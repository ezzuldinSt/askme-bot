mod admin;
mod config;
mod entities;
mod faqs;
mod gemini_client;
mod models;
mod qdrant_client;
mod qdrant_models;
mod search;
mod things_client;
mod tools;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use serde_json::{json, Value};
use tokio::io::AsyncBufReadExt;
use tokio::sync::{mpsc, oneshot, RwLock};
use tracing::{debug, error, info, warn};
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

use crate::config::RuntimeConfig;
use crate::entities::build_reply_with_entities;
use crate::gemini_client::{
    is_transient_exhausted, GeminiClient, GeminiError, GenerateTurn, KeyLease,
};
use crate::models::{
    Content, FileData, FunctionResponseData, Notification, Part, Post,
};
use crate::qdrant_client::{MemoryWrite, QdrantClient};
use crate::qdrant_models::{
    app_fact_point_id, user_fact_point_id, AppFactPayload, AppFactSource, AppFactStatus,
    AppKnowledgeSeed, FactCategory, MemoryEntry, MessagePayload, MessageType, UserFactPayload,
    PROCESSED_COLLECTION_NAME, THINGS_KNOWLEDGE_COLLECTION_NAME, USER_PROFILES_COLLECTION_NAME,
};
use crate::things_client::{is_auth_expired, is_not_found, ClientRejected, ThingsClient, TOKEN_FILE};
use crate::tools::{
    tool_declarations, ExtractionJob, ExtractionSource, ExtractionTask, FlowMeter, FlowSubject,
    ToolContext,
};

const BOT_USERNAME: &str = "AskMe";
const POLL_INTERVAL_MS: u64 = 3_000;
/// Reply text cap, in chars. Things rejects comment text over 2000 chars
/// (HTTP 422); truncation appends '…', so 1999 + '…' lands exactly at the
/// server's max.
const MAX_RESPONSE_LENGTH: usize = 1999;
const MAX_CONTEXT_DEPTH: usize = 20;
const MAX_MEDIA_FILES: usize = 5;
const MEMORY_WRITE_BATCH_SIZE: usize = 5;
const MEMORY_WRITE_FLUSH_MS: u64 = 2_000;
/// Notification pages fetched per poll (until all unread are covered).
const MAX_NOTIFICATION_PAGES: u32 = 5;
/// How often a notification may fail before it is poison-marked and skipped.
const MAX_PROCESS_ATTEMPTS: u8 = 3;
/// Polls with an unchanged unread count before a stuck count is called out.
const STAGNANT_WARN_CYCLES: u32 = 20;
/// Deleted-post sweep: runs this often after the boot sweep.
const SWEEP_INTERVAL_SECS: u64 = 6 * 3600;
/// Delay before the first sweep so a crash-looping service cannot hammer the
/// Things API.
const SWEEP_BOOT_DELAY_SECS: u64 = 60;
/// Spacing between per-post existence checks during a sweep (keeps the
/// request rate well under anything Cloudflare would frown at).
const SWEEP_REQUEST_SPACING_MS: u64 = 200;
/// Most recent conversation points verified per sweep.
const SWEEP_MAX_POINTS: u64 = 2_000;
/// A sweep only aborts on suspicion once at least this many posts were checked.
const SWEEP_ABORT_MIN_CHECKED: usize = 50;
/// ...and when more than this share of them 404 — mass 404s mean an API
/// malfunction, not a mass deletion.
const SWEEP_ABORT_STALE_RATIO: f64 = 0.9;
/// Timeout for the final memory-writer flush on shutdown.
const SHUTDOWN_FLUSH_TIMEOUT_SECS: u64 = 10;
/// Curated Things-app knowledge seeded into tier-3 memory on every boot.
const APP_KNOWLEDGE_SEED_FILE: &str = "things_knowledge.json";
/// Facts longer than this are almost certainly extraction noise.
const MAX_FACT_LENGTH: usize = 300;
/// Questions shorter than this never trigger app-knowledge retrieval: a
/// one-word question ("Hi", "Thanks") has diffuse semantics and matches
/// everything, while a genuine app question always names the app.
const MIN_APP_KNOWLEDGE_QUESTION_CHARS: usize = 12;
/// Max users auto-briefed per reply flow (each briefing costs one extra round).
const MAX_BRIEFED_SUBJECTS: usize = 2;

struct AppState {
    things: ThingsClient,
    gemini: GeminiClient,
    qdrant: Arc<QdrantClient>,
    memory_writer: mpsc::UnboundedSender<MemoryWrite>,
    extraction_writer: mpsc::UnboundedSender<ExtractionTask>,
    /// Live, hot-reloadable configuration (panel saves swap it atomically).
    runtime: Arc<RwLock<RuntimeConfig>>,
    system_prompt: String,
    /// In-memory mirror of processed notification ids (session-only safety net;
    /// Qdrant is the source of truth for cross-restart dedup).
    processed: HashSet<u64>,
    /// Per-notification failure counters for the current session.
    failures: HashMap<u64, u8>,
}

/// Result of attempting to handle one notification.
enum ProcessOutcome {
    /// A reply was generated and posted.
    Replied,
    /// Deliberately ignored (no post, bot's own post, empty question, ...).
    Skipped,
    /// Something transient failed; worth retrying on the next poll.
    Failed,
    /// Every Gemini key was rate-limited/cooling down. Retried on the next
    /// poll WITHOUT counting against MAX_PROCESS_ATTEMPTS: a quota cooldown
    /// heals on its own (RPM in a minute, daily cap at midnight Pacific), and
    /// poison-marking would lose the user's reply forever over a pause the
    /// bot simply had to wait out. Costs nothing while keys stay parked —
    /// no API call is made until one thaws.
    RateLimited,
}

#[tokio::main]
async fn main() -> Result<()> {
    let log_buffer = admin::new_log_buffer();
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .with(admin::LogLayer::new(log_buffer.clone()))
        .init();

    dotenvy::dotenv().ok();

    // Layered config: bot_config.json overrides > .env > defaults. Also
    // ensures the admin password hash exists (default CHANGEME on first boot).
    let mut bot_config = config::load();
    if let Err(e) = config::ensure_password_hash(&mut bot_config) {
        warn!("Failed to initialize admin password: {e}");
    }
    let bot_config = Arc::new(RwLock::new(bot_config));

    let gemini_api_keys = config::resolve_gemini_keys(&bot_config.read().await.overrides);
    if gemini_api_keys.is_empty() {
        error!("No Gemini API keys configured (bot_config.json or GEMINI_API_KEY(S) env)");
        std::process::exit(1);
    }
    let things_email = std::env::var("THINGS_EMAIL").expect("THINGS_EMAIL must be set in .env");
    let things_password =
        std::env::var("THINGS_PASSWORD").expect("THINGS_PASSWORD must be set in .env");

    let system_prompt = std::fs::read_to_string("SYSTEM_PROMPT.md")
        .expect("SYSTEM_PROMPT.md must exist in the working directory");

    let mut things = ThingsClient::new();

    if !things.load_cached_token() {
        info!("No cached token found. Logging in...");
        things.login(&things_email, &things_password).await?;
        info!("Login initiated. Check your email for the OTP code.");

        let otp = read_otp_from_stdin().await?;
        things.verify_otp(&things_email, &otp).await?;
    }

    info!("Authentication successful. Starting bot...");

    let args: Vec<String> = std::env::args().collect();

    let (generation_model, extraction_model, thinking_level, extraction_thinking_level, embedding_model, embedding_dimensions, qdrant_url) = {
        let cfg = bot_config.read().await;
        (
            config::resolve_generation_model(&cfg.overrides),
            config::resolve_extraction_model(&cfg.overrides),
            config::resolve_thinking_level(&cfg.overrides),
            config::resolve_extraction_thinking_level(&cfg.overrides),
            config::resolve_embedding_model(&cfg.overrides),
            config::resolve_embedding_dimensions(&cfg.overrides),
            config::resolve_qdrant_url(&cfg.overrides),
        )
    };
    let gemini = GeminiClient::with_keys(
        gemini_api_keys,
        crate::gemini_client::GeminiOptions {
            generation_model,
            extraction_model,
            thinking_level,
            extraction_thinking_level,
            media_resolution: config::resolve_media_resolution(),
            embedding_model: embedding_model.clone(),
            embedding_dimensions: embedding_dimensions as u32,
        },
    );

    let runtime_config = Arc::new(RwLock::new(RuntimeConfig::resolve(
        &bot_config.read().await.overrides,
    )));

    let qdrant = QdrantClient::connect(&qdrant_url, Arc::new(gemini.clone()), embedding_dimensions)
        .await;
    let qdrant = Arc::new(qdrant);

    // Embedding-model migration: vectors of different models are incompatible
    // even at the same dimension, so a model change wipes the vector memory
    // (processed markers kept; app knowledge re-seeds below; user profiles
    // rebuild organically). A missing marker means pre-migration data, which
    // was embedded with LEGACY_EMBEDDING_MODEL.
    if qdrant.is_available() {
        let last_model = bot_config
            .read()
            .await
            .state
            .last_embedding_model
            .clone()
            .unwrap_or_else(|| config::LEGACY_EMBEDDING_MODEL.to_string());
        if last_model != embedding_model {
            warn!(
                "Embedding model changed {last_model} -> {embedding_model}: \
                 wiping vector memory (incompatible embeddings)"
            );
            match qdrant.reset_memory().await {
                Ok(()) => {
                    let mut cfg = bot_config.write().await;
                    // Old thresholds were tuned for the previous model — fall
                    // back to the defaults tuned for the new one.
                    cfg.overrides.user_fact_supersede_threshold = None;
                    cfg.overrides.forget_similarity_threshold = None;
                    cfg.state.last_embedding_model = Some(embedding_model.clone());
                    if let Err(e) = config::save(&cfg) {
                        warn!("Failed to save config after embedding migration: {e}");
                    }
                    *runtime_config.write().await = RuntimeConfig::resolve(&cfg.overrides);
                    info!("Embedding migration complete: memory wiped, thresholds reset");
                }
                Err(e) => error!("Embedding migration wipe failed: {e}"),
            }
        } else if bot_config.read().await.state.last_embedding_model.is_none() {
            // First boot with tracking enabled: just record the marker.
            let mut cfg = bot_config.write().await;
            cfg.state.last_embedding_model = Some(embedding_model.clone());
            if let Err(e) = config::save(&cfg) {
                warn!("Failed to record embedding model marker: {e}");
            }
        }
    }

    // Ops mode: wipe the three memory collections (processed-notification
    // markers are kept) and exit. Run this once when switching schemas.
    if args.iter().any(|a| a == "--reset-memory") {
        if !qdrant.is_available() {
            error!("--reset-memory requires a reachable Qdrant at {qdrant_url}");
            std::process::exit(1);
        }
        return match qdrant.reset_memory().await {
            Ok(()) => {
                info!("Memory reset complete; processed-notification markers kept. Start the bot normally.");
                Ok(())
            }
            Err(e) => {
                error!("Memory reset failed: {e}");
                std::process::exit(1);
            }
        };
    }

    if qdrant.is_available() {
        // Collection setup failures (e.g. vector-dimension mismatch) are
        // configuration errors: fail loudly instead of running degraded.
        if let Err(e) = qdrant.ensure_collections().await {
            error!("Qdrant collection setup failed: {e}");
            std::process::exit(1);
        }
        info!(
            "Qdrant collections ready: {}, {}, {}, {}",
            qdrant.collection_name(),
            PROCESSED_COLLECTION_NAME,
            USER_PROFILES_COLLECTION_NAME,
            THINGS_KNOWLEDGE_COLLECTION_NAME
        );
        if let Err(e) = seed_app_knowledge(&qdrant, &gemini).await {
            warn!("Failed to seed Things app knowledge: {e}");
        }
        // Case-insensitive fact reads: converge any pre-normalization data.
        if let Err(e) = qdrant.normalize_usernames().await {
            warn!("Failed to normalize user-fact usernames: {e}");
        }
    }

    let processed = load_processed_ids(&qdrant).await;
    let memory_writer = qdrant.spawn_writer(
        MEMORY_WRITE_BATCH_SIZE,
        Duration::from_millis(MEMORY_WRITE_FLUSH_MS),
    );
    let extraction_writer =
        spawn_extraction_worker(gemini.clone(), qdrant.clone(), runtime_config.clone());

    let state = Arc::new(RwLock::new(AppState {
        things,
        gemini,
        qdrant,
        memory_writer,
        extraction_writer,
        runtime: runtime_config,
        system_prompt,
        processed,
        failures: HashMap::new(),
    }));

    if args.iter().any(|a| a == "--test-post") {
        let post_id = parse_test_post_id(&args)
            .ok_or_else(|| anyhow::anyhow!("--test-post requires a numeric post id"))?;
        let do_post = args.iter().any(|a| a == "--post");
        let prompt_override = args
            .iter()
            .position(|a| a == "--prompt")
            .and_then(|p| args.get(p + 1))
            .cloned();
        return test_post(&state, post_id, do_post, prompt_override.as_deref()).await;
    }

    // Bot mode only (never in --test-post): on the very first boot, silently
    // mark the existing notification backlog as processed instead of replying
    // to every historical mention.
    seed_existing_notifications(&state).await;

    // Admin panel: bind, announce, and serve in the background. Non-fatal if
    // the port is taken — the bot works fine without the panel.
    {
        let (bind, port) = {
            let cfg = bot_config.read().await;
            (cfg.admin.bind.clone(), cfg.admin.port)
        };
        let addr = format!("{bind}:{port}");
        match tokio::net::TcpListener::bind(&addr).await {
            Ok(listener) => {
                let lan_ip = local_ip_address::local_ip().ok();
                info!("Admin panel: http://localhost:{port}");
                println!("Admin panel: http://localhost:{port}");
                if let Some(ip) = lan_ip {
                    info!("Admin panel (LAN): http://{ip}:{port}");
                    println!("Admin panel (LAN): http://{ip}:{port}");
                }
                let admin_state = admin::AdminState::new(state.clone(), bot_config, log_buffer);
                tokio::spawn(async move {
                    if let Err(e) = admin::serve(listener, admin_state).await {
                        error!("Admin panel server failed: {e}");
                    }
                });
            }
            Err(e) => warn!("Admin panel disabled: failed to bind {addr}: {e}"),
        }
    }

    let (_shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

    {
        let state = state.clone();
        tokio::spawn(async move {
            poll_loop(state).await;
        });
    }

    // Forget posts that get deleted on Things (boot sweep + periodic sweeps).
    {
        let state = state.clone();
        tokio::spawn(async move {
            sweep_loop(state).await;
        });
    }

    wait_for_shutdown(&mut shutdown_rx).await;

    info!("Shutting down...");
    // Flush any queued memory writes before exiting.
    let (ack_tx, ack_rx) = oneshot::channel();
    if state
        .read()
        .await
        .memory_writer
        .send(MemoryWrite::Flush(ack_tx))
        .is_ok()
    {
        let _ = tokio::time::timeout(Duration::from_secs(SHUTDOWN_FLUSH_TIMEOUT_SECS), ack_rx).await;
    }
    // Drain queued fact-extraction jobs too (FIFO: the ack resolves once
    // every queued job ahead of it has completed).
    let (xack_tx, xack_rx) = oneshot::channel();
    if state
        .read()
        .await
        .extraction_writer
        .send(ExtractionTask::Flush(xack_tx))
        .is_ok()
    {
        let _ =
            tokio::time::timeout(Duration::from_secs(SHUTDOWN_FLUSH_TIMEOUT_SECS), xack_rx).await;
    }
    Ok(())
}

/// Extract the post id that must follow `--test-post` on the command line.
fn parse_test_post_id(args: &[String]) -> Option<u64> {
    let pos = args.iter().position(|a| a == "--test-post")?;
    args.get(pos + 1)?.parse::<u64>().ok()
}

async fn test_post(
    state: &RwLock<AppState>,
    post_id: u64,
    do_post: bool,
    prompt_override: Option<&str>,
) -> Result<()> {
    let post_data = {
        let things = &state.read().await.things;
        things.get_post(post_id).await?
    };

    let post = post_data
        .post
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Post {post_id} has no content"))?;

    println!("=== Post {post_id} by {} ===", post.author_username());
    println!("Comment: {}", post.content_text());

    let media_urls = extract_media_urls(post);
    println!("Media URLs found: {}", media_urls.len());
    for (i, url) in media_urls.iter().enumerate() {
        println!("  [{i}] {url}");
    }

    let question = extract_question(post);
    let is_follow_up = post.parent_id.is_some()
        && post_data
            .parent
            .as_ref()
            .map(|p| p.author_username().eq_ignore_ascii_case(BOT_USERNAME))
            .unwrap_or(false);
    let (conversation_id, ancestors) =
        resolve_conversation(state, &post_data, post, is_follow_up).await;

    if !ancestors.is_empty() {
        println!("=== Ancestor chain ({} posts above) ===", ancestors.len());
        for (author, content) in &ancestors {
            println!("  {}: {}", author, truncate_text(content, 120));
        }
    }

    let qdrant = state.read().await.qdrant.clone();
    println!("=== Qdrant status ===");
    if qdrant.is_available() {
        println!("Connected (collection: {})", qdrant.collection_name());

        let payload = MessagePayload {
            id: post_id,
            content: strip_mention(post.content_text()),
            username: post.author_username().to_string(),
            message_type: if is_follow_up {
                MessageType::Reply
            } else {
                MessageType::Post
            },
            parent_id: post.parent_id,
            conversation_id,
            timestamp: timestamp_from_post(post),
            media_urls: media_urls.clone(),
        };
        match qdrant.upsert(&payload).await {
            Ok(()) => println!("Stored post {post_id} in Qdrant (conversation {conversation_id})"),
            Err(e) => println!("Failed to store post {post_id} in Qdrant: {e}"),
        }

        let depth_limit = {
            let runtime = state.read().await.runtime.clone();
            let cfg = runtime.read().await;
            cfg.context_depth_limit
        };
        let conversation = qdrant
            .get_conversation_context(conversation_id, depth_limit as u64)
            .await
            .unwrap_or_default();
        println!(
            "=== Qdrant conversation context (conversation {conversation_id}, {} entries) ===",
            conversation.len()
        );
        for entry in &conversation {
            println!(
                "  [{}] {}: {}",
                entry.timestamp, entry.username, entry.content
            );
        }

        let (facts_limit, app_limit, app_min_score) = {
            let runtime = state.read().await.runtime.clone();
            let cfg = runtime.read().await;
            (
                cfg.memory.user_facts_limit,
                cfg.memory.app_knowledge_limit,
                cfg.memory.app_knowledge_min_score,
            )
        };
        let facts = qdrant
            .list_user_facts(post.author_username(), facts_limit)
            .await
            .unwrap_or_default();
        println!(
            "=== Qdrant user facts for {} ({} active) ===",
            post.author_username(),
            facts.len()
        );
        for (_, fact) in &facts {
            println!("  - {}", fact.fact);
        }

        let app_hits = qdrant
            .search_app_knowledge(&question, app_min_score, app_limit)
            .await
            .unwrap_or_default();
        println!("=== Qdrant app knowledge ({} hits) ===", app_hits.len());
        for fact in &app_hits {
            println!("  - {}", fact.fact);
        }
    } else {
        println!("UNREACHABLE — running degraded (no persistent memory)");
    }

    let gemini = state.read().await.gemini.clone();
    let system_prompt = state.read().await.system_prompt.clone();

    let mut media_files: Vec<DownloadedMedia> = Vec::new();
    if !media_urls.is_empty() {
        println!("=== Downloading media ===");
        for url in &media_urls {
            match download_media_file(state, url).await {
                Ok(file) => {
                    println!("Downloaded {} bytes ({})", file.data.len(), file.mime);
                    media_files.push(file);
                }
                Err(e) => {
                    println!("Media download failed for {url}: {e}");
                }
            }
        }
    }

    let meter = Arc::new(FlowMeter::default());
    let user_text = match prompt_override {
        Some(custom) => custom.to_string(),
        None => {
            if is_follow_up {
                println!("=== Follow-up reply (parent is {BOT_USERNAME}) ===");
                build_follow_up_prompt(state, post, &question, &post_data, conversation_id, Some(&meter))
                    .await
            } else {
                build_mention_prompt(state, post, &question, &ancestors, Some(&meter)).await
            }
        }
    };
    println!("=== Prompt ===");
    println!("{user_text}");

    let flow_subjects: Arc<Mutex<Vec<FlowSubject>>> = Arc::new(Mutex::new(Vec::new()));
    let tool_ctx = {
        let s = state.read().await;
        ToolContext::new(
            Arc::new(s.things.clone()),
            s.qdrant.clone(),
            s.runtime.clone(),
            s.extraction_writer.clone(),
            flow_subjects,
            meter.clone(),
        )
    };
    println!(
        "=== Generating response (key pool: {} keys) ===",
        gemini.pool_size()
    );
    let (response, sources) = generate_with_tools_failover(
        &gemini,
        &system_prompt,
        &user_text,
        &media_files,
        &tool_ctx,
    )
    .await?;
    println!("=== API calls used: {} ===", meter.summary());

    println!("=== Raw response ===");
    println!("{response}");

    let response = match clean_scaffold_leak(&gemini, response, &meter).await {
        Ok(t) => t,
        Err(()) => {
            println!("=== Reply was unsalvageable scaffold leakage; would be dropped ===");
            return Ok(());
        }
    };

    let reply_text = append_sources_footer(&response, &sources);
    let (reply_text, entities) = build_reply_with_entities(&reply_text, MAX_RESPONSE_LENGTH);
    println!("=== Clean reply text ===");
    println!("{reply_text}");
    println!("=== Entities ===");
    for e in &entities {
        println!(
            "  bold offset={} length={} (chars {}-{})",
            e.offset,
            e.length,
            e.offset,
            e.offset + e.length
        );
    }

    if do_post {
        println!("=== Posting reply ===");
        let things = &state.read().await.things;
        match things
            .reply_to_post(post_id, &reply_text, &entities, None, None)
            .await
        {
            Ok(reply_id) => println!("Posted reply {reply_id} to post {post_id}"),
            Err(e) => println!("Failed to post reply: {e}"),
        }
    }

    Ok(())
}

async fn read_otp_from_stdin() -> Result<String> {
    let stdin = tokio::io::stdin();
    let mut reader = tokio::io::BufReader::new(stdin);
    let mut line = String::new();
    println!("Enter OTP code: ");
    reader.read_line(&mut line).await?;
    Ok(line.trim().to_string())
}

async fn wait_for_shutdown(_rx: &mut mpsc::Receiver<()>) {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for Ctrl+C");
    info!("Received Ctrl+C signal");
}

/// Exit the process when the Things token has expired: the bot cannot recover
/// on its own (login needs an interactive OTP), so fail loudly — after
/// removing the stale token file — instead of error-looping forever.
fn exit_if_auth_expired(err: &anyhow::Error) {
    if is_auth_expired(err) {
        error!(
            "Things auth token expired (HTTP 401). Deleting stale {TOKEN_FILE}; \
             restart the bot and complete the OTP login again."
        );
        let _ = std::fs::remove_file(TOKEN_FILE);
        std::process::exit(2);
    }
}

/// Seed the processed-notification cache from Qdrant.
async fn load_processed_ids(qdrant: &Arc<QdrantClient>) -> HashSet<u64> {
    let mut ids = HashSet::new();

    if qdrant.is_available() {
        match qdrant.list_processed().await {
            Ok(list) => {
                ids.extend(list);
                info!(
                    "Loaded {} processed notification IDs from Qdrant",
                    ids.len()
                );
            }
            Err(e) => warn!("Failed to load processed IDs from Qdrant: {e}"),
        }
    }

    ids
}

/// On the very first boot (no processed markers anywhere), mark every
/// currently-listed notification as processed WITHOUT replying, so the bot
/// only answers mentions that arrive after startup.
async fn seed_existing_notifications(state: &RwLock<AppState>) {
    let (qdrant, already_have_ids) = {
        let s = state.read().await;
        (s.qdrant.clone(), !s.processed.is_empty())
    };
    if already_have_ids || !qdrant.is_available() {
        return;
    }

    let mut ids: Vec<u64> = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();
    for page in 1..=MAX_NOTIFICATION_PAGES {
        let batch = {
            let things = &state.read().await.things;
            match things.get_notifications(page).await {
                Ok(b) => b,
                Err(e) => {
                    exit_if_auth_expired(&e);
                    warn!("Failed to fetch notifications for seeding (page {page}): {e}");
                    break;
                }
            }
        };
        if batch.is_empty() {
            break;
        }
        for n in batch {
            if seen.insert(n.id) {
                ids.push(n.id);
            }
        }
    }

    if ids.is_empty() {
        return;
    }

    match qdrant.mark_processed_many(&ids).await {
        Ok(()) => {
            state.write().await.processed.extend(ids.iter().copied());
            info!(
                "First boot: seeded {} existing notifications as processed; \
                 only mentions arriving from now on will be answered",
                ids.len()
            );
        }
        Err(e) => warn!("Failed to seed existing notifications: {e}"),
    }
}

/// Fetch notification pages until every unread notification has been seen (or
/// the pages run out), so unread items buried past page 1 are never missed.
async fn collect_notifications(
    state: &RwLock<AppState>,
    unread_count: u64,
) -> Result<Vec<Notification>> {
    let mut all: Vec<Notification> = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();
    let mut unread_seen: u64 = 0;

    for page in 1..=MAX_NOTIFICATION_PAGES {
        let batch = {
            let things = &state.read().await.things;
            things.get_notifications(page).await?
        };
        if batch.is_empty() {
            break;
        }
        for notification in batch {
            if !seen.insert(notification.id) {
                continue;
            }
            if notification.is_read != Some(true) {
                unread_seen += 1;
            }
            all.push(notification);
        }
        if unread_seen >= unread_count {
            break;
        }
    }

    Ok(all)
}

async fn poll_loop(state: Arc<RwLock<AppState>>) {
    let mut stagnant: Option<(u64, u32)> = None;

    loop {
        tokio::time::sleep(Duration::from_millis(POLL_INTERVAL_MS)).await;

        let unread_count = {
            let things = &state.read().await.things;
            match things.get_unread_count().await {
                Ok(count) => count,
                Err(e) => {
                    exit_if_auth_expired(&e);
                    error!("Failed to get unread count: {e}");
                    continue;
                }
            }
        };

        if unread_count == 0 {
            stagnant = None;
            continue;
        }

        // A count that never changes means the unread notifications are not
        // visible in the fetched pages (deleted posts, server-side quirks).
        // Warn once, then stay quiet instead of spamming the log every poll.
        match &mut stagnant {
            Some((count, cycles)) if *count == unread_count => *cycles += 1,
            _ => stagnant = Some((unread_count, 0)),
        }
        let stagnant_cycles = stagnant.map(|(_, c)| c).unwrap_or(0);
        if stagnant_cycles == STAGNANT_WARN_CYCLES {
            warn!(
                "Unread notification count stuck at {unread_count} for ~{STAGNANT_WARN_CYCLES} \
                 polls; the unread notifications are not visible in the fetched pages \
                 (deleted posts?). Polling continues quietly."
            );
        }
        if stagnant_cycles > STAGNANT_WARN_CYCLES {
            debug!("{unread_count} unread notifications (count unchanged, nothing actionable)");
        } else {
            info!("{unread_count} unread notifications");
        }

        let notifications = match collect_notifications(&state, unread_count).await {
            Ok(n) => n,
            Err(e) => {
                exit_if_auth_expired(&e);
                error!("Failed to fetch notifications: {e}");
                continue;
            }
        };

        let mut to_process = Vec::new();
        let mut known_processed = Vec::new();
        let mut irrelevant = Vec::new();
        // Reply-target-uncertain post_replies: left unread, re-classified next poll.
        let mut uncertain: HashSet<u64> = HashSet::new();
        {
            let state_read = state.read().await;
            for notification in &notifications {
                if state_read.processed.contains(&notification.id) {
                    continue;
                }
                if state_read
                    .qdrant
                    .is_processed(notification.id)
                    .await
                    .unwrap_or(false)
                {
                    known_processed.push(notification.id);
                    continue;
                }
                if is_mention_notification(notification) {
                    to_process.push(notification.clone());
                    continue;
                }
                match classify_follow_up(&state_read, notification).await {
                    FollowUp::Yes => to_process.push(notification.clone()),
                    FollowUp::No => irrelevant.push(notification.id),
                    FollowUp::Unknown => {
                        uncertain.insert(notification.id);
                    }
                }
            }
        }
        if !uncertain.is_empty() {
            debug!(
                "{} post_reply notification(s) have an unknown reply target \
                 (payload lacks author info and memory is unreachable); leaving unread",
                uncertain.len()
            );
        }
        {
            let mut state_write = state.write().await;
            state_write.processed.extend(known_processed);
            // Irrelevant notifications (likes, follows, replies to other
            // people's posts) are cached session-locally so they are not
            // re-classified on every poll.
            state_write.processed.extend(irrelevant);
        }

        if !to_process.is_empty() {
            info!("Processing {} notifications", to_process.len());

            let mut handles = Vec::new();
            for notification in to_process {
                let state = state.clone();
                let handle = tokio::spawn(async move {
                    let id = notification.id;
                    let outcome = process_notification(state, notification).await;
                    (id, outcome)
                });
                handles.push(handle);
            }

            let mut retry_later: HashSet<u64> = HashSet::new();
            for handle in handles {
                match handle.await {
                    Ok((id, ProcessOutcome::Replied)) | Ok((id, ProcessOutcome::Skipped)) => {
                        state.write().await.failures.remove(&id);
                    }
                    Ok((id, ProcessOutcome::RateLimited)) => {
                        // Quota pause, not a real failure: stays unread and is
                        // retried next poll, but never poison-marked.
                        retry_later.insert(id);
                    }
                    Ok((id, ProcessOutcome::Failed)) => {
                        let attempts = {
                            let mut s = state.write().await;
                            let n = s.failures.entry(id).or_insert(0);
                            *n += 1;
                            *n
                        };
                        if attempts >= MAX_PROCESS_ATTEMPTS {
                            error!(
                                "Notification {id} failed {attempts} times; \
                                 marking it processed so it is not retried forever"
                            );
                            let qdrant = state.read().await.qdrant.clone();
                            if let Err(e) = qdrant.mark_processed(id).await {
                                warn!("Failed to poison-mark notification {id} in Qdrant: {e}");
                            }
                            let mut s = state.write().await;
                            s.processed.insert(id);
                            s.failures.remove(&id);
                        } else {
                            warn!(
                                "Notification {id} failed (attempt {attempts}/{MAX_PROCESS_ATTEMPTS}); \
                                 will retry on the next poll"
                            );
                            retry_later.insert(id);
                        }
                    }
                    Err(e) => {
                        error!("Notification processing task failed: {e}");
                    }
                }
            }

            // Mark read everything we saw EXCEPT failures that will be
            // retried (they stay unread so the next poll picks them up) and
            // reply-target-uncertain items (re-classified next poll).
            let ids: Vec<u64> = notifications
                .iter()
                .map(|n| n.id)
                .filter(|id| !retry_later.contains(id) && !uncertain.contains(id))
                .collect();
            if !ids.is_empty() {
                let things = &state.read().await.things;
                if let Err(e) = things.mark_notifications_read(&ids).await {
                    exit_if_auth_expired(&e);
                    error!("Failed to mark notifications as read: {e}");
                }
            }
        } else {
            // Nothing actionable: still drain the unread count for everything
            // we saw (likes, follows, already-handled items, ...) — except
            // reply-target-uncertain items, which stay unread.
            let ids: Vec<u64> = notifications
                .iter()
                .map(|n| n.id)
                .filter(|id| !uncertain.contains(id))
                .collect();
            if !ids.is_empty() {
                let things = &state.read().await.things;
                if let Err(e) = things.mark_notifications_read(&ids).await {
                    exit_if_auth_expired(&e);
                    error!("Failed to mark notifications as read: {e}");
                }
            }
        }
    }
}

/// Decide a sweep's fate from its hit counts. A run that finds nearly every
/// stored post gone is far more likely an API malfunction than a mass
/// deletion, so it refuses to delete anything.
fn sweep_should_abort(checked: usize, stale: usize) -> bool {
    checked >= SWEEP_ABORT_MIN_CHECKED
        && (stale as f64) / (checked as f64) > SWEEP_ABORT_STALE_RATIO
}

/// Forget posts that no longer exist on Things: verify stored conversation
/// messages against the API and purge the ones that 404.
///
/// Only a definitive 404 counts as deleted — any other failure (5xx, 429,
/// 403, transport) leaves the point untouched — and an implausibly high
/// stale ratio aborts the whole sweep instead of wiping memory on an API bug.
async fn sweep_deleted_posts(state: &RwLock<AppState>) {
    let (things, qdrant) = {
        let s = state.read().await;
        (s.things.clone(), s.qdrant.clone())
    };
    if !qdrant.is_available() {
        return;
    }
    let refs = match qdrant.list_conversation_refs(SWEEP_MAX_POINTS).await {
        Ok(refs) => refs,
        Err(e) => {
            warn!("Deleted-post sweep: failed to list conversation points: {e}");
            return;
        }
    };
    if refs.is_empty() {
        return;
    }

    let mut stale: Vec<u64> = Vec::new();
    let mut checked = 0usize;
    let mut skipped = 0usize;
    for (id, _) in &refs {
        match things.get_post(*id).await {
            Ok(_) => {}
            Err(e) => {
                exit_if_auth_expired(&e);
                if is_not_found(&e) {
                    stale.push(*id);
                } else {
                    skipped += 1;
                }
            }
        }
        checked += 1;
        tokio::time::sleep(Duration::from_millis(SWEEP_REQUEST_SPACING_MS)).await;
    }

    if stale.is_empty() {
        info!("Deleted-post sweep: {checked} checked, nothing stale ({skipped} skipped)");
        return;
    }
    if sweep_should_abort(checked, stale.len()) {
        warn!(
            "Deleted-post sweep ABORTED: {} of {} checked posts 404 (>{:.0}%); \
             that smells like an API malfunction, not a mass deletion — nothing purged",
            stale.len(),
            checked,
            SWEEP_ABORT_STALE_RATIO * 100.0,
        );
        return;
    }
    match qdrant.delete_conversation_points(&stale).await {
        Ok(()) => info!(
            "Deleted-post sweep: {checked} checked, {} stale purged, {skipped} skipped",
            stale.len(),
        ),
        Err(e) => error!(
            "Deleted-post sweep: failed to purge {} stale points: {e}",
            stale.len()
        ),
    }
}

/// Periodic reconciliation: Things posts are ephemeral and users delete
/// replies, but Qdrant memory would otherwise keep echoing them into
/// conversation context forever.
async fn sweep_loop(state: Arc<RwLock<AppState>>) {
    tokio::time::sleep(Duration::from_secs(SWEEP_BOOT_DELAY_SECS)).await;
    loop {
        sweep_deleted_posts(&state).await;
        tokio::time::sleep(Duration::from_secs(SWEEP_INTERVAL_SECS)).await;
    }
}

fn is_mention_notification(notification: &Notification) -> bool {
    let nt = notification.notification_type.as_deref().unwrap_or("");
    let group = notification.group.as_deref().unwrap_or("");
    nt == "user_mention" || nt == "mention" || group == "mentions"
}

/// How confident the bot is that a notification is a reply to the bot itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FollowUp {
    Yes,
    No,
    /// Cannot tell right now (payload carries no author info AND memory is
    /// unreachable). Left unread and re-evaluated on the next poll — never
    /// silently dropped, never marked read.
    Unknown,
}

/// A follow-up is a reply to a post that AskMe itself wrote (detected from the
/// notification payload, or from a `bot_reply` already persisted in Qdrant).
async fn classify_follow_up(state: &AppState, notification: &Notification) -> FollowUp {
    let nt = notification.notification_type.as_deref().unwrap_or("");
    if nt != "post_reply" {
        return FollowUp::No;
    }
    let original_post = notification.original_post_data.as_ref();
    // Only the unique username is trusted — display names are user-controlled
    // and anyone can call themselves "AskMe".
    let is_bot_author = original_post
        .and_then(|p| p.user.as_ref())
        .and_then(|u| u.username.as_deref())
        .map(|u| u.eq_ignore_ascii_case(BOT_USERNAME))
        .unwrap_or(false);
    if is_bot_author {
        return FollowUp::Yes;
    }
    // The replied-to post's id, directly or via the reply's parent link.
    let original_post_id = original_post
        .and_then(|p| p.id_value())
        .or_else(|| notification.reply_post_data.as_ref().and_then(|p| p.parent_id));
    let qdrant = state.qdrant.clone();
    if !qdrant.is_available() {
        // Payload gave no answer and memory cannot arbitrate: unknown.
        return if original_post_id.is_some() {
            FollowUp::Unknown
        } else {
            FollowUp::No
        };
    }
    let Some(original_post_id) = original_post_id else {
        return FollowUp::No;
    };
    match qdrant.get_point(original_post_id).await {
        Ok(Some(entry)) => {
            if entry.message_type == MessageType::BotReply {
                FollowUp::Yes
            } else {
                FollowUp::No
            }
        }
        // Not in memory: positively not a bot reply (bot replies are stored
        // synchronously right after posting).
        Ok(None) => FollowUp::No,
        // Memory lookup failed — retry next poll instead of dropping.
        Err(_) => FollowUp::Unknown,
    }
}

fn notification_post_id(notification: &Notification) -> Option<u64> {
    notification_post(notification).and_then(|p| p.id_value())
}

/// The full post row carried by the notification that corresponds to the post
/// the bot is answering (mirrors `notification_post_id`). Unlike the flat
/// `GET /post/{id}` response, notification payloads include `post_type` and
/// `expires_at`, so the reply can mirror the mentioned post's kind and
/// lifetime.
fn notification_post(notification: &Notification) -> Option<&Post> {
    if is_mention_notification(notification) {
        return notification.post_data.as_ref();
    }
    if notification.notification_type.as_deref() == Some("post_reply") {
        return notification
            .reply_post_data
            .as_ref()
            .or(notification.original_post_data.as_ref());
    }
    notification
        .post_data
        .as_ref()
        .or(notification.reply_post_data.as_ref())
}

/// Mark a notification as deliberately skipped (local cache + Qdrant marker),
/// so it is never re-evaluated, and report the outcome.
async fn skip_notification(state: &Arc<RwLock<AppState>>, notification_id: u64) -> ProcessOutcome {
    let qdrant = state.read().await.qdrant.clone();
    if let Err(e) = qdrant.mark_processed(notification_id).await {
        warn!("Failed to mark skipped notification {notification_id} in Qdrant: {e}");
    }
    state.write().await.processed.insert(notification_id);
    ProcessOutcome::Skipped
}

/// How a failed reply-post should be handled.
enum ReplyFailure {
    /// The reply definitely never committed AND a later attempt may succeed:
    /// connect errors (the request never reached the server) and 422
    /// validation rejections (the text is regenerated on every attempt).
    Retryable,
    /// The server definitively refused this TARGET (403 replies-disabled /
    /// forbidden, 404/410 gone, 400 or other 4xx payload problems a
    /// regenerated text won't fix). Retrying only burns Gemini quota.
    Permanent,
    /// Ambiguous (timeout after send, 5xx): the reply may have committed —
    /// retrying would post a duplicate, which is worse than a lost retry.
    Ambiguous,
}

/// Classify a failed reply-post. The old binary "safe to retry" treated every
/// 4xx as retryable, so a replies-disabled post (403) re-ran the ENTIRE
/// pipeline — fetch, embed, generate — three pointless times.
fn classify_reply_error(e: &anyhow::Error) -> ReplyFailure {
    let rejected_status = e
        .chain()
        .find_map(|cause| cause.downcast_ref::<ClientRejected>().map(|r| r.status));
    if let Some(status) = rejected_status {
        return match status {
            reqwest::StatusCode::UNPROCESSABLE_ENTITY => ReplyFailure::Retryable,
            _ => ReplyFailure::Permanent,
        };
    }
    let never_sent = e.chain().any(|cause| {
        cause
            .downcast_ref::<reqwest::Error>()
            .map(|re| re.is_connect())
            .unwrap_or(false)
    });
    if never_sent {
        ReplyFailure::Retryable
    } else {
        ReplyFailure::Ambiguous
    }
}

async fn process_notification(
    state: Arc<RwLock<AppState>>,
    notification: Notification,
) -> ProcessOutcome {
    let notification_id = notification.id;

    let post_id = match notification_post_id(&notification) {
        Some(id) => id,
        None => {
            warn!(
                "Notification {} has no post data, skipping",
                notification.id
            );
            return skip_notification(&state, notification_id).await;
        }
    };

    // The read guard must drop before the error path: the 404 branch below
    // takes further locks (and `skip_notification` a write lock).
    let post_data_result = {
        let things = &state.read().await.things;
        things.get_post(post_id).await
    };

    let post_data = match post_data_result {
        Ok(data) => data,
        Err(e) => {
            exit_if_auth_expired(&e);
            if is_not_found(&e) {
                // The post was deleted between notification and processing —
                // retrying can never succeed, so skip instead of failing.
                info!("Notification {notification_id}: post {post_id} is gone (404), skipping");
                let qdrant = state.read().await.qdrant.clone();
                if qdrant.is_available() {
                    if let Err(e) = qdrant.delete_conversation_points(&[post_id]).await {
                        warn!("Failed to purge deleted post {post_id} from Qdrant: {e}");
                    }
                }
                return skip_notification(&state, notification_id).await;
            }
            error!("Failed to fetch post {post_id}: {e}");
            return ProcessOutcome::Failed;
        }
    };

    let post = match post_data.post {
        Some(ref p) => p.clone(),
        None => {
            warn!("Post {post_id} has no content");
            return skip_notification(&state, notification_id).await;
        }
    };

    // Never answer our own posts (defensive: prevents self-conversations if a
    // notification ever points at one of the bot's replies).
    if post.author_username().eq_ignore_ascii_case(BOT_USERNAME) {
        info!("Notification {notification_id}: post authored by {BOT_USERNAME}, skipping");
        return skip_notification(&state, notification_id).await;
    }

    let is_follow_up = {
        let state_read = state.read().await;
        matches!(classify_follow_up(&state_read, &notification).await, FollowUp::Yes)
    };

    let question = extract_question(&post);
    // An empty mention (no question text) is not skipped: it gets a short
    // generated greeting instead. Only fully blank content falls through.
    let is_empty_mention = question.chars().count() < 2;
    if is_empty_mention {
        info!("Notification {notification_id}: empty mention, replying with a greeting");
    }

    let (conversation_id, ancestors) =
        resolve_conversation(&state, &post_data, &post, is_follow_up).await;
    let memory_payload = MessagePayload {
        id: post_id,
        content: strip_mention(post.content_text()),
        username: post.author_username().to_string(),
        message_type: if is_follow_up {
            MessageType::Reply
        } else {
            MessageType::Post
        },
        parent_id: post
            .parent_id
            .or_else(|| post_data.parent.as_ref().and_then(|p| p.id_value())),
        conversation_id,
        timestamp: timestamp_from_post(&post),
        media_urls: extract_media_urls(&post),
    };

    {
        let qdrant = state.read().await.qdrant.clone();
        // Empty mentions carry no content worth remembering; skip the store.
        // A retry re-processing the same notification finds the post already
        // stored with identical content — skip the re-store too: an upsert
        // would re-embed the identical text and burn an API call for nothing.
        let mut should_store = qdrant.is_available() && !is_empty_mention;
        if should_store {
            match qdrant.get_point(post_id).await {
                Ok(Some(existing)) if existing.content == memory_payload.content => {
                    should_store = false;
                }
                Ok(_) => {}
                Err(e) => warn!("Failed to check stored copy of post {post_id}: {e}"),
            }
        }
        if should_store {
            if is_follow_up {
                // Follow-ups must be immediately visible to the context builder.
                if let Err(e) = qdrant.upsert(&memory_payload).await {
                    warn!("Failed to store follow-up post {post_id} in Qdrant: {e}");
                }
            } else {
                let _ = state
                    .read()
                    .await
                    .memory_writer
                    .send(MemoryWrite::Upsert(memory_payload));
            }
        }
    }

    let meter = Arc::new(FlowMeter::default());
    let user_text = if is_empty_mention {
        build_greeting_prompt(&state, &post, &post_data, &ancestors, is_follow_up, conversation_id)
            .await
    } else if is_follow_up {
        build_follow_up_prompt(&state, &post, &question, &post_data, conversation_id, Some(&meter))
            .await
    } else {
        build_mention_prompt(&state, &post, &question, &ancestors, Some(&meter)).await
    };

    info!(
        "Prompt for notification {notification_id}: {}",
        truncate_text(&user_text, 400)
    );

    let media_urls = extract_media_urls(&post);
    let gemini = state.read().await.gemini.clone();
    let system_prompt = state.read().await.system_prompt.clone();
    let flow_subjects: Arc<Mutex<Vec<FlowSubject>>> = Arc::new(Mutex::new(Vec::new()));
    let tool_ctx = {
        let s = state.read().await;
        ToolContext::new(
            Arc::new(s.things.clone()),
            s.qdrant.clone(),
            s.runtime.clone(),
            s.extraction_writer.clone(),
            flow_subjects,
            meter.clone(),
        )
    };

    // Download media first (per-file tolerant). The reply flow then runs with
    // a sticky API-key lease: on a 429 the key is cooled down, the next key is
    // leased, and any media is re-uploaded into the new project.
    let mut media_files: Vec<DownloadedMedia> = Vec::new();
    for url in &media_urls {
        match download_media_file(&state, url).await {
            Ok(file) => media_files.push(file),
            Err(e) => {
                exit_if_auth_expired(&e);
                warn!("Failed to download media {url} for notification {notification_id}: {e}");
            }
        }
    }

    let response =
        generate_with_tools_failover(&gemini, &system_prompt, &user_text, &media_files, &tool_ctx)
            .await;
    info!(
        "Reply flow for notification {notification_id} used {} API call(s)",
        meter.summary()
    );

    let (reply_text, reply_entities) = match response {
        Ok((text, sources)) => {
            let text = match clean_scaffold_leak(&gemini, text, &meter).await {
                Ok(t) => t,
                Err(()) => {
                    error!(
                        "Reply for notification {notification_id} was unsalvageable \
                         scaffold leakage; dropping it"
                    );
                    return ProcessOutcome::Failed;
                }
            };
            let text = append_sources_footer(&text, &sources);
            let (reply_text, entities) = build_reply_with_entities(&text, MAX_RESPONSE_LENGTH);
            info!(
                "Generated response ({} chars) for notification {notification_id} with {} entities",
                reply_text.chars().count(),
                entities.len(),
            );
            (reply_text, entities)
        }
        Err(e) => {
            error!("Gemini generation failed for notification {notification_id}: {e}");
            if e.chain()
                .any(|cause| cause.downcast_ref::<AllKeysRateLimited>().is_some())
            {
                return ProcessOutcome::RateLimited;
            }
            return ProcessOutcome::Failed;
        }
    };

    let reply_result = {
        let things = &state.read().await.things;
        // Mirror the kind and lifetime of the post being answered: the
        // notification's full post row (not the flat /post/{id} response)
        // carries post_type and expires_at. Falls back to bot defaults.
        let mirrored = notification_post(&notification);
        let reply_type = mirrored.and_then(|p| p.post_type.as_deref());
        let reply_duration = crate::things_client::post_duration_string(
            mirrored.and_then(|p| p.created_at.as_deref()),
            mirrored.and_then(|p| p.expires_at.as_deref()),
        );
        things
            .reply_to_post(
                post_id,
                &reply_text,
                &reply_entities,
                reply_type,
                reply_duration.as_deref(),
            )
            .await
    };

    let reply_id = match reply_result {
        Ok(reply_id) => {
            info!("Posted reply {reply_id} to post {post_id}");
            reply_id
        }
        Err(e) => {
            exit_if_auth_expired(&e);
            error!("Failed to post reply to {post_id}: {e}");
            match classify_reply_error(&e) {
                ReplyFailure::Retryable => return ProcessOutcome::Failed,
                ReplyFailure::Permanent => {
                    warn!(
                        "Reply target {post_id} permanently rejected the reply; \
                         skipping notification {notification_id} (no quota-wasting retries)"
                    );
                    // A 404 at reply time means the post was deleted mid-flow:
                    // forget it right away instead of waiting for the sweeper.
                    if is_not_found(&e) {
                        let qdrant = state.read().await.qdrant.clone();
                        if qdrant.is_available() {
                            if let Err(e) = qdrant.delete_conversation_points(&[post_id]).await {
                                warn!("Failed to purge deleted post {post_id} from Qdrant: {e}");
                            }
                        }
                    }
                    return skip_notification(&state, notification_id).await;
                }
                ReplyFailure::Ambiguous => {
                    warn!(
                        "Not retrying reply for notification {notification_id} \
                         (reply may already be committed); marking processed"
                    );
                    return skip_notification(&state, notification_id).await;
                }
            }
        }
    };

    let qdrant = state.read().await.qdrant.clone();

    let reply_payload = MessagePayload {
        id: reply_id,
        content: reply_text,
        username: BOT_USERNAME.to_string(),
        message_type: MessageType::BotReply,
        parent_id: Some(post_id),
        conversation_id,
        timestamp: unix_now(),
        media_urls: Vec::new(),
    };
    // Batched writer (one embed call per flush, not per reply). A fast
    // follow-up still sees this reply via the API parent chain, and the next
    // poll is 3s out while the writer flushes every 2s.
    if state
        .read()
        .await
        .memory_writer
        .send(MemoryWrite::Upsert(reply_payload))
        .is_err()
    {
        warn!("Failed to queue bot reply {reply_id} for Qdrant");
    }
    if let Err(e) = qdrant.mark_processed(notification_id).await {
        warn!("Failed to mark notification {notification_id} processed in Qdrant: {e}");
    }
    state.write().await.processed.insert(notification_id);

    // Long-term memory pass: pull durable user/app facts out of the user's
    // message in the background. Idempotent, so retries are harmless. Empty
    // mentions — and trivially short ones (thanks/ok/emoji) — have nothing
    // durable to teach, so they skip the extraction call entirely.
    let extraction_min_chars = {
        let runtime = state.read().await.runtime.clone();
        let cfg = runtime.read().await;
        cfg.memory.extraction_min_chars
    };
    if !is_empty_mention && passes_extraction_gate(&question, extraction_min_chars) {
        let _ = state
            .read()
            .await
            .extraction_writer
            .send(ExtractionTask::Job(ExtractionJob {
                username: post.author_username().to_string(),
                text: question.clone(),
                post_id,
                conversation_id,
                source: ExtractionSource::Conversation,
            }));
    }

    ProcessOutcome::Replied
}

/// Whether a mention is substantial enough to justify a background
/// fact-extraction call (0 = extract everything).
fn passes_extraction_gate(question: &str, min_chars: usize) -> bool {
    question.trim().chars().count() >= min_chars
}

fn extract_question(post: &Post) -> String {
    strip_mention(post.content_text())
}

/// Remove every `@AskMe` mention from the content, keeping the surrounding
/// text on both sides. ASCII-case-insensitive and char-boundary safe (a
/// whole-string `to_lowercase()` would shift byte offsets for characters like
/// Turkish `İ` and could slice mid-character — panicking or corrupting text).
///
/// The `@` must start a token: preceded by nothing, whitespace, or punctuation
/// — never by a word character or `/`, so URLs ("things.cv/@AskMe") and
/// glued words ("x@AskMe") are left intact.
fn strip_mention(content: &str) -> String {
    const MENTION: &[u8] = b"@askme";
    let bytes = content.as_bytes();
    let mut out = String::with_capacity(content.len());
    let mut last = 0;
    let mut i = 0;

    while i < bytes.len() {
        // '@' is ASCII, so `i` is always a char boundary here.
        if bytes[i] == b'@' {
            // Non-ASCII bytes (UTF-8 continuations, >= 0x80) never match this
            // set, so a multi-byte character before '@' still counts as a
            // token boundary.
            let prev_ok = i == 0
                || !matches!(
                    bytes[i - 1],
                    b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'/'
                );
            let end = i + MENTION.len();
            if prev_ok
                && end <= bytes.len()
                && content.is_char_boundary(end)
                && content[i..end].eq_ignore_ascii_case("@askme")
            {
                // Don't strip inside longer handles like "@AskMeBot".
                let boundary_ok = match content[end..].chars().next() {
                    None => true,
                    Some(c) => !(c.is_ascii_alphanumeric() || c == '_'),
                };
                if boundary_ok {
                    out.push_str(&content[last..i]);
                    last = end;
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
    out.push_str(&content[last..]);
    collapse_spaces(&out)
}

/// Collapse runs of spaces/tabs (left behind by removed mentions) and trim.
/// Newlines are preserved.
fn collapse_spaces(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_space = false;
    for c in s.chars() {
        if c == ' ' || c == '\t' {
            pending_space = true;
        } else {
            if pending_space && !out.is_empty() && c != '\n' {
                out.push(' ');
            }
            pending_space = false;
            out.push(c);
        }
    }
    out.trim().to_string()
}

fn extract_media_urls(post: &Post) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();

    let mut push = |url: &str| {
        if urls.len() < MAX_MEDIA_FILES && !urls.iter().any(|u| u == url) {
            urls.push(url.to_string());
        }
    };

    if let Some(ref media) = post.media {
        for item in media {
            if let Some(ref url) = item.url {
                push(url);
            }
        }
    }
    if let Some(ref images) = post.images {
        for item in images {
            if let Some(ref url) = item.url {
                push(url);
            }
        }
    }
    if let Some(ref attachment) = post.image {
        if let Some(ref url) = attachment.url {
            push(url);
        }
    }
    if let Some(ref attachments) = post.attachments {
        for item in attachments {
            if let Some(ref url) = item.url {
                push(url);
            }
        }
    }
    // Voice notes/audio arrive via a dedicated `audio` field (shape unknown —
    // accept a single object or an array, with the usual url-ish keys).
    fn audio_url(v: &Value) -> Option<&str> {
        v.get("url")
            .or_else(|| v.get("path"))
            .or_else(|| v.get("src"))
            .or_else(|| v.get("file_url"))
            .and_then(|u| u.as_str())
    }
    if let Some(ref audio) = post.audio {
        match audio {
            Value::Array(items) => {
                for item in items {
                    if let Some(u) = audio_url(item) {
                        push(u);
                    }
                }
            }
            other => {
                if let Some(u) = audio_url(other) {
                    push(u);
                }
            }
        }
    }
    // Music attachments (Apple Music cards): the playable audio is the
    // preview URL (~30s AAC on Apple's public CDN).
    if let Some(ref music) = post.music {
        for item in music {
            if let Some(ref url) = item.preview_url {
                push(url);
            }
        }
    }

    urls
}

/// A short "[Attached music: ...]" prompt line naming the post's tracks, so
/// the model can discuss the song even when the audio upload fails or the
/// track has no preview. PROMPT-ONLY: never stored in memory or fed to fact
/// extraction (it lives outside `content_text()`/`extract_question`).
fn music_note(post: &Post) -> String {
    let tracks: Vec<String> = post
        .music
        .as_ref()
        .map(|items| {
            items
                .iter()
                .filter_map(|m| {
                    let title = m.title.as_deref().unwrap_or("").trim();
                    let artist = m.artist.as_deref().unwrap_or("").trim();
                    match (title.is_empty(), artist.is_empty()) {
                        (true, true) => None,
                        (false, true) => Some(format!("\"{title}\"")),
                        (true, false) => Some(artist.to_string()),
                        (false, false) => Some(format!("\"{title}\" — {artist}")),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    if tracks.is_empty() {
        String::new()
    } else {
        format!("\n[Attached music: {}]", tracks.join(", "))
    }
}

async fn build_mention_prompt(
    state: &RwLock<AppState>,
    post: &Post,
    question: &str,
    ancestors: &[(String, String)],
    meter: Option<&FlowMeter>,
) -> String {
    let author = post.author_username();
    let depth_limit = {
        let runtime = state.read().await.runtime.clone();
        let cfg = runtime.read().await;
        cfg.context_depth_limit
    };
    let above = format_ancestor_chain(ancestors, depth_limit);

    let profile = build_user_profile_section(state, author).await;
    let app_knowledge = build_app_knowledge_section(state, question, meter).await;

    format!(
        "[Post by {author}] {content}\n[Question] {question}{music}{above}{profile}{app_knowledge}",
        content = post.content_text(),
        question = question,
        author = author,
        music = music_note(post),
        above = above,
        profile = profile,
        app_knowledge = app_knowledge,
    )
}

/// Dedicated prompt for empty mentions: the user poked the bot with no
/// question text, so instead of skipping we generate a short greeting.
/// Greetings are written in ARABIC and acknowledge the surrounding context
/// (the thread above a fresh mention, or the ongoing conversation on
/// follow-ups) so they never feel generic or out of place.
async fn build_greeting_prompt(
    state: &RwLock<AppState>,
    post: &Post,
    post_data: &models::PostData,
    ancestors: &[(String, String)],
    is_follow_up: bool,
    conversation_id: u64,
) -> String {
    let author = post.author_username();
    let depth_limit = {
        let runtime = state.read().await.runtime.clone();
        let cfg = runtime.read().await;
        cfg.context_depth_limit
    };

    let context = if is_follow_up {
        let (ctx, _, _) = load_conversation_context(
            state,
            conversation_id,
            post.id_value(),
            post_data,
            depth_limit,
        )
        .await;
        ctx
    } else {
        format_ancestor_chain(ancestors, depth_limit)
    };

    format!(
        "The user @{author} mentioned you without any question text{}. \
         Reply with a short, warm greeting in ARABIC (1-2 sentences, no hashtags). \
         The greeting must naturally acknowledge what the user was discussing \
         (the context below), not feel generic{}. \
         End by briefly saying you are here to help. Keep it under 40 words.",
        if is_follow_up {
            " (as a follow-up to an ongoing conversation)"
        } else {
            ""
        },
        if context.is_empty() {
            "; if there is no context, keep the greeting warm and simple"
        } else {
            ""
        },
    ) + &music_note(post) + &context
}

/// Render the thread above a mention as prompt context, oldest first (the same
/// shape as `[Conversation so far]` on the follow-up path). Keeps the newest
/// `limit` entries when the chain is longer.
fn format_ancestor_chain(ancestors: &[(String, String)], limit: usize) -> String {
    if ancestors.is_empty() || limit == 0 {
        return String::new();
    }
    let start = ancestors.len().saturating_sub(limit);
    let mut section = String::from("\n[Conversation above]");
    for (author, content) in &ancestors[start..] {
        section.push_str(&format!("\n{author}: {content}"));
    }
    section
}

/// Tier 2 memory: durable facts about the user being replied to — the ONLY
/// memory that crosses conversations, and always scoped to exactly that user.
async fn build_user_profile_section(state: &RwLock<AppState>, author: &str) -> String {
    if author.eq_ignore_ascii_case(BOT_USERNAME) {
        return String::new();
    }
    let (qdrant, limit, enabled) = {
        let s = state.read().await;
        let runtime = s.runtime.clone();
        let cfg = runtime.read().await;
        (
            s.qdrant.clone(),
            cfg.memory.user_facts_limit,
            cfg.memory.fact_extraction_enabled,
        )
    };
    if !enabled || !qdrant.is_available() {
        return String::new();
    }
    match qdrant.list_user_facts(author, limit).await {
        Ok(facts) => format_user_profile_section(
            author,
            &facts.into_iter().map(|(_, f)| f).collect::<Vec<_>>(),
        ),
        Err(e) => {
            warn!("Failed to load user facts for {author}: {e}");
            String::new()
        }
    }
}

fn format_user_profile_section(author: &str, facts: &[UserFactPayload]) -> String {
    if facts.is_empty() {
        return String::new();
    }
    let mut section = format!("\n[About {author} — long-term memory]");
    for fact in facts {
        section.push_str(&format!("\n- {}", fact.fact));
    }
    section
}

/// Tier 3 memory: verified facts about the Things app, injected only when the
/// question semantically matches them (score-gated), so unrelated
/// conversations never see app knowledge.
async fn build_app_knowledge_section(
    state: &RwLock<AppState>,
    question: &str,
    meter: Option<&FlowMeter>,
) -> String {
    let (qdrant, limit, min_score) = {
        let s = state.read().await;
        let runtime = s.runtime.clone();
        let cfg = runtime.read().await;
        (
            s.qdrant.clone(),
            cfg.memory.app_knowledge_limit,
            cfg.memory.app_knowledge_min_score,
        )
    };
    if !qdrant.is_available()
        || question.trim().chars().count() < MIN_APP_KNOWLEDGE_QUESTION_CHARS
    {
        return String::new();
    }
    if let Some(m) = meter {
        m.embed(1);
    }
    match qdrant.search_app_knowledge(question, min_score, limit).await {
        Ok(facts) => format_app_knowledge_section(&facts),
        Err(e) => {
            warn!("Failed to search app knowledge: {e}");
            String::new()
        }
    }
}

fn format_app_knowledge_section(facts: &[AppFactPayload]) -> String {
    if facts.is_empty() {
        return String::new();
    }
    let mut section = String::from("\n[About Things — app knowledge]");
    for fact in facts {
        section.push_str(&format!("\n- {}", fact.fact));
    }
    section
}

/// Conversation context block: Qdrant history for the conversation (excluding
/// the current post), falling back to the Things thread above the post when
/// memory is unavailable. Returns (context, entry_count, source).
async fn load_conversation_context(
    state: &RwLock<AppState>,
    conversation_id: u64,
    current_id: Option<u64>,
    post_data: &models::PostData,
    depth_limit: usize,
) -> (String, usize, &'static str) {
    let qdrant = state.read().await.qdrant.clone();
    let mut context = String::new();
    let mut source = "things-api";
    let mut entry_count = 0usize;

    if qdrant.is_available() {
        match qdrant
            .get_conversation_context(conversation_id, depth_limit as u64)
            .await
        {
            Ok(mut entries) if !entries.is_empty() => {
                source = "qdrant";
                // The current question was just persisted; don't echo it back
                // inside its own context block.
                entries.retain(|e| Some(e.id) != current_id);
                entries.truncate(depth_limit);
                entry_count = entries.len();
                context = format_context_entries(&entries);
            }
            Ok(_) => {}
            Err(e) => {
                warn!("Qdrant conversation lookup failed for conversation {conversation_id}: {e}")
            }
        }
    }

    if context.is_empty() {
        let things = &state.read().await.things;
        context = build_thread_context(things, post_data.parent.as_ref()).await;
        entry_count = context.matches('\n').count().saturating_sub(1);
    }

    (context, entry_count, source)
}

/// Build the prompt for a follow-up question, pulling the conversation history
/// from Qdrant (strictly scoped to this conversation) and falling back to the
/// Things API only when memory is unavailable.
async fn build_follow_up_prompt(
    state: &RwLock<AppState>,
    post: &Post,
    question: &str,
    post_data: &models::PostData,
    conversation_id: u64,
    meter: Option<&FlowMeter>,
) -> String {
    let author = post.author_username();
    let current_id = post.id_value();
    let depth_limit = {
        let runtime = state.read().await.runtime.clone();
        let cfg = runtime.read().await;
        cfg.context_depth_limit
    };

    let (context, entry_count, source) = load_conversation_context(
        state,
        conversation_id,
        current_id,
        post_data,
        depth_limit,
    )
    .await;

    info!(
        "Follow-up context for conversation {conversation_id} from {source} ({entry_count} entries / {} chars)",
        context.chars().count(),
    );

    let profile = build_user_profile_section(state, author).await;
    let app_knowledge = build_app_knowledge_section(state, question, meter).await;

    format!(
        "[Follow-up question by {author}] {question}{music}{context}{profile}{app_knowledge}",
        author = author,
        question = question,
        music = music_note(post),
        context = context,
        profile = profile,
        app_knowledge = app_knowledge,
    )
}

/// Render memory entries in the same `author: content` shape the prompt uses.
fn format_context_entries(entries: &[MemoryEntry]) -> String {
    if entries.is_empty() {
        return String::new();
    }
    let mut chain = String::from("\n[Conversation so far]");
    for entry in entries {
        chain.push_str(&format!("\n{}: {}", entry.username, entry.content));
    }
    chain
}

pub(crate) async fn build_thread_context(things: &ThingsClient, start_post: Option<&Post>) -> String {
    let mut entries: Vec<(String, String)> = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();

    let mut current = start_post.cloned();
    let mut depth = 0;

    while depth < MAX_CONTEXT_DEPTH {
        let Some(post) = current else { break };
        let Some(id) = post.id_value() else { break };
        if !seen.insert(id) {
            break;
        }

        let author = post.author_username().to_string();
        let content = strip_mention(post.content_text());
        entries.push((author, content));

        current = match post.parent_id {
            Some(parent_id) => match things.get_post(parent_id).await {
                Ok(data) => data.post,
                Err(_) => break,
            },
            None => break,
        };
        depth += 1;
    }

    if entries.is_empty() {
        return String::new();
    }

    entries.reverse();

    let mut chain = String::from("\n[Conversation so far]");
    for (author, content) in entries {
        chain.push_str(&format!("\n{author}: {content}"));
    }
    chain
}

/// Determine the conversation a post belongs to.
///
/// Isolation boundary: a conversation is rooted at the post where the bot was
/// first @mentioned. Follow-ups (replies to the bot) inherit the conversation
/// from their parent point; a fresh @mention ALWAYS starts a new conversation
/// rooted at itself — even deep inside a bigger Things thread — so two
/// mention-conversations never share context.
async fn resolve_conversation(
    state: &RwLock<AppState>,
    post_data: &models::PostData,
    post: &Post,
    is_follow_up: bool,
) -> (u64, Vec<(String, String)>) {
    let post_id = post.id_value().unwrap_or_default();
    let parent_id = post
        .parent_id
        .or_else(|| post_data.parent.as_ref().and_then(|p| p.id_value()));

    if is_follow_up {
        if let Some(parent_id) = parent_id {
            let qdrant = state.read().await.qdrant.clone();
            if qdrant.is_available() {
                if let Ok(Some(entry)) = qdrant.get_point(parent_id).await {
                    // Follow-ups need no ancestor chain: the conversation
                    // memory already holds the context.
                    return (entry.conversation_id, Vec::new());
                }
            }
        }
    }

    // New mention (or unresolvable follow-up): a conversation of its own.
    let chain = match parent_id {
        Some(parent_id) => {
            backfill_ancestors(state, post_data.parent.clone(), parent_id, post_id).await
        }
        None => Vec::new(),
    };
    (post_id, chain)
}

/// Store the ancestor chain above a fresh @mention as that conversation's
/// lead-in context, and return it (oldest first) so the mention prompt can
/// include the thread above.
///
/// Already-known posts are never re-stored (they belong to some other
/// conversation, and backfill must not steal points from it) — but they still
/// contribute to the RETURNED chain: the thread above a mention is legitimate
/// context for the prompt regardless of which conversation stored it first.
async fn backfill_ancestors(
    state: &RwLock<AppState>,
    seed_parent: Option<Post>,
    start_parent_id: u64,
    conversation_id: u64,
) -> Vec<(String, String)> {
    // Nearest-first during the walk; reversed at the end.
    let mut chain: Vec<(String, String)> = Vec::new();
    // The direct parent usually arrives with the notification payload — use it
    // instead of re-fetching it from the Things API. It is a stand-in for
    // `start_parent_id` ONLY: if the walk advances past the first step the
    // seed must be discarded, never used as another ancestor.
    let mut seed = seed_parent.filter(|p| p.id_value() == Some(start_parent_id));
    let mut pending: Vec<MessagePayload> = Vec::new();
    let mut current_id = start_parent_id;
    let mut seen: HashSet<u64> = HashSet::new();
    let mut depth = 0;

    while depth < MAX_CONTEXT_DEPTH {
        if !seen.insert(current_id) {
            break;
        }

        let qdrant = state.read().await.qdrant.clone();
        if !qdrant.is_available() {
            break;
        }

        let next_id = match qdrant.get_point(current_id).await {
            // Already stored (in ANY conversation): keep it out of our stores
            // but in the prompt chain, and keep walking via its parent.
            Ok(Some(entry)) => {
                chain.push((entry.username.clone(), entry.content.clone()));
                entry.parent_id
            }
            Err(_) => break,
            Ok(None) => {
                let parent_post = match seed.take().filter(|_| current_id == start_parent_id) {
                    Some(p) => p,
                    None => {
                        let things = &state.read().await.things;
                        match things.get_post(current_id).await {
                            Ok(data) => match data.post {
                                Some(p) => p,
                                None => break,
                            },
                            Err(_) => break,
                        }
                    }
                };

                chain.push((
                    parent_post.author_username().to_string(),
                    strip_mention(parent_post.content_text()),
                ));

                // Bot-authored ancestors keep their identity (the follow-up
                // detector looks for BotReply points); the thread root (no
                // parent) is a root post, not a reply.
                let message_type = if parent_post
                    .author_username()
                    .eq_ignore_ascii_case(BOT_USERNAME)
                {
                    MessageType::BotReply
                } else if parent_post.parent_id.is_none() {
                    MessageType::Post
                } else {
                    MessageType::Reply
                };
                pending.push(MessagePayload {
                    id: parent_post.id_value().unwrap_or_default(),
                    content: strip_mention(parent_post.content_text()),
                    username: parent_post.author_username().to_string(),
                    message_type,
                    parent_id: parent_post.parent_id,
                    conversation_id,
                    timestamp: timestamp_from_post(&parent_post),
                    media_urls: extract_media_urls(&parent_post),
                });

                parent_post.parent_id
            }
        };

        match next_id {
            Some(next) => current_id = next,
            None => break,
        }
        depth += 1;
    }

    // Persist synchronously (one batch embed) so freshly backfilled ancestors
    // are immediately visible to the context builder that runs right after —
    // e.g. a follow-up whose parent was not in memory.
    if !pending.is_empty() {
        let qdrant = state.read().await.qdrant.clone();
        if let Err(e) = qdrant.upsert_many(&pending).await {
            warn!(
                "Failed to backfill {} ancestor posts into Qdrant: {e}",
                pending.len()
            );
        }
    }

    chain.reverse();
    chain
}

/// Upsert the curated `things_knowledge.json` seed facts plus every support
/// FAQ's facts into tier-3 memory. Idempotent (deterministic point ids); a
/// missing file is not an error. Runs on boot and after memory wipes.
async fn seed_app_knowledge(qdrant: &Arc<QdrantClient>, gemini: &GeminiClient) -> Result<()> {
    let content = match std::fs::read_to_string(APP_KNOWLEDGE_SEED_FILE) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    let seeds: Vec<AppKnowledgeSeed> = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Failed to parse {APP_KNOWLEDGE_SEED_FILE}: {e}"))?;
    if seeds.is_empty() {
        return Ok(());
    }
    let now = unix_now();
    let items: Vec<(uuid::Uuid, AppFactPayload)> = seeds
        .into_iter()
        .map(|seed| {
            let point_id = app_fact_point_id(&seed.fact);
            let payload = AppFactPayload {
                topic: seed.topic,
                fact: seed.fact,
                source: AppFactSource::Seed,
                status: AppFactStatus::Active,
                updated_at: now,
            };
            (point_id, payload)
        })
        .collect();
    let count = items.len();
    qdrant.upsert_app_facts(&items).await?;
    info!("Seeded {count} app-knowledge facts from {APP_KNOWLEDGE_SEED_FILE}");
    // Support FAQs ride the same tier-3 collection (source = faq).
    if let Err(e) = faqs::seed_support_faqs(qdrant, gemini).await {
        warn!("Failed to seed support FAQs: {e}");
    }
    Ok(())
}

/// Spawn the background fact-extraction worker. For every answered user
/// message it runs one lightweight Gemini call and turns the result into
/// tier-2 (user facts) and tier-3 (pending app facts) memory writes.
fn spawn_extraction_worker(
    gemini: GeminiClient,
    qdrant: Arc<QdrantClient>,
    runtime: Arc<RwLock<RuntimeConfig>>,
) -> mpsc::UnboundedSender<ExtractionTask> {
    let (tx, mut rx) = mpsc::unbounded_channel::<ExtractionTask>();
    tokio::spawn(async move {
        while let Some(task) = rx.recv().await {
            match task {
                ExtractionTask::Job(job) => {
                    // Snapshot the live config per job (panel edits apply immediately).
                    let config = runtime.read().await.clone();
                    run_extraction_job(&gemini, &qdrant, &config, job, None).await;
                }
                // The channel is FIFO: resolving now means every queued job
                // ahead of this Flush has completed.
                ExtractionTask::Flush(ack) => {
                    let _ = ack.send(());
                }
            }
        }
    });
    tx
}

async fn run_extraction_job(
    gemini: &GeminiClient,
    qdrant: &Arc<QdrantClient>,
    config: &RuntimeConfig,
    job: ExtractionJob,
    meter: Option<&FlowMeter>,
) {
    if !config.memory.fact_extraction_enabled
        || !qdrant.is_available()
        || job.username.eq_ignore_ascii_case(BOT_USERNAME)
    {
        return;
    }

    // Transient Gemini trouble (503 demand spikes) exhausts the in-client
    // retries and surfaces as TransientExhausted; give those a few spaced-out
    // extra attempts here instead of silently losing the facts. Anything else
    // is a genuinely broken request — fail the job immediately.
    const EXTRACTION_TRANSIENT_RETRIES: u32 = 2; // 3 attempts total
    let mut transient_attempt = 0u32;
    let extracted = loop {
        if let Some(m) = meter {
            m.gen();
        }
        match gemini.extract_facts(&job.username, &job.text).await {
            Ok(e) => break e,
            Err(e) => {
                if is_transient_exhausted(&e) && transient_attempt < EXTRACTION_TRANSIENT_RETRIES {
                    transient_attempt += 1;
                    // 30s, then 120s.
                    let wait = Duration::from_secs(30 * 4u64.pow(transient_attempt - 1));
                    warn!(
                        "Fact extraction for post {} hit transient Gemini trouble; \
                         retry {transient_attempt}/{EXTRACTION_TRANSIENT_RETRIES} in {}s",
                        job.post_id,
                        wait.as_secs()
                    );
                    tokio::time::sleep(wait).await;
                    continue;
                }
                warn!("Fact extraction failed for post {}: {e}", job.post_id);
                return;
            }
        }
    };
    let now = unix_now();

    // ── User facts: reinforce / supersede / insert ──
    // Profile scans (posts read during a user lookup) insert at most
    // `user_scan_fact_cap` NEW facts per scan; reinforcement of already-known
    // facts does not count against the cap.
    //
    // Phase 1: drop noise and reinforce exact restatements in place. A
    // forgotten or superseded fact stays retired: an exact restatement must
    // not silently resurrect it (forget is a deliberate user action; a
    // supersede already has a live replacement fact).
    struct PendingFact {
        point_id: uuid::Uuid,
        text: String,
        category: Option<String>,
    }
    let mut pending: Vec<PendingFact> = Vec::new();
    for fact in &extracted.user_facts {
        let text = fact.fact.trim();
        if text.chars().count() < 3 || text.chars().count() > MAX_FACT_LENGTH {
            continue;
        }
        let point_id = user_fact_point_id(&job.username, text);
        match qdrant.get_user_fact(point_id).await {
            Ok(Some(existing)) => {
                if existing.active {
                    let patch = serde_json::json!({
                        "last_seen": now,
                        "times_confirmed": existing.times_confirmed.saturating_add(1),
                    });
                    if let Err(e) = qdrant.patch_user_fact(point_id, patch).await {
                        warn!("Failed to reinforce user fact {point_id}: {e}");
                    }
                }
            }
            Ok(None) => pending.push(PendingFact {
                point_id,
                text: text.to_string(),
                category: fact.category.clone(),
            }),
            Err(e) => {
                warn!("Failed to check user fact {point_id}: {e}");
            }
        }
    }

    // Phase 2: embed every new candidate in ONE batched call (the supersede
    // check needs a vector per fact; the upsert re-uses it from the cache).
    let vectors: Vec<Vec<f32>> = if pending.is_empty() {
        Vec::new()
    } else {
        let texts: Vec<String> = pending.iter().map(|p| p.text.clone()).collect();
        if let Some(m) = meter {
            m.embed(texts.len());
        }
        match qdrant.embed_texts(&texts).await {
            Ok(v) => v,
            Err(e) => {
                warn!("Failed to batch-embed user facts for {}: {e}", job.username);
                return;
            }
        }
    };

    // Phase 3: near-duplicate or contradiction -> retire the old fact, keep
    // the new. The profile-scan cap applies HERE (before supersede), so an
    // over-cap fact never retires an existing one without inserting its
    // replacement.
    let mut new_facts = 0usize;
    for (candidate, vector) in pending.into_iter().zip(vectors) {
        if job.source == ExtractionSource::ProfileScan
            && new_facts >= config.tools.user_scan_fact_cap
        {
            continue;
        }
        let PendingFact {
            point_id,
            text,
            category,
        } = candidate;
        match qdrant
            .find_similar_user_fact(
                &job.username,
                &vector,
                config.memory.user_fact_supersede_threshold,
            )
            .await
        {
            Ok(Some((old_id, old))) => {
                let patch = serde_json::json!({
                    "active": false,
                    "superseded_by": point_id.to_string(),
                });
                if let Err(e) = qdrant.patch_user_fact(old_id, patch).await {
                    warn!("Failed to supersede user fact {old_id}: {e}");
                } else {
                    info!(
                        "Superseded fact about {}: '{}' -> '{}'",
                        job.username, old.fact, text
                    );
                }
            }
            Ok(None) => {}
            Err(e) => warn!("Similarity check failed for user fact: {e}"),
        }

        let payload = UserFactPayload {
            username: job.username.clone(),
            fact: text.clone(),
            category: category
                .as_deref()
                .and_then(FactCategory::parse)
                .unwrap_or(FactCategory::Other),
            source_post_id: job.post_id,
            source_conversation_id: job.conversation_id,
            first_seen: now,
            last_seen: now,
            times_confirmed: 1,
            active: true,
            superseded_by: None,
        };
        match qdrant.upsert_user_fact(point_id, &payload).await {
            Ok(()) => {
                new_facts += 1;
                info!("Learned fact about {}: {text}", job.username);
            }
            Err(e) => warn!("Failed to store user fact for {}: {e}", job.username),
        }
    }

    // ── App facts / forget requests: conversation messages only. Profile
    // scans are scoped to the scanned user — their posts should not mint
    // app knowledge or delete anyone's memory. ──
    if job.source != ExtractionSource::Conversation {
        return;
    }

    // ── App facts: store as pending, never authoritative ──
    for app_fact in extracted.app_facts {
        let text = app_fact.fact.trim();
        if text.chars().count() < 3 || text.chars().count() > MAX_FACT_LENGTH {
            continue;
        }
        let point_id = app_fact_point_id(text);
        match qdrant.get_app_fact(point_id).await {
            // Already known — never demote a seeded/promoted fact back to pending.
            Ok(Some(_)) => {}
            Ok(None) => {
                let payload = AppFactPayload {
                    topic: app_fact.topic.clone().unwrap_or_else(|| "general".to_string()),
                    fact: text.to_string(),
                    source: AppFactSource::User,
                    status: AppFactStatus::Pending,
                    updated_at: now,
                };
                match qdrant.upsert_app_facts(&[(point_id, payload)]).await {
                    Ok(()) => info!("Stored pending app fact: {text}"),
                    Err(e) => warn!("Failed to store app fact: {e}"),
                }
            }
            Err(e) => warn!("Failed to check app fact {point_id}: {e}"),
        }
    }

    // ── Forget requests: deactivate matching facts ──
    for forget in extracted.forget {
        let text = forget.trim();
        if text.chars().count() < 3 {
            continue;
        }
        let vector = match qdrant.embed(text).await {
            Ok(v) => v,
            Err(e) => {
                warn!("Failed to embed forget request for {}: {e}", job.username);
                continue;
            }
        };
        match qdrant
            .search_user_facts_semantic(
                &job.username,
                &vector,
                config.memory.forget_similarity_threshold,
                3,
            )
            .await
        {
            Ok(matches) => {
                for (id, fact) in matches {
                    let patch = serde_json::json!({ "active": false, "last_seen": now });
                    match qdrant.patch_user_fact(id, patch).await {
                        Ok(()) => info!("Forgot fact about {}: '{}'", job.username, fact.fact),
                        Err(e) => warn!("Failed to forget user fact {id}: {e}"),
                    }
                }
            }
            Err(e) => warn!("Forget search failed for {}: {e}", job.username),
        }
    }
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn timestamp_from_post(post: &Post) -> i64 {
    post.created_at
        .as_deref()
        .and_then(parse_timestamp)
        .unwrap_or_else(unix_now)
}

fn parse_timestamp(raw: &str) -> Option<i64> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(n) = s.parse::<i64>() {
        // Unix seconds, or milliseconds if the value is absurdly large.
        return Some(if n.abs() > 100_000_000_000 {
            n / 1000
        } else {
            n
        });
    }
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.timestamp())
        .ok()
}

fn truncate_text(text: &str, max_len: usize) -> String {
    if text.chars().count() <= max_len {
        text.to_string()
    } else {
        let mut truncated: String = text.chars().take(max_len).collect();
        truncated.push('…');
        truncated
    }
}

/// Append a "Sources:" footer listing the URLs the model fetched via the
/// url_context tool. Deduped, capped, and skipped when empty. URLs stay plain
/// text — Things auto-links them in the rendered post.
fn append_sources_footer(text: &str, sources: &[String]) -> String {
    const MAX_SOURCES: usize = 5;
    let mut seen: HashSet<&str> = HashSet::new();
    let mut urls: Vec<&str> = Vec::new();
    for url in sources {
        let url = url.trim();
        if url.is_empty() || !(url.starts_with("http://") || url.starts_with("https://")) {
            continue;
        }
        if seen.insert(url) {
            urls.push(url);
            if urls.len() >= MAX_SOURCES {
                break;
            }
        }
    }
    if urls.is_empty() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len() + 96 + urls.len() * 64);
    out.push_str(text);
    out.push_str("\n\nSources:");
    for url in urls {
        out.push('\n');
        out.push_str(url);
    }
    out
}

/// Key-failover arms for one reply flow: capped so a 429 storm degrades to
/// "retry next poll" instead of multiplying the flow's cost by the pool size.
fn flow_attempt_cap(pool_size: usize, configured: usize) -> usize {
    pool_size.min(configured).max(1)
}

/// Section markers injected into prompts (memory, briefings, conversation
/// logs). A final reply containing any of them means the model recited
/// scaffold text instead of answering — that must never be posted verbatim.
const SCAFFOLD_LEAK_MARKERS: &[&str] = &[
    "[About ",
    "[Attached music",
    "[Briefing for",
    "[Conversation so far]",
    "[Conversation above]",
    "[Follow-up question",
    "[Post by",
    "[Question]",
    "Saved facts about them",
    "Their recent posts:",
];

/// Byte index of the first scaffold marker in `text`, if any.
fn scaffold_leak_at(text: &str) -> Option<usize> {
    SCAFFOLD_LEAK_MARKERS
        .iter()
        .filter_map(|m| text.find(m))
        .min()
}

/// System prompt for the one-shot rewrite used when a reply leaks scaffold.
const REWRITE_PROMPT: &str = "You are the copy editor for AskMe, a social-network reply bot. \
The draft reply below accidentally includes raw internal context (memory sections, briefings, \
fact lists, conversation logs, bracketed section headers). Rewrite it as a natural reply in the \
SAME language and tone, keeping only the actual answer to the user. Output ONLY the cleaned \
reply text — no commentary, no section headers, no bracketed labels.";

/// Minimum chars worth keeping after cutting leaked scaffold out of a reply.
const MIN_STRIPPED_REPLY_CHARS: usize = 20;

/// Guard the final reply against scaffold leakage: clean replies pass through;
/// a leaking reply gets ONE rewrite pass (extraction model — a mechanical job
/// that should not burn reply quota); if the rewrite still leaks or fails, cut
/// everything from the first marker on; if nothing worth keeping remains, the
/// reply is dropped (Err) so raw context is never posted.
async fn clean_scaffold_leak(
    gemini: &GeminiClient,
    text: String,
    meter: &FlowMeter,
) -> Result<String, ()> {
    if scaffold_leak_at(&text).is_none() {
        return Ok(text);
    }
    warn!("Reply leaked scaffold context; attempting one rewrite pass");
    meter.gen();
    match gemini.rewrite_text(REWRITE_PROMPT, &text).await {
        Ok(clean) if !clean.trim().is_empty() && scaffold_leak_at(&clean).is_none() => {
            info!("Scaffold leak cleaned by rewrite pass");
            return Ok(clean);
        }
        Ok(_) => warn!("Rewrite pass still contained scaffold; stripping instead"),
        Err(e) => warn!("Rewrite pass failed ({e}); stripping instead"),
    }
    let cut = scaffold_leak_at(&text).unwrap_or(text.len());
    let stripped = text[..cut].trim().to_string();
    if stripped.chars().count() >= MIN_STRIPPED_REPLY_CHARS {
        warn!(
            "Reply stripped of leaked scaffold ({} chars kept)",
            stripped.chars().count()
        );
        Ok(stripped)
    } else {
        Err(())
    }
}

/// Media bytes downloaded from Things, staged for the reply flow's upload.
struct DownloadedMedia {
    data: Vec<u8>,
    mime: String,
    display_name: String,
}

async fn download_media_file(state: &RwLock<AppState>, url: &str) -> Result<DownloadedMedia> {
    let (data, mime) = {
        let things = &state.read().await.things;
        things.download_media(url).await?
    };
    Ok(DownloadedMedia {
        data,
        mime: normalize_audio_mime(&mime),
        display_name: format!("media_{}", uuid::Uuid::new_v4()),
    })
}

/// Gemini's documented audio MIME list has `audio/aac` but not the m4a/mp4
/// aliases Apple's preview CDN serves for music attachments — normalize so
/// the Files API doesn't reject the upload. Anything else passes through.
fn normalize_audio_mime(mime: &str) -> String {
    let base = mime.split(';').next().unwrap_or(mime).trim();
    match base.to_ascii_lowercase().as_str() {
        "audio/mp4" | "audio/x-m4a" | "audio/m4a" | "audio/x-m4p" | "audio/m4p"
        | "audio/mp4a-latm" => "audio/aac".to_string(),
        _ => mime.to_string(),
    }
}

/// How one tool-calling reply flow ended.
enum ToolsFlowError {
    /// 429/401/403 — the lease key is marked; re-lease with the next key.
    RateLimited,
    /// Non-retryable failure (bad request, rounds exhausted, etc.).
    Failed(anyhow::Error),
}

/// Marker error for "every key in the pool is cooling down": distinguished
/// from other generation failures so the poll loop never poison-marks a
/// notification whose only problem is a temporary quota pause.
#[derive(Debug, thiserror::Error)]
#[error("all Gemini API keys are currently rate-limited")]
struct AllKeysRateLimited;

/// Run one reply generation flow with rate-limit failover across the key pool.
///
/// The flow uses a sticky key lease: media uploads and the generation that
/// references them must live in the same Gemini project. On a 429 the lease
/// key is cooled down, the next key is leased, and any media is RE-UPLOADED
/// into the new project before retrying. On success the round-robin cursor
/// advances past the flow's key, so the next reply starts on the next key.
///
/// When tools are enabled, the model can call them in a multi-round loop on
/// the same lease: each functionCall (with its thought signature) is appended
/// to the history as a model-role part, executed via the ToolContext, and its
/// result appended as a user-role part — for up to `tools.max_rounds` turns.
///
/// Returns the final text plus the URLs the model grounded the answer on
/// (url_context retrievals), used for the Sources footer.
async fn generate_with_tools_failover(
    gemini: &GeminiClient,
    system_prompt: &str,
    user_text: &str,
    media_files: &[DownloadedMedia],
    tool_ctx: &ToolContext,
) -> Result<(String, Vec<String>)> {
    // Key-failover arms: each re-runs the whole flow (rounds + uploads), so
    // an uncapped pool (16 keys) could multiply one reply's cost by 16 in a
    // 429 storm. A small cap degrades to "retry next poll" instead.
    let max_flows = flow_attempt_cap(
        gemini.pool_size(),
        tool_ctx.runtime.read().await.tools.max_flow_attempts,
    );
    let mut last_err: Option<anyhow::Error> = None;
    // Profile-scan extraction jobs already flushed at the end of a flow arm,
    // deduped by username across key-failover retries of the same reply.
    let mut flushed: HashSet<String> = HashSet::new();

    for _ in 0..max_flows {
        let lease = match gemini.acquire_lease() {
            Some(l) => l,
            None => {
                // Every key is cooling down: wait for the earliest one to thaw.
                gemini.wait_for_cooldown().await;
                continue;
            }
        };

        let mut file_uris: Vec<(String, String)> = Vec::new();
        let mut rate_limited = false;
        for file in media_files {
            tool_ctx.meter.upload();
            match gemini
                .upload_file_with(&lease, &file.data, &file.mime, &file.display_name)
                .await
            {
                Ok(uri) => {
                    info!("Media uploaded to Gemini: {uri} ({})", file.mime);
                    file_uris.push((uri, file.mime.clone()));
                }
                Err(GeminiError::RateLimited) => {
                    rate_limited = true;
                    break;
                }
                Err(GeminiError::Failed(e)) => {
                    // Per-file tolerance: skip this attachment, keep the flow.
                    warn!("Media upload failed for {}: {e}", file.display_name);
                }
            }
        }
        if rate_limited {
            continue;
        }

        // Initial contents: the user's message plus uploaded media as file
        // parts. Tool-round results append to this history.
        let mut contents: Vec<Content> = vec![Content {
            role: "user".to_string(),
            parts: {
                let mut parts = vec![Part::Text {
                    text: user_text.to_string(),
                    thought_signature: None,
                }];
                for (uri, mime) in &file_uris {
                    parts.push(Part::FileData {
                        file_data: FileData {
                            mime_type: mime.clone(),
                            file_uri: uri.clone(),
                        },
                    });
                }
                parts
            },
        }];

        let (tools_enabled, max_rounds, url_context_enabled) = {
            let tools_enabled = tool_ctx.tools_enabled().await;
            let max_rounds = tool_ctx.max_tool_rounds().await;
            let url_context_enabled = tool_ctx.url_context_enabled().await;
            (tools_enabled, max_rounds, url_context_enabled)
        };

        match run_tool_rounds(
            gemini,
            &lease,
            system_prompt,
            &mut contents,
            tool_ctx,
            tools_enabled,
            max_rounds,
            url_context_enabled,
        )
        .await
        {
            Ok((text, sources)) => {
                gemini.flow_success(&lease);
                flush_pending_profile_scans(tool_ctx, &mut flushed).await;
                return Ok((text, sources));
            }
            Err(ToolsFlowError::RateLimited) => {
                flush_pending_profile_scans(tool_ctx, &mut flushed).await;
                continue;
            }
            Err(ToolsFlowError::Failed(e)) => {
                flush_pending_profile_scans(tool_ctx, &mut flushed).await;
                last_err = Some(e);
                break;
            }
        }
    }

    Err(last_err.unwrap_or_else(|| AllKeysRateLimited.into()))
}

/// The tool-calling loop on one lease: ask the model for a turn, execute any
/// function calls it requests, feed the results back, repeat until it produces
/// final text or the round budget is exhausted. Returns the final text plus
/// every URL the model fetched via the url_context tool across the rounds.
async fn run_tool_rounds(
    gemini: &GeminiClient,
    lease: &KeyLease,
    system_prompt: &str,
    contents: &mut Vec<Content>,
    tool_ctx: &ToolContext,
    tools_enabled: bool,
    max_rounds: usize,
    url_context_enabled: bool,
) -> Result<(String, Vec<String>), ToolsFlowError> {
    let declarations = if tools_enabled {
        tool_declarations(url_context_enabled)
    } else {
        Vec::new()
    };

    // Per-flow cache of executed calls: if the model repeats an identical call
    // (same name + args) in a later round, we answer with a one-line marker
    // instead of re-executing and re-echoing a large result into the history.
    // get_current_time is never cached (it must stay fresh).
    let mut call_cache: HashMap<String, Value> = HashMap::new();
    let mut sources: Vec<String> = Vec::new();

    for _ in 0..max_rounds {
        tool_ctx.meter.gen();
        let turn = match gemini
            .generate_turn_with(lease, system_prompt, contents, &declarations)
            .await
        {
            Ok(t) => t,
            Err(GeminiError::RateLimited) => return Err(ToolsFlowError::RateLimited),
            Err(GeminiError::Failed(e)) => return Err(ToolsFlowError::Failed(e)),
        };
        sources.extend(turn.retrieved_urls.clone());

        if turn.function_calls.is_empty() {
            match turn.text {
                Some(t) => return Ok((t, sources)),
                None => {
                    // A turn of only server-side tool parts (toolCall/
                    // toolResponse) with no final answer yet: circulate the
                    // parts and give the model another round.
                    append_tool_turn(turn, contents, tool_ctx, &mut call_cache).await;
                    continue;
                }
            }
        }

        let only_cached = append_tool_turn(turn, contents, tool_ctx, &mut call_cache).await;

        // Brief the model on any user it just looked up (saved facts + recent
        // posts, fresh facts extracted inline) IN THE SAME ROUND, so the next
        // turn is already the final answer — no extra briefing round.
        if let Some(briefing) = build_user_briefing(gemini, tool_ctx).await {
            append_user_parts(contents, briefing.parts);
        }

        // Every call this round was an identical repeat of an earlier one —
        // the model gained nothing new, so further rounds would only repeat.
        // Skip straight to the final no-tools chance below.
        if only_cached {
            break;
        }
    }

    // The round budget is spent but the model still wanted tools. Execute
    // nothing more: give it ONE final chance to answer with everything it has
    // collected — tools withheld so it cannot keep looping. This guarantees a
    // reply instead of dropping the notification after exhausting rounds.
    if let Some(briefing) = build_user_briefing(gemini, tool_ctx).await {
        append_user_parts(contents, briefing.parts);
    }
    tool_ctx.meter.gen();
    let final_turn = match gemini
        .generate_turn_with(lease, system_prompt, contents, &[])
        .await
    {
        Ok(t) => t,
        Err(GeminiError::RateLimited) => return Err(ToolsFlowError::RateLimited),
        Err(GeminiError::Failed(e)) => return Err(ToolsFlowError::Failed(e)),
    };
    if final_turn.function_calls.is_empty() {
        if let Some(text) = final_turn.text {
            return Ok((text, sources));
        }
    }
    Err(ToolsFlowError::Failed(anyhow::anyhow!(
        "model exhausted {max_rounds} tool rounds without a final answer"
    )))
}

/// Resolve a username to a numeric user id via the search API (exact username
/// match only) — a cheap Things call that lets the briefing fetch posts and
/// run the inline extraction even when the model only asked for saved facts.
async fn resolve_user_id_by_username(tool_ctx: &ToolContext, username: &str) -> Option<u64> {
    match tool_ctx.things.search_users(username, 5).await {
        Ok(rows) => rows
            .into_iter()
            .find(|r| {
                r.username
                    .as_deref()
                    .is_some_and(|u| u.eq_ignore_ascii_case(username))
            })
            .map(|r| r.id),
        Err(e) => {
            warn!("build_user_briefing: could not resolve id for @{username}: {e}");
            None
        }
    }
}

/// Auto-brief the model on the most recent user it looked up during this
/// reply flow: saved facts + recent posts, plus a fresh inline fact
/// extraction when posts were fetched, so the final reply includes facts
/// gathered BEFORE it is written. Returns the user-role part to append, or
/// None when there is nothing (no un-briefed subject, or nothing known).
async fn build_user_briefing(gemini: &GeminiClient, tool_ctx: &ToolContext) -> Option<Content> {
    let (username, user_id, posts) = {
        let mut subjects = tool_ctx.flow_subjects.lock().unwrap();
        if subjects.iter().filter(|s| s.briefed).count() >= MAX_BRIEFED_SUBJECTS {
            return None;
        }
        // A subject may carry only a user_id (e.g. a get_user_posts lookup of
        // a user with zero visible posts returned no username) — resolve the
        // username below instead of skipping the briefing entirely.
        let subject = subjects
            .iter_mut()
            .rev()
            .find(|s| !s.briefed && (s.username.is_some() || s.user_id.is_some()))?;
        let username = subject.username.clone();
        let user_id = subject.user_id;
        let posts = subject.posts.clone();
        subject.briefed = true;
        (username, user_id, posts)
    };

    let username = match username {
        Some(u) => u,
        None => {
            let id = user_id?;
            match tool_ctx.things.get_user(id).await {
                Ok(p) => {
                    let resolved = p.username?;
                    // Write it back so the extraction dedup below (which finds
                    // subjects by username) matches this subject.
                    let mut subjects = tool_ctx.flow_subjects.lock().unwrap();
                    if let Some(s) = subjects.iter_mut().find(|s| s.user_id == Some(id)) {
                        s.username = Some(resolved.clone());
                    }
                    resolved
                }
                Err(e) => {
                    warn!("build_user_briefing: could not resolve username for user {id}: {e}");
                    return None;
                }
            }
        }
    };
    if username.eq_ignore_ascii_case(BOT_USERNAME) {
        return None;
    }

    let config = tool_ctx.runtime.read().await.clone();

    // Posts: reuse the in-flow get_user_posts result, else fetch (bounded).
    let mut posts_value = posts;
    if posts_value.is_none() {
        // A facts-only subject (get_user_facts lookup) carries no user_id —
        // resolve it (cheap Things search, no Gemini call) so the briefing
        // can include posts and run the inline extraction instead of
        // degrading to facts-only.
        let user_id = match user_id {
            Some(id) => Some(id),
            None => resolve_user_id_by_username(tool_ctx, &username).await,
        };
        if let Some(id) = user_id {
            if let Ok(page) = tool_ctx
                .things
                .get_user_posts(id, config.tools.user_scan_posts_limit)
                .await
            {
                posts_value = Some(json!(
                    page.data.unwrap_or_default()
                        .iter()
                        .map(|p| {
                            json!({
                                "id": p.id,
                                "created_at": p.created_at,
                                "post_type": p.post_type,
                                "content": p.content_text(),
                            })
                        })
                        .collect::<Vec<_>>()
                ));
            }
        }
    }
    let post_list: Vec<String> = posts_value
        .as_ref()
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|p| p.get("content").and_then(|c| c.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // Fresh facts: run the profile-scan extraction inline (at most once per
    // reply flow) so facts learned from the posts already exist in memory
    // before the final reply is written.
    let mut should_extract = config.memory.fact_extraction_enabled
        && tool_ctx.qdrant.is_available()
        && !post_list.is_empty();
    if should_extract {
        let mut subjects = tool_ctx.flow_subjects.lock().unwrap();
        if subjects.iter().any(|s| s.extracted) {
            should_extract = false;
        } else if let Some(s) = subjects.iter_mut().find(|s| {
            s.username
                .as_deref()
                .is_some_and(|n| n.eq_ignore_ascii_case(&username))
        }) {
            s.extracted = true;
        } else {
            should_extract = false;
        }
    }
    if should_extract {
        let contents: Vec<&str> = post_list.iter().map(|s| s.as_str()).collect();
        let text = crate::tools::profile_scan_text(&username, &contents);
        if !text.is_empty() {
            let post_id = posts_value
                .as_ref()
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|p| p.get("id"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let job = ExtractionJob {
                username: username.clone(),
                text,
                post_id,
                conversation_id: 0,
                source: ExtractionSource::ProfileScan,
            };
            run_extraction_job(gemini, &tool_ctx.qdrant, &config, job, Some(tool_ctx.meter.as_ref())).await;
        }
    }

    // Saved facts (re-read after the inline extraction, so fresh facts show).
    let facts: Vec<String> = if tool_ctx.qdrant.is_available() {
        match tool_ctx
            .qdrant
            .list_user_facts(&username, config.memory.user_facts_limit.max(20))
            .await
        {
            Ok(list) => list.into_iter().map(|(_, f)| f.fact).collect(),
            Err(e) => {
                warn!("build_user_briefing: list_user_facts({username}) failed: {e}");
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    if facts.is_empty() && post_list.is_empty() {
        return None;
    }
    let text = user_briefing_text(&username, &facts, &post_list);
    info!(
        "Injected user briefing for @{username} ({} facts, {} posts)",
        facts.len(),
        post_list.len()
    );
    Some(Content {
        role: "user".to_string(),
        parts: vec![Part::Text {
            text,
            thought_signature: None,
        }],
    })
}

/// The briefing context part text: saved facts + numbered recent posts + the
/// instruction to base the reply on them (facts and a short post summary).
fn user_briefing_text(username: &str, facts: &[String], posts: &[String]) -> String {
    let mut out = format!("[Briefing for @{username}]\n");
    if !facts.is_empty() {
        out.push_str("Saved facts about them:\n");
        for f in facts {
            out.push_str(&format!("- {f}\n"));
        }
    }
    if !posts.is_empty() {
        out.push_str("Their recent posts:\n");
        for (i, p) in posts.iter().enumerate() {
            out.push_str(&format!("{}. {p}\n", i + 1));
        }
    }
    out.push_str("\nBase your reply on the facts above and include a short summary of their recent posts.");
    out
}

/// End-of-flow safety net: queue the profile-scan extraction for any subject
/// whose posts were fetched during the flow but whose facts were not already
/// extracted inline by the briefing step (e.g. the flow failed or no briefing
/// ran). `flushed` dedupes across key-failover retries of the same flow.
async fn flush_pending_profile_scans(tool_ctx: &ToolContext, flushed: &mut HashSet<String>) {
    let subjects = tool_ctx.flow_subjects.lock().unwrap().clone();
    for subject in subjects {
        let Some(username) = subject.username else { continue };
        let Some(posts) = subject.posts else { continue };
        if subject.extracted || username.eq_ignore_ascii_case(BOT_USERNAME) {
            continue;
        }
        if !flushed.insert(username.to_ascii_lowercase()) {
            continue;
        }
        let contents: Vec<&str> = posts
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|p| p.get("content").and_then(|c| c.as_str()))
                    .collect()
            })
            .unwrap_or_default();
        let text = crate::tools::profile_scan_text(&username, &contents);
        if text.is_empty() {
            continue;
        }
        let post_id = posts
            .as_array()
            .and_then(|a| a.first())
            .and_then(|p| p.get("id"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let job = ExtractionJob {
            username: username.clone(),
            text,
            post_id,
            conversation_id: 0,
            source: ExtractionSource::ProfileScan,
        };
        if tool_ctx.extraction_tx.send(ExtractionTask::Job(job)).is_err() {
            warn!("Extraction worker gone; profile-scan facts for {username} not queued");
        } else {
            info!("Queued profile-scan fact extraction for {username}");
        }
    }
}

/// Append the model's functionCall parts to the history, execute every call
/// via the ToolContext, and append the results as user-role parts.
///
/// Cost guards (quality-neutral — the model gets the same data either way):
/// - identical calls within one turn are executed ONCE; repeats get a short
///   "cached" marker instead of a duplicated full result,
/// - a call repeated in a later round (same name + args) also gets the marker
///   instead of re-executing and re-echoing a large result into the history —
///   the original result is still in the conversation from its first round.
///
/// A turn with no function calls at all (only server-side toolCall/
/// toolResponse parts) appends NO user content: an empty `parts` array is
/// rejected by the API and would kill the whole reply flow.
///
/// Returns true when the turn HAD function calls but every one was an
/// identical repeat (answered with the cached marker, zero fresh executions)
/// — the model gained nothing new this round.
async fn append_tool_turn(
    turn: GenerateTurn,
    contents: &mut Vec<Content>,
    tool_ctx: &ToolContext,
    call_cache: &mut HashMap<String, Value>,
) -> bool {
    // Append the model's parts verbatim (thought signatures and ids intact).
    // With tool context circulation this must include the server-side
    // toolCall/toolResponse parts, not just the functionCall parts — and even
    // interim text parts, which the API counts into the conversation.
    contents.push(Content {
        role: "model".to_string(),
        parts: turn.raw_parts.clone(),
    });

    // Pair each function call with its API-assigned id by position (parse_turn
    // pushes raw_parts and function_calls in the same order). Each call's own
    // id must echo on ITS response — required for tool context circulation.
    let mut call_ids: Vec<Option<String>> = Vec::with_capacity(turn.function_calls.len());
    for part in &turn.raw_parts {
        if let Part::FunctionCall {
            function_call, ..
        } = part
        {
            call_ids.push(function_call.id.clone());
        }
    }

    // Plan: which calls actually need fresh execution?
    let mut first_seen: HashMap<String, usize> = HashMap::new();
    let mut scheduled: Vec<(String, String, Value)> = Vec::new(); // (key, name, args)
    for (i, call) in turn.function_calls.iter().enumerate() {
        let key = call_cache_key(&call.name, &call.args);
        if first_seen.contains_key(&key) {
            continue; // duplicate within this turn
        }
        first_seen.insert(key.clone(), i);
        let cacheable = call.name != "get_current_time";
        if cacheable && call_cache.contains_key(&key) {
            continue; // already executed in an earlier round
        }
        scheduled.push((key, call.name.clone(), call.args.clone()));
    }
    let nothing_new = !turn.function_calls.is_empty() && scheduled.is_empty();

    // Execute the unique fresh calls and record their results.
    let mut turn_results: HashMap<String, Value> = HashMap::new();
    for (key, name, args) in scheduled {
        let result = tool_ctx.execute(&name, &args).await;
        if name != "get_current_time" {
            call_cache.insert(key.clone(), result.clone());
        }
        turn_results.insert(key, result);
    }

    // Emit one response part per call, in the original order.
    let mut response_parts: Vec<Part> = Vec::new();
    for (i, call) in turn.function_calls.iter().enumerate() {
        let key = call_cache_key(&call.name, &call.args);
        let response = if first_seen.get(&key) == Some(&i) && turn_results.contains_key(&key) {
            // Fresh result from this turn's execution.
            turn_results.get(&key).cloned().unwrap_or_default()
        } else {
            // Duplicate within this turn, or a repeat of an earlier round's
            // call: the full result is already in the history — a marker is
            // enough and keeps the context small.
            cached_call_marker(&call.name)
        };
        info!(
            "Tool {}({}) -> {}",
            call.name,
            truncate_text(&call.args.to_string(), 200),
            truncate_text(&response.to_string(), 300)
        );
        response_parts.push(Part::FunctionResponse {
            function_response: FunctionResponseData {
                name: call.name.clone(),
                response,
                id: call_ids.get(i).cloned().flatten(),
            },
        });
    }
    // Server-side-only turns (no function calls) must not produce an empty
    // user content — the API rejects empty parts arrays.
    if !response_parts.is_empty() {
        contents.push(Content {
            role: "user".to_string(),
            parts: response_parts,
        });
    }
    nothing_new
}

/// The one-line answer for a call whose full result is already in the history.
fn cached_call_marker(name: &str) -> Value {
    json!({
        "cached": true,
        "note": format!(
            "This exact call to {name}() with the same arguments was already executed \
             earlier in this conversation; the result shown above is unchanged."
        ),
    })
}

/// Append parts as a user-role content, merging into the trailing user
/// content when there is one (keeps the history alternating user/model).
fn append_user_parts(contents: &mut Vec<Content>, parts: Vec<Part>) {
    if let Some(last) = contents.last_mut() {
        if last.role == "user" {
            last.parts.extend(parts);
            return;
        }
    }
    contents.push(Content {
        role: "user".to_string(),
        parts,
    });
}

/// Stable key identifying one tool invocation (name + exact arguments).
fn call_cache_key(name: &str, args: &Value) -> String {
    serde_json::to_string(&(name, args)).unwrap_or_else(|_| format!("{name}:{args}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_abort_valve_boundaries() {
        // Nothing checked yet / tiny samples never abort.
        assert!(!sweep_should_abort(0, 0));
        assert!(!sweep_should_abort(SWEEP_ABORT_MIN_CHECKED - 1, SWEEP_ABORT_MIN_CHECKED - 1));
        // Exactly at the ratio is still allowed (abort is strictly above).
        assert!(!sweep_should_abort(100, 90));
        // Clearly suspicious: almost everything 404 -> refuse to purge.
        assert!(sweep_should_abort(100, 91));
        assert!(sweep_should_abort(SWEEP_ABORT_MIN_CHECKED, SWEEP_ABORT_MIN_CHECKED));
        // Healthy mixes never abort.
        assert!(!sweep_should_abort(397, 226));
        assert!(!sweep_should_abort(2000, 0));
    }

    #[test]
    fn strip_mention_basic() {
        assert_eq!(strip_mention("@AskMe what is this?"), "what is this?");
        assert_eq!(strip_mention("what is this? @AskMe"), "what is this?");
        assert_eq!(strip_mention("hey @askme what is this?"), "hey what is this?");
        assert_eq!(strip_mention("@ASKME hi"), "hi");
        assert_eq!(strip_mention("@AskMe @askme hi"), "hi");
        assert_eq!(strip_mention("no mention here"), "no mention here");
    }

    #[test]
    fn strip_mention_keeps_text_on_both_sides() {
        assert_eq!(
            strip_mention("please explain @AskMe this photo"),
            "please explain this photo"
        );
    }

    #[test]
    fn strip_mention_does_not_mangle_longer_handles() {
        assert_eq!(strip_mention("hi @AskMeBot"), "hi @AskMeBot");
        assert_eq!(strip_mention("email askme@example.com"), "email askme@example.com");
    }

    #[test]
    fn sources_footer_appends_deduped_capped_urls() {
        assert_eq!(append_sources_footer("answer", &[]), "answer");
        assert_eq!(
            append_sources_footer("answer", &["https://a.example".to_string()]),
            "answer\n\nSources:\nhttps://a.example"
        );
        let dupes = vec![
            "https://a.example".to_string(),
            "https://a.example".to_string(),
            "https://b.example".to_string(),
        ];
        let out = append_sources_footer("answer", &dupes);
        assert!(out.matches("https://a.example").count() == 1);
        assert!(out.contains("https://b.example"));

        let many: Vec<String> = (0..9).map(|i| format!("https://s{i}.example")).collect();
        assert_eq!(append_sources_footer("a", &many).lines().count(), 8, "1 text + 1 blank + 1 header + 5 urls");
        assert!(!append_sources_footer("a", &many).contains("https://s5.example"));
    }

    #[test]
    fn reply_error_422_stays_retryable() {
        // 422: the server validated and REJECTED the payload — nothing was
        // committed, and the text is regenerated on every attempt, so the
        // notification should be retried, not poison-marked.
        let err = anyhow::Error::new(ClientRejected {
            status: reqwest::StatusCode::UNPROCESSABLE_ENTITY,
            context: "Reply error".to_string(),
            body: "comment too long".to_string(),
        });
        assert!(matches!(classify_reply_error(&err), ReplyFailure::Retryable));
        // Still detected when wrapped in context layers.
        let err = err.context("posting reply failed");
        assert!(matches!(classify_reply_error(&err), ReplyFailure::Retryable));
    }

    #[test]
    fn reply_error_other_4xx_is_permanent() {
        // 403/404/400: the target definitively refused — retrying re-runs the
        // whole generate pipeline for nothing.
        for status in [
            reqwest::StatusCode::FORBIDDEN,
            reqwest::StatusCode::NOT_FOUND,
            reqwest::StatusCode::BAD_REQUEST,
            reqwest::StatusCode::GONE,
        ] {
            let err = anyhow::Error::new(ClientRejected {
                status,
                context: "Reply error".to_string(),
                body: String::new(),
            });
            assert!(
                matches!(classify_reply_error(&err), ReplyFailure::Permanent),
                "{status} must be permanent"
            );
        }
    }

    #[test]
    fn reply_error_not_safe_to_retry_when_ambiguous() {
        // 5xx/timeout-style failures may have committed server-side — retrying
        // would risk a duplicate reply.
        let err = anyhow::anyhow!("Reply error (HTTP 500): Internal Server Error");
        assert!(matches!(classify_reply_error(&err), ReplyFailure::Ambiguous));
    }

    #[test]
    fn scaffold_leak_detection() {
        assert!(scaffold_leak_at("إجابة طبيعية تماماً").is_none());
        assert!(scaffold_leak_at("a perfectly normal answer").is_none());
        let leaked = "تفضل:\n[About khalid — long-term memory]\n- lives in Riyadh";
        assert_eq!(scaffold_leak_at(leaked), Some(leaked.find("[About ").unwrap()));
        assert!(scaffold_leak_at("Saved facts about them:\n- x").is_some());
        assert!(scaffold_leak_at("[Conversation so far]\nu: hi").is_some());
        assert!(scaffold_leak_at("Their recent posts:\n1. hello").is_some());
        // Earliest marker wins.
        let both = "[Question] hi\n[Briefing for @x]";
        assert_eq!(scaffold_leak_at(both), Some(0));
    }

    #[test]
    fn extraction_gate_skips_trivial_mentions() {
        assert!(passes_extraction_gate("وش تعرف عن خالد؟", 0), "0 = extract everything");
        assert!(!passes_extraction_gate("شكراً", 24));
        assert!(!passes_extraction_gate("ok 👍", 24));
        assert!(passes_extraction_gate("اسمي فهد وأسكن في الرياض وأعمل مدرساً", 24));
        assert!(passes_extraction_gate("   padded question text   ", 10));
    }

    #[test]
    fn flow_attempt_cap_bounds_failover_storms() {
        assert_eq!(flow_attempt_cap(15, 3), 3, "big pool capped to the configured arms");
        assert_eq!(flow_attempt_cap(2, 3), 2, "small pool caps at pool size");
        assert_eq!(flow_attempt_cap(0, 3), 1, "empty pool still gets one attempt");
    }

    #[test]
    fn flow_meter_counts_and_summarizes() {
        let m = FlowMeter::default();
        m.gen();
        m.gen();
        m.upload();
        m.embed(3);
        assert_eq!(m.summary(), "2 generate, 1 upload, 1 embed (3 texts)");
    }

    #[test]
    fn reply_text_never_exceeds_things_comment_limit() {
        // Things rejects comment text over 2000 chars (HTTP 422). The cap plus
        // the truncation ellipsis must always land within the limit — even for
        // a huge model answer with a full Sources footer.
        const { assert!(MAX_RESPONSE_LENGTH < 2000) };
        let long = "**Riyadh** ".repeat(500); // ~5500 chars of bold markup
        let sources: Vec<String> = (0..5)
            .map(|i| format!("https://source-{i}.example/very/long/url/path"))
            .collect();
        let combined = append_sources_footer(&long, &sources);
        let (text, entities) = build_reply_with_entities(&combined, MAX_RESPONSE_LENGTH);
        assert!(text.chars().count() <= 2000);
        assert!(text.ends_with('…'));
        let len = text.chars().count() as u64;
        for e in &entities {
            assert!(e.offset + e.length <= len, "entity span must stay in bounds");
        }
    }

    #[test]
    fn sources_footer_skips_invalid_urls() {
        assert_eq!(
            append_sources_footer("a", &["not-a-url".to_string(), "".to_string(), "ftp://x".to_string()]),
            "a"
        );
    }

    #[test]
    fn user_briefing_text_lists_facts_posts_and_instruction() {
        let facts = vec!["thinks the app is beautiful and clear.".to_string()];
        let posts = vec!["بسم الله الرحمن الرحيم".to_string(), "تعالوا في عيديات".to_string()];
        let text = user_briefing_text("Fahad", &facts, &posts);
        assert!(text.contains("[Briefing for @Fahad]"));
        assert!(text.contains("Saved facts about them:"));
        assert!(text.contains("- thinks the app is beautiful and clear."));
        assert!(text.contains("Their recent posts:"));
        assert!(text.contains("1. بسم الله الرحمن الرحيم"));
        assert!(text.contains("2. تعالوا في عيديات"));
        assert!(text.contains("include a short summary of their recent posts"));
    }

    #[test]
    fn user_briefing_text_facts_only_when_no_posts() {
        let facts = vec!["lives in Riyadh".to_string()];
        let text = user_briefing_text("X", &facts, &[]);
        assert!(text.contains("Saved facts about them:"));
        assert!(!text.contains("Their recent posts:"));
        assert!(text.contains("include a short summary"));
    }

    #[test]
    fn call_cache_key_orders_and_unescapes() {
        let a = call_cache_key("get_user_profile", &json!({"user_id": 309}));
        let b = call_cache_key("get_user_profile", &json!({"user_id": 262}));
        assert_ne!(a, b);
        let c = call_cache_key("get_user_posts", &json!({"user_id": 309, "limit": 10}));
        assert_ne!(a, c);
        let d = call_cache_key("get_user_posts", &json!({"limit": 10, "user_id": 309}));
        assert_eq!(c, d);
        assert_eq!(
            call_cache_key("get_current_time", &json!({})),
            call_cache_key("get_current_time", &json!({}))
        );
    }

    #[test]
    fn strip_mention_unicode_safe() {
        // Turkish 'İ' lowercases to 'i' + combining dot (byte length changes).
        // A lowercase-then-slice implementation would panic here; ours must not.
        assert_eq!(
            strip_mention("İstanbul hakkında @askme ne düşünüyorsun"),
            "İstanbul hakkında ne düşünüyorsun"
        );
        // Mention followed directly by an Arabic (non-ASCII) letter is still a mention.
        assert_eq!(strip_mention("@AskMeهلا"), "هلا");
        // Arabic question with the mention at the end keeps the question.
        assert_eq!(strip_mention("ما هي عاصمة فرنسا؟ @AskMe"), "ما هي عاصمة فرنسا؟");
    }

    #[test]
    fn strip_mention_only_mention() {
        assert_eq!(strip_mention("@AskMe"), "");
        assert_eq!(strip_mention("  @AskMe  "), "");
    }

    #[test]
    fn collapse_spaces_works() {
        assert_eq!(collapse_spaces("a  b\t\tc"), "a b c");
        assert_eq!(collapse_spaces("  spaced  "), "spaced");
        assert_eq!(collapse_spaces("keep\nnewlines  x"), "keep\nnewlines x");
    }

    #[test]
    fn parse_test_post_id_works() {
        let to_args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        assert_eq!(parse_test_post_id(&to_args(&["bot", "--test-post", "123"])), Some(123));
        assert_eq!(
            parse_test_post_id(&to_args(&["bot", "--test-post", "123", "--post"])),
            Some(123)
        );
        assert_eq!(
            parse_test_post_id(&to_args(&["bot", "--test-post", "123", "--prompt", "hi"])),
            Some(123)
        );
        assert_eq!(parse_test_post_id(&to_args(&["bot", "--test-post"])), None);
        assert_eq!(parse_test_post_id(&to_args(&["bot", "--test-post", "--post"])), None);
        assert_eq!(parse_test_post_id(&to_args(&["bot"])), None);
    }

    #[test]
    fn parse_timestamp_formats() {
        assert_eq!(parse_timestamp("1700000000"), Some(1700000000));
        assert_eq!(parse_timestamp("1700000000000"), Some(1700000000)); // millis
        assert_eq!(parse_timestamp("2023-11-14T22:13:20Z"), Some(1700000000));
        assert_eq!(parse_timestamp(" 1700000000 "), Some(1700000000));
        assert_eq!(parse_timestamp("not a date"), None);
        assert_eq!(parse_timestamp(""), None);
    }

    fn notification(
        id: u64,
        nt: Option<&str>,
        group: Option<&str>,
        post_data: Option<Post>,
        reply_post_data: Option<Post>,
        original_post_data: Option<Post>,
    ) -> Notification {
        Notification {
            id,
            notification_type: nt.map(String::from),
            group: group.map(String::from),
            body: None,
            post_data,
            original_post_data,
            reply_post_data,
            is_read: None,
            created_at: None,
            action_url: None,
        }
    }

    fn post_with_id(id: u64) -> Post {
        Post {
            id: Some(id),
            post_id: None,
            user: None,
            parent_id: None,
            post_comment: None,
            media: None,
            audio: None,
            music: None,
            post_type: None,
            created_at: None,
            expires_at: None,
            content: None,
            comments: None,
            images: None,
            image: None,
            attachments: None,
            entities: None,
        }
    }

    #[test]
    fn mention_notification_detection() {
        assert!(is_mention_notification(&notification(
            1,
            Some("user_mention"),
            None,
            None,
            None,
            None
        )));
        assert!(is_mention_notification(&notification(
            1,
            Some("mention"),
            None,
            None,
            None,
            None
        )));
        assert!(is_mention_notification(&notification(
            1,
            None,
            Some("mentions"),
            None,
            None,
            None
        )));
        assert!(!is_mention_notification(&notification(
            1,
            Some("post_reply"),
            None,
            None,
            None,
            None
        )));
    }

    #[test]
    fn notification_post_id_routing() {
        // Mention: post_data wins.
        let n = notification(
            1,
            Some("user_mention"),
            None,
            Some(post_with_id(10)),
            Some(post_with_id(20)),
            None,
        );
        assert_eq!(notification_post_id(&n), Some(10));

        // Reply: reply_post_data wins, original is the fallback.
        let n = notification(
            1,
            Some("post_reply"),
            None,
            None,
            Some(post_with_id(30)),
            Some(post_with_id(40)),
        );
        assert_eq!(notification_post_id(&n), Some(30));
        let n = notification(
            1,
            Some("post_reply"),
            None,
            None,
            None,
            Some(post_with_id(40)),
        );
        assert_eq!(notification_post_id(&n), Some(40));

        // Other types: post_data, then reply_post_data.
        let n = notification(
            1,
            Some("like"),
            None,
            Some(post_with_id(50)),
            None,
            None,
        );
        assert_eq!(notification_post_id(&n), Some(50));
        let n = notification(1, Some("like"), None, None, Some(post_with_id(60)), None);
        assert_eq!(notification_post_id(&n), Some(60));
        let n = notification(1, Some("like"), None, None, None, None);
        assert_eq!(notification_post_id(&n), None);
    }

    // ── Scoped memory: section formatting & conversation resolution ──

    fn user_fact(fact: &str) -> UserFactPayload {
        UserFactPayload {
            username: "khalid".to_string(),
            fact: fact.to_string(),
            category: FactCategory::Other,
            source_post_id: 1,
            source_conversation_id: 1,
            first_seen: 100,
            last_seen: 200,
            times_confirmed: 1,
            active: true,
            superseded_by: None,
        }
    }

    fn app_fact(fact: &str) -> AppFactPayload {
        AppFactPayload {
            topic: "platform".to_string(),
            fact: fact.to_string(),
            source: AppFactSource::Seed,
            status: AppFactStatus::Active,
            updated_at: 100,
        }
    }

    #[test]
    fn user_profile_section_only_when_populated() {
        assert_eq!(format_user_profile_section("khalid", &[]), "");
        let section = format_user_profile_section(
            "khalid",
            &[user_fact("lives in Riyadh"), user_fact("is a teacher")],
        );
        assert_eq!(
            section,
            "\n[About khalid — long-term memory]\n- lives in Riyadh\n- is a teacher"
        );
    }

    #[test]
    fn app_knowledge_section_only_when_populated() {
        assert_eq!(format_app_knowledge_section(&[]), "");
        let section = format_app_knowledge_section(&[app_fact("Things is a social network")]);
        assert_eq!(
            section,
            "\n[About Things — app knowledge]\n- Things is a social network"
        );
    }

    /// AppState backed by an unavailable Qdrant: every memory read fails, so
    /// conversation resolution must fall back to self-rooted ids.
    pub(crate) fn test_state() -> Arc<RwLock<AppState>> {
        let gemini = GeminiClient::new("test-key".to_string());
        let qdrant = Arc::new(QdrantClient::unavailable(Arc::new(gemini.clone()), 4));
        let memory_writer = qdrant.spawn_writer(5, Duration::from_millis(50));
        let runtime = Arc::new(RwLock::new(RuntimeConfig {
            memory: config::MemoryConfig {
                user_facts_limit: 8,
                app_knowledge_limit: 3,
                app_knowledge_min_score: 0.72,
                user_fact_supersede_threshold: 0.85,
                forget_similarity_threshold: 0.80,
                fact_extraction_enabled: true,
                extraction_min_chars: 24,
            },
            context_depth_limit: 20,
            tools: config::ToolsConfig {
                enabled: true,
                max_rounds: 6,
                web_fetch_max_bytes: 512_000,
                web_fetch_timeout_secs: 15,
                user_scan_posts_limit: 10,
                user_scan_fact_cap: 3,
                url_context_enabled: true,
                max_flow_attempts: 3,
            },
        }));
        let extraction_writer =
            spawn_extraction_worker(gemini.clone(), qdrant.clone(), runtime.clone());
        Arc::new(RwLock::new(AppState {
            things: ThingsClient::new(),
            gemini,
            qdrant,
            memory_writer,
            extraction_writer,
            runtime,
            system_prompt: String::new(),
            processed: HashSet::new(),
            failures: HashMap::new(),
        }))
    }

    #[tokio::test]
    async fn resolve_conversation_roots_new_mentions_at_themselves() {
        let state = test_state();
        let mut post = post_with_id(100);
        post.parent_id = Some(50);
        let post_data = models::PostData {
            post: Some(post.clone()),
            parent: None,
            quoted: None,
        };
        // A fresh @mention NEVER inherits its parent's conversation — it is
        // always the root of its own isolated conversation.
        let (id, chain) = resolve_conversation(&state, &post_data, &post, false).await;
        assert_eq!(id, 100);
        // Memory is down in this fixture, so no ancestor walk can happen.
        assert!(chain.is_empty());
    }

    #[tokio::test]
    async fn resolve_conversation_roots_parentless_posts_at_themselves() {
        let state = test_state();
        let post = post_with_id(100);
        let post_data = models::PostData {
            post: Some(post.clone()),
            parent: None,
            quoted: None,
        };
        let (id, chain) = resolve_conversation(&state, &post_data, &post, false).await;
        assert_eq!(id, 100);
        assert!(chain.is_empty());
    }

    #[tokio::test]
    async fn resolve_conversation_follow_up_falls_back_to_self_when_memory_down() {
        let state = test_state();
        let mut post = post_with_id(100);
        post.parent_id = Some(50);
        let post_data = models::PostData {
            post: Some(post.clone()),
            parent: None,
            quoted: None,
        };
        // Follow-up whose parent cannot be looked up (memory down) starts its
        // own conversation instead of failing.
        let (id, _) = resolve_conversation(&state, &post_data, &post, true).await;
        assert_eq!(id, 100);
    }

    #[test]
    fn ancestor_chain_formats_oldest_first_with_limit() {
        assert_eq!(format_ancestor_chain(&[], 20), "");
        assert_eq!(
            format_ancestor_chain(&[("a".to_string(), "root".to_string())], 0),
            ""
        );
        let chain = vec![
            ("a".to_string(), "root".to_string()),
            ("b".to_string(), "middle".to_string()),
            ("c".to_string(), "direct parent".to_string()),
        ];
        assert_eq!(
            format_ancestor_chain(&chain, 20),
            "\n[Conversation above]\na: root\nb: middle\nc: direct parent"
        );
        // The limit keeps the NEWEST entries.
        assert_eq!(
            format_ancestor_chain(&chain, 2),
            "\n[Conversation above]\nb: middle\nc: direct parent"
        );
    }

    // ── Tool-turn handling (bugs #1, #6, #11) ──

    use crate::gemini_client::FunctionCallTurn;
    use crate::models::{FunctionCallData, ToolCallData, ToolResponseData};

    fn test_tool_ctx() -> ToolContext {
        let gemini = GeminiClient::new("test-key".to_string());
        let qdrant = Arc::new(QdrantClient::unavailable(Arc::new(gemini), 4));
        let (tx, _rx) = mpsc::unbounded_channel();
        ToolContext::new(
            Arc::new(ThingsClient::new()),
            qdrant,
            Arc::new(RwLock::new(RuntimeConfig {
                memory: config::MemoryConfig {
                    user_facts_limit: 8,
                    app_knowledge_limit: 3,
                    app_knowledge_min_score: 0.72,
                    user_fact_supersede_threshold: 0.85,
                    forget_similarity_threshold: 0.80,
                    fact_extraction_enabled: true,
                    extraction_min_chars: 24,
                },
                context_depth_limit: 20,
                tools: config::ToolsConfig {
                    enabled: true,
                    max_rounds: 6,
                    web_fetch_max_bytes: 512_000,
                    web_fetch_timeout_secs: 15,
                    user_scan_posts_limit: 10,
                    user_scan_fact_cap: 3,
                    url_context_enabled: true,
                    max_flow_attempts: 3,
                },
            })),
            tx,
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(FlowMeter::default()),
        )
    }

    fn facts_call_turn(id: &str) -> GenerateTurn {
        GenerateTurn {
            text: None,
            function_calls: vec![FunctionCallTurn {
                name: "get_user_facts".to_string(),
                args: json!({ "username": "khalid" }),
            }],
            retrieved_urls: vec![],
            raw_parts: vec![Part::FunctionCall {
                function_call: FunctionCallData {
                    name: "get_user_facts".to_string(),
                    args: json!({ "username": "khalid" }),
                    id: Some(id.to_string()),
                },
                thought_signature: None,
            }],
            finish_reason: Some("STOP".to_string()),
        }
    }

    /// Bug #1: a turn of only server-side tool parts (no function calls) must
    /// NOT append an empty user content — the API rejects empty parts.
    #[tokio::test]
    async fn server_side_only_turn_adds_no_empty_user_content() {
        let ctx = test_tool_ctx();
        let turn = GenerateTurn {
            text: None,
            function_calls: vec![],
            retrieved_urls: vec![],
            raw_parts: vec![
                Part::ToolCall {
                    tool_call: ToolCallData {
                        tool_type: "url_context".to_string(),
                        args: json!({}),
                        id: Some("t1".to_string()),
                    },
                    thought_signature: None,
                },
                Part::ToolResponse {
                    tool_response: ToolResponseData {
                        tool_type: "url_context".to_string(),
                        response: json!({}),
                        id: Some("t1".to_string()),
                    },
                    thought_signature: None,
                },
            ],
            finish_reason: Some("STOP".to_string()),
        };
        let mut contents = Vec::new();
        let mut cache = HashMap::new();
        append_tool_turn(turn, &mut contents, &ctx, &mut cache).await;
        assert_eq!(contents.len(), 1, "only the model turn is appended");
        assert_eq!(contents[0].role, "model");
        assert_eq!(contents[0].parts.len(), 2);
    }

    /// Bug #6: a call repeated in a LATER round gets the short marker (its
    /// full result is already in the history), not a re-echoed full result.
    /// Bug #11: every response echoes its OWN call id.
    #[tokio::test]
    async fn repeated_call_in_later_round_gets_marker_and_own_id() {
        let ctx = test_tool_ctx();
        let mut contents = Vec::new();
        let mut cache = HashMap::new();

        let first_fresh = append_tool_turn(facts_call_turn("call-1"), &mut contents, &ctx, &mut cache).await;
        assert!(!first_fresh, "first execution is fresh, not a cached repeat");
        let Part::FunctionResponse { function_response } = &contents[1].parts[0] else {
            panic!("expected a function response part");
        };
        assert!(
            function_response.response.get("error").is_some(),
            "qdrant unavailable -> the executed error result"
        );
        assert_eq!(function_response.id.as_deref(), Some("call-1"));

        let only_cached = append_tool_turn(facts_call_turn("call-2"), &mut contents, &ctx, &mut cache).await;
        assert!(only_cached, "identical repeat executes nothing new");
        let Part::FunctionResponse { function_response } = &contents[3].parts[0] else {
            panic!("expected a function response part");
        };
        assert_eq!(
            function_response.response.get("cached"),
            Some(&json!(true)),
            "repeat of an earlier round gets the marker"
        );        assert!(
            function_response.response.get("error").is_none(),
            "marker replaces the full result, not re-echoes it"
        );
        assert_eq!(function_response.id.as_deref(), Some("call-2"));
    }

    /// Bug #10: briefings merge into a trailing user content instead of
    /// creating two consecutive user contents.
    #[test]
    fn append_user_parts_merges_into_trailing_user_content() {
        let text = |s: &str| Part::Text {
            text: s.to_string(),
            thought_signature: None,
        };
        let mut contents = vec![Content {
            role: "user".to_string(),
            parts: vec![text("a")],
        }];
        append_user_parts(&mut contents, vec![text("b")]);
        assert_eq!(contents.len(), 1, "merged into the trailing user content");
        assert_eq!(contents[0].parts.len(), 2);

        contents.push(Content {
            role: "model".to_string(),
            parts: vec![],
        });
        append_user_parts(&mut contents, vec![text("c")]);
        assert_eq!(contents.len(), 3, "new user content after a model turn");
        assert_eq!(contents[2].role, "user");
    }

    /// Bug #9: mentions embedded in URLs or glued to words are not stripped.
    #[test]
    fn strip_mention_leaves_urls_and_glued_words_intact() {
        assert_eq!(
            strip_mention("see https://things.cv/@AskMe now"),
            "see https://things.cv/@AskMe now"
        );
        assert_eq!(strip_mention("x@AskMe"), "x@AskMe");
        assert_eq!(strip_mention("user_1@askme"), "user_1@askme");
        // Punctuation before the mention still counts as a token boundary.
        assert_eq!(strip_mention("(@AskMe) hi"), "() hi");
    }

    /// Bug #13: voice notes carried by the `audio` field are picked up.
    #[test]
    fn extract_media_urls_includes_audio_field() {
        let mut post = post_with_id(1);
        post.audio = Some(json!({ "url": "https://cdn.things.cv/voice1.ogg" }));
        assert_eq!(
            extract_media_urls(&post),
            vec!["https://cdn.things.cv/voice1.ogg".to_string()]
        );
        post.audio = Some(json!([{ "path": "https://cdn.things.cv/voice2.ogg" }]));
        assert_eq!(
            extract_media_urls(&post),
            vec!["https://cdn.things.cv/voice2.ogg".to_string()]
        );
        post.audio = Some(json!({ "unrelated": true }));
        assert!(extract_media_urls(&post).is_empty());
    }

    /// Music attachments (Apple Music cards): the preview URL is extracted as
    /// playable audio, and the metadata note names the track for the prompt.
    #[test]
    fn extract_media_urls_includes_music_previews() {
        let mut post = post_with_id(1);
        post.music = Some(vec![crate::models::MusicItem {
            id: Some(10925),
            title: Some("جدة غير".to_string()),
            artist: Some("Mooody محمود السفياني".to_string()),
            album_name: Some("جدة غير - Single".to_string()),
            artwork_url: Some("https://is1-ssl.mzstatic.com/artwork.jpg".to_string()),
            preview_url: Some(
                "https://audio-ssl.itunes.apple.com/itunes-assets/preview.m4a".to_string(),
            ),
        }]);
        assert_eq!(
            extract_media_urls(&post),
            vec!["https://audio-ssl.itunes.apple.com/itunes-assets/preview.m4a".to_string()]
        );
        assert_eq!(
            music_note(&post),
            "\n[Attached music: \"جدة غير\" — Mooody محمود السفياني]"
        );

        // A track without a preview URL yields no media but still gets the note.
        post.music = Some(vec![crate::models::MusicItem {
            id: None,
            title: Some("Song".to_string()),
            artist: Some("Artist".to_string()),
            album_name: None,
            artwork_url: None,
            preview_url: None,
        }]);
        assert!(extract_media_urls(&post).is_empty());
        assert_eq!(music_note(&post), "\n[Attached music: \"Song\" — Artist]");

        // No music at all -> no note.
        post.music = None;
        assert_eq!(music_note(&post), "");
    }

    /// Apple serves previews as audio/mp4 (or x-m4a) — Gemini's documented
    /// list only takes audio/aac, so uploads are normalized.
    #[test]
    fn normalize_audio_mime_maps_m4a_aliases_to_aac() {
        assert_eq!(normalize_audio_mime("audio/mp4"), "audio/aac");
        assert_eq!(normalize_audio_mime("audio/x-m4a"), "audio/aac");
        assert_eq!(normalize_audio_mime("audio/x-m4p"), "audio/aac");
        assert_eq!(normalize_audio_mime("audio/m4p"), "audio/aac");
        assert_eq!(normalize_audio_mime("audio/mp4a-latm"), "audio/aac");
        assert_eq!(normalize_audio_mime("audio/mp4; codecs=mp4a.40.2"), "audio/aac");
        // Everything else passes through untouched.
        assert_eq!(normalize_audio_mime("audio/ogg"), "audio/ogg");
        assert_eq!(normalize_audio_mime("image/jpeg"), "image/jpeg");
        assert_eq!(normalize_audio_mime("video/mp4"), "video/mp4");
    }

    /// Fixture test with a real Things payload (CR7's music post, 5757646):
    /// guards against field-name drift in the music card shape.
    #[test]
    fn parses_real_music_post_payload() {
        let raw = r#"{"data":{"post":{"id":5757646,"user":{"id":14348,"username":"CR7"},
            "parent_id":null,"post_comment":"مين زيي لا اخاف من المجتمع وحابتها مره وانطرب عليها",
            "entities":[],"media":[],"audio":null,
            "music":[{"id":10925,"title":"جدة غير","artist":"Mooody محمود السفياني",
                      "albumName":"جدة غير - Single",
                      "artworkURL":"https://is1-ssl.mzstatic.com/image/thumb/artwork.jpg",
                      "previewURL":"https://audio-ssl.itunes.apple.com/itunes-assets/mzaf_15182545737609936747.plus.aac.p.m4a"}],
            "created_at":"2026-08-05T16:23:09.000000Z","post_type":"b",
            "expires_at":"2026-08-07T16:23:09.000000Z"},"parent":null,"quoted":null}}"#;
        let envelope: crate::models::PostEnvelope = serde_json::from_str(raw).unwrap();
        let post = envelope.data.and_then(|d| d.post).expect("post parses");
        assert_eq!(post.id_value(), Some(5757646));
        assert_eq!(post.author_username(), "CR7");
        let urls = extract_media_urls(&post);
        assert_eq!(urls.len(), 1, "one music preview extracted");
        assert!(urls[0].starts_with("https://audio-ssl.itunes.apple.com/"));
        assert!(urls[0].ends_with(".m4a"));
        assert_eq!(
            music_note(&post),
            "\n[Attached music: \"جدة غير\" — Mooody محمود السفياني]"
        );
    }

    /// The music scaffold label must trip the leak guard like every other
    /// injected section header.
    #[test]
    fn scaffold_leak_detection_includes_music_note() {
        assert!(scaffold_leak_at("جواب\n[Attached music: \"x\" — y]").is_some());
    }
}
