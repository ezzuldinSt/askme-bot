mod admin;
mod config;
mod entities;
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
use crate::gemini_client::{GeminiClient, GeminiError, GenerateTurn, KeyLease};
use crate::models::{
    Content, FileData, FunctionResponseData, Notification, Part, Post,
};
use crate::qdrant_client::{MemoryWrite, QdrantClient};
use crate::qdrant_models::{
    app_fact_point_id, user_fact_point_id, AppFactPayload, AppFactSource, AppFactStatus,
    AppKnowledgeSeed, FactCategory, MemoryEntry, MessagePayload, MessageType, UserFactPayload,
    PROCESSED_COLLECTION_NAME, THINGS_KNOWLEDGE_COLLECTION_NAME, USER_PROFILES_COLLECTION_NAME,
};
use crate::things_client::{is_auth_expired, ThingsClient, TOKEN_FILE};
use crate::tools::{
    tool_declarations, ExtractionJob, ExtractionSource, FlowSubject, ToolContext,
};

const BOT_USERNAME: &str = "AskMe";
const POLL_INTERVAL_MS: u64 = 3_000;
const MAX_RESPONSE_LENGTH: usize = 8000;
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
    extraction_writer: mpsc::UnboundedSender<ExtractionJob>,
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

    let (generation_model, thinking_level, embedding_model, embedding_dimensions, qdrant_url) = {
        let cfg = bot_config.read().await;
        (
            config::resolve_generation_model(&cfg.overrides),
            config::resolve_thinking_level(&cfg.overrides),
            config::resolve_embedding_model(&cfg.overrides),
            config::resolve_embedding_dimensions(&cfg.overrides),
            config::resolve_qdrant_url(&cfg.overrides),
        )
    };
    let gemini = GeminiClient::with_keys(
        gemini_api_keys,
        generation_model,
        thinking_level,
        embedding_model.clone(),
        embedding_dimensions as u32,
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
        if let Err(e) = seed_app_knowledge(&qdrant).await {
            warn!("Failed to seed Things app knowledge: {e}");
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

    let user_text = match prompt_override {
        Some(custom) => custom.to_string(),
        None => {
            if is_follow_up {
                println!("=== Follow-up reply (parent is {BOT_USERNAME}) ===");
                build_follow_up_prompt(state, post, &question, &post_data, conversation_id).await
            } else {
                build_mention_prompt(state, post, &question, &ancestors).await
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

    println!("=== Raw response ===");
    println!("{response}");

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
                if is_mention_notification(notification)
                    || is_follow_up_notification(&state_read, notification).await
                {
                    to_process.push(notification.clone());
                } else {
                    irrelevant.push(notification.id);
                }
            }
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
            // retried — those stay unread so the next poll picks them up.
            let ids: Vec<u64> = notifications
                .iter()
                .map(|n| n.id)
                .filter(|id| !retry_later.contains(id))
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
            // we saw (likes, follows, already-handled items, ...).
            let ids: Vec<u64> = notifications.iter().map(|n| n.id).collect();
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

fn is_mention_notification(notification: &Notification) -> bool {
    let nt = notification.notification_type.as_deref().unwrap_or("");
    let group = notification.group.as_deref().unwrap_or("");
    nt == "user_mention" || nt == "mention" || group == "mentions"
}

/// A follow-up is a reply to a post that AskMe itself wrote (detected from the
/// notification payload, or from a `bot_reply` already persisted in Qdrant).
async fn is_follow_up_notification(state: &AppState, notification: &Notification) -> bool {
    let nt = notification.notification_type.as_deref().unwrap_or("");
    if nt != "post_reply" {
        return false;
    }
    let Some(original_post) = notification.original_post_data.as_ref() else {
        return false;
    };
    let Some(original_post_id) = original_post.id_value() else {
        return false;
    };
    // Only the unique username is trusted — display names are user-controlled
    // and anyone can call themselves "AskMe".
    let is_bot_author = original_post
        .user
        .as_ref()
        .and_then(|u| u.username.as_deref())
        .map(|u| u.eq_ignore_ascii_case(BOT_USERNAME))
        .unwrap_or(false);
    if is_bot_author {
        return true;
    }
    let qdrant = state.qdrant.clone();
    if !qdrant.is_available() {
        return false;
    }
    match qdrant.get_point(original_post_id).await {
        Ok(Some(entry)) => entry.message_type == MessageType::BotReply,
        _ => false,
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

    let post_data = {
        let things = &state.read().await.things;
        match things.get_post(post_id).await {
            Ok(data) => data,
            Err(e) => {
                exit_if_auth_expired(&e);
                error!("Failed to fetch post {post_id}: {e}");
                return ProcessOutcome::Failed;
            }
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
        is_follow_up_notification(&state_read, &notification).await
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
        if qdrant.is_available() && !is_empty_mention {
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

    let user_text = if is_empty_mention {
        build_greeting_prompt(&state, &post, &post_data, &ancestors, is_follow_up, conversation_id)
            .await
    } else if is_follow_up {
        build_follow_up_prompt(&state, &post, &question, &post_data, conversation_id).await
    } else {
        build_mention_prompt(&state, &post, &question, &ancestors).await
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

    let (reply_text, reply_entities) = match response {
        Ok((text, sources)) => {
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
            // Retry only when the request definitely never reached the server.
            // Otherwise the reply may have been committed server-side and
            // retrying would post a duplicate — worse than a lost retry.
            let never_sent = e.chain().any(|cause| {
                cause
                    .downcast_ref::<reqwest::Error>()
                    .map(|re| re.is_connect())
                    .unwrap_or(false)
            });
            if never_sent {
                return ProcessOutcome::Failed;
            }
            warn!(
                "Not retrying reply for notification {notification_id} \
                 (reply may already be committed); marking processed"
            );
            return skip_notification(&state, notification_id).await;
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
    if let Err(e) = qdrant.upsert(&reply_payload).await {
        warn!("Failed to store bot reply {reply_id} in Qdrant: {e}");
    }
    if let Err(e) = qdrant.mark_processed(notification_id).await {
        warn!("Failed to mark notification {notification_id} processed in Qdrant: {e}");
    }
    state.write().await.processed.insert(notification_id);

    // Long-term memory pass: pull durable user/app facts out of the user's
    // message in the background. Idempotent, so retries are harmless. Empty
    // mentions have no content to extract from.
    if !is_empty_mention {
        let _ = state
            .read()
            .await
            .extraction_writer
            .send(ExtractionJob {
                username: post.author_username().to_string(),
                text: question.clone(),
                post_id,
                conversation_id,
                source: ExtractionSource::Conversation,
            });
    }

    ProcessOutcome::Replied
}

fn extract_question(post: &Post) -> String {
    strip_mention(post.content_text())
}

/// Remove every `@AskMe` mention from the content, keeping the surrounding
/// text on both sides. ASCII-case-insensitive and char-boundary safe (a
/// whole-string `to_lowercase()` would shift byte offsets for characters like
/// Turkish `İ` and could slice mid-character — panicking or corrupting text).
fn strip_mention(content: &str) -> String {
    const MENTION: &[u8] = b"@askme";
    let bytes = content.as_bytes();
    let mut out = String::with_capacity(content.len());
    let mut last = 0;
    let mut i = 0;

    while i < bytes.len() {
        // '@' is ASCII, so `i` is always a char boundary here.
        if bytes[i] == b'@' {
            let end = i + MENTION.len();
            if end <= bytes.len()
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

    let mut push = |url: &String| {
        if urls.len() < MAX_MEDIA_FILES && !urls.contains(url) {
            urls.push(url.clone());
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

    urls
}

async fn build_mention_prompt(
    state: &RwLock<AppState>,
    post: &Post,
    question: &str,
    ancestors: &[(String, String)],
) -> String {
    let author = post.author_username();
    let depth_limit = {
        let runtime = state.read().await.runtime.clone();
        let cfg = runtime.read().await;
        cfg.context_depth_limit
    };
    let above = format_ancestor_chain(ancestors, depth_limit);

    let profile = build_user_profile_section(state, author).await;
    let app_knowledge = build_app_knowledge_section(state, question).await;

    format!(
        "[Post by {author}] {content}\n[Question] {question}{above}{profile}{app_knowledge}",
        content = post.content_text(),
        question = question,
        author = author,
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
    ) + &context
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
async fn build_app_knowledge_section(state: &RwLock<AppState>, question: &str) -> String {
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
    let app_knowledge = build_app_knowledge_section(state, question).await;

    format!(
        "[Follow-up question by {author}] {question}{context}{profile}{app_knowledge}",
        author = author,
        question = question,
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

/// Upsert the curated `things_knowledge.json` seed facts into tier-3 memory.
/// Idempotent (deterministic point ids); a missing file is not an error.
async fn seed_app_knowledge(qdrant: &Arc<QdrantClient>) -> Result<()> {
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
    Ok(())
}

/// Spawn the background fact-extraction worker. For every answered user
/// message it runs one lightweight Gemini call and turns the result into
/// tier-2 (user facts) and tier-3 (pending app facts) memory writes.
fn spawn_extraction_worker(
    gemini: GeminiClient,
    qdrant: Arc<QdrantClient>,
    runtime: Arc<RwLock<RuntimeConfig>>,
) -> mpsc::UnboundedSender<ExtractionJob> {
    let (tx, mut rx) = mpsc::unbounded_channel::<ExtractionJob>();
    tokio::spawn(async move {
        while let Some(job) = rx.recv().await {
            // Snapshot the live config per job (panel edits apply immediately).
            let config = runtime.read().await.clone();
            run_extraction_job(&gemini, &qdrant, &config, job).await;
        }
    });
    tx
}

async fn run_extraction_job(
    gemini: &GeminiClient,
    qdrant: &Arc<QdrantClient>,
    config: &RuntimeConfig,
    job: ExtractionJob,
) {
    if !config.memory.fact_extraction_enabled
        || !qdrant.is_available()
        || job.username.eq_ignore_ascii_case(BOT_USERNAME)
    {
        return;
    }

    let extracted = match gemini.extract_facts(&job.username, &job.text).await {
        Ok(e) => e,
        Err(e) => {
            warn!("Fact extraction failed for post {}: {e}", job.post_id);
            return;
        }
    };
    let now = unix_now();

    // ── User facts: reinforce / supersede / insert ──
    // Profile scans (posts read during a user lookup) insert at most
    // `user_scan_fact_cap` NEW facts per scan; reinforcement of already-known
    // facts does not count against the cap.
    let mut new_facts = 0usize;
    for fact in extracted.user_facts {
        let text = fact.fact.trim();
        if text.chars().count() < 3 || text.chars().count() > MAX_FACT_LENGTH {
            continue;
        }
        let point_id = user_fact_point_id(&job.username, text);

        // Exact restatement -> reinforce the existing point in place.
        match qdrant.get_user_fact(point_id).await {
            Ok(Some(existing)) => {
                let patch = serde_json::json!({
                    "last_seen": now,
                    "times_confirmed": existing.times_confirmed.saturating_add(1),
                    "active": true,
                });
                if let Err(e) = qdrant.patch_user_fact(point_id, patch).await {
                    warn!("Failed to reinforce user fact {point_id}: {e}");
                }
                continue;
            }
            Ok(None) => {}
            Err(e) => {
                warn!("Failed to check user fact {point_id}: {e}");
                continue;
            }
        }

        // Near-duplicate or contradiction -> retire the old fact, keep the new.
        let vector = match qdrant.embed(text).await {
            Ok(v) => v,
            Err(e) => {
                warn!("Failed to embed user fact for {}: {e}", job.username);
                continue;
            }
        };
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

        // Profile scans: stop inserting once the cap is hit (keep reinforcing
        // already-known facts, but do not keep adding brand-new ones).
        if job.source == ExtractionSource::ProfileScan
            && new_facts >= config.tools.user_scan_fact_cap
        {
            continue;
        }

        let payload = UserFactPayload {
            username: job.username.clone(),
            fact: text.to_string(),
            category: fact
                .category
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
        mime,
        display_name: format!("media_{}", uuid::Uuid::new_v4()),
    })
}

/// How one tool-calling reply flow ended.
enum ToolsFlowError {
    /// 429/401/403 — the lease key is marked; re-lease with the next key.
    RateLimited,
    /// Non-retryable failure (bad request, rounds exhausted, etc.).
    Failed(anyhow::Error),
}

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
    let max_flows = gemini.pool_size().max(1) + 1;
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

    Err(last_err
        .unwrap_or_else(|| anyhow::anyhow!("all Gemini API keys are currently rate-limited")))
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
            // The model is ready to answer, but if it looked up a user this
            // flow we first make sure its reply can draw on that user's saved
            // facts and a summary of their posts (fresh facts included).
            if let Some(briefing) = build_user_briefing(gemini, tool_ctx).await {
                contents.push(briefing);
                continue;
            }
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

        append_tool_turn(turn, contents, tool_ctx, &mut call_cache).await;
    }

    // The round budget is spent but the model still wanted tools. Execute
    // nothing more: give it ONE final chance to answer with everything it has
    // collected — tools withheld so it cannot keep looping. This guarantees a
    // reply instead of dropping the notification after exhausting rounds.
    if let Some(briefing) = build_user_briefing(gemini, tool_ctx).await {
        contents.push(briefing);
    }
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
        let subject = subjects
            .iter_mut()
            .rev()
            .find(|s| !s.briefed && s.username.is_some())?;
        let username = subject.username.clone()?;
        if username.eq_ignore_ascii_case(BOT_USERNAME) {
            return None;
        }
        let user_id = subject.user_id;
        let posts = subject.posts.clone();
        subject.briefed = true;
        (username, user_id, posts)
    };

    let config = tool_ctx.runtime.read().await.clone();

    // Posts: reuse the in-flow get_user_posts result, else fetch (bounded).
    let mut posts_value = posts;
    if posts_value.is_none() {
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
            run_extraction_job(gemini, &tool_ctx.qdrant, &config, job).await;
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
        parts: vec![Part::Text { text }],
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
        if tool_ctx.extraction_tx.send(job).is_err() {
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
/// - a call repeated in a later round (same name + args) is answered from
///   `call_cache` with the same marker instead of re-executing and re-echoing
///   a large result into the history.
async fn append_tool_turn(
    turn: GenerateTurn,
    contents: &mut Vec<Content>,
    tool_ctx: &ToolContext,
    call_cache: &mut HashMap<String, Value>,
) {
    // Append the model's parts verbatim (thought signatures and ids intact).
    // With tool context circulation this must include the server-side
    // toolCall/toolResponse parts, not just the functionCall parts — and even
    // interim text parts, which the API counts into the conversation.
    contents.push(Content {
        role: "model".to_string(),
        parts: turn.raw_parts.clone(),
    });

    // Map each executed call (name + args) to its API-assigned id, so the
    // functionResponse echoes it back (required for tool context circulation).
    let mut id_by_key: HashMap<String, String> = HashMap::new();
    for part in &turn.raw_parts {
        if let Part::FunctionCall {
            function_call,
            ..
        } = part
        {
            if let Some(id) = &function_call.id {
                id_by_key.insert(call_cache_key(&function_call.name, &function_call.args), id.clone());
            }
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
        let response = if first_seen.get(&key) == Some(&i) {
            match turn_results.get(&key).or_else(|| call_cache.get(&key)) {
                Some(r) => r.clone(),
                None => json!({ "error": "internal: missing tool result" }),
            }
        } else {
            json!({
                "cached": true,
                "note": format!(
                    "This exact call to {}() with the same arguments was already executed \
                     earlier in this conversation; the result shown above is unchanged.",
                    call.name
                ),
            })
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
                id: id_by_key.get(&key).cloned(),
            },
        });
    }
    contents.push(Content {
        role: "user".to_string(),
        parts: response_parts,
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
}
