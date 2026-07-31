mod entities;
mod gemini_client;
mod models;
mod qdrant_client;
mod qdrant_models;
mod things_client;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::io::AsyncBufReadExt;
use tokio::sync::{mpsc, RwLock};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use crate::entities::build_reply_with_entities;
use crate::gemini_client::GeminiClient;
use crate::models::{Notification, Post};
use crate::qdrant_client::{MemoryWrite, QdrantClient};
use crate::qdrant_models::{MemoryEntry, MessagePayload, MessageType, SearchOptions};
use crate::things_client::ThingsClient;

const BOT_USERNAME: &str = "AskMe";
const POLL_INTERVAL_MS: u64 = 3_000;
const MAX_RESPONSE_LENGTH: usize = 500;
const MAX_CONTEXT_DEPTH: usize = 20;
const MAX_MEDIA_FILES: usize = 5;
const PROCESSED_IDS_FILE: &str = ".processed-ids.json";
const MEMORY_WRITE_BATCH_SIZE: usize = 5;
const MEMORY_WRITE_FLUSH_MS: u64 = 2_000;

struct AppState {
    things: ThingsClient,
    gemini: GeminiClient,
    qdrant: Arc<QdrantClient>,
    memory_writer: mpsc::UnboundedSender<MemoryWrite>,
    system_prompt: String,
    /// In-memory mirror of processed notification ids (session-only safety net;
    /// Qdrant is the source of truth for cross-restart dedup).
    processed: HashSet<u64>,
    context_depth_limit: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    dotenvy::dotenv().ok();

    let gemini_api_key =
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set in .env");
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

    let gemini = GeminiClient::new(gemini_api_key);

    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
    let embedding_dimensions: u64 = std::env::var("EMBEDDING_DIMENSIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let context_search_limit: u64 = std::env::var("CONTEXT_SEARCH_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let context_depth_limit: usize = std::env::var("CONTEXT_DEPTH_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_CONTEXT_DEPTH);

    let qdrant = QdrantClient::connect(
        &qdrant_url,
        Arc::new(gemini.clone()),
        embedding_dimensions,
        context_search_limit,
    )
    .await;
    let qdrant = Arc::new(qdrant);

    if qdrant.is_available() {
        match qdrant.ensure_collection().await {
            Ok(()) => info!("Qdrant collection ready: {}", qdrant.collection_name()),
            Err(e) => warn!(
                "Qdrant reachable but collection setup failed: {e}; degrading to memory-only mode"
            ),
        }
    }

    let processed = load_processed_ids(&qdrant).await;
    let memory_writer = qdrant.spawn_writer(
        MEMORY_WRITE_BATCH_SIZE,
        Duration::from_millis(MEMORY_WRITE_FLUSH_MS),
    );

    let state = Arc::new(RwLock::new(AppState {
        things,
        gemini,
        qdrant,
        memory_writer,
        system_prompt,
        processed,
        context_depth_limit,
    }));

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--test-post") {
        let post_id = args
            .iter()
            .rev()
            .find(|a| !a.starts_with("--"))
            .ok_or_else(|| anyhow::anyhow!("--test-post requires a post id"))?
            .parse::<u64>()?;
        let do_post = args.iter().any(|a| a == "--post");
        let prompt_override = args
            .iter()
            .position(|a| a == "--prompt")
            .and_then(|p| args.get(p + 1))
            .cloned();
        return test_post(&state, post_id, do_post, prompt_override.as_deref()).await;
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
    Ok(())
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

    let qdrant = state.read().await.qdrant.clone();
    println!("=== Qdrant status ===");
    if qdrant.is_available() {
        println!("Connected (collection: {})", qdrant.collection_name());

        let thread_id = resolve_thread(state, &post_data, post).await;
        let payload = MessagePayload {
            id: post_id,
            content: post.content_text().to_string(),
            username: post.author_username().to_string(),
            message_type: MessageType::Post,
            parent_id: post.parent_id,
            thread_id,
            timestamp: timestamp_from_post(post),
            is_processed: false,
            media_urls: media_urls.clone(),
        };
        match qdrant.upsert(&payload).await {
            Ok(()) => println!("Stored post {post_id} in Qdrant (thread {thread_id})"),
            Err(e) => println!("Failed to store post {post_id} in Qdrant: {e}"),
        }

        let depth_limit = state.read().await.context_depth_limit;
        let thread = qdrant
            .get_thread_context(thread_id, depth_limit as u64)
            .await
            .unwrap_or_default();
        println!(
            "=== Qdrant thread context (thread {thread_id}, {} entries) ===",
            thread.len()
        );
        for entry in &thread {
            println!(
                "  [{}] {}: {}",
                entry.timestamp, entry.username, entry.content
            );
        }

        let related = qdrant
            .search_context(
                &question,
                &SearchOptions {
                    username: None,
                    min_timestamp: None,
                    limit: context_search_limit(),
                },
            )
            .await
            .unwrap_or_default();
        println!("=== Qdrant semantic search ({} hits) ===", related.len());
        for entry in related.iter().take(10) {
            println!(
                "  {}: {}",
                entry.username,
                truncate_text(&entry.content, 120)
            );
        }
    } else {
        println!("UNREACHABLE — running degraded (no persistent memory)");
    }

    let gemini = state.read().await.gemini.clone();
    let system_prompt = state.read().await.system_prompt.clone();

    let mut file_uris: Vec<(String, String)> = Vec::new();
    if !media_urls.is_empty() {
        println!("=== Downloading + uploading media ===");
        for url in &media_urls {
            match download_and_upload_media(state, &gemini, url).await {
                Ok((uri, mime)) => {
                    println!("Uploaded: {uri} ({mime})");
                    file_uris.push((uri, mime));
                }
                Err(e) => {
                    println!("Media processing failed for {url}: {e}");
                }
            }
        }
    }

    let user_text = match prompt_override {
        Some(custom) => custom.to_string(),
        None => {
            let is_follow_up = post.parent_id.is_some()
                && post_data
                    .parent
                    .as_ref()
                    .map(|p| p.author_username().eq_ignore_ascii_case(BOT_USERNAME))
                    .unwrap_or(false);
            if is_follow_up {
                println!("=== Follow-up reply (parent is {BOT_USERNAME}) ===");
                let thread_id = resolve_thread(state, &post_data, post).await;
                build_follow_up_prompt(state, post, &question, &post_data, thread_id).await
            } else {
                build_mention_prompt(&post_data, post, &question)
            }
        }
    };
    println!("=== Prompt ===");
    println!("{user_text}");

    println!("=== Generating response ===");
    let response = gemini
        .generate_content(&system_prompt, &user_text, &file_uris)
        .await?;

    println!("=== Raw response ===");
    println!("{response}");

    let (reply_text, entities) = build_reply_with_entities(&response, MAX_RESPONSE_LENGTH);
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
        match things.reply_to_post(post_id, &reply_text, &entities).await {
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

/// Seed the processed-notification cache from Qdrant (and migrate the legacy
/// JSON file on first startup).
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

    match std::fs::read_to_string(PROCESSED_IDS_FILE) {
        Ok(content) => match serde_json::from_str::<Vec<u64>>(&content) {
            Ok(list) if !list.is_empty() => {
                if qdrant.is_available() {
                    let mut migrated = 0;
                    let mut failures = Vec::new();
                    for id in &list {
                        if ids.contains(id) {
                            continue;
                        }
                        match qdrant.mark_processed(*id).await {
                            Ok(()) => {
                                ids.insert(*id);
                                migrated += 1;
                            }
                            Err(e) => {
                                warn!("Failed to migrate processed ID {id}: {e}");
                                failures.push(*id);
                            }
                        }
                    }
                    info!("Migrated {migrated} processed IDs from legacy file into Qdrant");
                    if failures.is_empty() {
                        match std::fs::remove_file(PROCESSED_IDS_FILE) {
                            Ok(()) => info!("Removed legacy {PROCESSED_IDS_FILE}"),
                            Err(e) => warn!("Failed to remove legacy {PROCESSED_IDS_FILE}: {e}"),
                        }
                    } else {
                        warn!(
                        "{} processed IDs failed to migrate (e.g. {:?}); keeping {PROCESSED_IDS_FILE} for retry on next boot",
                        failures.len(),
                        &failures[..failures.len().min(5)],
                    );
                    }
                } else {
                    for id in list {
                        ids.insert(id);
                    }
                    info!(
                        "Loaded {} processed notification IDs from legacy file (degraded mode)",
                        ids.len()
                    );
                }
            }
            Ok(_) => {}
            Err(e) => warn!("Failed to parse {PROCESSED_IDS_FILE}: {e}"),
        },
        Err(e) => {
            if e.kind() != std::io::ErrorKind::NotFound {
                warn!("Failed to read {PROCESSED_IDS_FILE}: {e}");
            }
        }
    }

    ids
}

async fn poll_loop(state: Arc<RwLock<AppState>>) {
    loop {
        tokio::time::sleep(Duration::from_millis(POLL_INTERVAL_MS)).await;

        let unread_count = {
            let things = &state.read().await.things;
            match things.get_unread_count().await {
                Ok(count) => count,
                Err(e) => {
                    error!("Failed to get unread count: {e}");
                    continue;
                }
            }
        };

        if unread_count == 0 {
            continue;
        }

        info!("{unread_count} unread notifications");

        let notifications = {
            let things = &state.read().await.things;
            match things.get_notifications(1).await {
                Ok(n) => n,
                Err(e) => {
                    error!("Failed to fetch notifications: {e}");
                    continue;
                }
            }
        };

        let mut to_process = Vec::new();
        let mut known_processed = Vec::new();
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
                    || is_follow_up_notification(&state, notification).await
                {
                    to_process.push(notification.clone());
                }
            }
        }
        if !known_processed.is_empty() {
            state.write().await.processed.extend(known_processed);
        }

        if to_process.is_empty() {
            continue;
        }

        info!("Processing {} notifications", to_process.len());

        let mut handles = Vec::new();
        for notification in to_process {
            let state = state.clone();
            let handle = tokio::spawn(async move {
                process_notification(state, notification).await;
            });
            handles.push(handle);
        }

        for handle in handles {
            if let Err(e) = handle.await {
                error!("Notification processing task failed: {e}");
            }
        }

        let ids: Vec<u64> = notifications.iter().map(|n| n.id).collect();
        {
            let things = &state.read().await.things;
            if let Err(e) = things.mark_notifications_read(&ids).await {
                error!("Failed to mark notifications as read: {e}");
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
async fn is_follow_up_notification(state: &RwLock<AppState>, notification: &Notification) -> bool {
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
    let is_bot_author = original_post
        .user
        .as_ref()
        .map(|u| {
            u.username
                .as_deref()
                .unwrap_or("")
                .eq_ignore_ascii_case(BOT_USERNAME)
                || u.name
                    .as_deref()
                    .unwrap_or("")
                    .eq_ignore_ascii_case(BOT_USERNAME)
        })
        .unwrap_or(false);
    if is_bot_author {
        return true;
    }
    let qdrant = state.read().await.qdrant.clone();
    if !qdrant.is_available() {
        return false;
    }
    match qdrant.get_point(original_post_id).await {
        Ok(Some(entry)) => entry.message_type == MessageType::BotReply,
        _ => false,
    }
}

fn notification_post_id(notification: &Notification) -> Option<u64> {
    if is_mention_notification(notification) {
        return notification.post_data.as_ref().and_then(|p| p.id_value());
    }
    if notification.notification_type.as_deref() == Some("post_reply") {
        return notification
            .reply_post_data
            .as_ref()
            .and_then(|p| p.id_value())
            .or_else(|| {
                notification
                    .original_post_data
                    .as_ref()
                    .and_then(|p| p.id_value())
            });
    }
    notification
        .post_data
        .as_ref()
        .and_then(|p| p.id_value())
        .or_else(|| {
            notification
                .reply_post_data
                .as_ref()
                .and_then(|p| p.id_value())
        })
}

async fn process_notification(state: Arc<RwLock<AppState>>, notification: Notification) {
    let notification_id = notification.id;

    let post_id = match notification_post_id(&notification) {
        Some(id) => id,
        None => {
            warn!(
                "Notification {} has no post data, skipping",
                notification.id
            );
            state.write().await.processed.insert(notification_id);
            return;
        }
    };

    let post_data = {
        let things = &state.read().await.things;
        match things.get_post(post_id).await {
            Ok(data) => data,
            Err(e) => {
                error!("Failed to fetch post {post_id}: {e}");
                return;
            }
        }
    };

    let post = match post_data.post {
        Some(ref p) => p.clone(),
        None => {
            warn!("Post {post_id} has no content");
            state.write().await.processed.insert(notification_id);
            return;
        }
    };

    let is_follow_up = is_follow_up_notification(&state, &notification).await;

    let question = extract_question(&post);
    if question.len() < 2 {
        info!("Notification {notification_id}: question too short, skipping");
        state.write().await.processed.insert(notification_id);
        return;
    }

    let thread_id = resolve_thread(&state, &post_data, &post).await;
    let memory_payload = MessagePayload {
        id: post_id,
        content: strip_mention(post.content_text()).to_string(),
        username: post.author_username().to_string(),
        message_type: if is_follow_up {
            MessageType::Reply
        } else {
            MessageType::Post
        },
        parent_id: post
            .parent_id
            .or_else(|| post_data.parent.as_ref().and_then(|p| p.id_value())),
        thread_id,
        timestamp: timestamp_from_post(&post),
        is_processed: false,
        media_urls: extract_media_urls(&post),
    };

    {
        let qdrant = state.read().await.qdrant.clone();
        if qdrant.is_available() {
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

    let user_text = if is_follow_up {
        build_follow_up_prompt(&state, &post, &question, &post_data, thread_id).await
    } else {
        build_mention_prompt(&post_data, &post, &question)
    };

    info!(
        "Prompt for notification {notification_id}: {}",
        truncate_text(&user_text, 400)
    );

    let media_urls = extract_media_urls(&post);

    let mut file_uris: Vec<(String, String)> = Vec::new();

    let gemini = state.read().await.gemini.clone();
    for url in &media_urls {
        match download_and_upload_media(&state, &gemini, url).await {
            Ok((uri, mime)) => {
                info!("Media uploaded to Gemini: {uri} ({mime})");
                file_uris.push((uri, mime));
            }
            Err(e) => {
                warn!("Failed to process media {url} for notification {notification_id}: {e}");
            }
        }
    }

    let system_prompt = state.read().await.system_prompt.clone();

    let response = gemini
        .generate_content(&system_prompt, &user_text, &file_uris)
        .await;

    let (reply_text, reply_entities) = match response {
        Ok(text) => {
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
            return;
        }
    };

    let reply_id = {
        let things = &state.read().await.things;
        match things
            .reply_to_post(post_id, &reply_text, &reply_entities)
            .await
        {
            Ok(reply_id) => {
                info!("Posted reply {reply_id} to post {post_id}");
                Some(reply_id)
            }
            Err(e) => {
                error!("Failed to post reply to {post_id}: {e}");
                None
            }
        }
    };

    if let Some(reply_id) = reply_id {
        let qdrant = state.read().await.qdrant.clone();

        let reply_payload = MessagePayload {
            id: reply_id,
            content: reply_text,
            username: BOT_USERNAME.to_string(),
            message_type: MessageType::BotReply,
            parent_id: Some(post_id),
            thread_id,
            timestamp: unix_now(),
            is_processed: true,
            media_urls: Vec::new(),
        };
        if let Err(e) = qdrant.upsert(&reply_payload).await {
            warn!("Failed to store bot reply {reply_id} in Qdrant: {e}");
        }
        if let Err(e) = qdrant.mark_processed(notification_id).await {
            warn!("Failed to mark notification {notification_id} processed in Qdrant: {e}");
        }
        state.write().await.processed.insert(notification_id);
    }
}

fn extract_question(post: &Post) -> String {
    strip_mention(post.content_text())
}

fn strip_mention(content: &str) -> String {
    let pattern = format!("@{BOT_USERNAME}");
    let lowered = content.to_lowercase();
    let pattern_lower = pattern.to_lowercase();

    if let Some(idx) = lowered.find(&pattern_lower) {
        let after = &content[idx + pattern.len()..];
        after.trim().to_string()
    } else {
        content.trim().to_string()
    }
}

fn extract_media_urls(post: &Post) -> Vec<String> {
    let mut urls = Vec::new();

    if let Some(ref media) = post.media {
        for item in media {
            if let Some(ref url) = item.url {
                if urls.len() < MAX_MEDIA_FILES {
                    urls.push(url.clone());
                }
            }
        }
    }
    if let Some(ref images) = post.images {
        for item in images {
            if let Some(ref url) = item.url {
                if urls.len() < MAX_MEDIA_FILES {
                    urls.push(url.clone());
                }
            }
        }
    }
    if let Some(ref attachment) = post.image {
        if let Some(ref url) = attachment.url {
            if urls.len() < MAX_MEDIA_FILES {
                urls.push(url.clone());
            }
        }
    }
    if let Some(ref attachments) = post.attachments {
        for item in attachments {
            if let Some(ref url) = item.url {
                if urls.len() < MAX_MEDIA_FILES {
                    urls.push(url.clone());
                }
            }
        }
    }

    urls
}

fn build_mention_prompt(post_data: &models::PostData, post: &Post, question: &str) -> String {
    let author = post.author_username();
    let parent_context = post_data
        .parent
        .as_ref()
        .map(|p| {
            let parent_author = p.author_username();
            format!("\n[Original post by {parent_author}] {}", p.content_text())
        })
        .unwrap_or_default();

    format!(
        "[Post by {author}] {content}\n[Question] {question}{parent_context}",
        content = post.content_text(),
        question = question,
        author = author,
        parent_context = parent_context,
    )
}

/// Build the prompt for a follow-up question, pulling the conversation history
/// from Qdrant (thread by id, augmented with semantic search) and falling back
/// to the Things API only when memory is unavailable.
async fn build_follow_up_prompt(
    state: &RwLock<AppState>,
    post: &Post,
    question: &str,
    post_data: &models::PostData,
    thread_id: u64,
) -> String {
    let author = post.author_username();
    let qdrant = state.read().await.qdrant.clone();
    let depth_limit = state.read().await.context_depth_limit;

    let mut context = String::new();
    let mut source = "things-api";

    if qdrant.is_available() {
        let search_options = || SearchOptions {
            username: None,
            min_timestamp: None,
            limit: context_search_limit(),
        };

        match qdrant
            .get_thread_context(thread_id, depth_limit as u64)
            .await
        {
            Ok(mut entries) if !entries.is_empty() => {
                source = "qdrant";
                if entries.len() < 3 {
                    if let Ok(related) = qdrant.search_context(question, &search_options()).await {
                        for entry in related {
                            if !entries.iter().any(|e| e.id == entry.id) {
                                entries.push(entry);
                            }
                        }
                        entries.sort_by_key(|e| e.timestamp);
                        entries.truncate(depth_limit);
                    }
                }
                context = format_context_entries(&entries);
            }
            Ok(_) => {
                if let Ok(related) = qdrant.search_context(question, &search_options()).await {
                    if !related.is_empty() {
                        source = "qdrant-semantic";
                        context = format_context_entries(&related);
                    }
                }
            }
            Err(e) => warn!("Qdrant thread lookup failed for thread {thread_id}: {e}"),
        }
    }

    if context.is_empty() {
        let things = &state.read().await.things;
        context = build_thread_context(things, post_data.parent.as_ref()).await;
    }

    info!(
        "Follow-up context for thread {thread_id} from {source} ({} entries / {} chars)",
        context
            .matches("[Conversation so far]")
            .count()
            .saturating_add(0),
        context.chars().count(),
    );

    format!(
        "[Follow-up question by {author}] {question}{context}",
        author = author,
        question = question,
        context = context,
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

async fn build_thread_context(things: &ThingsClient, start_post: Option<&Post>) -> String {
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

/// Determine the root post id of the thread a post belongs to.
///
/// Reuses the thread id already stored for the parent point in Qdrant when
/// possible; otherwise walks the parent chain through the Things API, backfilling
/// every ancestor into memory as it goes.
async fn resolve_thread(
    state: &RwLock<AppState>,
    post_data: &models::PostData,
    post: &Post,
) -> u64 {
    let post_id = post.id_value().unwrap_or_default();
    let parent_id = post
        .parent_id
        .or_else(|| post_data.parent.as_ref().and_then(|p| p.id_value()));

    match parent_id {
        Some(parent_id) => {
            let qdrant = state.read().await.qdrant.clone();
            if qdrant.is_available() {
                if let Ok(Some(entry)) = qdrant.get_point(parent_id).await {
                    return entry.thread_id;
                }
            }
            walk_thread_root_and_backfill(state, parent_id).await
        }
        None => post_id,
    }
}

async fn walk_thread_root_and_backfill(state: &RwLock<AppState>, start_parent_id: u64) -> u64 {
    let mut items: Vec<Post> = Vec::new();
    let mut current_id = start_parent_id;
    let mut seen: HashSet<u64> = HashSet::new();
    let mut root = start_parent_id;
    let mut depth = 0;

    while depth < MAX_CONTEXT_DEPTH {
        if !seen.insert(current_id) {
            break;
        }
        let things = &state.read().await.things;
        let Ok(data) = things.get_post(current_id).await else {
            break;
        };
        let Some(parent_post) = data.post.clone() else {
            break;
        };
        items.push(parent_post.clone());
        match parent_post.parent_id {
            Some(next) => {
                root = next;
                current_id = next;
            }
            None => break,
        }
        depth += 1;
    }

    let qdrant = state.read().await.qdrant.clone();
    if qdrant.is_available() {
        let writer = state.read().await.memory_writer.clone();
        for parent_post in items {
            let payload = MessagePayload {
                id: parent_post.id_value().unwrap_or_default(),
                content: strip_mention(parent_post.content_text()).to_string(),
                username: parent_post.author_username().to_string(),
                message_type: MessageType::Reply,
                parent_id: parent_post.parent_id,
                thread_id: root,
                timestamp: timestamp_from_post(&parent_post),
                is_processed: false,
                media_urls: extract_media_urls(&parent_post),
            };
            let _ = writer.send(MemoryWrite::Upsert(payload));
        }
    }

    root
}

fn context_search_limit() -> u64 {
    std::env::var("CONTEXT_SEARCH_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50)
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

async fn download_and_upload_media(
    state: &RwLock<AppState>,
    gemini: &GeminiClient,
    url: &str,
) -> Result<(String, String)> {
    let (data, mime_type) = {
        let things = &state.read().await.things;
        things.download_media(url).await?
    };

    let display_name = format!("media_{}", uuid::Uuid::new_v4());
    let file_uri = gemini.upload_file(data, &mime_type, &display_name).await?;

    Ok((file_uri, mime_type))
}
