mod entities;
mod gemini_client;
mod models;
mod things_client;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::io::AsyncBufReadExt;
use tokio::sync::{RwLock, mpsc};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use crate::entities::build_reply_with_entities;
use crate::gemini_client::GeminiClient;
use crate::models::{Notification, Post};
use crate::things_client::ThingsClient;

const BOT_USERNAME: &str = "AskMe";
const POLL_INTERVAL_MS: u64 = 3_000;
const MAX_RESPONSE_LENGTH: usize = 500;
const MAX_CONTEXT_DEPTH: usize = 20;
const MAX_MEDIA_FILES: usize = 5;
const PROCESSED_IDS_FILE: &str = ".processed-ids.json";
const MAX_PROCESSED_IDS: usize = 5000;

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct ReplyEntry {
    original_post_id: u64,
    original_content: String,
    ai_response: String,
    parent_context: Option<String>,
}

struct AppState {
    things: ThingsClient,
    gemini: GeminiClient,
    system_prompt: String,
    processed_notifications: HashSet<u64>,
    replied_posts: HashMap<u64, ReplyEntry>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    dotenvy::dotenv().ok();

    let gemini_api_key = std::env::var("GEMINI_API_KEY")
        .expect("GEMINI_API_KEY must be set in .env");
    let things_email = std::env::var("THINGS_EMAIL")
        .expect("THINGS_EMAIL must be set in .env");
    let things_password = std::env::var("THINGS_PASSWORD")
        .expect("THINGS_PASSWORD must be set in .env");

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

    let state = Arc::new(RwLock::new(AppState {
        things,
        gemini,
        system_prompt,
        processed_notifications: load_processed_ids(),
        replied_posts: HashMap::new(),
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

    let post = post_data.post.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Post {post_id} has no content"))?;

    println!("=== Post {post_id} by {} ===", post.author_username());
    println!("Comment: {}", post.content_text());

    let media_urls = extract_media_urls(post);
    println!("Media URLs found: {}", media_urls.len());
    for (i, url) in media_urls.iter().enumerate() {
        println!("  [{i}] {url}");
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

    let question = extract_question(post);
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
                build_follow_up_prompt(state, post, &question, &post_data, post_id).await
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
        println!("  bold offset={} length={} (chars {}-{})", e.offset, e.length, e.offset, e.offset + e.length);
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
    tokio::signal::ctrl_c().await.expect("Failed to listen for Ctrl+C");
    info!("Received Ctrl+C signal");
}

fn load_processed_ids() -> HashSet<u64> {
    let mut ids = HashSet::new();
    match std::fs::read_to_string(PROCESSED_IDS_FILE) {
        Ok(content) => match serde_json::from_str::<Vec<u64>>(&content) {
            Ok(list) => {
                for id in list {
                    ids.insert(id);
                }
                info!("Loaded {} processed notification IDs", ids.len());
            }
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

fn save_processed_ids(ids: &HashSet<u64>) {
    let mut list: Vec<u64> = ids.iter().copied().collect();
    list.sort_unstable();
    if list.len() > MAX_PROCESSED_IDS {
        list.drain(..list.len() - MAX_PROCESSED_IDS);
    }
    match serde_json::to_string_pretty(&list) {
        Ok(content) => {
            match std::fs::write(PROCESSED_IDS_FILE, content) {
                Ok(()) => info!("Persisted {} processed notification IDs to {PROCESSED_IDS_FILE}", list.len()),
                Err(e) => warn!("Failed to save {PROCESSED_IDS_FILE}: {e}"),
            }
        }
        Err(e) => warn!("Failed to serialize processed notification IDs: {e}"),
    }
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
        {
            let state_read = state.read().await;
            for notification in &notifications {
                if state_read.processed_notifications.contains(&notification.id) {
                    continue;
                }
                if is_mention_notification(notification) || is_follow_up_notification(notification, &state_read.replied_posts) {
                    to_process.push(notification.clone());
                }
            }
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

        {
            let state_read = state.read().await;
            save_processed_ids(&state_read.processed_notifications);
        }
    }
}

fn is_mention_notification(notification: &Notification) -> bool {
    let nt = notification.notification_type.as_deref().unwrap_or("");
    let group = notification.group.as_deref().unwrap_or("");
    nt == "user_mention" || nt == "mention" || group == "mentions"
}

fn is_follow_up_notification(notification: &Notification, replied_posts: &HashMap<u64, ReplyEntry>) -> bool {
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
    if replied_posts.contains_key(&original_post_id) {
        return true;
    }
    original_post
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
        .unwrap_or(false)
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
            .or_else(|| notification.original_post_data.as_ref().and_then(|p| p.id_value()));
    }
    notification
        .post_data
        .as_ref()
        .and_then(|p| p.id_value())
        .or_else(|| notification.reply_post_data.as_ref().and_then(|p| p.id_value()))
}

async fn process_notification(state: Arc<RwLock<AppState>>, notification: Notification) {
    let notification_id = notification.id;

    let post_id = match notification_post_id(&notification) {
        Some(id) => id,
        None => {
            warn!("Notification {} has no post data, skipping", notification.id);
            state.write().await.processed_notifications.insert(notification_id);
            return;
        }
    };

    let bot_reply_id = notification
        .original_post_data
        .as_ref()
        .and_then(|p| p.id_value());

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
            state.write().await.processed_notifications.insert(notification_id);
            return;
        }
    };

    let is_follow_up = is_follow_up_notification(&notification, &state.read().await.replied_posts);

    let question = extract_question(&post);
    if question.len() < 2 {
        info!("Notification {notification_id}: question too short, skipping");
        state.write().await.processed_notifications.insert(notification_id);
        return;
    }

    let user_text = if is_follow_up {
        let chain_start = bot_reply_id.unwrap_or(post_id);
        build_follow_up_prompt(&state, &post, &question, &post_data, chain_start).await
    } else {
        build_mention_prompt(&post_data, &post, &question)
    };

    info!("Prompt for notification {notification_id}: {}", truncate_text(&user_text, 400));

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
        match things.reply_to_post(post_id, &reply_text, &reply_entities).await {
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
        let mut state_write = state.write().await;
        state_write.replied_posts.insert(reply_id, ReplyEntry {
            original_post_id: post_id,
            original_content: question.clone(),
            ai_response: reply_text,
            parent_context: None,
        });
        state_write.processed_notifications.insert(notification_id);
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
    let parent_context = post_data.parent.as_ref()
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

async fn build_follow_up_prompt(
    state: &RwLock<AppState>,
    post: &Post,
    question: &str,
    post_data: &models::PostData,
    chain_start: u64,
) -> String {
    let author = post.author_username();

    let thread_context = {
        let things = &state.read().await.things;
        build_thread_context(things, post_data.parent.as_ref()).await
    };

    let context_chain = if thread_context.is_empty() {
        let replied_posts = &state.read().await.replied_posts;
        build_context_chain(replied_posts, chain_start)
    } else {
        thread_context
    };

    format!(
        "[Follow-up question by {author}] {question}\n{context_chain}",
        author = author,
        question = question,
        context_chain = context_chain,
    )
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

fn build_context_chain(replied_posts: &HashMap<u64, ReplyEntry>, post_id: u64) -> String {
    let mut chain = String::from("\n[Conversation so far]");
    let mut current = post_id;
    let mut depth = 0;

    while depth < MAX_CONTEXT_DEPTH {
        match replied_posts.get(&current) {
            Some(entry) => {
                let prev = format!(
                    "\nUser: {}\nAskMe: {}",
                    entry.original_content,
                    entry.ai_response,
                );
                chain.push_str(&prev);
                current = entry.original_post_id;
                depth += 1;
            }
            None => break,
        }
    }

    chain
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
