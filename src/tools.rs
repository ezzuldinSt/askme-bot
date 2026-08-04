//! Gemini tool calling: the tool set, their declarations, and the executor
//! the reply flow uses to run function calls the model requests.
//!
//! Tools let the bot fetch web pages, look up Things users/profiles/posts,
//! and read its own long-term memory about other users. Tool execution never
//! fails the reply flow: an error becomes an `{"error": ...}` result that the
//! model can honestly report to the user.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};

use crate::config::RuntimeConfig;
use crate::models::*;
use crate::qdrant_client::QdrantClient;
use crate::qdrant_models::UserFactPayload;
use crate::things_client::ThingsClient;

/// Where an extraction job's text came from (shapes how facts are saved).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionSource {
    /// A user's message in a conversation.
    Conversation,
    /// Posts scanned during a user-profile lookup (facts about that user).
    ProfileScan,
}

/// A text queued for the background fact-extraction pass.
pub struct ExtractionJob {
    pub username: String,
    /// The text to extract facts from (mention already stripped).
    pub text: String,
    pub post_id: u64,
    pub conversation_id: u64,
    pub source: ExtractionSource,
}

/// A user the model looked up during one reply flow (profile/posts/facts).
/// The reply flow may auto-brief the model on this user before the final
/// reply so the answer includes their saved facts and a post summary.
#[derive(Debug, Clone)]
pub struct FlowSubject {
    pub user_id: Option<u64>,
    pub username: Option<String>,
    /// Recent posts returned by get_user_posts during this flow (if any).
    pub posts: Option<Value>,
    /// A briefing context part was already injected for this user.
    pub briefed: bool,
    /// Facts for this user were already extracted inline during this flow.
    pub extracted: bool,
}

/// Lowest similarity for the cross-user fact search. Deliberately loose: the
/// model receives the results with their owners and picks the right person.
const GLOBAL_FACT_SEARCH_THRESHOLD: f32 = 0.30;
/// Max characters of extracted webpage text sent back to the model.
const WEB_FETCH_TEXT_CAP: usize = 12_000;
/// Max characters of joined posts sent to the background extractor.
const PROFILE_SCAN_TEXT_CAP: usize = 6_000;

/// Curated Arabic→Latin transliterations for common names. Most Things
/// usernames are Latin, so an Arabic query must be converted before the API
/// (which is ASCII-only) can find anyone. Keys are normalized (see
/// `normalize_arabic`) before lookup.
const ARABIC_NAME_VARIANTS: &[(&str, &[&str])] = &[
    ("خالد", &["khaled", "khalid"]),
    ("محمد", &["mohammed", "mohammad", "mohamed", "muhammed"]),
    ("احمد", &["ahmed", "ahmad"]),
    ("عبدالله", &["abdullah", "abdallah", "abdulla"]),
    ("عبدالرحمن", &["abdulrahman", "abdurrahman"]),
    ("عبدالعزيز", &["abdulaziz"]),
    ("علي", &["ali"]),
    ("عمر", &["omar", "omer"]),
    ("سارة", &["sara", "sarah"]),
    ("نورة", &["noura", "nora", "noora"]),
    ("فهد", &["fahad", "fahd"]),
    ("سلطان", &["sultan"]),
    ("فيصل", &["faisal", "faysal"]),
    ("ريم", &["reem", "rim"]),
    ("هند", &["hind"]),
    ("سعود", &["saud"]),
    ("يوسف", &["yousef", "yusuf"]),
    ("ابراهيم", &["ibrahim", "ebrahim"]),
    ("حسن", &["hasan", "hassan"]),
    ("حسين", &["hussein", "hussain", "husain"]),
    ("موسى", &["mousa", "musa"]),
    ("عيسى", &["issa", "isa"]),
    ("ليلى", &["laila", "layla"]),
    ("مريم", &["maryam", "mariam"]),
    ("سلمان", &["salman"]),
    ("خديجة", &["khadija", "khadijah"]),
    ("فاطمة", &["fatima", "fatimah"]),
    ("عائشة", &["aisha"]),
    ("تركي", &["turki"]),
    ("ناصر", &["nasser", "naser"]),
    ("ماجد", &["majed", "majid"]),
    ("وليد", &["waleed", "walid", "waled"]),
    ("سعيد", &["saeed", "said"]),
    ("خليفة", &["khalifa"]),
    ("حمد", &["hamad"]),
    ("راشد", &["rashid", "rashed"]),
    ("عبدالهادي", &["abdulhadi"]),
    ("طلال", &["talal"]),
    ("بدر", &["bader", "badr"]),
    ("مشعل", &["mishal", "meshal"]),
];

/// Shared handles the tools need. Built once per reply flow (cheap Arc clones).
#[derive(Clone)]
pub struct ToolContext {
    pub things: Arc<ThingsClient>,
    pub qdrant: Arc<QdrantClient>,
    pub runtime: Arc<RwLock<RuntimeConfig>>,
    pub extraction_tx: mpsc::UnboundedSender<ExtractionJob>,
    /// Users the model looked up during this reply flow (shared with the
    /// reply flow itself, which may auto-brief the model on them).
    pub flow_subjects: Arc<Mutex<Vec<FlowSubject>>>,
}

impl ToolContext {
    pub fn new(
        things: Arc<ThingsClient>,
        qdrant: Arc<QdrantClient>,
        runtime: Arc<RwLock<RuntimeConfig>>,
        extraction_tx: mpsc::UnboundedSender<ExtractionJob>,
        flow_subjects: Arc<Mutex<Vec<FlowSubject>>>,
    ) -> Self {
        Self {
            things,
            qdrant,
            runtime,
            extraction_tx,
            flow_subjects,
        }
    }

    pub async fn tools_enabled(&self) -> bool {
        self.runtime.read().await.tools.enabled
    }

    pub async fn max_tool_rounds(&self) -> usize {
        self.runtime.read().await.tools.max_rounds
    }

    pub async fn url_context_enabled(&self) -> bool {
        self.runtime.read().await.tools.url_context_enabled
    }

    /// Execute one requested tool call and return the JSON result the model
    /// should see. Unknown tools and handler failures become `{"error": ...}`.
    pub async fn execute(&self, name: &str, args: &Value) -> Value {
        let result = match name {
            "web_search" => self.web_search(args).await,
            "web_fetch" => self.web_fetch(args).await,
            "search_users" => self.search_users(args).await,
            "get_user_profile" => self.get_user_profile(args).await,
            "get_user_posts" => self.get_user_posts(args).await,
            "get_user_facts" => self.get_user_facts(args).await,
            "search_user_facts" => self.search_user_facts(args).await,
            "get_post" => self.get_post(args).await,
            "get_thread" => self.get_thread(args).await,
            "get_current_time" => self.get_current_time(args),
            other => {
                warn!("Tool call to unknown tool {other}");
                json!({ "error": format!("unknown tool: {other}") })
            }
        };
        self.observe_subject(name, args, &result);
        result
    }

    /// Record users the model looked up via profile/posts/facts tools, so the
    /// reply flow can auto-brief the model on them before the final answer.
    fn observe_subject(&self, name: &str, args: &Value, result: &Value) {
        if result.get("error").is_some() {
            return;
        }
        let (user_id, username, posts) = match name {
            "get_user_profile" | "get_user_posts" => (
                arg_u64(args, "user_id"),
                result
                    .get("username")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                if name == "get_user_posts" {
                    result.get("posts").cloned()
                } else {
                    None
                },
            ),
            "get_user_facts" => (None, arg_str(args, "username").map(String::from), None),
            _ => return,
        };
        if let Some(u) = &username {
            if u.eq_ignore_ascii_case(crate::BOT_USERNAME) {
                return;
            }
        }
        let mut subjects = self.flow_subjects.lock().unwrap();
        let existing = subjects.iter_mut().find(|s| {
            (user_id.is_some() && s.user_id == user_id)
                || (username.is_some()
                    && s.username.as_deref().is_some_and(|n| {
                        n.eq_ignore_ascii_case(username.as_deref().unwrap_or(""))
                    }))
        });
        match existing {
            Some(s) => {
                if s.user_id.is_none() {
                    s.user_id = user_id;
                }
                if s.username.is_none() {
                    s.username = username;
                }
                if posts.is_some() {
                    s.posts = posts;
                }
            }
            None => subjects.push(FlowSubject {
                user_id,
                username,
                posts,
                briefed: false,
                extracted: false,
            }),
        }
    }

    // ── web_search ──

    async fn web_search(&self, args: &Value) -> Value {
        let Some(query) = arg_str(args, "query") else {
            return err("missing query argument");
        };
        let num_results = crate::search::clamp_num_results(args.get("numResults").and_then(|v| v.as_u64()));
        let timeout_secs = self.runtime.read().await.tools.web_fetch_timeout_secs;
        let client = match reqwest::Client::builder()
            // Exa can be slower than a plain page fetch; floor at 25s.
            .timeout(Duration::from_secs(timeout_secs.max(25)))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
        {
            Ok(c) => c,
            Err(e) => return err(format!("failed to build search client: {e}")),
        };

        // Primary provider: Exa. Fallback: DDG Lite scraping.
        let mut provider = crate::search::SearchProvider::Exa;
        let results = match crate::search::exa_search(&client, query, num_results).await {
            Ok(r) if !r.is_empty() => r,
            Ok(_) => {
                info!("web_search {query:?}: exa returned no results, falling back to DDG");
                match crate::search::ddg_search(&client, query, num_results).await {
                    Ok(r) => {
                        provider = crate::search::SearchProvider::Ddg;
                        r
                    }
                    Err(e) => return err(format!("search failed: {e}")),
                }
            }
            Err(e) => {
                info!("web_search {query:?}: exa failed ({e}), falling back to DDG");
                match crate::search::ddg_search(&client, query, num_results).await {
                    Ok(r) => {
                        provider = crate::search::SearchProvider::Ddg;
                        r
                    }
                    Err(e2) => return err(format!("search failed: {e2}")),
                }
            }
        };

        info!(
            "web_search {query:?} via {}: {} results",
            crate::search::provider_label(provider),
            results.len()
        );
        if results.is_empty() {
            return json!({
                "query": query,
                "results": [],
                "note": "No results found. Try rephrasing the query (shorter, fewer words).",
            });
        }
        let results: Vec<Value> = results.iter().map(|r| r.to_json()).collect();
        json!({
            "query": query,
            "results": results,
            "note": "Use web_fetch on the most relevant url when you need details beyond the snippet.",
        })
    }

    // ── web_fetch ──

    async fn web_fetch(&self, args: &Value) -> Value {
        let Some(url) = arg_str(args, "url") else {
            return err("missing url argument");
        };
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return err("only http/https URLs are allowed");
        }
        let (max_bytes, timeout_secs) = {
            let runtime = self.runtime.read().await;
            (
                runtime.tools.web_fetch_max_bytes,
                runtime.tools.web_fetch_timeout_secs,
            )
        };
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
        {
            Ok(c) => c,
            Err(e) => return err(format!("failed to build fetch client: {e}")),
        };

        // Real-browser headers: many sites serve different (or no) content to
        // bots. If Cloudflare still answers with a challenge, retry once with
        // the honest bot UA (opencode's strategy for TLS-fingerprint blocks).
        const BROWSER_UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36";
        const BOT_UA: &str = "AskMeBot/1.0 (+https://things.cv/@AskMe)";

        let mut resp = match client
            .get(url)
            .header(reqwest::header::USER_AGENT, BROWSER_UA)
            .header(
                reqwest::header::ACCEPT,
                "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            )
            .header(reqwest::header::ACCEPT_LANGUAGE, "en-US,en;q=0.9")
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return err(format!("request failed: {e}")),
        };
        let cf_challenge = resp.status().as_u16() == 403
            && resp
                .headers()
                .get("cf-mitigated")
                .and_then(|v| v.to_str().ok())
                == Some("challenge");
        if cf_challenge {
            warn!("web_fetch {url}: Cloudflare challenge, retrying with honest UA");
            resp = match client.get(url).header(reqwest::header::USER_AGENT, BOT_UA).send().await {
                Ok(r) => r,
                Err(e) => return err(format!("request failed: {e}")),
            };
        }
        if !resp.status().is_success() {
            return err(format!("page returned HTTP {}", resp.status().as_u16()));
        }

        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();
        let is_text =
            content_type.contains("text/") || content_type.contains("json") || content_type.contains("xml");
        if !is_text {
            return err(format!("not a text page (content type: {content_type})"));
        }

        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => return err(format!("failed to read page: {e}")),
        };
        let oversized = bytes.len() > max_bytes;
        let raw = String::from_utf8_lossy(&bytes[..bytes.len().min(max_bytes)]);

        let (title, text) = if content_type.contains("json") {
            (String::new(), raw.trim().to_string())
        } else {
            html_to_text(&raw)
        };
        let text = truncate_chars(&text, WEB_FETCH_TEXT_CAP);
        info!("web_fetch {url}: {} bytes -> {} chars of text", bytes.len(), text.chars().count());
        json!({
            "url": url,
            "title": title,
            "truncated": oversized || text.chars().count() >= WEB_FETCH_TEXT_CAP,
            "text": text,
        })
    }

    // ── Things user lookups ──

    async fn search_users(&self, args: &Value) -> Value {
        let Some(query) = arg_str(args, "query") else {
            return err("missing query argument");
        };
        let queries = user_search_queries(query);

        // Search the original query plus any Arabic transliteration variants,
        // merging by user id so the same user found twice appears once.
        let mut rows: Vec<(UserSearchRow, String)> = Vec::new();
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for q in &queries {
            match self.things.search_users(q, 15).await {
                Ok(found) => {
                    for row in found {
                        if seen.insert(row.id) {
                            rows.push((row, q.clone()));
                        }
                    }
                }
                Err(e) => {
                    warn!("search_users({q}) failed: {e}");
                }
            }
        }

        if rows.is_empty() {
            return json!({
                "matches": [],
                "note": format!("No users matched '{query}'. Try a different spelling or a username."),
            });
        }

        let ranked = rank_user_rows(query, rows);
        let best_score = ranked.first().map(|r| r.1).unwrap_or(6);
        let best_count = ranked.iter().filter(|r| r.1 == best_score).count();
        let ambiguous = best_count > 1;
        let note = if ambiguous {
            let names: Vec<String> = ranked
                .iter()
                .take(best_count)
                .map(|r| r.0.username.as_deref().unwrap_or("?").to_string())
                .collect();
            format!(
                "Several users match '{query}' ({}). The asker's intended person is unclear — list these candidates and ask which one, instead of guessing.",
                names.join(", ")
            )
        } else if let Some((row, _, is_username_match, _)) = ranked.first() {
            let why = if *is_username_match {
                "exact username match"
            } else {
                "exact display-name match"
            };
            format!(
                "Picked @{} (id {}) — {why} for '{query}'. If the asker meant someone else, say so.",
                row.username.as_deref().unwrap_or("?"),
                row.id
            )
        } else {
            format!("No users matched '{query}'. Try a different spelling or a username.")
        };

        let matches: Vec<Value> = ranked
            .into_iter()
            .take(24)
            .map(|(row, _score, is_username_match, matched_query)| {
                json!({
                    "id": row.id,
                    "username": row.username,
                    "name": row.name,
                    "bio": row.bio,
                    "is_private": row.is_private,
                    "is_verified": row.is_verified,
                    "is_premium": row.is_premium,
                    "streak": row.streak,
                    "matched_query": matched_query,
                    "is_username_match": is_username_match,
                })
            })
            .collect();
        json!({ "matches": matches, "note": note })
    }


    async fn get_user_profile(&self, args: &Value) -> Value {
        let Some(user_id) = arg_u64(args, "user_id") else {
            return err("missing user_id argument");
        };
        match self.things.get_user(user_id).await {
            Ok(p) => json!({
                "id": p.id,
                "username": p.username,
                "name": p.name,
                "bio": p.bio,
                "joined_at": p.joined_at,
                "streak": p.streak,
                "is_private": p.is_private,
                "is_verified": p.is_verified,
                "is_premium": p.is_premium,
                "sticky_status": p.sticky_status,
            }),
            Err(e) => {
                warn!("get_user_profile({user_id}) failed: {e}");
                err(format!("failed to fetch profile: {e}"))
            }
        }
    }

    async fn get_user_posts(&self, args: &Value) -> Value {
        let Some(user_id) = arg_u64(args, "user_id") else {
            return err("missing user_id argument");
        };
        let requested = arg_u64(args, "limit").unwrap_or(10);
        // Config knob caps the default; the tool declaration promises 1-10.
        let limit = requested
            .min(self.runtime.read().await.tools.user_scan_posts_limit)
            .clamp(1, 10);
        let page = match self.things.get_user_posts(user_id, limit).await {
            Ok(p) => p,
            Err(e) => {
                warn!("get_user_posts({user_id}) failed: {e}");
                return err(format!("failed to fetch posts: {e}"));
            }
        };
        let rows = page.data.unwrap_or_default();
        let username = rows
            .first()
            .and_then(|p| p.user.as_ref())
            .and_then(|u| u.username.clone());
        let posts: Vec<Value> = rows
            .iter()
            .map(|p| {
                json!({
                    "id": p.id,
                    "created_at": p.created_at,
                    "post_type": p.post_type,
                    "content": p.content_text(),
                })
            })
            .collect();

        // Profile-scan fact extraction is handled by the reply flow: users
        // looked up via this tool are recorded as flow subjects, and the
        // extraction runs inline (before the final reply) or is queued as a
        // safety net at the end of the flow — so facts learned from these
        // posts are available before the reply is written.

        json!({
            "username": username,
            "posts": posts,
            "note": "only currently-visible posts are returned (Things posts expire after a few hours)",
        })
    }

    // ── Saved long-term memory ──

    async fn get_user_facts(&self, args: &Value) -> Value {
        let Some(username) = arg_str(args, "username") else {
            return err("missing username argument");
        };
        if username.eq_ignore_ascii_case(crate::BOT_USERNAME) {
            return json!({ "username": username, "facts": [] });
        }
        if !self.qdrant.is_available() {
            return err("memory is unavailable");
        }
        let limit = self.runtime.read().await.memory.user_facts_limit.max(20);
        match self.qdrant.list_user_facts(username, limit).await {
            Ok(facts) => json!({
                "username": username,
                "facts": facts.into_iter().map(|(_, f)| fact_json(&f)).collect::<Vec<_>>(),
            }),
            Err(e) => {
                warn!("get_user_facts({username}) failed: {e}");
                err(format!("failed to load facts: {e}"))
            }
        }
    }

    async fn search_user_facts(&self, args: &Value) -> Value {
        let Some(query) = arg_str(args, "query") else {
            return err("missing query argument");
        };
        if !self.qdrant.is_available() {
            return err("memory is unavailable");
        }
        let vector = match self.qdrant.embed(query).await {
            Ok(v) => v,
            Err(e) => return err(format!("failed to embed query: {e}")),
        };
        let mut hits = match self
            .qdrant
            .search_user_facts_global(&vector, GLOBAL_FACT_SEARCH_THRESHOLD, 25)
            .await
        {
            Ok(h) => h,
            Err(e) => {
                warn!("search_user_facts({query}) failed: {e}");
                return err(format!("fact search failed: {e}"));
            }
        };
        // Exact/substring username hits outrank pure-semantic ones.
        let needle = query.to_lowercase();
        hits.sort_by_key(|(username, _)| {
            if username.to_lowercase() == needle || username.to_lowercase().contains(&needle) {
                0
            } else {
                1
            }
        });
        hits.truncate(10);
        json!({
            "query": query,
            "matches": hits.into_iter().map(|(username, f)| {
                let mut v = fact_json(&f);
                v["username"] = json!(username);
                v
            }).collect::<Vec<_>>(),
        })
    }

    // ── Posts / threads ──

    async fn get_post(&self, args: &Value) -> Value {
        let Some(post_id) = arg_u64(args, "post_id") else {
            return err("missing post_id argument");
        };
        match self.things.get_post(post_id).await {
            Ok(data) => {
                let post = data.post.as_ref();
                json!({
                    "id": post_id,
                    "author": post.map(Post::author_username),
                    "content": post.map(|p| p.content_text()),
                    "post_type": post.and_then(|p| p.post_type.as_deref()),
                    "created_at": post.and_then(|p| p.created_at.as_deref()),
                    "has_parent": data.parent.is_some(),
                })
            }
            Err(e) => {
                warn!("get_post({post_id}) failed: {e}");
                err(format!("failed to fetch post: {e}"))
            }
        }
    }

    async fn get_thread(&self, args: &Value) -> Value {
        let Some(post_id) = arg_u64(args, "post_id") else {
            return err("missing post_id argument");
        };
        let data = match self.things.get_post(post_id).await {
            Ok(d) => d,
            Err(e) => {
                warn!("get_thread({post_id}) failed: {e}");
                return err(format!("failed to fetch post: {e}"));
            }
        };
        let Some(post) = data.post.as_ref() else {
            return err("post has no content");
        };
        let thread = crate::build_thread_context(&self.things, data.parent.as_ref()).await;
        json!({
            "post_id": post_id,
            "post": json!({
                "author": post.author_username(),
                "content": post.content_text(),
            }),
            "thread_above": thread,
        })
    }

    // ── Misc ──

    fn get_current_time(&self, _args: &Value) -> Value {
        let now = chrono::Utc::now();
        json!({
            "utc": now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            "unix_timestamp": now.timestamp(),
        })
    }
}

/// Query strings to try for a user search: the original first, then Latin
/// transliteration variants when the query is Arabic (most Things
/// usernames are Latin, so an Arabic name alone matches nothing).
fn user_search_queries(query: &str) -> Vec<String> {
    let mut out = vec![query.trim().to_string()];
    let mut push = |v: String| {
        if !out.iter().any(|x| x.eq_ignore_ascii_case(&v)) {
            out.push(v);
        }
    };
    let normalized = normalize_arabic(query.trim());
    if contains_arabic(&normalized) {
        let matched: Vec<&str> = ARABIC_NAME_VARIANTS
            .iter()
            .filter(|(ar, _)| normalize_arabic(ar) == normalized)
            .flat_map(|(_, variants)| variants.iter().copied())
            .collect();
        if !matched.is_empty() {
            for v in matched {
                push(v.to_string());
            }
        } else {
            for word in normalized.split_whitespace() {
                let latin = arabic_to_latin(word);
                if latin.chars().count() >= 3 {
                    push(latin);
                }
            }
        }
    }
    out
}

/// Score each row against the request and its variants (lower is better):
/// 0 exact username, 1 exact display name, 2 username prefix,
/// 3 display-name prefix, 4 username substring, 5 display-name substring,
/// 6 no field match (API order). Returns (row, score, is_username_match,
/// best-matched query).
fn rank_user_rows(
    requested: &str,
    rows: Vec<(UserSearchRow, String)>,
) -> Vec<(UserSearchRow, u8, bool, String)> {
    let variants = user_search_queries(requested);
    let mut ranked: Vec<(UserSearchRow, u8, bool, String)> = Vec::new();
    for (row, _from_query) in rows {
        let username = row.username.as_deref().unwrap_or("").trim();
        let name = row.name.as_deref().unwrap_or("").trim();
        let username_l = username.to_ascii_lowercase();
        let name_l = name.to_ascii_lowercase();
        let mut best: Option<(u8, bool, String)> = None;
        for v in &variants {
            let v = v.trim();
            let v_l = v.to_ascii_lowercase();
            let (score, is_user) = if username_l == v_l {
                (0u8, true)
            } else if name_l == v_l {
                (1u8, false)
            } else if username_l.starts_with(&v_l) {
                (2u8, false)
            } else if name_l.starts_with(&v_l) {
                (3u8, false)
            } else if username_l.contains(&v_l) {
                (4u8, false)
            } else if name_l.contains(&v_l) {
                (5u8, false)
            } else {
                continue;
            };
            let is_better = match &best {
                None => true,
                Some((bs, _, _)) => score < *bs,
            };
            if is_better {
                best = Some((score, is_user, v.to_string()));
            }
        }
        match best {
            Some((score, is_user, matched)) => ranked.push((row, score, is_user, matched)),
            None => ranked.push((row, 6, false, String::new())),
        }
    }
    ranked.sort_by_key(|r| r.1);
    ranked
}

/// True when a string contains Arabic script characters.
fn contains_arabic(s: &str) -> bool {
    s.chars().any(|c| {
        matches!(
            c,
            '\u{0600}'..='\u{06FF}'
                | '\u{0750}'..='\u{077F}'
                | '\u{08A0}'..='\u{08FF}'
                | '\u{FB50}'..='\u{FDFF}'
                | '\u{FE70}'..='\u{FEFF}'
        )
    })
}

/// Normalize Arabic: strip diacritics and tatweel, unify hamza forms and
/// ة/ھ/ى variants so spellings compare equal.
fn normalize_arabic(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '\u{064B}'..='\u{0652}' | '\u{0640}' | '\u{0670}'))
        .map(|c| match c {
            'أ' | 'إ' | 'آ' => 'ا',
            'ة' | 'ۀ' | 'ھ' => 'ه',
            'ى' => 'ي',
            _ => c,
        })
        .collect()
}

/// Generic Arabic→Latin letter mapping (produces vowel-less words — a
/// last-resort fallback when the name is not in the curated table).
fn arabic_to_latin(word: &str) -> String {
    let mut out = String::new();
    for c in word.chars() {
        let mapped = match c {
            'ا' | 'أ' | 'إ' | 'آ' | 'ى' => "a",
            'ب' => "b",
            'ت' | 'ة' => "t",
            'ث' => "th",
            'ج' => "j",
            'ح' => "h",
            'خ' => "kh",
            'د' => "d",
            'ذ' => "dh",
            'ر' => "r",
            'ز' => "z",
            'س' => "s",
            'ش' => "sh",
            'ص' => "s",
            'ض' => "d",
            'ط' => "t",
            'ظ' => "z",
            'ع' => "'",
            'غ' => "gh",
            'ف' => "f",
            'ق' => "q",
            'ك' => "k",
            'ل' => "l",
            'م' => "m",
            'ن' => "n",
            'ه' | 'ھ' => "h",
            'و' => "w",
            'ي' => "y",
            ' ' => " ",
            _ => continue,
        };
        out.push_str(mapped);
    }
    out
}

fn fact_json(fact: &UserFactPayload) -> Value {
    json!({
        "fact": fact.fact,
        "category": fact.category.as_str(),
        "times_confirmed": fact.times_confirmed,
    })
}

fn arg_str<'a>(args: &'a Value, name: &str) -> Option<&'a str> {
    args.get(name)
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn arg_u64(args: &Value, name: &str) -> Option<u64> {
    args.get(name).and_then(|v| v.as_u64())
}

fn err(message: impl Into<String>) -> Value {
    json!({ "error": message.into() })
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect()
    }
}

/// Join post contents into the extraction text for a profile scan (capped).
/// Empty contents are skipped; an all-empty input yields an empty string
/// (meaning there is nothing worth extracting).
pub fn profile_scan_text(username: &str, contents: &[&str]) -> String {
    let joined: Vec<&str> = contents
        .iter()
        .filter(|c| !c.trim().is_empty())
        .copied()
        .collect();
    if joined.is_empty() {
        return String::new();
    }
    truncate_chars(
        &format!("Recent posts by {username}:\n{}", joined.join("\n---\n")),
        PROFILE_SCAN_TEXT_CAP,
    )
}

/// The tool declarations offered to the model. Keep descriptions explicit
/// about WHEN to use each tool and what it returns. When `url_context_enabled`
/// is set, the built-in URL context tool is added first: the model then
/// auto-fetches any http(s) URL present in the conversation, with zero extra
/// tool rounds.
pub fn tool_declarations(url_context_enabled: bool) -> Vec<Tool> {
    let params = |properties: Value, required: &[&str]| {
        let mut p = json!({ "type": "object", "properties": properties });
        if !required.is_empty() {
            p["required"] = json!(required);
        }
        p
    };

    let mut tools: Vec<Tool> = Vec::new();
    if url_context_enabled {
        tools.push(Tool::url_context());
    }

    tools.extend(vec![
        Tool {
            url_context: None,
            function_declarations: vec![FunctionDeclaration {
                name: "web_search".to_string(),
                description: format!(
                    "Search the web for current information. Returns up to 8 results (title, URL, snippet) by default, more with numResults. Use whenever the answer needs FRESH data your training may not have: news, releases, availability, prices, dates, reviews, recommendations ('what's the latest...', 'has X been released yet?', 'find me chalets in Abha with good ratings'). Then call web_fetch on the most relevant url for details. The current year is {year}. Use this year when searching for recent information or news.",
                    year = chrono::Utc::now().format("%Y"),
                ),
                parameters: params(
                    json!({
                        "query": { "type": "string", "description": "The search query — concise and specific" },
                        "numResults": { "type": "integer", "description": "How many results to return (1-20, default 8)" },
                    }),
                    &["query"],
                ),
            }],
        },
        Tool {
            url_context: None,
            function_declarations: vec![FunctionDeclaration {
                name: "web_fetch".to_string(),
                description: "Fetch a webpage and return its readable text content. FALLBACK TOOL: URLs in the user's message are fetched automatically by the URL context tool — do NOT call web_fetch for those. Use web_fetch only (1) for a URL that came from web_search results or elsewhere in the conversation, or (2) when the automatic URL fetch failed. Only http/https URLs. Large pages are truncated.".to_string(),
                parameters: params(
                    json!({ "url": { "type": "string", "description": "The http(s) URL to fetch" } }),
                    &["url"],
                ),
            }],
        },
        Tool {
            url_context: None,
            function_declarations: vec![FunctionDeclaration {
                name: "search_users".to_string(),
                description: "Search Things users by username, display name, or Arabic name (Arabic queries are automatically transliterated to Latin variants). Results are ranked: exact username match first, then exact display name, then prefix/substring matches. If several users match the requested name, the result note says so — list the candidate usernames and ask the asker which one, never guess silently. Use to resolve a username (with or without @) or a person's name into a user id and basic info.".to_string(),
                parameters: params(
                    json!({ "query": { "type": "string", "description": "Username or name fragment to search (Arabic names are transliterated automatically)" } }),
                    &["query"],
                ),
            }],
        },
        Tool {
            url_context: None,
            function_declarations: vec![FunctionDeclaration {
                name: "get_user_profile".to_string(),
                description: "Fetch a user's full public profile by numeric id: bio, display name, username, join date, streak, verified/premium flags. Use after search_users to learn more about a user the asker mentioned. For the bot's SAVED long-term facts about someone, use get_user_facts instead.".to_string(),
                parameters: params(
                    json!({ "user_id": { "type": "integer", "description": "The user's numeric id from search_users" } }),
                    &["user_id"],
                ),
            }],
        },
        Tool {
            url_context: None,
            function_declarations: vec![FunctionDeclaration {
                name: "get_user_posts".to_string(),
                description: "Fetch a user's most recent posts (newest first, up to 10) by numeric user id. Use to learn what a user talks about, their style or opinions. Only currently-visible posts are returned (Things posts expire after a few hours), so the list may be short or empty. Durable facts from these posts are saved to memory automatically.".to_string(),
                parameters: params(
                    json!({
                        "user_id": { "type": "integer", "description": "The user's numeric id from search_users" },
                        "limit": { "type": "integer", "description": "How many posts to fetch (1-10, default 10)" },
                    }),
                    &["user_id"],
                ),
            }],
        },
        Tool {
            url_context: None,
            function_declarations: vec![FunctionDeclaration {
                name: "get_user_facts".to_string(),
                description: "Return the bot's saved long-term facts about a user, by exact username (no @). Facts are learned from past conversations and profile scans (e.g. location, job, preferences). Use when asked 'what do you know about X?' or when a question references another user's personal details.".to_string(),
                parameters: params(
                    json!({ "username": { "type": "string", "description": "Exact username without @" } }),
                    &["username"],
                ),
            }],
        },
        Tool {
            url_context: None,
            function_declarations: vec![FunctionDeclaration {
                name: "search_user_facts".to_string(),
                description: "Search the bot's saved long-term facts across ALL users by name or keyword. Use when the exact username is unknown or the asker gave a real name ('what do you know about Khaled?'). Returns facts tagged with the user they belong to.".to_string(),
                parameters: params(
                    json!({ "query": { "type": "string", "description": "A person's name, username, or topic to search for" } }),
                    &["query"],
                ),
            }],
        },
        Tool {
            url_context: None,
            function_declarations: vec![FunctionDeclaration {
                name: "get_post".to_string(),
                description: "Fetch a single post by its numeric id. Use when the user links or references a specific post and you need its exact content.".to_string(),
                parameters: params(
                    json!({ "post_id": { "type": "integer", "description": "The post id" } }),
                    &["post_id"],
                ),
            }],
        },
        Tool {
            url_context: None,
            function_declarations: vec![FunctionDeclaration {
                name: "get_thread".to_string(),
                description: "Fetch the conversation thread above a post (its ancestors, oldest first). Use when the user references a post whose surrounding discussion matters for the answer.".to_string(),
                parameters: params(
                    json!({ "post_id": { "type": "integer", "description": "The post id whose thread to read" } }),
                    &["post_id"],
                ),
            }],
        },
        Tool {
            url_context: None,
            function_declarations: vec![FunctionDeclaration {
                name: "get_current_time".to_string(),
                description: "Return the current UTC time. Use when the answer depends on the date or time.".to_string(),
                parameters: params(json!({}), &[]),
            }],
        },
    ]);

    tools
}

/// Best-effort HTML -> readable text: strips scripts/styles/comments/tags,
/// decodes entities, and preserves paragraph structure. Returns (title, text).
fn html_to_text(html: &str) -> (String, String) {
    const BLOCK_TAGS: &[&str] = &[
        "p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th", "section",
        "article", "blockquote", "pre", "ul", "ol", "table", "header", "footer", "hr", "summary",
        "details", "main",
    ];

    let mut title = String::new();
    let mut out = String::new();
    let mut tag = String::new();
    let mut in_tag = false;
    let mut in_comment = false;
    let mut in_title = false;
    let mut skip: Option<String> = None; // script/style/noscript content
    let mut pending_nl = false;

    let mut i = 0;
    let bytes = html.as_bytes();
    while i < bytes.len() {
        let c = bytes[i] as char;

        if in_comment {
            if c == '-' && i + 2 < bytes.len() && bytes[i + 1] == b'-' && bytes[i + 2] == b'>' {
                in_comment = false;
                i += 3;
            } else {
                i += 1;
            }
            continue;
        }

        if c == '<' {
            if i + 3 < bytes.len() && bytes[i + 1] == b'!' && bytes[i + 2] == b'-' && bytes[i + 3] == b'-'
            {
                in_comment = true;
                i += 4;
                continue;
            }
            in_tag = true;
            tag.clear();
            i += 1;
            continue;
        }

        if in_tag {
            if c == '>' {
                in_tag = false;
                let t = tag.to_lowercase();
                let is_close = t.starts_with('/');
                let name = t
                    .trim_start_matches('/')
                    .split([' ', '\t', '\n', '/', '>'])
                    .next()
                    .unwrap_or("")
                    .to_string();

                if let Some(active) = skip.as_mut() {
                    if is_close && *active == name {
                        skip = None;
                    }
                } else if !is_close
                    && matches!(
                        name.as_str(),
                        "script" | "style" | "noscript" | "template"
                    )
                {
                    skip = Some(name);
                } else if !is_close && name == "title" {
                    in_title = true;
                } else if is_close && name == "title" {
                    in_title = false;
                } else if BLOCK_TAGS.contains(&name.as_str()) {
                    pending_nl = true;
                }
            } else {
                tag.push(c);
            }
            i += 1;
            continue;
        }

        if skip.is_some() {
            i += 1;
            continue;
        }
        if in_title {
            if c != '\n' && c != '\r' {
                title.push(c);
            }
            i += 1;
            continue;
        }

        if c == '\n' || c == '\r' || c == '\t' || c == ' ' && pending_nl {
            pending_nl = true;
        } else {
            if pending_nl && !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            pending_nl = false;
            out.push(c);
        }
        i += 1;
    }

    let text = collapse_blank_lines(&crate::search::decode_entities(&out));
    let title = crate::search::decode_entities(&title).trim().to_string();
    (title, text.trim().to_string())
}

/// Collapse runs of 3+ newlines down to a blank line.
fn collapse_blank_lines(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut nl = 0usize;
    for ch in s.chars() {
        if ch == '\n' {
            nl += 1;
            if nl <= 2 {
                out.push(ch);
            }
        } else {
            nl = 0;
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_to_text_strips_tags_and_scripts() {
        let html = "<html><head><title>My Page</title></head><body>\
            <script>alert('x');</script>\
            <h1>Hello</h1><p>World &amp; more</p>\
            <style>.x{}</style><div>Second</div></body></html>";
        let (title, text) = html_to_text(html);
        assert_eq!(title, "My Page");
        assert!(!text.contains("alert"));
        assert!(!text.contains(".x{}"));
        assert!(text.contains("Hello"));
        assert!(text.contains("World & more"));
        assert!(text.contains("Second"));
    }

    #[test]
    fn html_to_text_blocks_become_newlines() {
        let html = "<p>one</p><p>two</p><br>three";
        let (_, text) = html_to_text(html);
        assert!(text.lines().count() >= 3);
        assert!(text.contains("one\ntwo\nthree") || text.contains("one\ntwo"));
    }

    #[test]
    fn html_to_text_handles_comments_and_title_skipping() {
        let html = "a<!-- hidden -->b<title>t</title>c";
        let (title, text) = html_to_text(html);
        assert_eq!(title, "t");
        assert!(!text.contains("hidden"));
        assert!(text.contains('a') && text.contains('b'));
    }

    #[test]
    fn tool_declarations_have_expected_tools() {
        let tools = tool_declarations(true);
        let names: Vec<&str> = tools
            .iter()
            .flat_map(|t| &t.function_declarations)
            .map(|f| f.name.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "web_search",
                "web_fetch",
                "search_users",
                "get_user_profile",
                "get_user_posts",
                "get_user_facts",
                "search_user_facts",
                "get_post",
                "get_thread",
                "get_current_time",
            ]
        );
        let serialized = serde_json::to_value(&tools).unwrap();
        assert_eq!(
            serialized[1]["functionDeclarations"][0]["parameters"]["type"],
            "object"
        );
    }

    #[test]
    fn url_context_tool_included_when_enabled() {
        let on = serde_json::to_value(tool_declarations(true)).unwrap();
        assert!(on[0]["urlContext"].is_object(), "url_context tool first when enabled");

        let off = serde_json::to_value(tool_declarations(false)).unwrap();
        let off_array = off.as_array().expect("serialized tools are an array");
        assert!(
            off_array
                .iter()
                .all(|t| t["urlContext"].is_null() && t["functionDeclarations"].is_array()),
            "no urlContext entry when disabled"
        );
    }

    #[test]
    fn user_search_queries_arabic_name_yields_variants() {
        assert_eq!(
            user_search_queries("خالد"),
            vec!["خالد".to_string(), "khaled".to_string(), "khalid".to_string()]
        );
        let queries = user_search_queries("محمد");
        assert_eq!(queries[0], "محمد");
        assert!(queries.contains(&"mohammed".to_string()));
        assert!(queries.contains(&"mohamed".to_string()));
        let hamza = user_search_queries("أحمد");
        assert!(hamza.contains(&"ahmed".to_string()));
    }

    #[test]
    fn user_search_queries_latin_passthrough() {
        assert_eq!(user_search_queries("khaled"), vec!["khaled".to_string()]);
        assert_eq!(user_search_queries("alialahmed"), vec!["alialahmed".to_string()]);
        assert_eq!(user_search_queries(""), vec!["".to_string()]);
    }

    #[test]
    fn user_search_queries_unknown_arabic_uses_generic_mapping() {
        let queries = user_search_queries("غانم");
        assert_eq!(queries[0], "غانم");
        assert!(queries.contains(&"ghanm".to_string()));
    }

    #[test]
    fn contains_arabic_detects_script() {
        assert!(contains_arabic("خالد"));
        assert!(contains_arabic("abc خالد def"));
        assert!(!contains_arabic("khaled"));
        assert!(!contains_arabic(""));
    }

    fn test_row(id: u64, username: &str, name: &str) -> UserSearchRow {
        UserSearchRow {
            id,
            username: Some(username.to_string()),
            name: Some(name.to_string()),
            bio: None,
            is_private: None,
            is_verified: None,
            is_premium: None,
            streak: None,
        }
    }

    #[test]
    fn rank_user_rows_prefers_exact_username_then_display() {
        let rows = vec![
            (test_row(4, "khaled", ""), "khaled".to_string()),
            (test_row(262, "KHA", "Khalid"), "khalid".to_string()),
            (test_row(309, "Khalid", "Dr. Khalid"), "khalid".to_string()),
            (test_row(5158, "5ld", "Khaled"), "khaled".to_string()),
            (test_row(3185, "khalidm", "Khalid"), "khalid".to_string()),
        ];
        let ranked = rank_user_rows("خالد", rows);
        assert_eq!(ranked[0].0.id, 4);
        assert!(ranked[0].2, "khaled row must be an exact username match");
        assert_eq!(ranked[1].0.id, 309, "exact username 'Khalid' ranks second");
        assert!(ranked[1].2);
        assert_eq!(ranked[2].0.id, 262, "exact display name after username matches");
        assert!(!ranked[2].2);
        assert_eq!(ranked[3].0.id, 5158);
        assert_eq!(ranked[4].0.id, 3185);
    }

    #[test]
    fn rank_user_rows_single_winner_for_latin_query() {
        let rows = vec![
            (test_row(4, "khaled", ""), "khaled".to_string()),
            (test_row(813, "thekhaled", ":)"), "khaled".to_string()),
        ];
        let ranked = rank_user_rows("khaled", rows);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0.id, 4);
        assert_eq!(ranked[0].1, 0);
        assert_eq!(ranked[1].0.id, 813);
        assert_eq!(ranked[1].1, 4, "substring match ranks below exact");
    }

    #[test]
    fn rank_user_rows_display_name_exact_wins_over_prefix() {
        let rows = vec![
            (test_row(262, "KHA", "Khalid"), "Khalid".to_string()),
            (test_row(3185, "khalidm", "Khalid"), "Khalid".to_string()),
        ];
        let ranked = rank_user_rows("Khalid", rows);
        assert_eq!(ranked[0].0.id, 262);
        assert!(!ranked[0].2);
        assert_eq!(ranked[0].1, 1);
        assert_eq!(ranked[1].0.id, 3185);
    }

    #[test]
    fn profile_scan_text_joins_skips_empty_and_caps() {
        let text = profile_scan_text("Fahad", &["post one", "", "post two"]);
        assert!(text.starts_with("Recent posts by Fahad:"));
        assert!(text.contains("post one"));
        assert!(text.contains("post two"));
        assert!(text.contains("\n---\n"));
        assert!(!text.contains("\n---\n---\n"), "empty contents are skipped");
        assert_eq!(profile_scan_text("X", &["", "  "]), "");

        let long = "ل".repeat(PROFILE_SCAN_TEXT_CAP + 100);
        let capped = profile_scan_text("X", &[&long]);
        assert_eq!(capped.chars().count(), PROFILE_SCAN_TEXT_CAP);
    }

    #[test]
    fn function_call_part_roundtrips_through_serde() {
        let call = Part::FunctionCall {
            function_call: FunctionCallData {
                name: "web_fetch".to_string(),
                args: json!({ "url": "https://x.test" }),
                id: None,
            },
            thought_signature: Some("sig-abc".to_string()),
        };
        let value = serde_json::to_value(&call).unwrap();
        assert_eq!(value["functionCall"]["name"], "web_fetch");
        assert_eq!(value["functionCall"]["args"]["url"], "https://x.test");
        assert_eq!(value["thoughtSignature"], "sig-abc");

        let response_part = Part::FunctionResponse {
            function_response: FunctionResponseData {
                name: "web_fetch".to_string(),
                response: json!({ "text": "hi" }),
                id: None,
            },
        };
        let value = serde_json::to_value(&response_part).unwrap();
        assert_eq!(value["functionResponse"]["name"], "web_fetch");
        assert_eq!(value["functionResponse"]["response"]["text"], "hi");
    }
}
