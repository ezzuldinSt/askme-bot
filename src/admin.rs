//! Admin panel: a small axum server (default 0.0.0.0:1330) exposing the bot's
//! configuration, live logs, memory stats and danger-zone operations behind a
//! password-protected session.
//!
//! Security model: argon2-hashed password in `bot_config.json`; opaque session
//! tokens (in-memory, 12h expiry) delivered as HttpOnly SameSite=Strict
//! cookies; login rate-limiting; a forced password-change gate while the
//! default password is still in effect. NOTE: plain HTTP — trusted LAN only.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::extract::{FromRequestParts, Query, State};
use axum::http::header;
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use tracing_subscriber::layer::{Context as LayerContext, Layer};
use uuid::Uuid;

use crate::config::{self, BotConfig, RuntimeConfig, RESTART_REQUIRED_FIELDS};
use crate::things_client::is_auth_expired;
use crate::AppState;


const INDEX_HTML: &str = include_str!("admin_ui.html");
const SESSION_COOKIE: &str = "askme_session";
const SESSION_TTL_SECS: u64 = 12 * 3600;
const LOG_BUFFER_CAPACITY: usize = 1000;
const LOGIN_MAX_FAILS: u32 = 5;
const LOGIN_LOCKOUT_SECS: u64 = 60;
const THINGS_STATUS_CACHE_TTL: Duration = Duration::from_secs(15);

// ── Live log ring buffer ──

#[derive(Debug, Clone, serde::Serialize)]
pub struct LogEvent {
    pub seq: u64,
    pub ts: String,
    pub level: String,
    pub target: String,
    pub text: String,
}

#[derive(Debug, Default)]
pub struct LogBuffer {
    inner: Mutex<VecDeque<LogEvent>>,
    next_seq: AtomicU64,
}

impl LogBuffer {
    fn push(&self, mut event: LogEvent) {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed) + 1;
        event.seq = seq;
        let mut buf = self.inner.lock().unwrap();
        if buf.len() >= LOG_BUFFER_CAPACITY {
            buf.pop_front();
        }
        buf.push_back(event);
    }

    /// Events with `seq > after`, plus the current high-water mark.
    fn since(&self, after: u64) -> (Vec<LogEvent>, u64) {
        let buf = self.inner.lock().unwrap();
        let events = buf.iter().filter(|e| e.seq > after).cloned().collect();
        (events, self.next_seq.load(Ordering::Relaxed))
    }
}

pub type SharedLogBuffer = Arc<LogBuffer>;

pub fn new_log_buffer() -> SharedLogBuffer {
    Arc::new(LogBuffer::default())
}

/// A tracing layer mirroring formatted log lines into the ring buffer (the
/// fmt layer keeps writing to stdout for journald).
pub struct LogLayer {
    buf: SharedLogBuffer,
}

impl LogLayer {
    pub fn new(buf: SharedLogBuffer) -> Self {
        Self { buf }
    }
}

#[derive(Default)]
struct MessageVisitor {
    message: String,
    fields: Vec<String>,
}

impl tracing::field::Visit for MessageVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else {
            self.fields.push(format!("{}={value}", field.name()));
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        } else {
            self.fields.push(format!("{}={value:?}", field.name()));
        }
    }
}

impl<S: tracing::Subscriber> Layer<S> for LogLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: LayerContext<'_, S>) {
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        let text = if visitor.message.is_empty() {
            visitor.fields.join(" ")
        } else {
            visitor.message
        };
        self.buf.push(LogEvent {
            seq: 0,
            ts: chrono::Local::now().format("%H:%M:%S%.3f").to_string(),
            level: event.metadata().level().to_string(),
            target: event.metadata().target().to_string(),
            text,
        });
    }
}

// ── Shared server state ──

#[derive(Debug, Default)]
struct Throttle {
    fails: u32,
    locked_until: Option<Instant>,
}

#[derive(Clone)]
pub struct AdminState {
    app: Arc<RwLock<AppState>>,
    bot_config: Arc<RwLock<BotConfig>>,
    sessions: Arc<RwLock<HashMap<String, Instant>>>,
    throttle: Arc<Mutex<Throttle>>,
    logs: SharedLogBuffer,
    started_at: Instant,
    needs_restart: Arc<AtomicBool>,
    things_auth_cache: Arc<Mutex<Option<(Instant, bool)>>>,
    /// Serializes FAQ read-modify-write cycles against concurrent panel clicks.
    faqs_lock: Arc<tokio::sync::Mutex<()>>,
}

impl AdminState {
    pub fn new(
        app: Arc<RwLock<AppState>>,
        bot_config: Arc<RwLock<BotConfig>>,
        logs: SharedLogBuffer,
    ) -> Self {
        Self {
            app,
            bot_config,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            throttle: Arc::new(Mutex::new(Throttle::default())),
            logs,
            started_at: Instant::now(),
            needs_restart: Arc::new(AtomicBool::new(false)),
            things_auth_cache: Arc::new(Mutex::new(None)),
            faqs_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }
}

pub async fn serve(listener: TcpListener, state: AdminState) -> Result<()> {
    axum::serve(listener, router(state))
        .await
        .context("Admin panel server error")
}

pub fn router(state: AdminState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/login", post(login))
        .route("/api/logout", post(logout))
        .route("/api/password", post(change_password))
        .route("/api/status", get(status))
        .route("/api/config", get(get_config).put(put_config))
        .route("/api/faqs", get(list_faqs).post(add_faq).delete(remove_faq))
        .route("/api/logs", get(get_logs))
        .route("/api/wipe", post(wipe))
        .route("/api/restart", post(restart))
        .with_state(state)
}

// ── Auth ──

type ApiResult<T> = Result<T, (StatusCode, Json<Value>)>;

fn err<T>(status: StatusCode, message: &str) -> ApiResult<T> {
    Err((status, Json(json!({ "error": message }))))
}

fn unauthorized() -> Response {
    (StatusCode::UNAUTHORIZED, Json(json!({ "error": "unauthorized" }))).into_response()
}

/// Session-cookie guard for every authenticated endpoint.
pub struct Auth;

impl FromRequestParts<AdminState> for Auth {
    type Rejection = Response;

    async fn from_request_parts(parts: &mut Parts, state: &AdminState) -> Result<Self, Self::Rejection> {
        let token = parts
            .headers
            .get(header::COOKIE)
            .and_then(|v| v.to_str().ok())
            .and_then(|cookies| {
                cookies
                    .split(';')
                    .map(str::trim)
                    .find_map(|c| c.strip_prefix("askme_session="))
                    .map(|s| s.to_string())
            });
        let Some(token) = token else {
            return Err(unauthorized());
        };
        let sessions = state.sessions.read().await;
        match sessions.get(&token) {
            Some(expiry) if *expiry > Instant::now() => Ok(Auth),
            _ => Err(unauthorized()),
        }
    }
}

/// While the default password is still in effect, every endpoint except
/// login/logout/password is blocked.
async fn gate(st: &AdminState) -> ApiResult<()> {
    if st.bot_config.read().await.admin.must_change {
        err(StatusCode::FORBIDDEN, "password change required")
    } else {
        Ok(())
    }
}

// ── Handlers ──

fn mask_key_for_view(key: &str) -> String {
    let tail: String = key.chars().rev().take(4).collect::<String>().chars().rev().collect();
    format!("••••{tail}")
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

#[derive(Deserialize)]
struct LoginRequest {
    password: String,
}

async fn login(State(st): State<AdminState>, Json(req): Json<LoginRequest>) -> ApiResult<Response> {
    {
        let throttle = st.throttle.lock().unwrap();
        if let Some(until) = throttle.locked_until {
            if until > Instant::now() {
                return err(StatusCode::TOO_MANY_REQUESTS, "too many attempts; try again in a minute");
            }
        }
    }

    let cfg = st.bot_config.read().await;
    if !config::verify_password(&req.password, &cfg.admin.password_hash) {
        let mut throttle = st.throttle.lock().unwrap();
        throttle.fails += 1;
        if throttle.fails >= LOGIN_MAX_FAILS {
            throttle.locked_until = Some(Instant::now() + Duration::from_secs(LOGIN_LOCKOUT_SECS));
            throttle.fails = 0;
        }
        warn!("Admin panel: failed login attempt");
        return err(StatusCode::UNAUTHORIZED, "invalid password");
    }
    st.throttle.lock().unwrap().fails = 0;

    let token = Uuid::new_v4().to_string();
    let expiry = Instant::now() + Duration::from_secs(SESSION_TTL_SECS);
    {
        let mut sessions = st.sessions.write().await;
        sessions.retain(|_, exp| *exp > Instant::now());
        sessions.insert(token.clone(), expiry);
    }
    info!("Admin panel: login");
    Ok((
        [(
            header::SET_COOKIE,
            format!("{SESSION_COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_TTL_SECS}"),
        )],
        Json(json!({ "ok": true, "must_change": cfg.admin.must_change })),
    )
        .into_response())
}

async fn logout(_: Auth, State(_st): State<AdminState>) -> Response {
    (
        [(
            header::SET_COOKIE,
            format!("{SESSION_COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"),
        )],
        Json(json!({ "ok": true })),
    )
        .into_response()
}

#[derive(Deserialize)]
struct PasswordUpdate {
    current: String,
    new: String,
}

async fn change_password(
    _: Auth,
    State(st): State<AdminState>,
    Json(req): Json<PasswordUpdate>,
) -> ApiResult<Json<Value>> {
    if req.new.chars().count() < 8 {
        return err(StatusCode::BAD_REQUEST, "new password must be at least 8 characters");
    }
    if req.new == config::DEFAULT_ADMIN_PASSWORD {
        return err(StatusCode::BAD_REQUEST, "choose a real password, not the default");
    }
    let mut cfg = st.bot_config.write().await;
    if !config::verify_password(&req.current, &cfg.admin.password_hash) {
        return err(StatusCode::UNAUTHORIZED, "current password is wrong");
    }
    cfg.admin.password_hash = config::hash_password(&req.new)
        .map_err(|e| {
            error!("Failed to hash new admin password: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "hash failed" })))
        })?;
    cfg.admin.must_change = false;
    config::save(&cfg).map_err(|e| {
        error!("Failed to save config: {e}");
        (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "save failed" })))
    })?;
    info!("Admin panel: password changed");
    Ok(Json(json!({ "ok": true })))
}

async fn status(_: Auth, State(st): State<AdminState>) -> ApiResult<Json<Value>> {
    gate(&st).await?;
    let (qdrant, runtime, gemini, things_auth) = {
        let app = st.app.read().await;
        (
            app.qdrant.clone(),
            app.runtime.clone(),
            app.gemini.clone(),
            check_things_auth(&st).await,
        )
    };
    let runtime = runtime.read().await;
    let collections = qdrant.collection_stats().await;
    Ok(Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "bot_username": crate::BOT_USERNAME,
        "uptime_secs": st.started_at.elapsed().as_secs(),
        "needs_restart": st.needs_restart.load(Ordering::Relaxed),
        "qdrant": {
            "available": qdrant.is_available(),
            "collections": collections,
        },
        "things": {
            "auth_ok": things_auth,
            "email": std::env::var("THINGS_EMAIL").unwrap_or_default(),
        },
        "gemini_pool": gemini.pool_status(),
        "config": {
            "generation_model": gemini.generation_model(),
            "extraction_model": gemini.extraction_model(),
            "thinking_level": gemini.thinking_level(),
            "extraction_thinking_level": gemini.extraction_thinking_level(),
            "user_facts_limit": runtime.memory.user_facts_limit,
            "app_knowledge_limit": runtime.memory.app_knowledge_limit,
            "app_knowledge_min_score": runtime.memory.app_knowledge_min_score,
            "user_fact_supersede_threshold": runtime.memory.user_fact_supersede_threshold,
            "forget_similarity_threshold": runtime.memory.forget_similarity_threshold,
            "fact_extraction_enabled": runtime.memory.fact_extraction_enabled,
            "context_depth_limit": runtime.context_depth_limit,
        },
    })))
}

/// Live Things auth probe with a short cache so the dashboard doesn't hammer
/// the API. None = unknown (network error rather than an auth failure).
async fn check_things_auth(st: &AdminState) -> Option<bool> {
    {
        let cache = st.things_auth_cache.lock().unwrap();
        if let Some((ts, ok)) = *cache {
            if ts.elapsed() < THINGS_STATUS_CACHE_TTL {
                return Some(ok);
            }
        }
    }
    let ok = {
        let app = st.app.read().await;
        match app.things.get_unread_count().await {
            Ok(_) => true,
            Err(e) => {
                if is_auth_expired(&e) {
                    false
                } else {
                    return None;
                }
            }
        }
    };
    *st.things_auth_cache.lock().unwrap() = Some((Instant::now(), ok));
    Some(ok)
}

async fn get_config(_: Auth, State(st): State<AdminState>) -> ApiResult<Json<Value>> {
    gate(&st).await?;
    let cfg = st.bot_config.read().await;
    let o = &cfg.overrides;
    let resolved = RuntimeConfig::resolve(o);
    let m = &resolved.memory;
    let fallback_generation_model = st
        .app
        .read()
        .await
        .gemini
        .fallback_generation_model();
    Ok(Json(json!({
        "hot": {
            "user_facts_limit": m.user_facts_limit,
            "app_knowledge_limit": m.app_knowledge_limit,
            "app_knowledge_min_score": m.app_knowledge_min_score,
            "user_fact_supersede_threshold": m.user_fact_supersede_threshold,
            "forget_similarity_threshold": m.forget_similarity_threshold,
            "fact_extraction_enabled": m.fact_extraction_enabled,
            "extraction_min_chars": m.extraction_min_chars,
            "context_depth_limit": resolved.context_depth_limit,
            "generation_model": config::resolve_generation_model(o),
            // Raw override ("" when unset) + the live client value for display.
            "fallback_generation_model": fallback_generation_model,
            // Raw override ("" when unset) for the form + the effective value
            // for display: unset means "same as the chat model".
            "extraction_model": o.extraction_model.clone().unwrap_or_default(),
            "extraction_model_effective": config::resolve_extraction_model(o)
                .unwrap_or_else(|| config::resolve_generation_model(o)),
            "thinking_level": config::resolve_thinking_level(o),
            // Raw override ("" when unset) for the form + the effective value
            // for display: unset means the built-in default ("low").
            "extraction_thinking_level": o.extraction_thinking_level.clone().unwrap_or_default(),
            "extraction_thinking_level_effective": config::resolve_extraction_thinking_level(o)
                .unwrap_or_default(),
            "tools": {
                "enabled": resolved.tools.enabled,
                "max_rounds": resolved.tools.max_rounds,
                "max_flow_attempts": resolved.tools.max_flow_attempts,
                "web_fetch_max_bytes": resolved.tools.web_fetch_max_bytes,
                "web_fetch_timeout_secs": resolved.tools.web_fetch_timeout_secs,
                "user_scan_posts_limit": resolved.tools.user_scan_posts_limit,
                "user_scan_fact_cap": resolved.tools.user_scan_fact_cap,
                "url_context_enabled": resolved.tools.url_context_enabled,
                "search_grounding_enabled": resolved.tools.search_grounding_enabled,
                "games_enabled": resolved.tools.games_enabled,
            },
        },
        "overridden": {
            "user_facts_limit": o.user_facts_limit.is_some(),
            "app_knowledge_limit": o.app_knowledge_limit.is_some(),
            "app_knowledge_min_score": o.app_knowledge_min_score.is_some(),
            "user_fact_supersede_threshold": o.user_fact_supersede_threshold.is_some(),
            "forget_similarity_threshold": o.forget_similarity_threshold.is_some(),
            "fact_extraction_enabled": o.fact_extraction_enabled.is_some(),
            "context_depth_limit": o.context_depth_limit.is_some(),
            "gemini_api_keys": !o.gemini_api_keys.is_empty(),
            "generation_model": o.generation_model.is_some(),
            "fallback_generation_model": o.fallback_generation_model.is_some(),
            "extraction_model": o.extraction_model.is_some(),
            "thinking_level": o.thinking_level.is_some(),
            "qdrant_url": o.qdrant_url.is_some(),
            "embedding_model": o.embedding_model.is_some(),
            "embedding_dimensions": o.embedding_dimensions.is_some(),
        },
        "restart": {
            "gemini_keys_count": config::resolve_gemini_keys(o).len(),
            "gemini_keys_masked": config::resolve_gemini_keys(o).iter().map(|k| mask_key_for_view(k)).collect::<Vec<_>>(),
            "qdrant_url": config::resolve_qdrant_url(o),
            "embedding_model": config::resolve_embedding_model(o),
            "embedding_dimensions": config::resolve_embedding_dimensions(o),
        },
        "restart_required_fields": RESTART_REQUIRED_FIELDS,
        "needs_restart": st.needs_restart.load(Ordering::Relaxed),
        "things_email": std::env::var("THINGS_EMAIL").unwrap_or_default(),
        "admin": { "bind": cfg.admin.bind, "port": cfg.admin.port },
    })))
}

#[derive(Deserialize)]
struct ConfigUpdate {
    // Hot-applied (always concrete values from the form).
    user_facts_limit: u64,
    app_knowledge_limit: u64,
    app_knowledge_min_score: f32,
    user_fact_supersede_threshold: f32,
    forget_similarity_threshold: f32,
    fact_extraction_enabled: bool,
    extraction_min_chars: usize,
    context_depth_limit: usize,
    // Tool calling (hot-applied, always concrete values from the form).
    tools_enabled: bool,
    max_tool_rounds: usize,
    max_flow_attempts: usize,
    web_fetch_max_bytes: usize,
    web_fetch_timeout_secs: u64,
    user_scan_posts_limit: u64,
    user_scan_fact_cap: usize,
    url_context_enabled: bool,
    search_grounding_enabled: bool,
    games_enabled: bool,
    // Hot-applied (empty/None = leave unchanged).
    gemini_api_keys: Option<Vec<String>>,
    /// Appended to the current pool (deduped). Ignored when gemini_api_keys
    /// (the wholesale replace) is non-empty — replace wins.
    gemini_api_keys_add: Option<Vec<String>>,
    generation_model: Option<String>,
    /// Saturation fallback for the chat model: empty = disabled, value =
    /// set. Hot-applied, no restart.
    fallback_generation_model: Option<String>,
    /// Extraction model: empty = "same as chat model" (clears the override).
    extraction_model: Option<String>,
    /// "default" or one of THINKING_LEVELS (empty/None = leave unchanged).
    thinking_level: Option<String>,
    /// "default" clears the override (back to the built-in "low"); one of
    /// THINKING_LEVELS sets it (empty/None = leave unchanged).
    extraction_thinking_level: Option<String>,
    // Restart-required (empty/None = leave unchanged).
    qdrant_url: Option<String>,
    embedding_model: Option<String>,
    embedding_dimensions: Option<u64>,
}

/// Map a dropdown value to the stored override: "default" -> None (model
/// default), a valid level -> Some(level), anything else -> Err.
fn parse_thinking_level(raw: &str) -> Result<Option<String>, String> {
    if raw == "default" {
        return Ok(None);
    }
    if config::THINKING_LEVELS.contains(&raw) {
        return Ok(Some(raw.to_string()));
    }
    Err(format!(
        "thinking level must be one of: default, {}",
        config::THINKING_LEVELS.join(", ")
    ))
}

/// True when the EFFECTIVE value of any restart-required field changed
/// (comparing resolved values, not override presence). This is the fix for
/// the phantom "restart required" after saving an unrelated section.
fn restart_fields_changed(old: &config::ConfigOverrides, new: &config::ConfigOverrides) -> bool {
    config::resolve_qdrant_url(old) != config::resolve_qdrant_url(new)
        || config::resolve_embedding_model(old) != config::resolve_embedding_model(new)
        || config::resolve_embedding_dimensions(old) != config::resolve_embedding_dimensions(new)
}

/// Apply one restart-required override field. Empty/None input = unchanged.
/// A value equal to the env/default fallback is normalized to `None` so the
/// config file never accumulates redundant overrides.
fn apply_restart_override(
    slot: &mut Option<String>,
    incoming: Option<String>,
    env_key: &str,
    default: &str,
) {
    let Some(v) = incoming else { return };
    let v = v.trim().to_string();
    if v.is_empty() {
        return;
    }
    let fallback = std::env::var(env_key).unwrap_or_else(|_| default.to_string());
    *slot = if v == fallback { None } else { Some(v) };
}

fn apply_restart_override_u64(slot: &mut Option<u64>, incoming: Option<u64>, env_key: &str, default: u64) {
    let Some(v) = incoming else { return };
    if v == 0 {
        return;
    }
    let fallback = std::env::var(env_key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default);
    *slot = if v == fallback { None } else { Some(v) };
}

/// Merge `adds` into `pool`, skipping exact duplicates and preserving order.
/// Returns the merged pool and how many keys were actually new.
fn merge_api_keys(pool: &[String], adds: &[String]) -> (Vec<String>, usize) {
    let mut pool = pool.to_vec();
    let before = pool.len();
    for k in adds {
        if !pool.contains(k) {
            pool.push(k.clone());
        }
    }
    let added = pool.len() - before;
    (pool, added)
}

async fn put_config(
    _: Auth,
    State(st): State<AdminState>,
    Json(req): Json<ConfigUpdate>,
) -> ApiResult<Json<Value>> {
    gate(&st).await?;

    for (name, v) in [
        ("app_knowledge_min_score", req.app_knowledge_min_score),
        ("user_fact_supersede_threshold", req.user_fact_supersede_threshold),
        ("forget_similarity_threshold", req.forget_similarity_threshold),
    ] {
        if !(0.0..=1.0).contains(&v) {
            return err(StatusCode::BAD_REQUEST, &format!("{name} must be between 0 and 1"));
        }
    }
    if !(1..=100).contains(&req.user_facts_limit) {
        return err(StatusCode::BAD_REQUEST, "user_facts_limit must be 1-100");
    }
    if !(1..=100).contains(&req.app_knowledge_limit) {
        return err(StatusCode::BAD_REQUEST, "app_knowledge_limit must be 1-100");
    }
    if !(1..=200).contains(&req.context_depth_limit) {
        return err(StatusCode::BAD_REQUEST, "context_depth_limit must be 1-200");
    }
    if !(1..=8).contains(&req.max_tool_rounds) {
        return err(StatusCode::BAD_REQUEST, "max_tool_rounds must be 1-8");
    }
    if !(1..=8).contains(&req.max_flow_attempts) {
        return err(StatusCode::BAD_REQUEST, "max_flow_attempts must be 1-8");
    }
    if req.extraction_min_chars > 500 {
        return err(StatusCode::BAD_REQUEST, "extraction_min_chars must be 0-500");
    }
    if !(16_384..=2_000_000).contains(&req.web_fetch_max_bytes) {
        return err(StatusCode::BAD_REQUEST, "web_fetch_max_bytes must be 16384-2000000");
    }
    if !(3..=60).contains(&req.web_fetch_timeout_secs) {
        return err(StatusCode::BAD_REQUEST, "web_fetch_timeout_secs must be 3-60");
    }
    if !(1..=30).contains(&req.user_scan_posts_limit) {
        return err(StatusCode::BAD_REQUEST, "user_scan_posts_limit must be 1-30");
    }

    let mut cfg = st.bot_config.write().await;
    let old = cfg.overrides.clone();
    let mut o = old.clone();
    o.user_facts_limit = Some(req.user_facts_limit);
    o.app_knowledge_limit = Some(req.app_knowledge_limit);
    o.app_knowledge_min_score = Some(req.app_knowledge_min_score);
    o.user_fact_supersede_threshold = Some(req.user_fact_supersede_threshold);
    o.forget_similarity_threshold = Some(req.forget_similarity_threshold);
    o.fact_extraction_enabled = Some(req.fact_extraction_enabled);
    o.extraction_min_chars = Some(req.extraction_min_chars);
    o.context_depth_limit = Some(req.context_depth_limit);
    o.tools_enabled = Some(req.tools_enabled);
    o.max_tool_rounds = Some(req.max_tool_rounds);
    o.max_flow_attempts = Some(req.max_flow_attempts);
    o.web_fetch_max_bytes = Some(req.web_fetch_max_bytes);
    o.web_fetch_timeout_secs = Some(req.web_fetch_timeout_secs);
    o.user_scan_posts_limit = Some(req.user_scan_posts_limit);
    o.user_scan_fact_cap = Some(req.user_scan_fact_cap);
    o.url_context_enabled = Some(req.url_context_enabled);
    o.search_grounding_enabled = Some(req.search_grounding_enabled);
    o.games_enabled = Some(req.games_enabled);

    // Key pool: empty/None = unchanged; otherwise replace wholesale (min 1).
    // Replace wins over append when both are filled.
    let new_keys = match req.gemini_api_keys {
        Some(keys) => {
            let cleaned: Vec<String> = keys
                .iter()
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty())
                .collect();
            if cleaned.is_empty() {
                None
            } else {
                if cleaned.iter().any(|k| k.chars().any(char::is_whitespace)) {
                    return err(StatusCode::BAD_REQUEST, "API keys must not contain whitespace");
                }
                Some(cleaned)
            }
        }
        None => None,
    };
    // Append path: merge new keys into the current pool (deduped). The pool
    // may live in env vars until the first panel edit — resolve materializes
    // it into the file.
    let mut keys_added: Option<(usize, usize)> = None; // (newly added, pool total)
    let new_keys = match new_keys {
        some @ Some(_) => some,
        None => {
            let adds: Vec<String> = req
                .gemini_api_keys_add
                .unwrap_or_default()
                .iter()
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty())
                .collect();
            if adds.is_empty() {
                None
            } else {
                if adds.iter().any(|k| k.chars().any(char::is_whitespace)) {
                    return err(StatusCode::BAD_REQUEST, "API keys must not contain whitespace");
                }
                let (pool, added) = merge_api_keys(&config::resolve_gemini_keys(&o), &adds);
                keys_added = Some((added, pool.len()));
                Some(pool)
            }
        }
    };
    if let Some(keys) = &new_keys {
        o.gemini_api_keys = keys.clone();
        o.gemini_api_key = None; // fold legacy single-key field into the pool
    }

    // Generation model: hot-applied, no restart.
    let new_generation_model = match req.generation_model {
        Some(m) => {
            let m = m.trim().to_string();
            if m.is_empty() {
                None
            } else {
                if m.chars().any(char::is_whitespace) {
                    return err(StatusCode::BAD_REQUEST, "generation model must not contain whitespace");
                }
                Some(m)
            }
        }
        None => None,
    };
    if let Some(m) = &new_generation_model {
        o.generation_model = Some(m.clone());
    }

    // Fallback model: empty string = disabled (clears the override); a
    // value sets it. Same shape as the extraction-model handling.
    let new_fallback_generation_model = match req.fallback_generation_model {
        Some(m) => {
            let m = m.trim().to_string();
            if m.chars().any(char::is_whitespace) {
                return err(StatusCode::BAD_REQUEST, "fallback model must not contain whitespace");
            }
            o.fallback_generation_model = (!m.is_empty()).then_some(m.clone());
            Some(o.fallback_generation_model.clone())
        }
        None => None,
    };

    // Extraction model: empty string = "same as chat model" (clears the
    // override); a value sets it. Hot-applied, no restart.
    let new_extraction_model = match req.extraction_model {
        Some(m) => {
            let m = m.trim().to_string();
            if m.chars().any(char::is_whitespace) {
                return err(StatusCode::BAD_REQUEST, "extraction model must not contain whitespace");
            }
            o.extraction_model = if m.is_empty() { None } else { Some(m) };
            Some(o.extraction_model.clone())
        }
        None => None,
    };

    // Thinking level: "default" clears the override (model default); one of
    // THINKING_LEVELS sets it; anything else is rejected. Hot-applied.
    let new_thinking_level = match req.thinking_level {
        Some(raw) => {
            let raw = raw.trim();
            if raw.is_empty() {
                None
            } else {
                match parse_thinking_level(raw) {
                    Ok(level) => {
                        o.thinking_level = level.clone();
                        Some(level)
                    }
                    Err(msg) => return err(StatusCode::BAD_REQUEST, &msg),
                }
            }
        }
        None => None,
    };

    // Extraction thinking level: "default" clears the override (back to the
    // built-in "low"); one of THINKING_LEVELS sets it. Hot-applied.
    let new_extraction_thinking_level = match req.extraction_thinking_level {
        Some(raw) => {
            let raw = raw.trim();
            if raw.is_empty() {
                None
            } else {
                match parse_thinking_level(raw) {
                    Ok(level) => {
                        o.extraction_thinking_level = level.clone();
                        Some(level)
                    }
                    Err(msg) => return err(StatusCode::BAD_REQUEST, &msg),
                }
            }
        }
        None => None,
    };

    if let Some(u) = req.qdrant_url.as_ref().filter(|s| !s.trim().is_empty()) {
        if !u.trim().starts_with("http") {
            return err(StatusCode::BAD_REQUEST, "qdrant_url must start with http");
        }
    }
    apply_restart_override(&mut o.qdrant_url, req.qdrant_url, "QDRANT_URL", config::DEFAULT_QDRANT_URL);
    apply_restart_override(
        &mut o.embedding_model,
        req.embedding_model,
        "EMBEDDING_MODEL",
        config::DEFAULT_EMBEDDING_MODEL,
    );
    if let Some(d) = req.embedding_dimensions.filter(|d| *d > 0) {
        if !(64..=3072).contains(&d) {
            return err(StatusCode::BAD_REQUEST, "embedding_dimensions must be 64-3072");
        }
    }
    apply_restart_override_u64(
        &mut o.embedding_dimensions,
        req.embedding_dimensions,
        "EMBEDDING_DIMENSIONS",
        config::DEFAULT_EMBEDDING_DIMENSIONS as u64,
    );

    let restart_changed = restart_fields_changed(&old, &o);

    cfg.overrides = o;
    if let Err(e) = config::save(&cfg) {
        error!("Failed to save config: {e}");
        return err(StatusCode::INTERNAL_SERVER_ERROR, "failed to save config");
    }
    let new_overrides = cfg.overrides.clone();
    drop(cfg);

    // Hot-apply the runtime knobs.
    let (runtime, gemini) = {
        let app = st.app.read().await;
        (app.runtime.clone(), app.gemini.clone())
    };
    *runtime.write().await = RuntimeConfig::resolve(&new_overrides);
    // Hot-apply the key pool and the generation model (no restart needed).
    if let Some(keys) = new_keys {
        gemini.set_keys(keys);
        info!("Gemini key pool hot-swapped via admin panel");
    }
    if let Some(m) = new_generation_model {
        gemini.set_generation_model(m);
    }
    if let Some(m) = new_fallback_generation_model {
        gemini.set_fallback_generation_model(m);
    }
    if let Some(m) = new_extraction_model {
        gemini.set_extraction_model(m);
    }
    if let Some(level) = new_thinking_level {
        gemini.set_thinking_level(level);
    }
    if let Some(level) = new_extraction_thinking_level {
        gemini.set_extraction_thinking_level(level);
    }
    if restart_changed {
        st.needs_restart.store(true, Ordering::Relaxed);
    }
    info!("Configuration updated via admin panel (restart_required={restart_changed})");
    Ok(Json(json!({
        "ok": true,
        "restart_required": restart_changed,
        "keys_added": keys_added.map(|(a, _)| a),
        "keys_total": keys_added.map(|(_, t)| t),
    })))
}

// ── Support FAQs ──

fn faq_json(f: &crate::faqs::SupportFaq) -> Value {
    json!({
        "id": f.id,
        "question": f.question,
        "answer": f.answer,
        "facts": f.facts,
        "created_at": f.created_at,
    })
}

async fn list_faqs(_: Auth, State(st): State<AdminState>) -> ApiResult<Json<Value>> {
    gate(&st).await?;
    let qdrant_available = st.app.read().await.qdrant.is_available();
    let faqs = crate::faqs::load();
    Ok(Json(json!({
        "faqs": faqs.iter().map(faq_json).collect::<Vec<_>>(),
        "qdrant_available": qdrant_available,
    })))
}

#[derive(Deserialize)]
struct FaqCreate {
    question: String,
    answer: String,
}

async fn add_faq(
    _: Auth,
    State(st): State<AdminState>,
    Json(req): Json<FaqCreate>,
) -> ApiResult<Json<Value>> {
    gate(&st).await?;
    let question = req.question.trim();
    let answer = req.answer.trim();
    if question.chars().count() < 3 || answer.chars().count() < 3 {
        return err(
            StatusCode::BAD_REQUEST,
            "question and answer must both be at least 3 characters",
        );
    }
    if question.chars().count() > 1000 || answer.chars().count() > 4000 {
        return err(
            StatusCode::BAD_REQUEST,
            "question is capped at 1000 chars, answer at 4000",
        );
    }
    let (qdrant, gemini) = {
        let app = st.app.read().await;
        (app.qdrant.clone(), app.gemini.clone())
    };
    let _guard = st.faqs_lock.lock().await;
    let (faq, synced) = crate::faqs::insert_faq(&qdrant, &gemini, question, answer)
        .await
        .map_err(|e| {
            error!("FAQ insert failed: {e}");
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("fact extraction failed: {e}") })),
            )
        })?;
    info!("Support FAQ added via admin panel ({} facts)", faq.facts.len());
    Ok(Json(json!({ "ok": true, "synced": synced, "faq": faq_json(&faq) })))
}

#[derive(Deserialize)]
struct FaqDelete {
    id: String,
}

async fn remove_faq(
    _: Auth,
    State(st): State<AdminState>,
    Query(q): Query<FaqDelete>,
) -> ApiResult<Json<Value>> {
    gate(&st).await?;
    let id = match Uuid::parse_str(q.id.trim()) {
        Ok(id) => id,
        Err(_) => return err(StatusCode::BAD_REQUEST, "invalid FAQ id"),
    };
    let qdrant = st.app.read().await.qdrant.clone();
    let _guard = st.faqs_lock.lock().await;
    match crate::faqs::delete_faq(&qdrant, id).await {
        Ok(Some(faq)) => {
            info!("Support FAQ {id} deleted via admin panel");
            Ok(Json(json!({ "ok": true, "deleted": faq_json(&faq) })))
        }
        Ok(None) => err(StatusCode::NOT_FOUND, "FAQ not found"),
        Err(e) => {
            error!("FAQ delete failed: {e}");
            err(StatusCode::INTERNAL_SERVER_ERROR, "failed to delete FAQ")
        }
    }
}

#[derive(Deserialize)]
struct LogsQuery {
    after: Option<u64>,
}

async fn get_logs(
    _: Auth,
    State(st): State<AdminState>,
    Query(q): Query<LogsQuery>,
) -> ApiResult<Json<Value>> {
    gate(&st).await?;
    let (lines, next) = st.logs.since(q.after.unwrap_or(0));
    Ok(Json(json!({ "next": next, "lines": lines })))
}

#[derive(Deserialize)]
struct WipeRequest {
    /// "memory" (keeps processed markers) or "all" (wipes markers too).
    scope: String,
    /// Must be exactly "WIPE".
    confirm: String,
}

async fn wipe(_: Auth, State(st): State<AdminState>, Json(req): Json<WipeRequest>) -> ApiResult<Json<Value>> {
    gate(&st).await?;
    if req.confirm != "WIPE" {
        return err(StatusCode::BAD_REQUEST, "type WIPE to confirm");
    }
    let qdrant = st.app.read().await.qdrant.clone();
    if !qdrant.is_available() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "Qdrant is unavailable");
    }
    match req.scope.as_str() {
        "memory" => {
            qdrant.reset_memory().await.map_err(|e| {
                error!("Memory wipe failed: {e}");
                (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "wipe failed" })))
            })?;
            let gemini = st.app.read().await.gemini.clone();
            if let Err(e) = crate::seed_app_knowledge(&qdrant, &gemini).await {
                warn!("Failed to re-seed app knowledge after wipe: {e}");
            }
            info!("Memory wiped via admin panel (processed markers kept)");
        }
        "all" => {
            qdrant.reset_memory().await.map_err(|e| {
                error!("Memory wipe failed: {e}");
                (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "wipe failed" })))
            })?;
            qdrant.wipe_processed().await.map_err(|e| {
                error!("Processed-marker wipe failed: {e}");
                (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "wipe failed" })))
            })?;
            let gemini = st.app.read().await.gemini.clone();
            if let Err(e) = crate::seed_app_knowledge(&qdrant, &gemini).await {
                warn!("Failed to re-seed app knowledge after wipe: {e}");
            }
            // The session cache must not shield old notifications either, and
            // the currently-visible backlog gets silently re-seeded so the bot
            // doesn't answer history.
            st.app.write().await.processed.clear();
            crate::seed_existing_notifications(&st.app).await;
            info!("FULL memory wipe via admin panel (processed markers included)");
        }
        _ => return err(StatusCode::BAD_REQUEST, "scope must be 'memory' or 'all'"),
    }
    Ok(Json(json!({ "ok": true })))
}

async fn restart(_: Auth, State(st): State<AdminState>) -> ApiResult<Json<Value>> {
    gate(&st).await?;
    // Respond first, then exit — systemd (Restart=always) brings the bot back.
    tokio::spawn(async {
        tokio::time::sleep(Duration::from_millis(600)).await;
        info!("Restarting via admin panel");
        std::process::exit(0);
    });
    Ok(Json(json!({ "ok": true })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_event(text: &str) -> LogEvent {
        LogEvent {
            seq: 0,
            ts: "00:00:00.000".to_string(),
            level: "INFO".to_string(),
            target: "test".to_string(),
            text: text.to_string(),
        }
    }

    #[test]
    fn log_buffer_caps_and_pages_by_seq() {
        let buf = new_log_buffer();
        for i in 0..(LOG_BUFFER_CAPACITY + 100) {
            buf.push(log_event(&format!("line {i}")));
        }
        let (all, next) = buf.since(0);
        assert_eq!(all.len(), LOG_BUFFER_CAPACITY);
        assert_eq!(next, (LOG_BUFFER_CAPACITY + 100) as u64);
        // Oldest entries were dropped; the remainder is contiguous.
        assert_eq!(all.first().unwrap().seq, 101);
        let (page, _) = buf.since(all.last().unwrap().seq - 1);
        assert_eq!(page.len(), 1);
    }

    #[test]
    fn merge_api_keys_appends_and_dedupes() {
        let pool = vec!["k1".to_string(), "k2".to_string()];
        let (merged, added) = merge_api_keys(&pool, &["k3".to_string()]);
        assert_eq!(merged, vec!["k1", "k2", "k3"]);
        assert_eq!(added, 1);
        // Duplicates (exact string match) are skipped; order preserved.
        let (merged, added) = merge_api_keys(&pool, &["k2".to_string(), "k4".to_string(), "k1".to_string()]);
        assert_eq!(merged, vec!["k1", "k2", "k4"]);
        assert_eq!(added, 1);
        // Nothing new: pool untouched, count zero.
        let (merged, added) = merge_api_keys(&pool, &["k1".to_string()]);
        assert_eq!(merged, pool);
        assert_eq!(added, 0);
        // Empty pool (fresh install): all adds land.
        let (merged, added) = merge_api_keys(&[], &["a".to_string(), "b".to_string()]);
        assert_eq!(merged, vec!["a", "b"]);
        assert_eq!(added, 2);
    }

    #[test]
    fn log_layer_captures_message_and_level() {
        let buf = new_log_buffer();
        {
            use tracing_subscriber::prelude::*;
            let registry = tracing_subscriber::registry().with(LogLayer::new(buf.clone()));
            let _guard = tracing::subscriber::set_default(registry);
            tracing::info!("hello admin");
            tracing::warn!(target: "askme_bot", "careful now");
        }
        let (events, _) = buf.since(0);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].text, "hello admin");
        assert_eq!(events[0].level, "INFO");
        assert_eq!(events[1].level, "WARN");
        assert_eq!(events[1].target, "askme_bot");
    }

    #[tokio::test]
    async fn throttle_locks_out_after_max_fails() {
        let throttle = Arc::new(Mutex::new(Throttle::default()));
        for _ in 0..LOGIN_MAX_FAILS {
            let mut t = throttle.lock().unwrap();
            t.fails += 1;
            if t.fails >= LOGIN_MAX_FAILS {
                t.locked_until = Some(Instant::now() + Duration::from_secs(LOGIN_LOCKOUT_SECS));
                t.fails = 0;
            }
        }
        let t = throttle.lock().unwrap();
        assert!(t.locked_until.unwrap() > Instant::now());
    }

    #[test]
    fn thinking_level_dropdown_mapping() {
        assert_eq!(parse_thinking_level("default"), Ok(None));
        for level in config::THINKING_LEVELS {
            assert_eq!(parse_thinking_level(level), Ok(Some(level.to_string())));
        }
        assert!(parse_thinking_level("very-hard").is_err());
        assert!(parse_thinking_level("LOW").is_err(), "case-sensitive enum like the API");
    }

    #[test]
    fn restart_changed_compares_effective_values_not_override_presence() {
        // The reported bug: saving an untouched prefilled form wrote
        // qdrant_url=Some(default) (was None) and falsely demanded a restart.
        let old = config::ConfigOverrides::default();
        let mut new = old.clone();
        apply_restart_override(
            &mut new.qdrant_url,
            Some(config::DEFAULT_QDRANT_URL.to_string()),
            "QDRANT_URL",
            config::DEFAULT_QDRANT_URL,
        );
        assert!(!restart_fields_changed(&old, &new), "same effective URL must not require restart");
        assert_eq!(new.qdrant_url, None, "redundant override is normalized away");

        // A genuinely different value DOES require a restart.
        let mut changed = old.clone();
        apply_restart_override(
            &mut changed.qdrant_url,
            Some("http://qdrant.internal:6334".to_string()),
            "QDRANT_URL",
            config::DEFAULT_QDRANT_URL,
        );
        assert!(restart_fields_changed(&old, &changed));
    }

    #[test]
    fn restart_override_normalization() {
        let mut slot = None;
        // Empty = unchanged.
        apply_restart_override(&mut slot, Some(String::new()), "__NO_SUCH_ENV__", "fallback");
        assert_eq!(slot, None);
        // Equal to fallback = normalized to None.
        apply_restart_override(&mut slot, Some("fallback".to_string()), "__NO_SUCH_ENV__", "fallback");
        assert_eq!(slot, None);
        // Different = stored.
        apply_restart_override(&mut slot, Some("custom".to_string()), "__NO_SUCH_ENV__", "fallback");
        assert_eq!(slot, Some("custom".to_string()));
        // Later emptied back to fallback = cleared.
        apply_restart_override(&mut slot, Some("fallback".to_string()), "__NO_SUCH_ENV__", "fallback");
        assert_eq!(slot, None);

        let mut num = Some(512u64);
        apply_restart_override_u64(&mut num, Some(0), "__NO_SUCH_ENV__", 512);
        assert_eq!(num, Some(512), "zero = unchanged");
        apply_restart_override_u64(&mut num, Some(512), "__NO_SUCH_ENV__", 512);
        assert_eq!(num, None, "equal to fallback normalizes to None");
        apply_restart_override_u64(&mut num, Some(768), "__NO_SUCH_ENV__", 512);
        assert_eq!(num, Some(768));
    }

    #[tokio::test]
    async fn unauthenticated_requests_are_rejected() {
        use tower::ServiceExt;

        let app_state = crate::tests::test_state();
        let state = AdminState::new(
            app_state,
            Arc::new(RwLock::new(BotConfig::default())),
            new_log_buffer(),
        );
        let router = router(state);
        for (method, uri) in [
            (axum::http::Method::GET, "/api/status"),
            (axum::http::Method::GET, "/api/faqs"),
            (axum::http::Method::POST, "/api/faqs"),
            (axum::http::Method::DELETE, "/api/faqs?id=x"),
        ] {
            let request = axum::http::Request::builder()
                .method(method)
                .uri(uri)
                .body(axum::body::Body::empty())
                .unwrap();
            let response = router.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{uri}");
        }
    }
}
