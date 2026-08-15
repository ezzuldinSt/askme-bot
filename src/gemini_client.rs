use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use reqwest::{Client, StatusCode};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

use crate::models::*;
use crate::qdrant_client::Embedder;

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com";
const DEFAULT_EMBEDDING_BATCH_SIZE: usize = 10;
/// Files API state polling: the docs poll every 5s; 24 attempts = ~2 min cap
/// (large videos need longer than images to reach ACTIVE).
const POLL_INTERVAL_MS: u64 = 5_000;
const MAX_POLL_ATTEMPTS: u32 = 24;
const EMBED_CACHE_MAX: usize = 2000;
const HTTP_TIMEOUT_SECS: u64 = 120;
/// In-flow transient retries (5xx/transport). Five attempts with the backoff
/// curve below tolerate ~75s of a Google-side overload spike before the
/// caller parks the model and fails over.
const GENERATE_MAX_ATTEMPTS: u32 = 5;
/// After a model's transient retries exhaust (503 "high demand" storms), it
/// is parked for this long: flows skip parked models with ZERO API calls so
/// the poll loop's uncounted retry can wait the spike out cheaply.
const MODEL_OVERLOAD_PARK_SECS: u64 = 60;
/// Backoff curve per transient retry (1-based). ~1.5s/4s/10s/20s/40s ≈ 75s
/// total, then the model parks. Longer than the 10s the old 1/2/4 curve
/// allowed — observed 503 spikes last minutes, not seconds.
const BACKOFF_STEPS_MS: [u64; 5] = [1500, 4000, 10000, 20000, 40000];
/// Default cooldown for a 429 with no parseable RetryInfo.
const DEFAULT_RATE_LIMIT_COOLDOWN_SECS: u64 = 60;
/// Cooldown for a 400 INVALID_ARGUMENT: long enough to mostly skip sticky
/// per-project rejections, short enough to probe again soon (and to bound
/// the cost of a malformed request, which 400s on every key).
const INVALID_ARGUMENT_COOLDOWN_SECS: u64 = 300;
/// Max sleep while waiting for a cooled-down key to come back.
const COOLDOWN_WAIT_CAP_SECS: u64 = 5;

// ── API key pool ──

#[derive(Debug, Clone)]
struct KeyState {
    key: String,
    uses: u64,
    rate_limited: u64,
    cooldown_until: Option<Instant>,
    /// Set when the daily quota is exhausted (parked much longer than an RPM hit).
    daily_park: bool,
    dead: Option<String>,
}

impl KeyState {
    fn new(key: String) -> Self {
        Self {
            key,
            uses: 0,
            rate_limited: 0,
            cooldown_until: None,
            daily_park: false,
            dead: None,
        }
    }

    fn usable(&self, now: Instant) -> bool {
        self.dead.is_none() && self.cooldown_until.is_none_or(|t| t <= now)
    }
}

#[derive(Debug, Default)]
struct KeyPool {
    keys: Vec<KeyState>,
    /// Round-robin cursor: reply flows advance it on success; stateless calls
    /// advance it on every acquire.
    cursor: usize,
}

/// A sticky key for one reply flow. Media uploads and the generation that
/// references them MUST share a key: uploaded files live in the key's project.
#[derive(Debug, Clone)]
pub struct KeyLease {
    pub(crate) idx: usize,
    pub(crate) key: String,
}

/// Masked per-key view for the admin panel.
#[derive(Debug, Clone, serde::Serialize)]
pub struct KeyStatusView {
    pub masked: String,
    /// "active" | "cooldown" | "daily" | "dead"
    pub state: String,
    pub state_secs: u64,
    pub reason: Option<String>,
    pub uses: u64,
    pub rate_limited: u64,
}

/// Outcome of a lease-based call that failed in a way the caller must handle.
#[derive(Debug)]
pub enum GeminiError {
    /// The lease key hit a quota/auth limit (already marked in the pool).
    /// The caller should re-lease with the next key and retry — re-uploading
    /// any media into the new project.
    RateLimited,
    /// Non-retryable failure (bad request, or transient retries exhausted).
    Failed(anyhow::Error),
}

/// Transient errors (5xx/transport) outlasted every in-client retry budget.
/// Typed so callers — e.g. the fact-extraction worker — can tell "try again
/// later" apart from a genuinely broken request.
#[derive(Debug, thiserror::Error)]
#[error("transient Gemini errors exhausted all retries: {0}")]
pub struct TransientExhausted(anyhow::Error);

impl TransientExhausted {
    pub fn new(err: anyhow::Error) -> Self {
        Self(err)
    }
}

/// True if the error (or anything in its chain) is a `TransientExhausted`.
pub fn is_transient_exhausted(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.downcast_ref::<TransientExhausted>().is_some())
}

/// A function call requested by the model during a tool-calling turn. The
/// thought signature and call id live in `GenerateTurn::raw_parts` and are
/// circulated verbatim; this struct only drives execution.
#[derive(Debug, Clone)]
pub struct FunctionCallTurn {
    pub name: String,
    pub args: serde_json::Value,
}

/// The result of one tool-calling generateContent turn: either a final text
/// answer (`function_calls` empty) or one-or-more calls to execute.
#[derive(Debug, Clone)]
pub struct GenerateTurn {
    pub text: Option<String>,
    pub function_calls: Vec<FunctionCallTurn>,
    /// URLs the URL context tool successfully fetched this turn (the answer
    /// may be based on them; the reply flow appends them as a Sources footer).
    pub retrieved_urls: Vec<String>,
    /// Source names from Google Search grounding chunks this turn (the chunk
    /// URIs are Google redirect URLs — only the titles are presentable, so
    /// only titles are collected here).
    pub grounded_sources: Vec<String>,
    /// Queries the google_search tool executed this turn. Gemini 3 bills
    /// grounding PER QUERY, so the reply flow logs these for cost visibility.
    pub grounded_queries: Vec<String>,
    /// Every part of the candidate's model-role content, re-serialized for
    /// circulation into the next request. MUST include toolCall/toolResponse
    /// and all thought signatures when tool context circulation is active.
    pub raw_parts: Vec<Part>,
    /// The candidate's finish reason (e.g. "STOP", "MAX_TOKENS") — lets the
    /// caller retry a truncated final answer instead of posting it.
    pub finish_reason: Option<String>,
}

/// Parse a generateContent response into a turn: collect every text part,
/// every functionCall (with its thought signature), and every server-side
/// toolCall/toolResponse part (preserved for tool context circulation). A
/// candidate with NO parts at all is a generation failure, not a valid turn.
fn parse_turn(response: GenerateContentResponse) -> Result<GenerateTurn, AttemptFailure> {
    let finish_reason = response
        .candidates
        .as_ref()
        .and_then(|c| c.first())
        .and_then(|c| c.finishReason.as_deref())
        .map(|s| s.to_string());
    let block_reason = response
        .promptFeedback
        .as_ref()
        .and_then(|f| f.blockReason.as_deref())
        .map(|s| s.to_string());

    let candidate = response
        .candidates
        .and_then(|c| c.into_iter().next())
        .ok_or_else(|| {
            AttemptFailure::Fatal(anyhow::anyhow!("Gemini returned no candidate"))
        })?;
    let retrieved_urls = candidate
        .url_context_metadata
        .as_ref()
        .and_then(|m| m.url_metadata.as_ref())
        .map(|entries| {
            entries
                .iter()
                .filter(|e| e.url_retrieval_status.as_deref() == Some("URL_RETRIEVAL_STATUS_SUCCESS"))
                .filter_map(|e| e.retrieved_url.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let (grounded_sources, grounded_queries) = match candidate.grounding_metadata.as_ref() {
        Some(g) => {
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            let sources = g
                .grounding_chunks
                .as_ref()
                .map(|chunks| {
                    chunks
                        .iter()
                        .filter_map(|c| c.web.as_ref())
                        .filter_map(|w| w.title.clone())
                        .map(|t| t.trim().to_string())
                        .filter(|t| !t.is_empty() && seen.insert(t.to_lowercase()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let queries = g.web_search_queries.clone().unwrap_or_default();
            (sources, queries)
        }
        None => (Vec::new(), Vec::new()),
    };
    let parts = candidate.content.and_then(|c| c.parts).unwrap_or_default();

    let mut text = String::new();
    let mut function_calls: Vec<FunctionCallTurn> = Vec::new();
    let mut raw_parts: Vec<Part> = Vec::with_capacity(parts.len());
    for part in parts {
        if let Some(t) = part.text {
            // Skip empty text parts with NO thought signature: they carry no
            // content, and the API rejects them on echo ("empty text
            // parameter") — a source of intermittent INVALID_ARGUMENTs on
            // later tool rounds. Empty parts WITH a signature must circulate
            // (docs: signatures may arrive on empty text parts).
            if !t.is_empty() || part.thought_signature.is_some() {
                // Thought-summary parts are NOT answer text: circulate them
                // (they can carry signatures) but never accumulate them into
                // the reply the user sees.
                if part.thought != Some(true) {
                    text.push_str(&t);
                }
                // Text parts can carry thought signatures (sometimes on an
                // EMPTY text part); circulate them verbatim like every other
                // part type.
                raw_parts.push(Part::Text {
                    text: t,
                    thought_signature: part.thought_signature.clone(),
                });
            }
        }
        if let Some(fc) = part.function_call {
            function_calls.push(FunctionCallTurn {
                name: fc.name.clone(),
                args: fc.args.clone(),
            });
            raw_parts.push(Part::FunctionCall {
                function_call: fc,
                thought_signature: part.thought_signature.clone(),
            });
        }
        if let Some(fr) = part.function_response {
            raw_parts.push(Part::FunctionResponse { function_response: fr });
        }
        if let Some(tc) = part.tool_call {
            raw_parts.push(Part::ToolCall {
                tool_call: tc,
                thought_signature: part.thought_signature.clone(),
            });
        }
        if let Some(tr) = part.tool_response {
            raw_parts.push(Part::ToolResponse {
                tool_response: tr,
                thought_signature: part.thought_signature.clone(),
            });
        }
    }

    if raw_parts.is_empty() {
        let reason = finish_reason
            .clone()
            .or(block_reason)
            .unwrap_or_else(|| "unknown".to_string());
        return Err(AttemptFailure::Fatal(anyhow::anyhow!(
            "Gemini returned an empty response (reason: {reason})"
        )));
    }

    if function_calls.is_empty() {
        return Ok(GenerateTurn {
            text: (!text.trim().is_empty()).then_some(text),
            function_calls,
            retrieved_urls,
            grounded_sources,
            grounded_queries,
            raw_parts,
            finish_reason,
        });
    }

    Ok(GenerateTurn {
        text: (!text.trim().is_empty()).then_some(text),
        function_calls,
        retrieved_urls,
        grounded_sources,
        grounded_queries,
        raw_parts,
        finish_reason,
    })
}

/// How a single API attempt failed (internal classification).
#[derive(Debug)]
enum AttemptFailure {
    /// 429 — cool the key down (or park it for the day).
    RateLimited { retry_after: Duration, daily: bool },
    /// 401/403 — the key is invalid; mark it dead.
    Dead(String),
    /// 400 INVALID_ARGUMENT — either the key's project rejects the request
    /// (e.g. model not enabled — sticky per key) or the request itself is
    /// malformed (same on every key). Cool the key briefly and fail over;
    /// if every key 400s, the error still surfaces, just slower.
    InvalidArgument(anyhow::Error),
    /// 5xx or transport error — worth a bounded backoff (not key-related).
    Transient(anyhow::Error),
    /// Other 4xx — the request itself is broken; never retry.
    Fatal(anyhow::Error),
}

#[derive(Clone)]
pub struct GeminiClient {
    client: Client,
    pool: Arc<Mutex<KeyPool>>,
    /// Chat model used for replies; hot-swappable (shared across clones).
    generation_model: Arc<std::sync::RwLock<String>>,
    /// Saturation fallback for the chat model (e.g. 3.6-flash while the
    /// primary is 3.7-flash): used for one whole-flow arm when the primary
    /// exhausts its transient retries. None = no fallback. Hot-swappable.
    fallback_generation_model: Arc<std::sync::RwLock<Option<String>>>,
    /// Model for extraction/FAQ/rewrite jobs; None = use generation_model.
    /// Hot-swappable (shared across clones).
    extraction_model: Arc<std::sync::RwLock<Option<String>>>,
    /// Gemini 3.x thinking level; None = model default. Hot-swappable.
    thinking_level: Arc<std::sync::RwLock<Option<String>>>,
    /// Thinking level for extraction/FAQ/rewrite jobs (Thinking docs:
    /// minimal/low for fact retrieval or classification). None = model
    /// default. Hot-swappable (shared across clones).
    extraction_thinking_level: Arc<std::sync::RwLock<Option<String>>>,
    /// Global media resolution enum value (e.g. "MEDIA_RESOLUTION_LOW") from
    /// the MEDIA_RESOLUTION env var; None = model default. Boot-time only.
    media_resolution: Option<String>,
    embedding_model: String,
    embedding_dimensions: u32,
    embedding_batch_size: usize,
    embed_cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    /// Models parked after their transient retries exhausted (Google-side
    /// overload): model name -> resume-at instant. Flows skip parked models
    /// entirely, spending zero API calls while the spike lasts.
    parked_models: Arc<Mutex<HashMap<String, Instant>>>,
}

fn mask_key(key: &str) -> String {
    let tail: String = key.chars().rev().take(4).collect::<String>().chars().rev().collect();
    format!("••••{tail}")
}

/// Construction knobs for the Gemini client (keeps the constructor tidy).
pub struct GeminiOptions {
    pub generation_model: String,
    /// Saturation fallback for the chat model; None = disabled.
    pub fallback_generation_model: Option<String>,
    pub extraction_model: Option<String>,
    pub thinking_level: Option<String>,
    pub extraction_thinking_level: Option<String>,
    pub media_resolution: Option<String>,
    pub embedding_model: String,
    pub embedding_dimensions: u32,
}

impl GeminiClient {
    /// Back-compatible single-key constructor (used by tests).
    #[allow(dead_code)]
    pub fn new(api_key: String) -> Self {
        Self::with_keys(
            vec![api_key],
            GeminiOptions {
                generation_model: crate::config::DEFAULT_GENERATION_MODEL.to_string(),
                fallback_generation_model: None,
                extraction_model: None,
                thinking_level: None,
                extraction_thinking_level: None,
                media_resolution: None,
                embedding_model: crate::config::DEFAULT_EMBEDDING_MODEL.to_string(),
                embedding_dimensions: crate::config::DEFAULT_EMBEDDING_DIMENSIONS,
            },
        )
    }

    pub fn with_keys(api_keys: Vec<String>, options: GeminiOptions) -> Self {
        let GeminiOptions {
            generation_model,
            fallback_generation_model,
            extraction_model,
            thinking_level,
            extraction_thinking_level,
            media_resolution,
            embedding_model,
            embedding_dimensions,
        } = options;
        let embedding_batch_size = std::env::var("EMBEDDING_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_EMBEDDING_BATCH_SIZE);
        if api_keys.is_empty() {
            error!("Gemini key pool is EMPTY — every API call will fail");
        } else {
            info!(
                "Gemini key pool: {} key(s); generation model: {generation_model}; embeddings: {embedding_model} ({embedding_dimensions}d)",
                api_keys.len()
            );
        }
        let client = Self {
            client: Client::builder()
                .timeout(Duration::from_secs(HTTP_TIMEOUT_SECS))
                .build()
                .expect("Failed to create Gemini HTTP client"),
            pool: Arc::new(Mutex::new(KeyPool {
                keys: api_keys.into_iter().map(KeyState::new).collect(),
                cursor: 0,
            })),
            generation_model: Arc::new(std::sync::RwLock::new(generation_model)),
            fallback_generation_model: Arc::new(std::sync::RwLock::new(fallback_generation_model)),
            extraction_model: Arc::new(std::sync::RwLock::new(extraction_model)),
            thinking_level: Arc::new(std::sync::RwLock::new(thinking_level)),
            extraction_thinking_level: Arc::new(std::sync::RwLock::new(extraction_thinking_level)),
            media_resolution,
            embedding_model,
            embedding_dimensions,
            embedding_batch_size,
            embed_cache: Arc::new(Mutex::new(HashMap::new())),
            parked_models: Arc::new(Mutex::new(HashMap::new())),
        };
        client.warn_on_unsupported_thinking_level();
        client
    }

    /// Hot-swap the chat model (propagates to every clone of this client).
    pub fn set_generation_model(&self, model: String) {
        let mut current = self.generation_model.write().unwrap();
        if *current != model {
            info!("Generation model changed: {} -> {model}", *current);
            *current = model;
        }
        drop(current);
        self.warn_on_unsupported_thinking_level();
    }

    /// The currently active chat model.
    pub fn generation_model(&self) -> String {
        self.generation_model.read().unwrap().clone()
    }

    /// Hot-swap the saturation fallback model (None = disabled; propagates to
    /// every clone). Setting it to the primary model disables the fallback.
    pub fn set_fallback_generation_model(&self, model: Option<String>) {
        let model = model.filter(|m| !m.trim().is_empty());
        let mut current = self.fallback_generation_model.write().unwrap();
        if *current != model {
            info!(
                "Fallback generation model changed: {:?} -> {:?}",
                *current, model
            );
            *current = model;
        }
    }

    /// The configured saturation fallback for the chat model, if any.
    pub fn fallback_generation_model(&self) -> Option<String> {
        self.fallback_generation_model.read().unwrap().clone()
    }

    /// Park a model after its transient retries exhausted — almost always a
    /// Google-side overload spike (503 "high demand"). Parked models are
    /// skipped by flows for `MODEL_OVERLOAD_PARK_SECS` with zero API calls.
    pub fn park_model(&self, model: &str) {
        let until = Instant::now() + Duration::from_secs(MODEL_OVERLOAD_PARK_SECS);
        self.parked_models
            .lock()
            .unwrap()
            .insert(model.to_string(), until);
        warn!(
            "Model {model} parked for {MODEL_OVERLOAD_PARK_SECS}s \
             (transient retries exhausted — likely Google-side overload)"
        );
    }

    /// True while the model is parked. Lazily drops expired entries.
    pub fn model_parked(&self, model: &str) -> bool {
        let mut parked = self.parked_models.lock().unwrap();
        parked.retain(|_, until| *until > Instant::now());
        parked.contains_key(model)
    }

    /// Hot-swap the extraction model (None = fall back to the chat model).
    pub fn set_extraction_model(&self, model: Option<String>) {
        let mut current = self.extraction_model.write().unwrap();
        if *current != model {
            info!("Extraction model changed: {:?} -> {:?}", *current, model);
            *current = model;
        }
        drop(current);
        self.warn_on_unsupported_thinking_level();
    }

    /// The effective extraction model: the override when set, else the chat model.
    pub fn extraction_model(&self) -> String {
        self.extraction_model
            .read()
            .unwrap()
            .clone()
            .unwrap_or_else(|| self.generation_model())
    }

    /// Hot-swap the thinking level (None = model default; shared across clones).
    pub fn set_thinking_level(&self, level: Option<String>) {
        let mut current = self.thinking_level.write().unwrap();
        if *current != level {
            info!("Thinking level changed: {:?} -> {:?}", *current, level);
            *current = level;
        }
        drop(current);
        self.warn_on_unsupported_thinking_level();
    }

    /// The currently configured thinking level (None = model default).
    pub fn thinking_level(&self) -> Option<String> {
        self.thinking_level.read().unwrap().clone()
    }

    /// Hot-swap the extraction thinking level (None = model default).
    pub fn set_extraction_thinking_level(&self, level: Option<String>) {
        let mut current = self.extraction_thinking_level.write().unwrap();
        if *current != level {
            info!("Extraction thinking level changed: {:?} -> {:?}", *current, level);
            *current = level;
        }
        drop(current);
        self.warn_on_unsupported_thinking_level();
    }

    /// Warn once when an active thinking level is unsupported by its model
    /// (Thinking docs' per-model matrix). The request path also silently
    /// drops such levels, so the misconfiguration degrades to the model
    /// default instead of 400ing every call.
    fn warn_on_unsupported_thinking_level(&self) {
        for (model, level) in [
            (self.generation_model(), self.thinking_level()),
            (self.extraction_model(), self.extraction_thinking_level()),
        ] {
            if let Some(level) = level {
                if !thinking_level_supported(&model, &level) {
                    warn!(
                        "Thinking level {level:?} is not supported by {model}; \
                         requests will use the model default instead"
                    );
                }
            }
        }
    }

    /// The currently configured extraction thinking level (None = model default).
    pub fn extraction_thinking_level(&self) -> Option<String> {
        self.extraction_thinking_level.read().unwrap().clone()
    }

    /// Assemble the generationConfig for one request family. `extraction`
    /// selects the extraction thinking level (classification-grade) instead
    /// of the chat level; `media` gates the global mediaResolution (only
    /// requests that can carry media parts get it). `model` is the model the
    /// request will actually run on (may be the fallback arm) — the thinking
    /// level is validated against IT. Returns None when every knob is unset
    /// so the field is omitted entirely.
    fn build_generation_config(
        &self,
        response_mime_type: Option<String>,
        response_schema: Option<serde_json::Value>,
        extraction: bool,
        media: bool,
        model: &str,
    ) -> Option<GenerationConfig> {
        let thinking_level = if extraction {
            self.extraction_thinking_level.read().unwrap().clone()
        } else {
            self.thinking_level.read().unwrap().clone()
        };
        // Drop a level the active model doesn't support (Thinking docs'
        // per-model matrix): sending it would 400 every request. None = the
        // model default, which is always safe. The setters warn once per
        // change, so this filter stays silent.
        let thinking_level =
            thinking_level.filter(|level| thinking_level_supported(model, level));
        let media_resolution = media.then(|| self.media_resolution.clone()).flatten();
        if response_mime_type.is_none()
            && response_schema.is_none()
            && thinking_level.is_none()
            && media_resolution.is_none()
        {
            return None;
        }
        Some(GenerationConfig {
            response_mime_type,
            response_schema,
            media_resolution,
            thinking_config: thinking_level.map(|level| ThinkingConfig { thinking_level: level }),
        })
    }

    /// Hot-swap the key pool (admin panel). Surviving keys keep their stats;
    /// removed keys drop out. Empty input is ignored.
    pub fn set_keys(&self, keys: Vec<String>) {
        if keys.is_empty() {
            return;
        }
        let mut pool = self.pool.lock().unwrap();
        let mut new_keys = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some(existing) = pool.keys.iter().find(|k| k.key == key) {
                let mut kept = existing.clone();
                kept.key = key;
                new_keys.push(kept);
            } else {
                new_keys.push(KeyState::new(key));
            }
        }
        pool.keys = new_keys;
        pool.cursor = 0;
        info!("Gemini key pool updated: {} key(s)", pool.keys.len());
    }

    pub fn pool_size(&self) -> usize {
        self.pool.lock().unwrap().keys.len()
    }

    /// Masked per-key status for the admin panel.
    pub fn pool_status(&self) -> Vec<KeyStatusView> {
        let now = Instant::now();
        let pool = self.pool.lock().unwrap();
        pool.keys
            .iter()
            .map(|k| {
                let (state, secs, reason) = if let Some(r) = &k.dead {
                    ("dead".to_string(), 0, Some(r.clone()))
                } else if let Some(t) = k.cooldown_until.filter(|t| *t > now) {
                    let secs = t.saturating_duration_since(now).as_secs();
                    (
                        if k.daily_park { "daily" } else { "cooldown" }.to_string(),
                        secs,
                        None,
                    )
                } else {
                    ("active".to_string(), 0, None)
                };
                KeyStatusView {
                    masked: mask_key(&k.key),
                    state,
                    state_secs: secs,
                    reason,
                    uses: k.uses,
                    rate_limited: k.rate_limited,
                }
            })
            .collect()
    }

    /// Next usable key, advancing the round-robin cursor (stateless calls).
    /// None = every key is cooling down or dead.
    fn acquire(&self) -> Option<(usize, String)> {
        let now = Instant::now();
        let mut pool = self.pool.lock().unwrap();
        let len = pool.keys.len();
        if len == 0 {
            return None;
        }
        for step in 0..len {
            let idx = (pool.cursor + step) % len;
            if pool.keys[idx].usable(now) {
                pool.keys[idx].uses += 1;
                pool.cursor = (idx + 1) % len;
                return Some((idx, pool.keys[idx].key.clone()));
            }
        }
        None
    }

    /// Sticky key for one reply flow; the cursor only moves on `flow_success`.
    pub fn acquire_lease(&self) -> Option<KeyLease> {
        let now = Instant::now();
        let mut pool = self.pool.lock().unwrap();
        let len = pool.keys.len();
        for step in 0..len {
            let idx = (pool.cursor + step) % len;
            if pool.keys[idx].usable(now) {
                pool.keys[idx].uses += 1;
                return Some(KeyLease {
                    idx,
                    key: pool.keys[idx].key.clone(),
                });
            }
        }
        None
    }

    /// A reply flow completed successfully — advance the round-robin cursor
    /// past its key, so the next flow starts on the next key.
    pub fn flow_success(&self, lease: &KeyLease) {
        let mut pool = self.pool.lock().unwrap();
        if lease.idx < pool.keys.len() {
            pool.cursor = (lease.idx + 1) % pool.keys.len();
        }
    }

    fn mark_rate_limited(&self, idx: usize, retry_after: Duration, daily: bool) {
        let mut pool = self.pool.lock().unwrap();
        if let Some(k) = pool.keys.get_mut(idx) {
            k.rate_limited += 1;
            k.daily_park = daily;
            let dur = if daily { daily_cooldown_duration() } else { retry_after };
            k.cooldown_until = Some(Instant::now() + dur);
            warn!(
                "Gemini key {} rate-limited{}; cooling down for {}s",
                mask_key(&k.key),
                if daily { " (daily cap)" } else { "" },
                dur.as_secs()
            );
        }
    }

    fn mark_dead(&self, idx: usize, reason: String) {
        let mut pool = self.pool.lock().unwrap();
        if let Some(k) = pool.keys.get_mut(idx) {
            error!("Gemini key {} marked DEAD ({reason}); skipping it from now on", mask_key(&k.key));
            k.dead = Some(reason);
        }
    }

    /// Cool a key down WITHOUT touching its 429 counter — used for 400
    /// INVALID_ARGUMENT rejections (sticky per-project, or a request bug
    /// that will show up on every key anyway).
    fn cool_key(&self, idx: usize, dur: Duration, why: &str) {
        let mut pool = self.pool.lock().unwrap();
        if let Some(k) = pool.keys.get_mut(idx) {
            k.cooldown_until = Some(Instant::now() + dur);
            warn!(
                "Gemini key {} rejected the request ({why}); cooling down for {}s",
                mask_key(&k.key),
                dur.as_secs()
            );
        }
    }

    /// Sleep until the earliest cooled-down key is due back (bounded).
    pub async fn wait_for_cooldown(&self) {
        let wake = {
            let pool = self.pool.lock().unwrap();
            pool.keys
                .iter()
                .filter(|k| k.dead.is_none())
                .filter_map(|k| k.cooldown_until)
                .min()
        };
        match wake {
            Some(t) => {
                let dur = t
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_secs(COOLDOWN_WAIT_CAP_SECS));
                sleep(dur).await;
            }
            None => sleep(Duration::from_secs(1)).await,
        }
    }

    /// Run a single-key attempt through the rotation: acquire the next usable
    /// key, and on quota/auth failures mark the key and instantly fail over to
    /// the next one. Only transient (5xx/transport) failures sleep, and only
    /// when every key is cooling do we wait for a cooldown to expire.
    async fn send_with_rotation<'a, T, F, Fut>(&'a self, f: F) -> Result<T>
    where
        F: Fn(String) -> Fut,
        Fut: std::future::Future<Output = Result<T, AttemptFailure>> + 'a,
    {
        let max_attempts = self.pool_size().max(2) + GENERATE_MAX_ATTEMPTS as usize;
        let mut transient_retries = 0u32;
        let mut last_invalid: Option<anyhow::Error> = None;
        for _ in 0..max_attempts {
            let Some((idx, key)) = self.acquire() else {
                self.wait_for_cooldown().await;
                continue;
            };
            match f(key).await {
                Ok(v) => return Ok(v),
                Err(AttemptFailure::RateLimited { retry_after, daily }) => {
                    self.mark_rate_limited(idx, retry_after, daily);
                }
                Err(AttemptFailure::Dead(reason)) => {
                    self.mark_dead(idx, reason);
                }
                Err(AttemptFailure::InvalidArgument(e)) => {
                    self.cool_key(idx, Duration::from_secs(INVALID_ARGUMENT_COOLDOWN_SECS), "400 INVALID_ARGUMENT");
                    last_invalid = Some(e);
                }
                Err(AttemptFailure::Transient(e)) => {
                    transient_retries += 1;
                    if transient_retries >= GENERATE_MAX_ATTEMPTS {
                        return Err(TransientExhausted(e).into());
                    }
                    warn!("Gemini transient error (retry {transient_retries}): {e}");
                    sleep(backoff(transient_retries)).await;
                }
                Err(AttemptFailure::Fatal(e)) => return Err(e),
            }
        }
        Err(last_invalid.unwrap_or_else(|| {
            anyhow::anyhow!("all Gemini API keys are currently rate-limited or dead")
        }))
    }

    // ── Files API (lease-based: files are project-scoped) ──

    /// Upload media using the lease's key. Single attempt — the caller's flow
    /// loop re-leases and re-uploads on `GeminiError::RateLimited`.
    pub async fn upload_file_with(
        &self,
        lease: &KeyLease,
        data: &[u8],
        mime_type: &str,
        display_name: &str,
    ) -> Result<String, GeminiError> {
        match self
            .try_upload_file(&lease.key, data, mime_type, display_name)
            .await
        {
            Ok(uri) => Ok(uri),
            Err(AttemptFailure::RateLimited { retry_after, daily }) => {
                self.mark_rate_limited(lease.idx, retry_after, daily);
                Err(GeminiError::RateLimited)
            }
            Err(AttemptFailure::Dead(reason)) => {
                self.mark_dead(lease.idx, reason);
                Err(GeminiError::RateLimited)
            }
            Err(AttemptFailure::InvalidArgument(e)) => {
                self.cool_key(lease.idx, Duration::from_secs(INVALID_ARGUMENT_COOLDOWN_SECS), "400 INVALID_ARGUMENT");
                warn!("Upload rejected (failing over): {e}");
                Err(GeminiError::RateLimited)
            }
            Err(AttemptFailure::Transient(e)) | Err(AttemptFailure::Fatal(e)) => {
                Err(GeminiError::Failed(e))
            }
        }
    }

    async fn try_upload_file(
        &self,
        key: &str,
        data: &[u8],
        mime_type: &str,
        display_name: &str,
    ) -> Result<String, AttemptFailure> {
        let boundary = format!("boundary_{}", uuid::Uuid::new_v4());
        let metadata = serde_json::json!({
            "file": {
                "display_name": display_name,
                "mime_type": mime_type,
            }
        });

        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(b"Content-Type: application/json; charset=UTF-8\r\n\r\n");
        body.extend_from_slice(&serde_json::to_vec(&metadata).map_err(|e| {
            AttemptFailure::Fatal(anyhow::Error::new(e).context("Failed to serialize file metadata"))
        })?);
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(format!("Content-Type: {mime_type}\r\n\r\n").as_bytes());
        body.extend_from_slice(data);
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

        let url = format!("{GEMINI_API_BASE}/upload/v1beta/files");
        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", key)
            .header("X-Goog-Upload-Protocol", "multipart")
            .header(
                "Content-Type",
                format!("multipart/related; boundary={boundary}"),
            )
            .body(body)
            .send()
            .await
            .map_err(|e| {
                AttemptFailure::Transient(
                    anyhow::Error::new(e).context("Failed to upload file to Gemini Files API"),
                )
            })?;

        let status = resp.status();
        let bytes = resp.bytes().await.map_err(|e| {
            AttemptFailure::Transient(anyhow::Error::new(e).context("Failed to read upload response"))
        })?;

        if !status.is_success() {
            return Err(classify_http_failure(status, &bytes));
        }

        let file_resp: GeminiFileResponse = serde_json::from_slice(&bytes).map_err(|e| {
            AttemptFailure::Fatal(anyhow::Error::new(e).context("Failed to parse Gemini Files API response"))
        })?;

        let file_name = file_resp.file.name.clone();
        let file_uri = file_resp.file.uri.clone();

        info!(
            "Uploaded file to Gemini: {file_name} -> {file_uri} (state: {})",
            file_resp.file.state
        );

        if file_resp.file.state == "ACTIVE" {
            return Ok(file_uri);
        }

        self.poll_file_until_ready(key, &file_name).await?;
        Ok(file_uri)
    }

    async fn poll_file_until_ready(&self, key: &str, file_name: &str) -> Result<(), AttemptFailure> {
        let url = format!("{GEMINI_API_BASE}/v1beta/{file_name}");

        for attempt in 0..MAX_POLL_ATTEMPTS {
            tokio::time::sleep(Duration::from_millis(POLL_INTERVAL_MS)).await;

            let resp = self
                .client
                .get(&url)
                .header("x-goog-api-key", key)
                .send()
                .await
                .map_err(|e| {
                    AttemptFailure::Transient(
                        anyhow::Error::new(e).context("Failed to poll Gemini file state"),
                    )
                })?;

            let status = resp.status();
            let bytes = resp.bytes().await.map_err(|e| {
                AttemptFailure::Transient(
                    anyhow::Error::new(e).context("Failed to read file state response"),
                )
            })?;

            if !status.is_success() {
                return Err(classify_http_failure(status, &bytes));
            }

            let file_resp: GeminiFileStateResponse = serde_json::from_slice(&bytes).map_err(|e| {
                AttemptFailure::Fatal(
                    anyhow::Error::new(e).context("Failed to parse Gemini file state response"),
                )
            })?;

            match file_resp.state.as_str() {
                "ACTIVE" => {
                    info!(
                        "File {file_name} is ACTIVE after ~{}s",
                        (attempt as u64 + 1) * POLL_INTERVAL_MS / 1000
                    );
                    return Ok(());
                }
                "FAILED" => {
                    let err = file_resp
                        .error
                        .as_ref()
                        .map(|e| e.message.as_deref().unwrap_or("unknown error"))
                        .unwrap_or("unknown error");
                    return Err(AttemptFailure::Fatal(anyhow::anyhow!(
                        "Gemini file processing FAILED: {err}"
                    )));
                }
                _ => {
                    info!(
                        "Gemini file {file_name} state: {} (attempt {}/{MAX_POLL_ATTEMPTS})",
                        file_resp.state,
                        attempt + 1
                    );
                }
            }
        }

        Err(AttemptFailure::Fatal(anyhow::anyhow!(
            "Timed out waiting for Gemini file {file_name} to become ACTIVE"
        )))
    }

    // ── GenerateContent ──

    /// A model to fail an extraction job over to when `exhausted` burned its
    /// transient retries: the configured generation fallback if usable, else
    /// the chat model if usable. None = nothing else to try.
    fn extraction_failover_model(&self, exhausted: &str) -> Option<String> {
        let mut candidates: Vec<String> = Vec::new();
        if let Some(fb) = self.fallback_generation_model() {
            candidates.push(fb);
        }
        candidates.push(self.generation_model());
        candidates
            .into_iter()
            .find(|c| c != exhausted && !self.model_parked(c))
    }

    /// The extraction model to use for a job: itself unless it is parked —
    /// then a usable failover model (generation fallback / chat model) so
    /// background memory keeps working through a saturation storm.
    fn unparked_extraction_model(&self) -> String {
        let model = self.extraction_model();
        if !self.model_parked(&model) {
            return model;
        }
        match self.extraction_failover_model(&model) {
            Some(c) => {
                warn!("{model} is parked; routing this extraction job to {c}");
                c
            }
            None => model, // nothing better: the rotation's retries will try it
        }
    }

    /// Generate a response constrained to a JSON Schema (`responseMimeType`
    /// = application/json + `responseSchema` — Structured Outputs docs).
    /// Stateless call: rotates per request. If the extraction model exhausts
    /// (overload spike), one retry runs on a failover model.
    pub async fn generate_json(
        &self,
        system_prompt: &str,
        user_text: &str,
        schema: serde_json::Value,
    ) -> Result<String> {
        let model = self.unparked_extraction_model();
        let result = self
            .send_with_rotation(|key: String| {
                let model = model.clone();
                let schema = schema.clone();
                async move {
                    self.try_generate(&key, &model, system_prompt, user_text, &[], Some(schema))
                        .await
                }
            })
            .await;
        match &result {
            Err(e) if is_transient_exhausted(e) => {
                self.park_model(&model);
                if let Some(fb) = self.extraction_failover_model(&model) {
                    warn!("generate_json: {model} exhausted; retrying once on {fb}");
                    return self
                        .send_with_rotation(|key: String| {
                            let fb = fb.clone();
                            let schema = schema.clone();
                            async move {
                                self.try_generate(&key, &fb, system_prompt, user_text, &[], Some(schema))
                                    .await
                            }
                        })
                        .await;
                }
            }
            _ => {}
        }
        result
    }

    /// One plain-text rewrite pass (scaffold-leak cleanup), routed to the
    /// extraction model — a mechanical job that should not burn reply quota.
    /// Falls back to a failover model when the extraction model is exhausted.
    pub async fn rewrite_text(&self, system_prompt: &str, user_text: &str) -> Result<String> {
        let model = self.unparked_extraction_model();
        let result = self
            .send_with_rotation(|key: String| {
                let model = model.clone();
                async move { self.try_generate(&key, &model, system_prompt, user_text, &[], None).await }
            })
            .await;
        match &result {
            Err(e) if is_transient_exhausted(e) => {
                self.park_model(&model);
                if let Some(fb) = self.extraction_failover_model(&model) {
                    warn!("rewrite_text: {model} exhausted; retrying once on {fb}");
                    return self
                        .send_with_rotation(|key: String| {
                            let fb = fb.clone();
                            async move {
                                self.try_generate(&key, &fb, system_prompt, user_text, &[], None).await
                            }
                        })
                        .await;
                }
            }
            _ => {}
        }
        result
    }

    /// One tool-calling round on the lease's key against an explicit model:
    /// sends the contents history plus tool declarations and returns the
    /// model's turn (text or calls). 429/401/403 mark the key and return
    /// `GeminiError::RateLimited` (fail over to the next KEY); transient
    /// errors get a bounded same-key backoff and, on exhaustion, park the
    /// MODEL and return `TransientExhausted` (fail over to the next MODEL —
    /// the caller detects it with `is_transient_exhausted`). The tool loop
    /// (in the caller) holds the lease across rounds.
    pub async fn generate_turn_with(
        &self,
        lease: &KeyLease,
        model: &str,
        system_prompt: &str,
        contents: &[Content],
        tools: &[Tool],
    ) -> Result<GenerateTurn, GeminiError> {
        let mut transient_retries = 0u32;
        let mut max_tokens_retries = 0u32;
        loop {
            match self
                .try_generate_turn(&lease.key, model, system_prompt, contents, tools)
                .await
            {
                Ok(turn) => {
                    // A FINAL answer cut off by the output budget would be
                    // posted mid-thought — retry it once before accepting.
                    let truncated_final = turn.function_calls.is_empty()
                        && turn.finish_reason.as_deref() == Some("MAX_TOKENS");
                    if truncated_final {
                        max_tokens_retries += 1;
                        if max_tokens_retries == 1 {
                            warn!("generateContent (tools): final answer hit MAX_TOKENS; retrying once");
                            continue;
                        }
                        warn!("generateContent (tools): still MAX_TOKENS after retry; accepting");
                    }
                    return Ok(turn);
                }
                Err(AttemptFailure::RateLimited { retry_after, daily }) => {
                    self.mark_rate_limited(lease.idx, retry_after, daily);
                    return Err(GeminiError::RateLimited);
                }
                Err(AttemptFailure::Dead(reason)) => {
                    self.mark_dead(lease.idx, reason);
                    return Err(GeminiError::RateLimited);
                }
                Err(AttemptFailure::InvalidArgument(e)) => {
                    self.cool_key(lease.idx, Duration::from_secs(INVALID_ARGUMENT_COOLDOWN_SECS), "400 INVALID_ARGUMENT");
                    warn!("generateContent rejected (failing over): {e}");
                    return Err(GeminiError::RateLimited);
                }
                Err(AttemptFailure::Transient(e)) => {
                    transient_retries += 1;
                    if transient_retries >= GENERATE_MAX_ATTEMPTS {
                        // Model-level exhaustion (5xx storm), not a key
                        // problem: park the model and mark the error so the
                        // reply flow can switch models instead of keys.
                        self.park_model(model);
                        return Err(GeminiError::Failed(TransientExhausted(e).into()));
                    }
                    warn!("generateContent (tools) transient error (retry {transient_retries}): {e}");
                    sleep(backoff(transient_retries)).await;
                }
                Err(AttemptFailure::Fatal(e)) => return Err(GeminiError::Failed(e)),
            }
        }
    }

    /// One generateContent attempt with an explicit key and model.
    /// `json_schema` constrains the response to a JSON Schema (Structured
    /// Outputs); None produces a plain-text response.
    async fn try_generate(
        &self,
        key: &str,
        model: &str,
        system_prompt: &str,
        user_text: &str,
        file_uris: &[(String, String)],
        json_schema: Option<serde_json::Value>,
    ) -> Result<String, AttemptFailure> {
        let mut parts: Vec<Part> = Vec::new();

        parts.push(Part::Text {
            text: user_text.to_string(),
            thought_signature: None,
        });

        for (uri, mime) in file_uris {
            parts.push(Part::FileData {
                file_data: FileData {
                    mime_type: mime.clone(),
                    file_uri: uri.clone(),
                },
            });
        }

        let request = GenerateContentRequest {
            system_instruction: Some(SystemInstruction {
                parts: vec![Part::Text {
                    text: system_prompt.to_string(),
                    thought_signature: None,
                }],
            }),
            contents: vec![Content {
                role: "user".to_string(),
                parts,
            }],
            tools: None,
            tool_config: None,
            generation_config: self.build_generation_config(
                json_schema.is_some().then(|| "application/json".to_string()),
                json_schema,
                true,
                false,
                model,
            ),
        };

        let response = self.send_generate(key, model, request).await?;

        let finish_reason = response
            .candidates
            .as_ref()
            .and_then(|c| c.first())
            .and_then(|c| c.finishReason.as_deref())
            .map(|s| s.to_string());
        let block_reason = response
            .promptFeedback
            .as_ref()
            .and_then(|f| f.blockReason.as_deref())
            .map(|s| s.to_string());

        // Concatenate EVERY non-thought text part: a response split across
        // multiple text parts must not lose its tail (JSON payloads
        // especially — a truncated document silently fails to parse).
        let text = response
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content)
            .and_then(|c| c.parts)
            .map(|parts| {
                parts
                    .into_iter()
                    .filter(|p| p.thought != Some(true))
                    .filter_map(|p| p.text)
                    .collect::<String>()
            })
            .unwrap_or_default();

        // An empty candidate (safety block, recitation filter, ...) must not be
        // posted as an empty reply — treat it as a generation failure.
        if text.trim().is_empty() {
            let reason = finish_reason
                .or(block_reason)
                .unwrap_or_else(|| "unknown".to_string());
            return Err(AttemptFailure::Fatal(anyhow::anyhow!(
                "Gemini returned an empty response (reason: {reason})"
            )));
        }

        Ok(text)
    }

    /// One generateContent POST with an explicit key and model,
    /// error-classified and parsed. Shared by the text-only and the
    /// tool-calling paths.
    async fn send_generate(
        &self,
        key: &str,
        model: &str,
        request: GenerateContentRequest,
    ) -> Result<GenerateContentResponse, AttemptFailure> {
        let url = format!("{GEMINI_API_BASE}/v1beta/models/{model}:generateContent");

        let send_result = self
            .client
            .post(&url)
            .header("x-goog-api-key", key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await;

        let resp = match send_result {
            Ok(r) => r,
            Err(e) => {
                return Err(AttemptFailure::Transient(
                    anyhow::Error::new(e).context("Failed to send generateContent request"),
                ))
            }
        };

        let status = resp.status();
        let bytes = resp.bytes().await.map_err(|e| {
            AttemptFailure::Transient(
                anyhow::Error::new(e).context("Failed to read Gemini response body"),
            )
        })?;

        if !status.is_success() {
            return Err(classify_http_failure(status, &bytes));
        }

        let response: GenerateContentResponse = serde_json::from_slice(&bytes).map_err(|e| {
            AttemptFailure::Fatal(
                anyhow::Error::new(e).context("Failed to parse Gemini generateContent response"),
            )
        })?;

        // Token accounting per call (Thinking docs: billed output = candidates
        // + thoughts) — visibility into what each reply/extraction costs. The
        // cached count shows implicit context-cache hits (on by default for
        // Gemini 2.5+; a stable request prefix above the model's minimum).
        if let Some(u) = &response.usageMetadata {
            info!(
                "Gemini {model} usage: {} prompt ({} cached) + {} candidates + {} thoughts = {} tokens",
                u.promptTokenCount.unwrap_or(0),
                u.cachedContentTokenCount.unwrap_or(0),
                u.candidatesTokenCount.unwrap_or(0),
                u.thoughtsTokenCount.unwrap_or(0),
                u.totalTokenCount.unwrap_or(0),
            );
        }
        Ok(response)
    }

    /// One tool-calling generateContent attempt against an explicit model:
    /// sends the full contents history plus the tool declarations, and
    /// returns whatever the model produced — a final text answer, or
    /// one-or-more function calls.
    async fn try_generate_turn(
        &self,
        key: &str,
        model: &str,
        system_prompt: &str,
        contents: &[Content],
        tools: &[Tool],
    ) -> Result<GenerateTurn, AttemptFailure> {
        // Built-in tools (url_context) combined with function declarations
        // require server-side tool context circulation: without the flag the
        // API rejects the request (400 INVALID_ARGUMENT). The flag must ALSO
        // stay on when the history already carries server-side toolCall/
        // toolResponse parts — e.g. the final no-tools answer call after a
        // run of tool rounds, which declares no tools at all.
        let request = GenerateContentRequest {
            system_instruction: Some(SystemInstruction {
                parts: vec![Part::Text {
                    text: system_prompt.to_string(),
                    thought_signature: None,
                }],
            }),
            contents: contents.to_vec(),
            tools: (!tools.is_empty()).then(|| tools.to_vec()),
            tool_config: needs_tool_config(tools, contents).then_some(ToolConfig {
                include_server_side_tool_invocations: true,
            }),
            generation_config: self.build_generation_config(None, None, false, true, model),
        };

        let response = self.send_generate(key, model, request).await?;
        parse_turn(response)
    }

    // ── Embeddings ──

    /// Embed a batch of texts into L2-normalized vectors using the embeddings API.
    ///
    /// Texts are split into batches of `EMBEDDING_BATCH_SIZE`; each batch is
    /// sent through the key rotation (a 429 fails over to the next key
    /// instantly). Duplicate content is served from an in-memory cache.
    async fn embed_texts_inner(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        let mut missing: Vec<usize> = Vec::new();
        let mut pending_texts: Vec<String> = Vec::new();

        {
            let cache = self.embed_cache.lock().unwrap();
            for (i, text) in texts.iter().enumerate() {
                match cache.get(text) {
                    Some(v) => results.push(v.clone()),
                    None => {
                        results.push(Vec::new());
                        missing.push(i);
                        pending_texts.push(text.clone());
                    }
                }
            }
        }

        if !pending_texts.is_empty() {
            let mut fetched: Vec<Vec<f32>> = Vec::new();
            for chunk in pending_texts.chunks(self.embedding_batch_size.max(1)) {
                let vectors = self
                    .send_with_rotation(|key: String| async move {
                        self.try_embed_chunk(&key, chunk).await
                    })
                    .await?;
                fetched.extend(vectors);
            }

            debug!(
                "Embedded {} texts (model {}, dim {})",
                fetched.len(),
                self.embedding_model,
                self.embedding_dimensions
            );

            {
                let mut cache = self.embed_cache.lock().unwrap();
                if cache.len() + fetched.len() > EMBED_CACHE_MAX {
                    cache.clear();
                }
                for (i, vector) in fetched.iter().cloned().enumerate() {
                    cache.insert(pending_texts[i].clone(), vector);
                }
            }

            for (slot, vector) in missing.into_iter().zip(fetched) {
                results[slot] = vector;
            }
        }

        Ok(results)
    }

    /// One batchEmbedContents attempt with an explicit key.
    async fn try_embed_chunk(&self, key: &str, chunk: &[String]) -> Result<Vec<Vec<f32>>, AttemptFailure> {
        let url = format!(
            "{GEMINI_API_BASE}/v1beta/models/{}:batchEmbedContents",
            self.embedding_model
        );

        let requests: Vec<EmbedContentRequest> = chunk
            .iter()
            .map(|text| EmbedContentRequest {
                model: format!("models/{}", self.embedding_model),
                content: Content {
                    role: "user".to_string(),
                    parts: vec![Part::Text {
                        text: text.clone(),
                        thought_signature: None,
                    }],
                },
                taskType: Some("SEMANTIC_SIMILARITY".to_string()),
                outputDimensionality: Some(self.embedding_dimensions),
            })
            .collect();

        let body = BatchEmbedContentsRequest { requests };
        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                AttemptFailure::Transient(
                    anyhow::Error::new(e).context("Failed to send batchEmbedContents request"),
                )
            })?;

        let status = resp.status();
        let bytes = resp.bytes().await.map_err(|e| {
            AttemptFailure::Transient(
                anyhow::Error::new(e).context("Failed to read embedding response"),
            )
        })?;

        if !status.is_success() {
            return Err(classify_http_failure(status, &bytes));
        }

        let parsed: BatchEmbedContentsResponse = serde_json::from_slice(&bytes).map_err(|e| {
            AttemptFailure::Fatal(
                anyhow::Error::new(e).context("Failed to parse embedding response"),
            )
        })?;
        collect_embeddings(parsed, chunk.len()).map_err(AttemptFailure::Fatal)
    }
}

/// Server-side tool context circulation must be enabled whenever the request
/// declares a built-in tool (url_context, google_search) OR the outgoing
/// history already carries server-side toolCall/toolResponse parts — otherwise
/// the API rejects the echo with 400 INVALID_ARGUMENT. The second case covers
/// the final no-tools answer call after a run of tool rounds.
fn needs_tool_config(tools: &[Tool], contents: &[Content]) -> bool {
    tools.iter().any(|t| t.url_context.is_some() || t.google_search.is_some())
        || contents.iter().any(|c| {
            c.parts
                .iter()
                .any(|p| matches!(p, Part::ToolCall { .. } | Part::ToolResponse { .. }))
        })
}

/// Exponential backoff over `BACKOFF_STEPS_MS` (clamped to the last step)
/// with ±25% jitter — the Troubleshooting docs recommend jitter so
/// concurrent workers/keys don't retry in lockstep. The jitter seed comes
/// from a UUID: no RNG dependency.
fn backoff(transient_retries: u32) -> Duration {
    let base_ms = BACKOFF_STEPS_MS[(transient_retries as usize).saturating_sub(1).min(BACKOFF_STEPS_MS.len() - 1)];
    let spread = base_ms / 4;
    let seed = u64::from(uuid::Uuid::new_v4().as_bytes()[0]); // 0..=255
    Duration::from_millis(base_ms - spread + (seed * spread * 2 / 255))
}

/// Per-model thinking-level support (Thinking docs matrix). Only models with
/// RESTRICTED level sets are listed; anything unrecognized is assumed to
/// accept every level, so newly released models are never blocked.
fn thinking_level_supported(model: &str, level: &str) -> bool {
    let name = model.rsplit('/').next().unwrap_or(model);
    let restricted: Option<&[&str]> = if name.starts_with("gemini-3-pro-preview") {
        Some(&["low", "high"])
    } else if name.starts_with("gemini-3.1-pro-preview") {
        Some(&["low", "medium", "high"])
    } else if name.starts_with("gemini-3.1-flash-lite-image") {
        Some(&["minimal", "high"])
    } else if name.starts_with("gemini-2.5") {
        Some(&["low", "medium", "high"])
    } else {
        None
    };
    match restricted {
        Some(levels) => levels.contains(&level),
        None => true,
    }
}

fn collect_embeddings(
    response: BatchEmbedContentsResponse,
    expected: usize,
) -> Result<Vec<Vec<f32>>> {
    let embeddings = response
        .embeddings
        .ok_or_else(|| anyhow::anyhow!("Embedding response missing embeddings field"))?;
    if embeddings.len() != expected {
        anyhow::bail!(
            "Embedding response count mismatch: got {}, expected {expected}",
            embeddings.len()
        );
    }
    Ok(embeddings
        .into_iter()
        .map(|e| normalize_vector(e.values))
        .collect())
}

fn normalize_vector(mut values: Vec<f32>) -> Vec<f32> {
    let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in values.iter_mut() {
            *v /= norm;
        }
    }
    values
}

// ── HTTP failure classification (429/401/403/5xx/other) ──

fn classify_http_failure(status: StatusCode, body: &[u8]) -> AttemptFailure {
    let text = String::from_utf8_lossy(body);
    if status == StatusCode::TOO_MANY_REQUESTS {
        let (retry_after, daily) = parse_rate_limit_details(body);
        return AttemptFailure::RateLimited { retry_after, daily };
    }
    if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
        // Google proactively blocks keys identified as leaked (Troubleshooting
        // docs) — fail over, but make the log actionable: rotating to another
        // EXISTING key won't help; the operator must mint a new one.
        if text.contains("reported as leaked") {
            return AttemptFailure::Dead(format!(
                "HTTP {status}: key reported as leaked — generate a NEW key in Google AI Studio"
            ));
        }
        return AttemptFailure::Dead(format!("HTTP {status}: {}", truncate(&text, 200)));
    }
    // Google's INVALID_ARGUMENT for a bad key comes back as HTTP 400, not 401.
    if status == StatusCode::BAD_REQUEST && text.contains("API key not valid") {
        return AttemptFailure::Dead(format!("HTTP {status}: API key not valid"));
    }
    let err = anyhow::anyhow!("Gemini API error (HTTP {status}): {}", truncate(&text, 400));
    if status == StatusCode::BAD_REQUEST {
        // Cool the key briefly and fail over: covers both sticky per-project
        // rejections and (slower, but still surfaced) malformed requests.
        return AttemptFailure::InvalidArgument(err);
    }
    // 408 and 5xx are transient (Troubleshooting docs: retry 429, 408, 5xx).
    if status == StatusCode::REQUEST_TIMEOUT || status.is_server_error() {
        AttemptFailure::Transient(err)
    } else {
        AttemptFailure::Fatal(err)
    }
}

fn truncate(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Parse a 429 body: RetryInfo.retryDelay (e.g. "34s") and whether the quota
/// failure is a DAILY cap (quotaId containing "PerDay").
fn parse_rate_limit_details(body: &[u8]) -> (Duration, bool) {
    #[derive(serde::Deserialize)]
    struct ErrBody {
        error: Option<ErrDetail>,
    }
    #[derive(serde::Deserialize)]
    struct ErrDetail {
        details: Option<Vec<Detail>>,
    }
    #[derive(serde::Deserialize)]
    struct Detail {
        #[serde(rename = "@type")]
        ty: Option<String>,
        #[serde(rename = "retryDelay")]
        retry_delay: Option<String>,
        violations: Option<Vec<Violation>>,
    }
    #[derive(serde::Deserialize)]
    struct Violation {
        #[serde(rename = "quotaId")]
        quota_id: Option<String>,
    }

    let mut retry_after = Duration::from_secs(DEFAULT_RATE_LIMIT_COOLDOWN_SECS);
    let mut daily = false;

    if let Ok(parsed) = serde_json::from_slice::<ErrBody>(body) {
        if let Some(details) = parsed.error.and_then(|e| e.details) {
            for detail in details {
                let ty = detail.ty.unwrap_or_default();
                if ty.ends_with("RetryInfo") {
                    if let Some(d) = detail.retry_delay.and_then(|s| parse_retry_delay(&s)) {
                        retry_after = d;
                    }
                }
                if ty.ends_with("QuotaFailure") {
                    if let Some(violations) = detail.violations {
                        if violations.iter().any(|v| {
                            v.quota_id
                                .as_deref()
                                .map(|id| id.contains("PerDay"))
                                .unwrap_or(false)
                        }) {
                            daily = true;
                        }
                    }
                }
            }
        }
    }
    (retry_after, daily)
}

/// "34s" / "0.5s" -> Duration.
fn parse_retry_delay(s: &str) -> Option<Duration> {
    let secs: f64 = s.trim().trim_end_matches('s').parse().ok()?;
    if !(0.0..=86400.0).contains(&secs) {
        return None;
    }
    Some(Duration::from_secs_f64(secs))
}

/// The nth Sunday of a month (US DST rule: PDT runs from the 2nd Sunday of
/// March to the 1st Sunday of November).
fn nth_sunday(year: i32, month: u32, n: u32) -> chrono::NaiveDate {
    use chrono::Datelike;
    let first = chrono::NaiveDate::from_ymd_opt(year, month, 1).expect("valid date");
    let offset = (7 - first.weekday().num_days_from_sunday()) % 7;
    first + chrono::Duration::days((offset + 7 * (n - 1)) as i64)
}

/// Hours behind UTC at midnight Pacific on the given date: 8 during PST,
/// 7 during PDT. The daily free-tier quota resets at midnight Pacific, so
/// the cooldown target moves with DST.
fn pacific_utc_offset_hours(date: chrono::NaiveDate) -> u32 {
    use chrono::Datelike;
    let year = date.year();
    if date >= nth_sunday(year, 3, 2) && date < nth_sunday(year, 11, 1) {
        7
    } else {
        8
    }
}

/// Daily free-tier quota resets at midnight Pacific (08:00 UTC in PST,
/// 07:00 UTC in PDT). Split from `daily_cooldown_duration` for testability.
fn daily_cooldown_from(now: chrono::DateTime<chrono::Utc>) -> Duration {
    let offset = pacific_utc_offset_hours(now.date_naive());
    let today_reset = now
        .date_naive()
        .and_hms_opt(offset, 0, 0)
        .map(|dt| dt.and_utc());
    let target = match today_reset {
        Some(t) if now < t => t,
        Some(t) => t + chrono::Duration::days(1),
        None => return Duration::from_secs(3600),
    };
    (target - now)
        .to_std()
        .unwrap_or_else(|_| Duration::from_secs(3600))
}

fn daily_cooldown_duration() -> Duration {
    daily_cooldown_from(chrono::Utc::now())
}

#[async_trait::async_trait]
impl Embedder for GeminiClient {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_texts_inner(texts).await
    }
}

// ── Long-term memory extraction ──

/// System prompt for the background fact-extraction pass. Deliberately strict:
/// most messages yield nothing, and anything extracted becomes long-term memory.
const EXTRACTION_PROMPT: &str = r#"You are the long-term memory extractor for AskMe, an AI bot on the Things social network.
Given ONE user message, decide what is worth remembering long-term.

Output ONLY a JSON object of this exact shape:
{
  "user_facts": [{"fact": "...", "category": "identity|location|occupation|preference|opinion|other"}],
  "app_facts": [{"fact": "...", "topic": "..."}],
  "forget": ["..."]
}

Rules:
- user_facts: durable facts about the message's AUTHOR only — name, location, job, stable preferences, lasting opinions. Write each as one short third-person sentence in English (translate if needed). No moods, no temporary states, no questions, no facts about other people.
- app_facts: factual claims about the Things app itself (features, rules, limits, how it works). Write each in English (translate if needed).
- forget: when the user asks to forget, delete, or correct something they previously shared ("forget I said I live in Riyadh", "I don't live there anymore"), put the fact to remove here, in English (translate if needed).
- Most messages contain nothing worth remembering — return empty arrays then.
- Never invent facts. When in doubt, leave it out."#;

/// What the extraction pass pulled out of one user message.
#[derive(Debug, Default, serde::Deserialize)]
pub struct ExtractedFacts {
    #[serde(default)]
    pub user_facts: Vec<ExtractedUserFact>,
    #[serde(default)]
    pub app_facts: Vec<ExtractedAppFact>,
    #[serde(default)]
    pub forget: Vec<String>,
}

#[derive(Debug, serde::Deserialize)]
pub struct ExtractedUserFact {
    pub fact: String,
    #[serde(default)]
    pub category: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub struct ExtractedAppFact {
    pub fact: String,
    #[serde(default)]
    pub topic: Option<String>,
}

impl GeminiClient {
    /// Run the extraction pass over one user message. Never fails the caller's
    /// workflow: unparseable output degrades to "nothing extracted".
    pub async fn extract_facts(&self, username: &str, text: &str) -> Result<ExtractedFacts> {
        let user_text = format!("[Message by {username}] {text}");
        let raw = self
            .generate_json(EXTRACTION_PROMPT, &user_text, extraction_schema())
            .await?;
        Ok(parse_extracted_facts(&raw))
    }
}

/// JSON Schema for the extraction pass (Structured Outputs docs: schema
/// constrains the shape instead of hoping the prose prompt does; every field
/// required per the troubleshooting guide). The lenient parser stays as a
/// fallback safety net.
fn extraction_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "user_facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact": { "type": "string" },
                        "category": {
                            "type": "string",
                            "enum": ["identity", "location", "occupation", "preference", "opinion", "other"]
                        }
                    },
                    "required": ["fact", "category"]
                }
            },
            "app_facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact": { "type": "string" },
                        "topic": { "type": "string" }
                    },
                    "required": ["fact", "topic"]
                }
            },
            "forget": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["user_facts", "app_facts", "forget"]
    })
}

/// Leniently parse the extractor's JSON output: strips markdown code fences
/// and ignores any prose around the JSON object.
fn parse_extracted_facts(raw: &str) -> ExtractedFacts {
    let trimmed = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let start = trimmed.find('{');
    let end = trimmed.rfind('}');
    let candidate = match (start, end) {
        (Some(s), Some(e)) if e > s => &trimmed[s..=e],
        _ => return ExtractedFacts::default(),
    };
    serde_json::from_str(candidate).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── extraction parsing (pre-existing coverage) ──

    #[test]
    fn parses_clean_json() {
        let raw = r#"{"user_facts":[{"fact":"lives in Riyadh","category":"location"}],"app_facts":[],"forget":[]}"#;
        let parsed = parse_extracted_facts(raw);
        assert_eq!(parsed.user_facts.len(), 1);
        assert_eq!(parsed.user_facts[0].fact, "lives in Riyadh");
        assert_eq!(parsed.user_facts[0].category.as_deref(), Some("location"));
        assert!(parsed.app_facts.is_empty());
        assert!(parsed.forget.is_empty());
    }

    #[test]
    fn parses_fenced_json_with_prose_around_it() {
        let raw = "Here is the extraction:\n```json\n{\"user_facts\":[],\"app_facts\":[{\"fact\":\"Things is a social network\",\"topic\":\"platform\"}],\"forget\":[\"lives in Riyadh\"]}\n```\nDone.";
        let parsed = parse_extracted_facts(raw);
        assert!(parsed.user_facts.is_empty());
        assert_eq!(parsed.app_facts.len(), 1);
        assert_eq!(parsed.forget, vec!["lives in Riyadh".to_string()]);
    }

    #[test]
    fn missing_sections_default_to_empty() {
        let parsed = parse_extracted_facts("{}");
        assert!(parsed.user_facts.is_empty());
        assert!(parsed.app_facts.is_empty());
        assert!(parsed.forget.is_empty());
    }

    #[test]
    fn garbage_degrades_to_nothing() {
        assert!(parse_extracted_facts("not json at all").user_facts.is_empty());
        assert!(parse_extracted_facts("").user_facts.is_empty());
        assert!(parse_extracted_facts("{broken").user_facts.is_empty());
    }

    // ── URL context metadata ──

    #[test]
    fn parse_turn_collects_successful_url_context_retrievals() {
        let raw = r#"{
            "candidates": [{
                "content": { "parts": [{"text": "answered from the page"}], "role": "model" },
                "urlContextMetadata": {
                    "urlMetadata": [
                        {"retrievedUrl": "https://ok.example", "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"},
                        {"retrievedUrl": "https://fail.example", "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_ERROR"},
                        {"retrievedUrl": "https://ok2.example", "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"}
                    ]
                }
            }]
        }"#;
        let response: GenerateContentResponse = serde_json::from_str(raw).unwrap();
        let turn = parse_turn(response).unwrap_or_else(|_| panic!("turn parses"));
        assert_eq!(turn.text.as_deref(), Some("answered from the page"));
        assert_eq!(
            turn.retrieved_urls,
            vec!["https://ok.example".to_string(), "https://ok2.example".to_string()]
        );
    }

    #[test]
    fn parse_turn_without_metadata_has_empty_sources() {
        let raw = r#"{
            "candidates": [{
                "content": { "parts": [{"text": "plain answer"}], "role": "model" }
            }]
        }"#;
        let response: GenerateContentResponse = serde_json::from_str(raw).unwrap();
        let turn = parse_turn(response).unwrap_or_else(|_| panic!("turn parses"));
        assert!(turn.retrieved_urls.is_empty());
    }

    #[test]
    fn parse_turn_skips_empty_text_without_signature_keeps_signed() {
        // An empty, unsigned text part must not circulate (the API rejects it
        // on echo); an empty part WITH a thought signature must circulate.
        let raw = r#"{
            "candidates": [{
                "content": { "parts": [
                    {"text": ""},
                    {"text": "", "thoughtSignature": "sig-1"},
                    {"text": "real answer"}
                ], "role": "model" }
            }]
        }"#;
        let response: GenerateContentResponse = serde_json::from_str(raw).unwrap();
        let turn = parse_turn(response).unwrap_or_else(|_| panic!("turn parses"));
        assert_eq!(turn.text.as_deref(), Some("real answer"));
        assert_eq!(turn.raw_parts.len(), 2, "unsigned empty part dropped");
        assert!(
            matches!(&turn.raw_parts[0], Part::Text { text, thought_signature }
                if text.is_empty() && thought_signature.as_deref() == Some("sig-1")),
            "signed empty part circulated with its signature"
        );
        assert!(
            matches!(&turn.raw_parts[1], Part::Text { text, .. } if text == "real answer"),
        );
    }

    #[test]
    fn parse_turn_concatenates_text_parts_and_skips_thought_parts() {
        // B1/B2: every non-thought text part contributes to the answer;
        // thought-summary parts circulate for signatures but are not text.
        let raw = r#"{
            "candidates": [{
                "content": { "parts": [
                    {"text": "let me think about this", "thought": true, "thoughtSignature": "sig-t"},
                    {"text": "{\"user_facts\":"},
                    {"text": " [],\"app_facts\":[],\"forget\":[]}"}
                ], "role": "model" }
            }]
        }"#;
        let response: GenerateContentResponse = serde_json::from_str(raw).unwrap();
        let turn = parse_turn(response).unwrap_or_else(|_| panic!("turn parses"));
        assert_eq!(
            turn.text.as_deref(),
            Some("{\"user_facts\": [],\"app_facts\":[],\"forget\":[]}"),
            "split text parts concatenated, thought text excluded"
        );
        assert_eq!(turn.raw_parts.len(), 3, "thought part still circulated");
        assert!(
            matches!(&turn.raw_parts[0], Part::Text { text, thought_signature }
                if text == "let me think about this" && thought_signature.as_deref() == Some("sig-t")),
            "thought part keeps its signature for circulation"
        );
    }

    #[test]
    fn needs_tool_config_covers_builtins_and_history_tool_parts() {
        use crate::models::{ToolCallData, ToolResponseData};
        let url_tool = Tool::url_context();
        let fn_tool = Tool {
            function_declarations: vec![crate::models::FunctionDeclaration {
                name: "f".to_string(),
                description: "d".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            url_context: None,
            google_search: None,
        };
        let text_content = |s: &str| Content {
            role: "user".to_string(),
            parts: vec![Part::Text {
                text: s.to_string(),
                thought_signature: None,
            }],
        };
        // Built-in tool declared -> on.
        assert!(needs_tool_config(std::slice::from_ref(&url_tool), &[text_content("hi")]));
        // google_search is a built-in server-side tool too -> on.
        assert!(needs_tool_config(std::slice::from_ref(&Tool::google_search()), &[text_content("hi")]));
        // Only function declarations, plain history -> off.
        assert!(!needs_tool_config(std::slice::from_ref(&fn_tool), &[text_content("hi")]));
        // No tools at all, plain history -> off.
        assert!(!needs_tool_config(&[], &[text_content("hi")]));
        // No tools, but the history carries server-side tool parts -> ON
        // (the final no-tools answer call after tool rounds — B3).
        let history_with_tool_parts = vec![
            text_content("hi"),
            Content {
                role: "model".to_string(),
                parts: vec![
                    Part::ToolCall {
                        tool_call: ToolCallData {
                            tool_type: "url_context".to_string(),
                            args: serde_json::json!({}),
                            id: Some("t1".to_string()),
                        },
                        thought_signature: None,
                    },
                    Part::ToolResponse {
                        tool_response: ToolResponseData {
                            tool_type: "url_context".to_string(),
                            response: serde_json::json!({}),
                            id: Some("t1".to_string()),
                        },
                        thought_signature: None,
                    },
                ],
            },
        ];
        assert!(needs_tool_config(&[], &history_with_tool_parts));
    }

    // ── Pacific daily-reset math (B4) ──

    fn utc(y: i32, m: u32, d: u32, hh: u32, mm: u32) -> chrono::DateTime<chrono::Utc> {
        chrono::NaiveDate::from_ymd_opt(y, m, d)
            .unwrap()
            .and_hms_opt(hh, mm, 0)
            .unwrap()
            .and_utc()
    }

    #[test]
    fn pacific_offset_follows_us_dst() {
        use chrono::NaiveDate;
        // PDT window: 2nd Sunday of March .. 1st Sunday of November.
        assert_eq!(pacific_utc_offset_hours(NaiveDate::from_ymd_opt(2026, 8, 5).unwrap()), 7);
        assert_eq!(pacific_utc_offset_hours(NaiveDate::from_ymd_opt(2026, 3, 8).unwrap()), 7);
        assert_eq!(pacific_utc_offset_hours(NaiveDate::from_ymd_opt(2026, 3, 7).unwrap()), 8);
        assert_eq!(pacific_utc_offset_hours(NaiveDate::from_ymd_opt(2026, 1, 15).unwrap()), 8);
        assert_eq!(pacific_utc_offset_hours(NaiveDate::from_ymd_opt(2026, 11, 1).unwrap()), 8);
        assert_eq!(pacific_utc_offset_hours(NaiveDate::from_ymd_opt(2026, 10, 31).unwrap()), 7);
        // 2026 transition sanity: Mar 8 and Nov 1 are Sundays.
        assert_eq!(nth_sunday(2026, 3, 2), NaiveDate::from_ymd_opt(2026, 3, 8).unwrap());
        assert_eq!(nth_sunday(2026, 11, 1), NaiveDate::from_ymd_opt(2026, 11, 1).unwrap());
    }

    #[test]
    fn daily_cooldown_targets_midnight_pacific() {
        // PDT (August): midnight Pacific = 07:00 UTC same day.
        let d = daily_cooldown_from(utc(2026, 8, 5, 12, 0));
        assert_eq!(d, Duration::from_secs(19 * 3600));
        // Before the reset on the same day: target is today's 07:00 UTC.
        let d = daily_cooldown_from(utc(2026, 8, 5, 6, 0));
        assert_eq!(d, Duration::from_secs(3600));
        // PST (January): midnight Pacific = 08:00 UTC.
        let d = daily_cooldown_from(utc(2026, 1, 15, 12, 0));
        assert_eq!(d, Duration::from_secs(20 * 3600));
        // Always in the future and within a day.
        for now in [utc(2026, 8, 5, 0, 0), utc(2026, 8, 5, 23, 59), utc(2026, 1, 15, 7, 59)] {
            let d = daily_cooldown_from(now);
            assert!(d > Duration::ZERO && d <= Duration::from_secs(86400));
        }
    }

    #[test]
    fn classify_400_invalid_argument_fails_over_not_fatal() {
        assert!(matches!(
            classify_http_failure(
                StatusCode::BAD_REQUEST,
                br#"{"error":{"code":400,"message":"Request contains an invalid argument.","status":"INVALID_ARGUMENT"}}"#
            ),
            AttemptFailure::InvalidArgument(_)
        ));
        // A bad key is still Dead, never cooled-and-retried.
        assert!(matches!(
            classify_http_failure(
                StatusCode::BAD_REQUEST,
                br#"{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT"}}"#
            ),
            AttemptFailure::Dead(_)
        ));
        // Other 4xx stay Fatal.
        assert!(matches!(
            classify_http_failure(StatusCode::NOT_FOUND, b"nope"),
            AttemptFailure::Fatal(_)
        ));
    }

    // ── key pool ──

    fn client_with_keys(n: usize) -> GeminiClient {
        GeminiClient::with_keys(
            (0..n).map(|i| format!("key-{i:04}")).collect(),
            GeminiOptions {
                generation_model: crate::config::DEFAULT_GENERATION_MODEL.to_string(),
                fallback_generation_model: None,
                extraction_model: None,
                thinking_level: None,
                extraction_thinking_level: None,
                media_resolution: None,
                embedding_model: crate::config::DEFAULT_EMBEDDING_MODEL.to_string(),
                embedding_dimensions: crate::config::DEFAULT_EMBEDDING_DIMENSIONS,
            },
        )
    }

    #[test]
    fn extraction_model_falls_back_to_generation_model() {
        let client = client_with_keys(1);
        assert_eq!(client.extraction_model(), client.generation_model());
        let clone = client.clone();
        client.set_extraction_model(Some("gemini-3.5-flash-lite".to_string()));
        assert_eq!(clone.extraction_model(), "gemini-3.5-flash-lite");
        assert_eq!(clone.generation_model(), crate::config::DEFAULT_GENERATION_MODEL);
        client.set_extraction_model(None);
        assert_eq!(clone.extraction_model(), crate::config::DEFAULT_GENERATION_MODEL);
    }

    #[test]
    fn acquire_rotates_through_all_keys() {
        let client = client_with_keys(3);
        let picks: Vec<String> = (0..6).map(|_| client.acquire().unwrap().1).collect();
        assert_eq!(
            picks,
            vec!["key-0000", "key-0001", "key-0002", "key-0000", "key-0001", "key-0002"]
        );
    }

    #[test]
    fn flow_lease_is_sticky_and_success_advances_cursor() {
        let client = client_with_keys(3);
        let a = client.acquire_lease().unwrap();
        let b = client.acquire_lease().unwrap();
        // No success yet: both flows get the key at the cursor.
        assert_eq!(a.key, b.key);
        assert_eq!(a.key, "key-0000");
        client.flow_success(&a);
        let c = client.acquire_lease().unwrap();
        assert_eq!(c.key, "key-0001", "cursor advanced past the successful key");
        client.flow_success(&c);
        let d = client.acquire_lease().unwrap();
        assert_eq!(d.key, "key-0002");
        client.flow_success(&d);
        let e = client.acquire_lease().unwrap();
        assert_eq!(e.key, a.key, "rotation wraps around the pool");
    }

    #[test]
    fn dead_and_cooling_keys_are_skipped() {
        let client = client_with_keys(3);
        client.mark_dead(0, "401".to_string());
        client.mark_rate_limited(1, Duration::from_secs(60), false);
        for _ in 0..4 {
            assert_eq!(client.acquire().unwrap().1, "key-0002");
        }
        // Cool down key 2 as well: nothing usable.
        client.mark_rate_limited(2, Duration::from_secs(60), false);
        assert!(client.acquire().is_none());
    }

    #[test]
    fn pool_status_masks_and_reports_state() {
        let client = client_with_keys(2);
        client.mark_rate_limited(1, Duration::from_secs(30), false);
        let status = client.pool_status();
        assert_eq!(status.len(), 2);
        assert_eq!(status[0].masked, "••••0000");
        assert_eq!(status[0].state, "active");
        assert_eq!(status[1].state, "cooldown");
        assert!(status[1].state_secs > 0 && status[1].state_secs <= 30);
        assert_eq!(status[1].rate_limited, 1);
    }

    #[test]
    fn set_keys_preserves_stats_and_swaps_pool() {
        let client = client_with_keys(2);
        client.acquire(); // key-0000 uses=1
        client.acquire(); // key-0001 uses=1
        client.mark_rate_limited(0, Duration::from_secs(30), false);
        client.set_keys(vec!["key-0001".to_string(), "brand-new".to_string()]);
        let status = client.pool_status();
        assert_eq!(status.len(), 2);
        assert_eq!(status[0].masked, "••••0001");
        assert_eq!(status[0].uses, 1, "surviving key keeps its stats");
        assert_eq!(status[1].masked, "••••-new");
        assert_eq!(status[1].uses, 0);
        // Empty input is ignored.
        client.set_keys(vec![]);
        assert_eq!(client.pool_status().len(), 2);
    }

    // ── 429 body parsing ──

    #[test]
    fn parses_retry_info_delay() {
        let body = br#"{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","details":[
            {"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"34s"}
        ]}}"#;
        let (d, daily) = parse_rate_limit_details(body);
        assert_eq!(d, Duration::from_secs(34));
        assert!(!daily);
    }

    #[test]
    fn parses_daily_quota_violation() {
        let body = br#"{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","details":[
            {"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[
                {"quotaMetric":"generativelanguage.googleapis.com/generate_requests_per_model_per_day",
                 "quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier","quotaValue":"50"}]},
            {"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"0.5s"}
        ]}}"#;
        let (d, daily) = parse_rate_limit_details(body);
        assert!(daily);
        assert_eq!(d, Duration::from_secs_f64(0.5));
    }

    #[test]
    fn unparseable_429_body_defaults_to_60s() {
        let (d, daily) = parse_rate_limit_details(b"not json");
        assert_eq!(d, Duration::from_secs(60));
        assert!(!daily);
    }

    #[test]
    fn retry_delay_parser_rejects_garbage() {
        assert_eq!(parse_retry_delay("12s"), Some(Duration::from_secs(12)));
        assert_eq!(parse_retry_delay("0.25s"), Some(Duration::from_secs_f64(0.25)));
        assert_eq!(parse_retry_delay("abc"), None);
        assert_eq!(parse_retry_delay("99999999s"), None);
    }

    #[test]
    fn daily_cooldown_is_bounded() {
        let d = daily_cooldown_duration();
        assert!(d > Duration::ZERO && d <= Duration::from_secs(86400));
    }

    #[test]
    fn transient_exhausted_is_downcastable_through_chain() {
        let inner = anyhow::anyhow!("Gemini API error (HTTP 503 Service Unavailable)");
        let err: anyhow::Error = TransientExhausted(inner).into();
        let wrapped = err.context("generate_json failed");
        assert!(is_transient_exhausted(&wrapped));

        let plain = anyhow::anyhow!("Gemini API error (HTTP 400 Bad Request)");
        assert!(!is_transient_exhausted(&plain));
    }

    #[test]
    fn generation_model_hot_swap_propagates_to_clones() {
        let client = client_with_keys(1);
        let clone = client.clone();
        assert_eq!(client.generation_model(), crate::config::DEFAULT_GENERATION_MODEL);
        client.set_generation_model("gemini-3.5-flash-lite".to_string());
        assert_eq!(clone.generation_model(), "gemini-3.5-flash-lite");
        // Setting the same value again is a no-op (no churn).
        client.set_generation_model("gemini-3.5-flash-lite".to_string());
        assert_eq!(clone.generation_model(), "gemini-3.5-flash-lite");
    }

    #[test]
    fn thinking_level_hot_swap_propagates_to_clones() {
        let client = client_with_keys(1);
        let clone = client.clone();
        assert_eq!(client.thinking_level(), None);
        client.set_thinking_level(Some("low".to_string()));
        assert_eq!(clone.thinking_level(), Some("low".to_string()));
        client.set_thinking_level(None);
        assert_eq!(clone.thinking_level(), None);
    }

    #[test]
    fn thinking_config_serializes_camel_case_like_the_rest_docs() {
        let cfg = GenerationConfig {
            response_mime_type: None,
            response_schema: None,
            media_resolution: None,
            thinking_config: Some(ThinkingConfig {
                thinking_level: "low".to_string(),
            }),
        };
        let json = serde_json::to_value(&cfg).unwrap();
        assert_eq!(json, serde_json::json!({"thinkingConfig": {"thinkingLevel": "low"}}));
        // Without a level the field is omitted entirely (model default path).
        let plain = GenerationConfig {
            response_mime_type: None,
            response_schema: None,
            media_resolution: None,
            thinking_config: None,
        };
        assert_eq!(serde_json::to_value(&plain).unwrap(), serde_json::json!({}));
    }

    #[test]
    fn generation_config_serializes_schema_and_media_resolution() {
        let cfg = GenerationConfig {
            response_mime_type: Some("application/json".to_string()),
            response_schema: Some(serde_json::json!({"type": "object"})),
            media_resolution: Some("MEDIA_RESOLUTION_LOW".to_string()),
            thinking_config: None,
        };
        let json = serde_json::to_value(&cfg).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "responseMimeType": "application/json",
                "responseSchema": {"type": "object"},
                "mediaResolution": "MEDIA_RESOLUTION_LOW"
            })
        );
    }

    #[test]
    fn extraction_thinking_level_hot_swap_propagates_to_clones() {
        let client = client_with_keys(1);
        let clone = client.clone();
        assert_eq!(client.extraction_thinking_level(), None);
        client.set_extraction_thinking_level(Some("minimal".to_string()));
        assert_eq!(clone.extraction_thinking_level(), Some("minimal".to_string()));
        client.set_extraction_thinking_level(None);
        assert_eq!(clone.extraction_thinking_level(), None);
    }

    #[test]
    fn build_generation_config_picks_the_right_thinking_level() {
        let client = client_with_keys(1);
        client.set_thinking_level(Some("high".to_string()));
        client.set_extraction_thinking_level(Some("minimal".to_string()));
        let model = crate::config::DEFAULT_GENERATION_MODEL;
        // Chat path: chat level + media resolution gate.
        let chat = client
            .build_generation_config(None, None, false, true, model)
            .expect("chat config present");
        assert_eq!(
            chat.thinking_config.map(|t| t.thinking_level).as_deref(),
            Some("high")
        );
        // Extraction path: the cheaper extraction level, no media resolution.
        let extract = client
            .build_generation_config(Some("application/json".to_string()), None, true, false, model)
            .expect("extraction config present");
        assert_eq!(
            extract.thinking_config.map(|t| t.thinking_level).as_deref(),
            Some("minimal")
        );
        assert_eq!(extract.response_mime_type.as_deref(), Some("application/json"));
        // Everything unset -> the whole field is omitted.
        let bare = client_with_keys(1);
        assert!(bare.build_generation_config(None, None, false, false, model).is_none());
    }

    #[test]
    fn classify_leaked_key_gets_actionable_reason() {
        // Google's leaked-key block message must surface as "mint a NEW key",
        // not a generic 403 dump (Troubleshooting docs).
        let body = br#"{"error":{"code":403,"message":"Your API key was reported as leaked. Please use another API key.","status":"PERMISSION_DENIED"}}"#;
        match classify_http_failure(StatusCode::FORBIDDEN, body) {
            AttemptFailure::Dead(reason) => {
                assert!(reason.contains("leaked"), "reason names the cause: {reason}");
                assert!(reason.contains("NEW key"), "reason is actionable: {reason}");
            }
            other => panic!("leaked key must be Dead, got {other:?}"),
        }
        // An ordinary 403 keeps the raw-body reason.
        match classify_http_failure(StatusCode::FORBIDDEN, b"forbidden") {
            AttemptFailure::Dead(reason) => assert!(reason.contains("forbidden")),
            other => panic!("plain 403 must be Dead, got {other:?}"),
        }
    }

    #[test]
    fn backoff_stays_within_jitter_bounds() {
        // New curve: ~1.5s/4s/10s/20s/40s (clamped at the last step) —
        // ~75s total tolerance for 5xx overload spikes.
        let steps: [(u32, u64); 6] = [
            (1, 1500),
            (2, 4000),
            (3, 10000),
            (4, 20000),
            (5, 40000),
            (99, 40000),
        ];
        for (retries, base_ms) in steps {
            for _ in 0..50 {
                let d = backoff(retries).as_millis() as u64;
                let spread = base_ms / 4;
                assert!(
                    (base_ms - spread..=base_ms + spread).contains(&d),
                    "backoff({retries}) = {d}ms outside ±25% of {base_ms}ms"
                );
            }
        }
    }

    #[test]
    fn model_park_blocks_then_expires() {
        let client = client_with_keys(1);
        assert!(!client.model_parked("gemini-3.7-flash"));
        client.park_model("gemini-3.7-flash");
        assert!(client.model_parked("gemini-3.7-flash"));
        assert!(!client.model_parked("gemini-3.6-flash"), "other models unaffected");
        // Expired entries are dropped lazily (simulated by backdating).
        {
            let mut parked = client.parked_models.lock().unwrap();
            if let Some(until) = parked.get_mut("gemini-3.7-flash") {
                *until = std::time::Instant::now();
            }
        }
        assert!(!client.model_parked("gemini-3.7-flash"), "park expired");
    }

    #[test]
    fn fallback_model_hot_swap_propagates_to_clones() {
        let client = client_with_keys(1);
        let clone = client.clone();
        assert_eq!(clone.fallback_generation_model(), None);
        client.set_fallback_generation_model(Some("gemini-3.6-flash".to_string()));
        assert_eq!(
            clone.fallback_generation_model().as_deref(),
            Some("gemini-3.6-flash")
        );
        // Blank input clears it (the panel sends "" for "disabled").
        client.set_fallback_generation_model(Some("   ".to_string()));
        assert_eq!(clone.fallback_generation_model(), None);
        // Same model as the primary is allowed to be stored; the flow skips
        // it as a duplicate arm at use time.
        client.set_fallback_generation_model(Some(clone.generation_model()));
        assert_eq!(clone.fallback_generation_model(), Some(clone.generation_model()));
    }

    #[test]
    fn extraction_failover_model_skips_parked_and_duplicates() {
        let client = client_with_keys(1);
        client.set_generation_model("gemini-3.7-flash".to_string());
        client.set_fallback_generation_model(Some("gemini-3.6-flash".to_string()));
        // Nothing parked: the configured fallback is the first candidate.
        assert_eq!(
            client.extraction_failover_model("gemini-3.7-flash").as_deref(),
            Some("gemini-3.6-flash")
        );
        // The exhausted model is never offered back: falls through to chat.
        assert_eq!(
            client.extraction_failover_model("gemini-3.6-flash").as_deref(),
            Some("gemini-3.7-flash")
        );
        // Parked candidates are skipped; nothing usable -> None.
        client.park_model("gemini-3.6-flash");
        client.park_model("gemini-3.7-flash");
        assert_eq!(client.extraction_failover_model("anything"), None);
        // No fallback configured: the chat model is the only candidate.
        let bare = client_with_keys(1);
        assert_eq!(
            bare.extraction_failover_model("other").as_deref(),
            Some(bare.generation_model().as_str())
        );
    }

    #[test]
    fn thinking_level_support_matrix() {
        // Unrestricted models (Thinking docs): every level passes.
        for level in ["minimal", "low", "medium", "high"] {
            assert!(thinking_level_supported("gemini-3.6-flash", level));
            assert!(thinking_level_supported("gemini-3.5-flash-lite", level));
            assert!(thinking_level_supported("gemini-some-future-model", level), "unknown models are never blocked");
        }
        // gemini-3-pro-preview: low/high only.
        assert!(thinking_level_supported("gemini-3-pro-preview", "low"));
        assert!(thinking_level_supported("gemini-3-pro-preview", "high"));
        assert!(!thinking_level_supported("gemini-3-pro-preview", "minimal"));
        assert!(!thinking_level_supported("gemini-3-pro-preview", "medium"));
        // gemini-3.1-pro-preview: no minimal.
        assert!(!thinking_level_supported("gemini-3.1-pro-preview", "minimal"));
        assert!(thinking_level_supported("gemini-3.1-pro-preview", "medium"));
        // gemini-3.1-flash-lite-image: minimal/high only.
        assert!(thinking_level_supported("gemini-3.1-flash-lite-image", "minimal"));
        assert!(!thinking_level_supported("gemini-3.1-flash-lite-image", "low"));
        // gemini-2.5 family: no minimal. "models/"-prefixed names work too.
        assert!(!thinking_level_supported("models/gemini-2.5-flash", "minimal"));
        assert!(thinking_level_supported("models/gemini-2.5-pro", "medium"));
    }

    #[test]
    fn build_generation_config_drops_unsupported_level() {
        let client = client_with_keys(1);
        client.set_generation_model("gemini-3-pro-preview".to_string());
        client.set_thinking_level(Some("minimal".to_string()));
        // The unsupported level is dropped (validated against the REQUEST'S
        // model, not the configured one) -> every knob unset -> omitted.
        assert!(client
            .build_generation_config(None, None, false, false, "gemini-3-pro-preview")
            .is_none());
        // A supported level survives.
        client.set_thinking_level(Some("high".to_string()));
        let cfg = client
            .build_generation_config(None, None, false, false, "gemini-3-pro-preview")
            .expect("supported level produces a config");
        assert_eq!(
            cfg.thinking_config.map(|t| t.thinking_level).as_deref(),
            Some("high")
        );
        // The SAME level on the fallback arm's model (accepts all levels)
        // survives too — the guard follows the request, not the config.
        let cfg = client
            .build_generation_config(None, None, false, false, "gemini-3.6-flash")
            .expect("fallback-arm config produced");
        assert_eq!(
            cfg.thinking_config.map(|t| t.thinking_level).as_deref(),
            Some("high")
        );
    }

    #[test]
    fn parse_turn_collects_grounding_metadata() {
        let raw = r#"{
            "candidates": [{
                "content": { "parts": [{"text": "Spain won."}], "role": "model" },
                "groundingMetadata": {
                    "webSearchQueries": ["euro 2024 winner", "who won euro 2024"],
                    "searchEntryPoint": { "renderedContent": "<!-- widget -->" },
                    "groundingChunks": [
                        {"web": {"uri": "https://vertexaisearch.cloud.google.com/1", "title": "aljazeera.com"}},
                        {"web": {"uri": "https://vertexaisearch.cloud.google.com/2", "title": "Aljazeera.com"}},
                        {"web": {"uri": "https://vertexaisearch.cloud.google.com/3", "title": "uefa.com"}},
                        {"web": {"uri": "https://vertexaisearch.cloud.google.com/4"}},
                        {"images": {"uri": "https://x"}}
                    ]
                }
            }]
        }"#;
        let response: GenerateContentResponse = serde_json::from_str(raw).unwrap();
        let turn = parse_turn(response).unwrap_or_else(|_| panic!("turn parses"));
        assert_eq!(turn.text.as_deref(), Some("Spain won."));
        assert_eq!(
            turn.grounded_queries,
            vec!["euro 2024 winner".to_string(), "who won euro 2024".to_string()]
        );
        assert_eq!(
            turn.grounded_sources,
            vec!["aljazeera.com".to_string(), "uefa.com".to_string()],
            "titles deduped case-insensitively; missing titles and non-web chunks skipped"
        );
    }

    #[test]
    fn classify_maps_statuses() {
        assert!(matches!(
            classify_http_failure(StatusCode::TOO_MANY_REQUESTS, b"{}"),
            AttemptFailure::RateLimited { .. }
        ));
        assert!(matches!(
            classify_http_failure(StatusCode::UNAUTHORIZED, b"bad key"),
            AttemptFailure::Dead(_)
        ));
        assert!(matches!(
            classify_http_failure(StatusCode::FORBIDDEN, b"forbidden"),
            AttemptFailure::Dead(_)
        ));
        assert!(matches!(
            classify_http_failure(StatusCode::INTERNAL_SERVER_ERROR, b"boom"),
            AttemptFailure::Transient(_)
        ));
        // 408 is transient per the Troubleshooting docs (retry 429, 408, 5xx).
        assert!(matches!(
            classify_http_failure(StatusCode::REQUEST_TIMEOUT, b"slow"),
            AttemptFailure::Transient(_)
        ));
        // Plain 400: cool the key and fail over (per-project rejection or
        // malformed request), no longer Fatal.
        assert!(matches!(
            classify_http_failure(StatusCode::BAD_REQUEST, b"bad request"),
            AttemptFailure::InvalidArgument(_)
        ));
        // Non-400 4xx stays Fatal.
        assert!(matches!(
            classify_http_failure(StatusCode::UNPROCESSABLE_ENTITY, b"bad request"),
            AttemptFailure::Fatal(_)
        ));
    }

    /// LIVE verification that url_context + function declarations coexist in
    /// one generateContent request AND that server-side tool parts circulate
    /// across rounds (the bot's tool loop shape). Run manually:
    /// `GEMINI_API_KEY=... cargo test url_context_and_functions_coexist -- --ignored`
    #[tokio::test]
    #[ignore]
    async fn url_context_and_functions_coexist() {
        let key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY");
        let client = GeminiClient::new(key);
        let lease = client.acquire_lease().expect("lease");
        let tools = vec![
            crate::models::Tool::url_context(),
            crate::models::Tool {
                function_declarations: vec![crate::models::FunctionDeclaration {
                    name: "get_fact".to_string(),
                    description: "Look up a saved fact by query string.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": { "query": { "type": "string" } },
                        "required": ["query"],
                    }),
                }],
                url_context: None,
                google_search: None,
            },
        ];
        let mut contents = vec![crate::models::Content {
            role: "user".to_string(),
            parts: vec![crate::models::Part::Text {
                text: "What is the main heading of https://en.wikipedia.org/wiki/UEFA_Euro_2024_final ? \
                       Then call get_fact with query \"euro\". Answer in one sentence after the tool result."
                    .to_string(),
                thought_signature: None,
            }],
        }];
        let mut sources: Vec<String> = Vec::new();
        for _round in 0..3 {
            let turn = client
                .generate_turn_with(&lease, &client.generation_model(), "You are a helpful assistant.", &contents, &tools)
                .await
                .expect("live generateContent with url_context + functions");
            println!(
                "round {_round}: text={:?} calls={:?} urls={:?} parts={}",
                turn.text,
                turn.function_calls.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
                turn.retrieved_urls,
                turn.raw_parts.len(),
            );
            sources.extend(turn.retrieved_urls.clone());
            if turn.function_calls.is_empty() {
                let text = turn.text.expect("final answer text");
                assert!(
                    text.to_lowercase().contains("final"),
                    "final answer must reflect the canned tool result"
                );
                assert!(
                    sources.iter().any(|u| u.contains("wikipedia")),
                    "url_context retrieval must be reported: {sources:?}"
                );
                return;
            }
            contents.push(crate::models::Content {
                role: "model".to_string(),
                parts: turn.raw_parts.clone(),
            });
            let mut responses: Vec<crate::models::Part> = Vec::new();
            for call in &turn.function_calls {
                let id = turn
                    .raw_parts
                    .iter()
                    .find_map(|p| match p {
                        crate::models::Part::FunctionCall { function_call, .. }
                            if function_call.name == call.name =>
                        {
                            function_call.id.clone()
                        }
                        _ => None,
                    });
                println!("  executing {} id={id:?}", call.name);
                responses.push(crate::models::Part::FunctionResponse {
                    function_response: crate::models::FunctionResponseData {
                        name: call.name.clone(),
                        response: serde_json::json!({ "fact": "euro 2024 final was played in Berlin" }),
                        id,
                    },
                });
            }
            contents.push(crate::models::Content {
                role: "user".to_string(),
                parts: responses,
            });
        }
        panic!("loop never produced a final answer");
    }

    /// LIVE verification that the built-in google_search tool grounds an
    /// answer in one turn AND that its queries/source titles are collected.
    /// Run manually:
    /// `GEMINI_API_KEY=... cargo test google_search_grounds_answer -- --ignored`
    #[tokio::test]
    #[ignore]
    async fn google_search_grounds_answer() {
        let key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY");
        let client = GeminiClient::new(key);
        let lease = client.acquire_lease().expect("lease");
        let tools = vec![crate::models::Tool::google_search()];
        let contents = vec![crate::models::Content {
            role: "user".to_string(),
            parts: vec![crate::models::Part::Text {
                text: "Who won the most recent UEFA European Championship? Answer in one sentence."
                    .to_string(),
                thought_signature: None,
            }],
        }];
        let turn = client
            .generate_turn_with(&lease, &client.generation_model(), "You are a helpful assistant.", &contents, &tools)
            .await
            .expect("live generateContent with google_search");
        println!(
            "text={:?} queries={:?} sources={:?}",
            turn.text, turn.grounded_queries, turn.grounded_sources
        );
        assert!(turn.function_calls.is_empty(), "built-in tool: no client-side calls");
        assert!(turn.text.as_deref().unwrap_or("").len() > 10, "a real answer came back");
        assert!(
            !turn.grounded_queries.is_empty(),
            "the model searched (queries are the billable unit)"
        );
        assert!(
            !turn.grounded_sources.is_empty(),
            "grounding chunk titles collected for the Sources footer"
        );
    }
}
