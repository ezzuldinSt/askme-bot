//! Layered bot configuration and admin password storage.
//!
//! Precedence (highest first):
//!   1. `bot_config.json` overrides (written by the admin panel)
//!   2. environment variables (`.env`)
//!   3. built-in defaults
//!
//! Admin bind/port are the exception: env always wins over the file, so an
//! operator can relocate the panel without editing JSON.
//!
//! The file is written atomically (tmp + rename) with mode 600 — it holds the
//! argon2 password hash and, if edited via the panel, API secrets.

use anyhow::{Context, Result};
use argon2::password_hash::{rand_core::OsRng, SaltString};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Persisted configuration file (working directory).
pub const CONFIG_FILE: &str = "bot_config.json";

/// Initial admin password on first boot; the panel forces a change on first
/// login, so it never survives beyond initial setup.
pub const DEFAULT_ADMIN_PASSWORD: &str = "CHANGEME";

/// Default chat model for replies and extraction (hot-reloadable).
pub const DEFAULT_GENERATION_MODEL: &str = "gemini-3.6-flash";
/// Default embeddings model for memory vectors (restart-required).
pub const DEFAULT_EMBEDDING_MODEL: &str = "gemini-embedding-2";
/// Default embedding vector size (gemini-embedding-2 supports Matryoshka
/// outputDimensionality; 512 keeps collections small).
pub const DEFAULT_EMBEDDING_DIMENSIONS: u32 = 512;
/// Default Qdrant gRPC endpoint.
pub const DEFAULT_QDRANT_URL: &str = "http://localhost:6334";
/// Valid Gemini 3.x thinking levels (None = let the model use its own default).
pub const THINKING_LEVELS: [&str; 4] = ["minimal", "low", "medium", "high"];
/// Default thinking level for extraction/FAQ/rewrite jobs. The Thinking docs
/// recommend minimal/low for fact retrieval and classification — the chat
/// model's own default (medium) is wasted on these mechanical calls.
pub const DEFAULT_EXTRACTION_THINKING_LEVEL: &str = "low";
/// The embedding model used before DEFAULT_EMBEDDING_MODEL became the default;
/// a missing migration marker is assumed to mean vectors of this model.
pub const LEGACY_EMBEDDING_MODEL: &str = "gemini-embedding-001";

/// Matches at/above this cosine similarity are treated as the same fact
/// restated (or a direct contradiction) and supersede the old fact.
/// Measured against gemini-embedding-2 @512: phrased contradictions ("moved to
/// Jeddah recently" vs "lives in Riyadh") score ~0.81, genuinely different
/// facts ("is a teacher" vs "lives in Riyadh") score ~0.62 — 0.78 sits between.
pub const DEFAULT_USER_FACT_SUPERSEDE_THRESHOLD: f32 = 0.78;
/// Matches at/above this similarity are deactivated on a forget request.
/// Measured against gemini-embedding-2 @512: an Arabic forget phrase vs its
/// English fact scores ~0.78 — 0.75 passes it with margin.
pub const DEFAULT_FORGET_SIMILARITY_THRESHOLD: f32 = 0.75;

/// Fields that only take effect after a process restart (the corresponding
/// clients are constructed once at boot). API keys are NOT here: the key pool
/// is hot-swapped live.
pub const RESTART_REQUIRED_FIELDS: &[&str] = &[
    "qdrant_url",
    "embedding_model",
    "embedding_dimensions",
];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct BotConfig {
    pub admin: AdminConfig,
    pub overrides: ConfigOverrides,
    /// Bookkeeping written by the bot itself (not user-editable).
    pub state: BotState,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct BotState {
    /// Embedding model whose vectors are currently in Qdrant. Used to detect
    /// model changes and auto-wipe the (incompatible) vector memory. `None`
    /// means "pre-migration" and is treated as `LEGACY_EMBEDDING_MODEL`.
    pub last_embedding_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AdminConfig {
    /// argon2 hash of the admin panel password (never the password itself).
    pub password_hash: String,
    /// True until the default password has been replaced; the panel blocks
    /// everything except the password-change endpoint while set.
    pub must_change: bool,
    pub bind: String,
    pub port: u16,
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            password_hash: String::new(),
            must_change: true,
            bind: "0.0.0.0".to_string(),
            port: 1330,
        }
    }
}

/// Optional overrides written by the admin panel. `None` = fall through to
/// the env var (or built-in default).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ConfigOverrides {
    // Hot-applied (take effect immediately).
    pub user_facts_limit: Option<u64>,
    pub app_knowledge_limit: Option<u64>,
    pub app_knowledge_min_score: Option<f32>,
    pub user_fact_supersede_threshold: Option<f32>,
    pub forget_similarity_threshold: Option<f32>,
    pub fact_extraction_enabled: Option<bool>,
    pub context_depth_limit: Option<usize>,
    // Tool calling (hot-applied).
    /// Master switch for the Gemini tool loop (web_fetch, user lookup, ...).
    pub tools_enabled: Option<bool>,
    /// Max tool-execution rounds per reply (each round is one API call).
    pub max_tool_rounds: Option<usize>,
    /// Built-in Gemini URL context tool: the model auto-fetches http(s) URLs
    /// from the conversation instead of calling web_fetch for them.
    pub url_context_enabled: Option<bool>,
    /// Built-in Google Search grounding tool (replaces the custom web_search
    /// when on). Billed per executed search query past the free allowance —
    /// default OFF.
    pub search_grounding_enabled: Option<bool>,
    /// Gaming mode: the bot hosts text games (hangman, 20 questions, ...)
    /// when users ask to play. Default ON.
    pub games_enabled: Option<bool>,
    /// Max bytes web_fetch downloads per URL.
    pub web_fetch_max_bytes: Option<usize>,
    /// Per-request timeout for web_fetch, in seconds.
    pub web_fetch_timeout_secs: Option<u64>,
    /// How many posts a user-profile scan looks at.
    pub user_scan_posts_limit: Option<u64>,
    /// Max NEW facts auto-saved from one profile scan.
    pub user_scan_fact_cap: Option<usize>,
    /// Mentions shorter than this skip background fact extraction (0 = extract all).
    pub extraction_min_chars: Option<usize>,
    /// Max key-failover attempts per reply flow (each arm re-runs the flow).
    pub max_flow_attempts: Option<usize>,
    // Hot-applied.
    /// Gemini API key pool (round-robin). Legacy single `gemini_api_key`
    /// migrates into a pool of one.
    pub gemini_api_keys: Vec<String>,
    /// Chat model for replies (hot-applied, no restart needed).
    pub generation_model: Option<String>,
    /// Saturation fallback for the chat model: one whole-flow arm runs on it
    /// when the primary exhausts its transient retries (503 storms). None or
    /// empty = disabled. Hot-applied, no restart needed.
    pub fallback_generation_model: Option<String>,
    /// Cheaper model for fact extraction/FAQ jobs; None = use generation_model.
    pub extraction_model: Option<String>,
    /// Gemini 3.x thinking level: "minimal"|"low"|"medium"|"high", or None for
    /// the model's own default (hot-applied, no restart needed).
    pub thinking_level: Option<String>,
    /// Thinking level for extraction/FAQ/rewrite jobs (hot-applied). None =
    /// fall through to env, then DEFAULT_EXTRACTION_THINKING_LEVEL.
    pub extraction_thinking_level: Option<String>,
    // Restart-required (clients built at boot).
    /// Legacy single-key field, kept for backward compatibility with older
    /// config files; folded into `gemini_api_keys` on load/save.
    pub gemini_api_key: Option<String>,
    pub qdrant_url: Option<String>,
    pub embedding_model: Option<String>,
    pub embedding_dimensions: Option<u64>,
}

/// override > env `GENERATION_MODEL` > default.
pub fn resolve_generation_model(overrides: &ConfigOverrides) -> String {
    overrides
        .generation_model
        .clone()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::env::var("GENERATION_MODEL").ok().filter(|s| !s.trim().is_empty()))
        .unwrap_or_else(|| DEFAULT_GENERATION_MODEL.to_string())
}

/// override > env `FALLBACK_GENERATION_MODEL` > None (disabled). A value
/// equal to the primary model is treated as disabled at use time.
pub fn resolve_fallback_generation_model(overrides: &ConfigOverrides) -> Option<String> {
    overrides
        .fallback_generation_model
        .clone()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            std::env::var("FALLBACK_GENERATION_MODEL")
                .ok()
                .filter(|s| !s.trim().is_empty())
        })
}

/// override > env `THINKING_LEVEL` > None (model default). Invalid values are
/// ignored rather than propagated into requests.
pub fn resolve_thinking_level(overrides: &ConfigOverrides) -> Option<String> {
    let valid = |s: &String| THINKING_LEVELS.contains(&s.trim()).then(|| s.trim().to_string());
    overrides
        .thinking_level
        .as_ref()
        .and_then(valid)
        .or_else(|| std::env::var("THINKING_LEVEL").ok().as_ref().and_then(valid))
}

/// override > env `EXTRACTION_THINKING_LEVEL` > DEFAULT_EXTRACTION_THINKING_LEVEL.
/// Invalid values are ignored rather than propagated into requests. Always
/// Some under normal operation (the built-in default is "low").
pub fn resolve_extraction_thinking_level(overrides: &ConfigOverrides) -> Option<String> {
    let valid = |s: &String| THINKING_LEVELS.contains(&s.trim()).then(|| s.trim().to_string());
    overrides
        .extraction_thinking_level
        .as_ref()
        .and_then(valid)
        .or_else(|| std::env::var("EXTRACTION_THINKING_LEVEL").ok().as_ref().and_then(valid))
        .or_else(|| Some(DEFAULT_EXTRACTION_THINKING_LEVEL.to_string()))
}

/// Map a MEDIA_RESOLUTION value ("low"/"medium"/"high", full enum names
/// accepted) to the API enum value. None = invalid.
fn map_media_resolution(v: &str) -> Option<&'static str> {
    match v.trim().to_ascii_lowercase().as_str() {
        "low" | "media_resolution_low" => Some("MEDIA_RESOLUTION_LOW"),
        "medium" | "media_resolution_medium" => Some("MEDIA_RESOLUTION_MEDIUM"),
        "high" | "media_resolution_high" => Some("MEDIA_RESOLUTION_HIGH"),
        _ => None,
    }
}

/// env `MEDIA_RESOLUTION` ("low"|"medium"|"high", full enum names accepted)
/// > None (model default). Boot-time only; invalid values are ignored.
pub fn resolve_media_resolution() -> Option<String> {
    let raw = std::env::var("MEDIA_RESOLUTION").ok()?;
    let v = raw.trim();
    if v.is_empty() {
        return None;
    }
    match map_media_resolution(v) {
        Some(m) => Some(m.to_string()),
        None => {
            warn!("Ignoring invalid MEDIA_RESOLUTION {v:?} (expected low|medium|high)");
            None
        }
    }
}

/// override > env `EXTRACTION_MODEL` > None (None = use the generation model).
pub fn resolve_extraction_model(overrides: &ConfigOverrides) -> Option<String> {
    overrides
        .extraction_model
        .clone()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::env::var("EXTRACTION_MODEL").ok().filter(|s| !s.trim().is_empty()))
}

/// override > env `EMBEDDING_MODEL` > default.
pub fn resolve_embedding_model(overrides: &ConfigOverrides) -> String {
    overrides
        .embedding_model
        .clone()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::env::var("EMBEDDING_MODEL").ok().filter(|s| !s.trim().is_empty()))
        .unwrap_or_else(|| DEFAULT_EMBEDDING_MODEL.to_string())
}

/// override > env `EMBEDDING_DIMENSIONS` > default.
pub fn resolve_embedding_dimensions(overrides: &ConfigOverrides) -> u64 {
    overrides
        .embedding_dimensions
        .or_else(|| env_u64("EMBEDDING_DIMENSIONS"))
        .unwrap_or(DEFAULT_EMBEDDING_DIMENSIONS as u64)
}

/// override > env `QDRANT_URL` > default.
pub fn resolve_qdrant_url(overrides: &ConfigOverrides) -> String {
    overrides
        .qdrant_url
        .clone()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::env::var("QDRANT_URL").ok().filter(|s| !s.trim().is_empty()))
        .unwrap_or_else(|| DEFAULT_QDRANT_URL.to_string())
}

/// Resolve the effective Gemini API key pool:
/// file keys > env `GEMINI_API_KEYS` (CSV) > legacy file key > env key.
pub fn resolve_gemini_keys(overrides: &ConfigOverrides) -> Vec<String> {
    if !overrides.gemini_api_keys.is_empty() {
        return overrides.gemini_api_keys.clone();
    }
    if let Ok(csv) = std::env::var("GEMINI_API_KEYS") {
        let keys: Vec<String> = csv
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if !keys.is_empty() {
            return keys;
        }
    }
    if let Some(k) = &overrides.gemini_api_key {
        if !k.trim().is_empty() {
            return vec![k.trim().to_string()];
        }
    }
    if let Ok(k) = std::env::var("GEMINI_API_KEY") {
        if !k.trim().is_empty() {
            return vec![k.trim().to_string()];
        }
    }
    vec![]
}

/// Knobs controlling the three memory tiers.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub user_facts_limit: u64,
    pub app_knowledge_limit: u64,
    pub app_knowledge_min_score: f32,
    pub user_fact_supersede_threshold: f32,
    pub forget_similarity_threshold: f32,
    pub fact_extraction_enabled: bool,
    /// Mentions shorter than this skip background fact extraction (0 = extract all).
    pub extraction_min_chars: usize,
}

/// The live, hot-reloadable configuration (swapped atomically on panel save).
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub memory: MemoryConfig,
    pub context_depth_limit: usize,
    pub tools: ToolsConfig,
}

/// Tool-calling knobs.
#[derive(Debug, Clone)]
pub struct ToolsConfig {
    pub enabled: bool,
    pub max_rounds: usize,
    pub web_fetch_max_bytes: usize,
    pub web_fetch_timeout_secs: u64,
    pub user_scan_posts_limit: u64,
    pub user_scan_fact_cap: usize,
    pub url_context_enabled: bool,
    /// Built-in Google Search grounding: the model searches the web
    /// server-side in the same turn. When on, the custom web_search tool is
    /// not declared. Default false (bills per executed query past the free
    /// allowance).
    pub search_grounding_enabled: bool,
    /// Gaming mode: host text games when users ask to play (manage_game tool
    /// + per-thread game state). Default true.
    pub games_enabled: bool,
    /// Max key-failover attempts per reply flow (each arm re-runs the flow).
    pub max_flow_attempts: usize,
}

fn env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}
fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}
fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

impl RuntimeConfig {
    /// Merge overrides > env > defaults into the effective runtime config.
    pub fn resolve(overrides: &ConfigOverrides) -> Self {
        let memory = MemoryConfig {
            user_facts_limit: overrides
                .user_facts_limit
                .or_else(|| env_u64("USER_FACTS_LIMIT"))
                .unwrap_or(8),
            app_knowledge_limit: overrides
                .app_knowledge_limit
                .or_else(|| env_u64("APP_KNOWLEDGE_LIMIT"))
                .unwrap_or(3),
            // Measured against gemini-embedding-2 @512 with live support
            // FAQs: genuine app questions score >=0.75 (relevant FAQ facts
            // >=0.68), unrelated chatter <=0.61 — 0.65 separates both sides
            // with margin. (The previous 0.72 was tuned on
            // gemini-embedding-001 and gates out legitimate cross-lingual
            // FAQ matches.)
            app_knowledge_min_score: overrides
                .app_knowledge_min_score
                .or_else(|| env_f32("APP_KNOWLEDGE_MIN_SCORE"))
                .unwrap_or(0.65),
            user_fact_supersede_threshold: overrides
                .user_fact_supersede_threshold
                .or_else(|| env_f32("USER_FACT_SUPERSEDE_THRESHOLD"))
                .unwrap_or(DEFAULT_USER_FACT_SUPERSEDE_THRESHOLD),
            forget_similarity_threshold: overrides
                .forget_similarity_threshold
                .or_else(|| env_f32("FORGET_SIMILARITY_THRESHOLD"))
                .unwrap_or(DEFAULT_FORGET_SIMILARITY_THRESHOLD),
            fact_extraction_enabled: overrides
                .fact_extraction_enabled
                .unwrap_or_else(|| {
                    std::env::var("FACT_EXTRACTION_ENABLED")
                        .map(|v| !matches!(v.as_str(), "0" | "false" | "no" | "off"))
                        .unwrap_or(true)
                }),
            extraction_min_chars: overrides
                .extraction_min_chars
                .or_else(|| env_usize("EXTRACTION_MIN_CHARS"))
                .unwrap_or(24)
                .clamp(0, 500),
        };
        Self {
            memory,
            context_depth_limit: overrides
                .context_depth_limit
                .or_else(|| env_usize("CONTEXT_DEPTH_LIMIT"))
                .unwrap_or(20),
            tools: ToolsConfig {
                enabled: overrides.tools_enabled.unwrap_or_else(|| {
                    std::env::var("TOOLS_ENABLED")
                        .map(|v| !matches!(v.as_str(), "0" | "false" | "no" | "off"))
                        .unwrap_or(true)
                }),
                max_rounds: overrides
                    .max_tool_rounds
                    .or_else(|| env_usize("MAX_TOOL_ROUNDS"))
                    .unwrap_or(6)
                    .clamp(1, 8),
                web_fetch_max_bytes: overrides
                    .web_fetch_max_bytes
                    .or_else(|| env_usize("WEB_FETCH_MAX_BYTES"))
                    .unwrap_or(512_000)
                    .clamp(16_384, 2_000_000),
                web_fetch_timeout_secs: overrides
                    .web_fetch_timeout_secs
                    .or_else(|| env_u64("WEB_FETCH_TIMEOUT_SECS"))
                    .unwrap_or(15)
                    .clamp(3, 60),
                user_scan_posts_limit: overrides
                    .user_scan_posts_limit
                    .or_else(|| env_u64("USER_SCAN_POSTS_LIMIT"))
                    .unwrap_or(10)
                    .clamp(1, 30),
                user_scan_fact_cap: overrides
                    .user_scan_fact_cap
                    .or_else(|| env_usize("USER_SCAN_FACT_CAP"))
                    .unwrap_or(3),
                url_context_enabled: overrides.url_context_enabled.unwrap_or_else(|| {
                    std::env::var("URL_CONTEXT_ENABLED")
                        .map(|v| !matches!(v.as_str(), "0" | "false" | "no" | "off"))
                        .unwrap_or(true)
                }),
                // Opt-in: OFF unless explicitly enabled (bills per query).
                search_grounding_enabled: overrides.search_grounding_enabled.unwrap_or_else(|| {
                    std::env::var("SEARCH_GROUNDING_ENABLED")
                        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
                        .unwrap_or(false)
                }),
                games_enabled: overrides.games_enabled.unwrap_or_else(|| {
                    std::env::var("GAMES_ENABLED")
                        .map(|v| !matches!(v.as_str(), "0" | "false" | "no" | "off"))
                        .unwrap_or(true)
                }),
                max_flow_attempts: overrides
                    .max_flow_attempts
                    .or_else(|| env_usize("MAX_FLOW_ATTEMPTS"))
                    .unwrap_or(3)
                    .clamp(1, 8),
            },
        }
    }
}

/// Load `bot_config.json` (missing/corrupt -> defaults) and apply env
/// overrides for the admin bind address.
pub fn load() -> BotConfig {
    let mut cfg = match std::fs::read_to_string(CONFIG_FILE) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_else(|e| {
            warn!("Failed to parse {CONFIG_FILE}: {e}; starting with defaults");
            BotConfig::default()
        }),
        Err(e) => {
            if e.kind() != std::io::ErrorKind::NotFound {
                warn!("Failed to read {CONFIG_FILE}: {e}; starting with defaults");
            }
            BotConfig::default()
        }
    };
    if let Some(port) = env_u64("ADMIN_PORT") {
        cfg.admin.port = port as u16;
    }
    if let Ok(bind) = std::env::var("ADMIN_BIND") {
        if !bind.trim().is_empty() {
            cfg.admin.bind = bind;
        }
    }
    cfg
}

/// Persist the configuration atomically with owner-only permissions.
pub fn save(cfg: &BotConfig) -> Result<()> {
    let content = serde_json::to_string_pretty(cfg)?;
    let tmp = format!("{CONFIG_FILE}.tmp");
    std::fs::write(&tmp, content).with_context(|| format!("Failed to write {tmp}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600)).ok();
    }
    std::fs::rename(&tmp, CONFIG_FILE).with_context(|| format!("Failed to rename {tmp}"))?;
    Ok(())
}

/// Ensure a password hash exists (first boot: hash the default `CHANGEME`).
pub fn ensure_password_hash(cfg: &mut BotConfig) -> Result<()> {
    if !cfg.admin.password_hash.is_empty() {
        return Ok(());
    }
    cfg.admin.password_hash = hash_password(DEFAULT_ADMIN_PASSWORD)?;
    cfg.admin.must_change = true;
    save(cfg)?;
    info!("Initialized admin password to the default; change it on first login");
    Ok(())
}

pub fn hash_password(password: &str) -> Result<String> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| anyhow::anyhow!("Failed to hash password: {e}"))
}

pub fn verify_password(password: &str, hash: &str) -> bool {
    PasswordHash::new(hash)
        .ok()
        .and_then(|parsed| Argon2::default().verify_password(password.as_bytes(), &parsed).ok())
        .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_beat_defaults() {
        let overrides = ConfigOverrides {
            user_facts_limit: Some(12),
            fact_extraction_enabled: Some(false),
            ..Default::default()
        };
        let cfg = RuntimeConfig::resolve(&overrides);
        assert_eq!(cfg.memory.user_facts_limit, 12);
        assert!(!cfg.memory.fact_extraction_enabled);
        // Untouched fields fall through to env or default.
        assert_eq!(cfg.memory.app_knowledge_limit, 3);
        assert_eq!(cfg.context_depth_limit, 20);
    }

    #[test]
    fn model_resolvers_use_defaults_and_overrides() {
        let empty = ConfigOverrides::default();
        assert_eq!(resolve_generation_model(&empty), DEFAULT_GENERATION_MODEL);
        assert_eq!(resolve_embedding_model(&empty), DEFAULT_EMBEDDING_MODEL);
        assert_eq!(resolve_embedding_dimensions(&empty), DEFAULT_EMBEDDING_DIMENSIONS as u64);
        assert_eq!(resolve_qdrant_url(&empty), DEFAULT_QDRANT_URL);

        let with = ConfigOverrides {
            generation_model: Some("gemini-3.5-flash-lite".to_string()),
            ..Default::default()
        };
        assert_eq!(resolve_generation_model(&with), "gemini-3.5-flash-lite");
        // Blank override falls through to default.
        let blank = ConfigOverrides {
            generation_model: Some("   ".to_string()),
            ..Default::default()
        };
        assert_eq!(resolve_generation_model(&blank), DEFAULT_GENERATION_MODEL);
    }

    #[test]
    fn fallback_model_resolver() {
        let empty = ConfigOverrides::default();
        assert_eq!(resolve_fallback_generation_model(&empty), None, "unset = disabled");
        let with = ConfigOverrides {
            fallback_generation_model: Some("gemini-3.6-flash".to_string()),
            ..Default::default()
        };
        assert_eq!(
            resolve_fallback_generation_model(&with).as_deref(),
            Some("gemini-3.6-flash")
        );
        let blank = ConfigOverrides {
            fallback_generation_model: Some("   ".to_string()),
            ..Default::default()
        };
        assert_eq!(resolve_fallback_generation_model(&blank), None, "blank = disabled");
    }

    #[test]
    fn extraction_model_resolver_defaults_to_none() {
        let empty = ConfigOverrides::default();
        assert_eq!(resolve_extraction_model(&empty), None, "unset = same as generation model");
        let with = ConfigOverrides {
            extraction_model: Some("gemini-3.5-flash-lite".to_string()),
            ..Default::default()
        };
        assert_eq!(
            resolve_extraction_model(&with),
            Some("gemini-3.5-flash-lite".to_string())
        );
        let blank = ConfigOverrides {
            extraction_model: Some("   ".to_string()),
            ..Default::default()
        };
        assert_eq!(resolve_extraction_model(&blank), None);
    }

    #[test]
    fn cost_knobs_resolve_with_defaults_and_clamps() {
        let cfg = RuntimeConfig::resolve(&ConfigOverrides::default());
        assert_eq!(cfg.memory.extraction_min_chars, 24);
        assert_eq!(cfg.tools.max_flow_attempts, 3);
        let with = ConfigOverrides {
            extraction_min_chars: Some(0),
            max_flow_attempts: Some(99),
            ..Default::default()
        };
        let cfg = RuntimeConfig::resolve(&with);
        assert_eq!(cfg.memory.extraction_min_chars, 0, "0 = extract everything");
        assert_eq!(cfg.tools.max_flow_attempts, 8, "clamped to the 1-8 range");
    }

    #[test]
    fn thinking_level_resolver() {
        let empty = ConfigOverrides::default();
        assert_eq!(resolve_thinking_level(&empty), None, "untouched = model default");
        let with = ConfigOverrides {
            thinking_level: Some("low".to_string()),
            ..Default::default()
        };
        assert_eq!(resolve_thinking_level(&with), Some("low".to_string()));
        // Invalid values are ignored, never propagated.
        let bogus = ConfigOverrides {
            thinking_level: Some("very-hard".to_string()),
            ..Default::default()
        };
        assert_eq!(resolve_thinking_level(&bogus), None);
    }

    #[test]
    fn extraction_thinking_level_resolver_defaults_to_low() {
        let empty = ConfigOverrides::default();
        assert_eq!(
            resolve_extraction_thinking_level(&empty),
            Some(DEFAULT_EXTRACTION_THINKING_LEVEL.to_string()),
            "untouched = the built-in low (Thinking docs: minimal/low for classification)"
        );
        let with = ConfigOverrides {
            extraction_thinking_level: Some("minimal".to_string()),
            ..Default::default()
        };
        assert_eq!(resolve_extraction_thinking_level(&with), Some("minimal".to_string()));
        // Invalid override falls back to the built-in default, never propagated.
        let bogus = ConfigOverrides {
            extraction_thinking_level: Some("very-hard".to_string()),
            ..Default::default()
        };
        assert_eq!(
            resolve_extraction_thinking_level(&bogus),
            Some(DEFAULT_EXTRACTION_THINKING_LEVEL.to_string())
        );
    }

    #[test]
    fn media_resolution_mapping() {
        assert_eq!(map_media_resolution("low"), Some("MEDIA_RESOLUTION_LOW"));
        assert_eq!(map_media_resolution("MEDIUM"), Some("MEDIA_RESOLUTION_MEDIUM"));
        assert_eq!(map_media_resolution(" media_resolution_high "), Some("MEDIA_RESOLUTION_HIGH"));
        assert_eq!(map_media_resolution("ultra"), None);
        assert_eq!(map_media_resolution(""), None);
    }

    #[test]
    fn bot_state_defaults_to_no_marker() {
        let cfg = BotConfig::default();
        assert!(cfg.state.last_embedding_model.is_none());
        // Older config files without the section still parse.
        let parsed: BotConfig = serde_json::from_str(r#"{"admin":{"password_hash":"x"}}"#).unwrap();
        assert!(parsed.state.last_embedding_model.is_none());
    }

    #[test]
    fn password_hash_roundtrip() {
        let hash = hash_password("s3cret!").unwrap();
        assert!(verify_password("s3cret!", &hash));
        assert!(!verify_password("wrong", &hash));
        assert!(!verify_password("CHANGEME", &hash));
        // Garbage hashes never panic.
        assert!(!verify_password("s3cret!", "not-a-hash"));
    }

    #[test]
    fn config_file_roundtrip() {
        let dir = std::env::temp_dir().join(format!("askme-cfg-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(CONFIG_FILE);
        let cfg = BotConfig {
            admin: AdminConfig {
                password_hash: "hash".to_string(),
                must_change: false,
                bind: "127.0.0.1".to_string(),
                port: 9999,
            },
            overrides: ConfigOverrides {
                user_facts_limit: Some(5),
                ..Default::default()
            },
            state: BotState::default(),
        };
        std::fs::write(&path, serde_json::to_string_pretty(&cfg).unwrap()).unwrap();
        let loaded: BotConfig =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(loaded.admin.port, 9999);
        assert!(!loaded.admin.must_change);
        assert_eq!(loaded.overrides.user_facts_limit, Some(5));
        std::fs::remove_dir_all(&dir).ok();
    }
}
