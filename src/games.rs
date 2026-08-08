//! Gaming mode: per-conversation game state and all-time per-user scores.
//!
//! `game_states.json` (working dir) is the durable record, written atomically
//! like `support_faqs.json`. One active game per conversation thread; a game
//! idle longer than `GAME_IDLE_EXPIRY_SECS` is pruned as abandoned (Things
//! threads expire within hours anyway). Secrets — the word/answer the bot is
//! "thinking of" — live ONLY in this store, never in the posted transcript.
//!
//! The model drives everything through the `manage_game` tool (start / update
//! / end / score); the rules of each game live in SYSTEM_PROMPT.md. This file
//! only validates game names and keeps the state honest.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::warn;

/// Per-instance game store (working directory). Instance data, not secrets —
/// gitignored like `bot_config.json`.
pub const GAMES_FILE: &str = "game_states.json";

/// A game with no activity for this long is treated as abandoned.
pub const GAME_IDLE_EXPIRY_SECS: i64 = 6 * 3600;

/// The game catalog: canonical key -> Arabic display name. The system prompt
/// teaches the model the rules; the catalog only validates names and renders
/// the active-game prompt section.
pub const GAME_CATALOG: &[(&str, &str)] = &[
    ("categories", "إنسان حيوان جماد"),
    ("word_chain", "سلسلة الكلمات"),
    ("taboo", "تابو"),
    ("twenty_questions", "20 سؤال"),
    ("riddles", "ألغاز وأحاجي"),
    ("guess_the_figure", "من القائل / خمّن الشخصية"),
    ("choose_adventure", "اختر مغامرتك"),
    ("story_chain", "أكمل القصة"),
    ("hangman", "المشنوق"),
    ("emoji_guess", "خمّن من الإيموجي"),
    ("two_truths", "حقيقتان وكذبة"),
    ("would_you_rather", "لو خيروك"),
    ("trivia", "مسابقة الثقافة"),
    ("true_false", "صح أم خطأ"),
    ("tic_tac_toe", "إكس أو"),
];

/// The Arabic display name for a catalog key, if valid.
pub fn game_display_name(key: &str) -> Option<&'static str> {
    GAME_CATALOG
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, name)| *name)
}

/// Comma-separated valid game keys, for tool-error messages.
pub fn valid_game_names() -> String {
    GAME_CATALOG
        .iter()
        .map(|(k, _)| *k)
        .collect::<Vec<_>>()
        .join(", ")
}

/// One live game in one conversation thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    /// Catalog key (e.g. "hangman").
    pub game: String,
    /// Username of the opponent (the person playing against the bot).
    pub player: String,
    /// The hidden answer, when the game has one. NEVER shown in the
    /// transcript; the prompt section marks it as secret.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<String>,
    #[serde(default)]
    pub turn: u32,
    /// Game-specific state blob owned by the model (guessed letters,
    /// questions left, board grid, story chapter, score in the match, ...).
    #[serde(default)]
    pub data: serde_json::Value,
    pub started_at: i64,
    pub updated_at: i64,
}

/// How a finished game went for the PLAYER (win = the player beat the bot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameOutcome {
    Win,
    Loss,
    Draw,
}

impl GameOutcome {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "win" => Some(Self::Win),
            "loss" => Some(Self::Loss),
            "draw" => Some(Self::Draw),
            _ => None,
        }
    }
}

/// Per-game tally inside a user's all-time record.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GameRecord {
    pub plays: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

impl GameRecord {
    fn credit(&mut self, outcome: GameOutcome) {
        self.plays += 1;
        match outcome {
            GameOutcome::Win => self.wins += 1,
            GameOutcome::Loss => self.losses += 1,
            GameOutcome::Draw => self.draws += 1,
        }
    }

    /// "3W/1L/0D" — compact form for the prompt section.
    fn compact(&self) -> String {
        format!("{}W/{}L/{}D", self.wins, self.losses, self.draws)
    }
}

/// A user's all-time record across all games.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct UserScore {
    pub plays: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    #[serde(default)]
    pub per_game: HashMap<String, GameRecord>,
}

/// The on-disk shape.
#[derive(Debug, Default, Serialize, Deserialize)]
struct GamesFile {
    #[serde(default)]
    active: HashMap<u64, GameState>,
    #[serde(default)]
    scores: HashMap<String, UserScore>,
}

/// The live store: active games by conversation id + all-time scores by
/// (lowercased) username. `path: None` = in-memory only (tests) — saves are
/// no-ops so the test suite never touches the real file.
#[derive(Debug, Default)]
pub struct GameStore {
    active: HashMap<u64, GameState>,
    scores: HashMap<String, UserScore>,
    path: Option<String>,
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

impl GameStore {
    /// Load the store (missing/corrupt -> empty, like config::load).
    pub fn load() -> Self {
        let file: GamesFile = match std::fs::read_to_string(GAMES_FILE) {
            Ok(content) => serde_json::from_str(&content).unwrap_or_else(|e| {
                warn!("Failed to parse {GAMES_FILE}: {e}; starting with an empty game store");
                GamesFile::default()
            }),
            Err(_) => GamesFile::default(),
        };
        Self {
            active: file.active,
            scores: file.scores,
            path: Some(GAMES_FILE.to_string()),
        }
    }

    /// Persist the store atomically with owner-only permissions. In-memory
    /// stores (path None, used by tests) save as no-ops.
    pub fn save(&self) -> Result<()> {
        let Some(path) = &self.path else { return Ok(()) };
        let file = GamesFile {
            active: self.active.clone(),
            scores: self.scores.clone(),
        };
        let content = serde_json::to_string_pretty(&file)?;
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, content).with_context(|| format!("Failed to write {tmp}"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600)).ok();
        }
        std::fs::rename(&tmp, path).with_context(|| format!("Failed to rename {tmp}"))?;
        Ok(())
    }

    fn save_warn(&self, what: &str) {
        if let Err(e) = self.save() {
            warn!("Failed to save game store after {what}: {e}");
        }
    }

    /// The active game in a thread, pruning it first if it went idle.
    pub fn active_game(&mut self, conversation_id: u64) -> Option<GameState> {
        let expired = self
            .active
            .get(&conversation_id)
            .is_some_and(|g| now() - g.updated_at > GAME_IDLE_EXPIRY_SECS);
        if expired {
            self.active.remove(&conversation_id);
            self.save_warn("idle expiry");
            return None;
        }
        self.active.get(&conversation_id).cloned()
    }

    /// True when a (live) game is active in the thread — cheap check for the
    /// extraction guard, no pruning side effects beyond the same expiry rule.
    pub fn has_active_game(&mut self, conversation_id: u64) -> bool {
        self.active_game(conversation_id).is_some()
    }

    /// Start (or replace) the game in a thread.
    pub fn start_game(
        &mut self,
        conversation_id: u64,
        game: &str,
        player: &str,
        secret: Option<String>,
        data: serde_json::Value,
    ) -> GameState {
        let ts = now();
        let state = GameState {
            game: game.to_string(),
            player: player.to_string(),
            secret,
            turn: 1,
            data,
            started_at: ts,
            updated_at: ts,
        };
        self.active.insert(conversation_id, state.clone());
        self.save_warn("game start");
        state
    }

    /// Apply a move: provided fields overwrite; an absent turn increments.
    /// None when no game is active in the thread.
    pub fn update_game(
        &mut self,
        conversation_id: u64,
        secret: Option<String>,
        turn: Option<u32>,
        data: Option<serde_json::Value>,
    ) -> Option<GameState> {
        let state = self.active.get_mut(&conversation_id)?;
        if let Some(secret) = secret {
            state.secret = Some(secret);
        }
        state.turn = turn.unwrap_or(state.turn + 1);
        if let Some(data) = data {
            state.data = data;
        }
        state.updated_at = now();
        let state = state.clone();
        self.save_warn("game update");
        Some(state)
    }

    /// End the thread's game and credit the player's all-time record.
    /// Returns the removed game (None when nothing was active).
    pub fn end_game(&mut self, conversation_id: u64, outcome: GameOutcome) -> Option<GameState> {
        let state = self.active.remove(&conversation_id)?;
        let key = state.player.to_lowercase();
        let score = self.scores.entry(key).or_default();
        score.plays += 1;
        match outcome {
            GameOutcome::Win => score.wins += 1,
            GameOutcome::Loss => score.losses += 1,
            GameOutcome::Draw => score.draws += 1,
        }
        score
            .per_game
            .entry(state.game.clone())
            .or_default()
            .credit(outcome);
        self.save_warn("game end");
        Some(state)
    }

    /// A user's all-time record (case-insensitive username).
    pub fn score(&self, username: &str) -> Option<UserScore> {
        self.scores.get(&username.to_lowercase()).cloned()
    }

    /// The `[Active game: ...]` prompt section for a thread (None when no
    /// live game). Compact: the full rules live in the system prompt; this
    /// just pins the state so the model continues coherently across turns.
    pub fn prompt_section(&mut self, conversation_id: u64) -> Option<String> {
        let game = self.active_game(conversation_id)?;
        let name = game_display_name(&game.game).unwrap_or(&game.game);
        let record = self
            .score(&game.player)
            .map(|s| {
                format!(
                    "all-time {} (this game: {})",
                    GameRecord {
                        plays: s.plays,
                        wins: s.wins,
                        losses: s.losses,
                        draws: s.draws,
                    }
                    .compact(),
                    s.per_game
                        .get(&game.game)
                        .map(|r| r.compact())
                        .unwrap_or_else(|| "0W/0L/0D".to_string())
                )
            })
            .unwrap_or_else(|| "first recorded game".to_string());
        let secret = game
            .secret
            .as_ref()
            .map(|s| format!(" Secret: \"{s}\" — NEVER reveal it until the game ends."))
            .unwrap_or_default();
        Some(format!(
            "\n[Active game: {name} ({}) — turn {}. Player: @{} ({record}).{secret} State: {}]",
            game.game,
            game.turn,
            game.player,
            serde_json::to_string(&game.data).unwrap_or_else(|_| "{}".to_string())
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> GameStore {
        GameStore::default()
    }

    #[test]
    fn catalog_names_resolve() {
        assert_eq!(game_display_name("hangman"), Some("المشنوق"));
        assert_eq!(game_display_name("nope"), None);
        assert!(valid_game_names().contains("tic_tac_toe"));
        assert_eq!(GAME_CATALOG.len(), 15);
    }

    #[test]
    fn start_update_end_cycle_credits_scores() {
        let mut s = store();
        let st = s.start_game(7, "taboo", "Sara", Some("قهوة".to_string()), serde_json::json!({"clues": 0}));
        assert_eq!(st.turn, 1);
        assert_eq!(st.player, "Sara");

        let st = s
            .update_game(7, None, None, Some(serde_json::json!({"clues": 1})))
            .expect("game active");
        assert_eq!(st.turn, 2, "absent turn increments");
        assert_eq!(st.secret.as_deref(), Some("قهوة"), "secret survives an update that doesn't touch it");

        let st = s
            .update_game(7, None, Some(9), None)
            .expect("game active");
        assert_eq!(st.turn, 9, "explicit turn wins");

        let ended = s.end_game(7, GameOutcome::Win).expect("game was active");
        assert_eq!(ended.game, "taboo");
        assert!(s.active_game(7).is_none(), "ended game is gone");

        let score = s.score("sara").expect("score credited (case-insensitive)");
        assert_eq!((score.plays, score.wins, score.losses, score.draws), (1, 1, 0, 0));
        assert_eq!(score.per_game["taboo"].wins, 1);
    }

    #[test]
    fn end_without_active_game_is_none() {
        let mut s = store();
        assert!(s.end_game(42, GameOutcome::Loss).is_none());
        assert!(s.score("ghost").is_none());
    }

    #[test]
    fn idle_game_expires() {
        let mut s = store();
        let mut st = s.start_game(1, "riddles", "mo", None, serde_json::json!({}));
        st.updated_at = now() - GAME_IDLE_EXPIRY_SECS - 10;
        s.active.insert(1, st);
        assert!(s.active_game(1).is_none(), "stale game pruned");
        assert!(!s.has_active_game(1));
    }

    #[test]
    fn prompt_section_marks_secret_and_record() {
        let mut s = store();
        // A finished hangman game credits the player's record...
        s.start_game(3, "hangman", "Khaled", Some("شمس".to_string()), serde_json::json!({"wrong": 1}));
        s.end_game(3, GameOutcome::Loss);
        // ...so the NEXT game's prompt section carries the all-time record.
        s.start_game(3, "hangman", "Khaled", Some("قمر".to_string()), serde_json::json!({"wrong": 0}));
        let section = s.prompt_section(3).expect("active game section");
        assert!(section.contains("المشنوق"));
        assert!(section.contains("قمر"), "current secret shown to the model");
        assert!(section.contains("NEVER reveal"));
        assert!(section.contains("0W/1L/0D"), "all-time record from the earlier loss");
        assert!(section.contains("\"wrong\":0"));
        assert!(s.prompt_section(99).is_none(), "no game in another thread");
    }

    #[test]
    fn outcome_parsing() {
        assert_eq!(GameOutcome::parse("win"), Some(GameOutcome::Win));
        assert_eq!(GameOutcome::parse(" LOSS "), Some(GameOutcome::Loss));
        assert_eq!(GameOutcome::parse("Draw"), Some(GameOutcome::Draw));
        assert_eq!(GameOutcome::parse("whatever"), None);
    }
}
