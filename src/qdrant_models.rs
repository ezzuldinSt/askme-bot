#![allow(dead_code)]

use qdrant_client::qdrant::RetrievedPoint;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use uuid::Uuid;

/// Name of the Qdrant collection that stores per-conversation episodic memory.
///
/// Tier 1 memory: every post/reply the bot sees, scoped by `conversation_id`.
/// Context is ONLY ever read through a `conversation_id` filter, so one
/// conversation can never bleed into another.
pub const COLLECTION_NAME: &str = "conversation_memory";

/// Name of the Qdrant collection that stores processed-notification markers.
///
/// Kept separate from conversation memory because notification ids and post ids
/// come from different server-side id spaces: sharing one point-id space would
/// let a marker silently overwrite a conversation point with the same number.
pub const PROCESSED_COLLECTION_NAME: &str = "processed_notifications";

/// Name of the Qdrant collection that stores durable per-user facts.
///
/// Tier 2 memory: the ONLY memory that crosses conversations, and it carries
/// facts about a user — never the content of other conversations. Every read
/// is filtered by `username`, so one user's profile never leaks into another
/// user's context.
pub const USER_PROFILES_COLLECTION_NAME: &str = "user_profiles";

/// Name of the Qdrant collection that stores global knowledge about the
/// Things app itself (tier 3). Retrieved only through a score-gated semantic
/// search, and only `status = active` facts are ever injected into prompts.
pub const THINGS_KNOWLEDGE_COLLECTION_NAME: &str = "things_knowledge";

/// Kind of message stored in the conversation memory collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    /// A user's root post (mention of the bot or otherwise).
    Post,
    /// A user's reply within a thread.
    Reply,
    /// A reply posted by the bot (username is always `AskMe`).
    BotReply,
}

impl MessageType {
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageType::Post => "post",
            MessageType::Reply => "reply",
            MessageType::BotReply => "bot_reply",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "post" => Some(MessageType::Post),
            "reply" => Some(MessageType::Reply),
            "bot_reply" => Some(MessageType::BotReply),
            _ => None,
        }
    }
}

/// Payload stored on every point in the conversation memory collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePayload {
    /// Unique post/reply id (matches the Qdrant point id).
    pub id: u64,
    /// Full text of the post/reply.
    pub content: String,
    /// Author's username (always `AskMe` for bot replies).
    pub username: String,
    pub message_type: MessageType,
    /// Parent post id; absent for root posts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<u64>,
    /// Id of the post where this bot conversation started (the first @mention).
    /// The isolation boundary: context is only ever read per `conversation_id`.
    pub conversation_id: u64,
    /// Unix timestamp (seconds) of the message.
    pub timestamp: i64,
    /// Media URLs attached to the message.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media_urls: Vec<String>,
}

/// A decoded memory entry used when building conversation context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: u64,
    pub content: String,
    pub username: String,
    pub message_type: MessageType,
    pub parent_id: Option<u64>,
    pub conversation_id: u64,
    pub timestamp: i64,
    /// Absent from the stored payload when empty (`skip_serializing_if`).
    #[serde(default)]
    pub media_urls: Vec<String>,
}

impl From<&MessagePayload> for MemoryEntry {
    fn from(p: &MessagePayload) -> Self {
        MemoryEntry {
            id: p.id,
            content: p.content.clone(),
            username: p.username.clone(),
            message_type: p.message_type,
            parent_id: p.parent_id,
            conversation_id: p.conversation_id,
            timestamp: p.timestamp,
            media_urls: p.media_urls.clone(),
        }
    }
}

impl MemoryEntry {
    /// Decode a retrieved Qdrant point into a `MemoryEntry`.
    pub fn from_retrieved_point(point: &RetrievedPoint) -> Option<Self> {
        decode_payload(&point.payload)
    }

    /// Decode a scored Qdrant point (semantic search result) into a `MemoryEntry`.
    pub fn from_scored_point(point: &qdrant_client::qdrant::ScoredPoint) -> Option<Self> {
        decode_payload(&point.payload)
    }
}

/// Decode a Qdrant payload map into any deserializable payload type.
pub fn decode_payload<T: serde::de::DeserializeOwned>(
    payload: &std::collections::HashMap<String, qdrant_client::qdrant::Value>,
) -> Option<T> {
    if payload.is_empty() {
        return None;
    }
    let mut map = serde_json::Map::new();
    for (key, value) in payload {
        map.insert(key.clone(), value.clone().into());
    }
    serde_json::from_value(JsonValue::Object(map)).ok()
}

// ── Tier 2: per-user durable facts ──

/// Coarse category of a durable user fact, as produced by the extractor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactCategory {
    Identity,
    Location,
    Occupation,
    Preference,
    Opinion,
    Other,
}

impl FactCategory {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "identity" => Some(FactCategory::Identity),
            "location" => Some(FactCategory::Location),
            "occupation" => Some(FactCategory::Occupation),
            "preference" => Some(FactCategory::Preference),
            "opinion" => Some(FactCategory::Opinion),
            "other" => Some(FactCategory::Other),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            FactCategory::Identity => "identity",
            FactCategory::Location => "location",
            FactCategory::Occupation => "occupation",
            FactCategory::Preference => "preference",
            FactCategory::Opinion => "opinion",
            FactCategory::Other => "other",
        }
    }
}

/// Payload stored on every point in the user-profiles collection.
///
/// One point = one durable fact about one user. The point id is a UUIDv5 of
/// `(username, normalized fact)`, so restating the same fact updates the same
/// point instead of duplicating it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFactPayload {
    /// The user this fact is about (keyword index; EVERY read filters on it).
    pub username: String,
    /// One short third-person sentence, e.g. "lives in Riyadh".
    pub fact: String,
    #[serde(default = "default_fact_category")]
    pub category: FactCategory,
    /// Where the fact was stated (for audit/debugging).
    pub source_post_id: u64,
    pub source_conversation_id: u64,
    pub first_seen: i64,
    pub last_seen: i64,
    /// How many times the user has restated this fact.
    #[serde(default = "default_times_confirmed")]
    pub times_confirmed: u32,
    /// Inactive facts (forgotten or superseded) are never injected into prompts.
    #[serde(default = "default_active")]
    pub active: bool,
    /// Point id (UUID string) of the fact that replaced this one, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>,
}

fn default_fact_category() -> FactCategory {
    FactCategory::Other
}
fn default_times_confirmed() -> u32 {
    1
}
fn default_active() -> bool {
    true
}

/// Deterministic point id for a user fact: restating the same fact yields the
/// same UUID, so duplicates update in place.
pub fn user_fact_point_id(username: &str, fact: &str) -> Uuid {
    Uuid::new_v5(
        &Uuid::NAMESPACE_OID,
        format!("askme:user:{}:{}", normalize_for_id(username), normalize_for_id(fact)).as_bytes(),
    )
}

/// Deterministic point id for an app-knowledge fact.
pub fn app_fact_point_id(fact: &str) -> Uuid {
    Uuid::new_v5(
        &Uuid::NAMESPACE_OID,
        format!("askme:app:{}", normalize_for_id(fact)).as_bytes(),
    )
}

/// Lowercase + collapse whitespace so trivially-different restatements of a
/// fact map onto the same deterministic point id.
fn normalize_for_id(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

// ── Tier 3: Things app knowledge ──

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppFactSource {
    /// From the curated `things_knowledge.json` seed file (authoritative).
    Seed,
    /// Stated by a user in conversation (never authoritative).
    User,
    /// Extracted from an admin-entered support FAQ (authoritative how-to).
    Faq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppFactStatus {
    /// May be injected into prompts.
    Active,
    /// Stored but NEVER injected (user-stated facts until promoted).
    Pending,
}

impl AppFactStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            AppFactStatus::Active => "active",
            AppFactStatus::Pending => "pending",
        }
    }
}

/// Payload stored on every point in the things-knowledge collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppFactPayload {
    pub topic: String,
    pub fact: String,
    pub source: AppFactSource,
    pub status: AppFactStatus,
    pub updated_at: i64,
}

/// One entry of the curated `things_knowledge.json` seed file.
#[derive(Debug, Clone, Deserialize)]
pub struct AppKnowledgeSeed {
    pub topic: String,
    pub fact: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_type_roundtrip() {
        for (ty, s) in [
            (MessageType::Post, "post"),
            (MessageType::Reply, "reply"),
            (MessageType::BotReply, "bot_reply"),
        ] {
            assert_eq!(ty.as_str(), s);
            assert_eq!(MessageType::parse(s), Some(ty));
        }
        assert_eq!(MessageType::parse("notification"), None);
        assert_eq!(MessageType::parse("unknown"), None);
    }

    #[test]
    fn payload_serde_roundtrip() {
        let payload = MessagePayload {
            id: 42,
            content: "hello world".to_string(),
            username: "toast".to_string(),
            message_type: MessageType::Reply,
            parent_id: Some(7),
            conversation_id: 3,
            timestamp: 1_700_000_000,
            media_urls: vec!["https://example.com/a.jpg".to_string()],
        };
        let json = serde_json::to_value(&payload).unwrap();
        let back: MessagePayload = serde_json::from_value(json).unwrap();
        assert_eq!(back.id, 42);
        assert_eq!(back.message_type, MessageType::Reply);
        assert_eq!(back.parent_id, Some(7));
        assert_eq!(back.conversation_id, 3);
        assert_eq!(back.media_urls, payload.media_urls);
    }

    #[test]
    fn memory_entry_decodes_payload_without_media_urls() {
        let json = serde_json::json!({
            "id": 5799887,
            "content": "طيب عادي",
            "username": "toast",
            "message_type": "post",
            "parent_id": 5799876,
            "conversation_id": 5799837,
            "timestamp": 1785438790,
        });
        let entry: MemoryEntry = serde_json::from_value(json).unwrap();
        assert_eq!(entry.id, 5799887);
        assert_eq!(entry.conversation_id, 5799837);
        assert_eq!(entry.parent_id, Some(5799876));
        assert!(entry.media_urls.is_empty());
    }

    #[test]
    fn user_fact_point_id_is_deterministic_and_normalized() {
        let a = user_fact_point_id("Khalid", "Lives in Riyadh");
        let b = user_fact_point_id("khalid", "lives  in   riyadh");
        let c = user_fact_point_id("khalid", "is a teacher");
        let d = user_fact_point_id("someone_else", "lives in riyadh");
        assert_eq!(a, b, "same fact, different casing/spacing -> same id");
        assert_ne!(a, c, "different facts -> different ids");
        assert_ne!(a, d, "same fact, different users -> different ids");
    }

    #[test]
    fn app_fact_point_id_is_deterministic() {
        assert_eq!(
            app_fact_point_id("Things is a social network"),
            app_fact_point_id("things  is a social network")
        );
    }

    #[test]
    fn user_fact_payload_defaults() {
        let json = serde_json::json!({
            "username": "khalid",
            "fact": "lives in Riyadh",
            "source_post_id": 1,
            "source_conversation_id": 2,
            "first_seen": 100,
            "last_seen": 200,
        });
        let payload: UserFactPayload = serde_json::from_value(json).unwrap();
        assert_eq!(payload.category, FactCategory::Other);
        assert_eq!(payload.times_confirmed, 1);
        assert!(payload.active);
        assert!(payload.superseded_by.is_none());
    }

    #[test]
    fn app_fact_payload_roundtrip() {
        let payload = AppFactPayload {
            topic: "platform".to_string(),
            fact: "Things is a social network".to_string(),
            source: AppFactSource::Seed,
            status: AppFactStatus::Active,
            updated_at: 123,
        };
        let json = serde_json::to_value(&payload).unwrap();
        let back: AppFactPayload = serde_json::from_value(json).unwrap();
        assert_eq!(back.source, AppFactSource::Seed);
        assert_eq!(back.status, AppFactStatus::Active);
    }

    #[test]
    fn app_fact_source_faq_roundtrips() {
        assert_eq!(
            serde_json::to_value(AppFactSource::Faq).unwrap(),
            serde_json::json!("faq")
        );
        let back: AppFactSource = serde_json::from_value(serde_json::json!("faq")).unwrap();
        assert_eq!(back, AppFactSource::Faq);
        // Pre-existing points keep parsing.
        let seed: AppFactSource = serde_json::from_value(serde_json::json!("seed")).unwrap();
        assert_eq!(seed, AppFactSource::Seed);
    }
}
