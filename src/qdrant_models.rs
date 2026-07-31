#![allow(dead_code)]

use qdrant_client::qdrant::RetrievedPoint;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Name of the Qdrant collection that stores the bot's persistent memory.
pub const COLLECTION_NAME: &str = "conversation_memory";

/// Kind of message stored in the memory collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    /// A user's root post (mention of the bot or otherwise).
    Post,
    /// A user's reply within a thread.
    Reply,
    /// A reply posted by the bot (username is always `AskMe`).
    BotReply,
    /// Internal marker used to deduplicate processed notifications.
    Notification,
}

impl MessageType {
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageType::Post => "post",
            MessageType::Reply => "reply",
            MessageType::BotReply => "bot_reply",
            MessageType::Notification => "notification",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "post" => Some(MessageType::Post),
            "reply" => Some(MessageType::Reply),
            "bot_reply" => Some(MessageType::BotReply),
            "notification" => Some(MessageType::Notification),
            _ => None,
        }
    }
}

/// Payload stored on every point in the Qdrant collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePayload {
    /// Unique post/reply/notification id (matches the Qdrant point id).
    pub id: u64,
    /// Full text of the post/reply (or empty for notification markers).
    pub content: String,
    /// Author's username (always `AskMe` for bot replies).
    pub username: String,
    pub message_type: MessageType,
    /// Parent post id; absent for root posts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<u64>,
    /// Root post id of the thread this message belongs to.
    pub thread_id: u64,
    /// Unix timestamp (seconds) of the message.
    pub timestamp: i64,
    /// True once the bot has responded to this message.
    #[serde(default)]
    pub is_processed: bool,
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
    pub thread_id: u64,
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
            thread_id: p.thread_id,
            timestamp: p.timestamp,
            media_urls: p.media_urls.clone(),
        }
    }
}

impl MemoryEntry {
    /// Decode a retrieved Qdrant point into a `MemoryEntry`.
    pub fn from_retrieved_point(point: &RetrievedPoint) -> Option<Self> {
        Self::from_payload_map(&point.payload)
    }

    /// Decode a scored Qdrant point (semantic search result) into a `MemoryEntry`.
    pub fn from_scored_point(point: &qdrant_client::qdrant::ScoredPoint) -> Option<Self> {
        Self::from_payload_map(&point.payload)
    }

    fn from_payload_map(
        payload: &std::collections::HashMap<String, qdrant_client::qdrant::Value>,
    ) -> Option<Self> {
        if payload.is_empty() {
            return None;
        }
        let mut map = serde_json::Map::new();
        for (key, value) in payload {
            map.insert(key.clone(), value.clone().into());
        }
        serde_json::from_value(JsonValue::Object(map)).ok()
    }
}

/// Options that narrow a semantic search over the memory collection.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Only match messages authored by this username.
    pub username: Option<String>,
    /// Only match messages newer than this timestamp.
    pub min_timestamp: Option<i64>,
    /// Maximum number of results to return.
    pub limit: u64,
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
            (MessageType::Notification, "notification"),
        ] {
            assert_eq!(ty.as_str(), s);
            assert_eq!(MessageType::parse(s), Some(ty));
        }
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
            thread_id: 3,
            timestamp: 1_700_000_000,
            is_processed: true,
            media_urls: vec!["https://example.com/a.jpg".to_string()],
        };
        let json = serde_json::to_value(&payload).unwrap();
        let back: MessagePayload = serde_json::from_value(json).unwrap();
        assert_eq!(back.id, 42);
        assert_eq!(back.message_type, MessageType::Reply);
        assert_eq!(back.parent_id, Some(7));
        assert_eq!(back.media_urls, payload.media_urls);
        assert!(back.is_processed);
    }

    #[test]
    fn payload_serde_defaults_for_missing_fields() {
        let json = serde_json::json!({
            "id": 1,
            "content": "x",
            "username": "u",
            "message_type": "notification",
            "thread_id": 0,
            "timestamp": 123,
        });
        let payload: MessagePayload = serde_json::from_value(json).unwrap();
        assert_eq!(payload.parent_id, None);
        assert!(!payload.is_processed);
        assert!(payload.media_urls.is_empty());
    }

    #[test]
    fn memory_entry_decodes_payload_without_media_urls() {
        let json = serde_json::json!({
            "id": 5799887,
            "content": "طيب عادي",
            "username": "toast",
            "message_type": "post",
            "parent_id": 5799876,
            "thread_id": 5799837,
            "timestamp": 1785438790,
            "is_processed": false,
        });
        let entry: MemoryEntry = serde_json::from_value(json).unwrap();
        assert_eq!(entry.id, 5799887);
        assert_eq!(entry.thread_id, 5799837);
        assert_eq!(entry.parent_id, Some(5799876));
        assert!(entry.media_urls.is_empty());
    }
}
