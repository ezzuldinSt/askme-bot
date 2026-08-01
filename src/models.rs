#![allow(non_snake_case, dead_code)]

use serde::{Deserialize, Serialize};

// ── Things API Auth ──

#[derive(Debug, Deserialize)]
pub struct LoginResponse {
    pub message: Option<String>,
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct VerifyOtpResponse {
    pub data: Option<VerifyOtpData>,
    pub token: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct VerifyOtpData {
    pub authToken: Option<AuthToken>,
}

#[derive(Debug, Deserialize)]
pub struct AuthToken {
    pub token: String,
}

// ── Things API Notifications ──

#[derive(Debug, Deserialize)]
pub struct UnreadCountResponse {
    pub count: Option<u64>,
    pub data: Option<UnreadCountData>,
}

#[derive(Debug, Deserialize)]
pub struct UnreadCountData {
    pub count: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct NotificationsEnvelope {
    pub data: Option<Vec<Notification>>,
    pub notifications: Option<Vec<Notification>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Notification {
    pub id: u64,
    #[serde(rename = "type")]
    pub notification_type: Option<String>,
    pub group: Option<String>,
    pub body: Option<String>,
    pub post_data: Option<Post>,
    pub original_post_data: Option<Post>,
    pub reply_post_data: Option<Post>,
    pub is_read: Option<bool>,
    pub created_at: Option<String>,
    pub action_url: Option<String>,
}

// ── Things API Posts ──

#[derive(Debug, Deserialize)]
pub struct PostEnvelope {
    pub data: Option<PostData>,
}

#[derive(Debug, Deserialize)]
pub struct PostData {
    pub post: Option<Post>,
    pub parent: Option<Post>,
    pub quoted: Option<Post>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Post {
    pub id: Option<u64>,
    pub post_id: Option<u64>,
    pub user: Option<User>,
    pub parent_id: Option<u64>,
    pub post_comment: Option<String>,
    pub media: Option<Vec<MediaItem>>,
    pub audio: Option<serde_json::Value>,
    pub post_type: Option<String>,
    pub created_at: Option<String>,
    pub content: Option<String>,
    pub comments: Option<String>,
    pub images: Option<Vec<MediaItem>>,
    pub image: Option<MediaItem>,
    pub attachments: Option<Vec<MediaItem>>,
    pub entities: Option<Vec<PostEntity>>,
}

impl Post {
    pub fn content_text(&self) -> &str {
        self.post_comment
            .as_deref()
            .or(self.comments.as_deref())
            .or(self.content.as_deref())
            .unwrap_or("")
    }

    pub fn author_username(&self) -> &str {
        self.user
            .as_ref()
            .and_then(|u| u.username.as_deref())
            .unwrap_or("someone")
    }

    pub fn id_value(&self) -> Option<u64> {
        self.id.or(self.post_id)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct User {
    pub id: Option<u64>,
    pub name: Option<String>,
    pub username: Option<String>,
    pub profile_pic_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MediaItem {
    #[serde(alias = "path", alias = "src", alias = "file_url")]
    pub url: Option<String>,
    pub id: Option<u64>,
    #[serde(rename = "type")]
    pub media_type: Option<String>,
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostEntity {
    #[serde(rename = "type")]
    pub entity_type: String,
    pub offset: u64,
    pub length: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_size_value: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ReplyResponse {
    pub id: Option<u64>,
    pub post_id: Option<u64>,
    pub data: Option<ReplyData>,
}

#[derive(Debug, Deserialize)]
pub struct ReplyData {
    pub id: Option<u64>,
    pub post_id: Option<u64>,
}

// ── Things API Errors ──

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub error: Option<String>,
    pub message: Option<String>,
}

// ── Gemini Files API ──

#[derive(Debug, Deserialize)]
pub struct GeminiFileResponse {
    pub file: GeminiFile,
}

#[derive(Debug, Deserialize)]
pub struct GeminiFile {
    pub name: String,
    pub uri: String,
    pub state: String,
    pub mimeType: Option<String>,
    pub error: Option<GeminiFileError>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiFileError {
    pub code: Option<i32>,
    pub message: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiFileStateResponse {
    pub state: String,
    pub error: Option<GeminiFileError>,
}

// ── Gemini GenerateContent API ──

#[derive(Debug, Serialize)]
pub struct GenerateContentRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    pub contents: Vec<Content>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

#[derive(Debug, Serialize)]
pub struct GenerationConfig {
    #[serde(rename = "responseMimeType", skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

/// Gemini 3.x reasoning effort control (see ai.google.dev/gemini-api/docs/thinking).
#[derive(Debug, Serialize)]
pub struct ThinkingConfig {
    #[serde(rename = "thinkingLevel")]
    pub thinking_level: String,
}

#[derive(Debug, Serialize)]
pub struct SystemInstruction {
    pub parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
pub struct Content {
    pub role: String,
    pub parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Part {
    Text { text: String },
    FileData { file_data: FileData },
    InlineData { inline_data: InlineData },
}

#[derive(Debug, Serialize)]
pub struct FileData {
    pub mime_type: String,
    pub file_uri: String,
}

#[derive(Debug, Serialize)]
pub struct InlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Deserialize)]
pub struct GenerateContentResponse {
    pub candidates: Option<Vec<Candidate>>,
    pub usageMetadata: Option<UsageMetadata>,
    pub promptFeedback: Option<PromptFeedback>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Candidate {
    pub content: Option<ContentResponse>,
    pub finishReason: Option<String>,
    pub safetyRatings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentResponse {
    pub parts: Option<Vec<PartResponse>>,
    pub role: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PartResponse {
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UsageMetadata {
    pub promptTokenCount: Option<i32>,
    pub candidatesTokenCount: Option<i32>,
    pub totalTokenCount: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PromptFeedback {
    pub blockReason: Option<String>,
    pub safetyRatings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SafetyRating {
    pub category: String,
    pub probability: String,
}

// ── Gemini Embeddings API ──

#[derive(Debug, Serialize)]
pub struct EmbedContentRequest {
    pub model: String,
    pub content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub taskType: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputDimensionality: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct BatchEmbedContentsRequest {
    pub requests: Vec<EmbedContentRequest>,
}

#[derive(Debug, Deserialize)]
pub struct BatchEmbedContentsResponse {
    pub embeddings: Option<Vec<ContentEmbedding>>,
    pub usageMetadata: Option<UsageMetadata>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentEmbedding {
    pub values: Vec<f32>,
}
