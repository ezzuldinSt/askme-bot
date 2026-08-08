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

// ── Things API Users ──

#[derive(Debug, Clone, Deserialize)]
pub struct UsersEnvelope {
    pub data: Option<Vec<UserSearchRow>>,
}

/// One row of `GET /users?search=...` — already includes the bio, so a
/// username search doubles as a lightweight profile lookup.
#[derive(Debug, Clone, Deserialize)]
pub struct UserSearchRow {
    pub id: u64,
    pub username: Option<String>,
    pub name: Option<String>,
    pub bio: Option<String>,
    pub is_private: Option<bool>,
    pub is_verified: Option<bool>,
    pub is_premium: Option<bool>,
    pub streak: Option<u64>,
}

/// Full profile from `GET /user/{id}`.
#[derive(Debug, Clone, Deserialize)]
pub struct UserProfile {
    pub id: u64,
    pub username: Option<String>,
    pub name: Option<String>,
    pub bio: Option<String>,
    pub joined_at: Option<String>,
    pub is_private: Option<bool>,
    pub is_verified: Option<bool>,
    pub is_premium: Option<bool>,
    pub streak: Option<u64>,
    #[serde(rename = "sticky_status")]
    pub sticky_status: Option<String>,
}

/// Cursor-paginated page of `GET /user/{id}/posts`.
#[derive(Debug, Clone, Deserialize)]
pub struct UserPostsPage {
    pub data: Option<Vec<UserPostRow>>,
    pub next_cursor: Option<String>,
    #[serde(rename = "has_more")]
    pub has_more: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UserPostRow {
    pub id: u64,
    #[serde(default)]
    pub post_comment: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub comments: Option<String>,
    pub user: Option<User>,
    pub parent_id: Option<u64>,
    pub post_type: Option<String>,
    pub created_at: Option<String>,
}

impl UserPostRow {
    pub fn content_text(&self) -> &str {
        self.post_comment
            .as_deref()
            .or(self.content.as_deref())
            .or(self.comments.as_deref())
            .unwrap_or("")
    }
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
    /// Music shared on the post (Apple Music card). The audio lives at
    /// `preview_url` (a ~30s AAC preview on Apple's public CDN).
    pub music: Option<Vec<MusicItem>>,
    pub post_type: Option<String>,
    pub created_at: Option<String>,
    pub expires_at: Option<String>,
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

/// One music track attached to a post (Things sends an Apple Music card).
#[derive(Debug, Clone, Deserialize)]
pub struct MusicItem {
    pub id: Option<u64>,
    pub title: Option<String>,
    pub artist: Option<String>,
    #[serde(rename = "albumName")]
    pub album_name: Option<String>,
    #[serde(rename = "artworkURL")]
    pub artwork_url: Option<String>,
    /// ~30s AAC preview on Apple's public CDN (audio-ssl.itunes.apple.com) —
    /// no auth needed, downloadable like any other media URL.
    #[serde(rename = "previewURL")]
    pub preview_url: Option<String>,
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
    /// Language tag for `code_block` entities (e.g. "rust", "arduino").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
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

// ── Gemini Tool Calling ──

/// A function call requested by the model (response) or echoed back into
/// history (request). `args` is the JSON argument object. `id` links the call
/// to its response and MUST be echoed back verbatim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallData {
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// The result of executing a model-requested function call, sent back as a
/// user-role part so the model can reason over the outcome. Echoes the call's
/// `id` when the API provided one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponseData {
    pub name: String,
    pub response: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// A server-side built-in tool invocation (url_context, google_search, ...)
/// surfaced when `toolConfig.includeServerSideToolInvocations` is set. The
/// server executes the tool itself; the app only circulates the parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallData {
    #[serde(rename = "toolType")]
    pub tool_type: String,
    #[serde(default)]
    pub args: serde_json::Value,
    pub id: Option<String>,
}

/// The server's result for a server-side tool invocation, paired with its
/// `toolCall` by `id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResponseData {
    #[serde(rename = "toolType")]
    pub tool_type: String,
    #[serde(default)]
    pub response: serde_json::Value,
    pub id: Option<String>,
}

/// One function the model may call (see ai.google.dev/gemini-api/docs/
/// function-calling). `parameters` is a JSON Schema object.
#[derive(Debug, Clone, Serialize)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    #[serde(rename = "functionDeclarations", skip_serializing_if = "Vec::is_empty")]
    pub function_declarations: Vec<FunctionDeclaration>,
    /// The built-in URL context tool: the model auto-fetches http(s) URLs
    /// present in the conversation. Takes no arguments — URLs come from the
    /// prompt. Serializes as `{"urlContext": {}}` when set.
    #[serde(rename = "urlContext", skip_serializing_if = "Option::is_none")]
    pub url_context: Option<serde_json::Value>,
    /// The built-in Google Search grounding tool: the model searches the web
    /// server-side and answers with grounding metadata (queries + source
    /// chunks) in the same turn. Serializes as `{"googleSearch": {}}` when set.
    #[serde(rename = "googleSearch", skip_serializing_if = "Option::is_none")]
    pub google_search: Option<serde_json::Value>,
}

impl Tool {
    pub fn url_context() -> Self {
        Self {
            function_declarations: Vec::new(),
            url_context: Some(serde_json::json!({})),
            google_search: None,
        }
    }

    pub fn google_search() -> Self {
        Self {
            function_declarations: Vec::new(),
            url_context: None,
            google_search: Some(serde_json::json!({})),
        }
    }
}

// ── Gemini GenerateContent API ──

#[derive(Debug, Serialize)]
pub struct GenerateContentRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

/// `includeServerSideToolInvocations` is REQUIRED whenever built-in tools
/// (url_context, ...) are combined with function declarations: it surfaces the
/// server-side `toolCall`/`toolResponse` parts so the app can circulate them.
#[derive(Debug, Serialize)]
pub struct ToolConfig {
    #[serde(rename = "includeServerSideToolInvocations")]
    pub include_server_side_tool_invocations: bool,
}

#[derive(Debug, Serialize)]
pub struct GenerationConfig {
    #[serde(rename = "responseMimeType", skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    /// JSON Schema constraining the response (requires `responseMimeType` =
    /// "application/json"). Guarantees field names/types instead of relying
    /// on the model to honor a prose-described shape.
    #[serde(rename = "responseSchema", skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<serde_json::Value>,
    /// Global media resolution for every media part in the request
    /// ("MEDIA_RESOLUTION_LOW" / ..._MEDIUM / ..._HIGH). Per-part overrides
    /// are v1alpha-only, so the global config is the v1beta way.
    #[serde(rename = "mediaResolution", skip_serializing_if = "Option::is_none")]
    pub media_resolution: Option<String>,
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

#[derive(Debug, Clone, Serialize)]
pub struct Content {
    pub role: String,
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Part {
    Text {
        text: String,
        /// Gemini 3 models may attach a thought signature to ANY part type
        /// (including text — sometimes an empty text part). It must be echoed
        /// back verbatim when the turn is circulated, or the API rejects the
        /// next request in a function-calling flow.
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FileData { file_data: FileData },
    InlineData { inline_data: InlineData },
    /// Echo of a model functionCall back into history. Thinking models attach
    /// a `thoughtSignature` that MUST be echoed verbatim on the way back.
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: FunctionCallData,
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponseData,
    },
    /// Server-side built-in tool invocation, circulated back into history
    /// verbatim (with its thought signature).
    ToolCall {
        #[serde(rename = "toolCall")]
        tool_call: ToolCallData,
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    /// Server-side tool result, circulated back into history verbatim
    /// (tool responses carry their own thought signature).
    ToolResponse {
        #[serde(rename = "toolResponse")]
        tool_response: ToolResponseData,
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct FileData {
    pub mime_type: String,
    pub file_uri: String,
}

#[derive(Debug, Clone, Serialize)]
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
    /// Which URLs the URL context tool retrieved this turn, with per-URL status.
    #[serde(rename = "urlContextMetadata")]
    pub url_context_metadata: Option<UrlContextMetadata>,
    /// Google Search grounding metadata (present when the built-in
    /// google_search tool ran this turn).
    #[serde(rename = "groundingMetadata")]
    pub grounding_metadata: Option<GroundingMetadata>,
}

/// Google Search grounding metadata. Only the fields this bot reads are
/// modeled: the executed queries (cost visibility — Gemini 3 bills per
/// query) and the source chunks (their titles feed the Sources footer).
/// `searchEntryPoint`/`groundingSupports` are deliberately not modeled.
#[derive(Debug, Clone, Deserialize)]
pub struct GroundingMetadata {
    #[serde(rename = "webSearchQueries")]
    pub web_search_queries: Option<Vec<String>>,
    #[serde(rename = "groundingChunks")]
    pub grounding_chunks: Option<Vec<GroundingChunk>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GroundingChunk {
    pub web: Option<GroundingWeb>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GroundingWeb {
    /// A vertexaisearch.cloud.google.com REDIRECT URL, not the original
    /// link — never posted; only `title` is presentable.
    pub uri: Option<String>,
    /// The source name (e.g. "aljazeera.com") shown in the Sources footer.
    pub title: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UrlContextMetadata {
    #[serde(rename = "urlMetadata")]
    pub url_metadata: Option<Vec<UrlMetadataEntry>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UrlMetadataEntry {
    #[serde(rename = "retrievedUrl")]
    pub retrieved_url: Option<String>,
    #[serde(rename = "urlRetrievalStatus")]
    pub url_retrieval_status: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentResponse {
    pub parts: Option<Vec<PartResponse>>,
    pub role: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PartResponse {
    pub text: Option<String>,
    /// True on thought-summary parts (Thinking docs). Thought text is NOT
    /// answer text: it must never be concatenated into a reply (or a JSON
    /// payload), only circulated for its signature.
    #[serde(rename = "thought", default)]
    pub thought: Option<bool>,
    #[serde(rename = "functionCall")]
    pub function_call: Option<FunctionCallData>,
    #[serde(rename = "functionResponse")]
    pub function_response: Option<FunctionResponseData>,
    #[serde(rename = "toolCall")]
    pub tool_call: Option<ToolCallData>,
    #[serde(rename = "toolResponse")]
    pub tool_response: Option<ToolResponseData>,
    #[serde(rename = "thoughtSignature")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UsageMetadata {
    pub promptTokenCount: Option<i32>,
    pub candidatesTokenCount: Option<i32>,
    pub totalTokenCount: Option<i32>,
    /// Billed thinking tokens (Thinking docs: output price = output + thoughts).
    pub thoughtsTokenCount: Option<i32>,
    /// Prompt tokens served from an implicit context-cache hit (caching is on
    /// by default for Gemini 2.5+; zero means no hit — e.g. below the
    /// per-model minimum prefix size).
    pub cachedContentTokenCount: Option<i32>,
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
