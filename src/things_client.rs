use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::multipart::Form;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::models::*;

const THINGS_API_BASE: &str = "https://things.cv/api";
pub const TOKEN_FILE: &str = ".token.json";
const MAX_RETRIES: u32 = 3;
const BASE_RETRY_DELAY_MS: u64 = 1000;
/// Largest attachment the bot will download for a reply (Gemini upload
/// staging is in-memory, so unbounded downloads are a memory hazard).
const MAX_MEDIA_DOWNLOAD_BYTES: usize = 25 * 1024 * 1024;

/// Read a response body into memory, stopping at `max` bytes. Returns the
/// (possibly truncated) body and whether truncation happened — the cap is
/// enforced DURING the download, not after, so a huge page/file can never
/// be fully buffered.
pub(crate) async fn read_body_capped(
    resp: &mut reqwest::Response,
    max: usize,
) -> Result<(Vec<u8>, bool)> {
    let declared = resp.content_length().map(|n| n as usize);
    let mut buf: Vec<u8> = Vec::with_capacity(declared.unwrap_or(0).min(max));
    let mut truncated = declared.map(|n| n > max).unwrap_or(false);
    while let Some(chunk) = resp.chunk().await.context("Failed to read response body")? {
        let remaining = max.saturating_sub(buf.len());
        if remaining == 0 {
            truncated = true;
            break;
        }
        if chunk.len() > remaining {
            buf.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            break;
        }
        buf.extend_from_slice(&chunk);
    }
    Ok((buf, truncated))
}

/// Post type used for bot replies. "b" is the code the Things app's own post
/// composer sends; the bot previously used "r" (reply). Keeping the app's code
/// makes replies behave like normal app posts. Used when the mentioned post
/// carries no post_type of its own.
const REPLY_POST_TYPE: &str = "b";
/// How long bot replies stay live before expiring (Things posts are
/// ephemeral). "1w" = one week. Used when the mentioned post carries no
/// expiry to mirror.
const REPLY_POST_DURATION: &str = "1w";

/// Derive the API's `post_duration` string from a post's created/expiry
/// timestamps, so a reply can mirror the lifetime of the post it answers.
/// Only emits units the app is known to accept ("Nh" hours, "Nw" weeks);
/// anything else (unparseable, not whole hours/weeks) yields None and the
/// caller falls back to `REPLY_POST_DURATION`.
pub fn post_duration_string(created_at: Option<&str>, expires_at: Option<&str>) -> Option<String> {
    let (Some(created_at), Some(expires_at)) = (created_at, expires_at) else {
        return None;
    };
    let created = chrono::DateTime::parse_from_rfc3339(created_at).ok()?;
    let expires = chrono::DateTime::parse_from_rfc3339(expires_at).ok()?;
    let secs = expires.signed_duration_since(created).num_seconds();
    if secs <= 0 {
        return None;
    }
    if secs % (7 * 24 * 3600) == 0 {
        return Some(format!("{}w", secs / (7 * 24 * 3600)));
    }
    if secs % 3600 == 0 {
        return Some(format!("{}h", secs / 3600));
    }
    None
}

/// Raised on HTTP 401: the cached auth token is expired or invalid and the bot
/// cannot recover on its own (login requires an interactive OTP).
#[derive(Debug, thiserror::Error)]
#[error("Things auth token expired or invalid (HTTP 401)")]
pub struct AuthExpired;

/// True if the error (or anything in its chain) is an `AuthExpired`.
pub fn is_auth_expired(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.downcast_ref::<AuthExpired>().is_some())
}

/// Raised on 4xx (non-401): the server validated and REJECTED the request,
/// so nothing was committed. Callers can downcast to this to tell a
/// definitive rejection (safe to retry with a corrected payload) apart from
/// an ambiguous failure (timeout, 5xx) where the request may have committed.
#[derive(Debug, thiserror::Error)]
#[error("{context} (HTTP {status}): {body}")]
pub struct ClientRejected {
    pub status: StatusCode,
    pub context: String,
    pub body: String,
}

/// Internal error classification so the retry loop only retries failures that
/// can actually heal on their own.
enum RequestError {
    /// Transport errors, timeouts, 429 and 5xx — worth retrying.
    Retryable(anyhow::Error),
    /// Other 4xx client errors — retrying the same request won't help.
    Fatal(anyhow::Error),
    /// 401 — surfaced to the caller as `AuthExpired`.
    AuthExpired,
}

impl From<RequestError> for anyhow::Error {
    fn from(e: RequestError) -> Self {
        match e {
            RequestError::Retryable(e) | RequestError::Fatal(e) => e,
            RequestError::AuthExpired => AuthExpired.into(),
        }
    }
}

fn transport_error(context: &'static str) -> impl Fn(reqwest::Error) -> RequestError {
    move |e| RequestError::Retryable(anyhow::Error::new(e).context(context))
}

fn http_error(status: StatusCode, context: &str, body: &[u8]) -> RequestError {
    if status == StatusCode::UNAUTHORIZED {
        return RequestError::AuthExpired;
    }
    // Error pages (404/500 HTML) can be kilobytes of markup; tool-facing
    // errors are fed back to the model, so keep only a snippet.
    let text: String = String::from_utf8_lossy(body).chars().take(500).collect();
    if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
        RequestError::Retryable(anyhow::anyhow!("{context} (HTTP {status}): {text}"))
    } else {
        RequestError::Fatal(
            ClientRejected {
                status,
                context: context.to_string(),
                body: text,
            }
            .into(),
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedToken {
    token: String,
}

#[derive(Clone)]
pub struct ThingsClient {
    client: Client,
    api_base: String,
    token: Option<String>,
}

impl ThingsClient {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            api_base: THINGS_API_BASE.to_string(),
            token: None,
        }
    }

    pub fn load_cached_token(&mut self) -> bool {
        let path = Path::new(TOKEN_FILE);
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(cached) = serde_json::from_str::<CachedToken>(&content) {
                    self.token = Some(cached.token);
                    info!("Loaded cached auth token");
                    return true;
                }
            }
        }
        false
    }

    fn save_token(&self) -> Result<()> {
        if let Some(ref token) = self.token {
            let cached = CachedToken {
                token: token.clone(),
            };
            let content = serde_json::to_string_pretty(&cached)?;
            std::fs::write(TOKEN_FILE, content)?;
            info!("Saved auth token to {TOKEN_FILE}");
        }
        Ok(())
    }

    pub async fn login(&self, email: &str, password: &str) -> Result<LoginResponse> {
        let url = format!("{}/login", self.api_base);
        let body = serde_json::json!({ "email": email, "password": password });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to send login request")?;

        let status = resp.status();
        let bytes = resp.bytes().await?;

        if !status.is_success() {
            let text = String::from_utf8_lossy(&bytes);
            anyhow::bail!("Login error (HTTP {status}): {text}");
        }

        serde_json::from_slice(&bytes).context("Failed to parse login response")
    }

    pub async fn verify_otp(&mut self, email: &str, otp: &str) -> Result<()> {
        let url = format!("{}/verify-login-otp", self.api_base);
        let body = serde_json::json!({ "email": email, "otp": otp });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to send OTP verification request")?;

        let status = resp.status();
        let bytes = resp.bytes().await?;

        if !status.is_success() {
            let text = String::from_utf8_lossy(&bytes);
            anyhow::bail!("OTP verification error (HTTP {status}): {text}");
        }

        let verify_resp: VerifyOtpResponse =
            serde_json::from_slice(&bytes).context("Failed to parse OTP verification response")?;

        let token = verify_resp
            .data
            .and_then(|d| d.authToken)
            .map(|a| a.token)
            .or(verify_resp.token)
            .ok_or_else(|| anyhow::anyhow!("No auth token in OTP verification response"))?;

        self.token = Some(token);
        self.save_token()?;

        info!("OTP verification successful, token acquired");
        Ok(())
    }

    pub async fn get_unread_count(&self) -> Result<u64> {
        self.retry(|| async {
            let url = format!("{}/notifications/unread-count", self.api_base);
            let resp = self
                .client
                .get(&url)
                .headers(self.auth_headers().map_err(RequestError::Fatal)?)
                .send()
                .await
                .map_err(transport_error("Failed to fetch unread count"))?;

            let status = resp.status();
            let bytes = resp
                .bytes()
                .await
                .map_err(transport_error("Failed to read unread count response"))?;

            if !status.is_success() {
                return Err(http_error(status, "Unread count error", &bytes));
            }

            let envelope: UnreadCountResponse = serde_json::from_slice(&bytes)
                .map_err(|e| {
                    RequestError::Fatal(
                        anyhow::Error::new(e).context("Failed to parse unread count response"),
                    )
                })?;

            Ok(envelope
                .count
                .or_else(|| envelope.data.and_then(|d| d.count))
                .unwrap_or(0))
        })
        .await
    }

    pub async fn get_notifications(&self, page: u32) -> Result<Vec<Notification>> {
        self.retry(|| async {
            let url = format!("{}/notifications?page={page}", self.api_base);
            let resp = self
                .client
                .get(&url)
                .headers(self.auth_headers().map_err(RequestError::Fatal)?)
                .send()
                .await
                .map_err(transport_error("Failed to fetch notifications"))?;

            let status = resp.status();
            let bytes = resp
                .bytes()
                .await
                .map_err(transport_error("Failed to read notifications response"))?;

            if !status.is_success() {
                return Err(http_error(status, "Notifications error", &bytes));
            }

            let envelope: NotificationsEnvelope = serde_json::from_slice(&bytes)
                .map_err(|e| {
                    RequestError::Fatal(
                        anyhow::Error::new(e).context("Failed to parse notifications response"),
                    )
                })?;

            Ok(envelope.data.or(envelope.notifications).unwrap_or_default())
        })
        .await
    }

    pub async fn get_post(&self, post_id: u64) -> Result<PostData> {
        self.retry(|| async {
            let url = format!("{}/posts/{post_id}", self.api_base);
            let resp = self
                .client
                .get(&url)
                .headers(self.auth_headers().map_err(RequestError::Fatal)?)
                .send()
                .await
                .map_err(transport_error("Failed to fetch post"))?;

            let status = resp.status();
            let bytes = resp
                .bytes()
                .await
                .map_err(transport_error("Failed to read post response"))?;

            if !status.is_success() {
                return Err(http_error(status, "Get post error", &bytes));
            }

            let envelope: PostEnvelope = serde_json::from_slice(&bytes).map_err(|e| {
                RequestError::Fatal(
                    anyhow::Error::new(e).context("Failed to parse post response"),
                )
            })?;

            envelope
                .data
                .context("Post response missing data field")
                .map_err(RequestError::Fatal)
        })
        .await
    }

    /// Search users globally (substring match on username/name). The caller
    /// filters for the exact case-insensitive username.
    pub async fn search_users(&self, query: &str, per_page: u64) -> Result<Vec<UserSearchRow>> {
        self.retry(|| async {
            let url = reqwest::Url::parse_with_params(
                &format!("{}/users", self.api_base),
                &[
                    ("search", query),
                    ("per_page", &per_page.to_string()),
                ],
            )
            .map_err(|e| {
                RequestError::Fatal(anyhow::Error::new(e).context("Failed to build users URL"))
            })?;

            let resp = self
                .client
                .get(url)
                .headers(self.auth_headers().map_err(RequestError::Fatal)?)
                .send()
                .await
                .map_err(transport_error("Failed to search users"))?;

            let status = resp.status();
            let bytes = resp
                .bytes()
                .await
                .map_err(transport_error("Failed to read users response"))?;

            if !status.is_success() {
                return Err(http_error(status, "Users search error", &bytes));
            }

            let envelope: UsersEnvelope = serde_json::from_slice(&bytes).map_err(|e| {
                RequestError::Fatal(
                    anyhow::Error::new(e).context("Failed to parse users search response"),
                )
            })?;

            Ok(envelope.data.unwrap_or_default())
        })
        .await
    }

    /// Full user profile (bio, joined_at, streak, ...) by numeric id.
    pub async fn get_user(&self, user_id: u64) -> Result<UserProfile> {
        self.retry(|| async {
            let url = format!("{}/user/{user_id}", self.api_base);
            let resp = self
                .client
                .get(&url)
                .headers(self.auth_headers().map_err(RequestError::Fatal)?)
                .send()
                .await
                .map_err(transport_error("Failed to fetch user profile"))?;

            let status = resp.status();
            let bytes = resp
                .bytes()
                .await
                .map_err(transport_error("Failed to read user profile response"))?;

            if !status.is_success() {
                return Err(http_error(status, "Get user error", &bytes));
            }

            serde_json::from_slice(&bytes).map_err(|e| {
                RequestError::Fatal(
                    anyhow::Error::new(e).context("Failed to parse user profile response"),
                )
            })
        })
        .await
    }

    /// A user's recent posts (cursor-paginated, newest first).
    pub async fn get_user_posts(&self, user_id: u64, per_page: u64) -> Result<UserPostsPage> {
        self.retry(|| async {
            let url = reqwest::Url::parse_with_params(
                &format!("{}/user/{user_id}/posts", self.api_base),
                &[("per_page", per_page.to_string())],
            )
            .map_err(|e| {
                RequestError::Fatal(anyhow::Error::new(e).context("Failed to build user posts URL"))
            })?;

            let resp = self
                .client
                .get(url)
                .headers(self.auth_headers().map_err(RequestError::Fatal)?)
                .send()
                .await
                .map_err(transport_error("Failed to fetch user posts"))?;

            let status = resp.status();
            let bytes = resp
                .bytes()
                .await
                .map_err(transport_error("Failed to read user posts response"))?;

            if !status.is_success() {
                return Err(http_error(status, "Get user posts error", &bytes));
            }

            serde_json::from_slice(&bytes).map_err(|e| {
                RequestError::Fatal(
                    anyhow::Error::new(e).context("Failed to parse user posts response"),
                )
            })
        })
        .await
    }

    /// Post a reply. Intentionally NOT retried: if the server commits the reply
    /// but the response is lost (timeout, dropped connection), retrying would
    /// post a duplicate — the bot's cardinal sin. A single attempt is safer.
    ///
    /// `post_type`/`post_duration` mirror the post being answered (None falls
    /// back to the bot's own defaults).
    pub async fn reply_to_post(
        &self,
        parent_post_id: u64,
        content: &str,
        entities: &[PostEntity],
        post_type: Option<&str>,
        post_duration: Option<&str>,
    ) -> Result<u64> {
        let form = self.build_reply_form(parent_post_id, content, entities, post_type, post_duration);

        let url = format!("{}/posts", self.api_base);
        let resp = self
            .client
            .post(&url)
            .headers(self.auth_headers()?)
            .header("Accept", "application/json")
            .multipart(form)
            .send()
            .await
            .context("Failed to send reply")?;

        let status = resp.status();
        let bytes = resp.bytes().await?;

        if !status.is_success() {
            return Err(http_error(status, "Reply error", &bytes).into());
        }

        let reply: ReplyResponse =
            serde_json::from_slice(&bytes).context("Failed to parse reply response")?;

        reply
            .id
            .or(reply.post_id)
            .or_else(|| reply.data.as_ref().and_then(|d| d.id.or(d.post_id)))
            .ok_or_else(|| anyhow::anyhow!("Reply response missing post ID"))
    }

    pub async fn mark_notifications_read(&self, ids: &[u64]) -> Result<()> {
        self.retry(|| async {
            let url = format!("{}/notifications/read", self.api_base);
            let body = serde_json::json!({ "ids": ids });

            let resp = self
                .client
                .post(&url)
                .headers(self.auth_headers().map_err(RequestError::Fatal)?)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(transport_error("Failed to mark notifications as read"))?;

            let status = resp.status();
            if !status.is_success() {
                let bytes = resp
                    .bytes()
                    .await
                    .map_err(transport_error("Failed to read mark-read response"))?;
                return Err(http_error(status, "Mark read error", &bytes));
            }

            Ok(())
        })
        .await
    }

    pub async fn download_media(&self, url: &str) -> Result<(Vec<u8>, String)> {
        let mut resp = self
            .client
            .get(url)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .context("Failed to download media")?;

        let status = resp.status();
        if status == StatusCode::UNAUTHORIZED {
            return Err(AuthExpired.into());
        }
        if !status.is_success() {
            anyhow::bail!("Media download error (HTTP {status}) for {url}");
        }

        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();

        let (bytes, truncated) = read_body_capped(&mut resp, MAX_MEDIA_DOWNLOAD_BYTES).await?;
        if truncated {
            // Partial media (e.g. half a video) is useless to the model —
            // skip this attachment entirely rather than uploading corrupt data.
            anyhow::bail!(
                "Media exceeds the {} MB cap; skipped: {url}",
                MAX_MEDIA_DOWNLOAD_BYTES / (1024 * 1024)
            );
        }

        let len = bytes.len();
        info!("Downloaded {len} bytes from {url} (type: {content_type})");

        Ok((bytes, content_type))
    }

    fn auth_headers(&self) -> Result<reqwest::header::HeaderMap> {
        let token = self
            .token
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Not authenticated"))?;
        let mut headers = reqwest::header::HeaderMap::new();
        let auth_value = format!("Bearer {token}");
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&auth_value)
                .map_err(|e| anyhow::anyhow!("Invalid header value: {e}"))?,
        );
        Ok(headers)
    }

    fn build_reply_form(
        &self,
        parent_post_id: u64,
        content: &str,
        entities: &[PostEntity],
        post_type: Option<&str>,
        post_duration: Option<&str>,
    ) -> Form {
        let mut form = Form::new()
            .text("location_name", "")
            .text("address", "")
            .text("coordinate", "")
            .text("region", "")
            .text("city", "")
            .text("comments", content.to_string())
            .text("post_id", parent_post_id.to_string())
            .text(
                "post_type",
                post_type.unwrap_or(REPLY_POST_TYPE).to_string(),
            )
            .text(
                "post_duration",
                post_duration.unwrap_or(REPLY_POST_DURATION).to_string(),
            )
            .text("allow_screenshot", "1".to_string())
            .text("is_private", "0".to_string());

        if !entities.is_empty() {
            if let Ok(json) = serde_json::to_string(entities) {
                form = form.text("entities", json);
            }
        }

        form
    }

    /// Retry only failures that can heal (transport errors, 429, 5xx).
    /// Fatal 4xx and 401s return immediately.
    async fn retry<T, F, Fut>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, RequestError>>,
    {
        let mut last_error = None;
        for attempt in 1..=MAX_RETRIES {
            match operation().await {
                Ok(val) => return Ok(val),
                Err(e @ RequestError::AuthExpired) | Err(e @ RequestError::Fatal(_)) => {
                    return Err(e.into())
                }
                Err(RequestError::Retryable(e)) => {
                    warn!("Request failed (attempt {attempt}/{MAX_RETRIES}): {e}");
                    last_error = Some(e);
                    if attempt < MAX_RETRIES {
                        let delay = BASE_RETRY_DELAY_MS * (1u64 << (attempt - 1));
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                    }
                }
            }
        }
        Err(last_error
            .unwrap_or_else(|| anyhow::anyhow!("Request failed after {MAX_RETRIES} retries")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_string_mirrors_known_units() {
        let one_week = (
            Some("2026-08-02T21:06:26.000000Z"),
            Some("2026-08-09T21:06:26.000000Z"),
        );
        assert_eq!(post_duration_string(one_week.0, one_week.1).as_deref(), Some("1w"));

        let three_hours = (
            Some("2026-08-03T18:00:00Z"),
            Some("2026-08-03T21:00:00Z"),
        );
        assert_eq!(
            post_duration_string(three_hours.0, three_hours.1).as_deref(),
            Some("3h")
        );

        let two_weeks = (
            Some("2026-08-01T00:00:00Z"),
            Some("2026-08-15T00:00:00Z"),
        );
        assert_eq!(
            post_duration_string(two_weeks.0, two_weeks.1).as_deref(),
            Some("2w")
        );
    }

    #[test]
    fn http_error_classifies_by_status() {
        // 4xx -> Fatal carrying a downcastable ClientRejected.
        let err: anyhow::Error =
            http_error(StatusCode::UNPROCESSABLE_ENTITY, "Reply error", b"{\"errors\":{}}").into();
        let rejected = err
            .chain()
            .find_map(|c| c.downcast_ref::<ClientRejected>())
            .expect("422 must surface as ClientRejected");
        assert_eq!(rejected.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(err
            .to_string()
            .contains("Reply error (HTTP 422 Unprocessable Entity)"));

        // 5xx -> Retryable, and NOT a definitive rejection.
        let err: anyhow::Error =
            http_error(StatusCode::INTERNAL_SERVER_ERROR, "Reply error", b"oops").into();
        assert!(
            err.chain()
                .all(|c| c.downcast_ref::<ClientRejected>().is_none()),
            "5xx is ambiguous — never ClientRejected"
        );

        // 429 -> Retryable (heals on its own via the retry loop).
        let err: anyhow::Error =
            http_error(StatusCode::TOO_MANY_REQUESTS, "Reply error", b"").into();
        assert!(
            err.chain()
                .all(|c| c.downcast_ref::<ClientRejected>().is_none()),
            "429 is ambiguous for reply purposes — never ClientRejected"
        );

        // 401 -> AuthExpired.
        let err: anyhow::Error = http_error(StatusCode::UNAUTHORIZED, "Reply error", b"").into();
        assert!(is_auth_expired(&err));
    }

    #[test]
    fn duration_string_rejects_unmappable_or_missing() {
        // Odd interval (90 minutes) is not a known unit -> fallback needed.
        assert_eq!(
            post_duration_string(
                Some("2026-08-03T18:00:00Z"),
                Some("2026-08-03T19:30:00Z")
            ),
            None
        );
        // Expiry before creation is nonsense.
        assert_eq!(
            post_duration_string(
                Some("2026-08-03T21:00:00Z"),
                Some("2026-08-03T18:00:00Z")
            ),
            None
        );
        // Missing timestamps -> None.
        assert_eq!(post_duration_string(None, None), None);
        assert_eq!(post_duration_string(Some("2026-08-03T18:00:00Z"), None), None);
        // Garbage dates -> None.
        assert_eq!(
            post_duration_string(Some("not-a-date"), Some("also-not-a-date")),
            None
        );
    }
}
