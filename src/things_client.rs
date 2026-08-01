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
    let text = String::from_utf8_lossy(body);
    let err = anyhow::anyhow!("{context} (HTTP {status}): {text}");
    if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
        RequestError::Retryable(err)
    } else {
        RequestError::Fatal(err)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedToken {
    token: String,
}

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

    /// Post a reply. Intentionally NOT retried: if the server commits the reply
    /// but the response is lost (timeout, dropped connection), retrying would
    /// post a duplicate — the bot's cardinal sin. A single attempt is safer.
    pub async fn reply_to_post(
        &self,
        parent_post_id: u64,
        content: &str,
        entities: &[PostEntity],
    ) -> Result<u64> {
        let form = self.build_reply_form(parent_post_id, content, entities);

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
        let resp = self
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

        let bytes = resp.bytes().await.context("Failed to read media bytes")?;

        let len = bytes.len();
        info!("Downloaded {len} bytes from {url} (type: {content_type})");

        Ok((bytes.to_vec(), content_type))
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
    ) -> Form {
        let mut form = Form::new()
            .text("location_name", "")
            .text("address", "")
            .text("coordinate", "")
            .text("region", "")
            .text("city", "")
            .text("comments", content.to_string())
            .text("post_id", parent_post_id.to_string())
            .text("post_type", "r".to_string())
            .text("post_duration", "3h".to_string())
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
