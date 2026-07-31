use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::Client;
use tracing::info;

use crate::models::*;

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com";
const MODEL: &str = "gemini-3.5-flash-lite";
const POLL_INTERVAL_MS: u64 = 1000;
const MAX_POLL_ATTEMPTS: u32 = 60;

#[derive(Clone)]
pub struct GeminiClient {
    client: Client,
    api_key: String,
}

impl GeminiClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }

    pub async fn upload_file(&self, data: Vec<u8>, mime_type: &str, display_name: &str) -> Result<String> {
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
        body.extend_from_slice(&serde_json::to_vec(&metadata)?);
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(format!("Content-Type: {mime_type}\r\n\r\n").as_bytes());
        body.extend_from_slice(&data);
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

        let url = format!("{GEMINI_API_BASE}/upload/v1beta/files");
        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("X-Goog-Upload-Protocol", "multipart")
            .header("Content-Type", format!("multipart/related; boundary={boundary}"))
            .body(body)
            .send()
            .await
            .context("Failed to upload file to Gemini Files API")?;

        let status = resp.status();
        let bytes = resp.bytes().await?;

        if !status.is_success() {
            let text = String::from_utf8_lossy(&bytes);
            anyhow::bail!("Gemini Files API error (HTTP {status}): {text}");
        }

        let file_resp: GeminiFileResponse = serde_json::from_slice(&bytes)
            .context("Failed to parse Gemini Files API response")?;

        let file_name = file_resp.file.name.clone();
        let file_uri = file_resp.file.uri.clone();

        info!("Uploaded file to Gemini: {file_name} -> {file_uri} (state: {})", file_resp.file.state);

        if file_resp.file.state == "ACTIVE" {
            return Ok(file_uri);
        }

        self.poll_file_until_ready(&file_name).await?;

        Ok(file_uri)
    }

    async fn poll_file_until_ready(&self, file_name: &str) -> Result<()> {
        let url = format!("{GEMINI_API_BASE}/v1beta/{file_name}");

        for attempt in 0..MAX_POLL_ATTEMPTS {
            tokio::time::sleep(Duration::from_millis(POLL_INTERVAL_MS)).await;

            let resp = self
                .client
                .get(&url)
                .header("x-goog-api-key", &self.api_key)
                .send()
                .await
                .context("Failed to poll Gemini file state")?;

            let status = resp.status();
            let bytes = resp.bytes().await?;

            if !status.is_success() {
                let text = String::from_utf8_lossy(&bytes);
                anyhow::bail!("Gemini file state poll error (HTTP {status}): {text}");
            }

            let file_resp: GeminiFileStateResponse = serde_json::from_slice(&bytes)
                .context("Failed to parse Gemini file state response")?;

            match file_resp.state.as_str() {
                "ACTIVE" => {
                    info!("File {file_name} is ACTIVE after {attempt}s");
                    return Ok(());
                }
                "FAILED" => {
                    let err = file_resp.error.as_ref()
                        .map(|e| e.message.as_deref().unwrap_or("unknown error"))
                        .unwrap_or("unknown error");
                    anyhow::bail!("Gemini file processing FAILED: {err}");
                }
                _ => {
                    info!("File {file_name} state: {} (attempt {}/{MAX_POLL_ATTEMPTS})", file_resp.state, attempt + 1);
                }
            }
        }

        anyhow::bail!("Timed out waiting for Gemini file {file_name} to become ACTIVE")
    }

    pub async fn generate_content(
        &self,
        system_prompt: &str,
        user_text: &str,
        file_uris: &[(String, String)],
    ) -> Result<String> {
        let mut parts: Vec<Part> = Vec::new();

        parts.push(Part::Text { text: user_text.to_string() });

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
                }],
            }),
            contents: vec![Content {
                role: "user".to_string(),
                parts,
            }],
        };

        let url = format!("{GEMINI_API_BASE}/v1beta/models/{MODEL}:generateContent");

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send generateContent request")?;

        let status = resp.status();
        let bytes = resp.bytes().await?;

        if !status.is_success() {
            let text = String::from_utf8_lossy(&bytes);
            if status.as_u16() == 429 {
                return Err(anyhow::anyhow!("Gemini API rate limited (HTTP 429): {text}"));
            }
            if status.as_u16() == 400 {
                if let Ok(api_err) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    anyhow::bail!("Gemini API bad request (HTTP 400): {}", api_err);
                }
            }
            anyhow::bail!("Gemini API error (HTTP {status}): {text}");
        }

        let response: GenerateContentResponse = serde_json::from_slice(&bytes)
            .context("Failed to parse Gemini generateContent response")?;

        let text = response
            .candidates
            .clone()
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content)
            .and_then(|c| c.parts)
            .and_then(|p| p.into_iter().next())
            .and_then(|p| p.text)
            .unwrap_or_default();

        if text.is_empty() {
            let reason = response.candidates
                .as_ref()
                .and_then(|c| c.first())
                .and_then(|c| c.finishReason.as_deref())
                .unwrap_or("unknown");
            info!("Gemini returned empty response, finishReason: {reason}");
        }

        Ok(text)
    }
}
