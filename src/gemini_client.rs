use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::Client;
use tokio::time::sleep;
use tracing::{debug, info, warn};

use crate::models::*;
use crate::qdrant_client::Embedder;

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com";
const MODEL: &str = "gemini-3.5-flash-lite";
const DEFAULT_EMBEDDING_MODEL: &str = "gemini-embedding-001";
const DEFAULT_EMBEDDING_DIMENSIONS: u32 = 512;
const DEFAULT_EMBEDDING_BATCH_SIZE: usize = 10;
const POLL_INTERVAL_MS: u64 = 1000;
const MAX_POLL_ATTEMPTS: u32 = 60;
const EMBED_CACHE_MAX: usize = 2000;

#[derive(Clone)]
pub struct GeminiClient {
    client: Client,
    api_key: String,
    embedding_model: String,
    embedding_dimensions: u32,
    embedding_batch_size: usize,
    embed_cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
}

impl GeminiClient {
    pub fn new(api_key: String) -> Self {
        let embedding_model = std::env::var("EMBEDDING_MODEL")
            .unwrap_or_else(|_| DEFAULT_EMBEDDING_MODEL.to_string());
        let embedding_dimensions = std::env::var("EMBEDDING_DIMENSIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_EMBEDDING_DIMENSIONS);
        let embedding_batch_size = std::env::var("EMBEDDING_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_EMBEDDING_BATCH_SIZE);
        Self {
            client: Client::new(),
            api_key,
            embedding_model,
            embedding_dimensions,
            embedding_batch_size,
            embed_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn upload_file(
        &self,
        data: Vec<u8>,
        mime_type: &str,
        display_name: &str,
    ) -> Result<String> {
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
            .header(
                "Content-Type",
                format!("multipart/related; boundary={boundary}"),
            )
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

        let file_resp: GeminiFileResponse =
            serde_json::from_slice(&bytes).context("Failed to parse Gemini Files API response")?;

        let file_name = file_resp.file.name.clone();
        let file_uri = file_resp.file.uri.clone();

        info!(
            "Uploaded file to Gemini: {file_name} -> {file_uri} (state: {})",
            file_resp.file.state
        );

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
                    let err = file_resp
                        .error
                        .as_ref()
                        .map(|e| e.message.as_deref().unwrap_or("unknown error"))
                        .unwrap_or("unknown error");
                    anyhow::bail!("Gemini file processing FAILED: {err}");
                }
                _ => {
                    info!(
                        "File {file_name} state: {} (attempt {}/{MAX_POLL_ATTEMPTS})",
                        file_resp.state,
                        attempt + 1
                    );
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

        parts.push(Part::Text {
            text: user_text.to_string(),
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
                return Err(anyhow::anyhow!(
                    "Gemini API rate limited (HTTP 429): {text}"
                ));
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
            let reason = response
                .candidates
                .as_ref()
                .and_then(|c| c.first())
                .and_then(|c| c.finishReason.as_deref())
                .unwrap_or("unknown");
            info!("Gemini returned empty response, finishReason: {reason}");
        }

        Ok(text)
    }

    /// Embed a batch of texts into L2-normalized vectors using the embeddings API.
    ///
    /// Texts are split into batches of `EMBEDDING_BATCH_SIZE` and sent through
    /// the `batchEmbedContents` endpoint. Duplicate content is served from an
    /// in-memory cache to avoid redundant API calls.
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
            let url = format!(
                "{GEMINI_API_BASE}/v1beta/models/{}:batchEmbedContents",
                self.embedding_model
            );

            let mut fetched: Vec<Vec<f32>> = Vec::new();
            for chunk in pending_texts.chunks(self.embedding_batch_size.max(1)) {
                let requests: Vec<EmbedContentRequest> = chunk
                    .iter()
                    .map(|text| EmbedContentRequest {
                        model: format!("models/{}", self.embedding_model),
                        content: Content {
                            role: "user".to_string(),
                            parts: vec![Part::Text { text: text.clone() }],
                        },
                        taskType: Some("SEMANTIC_SIMILARITY".to_string()),
                        outputDimensionality: Some(self.embedding_dimensions),
                    })
                    .collect();

                let body = BatchEmbedContentsRequest { requests };
                let response = self
                    .client
                    .post(&url)
                    .header("x-goog-api-key", &self.api_key)
                    .header("Content-Type", "application/json")
                    .json(&body)
                    .send()
                    .await
                    .context("Failed to send batchEmbedContents request")?;

                let status = response.status();
                let bytes = response.bytes().await?;

                if !status.is_success() {
                    if status.as_u16() == 429 || status.as_u16() >= 500 {
                        warn!("Embedding API error (HTTP {status}), retrying once after backoff");
                        sleep(Duration::from_secs(2)).await;
                        let retry = self
                            .client
                            .post(&url)
                            .header("x-goog-api-key", &self.api_key)
                            .header("Content-Type", "application/json")
                            .json(&body)
                            .send()
                            .await
                            .context("Failed to retry batchEmbedContents request")?;
                        let retry_status = retry.status();
                        let retry_bytes = retry.bytes().await?;
                        if !retry_status.is_success() {
                            let text = String::from_utf8_lossy(&retry_bytes);
                            anyhow::bail!(
                                "Embedding API error (HTTP {retry_status}) on retry: {text}"
                            );
                        }
                        let parsed: BatchEmbedContentsResponse =
                            serde_json::from_slice(&retry_bytes)
                                .context("Failed to parse embedding retry response")?;
                        fetched.extend(collect_embeddings(parsed, chunk.len())?);
                        continue;
                    }
                    let text = String::from_utf8_lossy(&bytes);
                    anyhow::bail!("Embedding API error (HTTP {status}): {text}");
                }

                let parsed: BatchEmbedContentsResponse = serde_json::from_slice(&bytes)
                    .context("Failed to parse batchEmbedContents response")?;
                fetched.extend(collect_embeddings(parsed, chunk.len())?);
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

#[async_trait::async_trait]
impl Embedder for GeminiClient {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_texts_inner(texts).await
    }
}
