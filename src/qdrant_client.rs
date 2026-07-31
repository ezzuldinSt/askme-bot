#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, Distance, Filter, GetPointsBuilder, PointStruct,
    QueryPointsBuilder, ScrollPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};

use crate::qdrant_models::{
    MemoryEntry, MessagePayload, MessageType, SearchOptions, COLLECTION_NAME,
};

/// Producer of vector embeddings used to index and query message content.
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts into normalized vectors (one per input, same order).
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// Item sent to the background batch writer.
pub enum MemoryWrite {
    /// Queue a message to be embedded and persisted in the background.
    Upsert(MessagePayload),
    /// Flush any pending writes and resolve the oneshot once persisted.
    Flush(oneshot::Sender<()>),
}

/// Wrapper around a local Qdrant instance that stores the bot's persistent memory.
///
/// The client is used for all conversation memory: every post, reply and bot
/// reply is indexed with an embedding of its content, so past conversations can
/// be recalled both by thread id (structured) and by semantic similarity.
pub struct QdrantClient {
    client: Option<Arc<Qdrant>>,
    available: AtomicBool,
    embedder: Arc<dyn Embedder>,
    collection: String,
    dimensions: u64,
    search_limit: u64,
}

impl QdrantClient {
    /// Build a client for the given gRPC endpoint and check connectivity.
    ///
    /// This never fails hard: if Qdrant is unreachable the client is created in
    /// a degraded state (`is_available() == false`) and all operations no-op
    /// with an error, so the bot keeps working without persistent memory.
    pub async fn connect(
        url: &str,
        embedder: Arc<dyn Embedder>,
        dimensions: u64,
        search_limit: u64,
    ) -> Self {
        let client = match Qdrant::from_url(url).build() {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                warn!("Failed to build Qdrant client for {url}: {e}");
                None
            }
        };

        let this = Self {
            client,
            available: AtomicBool::new(false),
            embedder,
            collection: COLLECTION_NAME.to_string(),
            dimensions,
            search_limit,
        };

        if this.health_check().await {
            this.available.store(true, Ordering::Relaxed);
            info!("Connected to Qdrant at {url}");
        } else {
            warn!(
                "Qdrant unavailable at {url}; running without persistent memory. \
                 Start it with: scripts/start_qdrant.sh"
            );
        }
        this
    }

    /// Create a client that is always unavailable (used only when Qdrant is optional).
    pub fn unavailable(embedder: Arc<dyn Embedder>, dimensions: u64, search_limit: u64) -> Self {
        Self {
            client: None,
            available: AtomicBool::new(false),
            embedder,
            collection: COLLECTION_NAME.to_string(),
            dimensions,
            search_limit,
        }
    }

    /// True when the underlying Qdrant server is reachable.
    pub fn is_available(&self) -> bool {
        self.available.load(Ordering::Relaxed)
    }

    /// Name of the collection this client operates on.
    pub fn collection_name(&self) -> &str {
        &self.collection
    }

    /// Ping the server to verify connectivity.
    pub async fn health_check(&self) -> bool {
        let Some(client) = &self.client else {
            return false;
        };
        match client.health_check().await {
            Ok(_) => true,
            Err(e) => {
                warn!("Qdrant health check failed: {e}");
                false
            }
        }
    }

    /// Create the collection if it does not exist yet.
    pub async fn ensure_collection(&self) -> Result<()> {
        let client = self.client()?;
        if client.collection_exists(&self.collection).await? {
            return Ok(());
        }
        client
            .create_collection(
                CreateCollectionBuilder::new(self.collection.clone())
                    .vectors_config(VectorParamsBuilder::new(self.dimensions, Distance::Cosine)),
            )
            .await
            .context("Failed to create Qdrant collection")?;
        info!("Created Qdrant collection {}", self.collection);
        Ok(())
    }

    /// Embed a single text and return the normalized vector.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut vectors = self.embedder.embed_texts(&[text.to_string()]).await?;
        vectors
            .pop()
            .ok_or_else(|| anyhow::anyhow!("Embedder returned no vector"))
    }

    /// Store a single message, computing its embedding on the spot.
    ///
    /// Used for writes that must be immediately visible (bot replies and
    /// follow-up posts); user posts are normally queued via `spawn_writer`.
    ///
    /// Messages without content (processed-notification markers) are stored
    /// with a zero vector — the Gemini embeddings API rejects empty text.
    pub async fn upsert(&self, payload: &MessagePayload) -> Result<()> {
        let client = self.client()?;
        let vector = if payload.content.is_empty() {
            self.zero_vector()
        } else {
            self.embed(&payload.content).await?
        };
        self.upsert_points(&client, &[(payload.clone(), vector)])
            .await
    }

    /// Store a batch of messages, embedding all of them in one call.
    ///
    /// Only non-empty texts are sent to the embedder; empty-content messages
    /// (notification markers) are stored with a zero vector.
    pub async fn upsert_many(&self, items: &[MessagePayload]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let client = self.client()?;
        let embed_indices: Vec<usize> = items
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.content.is_empty())
            .map(|(i, _)| i)
            .collect();
        let texts: Vec<String> = embed_indices
            .iter()
            .map(|&i| items[i].content.clone())
            .collect();
        let vectors = if texts.is_empty() {
            Vec::new()
        } else {
            self.embedder.embed_texts(&texts).await?
        };
        let mut vector_iter = vectors.into_iter();
        let zipped: Vec<(MessagePayload, Vec<f32>)> = items
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let vector = if embed_indices.contains(&i) {
                    vector_iter.next().unwrap_or_else(|| self.zero_vector())
                } else {
                    self.zero_vector()
                };
                (p.clone(), vector)
            })
            .collect();
        self.upsert_points(&client, &zipped).await
    }

    fn zero_vector(&self) -> Vec<f32> {
        vec![0.0; self.dimensions as usize]
    }

    async fn upsert_points(
        &self,
        client: &Qdrant,
        points: &[(MessagePayload, Vec<f32>)],
    ) -> Result<()> {
        let structs: Vec<PointStruct> = points
            .iter()
            .map(|(payload, vector)| {
                let json = serde_json::to_value(payload)
                    .unwrap_or_else(|_| serde_json::json!({ "id": payload.id }));
                PointStruct::new(
                    payload.id,
                    vector.clone(),
                    Payload::try_from(json).unwrap_or_default(),
                )
            })
            .collect();
        client
            .upsert_points(UpsertPointsBuilder::new(self.collection.clone(), structs))
            .await
            .context("Failed to upsert points to Qdrant")?;
        Ok(())
    }

    /// Look up a single message by its id.
    pub async fn get_point(&self, id: u64) -> Result<Option<MemoryEntry>> {
        let client = self.client()?;
        let response = client
            .get_points(
                GetPointsBuilder::new(self.collection.clone(), vec![id.into()]).with_payload(true),
            )
            .await
            .context("Failed to get point from Qdrant")?;
        Ok(response
            .result
            .into_iter()
            .next()
            .and_then(|p| MemoryEntry::from_retrieved_point(&p)))
    }

    /// Fetch the messages of a thread in chronological order.
    pub async fn get_thread_context(&self, thread_id: u64, limit: u64) -> Result<Vec<MemoryEntry>> {
        let client = self.client()?;
        let filter = Filter::all([Condition::matches("thread_id", thread_id as i64)]);
        let mut entries = self.scroll(&client, filter, limit).await?;
        entries.sort_by_key(|e| e.timestamp);
        Ok(entries)
    }

    /// Semantic search over past conversations, with optional payload filters.
    ///
    /// Notification markers are always excluded from results.
    pub async fn search_context(
        &self,
        query_text: &str,
        options: &SearchOptions,
    ) -> Result<Vec<MemoryEntry>> {
        let client = self.client()?;
        let vector = self.embed(query_text).await?;

        let mut conditions: Vec<Condition> = Vec::new();
        if let Some(username) = &options.username {
            conditions.push(Condition::matches("username", username.clone()));
        }
        if let Some(min_ts) = options.min_timestamp {
            conditions.push(Condition::range(
                "timestamp",
                qdrant_client::qdrant::Range {
                    gte: Some(min_ts as f64),
                    ..Default::default()
                },
            ));
        }

        let filter = Filter {
            must: conditions,
            must_not: vec![Condition::matches(
                "message_type",
                MessageType::Notification.as_str().to_string(),
            )],
            ..Default::default()
        };

        let response = client
            .query(
                QueryPointsBuilder::new(self.collection.clone())
                    .query(vector)
                    .filter(filter)
                    .limit(options.limit.max(1))
                    .with_payload(true)
                    .with_vectors(false),
            )
            .await
            .context("Failed to query Qdrant")?;

        Ok(response
            .result
            .into_iter()
            .filter_map(|p| MemoryEntry::from_scored_point(&p))
            .collect())
    }

    /// Persist a processed-notification marker (replaces the legacy JSON file).
    pub async fn mark_processed(&self, notification_id: u64) -> Result<()> {
        let payload = MessagePayload {
            id: notification_id,
            content: String::new(),
            username: "system".to_string(),
            message_type: MessageType::Notification,
            parent_id: None,
            thread_id: 0,
            timestamp: unix_now(),
            is_processed: true,
            media_urls: Vec::new(),
        };
        self.upsert(&payload).await
    }

    /// True if the given notification id has already been processed.
    pub async fn is_processed(&self, id: u64) -> Result<bool> {
        Ok(self
            .get_point(id)
            .await?
            .map(|e| e.message_type == MessageType::Notification)
            .unwrap_or(false))
    }

    /// List every processed notification id (used to seed the session cache).
    pub async fn list_processed(&self) -> Result<Vec<u64>> {
        let client = self.client()?;
        let filter = Filter::all([Condition::matches(
            "message_type",
            MessageType::Notification.as_str().to_string(),
        )]);
        let mut ids = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        loop {
            let mut builder = ScrollPointsBuilder::new(self.collection.clone())
                .filter(filter.clone())
                .limit(100)
                .with_payload(true)
                .with_vectors(false);
            if let Some(offset_id) = offset.clone() {
                builder = builder.offset(offset_id);
            }
            let response = client.scroll(builder).await?;
            for point in response.result {
                if let Some(entry) = MemoryEntry::from_retrieved_point(&point) {
                    ids.push(entry.id);
                }
            }
            match response.next_page_offset {
                Some(next) => offset = Some(next),
                None => break,
            }
        }
        Ok(ids)
    }

    /// Spawn the background batch writer and return its channel.
    ///
    /// User posts/replies are queued here and written in batches (one embedding
    /// call per batch) on a short flush interval, keeping the poll loop cheap.
    pub fn spawn_writer(
        self: &Arc<Self>,
        batch_size: usize,
        flush_interval: Duration,
    ) -> mpsc::UnboundedSender<MemoryWrite> {
        let (tx, mut rx) = mpsc::unbounded_channel::<MemoryWrite>();
        let qdrant = self.clone();
        tokio::spawn(async move {
            let mut pending: Vec<MessagePayload> = Vec::new();
            let mut interval = tokio::time::interval(flush_interval);
            interval.tick().await; // consume the immediate first tick

            loop {
                tokio::select! {
                    maybe = rx.recv() => match maybe {
                        Some(MemoryWrite::Upsert(payload)) => {
                            pending.push(payload);
                            if pending.len() >= batch_size {
                                flush_batch(&qdrant, &mut pending).await;
                            }
                        }
                        Some(MemoryWrite::Flush(ack)) => {
                            flush_batch(&qdrant, &mut pending).await;
                            let _ = ack.send(());
                        }
                        None => {
                            flush_batch(&qdrant, &mut pending).await;
                            break;
                        }
                    },
                    _ = interval.tick() => {
                        flush_batch(&qdrant, &mut pending).await;
                    }
                }
            }
        });
        tx
    }

    fn client(&self) -> Result<Arc<Qdrant>> {
        if !self.is_available() {
            return Err(anyhow::anyhow!("Qdrant is unavailable"));
        }
        self.client
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Qdrant client not initialized"))
    }

    async fn scroll(
        &self,
        client: &Qdrant,
        filter: Filter,
        limit: u64,
    ) -> Result<Vec<MemoryEntry>> {
        let mut entries = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        loop {
            let mut builder = ScrollPointsBuilder::new(self.collection.clone())
                .filter(filter.clone())
                .limit(limit.min(100) as u32)
                .with_payload(true)
                .with_vectors(false);
            if let Some(offset_id) = offset.clone() {
                builder = builder.offset(offset_id);
            }
            let response = client.scroll(builder).await?;
            for point in response.result {
                if let Some(entry) = MemoryEntry::from_retrieved_point(&point) {
                    entries.push(entry);
                    if entries.len() as u64 >= limit {
                        return Ok(entries);
                    }
                }
            }
            match response.next_page_offset {
                Some(next) => offset = Some(next),
                None => break,
            }
        }
        Ok(entries)
    }
}

async fn flush_batch(qdrant: &QdrantClient, pending: &mut Vec<MessagePayload>) {
    if pending.is_empty() {
        return;
    }
    let items = std::mem::take(pending);
    match qdrant.upsert_many(&items).await {
        Ok(()) => info!("Persisted {} messages to Qdrant", items.len()),
        Err(e) => error!("Failed to persist {} messages to Qdrant: {e}", items.len()),
    }
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_client_errors_gracefully() {
        let embedder = FakeEmbedder;
        let client = QdrantClient::unavailable(Arc::new(embedder), 4, 10);
        assert!(!client.is_available());
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            assert!(client
                .upsert(&MessagePayload {
                    id: 1,
                    content: "x".into(),
                    username: "u".into(),
                    message_type: MessageType::Post,
                    parent_id: None,
                    thread_id: 1,
                    timestamp: 0,
                    is_processed: false,
                    media_urls: vec![],
                })
                .await
                .is_err());
            assert!(client.is_processed(1).await.is_err());
        });
    }

    struct FakeEmbedder;
    #[async_trait::async_trait]
    impl Embedder for FakeEmbedder {
        async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v = vec![0.0; 4];
                    for (i, b) in t.bytes().enumerate() {
                        v[i % 4] += b as f32;
                    }
                    normalize(&mut v);
                    v
                })
                .collect())
        }
    }

    fn normalize(v: &mut [f32]) {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    #[test]
    fn embedder_vectors_are_normalized() {
        let embedder = FakeEmbedder;
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let vectors = embedder.embed_texts(&["hello".to_string()]).await.unwrap();
            let norm = vectors[0]
                .iter()
                .map(|x| (x * x) as f64)
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-3);
        });
    }
}
