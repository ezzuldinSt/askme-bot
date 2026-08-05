#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use qdrant_client::qdrant::{
    points_selector::PointsSelectorOneOf, Condition, CreateCollectionBuilder,
    CreateFieldIndexCollectionBuilder, DeletePointsBuilder, Direction, FieldType, Filter,
    GetPointsBuilder, OrderBy, PointId, PointStruct, PointsIdsList, QueryPointsBuilder,
    ScrollPointsBuilder, SetPayloadPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::qdrant::Distance;
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::qdrant_models::{
    decode_payload, AppFactPayload, MemoryEntry, MessagePayload, UserFactPayload, COLLECTION_NAME,
    PROCESSED_COLLECTION_NAME, THINGS_KNOWLEDGE_COLLECTION_NAME, USER_PROFILES_COLLECTION_NAME,
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

/// Point count of one collection (None when unavailable/unknown).
#[derive(Debug, Clone, serde::Serialize)]
pub struct CollectionStat {
    pub name: String,
    pub points: Option<u64>,
}

/// Wrapper around a local Qdrant instance that stores the bot's memory.
///
/// Memory is split into three strictly-scoped tiers:
///
/// 1. `conversation_memory` — episodic memory, one conversation per post where
///    the bot was @mentioned. Reads are ONLY ever filtered by
///    `conversation_id`, so conversations are fully isolated from each other.
/// 2. `user_profiles` — durable facts about individual users (the ChatGPT-style
///    long-term memory). The only memory that crosses conversations, and every
///    read is filtered by `username`.
/// 3. `things_knowledge` — curated-first facts about the Things app itself,
///    injected only when a score-gated semantic search matches the question.
pub struct QdrantClient {
    client: Option<Arc<Qdrant>>,
    available: AtomicBool,
    embedder: Arc<dyn Embedder>,
    collection: String,
    dimensions: u64,
}

impl QdrantClient {
    /// Build a client for the given gRPC endpoint and check connectivity.
    ///
    /// This never fails hard: if Qdrant is unreachable the client is created in
    /// a degraded state (`is_available() == false`) and all operations no-op
    /// with an error, so the bot keeps working without persistent memory.
    pub async fn connect(url: &str, embedder: Arc<dyn Embedder>, dimensions: u64) -> Self {
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
    pub fn unavailable(embedder: Arc<dyn Embedder>, dimensions: u64) -> Self {
        Self {
            client: None,
            available: AtomicBool::new(false),
            embedder,
            collection: COLLECTION_NAME.to_string(),
            dimensions,
        }
    }

    /// True when the underlying Qdrant server is reachable.
    pub fn is_available(&self) -> bool {
        self.available.load(Ordering::Relaxed)
    }

    /// Name of the conversation collection this client operates on.
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

    /// Create every collection the bot needs if it does not exist yet, verify
    /// vector sizes on existing ones, and ensure payload indexes are in place.
    pub async fn ensure_collections(&self) -> Result<()> {
        let conversation = self.collection.clone();
        self.ensure_one_collection(&conversation).await?;
        self.ensure_one_collection(PROCESSED_COLLECTION_NAME).await?;
        self.ensure_one_collection(USER_PROFILES_COLLECTION_NAME).await?;
        self.ensure_one_collection(THINGS_KNOWLEDGE_COLLECTION_NAME).await?;
        self.ensure_payload_indexes().await;
        Ok(())
    }

    async fn ensure_one_collection(&self, name: &str) -> Result<()> {
        let client = self.client()?;
        if client.collection_exists(name).await? {
            self.verify_dimensions(name).await?;
            return Ok(());
        }
        client
            .create_collection(
                CreateCollectionBuilder::new(name.to_string())
                    .vectors_config(VectorParamsBuilder::new(self.dimensions, Distance::Cosine)),
            )
            .await
            .with_context(|| format!("Failed to create Qdrant collection {name}"))?;
        info!("Created Qdrant collection {name}");
        Ok(())
    }

    /// Payload indexes keep `conversation_id`/`username`-filtered reads fast as
    /// the collections grow. Failures are non-fatal (queries still work with a
    /// full scan), so they are only logged.
    async fn ensure_payload_indexes(&self) {
        let Ok(client) = self.client() else {
            return;
        };
        let indexes: &[(&str, &str, FieldType)] = &[
            (COLLECTION_NAME, "conversation_id", FieldType::Integer),
            (COLLECTION_NAME, "username", FieldType::Keyword),
            (COLLECTION_NAME, "timestamp", FieldType::Integer),
            (USER_PROFILES_COLLECTION_NAME, "username", FieldType::Keyword),
            (USER_PROFILES_COLLECTION_NAME, "active", FieldType::Bool),
            (USER_PROFILES_COLLECTION_NAME, "last_seen", FieldType::Integer),
            (THINGS_KNOWLEDGE_COLLECTION_NAME, "status", FieldType::Keyword),
        ];
        for (collection, field, field_type) in indexes {
            let result = client
                .create_field_index(CreateFieldIndexCollectionBuilder::new(
                    collection.to_string(),
                    field.to_string(),
                    *field_type,
                ))
                .await;
            match result {
                Ok(_) => info!("Payload index ready: {collection}.{field}"),
                Err(e) => warn!("Failed to create payload index {collection}.{field}: {e}"),
            }
        }
    }

    /// Delete the three memory collections (conversation, user profiles, app
    /// knowledge) and recreate them empty. Processed-notification markers are
    /// deliberately kept so the bot never re-answers old mentions.
    pub async fn reset_memory(&self) -> Result<()> {
        let client = self.client()?;
        for name in [
            COLLECTION_NAME,
            USER_PROFILES_COLLECTION_NAME,
            THINGS_KNOWLEDGE_COLLECTION_NAME,
        ] {
            if client.collection_exists(name).await? {
                client
                    .delete_collection(name.to_string())
                    .await
                    .with_context(|| format!("Failed to delete Qdrant collection {name}"))?;
                info!("Deleted Qdrant collection {name}");
            }
        }
        self.ensure_collections().await
    }

    /// Delete and recreate the processed-notification markers collection.
    /// DANGEROUS: afterwards the bot may re-answer old mentions. Admin panel
    /// danger-zone only.
    pub async fn wipe_processed(&self) -> Result<()> {
        let client = self.client()?;
        if client.collection_exists(PROCESSED_COLLECTION_NAME).await? {
            client
                .delete_collection(PROCESSED_COLLECTION_NAME.to_string())
                .await
                .context("Failed to delete processed markers collection")?;
            info!("Deleted Qdrant collection {PROCESSED_COLLECTION_NAME}");
        }
        self.ensure_one_collection(PROCESSED_COLLECTION_NAME).await
    }

    /// Point counts of every collection the bot uses (admin dashboard).
    pub async fn collection_stats(&self) -> Vec<CollectionStat> {
        let names = [
            COLLECTION_NAME,
            PROCESSED_COLLECTION_NAME,
            USER_PROFILES_COLLECTION_NAME,
            THINGS_KNOWLEDGE_COLLECTION_NAME,
        ];
        let mut out = Vec::with_capacity(names.len());
        let Ok(client) = self.client() else {
            return names
                .into_iter()
                .map(|name| CollectionStat {
                    name: name.to_string(),
                    points: None,
                })
                .collect();
        };
        for name in names {
            let points = client
                .collection_info(name.to_string())
                .await
                .ok()
                .and_then(|info| info.result)
                .and_then(|r| r.points_count);
            out.push(CollectionStat {
                name: name.to_string(),
                points,
            });
        }
        out
    }

    /// Fail loudly when an existing collection's vector size does not match
    /// `EMBEDDING_DIMENSIONS` — otherwise every write would fail forever.
    async fn verify_dimensions(&self, name: &str) -> Result<()> {
        let client = self.client()?;
        let info = client
            .collection_info(name.to_string())
            .await
            .with_context(|| format!("Failed to inspect Qdrant collection {name}"))?;
        let size = info
            .result
            .and_then(|r| r.config)
            .and_then(|c| c.params)
            .and_then(|p| p.vectors_config)
            .and_then(|v| v.config)
            .and_then(|config| match config {
                qdrant_client::qdrant::vectors_config::Config::Params(params) => Some(params.size),
                _ => None,
            });
        match size {
            Some(size) if size == self.dimensions => Ok(()),
            Some(size) => Err(anyhow::anyhow!(
                "Qdrant collection {name} has vector size {size} but EMBEDDING_DIMENSIONS is {}; \
                 delete the collection or fix the env var",
                self.dimensions
            )),
            // Unknown/named-vector layout — leave it untouched.
            None => Ok(()),
        }
    }

    /// Embed a single text and return the normalized vector.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut vectors = self.embedder.embed_texts(&[text.to_string()]).await?;
        vectors
            .pop()
            .ok_or_else(|| anyhow::anyhow!("Embedder returned no vector"))
    }

    /// Embed several texts in one batched request (one API call instead of N).
    pub async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embedder.embed_texts(texts).await
    }

    /// Store a single conversation message, computing its embedding on the spot.
    ///
    /// Used for writes that must be immediately visible (bot replies and
    /// follow-up posts); user posts are normally queued via `spawn_writer`.
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

    /// Store a batch of conversation messages, embedding all of them in one call.
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
        let mut is_embedded = vec![false; items.len()];
        for &i in &embed_indices {
            is_embedded[i] = true;
        }
        let mut vector_iter = vectors.into_iter();
        let zipped: Vec<(MessagePayload, Vec<f32>)> = items
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let vector = if is_embedded[i] {
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

    /// Look up a single conversation message by its id.
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

    /// Fetch the most recent `limit` messages of ONE conversation in
    /// chronological order. This is the only way conversation context is read:
    /// always filtered by `conversation_id`, never across conversations.
    ///
    /// The scroll is server-side ordered by timestamp DESC, so the "most
    /// recent" window stays correct no matter how long the conversation gets
    /// (a plain scroll would truncate the wrong end past the page cap).
    pub async fn get_conversation_context(
        &self,
        conversation_id: u64,
        limit: u64,
    ) -> Result<Vec<MemoryEntry>> {
        let client = self.client()?;
        let filter = Filter::all([Condition::matches(
            "conversation_id",
            conversation_id as i64,
        )]);
        let collection = self.collection.clone();
        let points = self
            .scroll_raw::<MemoryEntry>(
                &client,
                &collection,
                filter,
                Some(("timestamp", Direction::Desc)),
                limit,
            )
            .await?;
        let mut entries: Vec<MemoryEntry> =
            points.into_iter().filter_map(|(_, p)| p).collect();
        // Chronological order, oldest first. The id tie-break keeps same-
        // second entries in posting order too (Things ids grow over time);
        // a plain timestamp sort would leave ties in the fetched (newest-
        // first) order.
        entries.sort_by_key(|e| (e.timestamp, e.id));
        Ok(entries)
    }

    /// Semantic search WITHIN a single conversation (debug/dry-run helper).
    /// Like `get_conversation_context`, the `conversation_id` filter is
    /// mandatory — there is deliberately no cross-conversation search.
    pub async fn search_conversation(
        &self,
        query_text: &str,
        conversation_id: u64,
        limit: u64,
    ) -> Result<Vec<MemoryEntry>> {
        let client = self.client()?;
        let vector = self.embed(query_text).await?;
        let filter = Filter::all([Condition::matches(
            "conversation_id",
            conversation_id as i64,
        )]);
        let response = client
            .query(
                QueryPointsBuilder::new(self.collection.clone())
                    .query(vector)
                    .filter(filter)
                    .limit(limit.max(1))
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

    // ── Tier 2: per-user durable facts ──

    /// Insert (or fully replace) a user fact, embedding the fact text.
    ///
    /// The stored `username` is lowercased: keyword filters are exact-match,
    /// and fact point ids already normalize case — so reads must be able to
    /// assume one canonical (case-insensitive) form.
    pub async fn upsert_user_fact(&self, point_id: Uuid, payload: &UserFactPayload) -> Result<()> {
        let client = self.client()?;
        let mut payload = payload.clone();
        payload.username = unorm(&payload.username);
        let vector = self.embed(&payload.fact).await?;
        let json = serde_json::to_value(&payload)
            .unwrap_or_else(|_| serde_json::json!({ "fact": payload.fact }));
        let point = PointStruct::new(
            point_id.to_string(),
            vector,
            Payload::try_from(json).unwrap_or_default(),
        );
        client
            .upsert_points(UpsertPointsBuilder::new(
                USER_PROFILES_COLLECTION_NAME.to_string(),
                vec![point],
            ))
            .await
            .context("Failed to upsert user fact to Qdrant")?;
        Ok(())
    }

    /// Fetch a single user fact by its deterministic point id.
    pub async fn get_user_fact(&self, point_id: Uuid) -> Result<Option<UserFactPayload>> {
        let client = self.client()?;
        let response = client
            .get_points(
                GetPointsBuilder::new(
                    USER_PROFILES_COLLECTION_NAME.to_string(),
                    vec![point_id.to_string().into()],
                )
                .with_payload(true),
            )
            .await
            .context("Failed to get user fact from Qdrant")?;
        Ok(response
            .result
            .into_iter()
            .next()
            .and_then(|p| decode_payload(&p.payload)))
    }

    /// Merge-update fields of an existing user fact (no re-embedding).
    ///
    /// Used to reinforce (bump `last_seen`/`times_confirmed`), supersede
    /// (`active=false` + `superseded_by`), and forget (`active=false`) facts.
    pub async fn patch_user_fact(&self, point_id: Uuid, patch: serde_json::Value) -> Result<()> {
        let client = self.client()?;
        let payload = Payload::try_from(patch).unwrap_or_default();
        client
            .set_payload(
                SetPayloadPointsBuilder::new(USER_PROFILES_COLLECTION_NAME.to_string(), payload)
                    .points_selector(PointsSelectorOneOf::Points(PointsIdsList {
                        ids: vec![point_id.to_string().into()],
                    })),
            )
            .await
            .context("Failed to patch user fact in Qdrant")?;
        Ok(())
    }

    /// List a user's active facts, most recently confirmed first
    /// (server-side ordered by `last_seen` DESC).
    pub async fn list_user_facts(
        &self,
        username: &str,
        limit: u64,
    ) -> Result<Vec<(Uuid, UserFactPayload)>> {
        let client = self.client()?;
        let filter = Filter::all([
            Condition::matches("username", unorm(username)),
            Condition::matches("active", true),
        ]);
        let points = self
            .scroll_raw::<UserFactPayload>(
                &client,
                USER_PROFILES_COLLECTION_NAME,
                filter,
                Some(("last_seen", Direction::Desc)),
                limit,
            )
            .await?;
        Ok(points
            .into_iter()
            .filter_map(|(id, payload)| {
                let uuid = id.and_then(point_id_uuid)?;
                let payload = payload?;
                Some((uuid, payload))
            })
            .collect())
    }

    /// One-time migration: lowercase the stored `username` of every ACTIVE
    /// user fact written before reads became case-insensitive. New writes are
    /// normalized at insert time, so this converges old data in the
    /// background and becomes a no-op once everything is normalized.
    pub async fn normalize_usernames(&self) -> Result<()> {
        let client = self.client()?;
        let filter = Filter::all([Condition::matches("active", true)]);
        let points = self
            .scroll_raw::<UserFactPayload>(
                &client,
                USER_PROFILES_COLLECTION_NAME,
                filter,
                None,
                100_000,
            )
            .await?;
        let mut fixed = 0usize;
        for (id, payload) in points {
            let (Some(id), Some(p)) = (id.and_then(point_id_uuid), payload) else {
                continue;
            };
            let lower = unorm(&p.username);
            if lower != p.username {
                if let Err(e) = self
                    .patch_user_fact(id, serde_json::json!({ "username": lower }))
                    .await
                {
                    warn!("Failed to normalize username on fact {id}: {e}");
                } else {
                    fixed += 1;
                }
            }
        }
        if fixed > 0 {
            info!("Normalized username casing on {fixed} user facts");
        }
        Ok(())
    }

    /// Find the most similar ACTIVE fact of the same user above `threshold`
    /// (used for contradiction/supersede detection).
    pub async fn find_similar_user_fact(        &self,
        username: &str,
        vector: &[f32],
        threshold: f32,
    ) -> Result<Option<(Uuid, UserFactPayload)>> {
        let mut hits = self
            .query_user_facts(username, vector, threshold, 1)
            .await?;
        Ok(hits.pop())
    }

    /// Semantic search over a user's ACTIVE facts (used to locate facts the
    /// user asked to forget).
    pub async fn search_user_facts_semantic(
        &self,
        username: &str,
        vector: &[f32],
        threshold: f32,
        limit: u64,
    ) -> Result<Vec<(Uuid, UserFactPayload)>> {
        self.query_user_facts(username, vector, threshold, limit)
            .await
    }

    /// Semantic search across EVERY user's ACTIVE facts (no username scope).
    /// Used by the bot's user-lookup tool, where the queried user may only be
    /// known by a name or a username fragment. Results carry their owner.
    pub async fn search_user_facts_global(
        &self,
        vector: &[f32],
        threshold: f32,
        limit: u64,
    ) -> Result<Vec<(String, UserFactPayload)>> {
        let client = self.client()?;
        let filter = Filter::all([Condition::matches("active", true)]);
        let builder = QueryPointsBuilder::new(USER_PROFILES_COLLECTION_NAME.to_string())
            .query(vector.to_vec())
            .filter(filter)
            .limit(limit.max(1))
            .score_threshold(threshold)
            .with_payload(true)
            .with_vectors(false);
        let response = client
            .query(builder)
            .await
            .context("Failed to query user facts from Qdrant")?;
        Ok(response
            .result
            .into_iter()
            .filter_map(|p| {
                let payload: UserFactPayload = decode_payload(&p.payload)?;
                Some((payload.username.clone(), payload))
            })
            .collect())
    }

    async fn query_user_facts(
        &self,
        username: &str,
        vector: &[f32],
        threshold: f32,
        limit: u64,
    ) -> Result<Vec<(Uuid, UserFactPayload)>> {
        let client = self.client()?;
        let filter = Filter::all([
            Condition::matches("username", unorm(username)),
            Condition::matches("active", true),
        ]);
        let builder = QueryPointsBuilder::new(USER_PROFILES_COLLECTION_NAME.to_string())
            .query(vector.to_vec())
            .filter(filter)
            .limit(limit.max(1))
            .score_threshold(threshold)
            .with_payload(true)
            .with_vectors(false);
        let response = client
            .query(builder)
            .await
            .context("Failed to query user facts from Qdrant")?;
        Ok(response
            .result
            .into_iter()
            .filter_map(|p| {
                let id = p.id.clone().and_then(point_id_uuid)?;
                let payload = decode_payload(&p.payload)?;
                Some((id, payload))
            })
            .collect())
    }

    // ── Tier 3: Things app knowledge ──

    /// Fetch a single app-knowledge fact by its deterministic point id.
    pub async fn get_app_fact(&self, point_id: Uuid) -> Result<Option<AppFactPayload>> {
        let client = self.client()?;
        let response = client
            .get_points(
                GetPointsBuilder::new(
                    THINGS_KNOWLEDGE_COLLECTION_NAME.to_string(),
                    vec![point_id.to_string().into()],
                )
                .with_payload(true),
            )
            .await
            .context("Failed to get app fact from Qdrant")?;
        Ok(response
            .result
            .into_iter()
            .next()
            .and_then(|p| decode_payload(&p.payload)))
    }

    /// Insert or replace app-knowledge facts, embedding them in one batch.
    pub async fn upsert_app_facts(&self, items: &[(Uuid, AppFactPayload)]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let client = self.client()?;
        let texts: Vec<String> = items.iter().map(|(_, f)| f.fact.clone()).collect();
        let vectors = self.embedder.embed_texts(&texts).await?;
        let points: Vec<PointStruct> = items
            .iter()
            .zip(vectors)
            .map(|((id, payload), vector)| {
                let json = serde_json::to_value(payload)
                    .unwrap_or_else(|_| serde_json::json!({ "fact": payload.fact }));
                PointStruct::new(
                    id.to_string(),
                    vector,
                    Payload::try_from(json).unwrap_or_default(),
                )
            })
            .collect();
        client
            .upsert_points(UpsertPointsBuilder::new(
                THINGS_KNOWLEDGE_COLLECTION_NAME.to_string(),
                points,
            ))
            .await
            .context("Failed to upsert app facts to Qdrant")?;
        Ok(())
    }

    /// Delete app-knowledge points by id (used when a support FAQ is removed).
    pub async fn delete_app_facts(&self, point_ids: &[Uuid]) -> Result<()> {
        if point_ids.is_empty() {
            return Ok(());
        }
        let client = self.client()?;
        client
            .delete_points(
                DeletePointsBuilder::new(THINGS_KNOWLEDGE_COLLECTION_NAME.to_string())
                    .points(PointsIdsList {
                        ids: point_ids.iter().map(|id| id.to_string().into()).collect(),
                    })
                    .wait(true),
            )
            .await
            .context("Failed to delete app facts from Qdrant")?;
        Ok(())
    }

    /// Delete conversation messages by their Things post ids (the point ids of
    /// the conversation collection). Used to forget posts that no longer exist
    /// on Things.
    pub async fn delete_conversation_points(&self, ids: &[u64]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let client = self.client()?;
        client
            .delete_points(
                DeletePointsBuilder::new(self.collection.clone())
                    .points(PointsIdsList {
                        ids: ids.iter().map(|id| (*id).into()).collect(),
                    })
                    .wait(true),
            )
            .await
            .context("Failed to delete conversation points from Qdrant")?;
        Ok(())
    }

    /// List the `(id, parent_id)` of conversation messages (capped at `max`).
    /// The deleted-post sweeper verifies each of these against the Things API.
    ///
    /// Deliberately UNORDERED: plain id-offset pagination walks the whole
    /// collection, while `order_by` pagination (used by the context readers,
    /// which stay under one page) would silently stop after the first page.
    pub async fn list_conversation_refs(&self, max: u64) -> Result<Vec<(u64, Option<u64>)>> {
        let client = self.client()?;
        let points = self
            .scroll_raw::<MemoryEntry>(&client, &self.collection.clone(), Filter::default(), None, max)
            .await?;
        Ok(points
            .into_iter()
            .filter_map(|(_, p)| p.map(|e| (e.id, e.parent_id)))
            .collect())
    }

    /// Semantic search over ACTIVE app-knowledge facts, gated by `min_score` so
    /// unrelated questions never see app knowledge.
    pub async fn search_app_knowledge(
        &self,
        query_text: &str,
        min_score: f32,
        limit: u64,
    ) -> Result<Vec<AppFactPayload>> {
        let client = self.client()?;
        let vector = self.embed(query_text).await?;
        let filter = Filter::all([Condition::matches(
            "status",
            crate::qdrant_models::AppFactStatus::Active
                .as_str()
                .to_string(),
        )]);
        let response = client
            .query(
                QueryPointsBuilder::new(THINGS_KNOWLEDGE_COLLECTION_NAME.to_string())
                    .query(vector)
                    .filter(filter)
                    .limit(limit.max(1))
                    .score_threshold(min_score)
                    .with_payload(true)
                    .with_vectors(false),
            )
            .await
            .context("Failed to query app knowledge from Qdrant")?;
        Ok(response
            .result
            .into_iter()
            .filter_map(|p| decode_payload(&p.payload))
            .collect())
    }

    // ── Processed-notification markers ──

    /// Persist a processed-notification marker (replaces the legacy JSON file).
    ///
    /// Markers live in their own collection with a zero vector: no embedding
    /// call is needed and they can never collide with conversation points.
    pub async fn mark_processed(&self, notification_id: u64) -> Result<()> {
        self.mark_processed_many(&[notification_id]).await
    }

    /// Persist many processed-notification markers in one write.
    pub async fn mark_processed_many(&self, notification_ids: &[u64]) -> Result<()> {
        if notification_ids.is_empty() {
            return Ok(());
        }
        let client = self.client()?;
        let processed_at = unix_now();
        let points: Vec<PointStruct> = notification_ids
            .iter()
            .map(|id| {
                let json = serde_json::json!({ "id": id, "processed_at": processed_at });
                PointStruct::new(
                    *id,
                    self.zero_vector(),
                    Payload::try_from(json).unwrap_or_default(),
                )
            })
            .collect();
        client
            .upsert_points(UpsertPointsBuilder::new(
                PROCESSED_COLLECTION_NAME.to_string(),
                points,
            ))
            .await
            .context("Failed to upsert processed markers to Qdrant")?;
        Ok(())
    }

    /// True if the given notification id has already been processed.
    pub async fn is_processed(&self, id: u64) -> Result<bool> {
        let client = self.client()?;
        let response = client
            .get_points(
                GetPointsBuilder::new(PROCESSED_COLLECTION_NAME.to_string(), vec![id.into()])
                    .with_payload(false),
            )
            .await
            .context("Failed to get processed marker from Qdrant")?;
        Ok(!response.result.is_empty())
    }

    /// List processed notification ids (used to seed the session cache).
    /// Capped: Qdrant remains the source of truth (every unprocessed
    /// notification is checked individually anyway), so there is no reason to
    /// load an ever-growing history into memory on every boot.
    pub async fn list_processed(&self) -> Result<Vec<u64>> {
        /// Safety-net cache size; older markers fall back to per-item lookups.
        const MAX_SEEDED_IDS: usize = 10_000;
        let client = self.client()?;
        let mut ids = Vec::new();
        let mut offset: Option<PointId> = None;
        loop {
            let mut builder = ScrollPointsBuilder::new(PROCESSED_COLLECTION_NAME.to_string())
                .limit(100)
                .with_payload(false)
                .with_vectors(false);
            if let Some(offset_id) = offset {
                builder = builder.offset(offset_id);
            }
            let response = client.scroll(builder).await?;
            for point in response.result {
                if let Some(id) = point.id.and_then(point_id_num) {
                    ids.push(id);
                }
            }
            if ids.len() >= MAX_SEEDED_IDS {
                warn!(
                    "Processed-marker history exceeds {MAX_SEEDED_IDS}; seeding the session \
                     cache with the first page only (older markers are checked on demand)"
                );
                break;
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

    /// Page through a collection with a filter, optionally server-side ordered
    /// by a payload key (requires a payload index on that key), decoding each
    /// point into `(PointId, payload)` pairs.
    async fn scroll_raw<T: serde::de::DeserializeOwned>(
        &self,
        client: &Qdrant,
        collection: &str,
        filter: Filter,
        order: Option<(&str, Direction)>,
        limit: u64,
    ) -> Result<Vec<(Option<PointId>, Option<T>)>> {
        let mut entries = Vec::new();
        let mut offset: Option<PointId> = None;
        loop {
            let mut builder = ScrollPointsBuilder::new(collection.to_string())
                .filter(filter.clone())
                .limit(limit.min(100) as u32)
                .with_payload(true)
                .with_vectors(false);
            if let Some((key, direction)) = order {
                builder = builder.order_by(OrderBy {
                    key: key.to_string(),
                    direction: Some(direction as i32),
                    start_from: None,
                });
            }
            if let Some(offset_id) = offset.clone() {
                builder = builder.offset(offset_id);
            }
            let response = client.scroll(builder).await?;
            for point in response.result {
                let decoded = decode_payload::<T>(&point.payload);
                entries.push((point.id, decoded));
                if entries.len() as u64 >= limit {
                    return Ok(entries);
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

/// Canonical username form for storage and filtering: Things usernames are
/// case-insensitive, but Qdrant keyword matches are exact — so every write
/// and every read goes through this one normalization point.
fn unorm(username: &str) -> String {
    username.trim().to_lowercase()
}

/// Extract the numeric id of a Qdrant point (returns `None` for UUID points,
/// which conversation points never are).
fn point_id_num(point_id: PointId) -> Option<u64> {
    match point_id.point_id_options {
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => Some(n),
        _ => None,
    }
}

/// Extract the UUID of a Qdrant point (returns `None` for numeric points).
fn point_id_uuid(point_id: PointId) -> Option<Uuid> {
    match point_id.point_id_options {
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s)) => {
            Uuid::parse_str(&s).ok()
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qdrant_models::MessageType;

    #[test]
    fn unavailable_client_errors_gracefully() {
        let embedder = FakeEmbedder;
        let client = QdrantClient::unavailable(Arc::new(embedder), 4);
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
                    conversation_id: 1,
                    timestamp: 0,
                    media_urls: vec![],
                })
                .await
                .is_err());
            assert!(client.is_processed(1).await.is_err());
            assert!(client
                .list_user_facts("someone", 8)
                .await
                .is_err());
        });
    }

    #[test]
    fn unorm_normalizes_case_and_surrounding_whitespace() {
        assert_eq!(unorm("Khaled"), "khaled");
        assert_eq!(unorm("  KHALED "), "khaled");
        assert_eq!(unorm("khaled"), "khaled");
    }

    struct FakeEmbedder;
    #[async_trait::async_trait]
    impl Embedder for FakeEmbedder {        async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
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
                .map(|x| x * x)
                .sum::<f32>() as f64;
            let norm = norm.sqrt();
            assert!((norm - 1.0).abs() < 1e-3);
        });
    }
}
