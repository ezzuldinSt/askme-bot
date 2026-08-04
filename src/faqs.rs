//! Support FAQs: admin-entered question/answer pairs distilled into
//! authoritative "support facts" in tier-3 memory (the `things_knowledge`
//! collection, `source = faq`, `status = active`).
//!
//! `support_faqs.json` (working dir) is the durable record: it drives the
//! admin panel's list view, records which fact points each FAQ owns (so
//! deletion removes exactly those), and lets boot/wipe re-seeds restore the
//! facts without re-running extraction.

use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use uuid::Uuid;

use crate::gemini_client::GeminiClient;
use crate::qdrant_client::QdrantClient;
use crate::qdrant_models::{app_fact_point_id, AppFactPayload, AppFactSource, AppFactStatus};

/// Per-instance FAQ store (working directory). Instance data, not secrets —
/// gitignored like `bot_config.json`.
pub const FAQS_FILE: &str = "support_faqs.json";

/// Topic stamped on support facts (groups them inside things_knowledge).
const SUPPORT_TOPIC: &str = "support";
/// Facts longer than this are almost certainly extraction noise.
const MAX_FACT_LENGTH: usize = 300;
/// How many facts one FAQ may contribute.
const MAX_FACTS_PER_FAQ: usize = 6;

/// One admin-entered FAQ and the facts distilled from it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportFaq {
    pub id: Uuid,
    pub question: String,
    pub answer: String,
    /// English facts distilled from the Q/A. Empty = extraction pending (e.g.
    /// the FAQ was saved while Gemini was failing); the next re-seed retries.
    #[serde(default)]
    pub facts: Vec<String>,
    pub created_at: i64,
}

/// Load the FAQ store (missing/corrupt -> empty, like config::load).
pub fn load() -> Vec<SupportFaq> {
    match std::fs::read_to_string(FAQS_FILE) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_else(|e| {
            warn!("Failed to parse {FAQS_FILE}: {e}; starting with an empty FAQ store");
            Vec::new()
        }),
        Err(_) => Vec::new(),
    }
}

/// Persist the FAQ store atomically with owner-only permissions.
pub fn save(faqs: &[SupportFaq]) -> Result<()> {
    let content = serde_json::to_string_pretty(faqs)?;
    let tmp = format!("{FAQS_FILE}.tmp");
    std::fs::write(&tmp, content).with_context(|| format!("Failed to write {tmp}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600)).ok();
    }
    std::fs::rename(&tmp, FAQS_FILE).with_context(|| format!("Failed to rename {tmp}"))?;
    Ok(())
}

/// Extraction prompt: FAQ Q/A -> standalone English support facts.
const FAQ_EXTRACTION_PROMPT: &str = r#"You are the support-knowledge distiller for AskMe, the support bot of the Things social network app.
Given ONE support FAQ (a question and its official answer), distill it into durable facts a support agent can answer from.

Output ONLY a JSON object of this exact shape:
{"facts": ["...", "..."]}

Rules:
- 1 to 6 facts; each one short, self-contained, and written in English (translate if needed) — steps, rules, limits, or settings paths.
- Facts must stand alone WITHOUT the question text: include the subject ("To change your profile picture: go to Profile, tap Edit, then tap the photo" — not "Tap the photo").
- Never invent steps or features that are not in the answer. If the answer contains nothing actionable, return an empty array."#;

/// Distill one FAQ into English support facts via Gemini (stateless call,
/// key-rotating). Unparseable output degrades to "no facts".
pub async fn extract_faq_facts(
    gemini: &GeminiClient,
    question: &str,
    answer: &str,
) -> Result<Vec<String>> {
    let user_text = format!("[FAQ]\nQuestion: {question}\nAnswer: {answer}");
    let raw = gemini
        .generate_json(FAQ_EXTRACTION_PROMPT, &user_text)
        .await?;
    Ok(parse_faq_facts(&raw))
}

/// Leniently parse the distiller's JSON output: strips markdown fences and
/// ignores prose around the object; caps and dedupes the result.
fn parse_faq_facts(raw: &str) -> Vec<String> {
    let trimmed = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let start = trimmed.find('{');
    let end = trimmed.rfind('}');
    let candidate = match (start, end) {
        (Some(s), Some(e)) if e > s => &trimmed[s..=e],
        _ => return Vec::new(),
    };
    #[derive(Deserialize, Default)]
    struct Out {
        #[serde(default)]
        facts: Vec<String>,
    }
    let parsed: Out = serde_json::from_str(candidate).unwrap_or_default();
    let mut out: Vec<String> = Vec::new();
    for fact in parsed.facts {
        let text = fact.trim();
        let chars = text.chars().count();
        if !(3..=MAX_FACT_LENGTH).contains(&chars) {
            continue;
        }
        if !out.iter().any(|f| f.eq_ignore_ascii_case(text)) {
            out.push(text.to_string());
        }
        if out.len() >= MAX_FACTS_PER_FAQ {
            break;
        }
    }
    out
}

/// The Qdrant payloads for one FAQ's facts (deterministic point ids).
fn faq_fact_items(faq: &SupportFaq, now: i64) -> Vec<(Uuid, AppFactPayload)> {
    faq.facts
        .iter()
        .map(|fact| {
            (
                app_fact_point_id(fact),
                AppFactPayload {
                    topic: SUPPORT_TOPIC.to_string(),
                    fact: fact.clone(),
                    source: AppFactSource::Faq,
                    status: AppFactStatus::Active,
                    updated_at: now,
                },
            )
        })
        .collect()
}

/// Upsert one FAQ's facts into memory. No-op when Qdrant is unavailable or
/// the FAQ has no facts yet (a later re-seed syncs).
async fn sync_faq_facts(qdrant: &Arc<QdrantClient>, faq: &SupportFaq) -> Result<()> {
    if !qdrant.is_available() || faq.facts.is_empty() {
        return Ok(());
    }
    let items = faq_fact_items(faq, unix_now());
    qdrant.upsert_app_facts(&items).await
}

/// Insert a new FAQ: distill facts, push them into memory IMMEDIATELY (hot —
/// the next reply can answer from them), then persist. Returns the stored
/// FAQ plus whether its facts reached Qdrant this instant (false = a boot or
/// wipe re-seed will sync them later).
pub async fn insert_faq(
    qdrant: &Arc<QdrantClient>,
    gemini: &GeminiClient,
    question: &str,
    answer: &str,
) -> Result<(SupportFaq, bool)> {
    let facts = extract_faq_facts(gemini, question, answer).await?;
    let faq = SupportFaq {
        id: Uuid::new_v4(),
        question: question.trim().to_string(),
        answer: answer.trim().to_string(),
        facts,
        created_at: unix_now(),
    };
    let synced = match sync_faq_facts(qdrant, &faq).await {
        Ok(()) => qdrant.is_available() && !faq.facts.is_empty(),
        Err(e) => {
            warn!("Support facts for new FAQ {} not synced yet: {e}", faq.id);
            false
        }
    };
    let mut faqs = load();
    faqs.push(faq.clone());
    save(&faqs)?;
    info!(
        "Support FAQ {} added ({} facts, synced={synced})",
        faq.id,
        faq.facts.len()
    );
    Ok((faq, synced))
}

/// Delete a FAQ: remove its fact points from memory, then from the store.
/// Returns the removed FAQ (None when the id is unknown).
pub async fn delete_faq(qdrant: &Arc<QdrantClient>, id: Uuid) -> Result<Option<SupportFaq>> {
    let mut faqs = load();
    let Some(pos) = faqs.iter().position(|f| f.id == id) else {
        return Ok(None);
    };
    let faq = faqs.remove(pos);
    let ids: Vec<Uuid> = faq.facts.iter().map(|f| app_fact_point_id(f)).collect();
    if qdrant.is_available() {
        if let Err(e) = qdrant.delete_app_facts(&ids).await {
            // Not fatal: the point ids are deterministic, a later delete or
            // wipe clears them. The FAQ itself is already gone from the store.
            warn!("Failed to delete support fact points for FAQ {id}: {e}");
        }
    }
    save(&faqs)?;
    info!("Support FAQ {id} deleted ({} facts)", faq.facts.len());
    Ok(Some(faq))
}

/// Upsert every FAQ's facts into tier-3 memory (idempotent — deterministic
/// point ids). FAQs with no stored facts are re-extracted first and the file
/// is updated. Called on boot and after memory wipes.
pub async fn seed_support_faqs(qdrant: &Arc<QdrantClient>, gemini: &GeminiClient) -> Result<()> {
    let mut faqs = load();
    if faqs.is_empty() {
        return Ok(());
    }
    let now = unix_now();
    let mut store_changed = false;
    let mut items: Vec<(Uuid, AppFactPayload)> = Vec::new();
    for faq in faqs.iter_mut() {
        if faq.facts.is_empty() {
            match extract_faq_facts(gemini, &faq.question, &faq.answer).await {
                Ok(facts) if !facts.is_empty() => {
                    faq.facts = facts;
                    store_changed = true;
                }
                Ok(_) => warn!("FAQ {}: extraction returned no facts", faq.id),
                Err(e) => warn!("FAQ {}: fact extraction failed: {e}", faq.id),
            }
        }
        items.extend(faq_fact_items(faq, now));
    }
    if store_changed {
        save(&faqs)?;
    }
    let count = items.len();
    qdrant.upsert_app_facts(&items).await?;
    info!("Seeded {count} support facts from {FAQS_FILE}");
    Ok(())
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
    fn parse_faq_facts_clean_json() {
        let raw = r#"{"facts": ["To change your profile picture: go to Profile, tap Edit, then tap the photo.", "Posts expire after 24 hours by default."]}"#;
        let facts = parse_faq_facts(raw);
        assert_eq!(facts.len(), 2);
        assert!(facts[0].starts_with("To change your profile picture"));
    }

    #[test]
    fn parse_faq_facts_fenced_with_prose_and_noise() {
        let raw = "Sure!\n```json\n{\"facts\": [\"ok fact\", \"x\", \"ok fact\", \"\"]}\n```\nDone";
        let facts = parse_faq_facts(raw);
        assert_eq!(facts, vec!["ok fact".to_string()], "deduped, noise dropped");
        assert!(parse_faq_facts("not json").is_empty());
        assert!(parse_faq_facts("").is_empty());
    }

    #[test]
    fn parse_faq_facts_caps_at_six() {
        let raw = serde_json::json!({
            "facts": (1..=9).map(|i| format!("fact number {i}")).collect::<Vec<_>>()
        })
        .to_string();
        assert_eq!(parse_faq_facts(&raw).len(), MAX_FACTS_PER_FAQ);
    }

    #[test]
    fn faq_store_roundtrip() {
        let faq = SupportFaq {
            id: Uuid::new_v4(),
            question: "كيف أغير صورتي؟".to_string(),
            answer: "من الملف الشخصي".to_string(),
            facts: vec!["To change your profile picture: go to Profile, tap Edit.".to_string()],
            created_at: 123,
        };
        let json = serde_json::to_string(&vec![faq.clone()]).unwrap();
        let back: Vec<SupportFaq> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].id, faq.id);
        assert_eq!(back[0].facts, faq.facts);
        // Older files without `facts` still parse.
        let legacy: Vec<SupportFaq> = serde_json::from_str(
            &serde_json::json!([{ "id": faq.id, "question": "q", "answer": "a", "created_at": 1 }])
                .to_string(),
        )
        .unwrap();
        assert!(legacy[0].facts.is_empty());
    }

    #[test]
    fn faq_fact_items_are_deterministic_and_active() {
        let faq = SupportFaq {
            id: Uuid::new_v4(),
            question: "q".to_string(),
            answer: "a".to_string(),
            facts: vec!["Posts expire after 24 hours.".to_string()],
            created_at: 0,
        };
        let items = faq_fact_items(&faq, 42);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, app_fact_point_id("Posts expire after 24 hours."));
        assert_eq!(items[0].1.source, AppFactSource::Faq);
        assert_eq!(items[0].1.status, AppFactStatus::Active);
        assert_eq!(items[0].1.topic, SUPPORT_TOPIC);
    }
}
