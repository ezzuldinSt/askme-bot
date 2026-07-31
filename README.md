> **العربية:** [اقرأ النسخة العربية](README.ar.md)

# AskMe — the Things AI bot 🤖

Meet **AskMe**, a friendly AI assistant that lives on the [Things](https://things.cv) social network. When you need a quick answer, an opinion, or a look at a photo, just @mention AskMe — it will jump in and reply.

---

## What AskMe can do

### 🗣️ Answer your questions
Mention **@AskMe** in a post or in a reply, and it will answer right away. Ask about anything — a fact, an idea, an opinion, a "what should I do?" moment.

### 💬 Follows the conversation
AskMe reads the post you're replying to, so it understands the context. If you follow up with a second question, it **remembers what was said before** — even hours later — so you can keep the conversation going naturally.

### 🖼️ Sees images and videos
Attach a photo, screenshot, or video to your question and AskMe will look at it. Show it a product, a place, a piece of text, or anything else and it will tell you what it sees and how it relates to your question.

### 🌍 Speaks your language
AskMe matches the language you write in — Arabic, English, and more — and keeps your tone: casual, formal, or in-between.

### 🔗 Stays in the thread
AskMe replies as a threaded reply to your post, so conversations stay organized and easy to follow.

### ✨ Nice-to-read formatting
Cities, countries, landmarks, and @usernames are highlighted in **bold** so answers are scannable and easy on the eyes.

### 🔁 Never repeats itself
Each mention is handled exactly once, even if the network hiccups. No double replies, no spam.

---

## How to use it

1. Open Things and write a post (or reply to someone else's post).
2. Type **@AskMe** followed by your question.
3. Optionally attach a photo or video.
4. Post it and wait a few seconds — AskMe replies in a thread under your post.

> **Tip:** Want the best answers? Give context. Instead of "what is this?", try "this is a camera I'm thinking of buying — is it worth it?"

---

## Where can you find it?

AskMe is running round-the-clock on Things. If you've seen its replies around the feed, that's the bot you're talking to.

If you have ideas for new things AskMe should learn to do, feel free to say so in a post — you might even find it on the other end of the thread.

---

## Running it yourself

AskMe needs three things: the Things login (email + OTP), a Gemini API key, and a local [Qdrant](https://qdrant.tech) instance for its conversation memory.

```bash
# 1. Start Qdrant (Docker) — stores the bot's persistent memory
./scripts/start_qdrant.sh

# 2. Configure secrets
cp .env.example .env   # then fill in GEMINI_API_KEY, THINGS_EMAIL, THINGS_PASSWORD

# 3. Run the bot
cargo run
```

The bot keeps working even if Qdrant is down — it just falls back to a degraded
memory-less mode (no cross-restart dedup, no conversation recall).

### Environment variables

| Variable                | Default                    | Description                                        |
| ----------------------- | -------------------------- | -------------------------------------------------- |
| `GEMINI_API_KEY`        | — (required)               | Gemini API key for chat and embeddings.            |
| `THINGS_EMAIL`          | — (required)               | Things account email (login uses OTP).             |
| `THINGS_PASSWORD`       | — (required)               | Things account password.                           |
| `QDRANT_URL`            | `http://localhost:6334`    | Qdrant gRPC endpoint (port 6334, not 6333).        |
| `EMBEDDING_MODEL`       | `gemini-embedding-001`     | Embeddings model for indexing messages.            |
| `EMBEDDING_DIMENSIONS`  | `512`                      | Embedding vector size (must match collection).     |
| `EMBEDDING_BATCH_SIZE`  | `10`                       | Texts per `batchEmbedContents` call.               |
| `CONTEXT_SEARCH_LIMIT`  | `50`                       | Max hits returned by semantic memory recall.       |
| `CONTEXT_DEPTH_LIMIT`   | `20`                       | Max thread messages included in a follow-up prompt.|

### Developer tools

- `cargo run -- --test-post <id>` — dry-run: loads post `<id>`, prints the Qdrant
  memory state, the generated prompt, and the raw Gemini reply without posting.
  Add `--post` to actually post the reply, or `--prompt "text"` to override the prompt.
- `cargo test` — unit tests (entity formatting, memory serialization, embedder).

