> **العربية:** [اقرأ النسخة العربية](README.ar.md)

# AskMe — the Things AI bot 🤖

Meet **AskMe**, a friendly AI assistant that lives on the [Things](https://things.cv) social network. When you need a quick answer, an opinion, or a look at a photo, just @mention AskMe — it will jump in and reply.

---

## What AskMe can do

### 🗣️ Answer your questions
Mention **@AskMe** in a post or in a reply, and it will answer right away. Ask about anything — a fact, an idea, an opinion, a "what should I do?" moment.

### 💬 Follows the conversation
AskMe reads the post you're replying to, so it understands the context. If you follow up with a second question, it **remembers what was said before** — even hours later — so you can keep the conversation going naturally. Each conversation is isolated: what was said under one post never leaks into another post's thread.

### 🧠 Remembers you
AskMe keeps a long-term memory of durable facts you share about yourself — where you live, what you do, your preferences — and brings them into future conversations, just like ChatGPT memory. Try "@AskMe describe me" to see what it knows about you, or "@AskMe forget that I live in Riyadh" to make it drop a fact.

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

### 🎮 Plays games with you
Say "@AskMe نلعب" and it hosts one of 15 text games in the thread — المشنوق (hangman), 20 سؤال, إنسان حيوان جماد, تابو, ألغاز, تخمين الشخصيات, اختر مغامرتك, أكمل القصة, خمّن من الإيموجي, حقيقتان وكذبة, لو خيروك, مسابقة الثقافة, صح أم خطأ, سلسلة الكلمات, and إكس أو. It keeps the game state (and its secret answers) server-side, tracks the score, and remembers your all-time record across threads.

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

## Admin panel

Once the bot starts it serves a web admin panel (default `http://localhost:1330`,
also reachable at `http://<machine-ip>:1330`; the exact URLs are printed at
startup). Sign in with the default password **`CHANGEME`** — the panel forces
you to set a new one before anything else works.

The panel offers:

- **Dashboard** — uptime, Qdrant status with per-collection point counts, Things
  auth health, **API key pool** state, live configuration.
- **Configuration** — every bot knob. Memory/retrieval settings and the **Gemini
  API key pool** apply *instantly*; boot-time settings (Qdrant URL, embedding
  model) are applied on restart. Saved to `bot_config.json` (overrides `.env`).
- **Logs** — live bot log with level filter, auto-scroll and one-click copy.
- **Security** — change the admin password.
- **Danger Zone** — wipe memory (markers kept) / wipe everything (typed
  confirmation), and restart the bot.

### Gemini API key pool

To survive tight free-tier rate limits you can give the bot **any number of
Gemini API keys** (panel → Configuration, or `GEMINI_API_KEYS=k1,k2,k3` in
`.env`):

- Successful replies rotate through the pool **round-robin**.
- Background calls (embeddings, fact extraction) rotate per request.
- On a **429** the key enters an automatic cooldown (parsed from Google's
  `RetryInfo`, or until the daily reset for per-day quotas) and the request
  **instantly fails over to the next key** — you only wait when *every* key is
  cooling.
- Keys that return 401/403 are marked **dead** and skipped. Media uploads stay
  pinned to the generating key (Gemini Files are project-scoped); on failover
  media is re-uploaded automatically.

The dashboard shows each key masked with its state (active / cooldown / daily
cap / dead) and request/429 counters.

> ⚠️ The panel is plain HTTP on `0.0.0.0` — use it on a **trusted LAN only**
> (no TLS). Bind to `127.0.0.1` via `ADMIN_BIND` if you want it local-only.

### Run as a service (background + survives reboots)

```bash
./scripts/install_service.sh
```

This builds the release binary, installs a systemd unit (`Restart=always`),
enables it at boot and starts it immediately. The first-ever OTP login must be
done in the foreground (`cargo run`) beforehand — the cached `.token.json` is
reused by the service afterwards. Logs: `journalctl -u askme-bot -f`.

The bot keeps working even if Qdrant is down — it just falls back to a degraded
memory-less mode (no cross-restart dedup, no conversation recall). If Qdrant is
reachable but misconfigured (e.g. an embedding-dimension mismatch), the bot
exits with a clear error instead of running half-broken.

### Memory architecture

AskMe's memory is split into three strictly-scoped Qdrant collections:

- **`conversation_memory`** — episodic memory. Each post where the bot is
  @mentioned starts its own conversation (the replies under it included), and
  context is only ever read per conversation — conversations are fully isolated
  from each other, even inside one big Things thread.
- **`user_profiles`** — durable per-user facts, extracted by a background pass
  after each reply. This is the only memory that crosses conversations, and it
  is always scoped to exactly one user. Restating a fact reinforces it;
  contradicting it supersedes the old fact; asking to forget deactivates it.
- **`things_knowledge`** — curated facts about the Things app, seeded from
  `things_knowledge.json` on every boot. Facts users *claim* about the app are
  stored as `pending` and never injected into prompts until promoted. App
  knowledge only enters a prompt when the question is actually about the app
  (score-gated semantic search).

Reliability notes:

- On the **very first boot** (empty memory), the bot silently marks the existing
  notification backlog as processed — it only answers mentions that arrive
  after startup, instead of replying to history.
- Every poll fetches notification pages until all unread items are covered, so
  mentions buried past page 1 are never missed. Notifications that fail to
  process (deleted post, Gemini error, ...) are retried a few times before
  being skipped, and are only marked read once handled.
- If the Things token expires (HTTP 401), the bot deletes the stale
  `.token.json` and exits with a clear error — restart it to log in again.

### Environment variables

| Variable                | Default                    | Description                                        |
| ----------------------- | -------------------------- | -------------------------------------------------- |
| `GEMINI_API_KEY`        | — (required*)              | Gemini API key for chat and embeddings.            |
| `GEMINI_API_KEYS`       | —                          | Comma-separated key pool (alternative to single).  |
| `THINGS_EMAIL`          | — (required)               | Things account email (login uses OTP).             |
| `THINGS_PASSWORD`       | — (required)               | Things account password.                           |
| `QDRANT_URL`                    | `http://localhost:6334`    | Qdrant gRPC endpoint (port 6334, not 6333).            |
| `EMBEDDING_MODEL`               | `gemini-embedding-2`       | Embeddings model (changing wipes vector memory).       |
| `EMBEDDING_DIMENSIONS`          | `512`                      | Embedding vector size (must match collection).         |
| `EMBEDDING_BATCH_SIZE`          | `10`                       | Texts per `batchEmbedContents` call.                   |
| `CONTEXT_DEPTH_LIMIT`           | `20`                       | Max conversation messages included in a prompt.        |
| `FACT_EXTRACTION_ENABLED`       | `true`                     | Background extraction of long-term user/app facts.     |
| `USER_FACTS_LIMIT`              | `8`                        | Max user facts injected into a prompt.                 |
| `GENERATION_MODEL`              | `gemini-3.6-flash`         | Chat model for replies (hot-reloadable via panel).     |
| `THINKING_LEVEL`                | — (model default)          | `minimal`/`low`/`medium`/`high` (hot via panel).       |
| `EXTRACTION_THINKING_LEVEL`     | `low`                      | Thinking level for extraction/FAQ/rewrite (hot via panel). |
| `MEDIA_RESOLUTION`              | — (model default)          | `low`/`medium`/`high` media token budget (restart).    |
| `SEARCH_GROUNDING_ENABLED`      | `false`                    | Google Search grounding; replaces the custom web_search when on. Billed per executed query past the free allowance (hot via panel). |
| `GAMES_ENABLED`                 | `true`                     | Gaming mode: the bot hosts 15 text games (hangman, 20 questions, trivia, ...) with per-thread state and all-time player scores (hot via panel). |
| `USER_FACT_SUPERSEDE_THRESHOLD` | `0.78`                     | Similarity at which a new fact supersedes an old one.  |
| `FORGET_SIMILARITY_THRESHOLD`   | `0.75`                     | Similarity for locating facts a user asked to forget.  |
| `APP_KNOWLEDGE_LIMIT`           | `3`                        | Max app-knowledge facts injected into a prompt.        |
| `APP_KNOWLEDGE_MIN_SCORE`       | `0.72`                     | Min semantic score for app knowledge to be injected.   |
| `ADMIN_BIND`                    | `0.0.0.0`                  | Admin panel bind address.                              |
| `ADMIN_PORT`                    | `1330`                     | Admin panel port.                                      |

### Developer tools

- `cargo run -- --test-post <id>` — dry-run: loads post `<id>`, prints the Qdrant
  memory state (conversation context, user facts, app knowledge), the generated
  prompt, and the raw Gemini reply without posting.
  Add `--post` to actually post the reply, or `--prompt "text"` to override the prompt.
- `cargo run -- --reset-memory` — wipe the three memory collections
  (`conversation_memory`, `user_profiles`, `things_knowledge`) and recreate them
  empty. Processed-notification markers are kept, so no old mention is
  re-answered. Run once when switching to a new memory schema.
- `cargo test` — unit tests (entity formatting, memory serialization, extraction
  parsing, embedder).

