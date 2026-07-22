# DocuBot Model Card — Answering Modes Compared

DocuBot can answer a developer question in three ways. This card documents how
each mode behaves on the sample docs corpus, with concrete examples of where
each one is strong and where it breaks down.

| Mode | Method | Uses the docs? | Grounded? |
|------|--------|----------------|-----------|
| Naive generation | `GeminiClient.naive_answer_over_full_docs` | **No** — see note | No |
| Retrieval only | `DocuBot.answer_retrieval_only` | Yes (raw chunks) | Yes, but unsynthesized |
| RAG | `DocuBot.answer_rag` | Yes (retrieved chunks → LLM) | Yes, synthesized |

> **Important about naive mode:** `naive_answer_over_full_docs` currently
> *ignores* the `all_text` it is given and sends only the bare question to the
> model (`llm_client.py`: "We ignore all_text and send a generic prompt
> instead"). So naive answers come entirely from the model's prior knowledge,
> not from this project's documentation. That is exactly why it sounds fluent
> but is weakly grounded.

> **On the examples below:** retrieval-only outputs are **captured live** from
> the current code. Naive and RAG outputs are **illustrative reconstructions** —
> `google.genai` and a `GEMINI_API_KEY` are not available in the environment
> used to write this card, so the LLM could not be called. They reflect the
> behavior the prompts and code paths are designed to produce, not recorded runs.

---

## 1. Naive generation: confident but weakly grounded

**Query:** *"How long is an access token valid?"*

**Naive output (illustrative):**
> "Access tokens are typically valid for around 15 minutes, after which the
> client must request a new one using a refresh token."

**Why this is a problem:** it reads as authoritative, but it is a *generic*
answer from the model's training, not from these docs. The real, project-specific
answer lives in `AUTH.md`:

> `TOKEN_LIFETIME_SECONDS` — Controls how long a generated token remains valid.
> **Defaults to 3600 seconds if not set.**

The correct answer is **3600 seconds (1 hour) by default**. The naive "15
minutes" is plausibly worded and completely wrong for this application. Because
naive mode never reads `all_text`, it cannot know project-specific facts like the
`3600` default, the `generate_access_token` function name, or the `/api/refresh`
route — it will confidently invent plausible substitutes for any of them.

**Takeaway:** fluent tone is not evidence. Naive generation optimizes for sounding
right, not for being grounded in the actual docs.

---

## 2. Retrieval only: accurate but hard to interpret

**Query:** *"How long is an access token valid?"*

**Retrieval-only output (live):**
```
[AUTH.md]
## Environment Variables

The authentication system depends on two variables:

- `AUTH_SECRET_KEY`
  A secret string used to sign all access tokens. Must be long and unpredictable.

- `TOKEN_LIFETIME_SECONDS`
  Controls how long a generated token remains valid. Defaults to 3600 seconds if not set.

Both variables must be configured before starting the server.
---
[SETUP.md]
### Optional Variables

- `TOKEN_LIFETIME_SECONDS`
  Controls how long access tokens remain valid.

- `GEMINI_API_KEY`
  Enables LLM powered features ...
```

**Why this is a problem:** the answer *is* here and it *is* correct — nothing is
invented. But the user asked a one-line question and got two full paragraphs
about environment variables. The actual answer ("3600 seconds") is buried mid-chunk,
next to unrelated details (`AUTH_SECRET_KEY`, `GEMINI_API_KEY`). The user has to
read and extract it themselves.

**Takeaway:** retrieval-only trades interpretability for trustworthiness. It never
lies, but it makes the human do the synthesis.

---

## 3. RAG: balances clarity and evidence

**Query:** *"How long is an access token valid?"*

**RAG output (illustrative):**
> "By default, access tokens are valid for **3600 seconds (1 hour)**. This is
> controlled by the `TOKEN_LIFETIME_SECONDS` environment variable, which falls
> back to 3600 when unset. (Source: `AUTH.md`.)"

**Why this works:** RAG gets the *clarity* of naive generation (one direct
sentence) and the *grounding* of retrieval only (the number comes from the
retrieved `AUTH.md` chunk, and the file is cited). The LLM's job is narrowed to
summarizing evidence it was handed, not recalling facts from memory — so it states
the project's real default instead of a generic guess.

**Takeaway:** RAG is the best of both when — and only when — retrieval actually
surfaces the right chunk. That condition is where it can still fail.

---

## 4. Where RAG still fails

RAG is only as good as the chunks retrieval feeds it. When lexical (keyword)
retrieval ranks the wrong chunk first, RAG inherits the mistake.

**Query:** *"Which endpoint returns all users?"*

**Correct answer:** `GET /api/users` — "Returns a list of all users. Only
accessible to admins." (`API_REFERENCE.md`).

**Retrieval-only output, top 2 (live):**
```
[API_REFERENCE.md]
### POST /api/login
Authenticates a user and returns a short lived access token.
...
---
[DATABASE.md]
## Query Helpers
- get_user_by_id(user_id) ...
- get_all_users()   Returns a list of all user records.
...
```

**The failure:** the two top-ranked chunks are both wrong for this question:
- `POST /api/login` is the *login* endpoint — it ranks first only because its
  text literally contains the words "user", "returns", and "endpoint".
- `get_all_users()` is a **database helper function**, not a REST endpoint.

The genuinely correct chunk, `GET /api/users`, does **not** make the top 2 (it
ranks 3rd). So:
- With `top_k=2`, RAG never sees the right chunk and will either answer with the
  wrong endpoint or refuse.
- Even with `top_k=3`, the correct chunk arrives buried below two distractors,
  and the LLM may anchor on `get_all_users()` and report a *function* when the
  user asked for an *endpoint*.

**Root cause:** this is a limitation of **lexical retrieval**, not of the LLM. The
correct paragraph describes the endpoint without repeating the exact query words
as densely as the distractors do. Keyword matching cannot tell that
`GET /api/users` is more *about* "returning all users" than the login route is.

**What would fix it:** semantic retrieval (embeddings) that matches meaning rather
than surface words would rank `GET /api/users` first. That is a future phase
beyond the current keyword-based pipeline.

---

## Summary

| Dimension | Naive | Retrieval only | RAG |
|-----------|-------|----------------|-----|
| Clarity | High | Low | High |
| Grounding | Weak / none | Strong | Strong (when retrieval is right) |
| Interpretability | High | Low | High |
| Main failure mode | Confident hallucination | Burden on the reader | Inherits retrieval errors |

**Overall:** RAG is the best default for this corpus because it pairs a clear,
synthesized answer with cited evidence. But it is not self-correcting — it trusts
whatever retrieval hands it, so improving retrieval quality (better chunking,
and eventually semantic search) is the highest-leverage way to make RAG more
reliable. DocuBot's evidence guardrail (`has_sufficient_evidence`) mitigates the
worst case by refusing when even the best chunk is too weak, but it cannot fix a
case where a *wrong* chunk scores highly.
