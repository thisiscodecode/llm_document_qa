# API documentation

The base URL for local development is `http://127.0.0.1:8000/`.

## Authentication

Sign in at `/accounts/login/` before using the browser app. Document, chat,
history, and evaluation data are scoped to the authenticated user. Session
requests that upload, delete, or change data must send `X-CSRFToken` with the
`csrftoken` cookie. The `/api/ask/` and `/api/history/` endpoints also accept
HTTP Basic authentication over HTTPS for scripted clients. Login is available
to active nonstaff accounts; staff privileges are required for operational
endpoints.

## Documents

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/upload/` | Upload a DOCX or PDF; returns `202` with document and task IDs |
| `GET` | `/documents/` | List your documents |
| `GET` | `/documents/<id>/status/` | Check processing state and chunk count |
| `DELETE` | `/documents/<id>/delete/` | Remove your document and its vectors |

Upload with multipart form field `file`. Optional fields are `document_type`
(`default`, `code`, `legal`, `academic`, or `presentation`), `chunk_size`
(50–4000), and `chunk_overlap` (0 to less than `chunk_size`). Wait for the
document to become `ready` before asking about it. A missing or another
user's document returns `404`.

## Ask questions

`POST /api/ask/` accepts JSON:

```json
{
  "question": "What is the warranty period?",
  "search_method": "hybrid",
  "document_ids": [4],
  "session_id": 7
}
```

`question` is required (up to 5000 characters). `search_method` can be
`simple`, `bm25`, `vector`, or `hybrid` (default). Omit `document_ids` or send
`[]` to search all of your ready documents; explicit IDs must all be your
ready documents. Omit `session_id` to start a chat.

A successful response contains `answer`, `context`, `sources`, `history_id`,
`session_id`, `search_method`, and `request_id`. Each source includes its
document and chunk IDs, title, page, preview, and retrieval scores. An
example shortened response is:

```json
{
  "answer": "The warranty period is 12 months [Source 1].",
  "sources": [{"document_id": 4, "chunk_id": 12, "document": "Warranty.pdf", "page": 2}],
  "history_id": 18,
  "session_id": 7,
  "search_method": "hybrid",
  "request_id": "e62af017"
}
```

If no ready document exists, the response explains that processing is needed
and does not create a chat session. Invalid or inaccessible explicit document
IDs return `400`. Retrieval with no relevant content returns `422`; a model
or pipeline failure returns `503`. Failed answers are not saved to chat history.

## Chats and history

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/history/` | List your question history |
| `DELETE` | `/api/history/<id>/delete/` | Delete one history entry |
| `GET` | `/api/sessions/` | List your chats with messages and stored sources |
| `DELETE` | `/api/sessions/<id>/delete/` | Delete one chat and its messages |

## Operations and evaluation

`GET /health/live/` checks that the web process responds. `GET /health/ready/`
checks SQLite and Chroma connectivity. Both return only
`{"status":"ok"}` or, for failed readiness, `{"status":"unavailable"}`
with HTTP `503`.

Staff users can call `GET /api/stats/`, `GET /api/flags/`, and
`POST /api/flags/` to inspect or update runtime feature flags. A flag update body
contains `name` plus optional `rollout_pct` (0–100) and `variants` (a list of
1–10 names). Flag settings are in process memory and reset on restart.

Staff can call `GET /api/evaluate/?method=hybrid` or `GET /api/compare/` after
configuring `RAG_EVAL_DATASET` to point to a JSONL file. Each record needs a
`question` and `relevant_chunk_ids`; optional `document_ids` scope that case.
These routes evaluate the staff user's ready documents and report retrieval
metrics without calling an LLM. The equivalent command line tool is
`python manage.py evaluate_rag --dataset path/to/eval.jsonl --username your-user`.

Authentication failures return `401`, missing ownership or records return
`404`, and rate-limited `/api/ask/` requests return `429`.
