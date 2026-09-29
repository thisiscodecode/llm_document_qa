# 📄 LLM Document QA System

![LLM Document QA System Architecture](assets/architecture.png)

A Django RAG application for asking grounded questions about your own PDF and DOCX documents.

This project extracts text from documents, splits the content into searchable chunks, retrieves the most relevant parts, and sends them to an LLM through OpenRouter to generate grounded answers.

---

## 📌 Project Overview

**LLM Document QA System** is an authenticated web app and API for document-based question answering.

Users upload PDF or DOCX files. Background processing extracts text, creates
chunks, stores document metadata in SQLite, and indexes embeddings in Chroma.
When a question arrives, the backend retrieves relevant chunks with BM25,
vector search, or their hybrid rank fusion, then sends cited context to the
LLM. Documents and chats are isolated per user.

This makes the answer more focused, more document-aware, and more efficient than sending the full document every time.

### Useful For

* Internal document Q&A
* Knowledge-base assistants
* Report analysis
* Resume and profile analysis
* Business document review
* Academic document search
* Document-backed chat in the included web app

---

## 🧰 Tech Stack

| Part               | Technology                  |
| ------------------ | --------------------------- |
| Backend            | Django                      |
| API Layer          | Django REST Framework       |
| AI / LLM Layer     | LangChain, OpenRouter       |
| Document Parsing   | python-docx, PyPDF2         |
| Retrieval          | Chroma vector search, BM25, hybrid RRF |
| Database           | SQLite + Chroma             |
| Admin Panel        | Django Admin                |
| Environment Config | python-dotenv               |
| Containerization   | Docker, Docker Compose      |
| Language           | Python                      |

---

## 🏗️ Architecture

The project follows a practical RAG pipeline.

```text
PDF or DOCX document
     ↓
Authenticated web upload
     ↓
Background text extraction
     ↓
Chunking with page metadata
     ↓
SQLite + persistent Chroma vector store
     ↓
Authenticated question API
     ↓
BM25 + Chroma vector retrieval (hybrid by default)
     ↓
Reranking and cited context construction
     ↓
LangChain
     ↓
OpenRouter LLM
     ↓
Grounded answer with sources
     ↓
User-owned chat history
```

### 1. Document Upload Layer

Documents are uploaded in the web app or through `POST /upload/`.

After a PDF or DOCX upload, the background processor:

* Reads the uploaded file
* Extracts text from the document
* Stores the full extracted text
* Splits the document into smaller chunks
* Saves those chunks in the database
* Creates embeddings and upserts their vectors into Chroma

The status endpoint reports progress until the document becomes `ready` or
`failed`.

This prepares the document for fast retrieval later.

---

### 2. Retrieval Layer

When a question is submitted, the backend searches the saved chunks and selects the most relevant ones.

The project supports four search methods:

| Method   | Description                                     |
| -------- | ----------------------------------------------- |
| `simple` | Basic keyword-based matching                    |
| `bm25`   | Ranked information retrieval using BM25 scoring |
| `vector` | Semantic cosine search over Chroma embeddings   |
| `hybrid` | BM25 + vector search combined with reciprocal rank fusion |

Retrieval stays inside the caller's ready documents. Hybrid search combines
lexical and semantic rankings so exact terms and paraphrases can both surface.

---

### 3. Prompt Construction Layer

After the best chunks are selected, the backend builds a prompt for the LLM.

The prompt includes:

* The user question
* The retrieved document context
* Instructions to answer based only on the provided context

This helps keep answers grounded in the uploaded documents.

---

### 4. LLM Layer

The system uses LangChain to connect the backend to OpenRouter.

OpenRouter allows the project to use different LLM models through environment variables without changing the main application code.

---

### 5. History Layer

Successful questions and answers are saved to the user's chat history.

The saved history includes:

* User question
* Generated answer
* Retrieved context
* Cited source chunks
* Creation timestamp

This makes the system easier to test, debug, and review.

---

## 🔄 Data Flow

```text
User Question
   ↓
Django REST API
   ↓
Request Validation
   ↓
Search Stored Chunks
   ↓
Select Top Relevant Context
   ↓
Build LLM Prompt
   ↓
Send Request to OpenRouter
   ↓
Receive Answer
   ↓
Save Question History
   ↓
Return JSON Response
```

### Step-by-Step Flow

1. An authenticated user uploads a PDF or DOCX in the web app.
2. A background task extracts text and creates page-aware chunks.
3. Chunks are saved in SQLite and indexed in Chroma.
5. A user sends a question to the API.
6. The backend validates the request.
7. Retrieval finds and reranks chunks from that user's ready documents.
8. Selected chunks and source labels are used as context for the LLM.
9. LangChain sends the prompt to OpenRouter.
10. The LLM generates an answer.
11. A supported answer, context, and source chunks are saved in chat history.
12. The API returns the final answer as JSON.

---

## ✨ Features

### Core Features

* Authenticated PDF and DOCX upload
* Automatic text extraction
* Page-aware document chunking and Chroma indexing
* Retrieval-based question answering
* Simple keyword search
* BM25 search option
* LangChain integration
* OpenRouter LLM integration
* Question history storage
* Source citations and user-owned chat sessions
* Labeled retrieval evaluation
* Django Admin interface
* REST API endpoints
* Dockerized deployment

---

### Retrieval Features

* Search across a user's ready document chunks
* Rank chunks by relevance
* Select top matching chunks
* Choose search method per request
* Store retrieved context with each answer
* Return a fallback message when no useful context is found

---

### Admin Features

Django Admin can be used to manage and inspect:

* Uploaded documents
* Extracted document text
* Generated chunks
* Question history
* Retrieved context
* LLM-generated answers

This is helpful for testing, reviewing results, and improving the retrieval flow.

---

## 📁 Project Structure

```text
llm_document_qa/
│
├── config/                    # Django project configuration
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── documents/                 # Main document QA app
│   ├── admin.py               # Django Admin setup
│   ├── models.py              # Document, chunk, and history models
│   ├── serializers.py         # API serializers
│   ├── services.py            # Retrieval and LLM logic
│   ├── urls.py                # App routes
│   ├── views.py               # API views
│   └── migrations/
│
├── assets/                    # README and project assets
├── screenshots/               # Existing visual assets
├── API_DOCUMENTATION.md       # Detailed API documentation
├── Dockerfile
├── docker-compose.yml
├── manage.py
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/thisiscodecode/llm_document_qa.git
cd llm_document_qa
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Create `.env` File

Copy `.env.example` to `.env` and set the OpenRouter values. The file stays out of Git.
For local development, `DJANGO_DEBUG=true` permits a temporary session key. Set a
stable `DJANGO_SECRET_KEY` to keep sessions valid across restarts.

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=your_selected_model
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
CHROMA_PATH=chroma_db
DJANGO_DEBUG=true
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
```

Example:

```env
OPENROUTER_MODEL=openrouter/free
```

---

## 5. Run Migrations

```bash
python manage.py migrate
```

If the project already contains processed chunks, build Chroma after migrating:

```bash
python manage.py rebuild_chroma_index
```

---

## 6. Create Superuser

```bash
python manage.py createsuperuser
```

---

## 7. Run Development Server

```bash
python manage.py runserver
```

Server URL:

```text
http://127.0.0.1:8000/
```

Django login and Admin:

```text
http://127.0.0.1:8000/accounts/login/
http://127.0.0.1:8000/admin/
```

Sign in at `/accounts/login/` to use the app. The initial superuser can create
ordinary user accounts in Django Admin. Each user sees only their own uploaded
documents, chat sessions, and question history. Admin access is for staff users.

---

## 📤 Uploading Documents

Sign in, open the app at `/`, and upload a DOCX or PDF. The upload endpoint is
`POST /upload/` with multipart field `file`; optional `document_type`,
`chunk_size`, and `chunk_overlap` fields control chunking. Processing happens
in the background. Poll `/documents/<id>/status/` or watch the app for its
`ready` state before asking questions. The background queue runs inside the
web process, so interrupted jobs need reprocessing after a restart.

---

## 🔌 API Endpoints

Authentication is required. The browser uses a Django session and CSRF token;
`/api/ask/` and `/api/history/` also accept HTTP Basic authentication over
HTTPS. Users can access only their own records. Operational endpoints
(`/api/stats/`, `/api/flags/`, `/api/evaluate/`, `/api/compare/`) require staff.
`/health/live/` and `/health/ready/` are available for service probes.

## Ask Question

```http
POST /api/ask/
```

### Request

```json
{
  "question": "What is this document about?"
}
```

### Request with BM25 Search

```json
{
  "question": "What is this document about?",
  "search_method": "bm25"
}
```

### Response

```json
{
  "question": "What is this document about?",
  "search_method": "bm25",
  "answer": "The document explains...",
  "context": "Retrieved document chunks...",
  "history_id": 1
}
```

---

## Question History

```http
GET /api/history/
```

### Response

```json
[
  {
    "id": 1,
    "question": "What is this document about?",
    "answer": "The document explains...",
    "retrieved_context": "Relevant document chunks...",
    "created_at": "2026-05-19T04:43:55.521861Z"
  }
]
```

---

## 🔍 Search Methods

## Simple Search

Simple search compares words from the question with words inside each document chunk.

It is useful for basic matching and small document collections.

```json
{
  "question": "What skills are mentioned?",
  "search_method": "simple"
}
```

---

## BM25 Search

BM25 is a stronger ranking method for information retrieval.

It scores chunks based on how relevant they are to the question and returns the best matches.

```json
{
  "question": "What skills are mentioned?",
  "search_method": "bm25"
}
```

---

## Vector and Hybrid Search

Vector search uses Chroma cosine similarity. Hybrid search is the recommended
default: it fuses Chroma semantic results with BM25 lexical results before
reranking and building a token-bounded context.

```json
{
  "question": "What skills are mentioned?",
  "search_method": "hybrid"
}
```

---

## 🐳 Docker Setup

Set `DJANGO_SECRET_KEY` in `.env` before starting Docker. Generate a strong
value with `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"`.
Compose runs with `DJANGO_DEBUG=false`, applies migrations, collects static
files, and starts one Gunicorn worker. It binds to `127.0.0.1:8000`; place an
HTTPS reverse proxy in front of it for remote access and configure
`DJANGO_ALLOWED_HOSTS`, `DJANGO_CSRF_TRUSTED_ORIGINS`,
`DJANGO_SESSION_COOKIE_SECURE=true`, and `DJANGO_CSRF_COOKIE_SECURE=true` for
that host. Compose leaves secure cookies off for its localhost HTTP binding.
SQLite,
uploaded media, and Chroma each have persistent Docker volumes.

## Build Docker Image

```bash
docker compose build
```

---

## Run Container

```bash
docker compose up -d
```

---

## Create Superuser Inside Docker

```bash
docker compose exec web python manage.py createsuperuser
```

The API will be available at:

```text
http://127.0.0.1:8000/
```

---

## ⚙️ Environment Variables

| Variable             | Required | Description                           |
| -------------------- | -------- | ------------------------------------- |
| `OPENROUTER_API_KEY` | Yes      | API key used to connect to OpenRouter |
| `OPENROUTER_MODEL`   | Yes      | Selected LLM model name               |
| `OPENROUTER_EMBEDDING_MODEL` | No | Embedding model (defaults to `openai/text-embedding-3-small`) |
| `CHROMA_PATH` | No | Local persistent Chroma directory (defaults to `chroma_db`) |
| `CHROMA_HOST` | No | Chroma server hostname for production client/server mode |
| `CHROMA_PORT` | No | Chroma server port (defaults to `8000`) |
| `DJANGO_DEBUG` | No | Local development only; Docker forces `false` |
| `DJANGO_SECRET_KEY` | Docker/production | Stable secret for signed sessions and CSRF |
| `DJANGO_ALLOWED_HOSTS` | Production | Comma-separated public hostnames |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | If behind another origin | Comma-separated trusted HTTPS origins |
| `DJANGO_DB_PATH` | No | SQLite location; Docker uses `/app/data/db.sqlite3` |
| `DJANGO_MEDIA_ROOT` | No | Uploaded file directory; Docker uses `/app/media` |
| `RAG_EVAL_DATASET` | For evaluation API | Path to a labeled JSONL retrieval dataset |

Keep these values private and never commit them to Git.

---

## 🧪 Example Workflow

1. Start the Django server and sign in at `/accounts/login/`.
2. Upload a DOCX or PDF document in the app.
3. Wait for the document status to become `ready` after extraction and indexing.
4. Ask a question in the app or send it to `/api/ask/`.
5. Use the default `hybrid` search (or select `simple`, `bm25`, or `vector`).
6. Review the cited sources and the saved chat at `/api/history/`.

---

## Retrieval evaluation

Build a JSONL file with questions and ground-truth chunk IDs from your own
corpus. For example, one line can be:

```json
{"question":"What is the warranty period?","relevant_chunk_ids":[12,13],"document_ids":[4]}
```

Run `python manage.py evaluate_rag --dataset path/to/eval.jsonl --username your-user`
to compare simple, BM25, vector, and hybrid retrieval. Use `--method hybrid`
for one method and `--k 5` for a cutoff. The report includes hit rate,
recall, precision, reciprocal rank, and latency. The staff-only evaluation
API uses the path in `RAG_EVAL_DATASET` and returns a configuration error if
no dataset is set. Evaluation uses retrieval only; it does not call an LLM.

## Existing anonymous data

Older versions stored documents and chats without an owner. Those records are
hidden after login is enabled. If all of that legacy data belongs to one
account, choose its username and run this one-time command locally after a
database backup:

```bash
python manage.py shell -c "from django.contrib.auth import get_user_model; from documents.models import Document, ChatSession, QuestionHistory; user=get_user_model().objects.get(username='your-username'); Document.objects.filter(owner__isnull=True).update(owner=user); ChatSession.objects.filter(owner__isnull=True).update(owner=user); QuestionHistory.objects.filter(owner__isnull=True).update(owner=user)"
```

If records belong to different users, assign selected IDs separately instead
of applying the command to every ownerless record.

## Security and deployment

All document and chat routes require login. Session-authenticated write
requests require CSRF tokens. Staff-only controls include runtime statistics,
feature flags, and retrieval evaluation. The answer endpoint is rate-limited
to 10 requests per minute per user by default. Use HTTPS for production,
set a persistent `DJANGO_SECRET_KEY`, restrict allowed hosts, and configure
secure cookies and trusted origins behind your reverse proxy. Keep `.env`,
database files, uploaded media, and vector data out of source control.

---

## ⚠️ Current Limitations

* SQLite remains the default relational database.
* Background processing uses an in-process queue and does not survive a web
  process restart. A durable worker queue is needed for larger deployments.
* Responses are not streamed.

---

## 🧭 Future Improvements

Possible improvements for future versions:

* PostgreSQL support
* Streaming LLM responses
* Durable background jobs and retry scheduling

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👥 Author

Built by the **codecode** team.
