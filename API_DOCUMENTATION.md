\# API Documentation



\---



\# Base URL



```text

http://127.0.0.1:8000/

```



\---



\# 1. Ask Question API



\## Endpoint



```text

POST /api/ask/

```



\---



\## Description



This endpoint receives a user question, retrieves the most relevant document chunks from the database, sends the context to the LLM using LangChain and OpenRouter, stores the interaction in history, and returns the generated answer.



\---



\## Request Headers



```http

Content-Type: application/json

```



\---



\## Request Body



```json

{

&#x20; "question": "What programming languages does Mahyar know?"

}

```



\---



\## Successful Response



\### Status Code



```text

200 OK

```



\### Response Body



```json

{

&#x20; "question": "What programming languages does Mahyar know?",

&#x20; "answer": "Java, Python, C, SQL",

&#x20; "context": "Retrieved document chunks...",

&#x20; "history\_id": 1

}

```



\---



\## Response Fields



| Field | Type | Description |

|---|---|---|

| question | string | User question |

| answer | string | Generated answer from LLM |

| context | string | Retrieved document chunks |

| history\_id | integer | Saved history record ID |



\---



\## Internal Workflow



1\. User submits a question

2\. Question is tokenized

3\. Relevant chunks are retrieved

4\. Prompt is constructed

5\. LangChain sends request to OpenRouter

6\. LLM generates response

7\. Result is stored in QuestionHistory model

8\. JSON response returned



\---



\## Possible Errors



\### LLM Rate Limit Error



```json

{

&#x20; "answer": "LLM service error: Error code: 429"

}

```



Cause:

\- Free model rate limit exceeded



Solution:

\- Retry later

\- Change model

\- Add OpenRouter credits



\---



\### Invalid API Key



```json

{

&#x20; "answer": "LLM service error: Error code: 401"

}

```



Cause:

\- Invalid OpenRouter API key



Solution:

\- Check `.env`



\---



\### Insufficient Credits



```json

{

&#x20; "answer": "LLM service error: Error code: 402"

}

```



Cause:

\- No OpenRouter credits



Solution:

\- Add credits

\- Use free model



\---



\# 2. Question History API



\## Endpoint



```text

GET /api/history/

```



\---



\## Description



Returns all stored questions, generated answers, retrieved contexts, and timestamps from the database.



\---



\## Successful Response



\### Status Code



```text

200 OK

```



\### Response Body



```json

\[

&#x20; {

&#x20;   "id": 1,

&#x20;   "question": "What university does Mahyar study at?",

&#x20;   "answer": "Mahyar studies at Sharif University of Technology.",

&#x20;   "retrieved\_context": "Relevant document chunks...",

&#x20;   "created\_at": "2026-05-19T04:43:55.521861Z"

&#x20; }

]

```



\---



\## Response Fields



| Field | Type | Description |

|---|---|---|

| id | integer | History record ID |

| question | string | User question |

| answer | string | Generated answer |

| retrieved\_context | string | Retrieved chunks |

| created\_at | datetime | Creation timestamp |



\---



\# AI System Design



\---



\## Retrieval Layer



Current retrieval system uses:



\- Keyword matching

\- Simple scoring mechanism

\- Chunk ranking



Each chunk receives a score based on overlap between question words and chunk content.



Top matching chunks are selected as context.



\---



\## Prompt Engineering



The prompt template:



```text

Answer the question using ONLY the provided context.



If the answer is not in the context, say:

"I could not find the answer in the uploaded documents."

```



This reduces hallucination and forces grounded answers.



\---



\## LangChain Integration



LangChain is used to:



\- Connect application to OpenRouter

\- Send prompts

\- Receive responses

\- Manage LLM abstraction



\---



\## OpenRouter Integration



The project uses OpenRouter as the LLM gateway.



Advantages:



\- Multiple models

\- Unified API

\- Free models available

\- Easy model switching



\---



\# Current Retrieval Limitations



Current retrieval is basic keyword matching.



Limitations:



\- No semantic understanding

\- No embeddings

\- Weak ranking quality

\- Sensitive to wording



\---



\# Planned Retrieval Improvements



Future improvements include:



\- BM25 ranking

\- Inverted Index

\- Embedding search

\- Vector databases

\- Hybrid retrieval



Possible vector databases:



\- ChromaDB

\- FAISS

\- Pinecone



\---



\# Docker Deployment



\---



\## Build Image



```bash

docker compose build --no-cache

```



\---



\## Run Container



```bash

docker compose up

```



\---



\## Run Migrations



```bash

docker compose exec web python manage.py migrate

```



\---



\## Create Superuser



```bash

docker compose exec web python manage.py createsuperuser

```



\---



\# Security Notes



Never upload:



\- `.env`

\- API keys

\- `venv/`

\- `db.sqlite3`



\---



\# Future Improvements



\- PDF support

\- Authentication

\- PostgreSQL

\- React frontend

\- Semantic retrieval

\- Streaming responses

\- Chat interface

\- Upload progress tracking

\- Swagger/OpenAPI support



\---



\# Author



Mahyar Rezaeepoor



GitHub:



```text

https://github.com/Mahyarelect

```

