# Policy Intelligence Assistant (Mini-RAG)

Production-oriented Mini-RAG application for querying policy PDFs with grounded answers, document attribution, and confidence scoring.

## Architecture

- `backend/`: FastAPI API, ingestion pipeline, retrieval, and LLM orchestration
- `frontend/`: Next.js chat application for Vercel
- `raw/`: drop source PDFs here before ingestion
- `faiss_index/`: generated FAISS index and chunk metadata

## Backend flow

1. `backend/ingest.py` loads PDFs from `raw/`
2. Text is extracted with `pdfplumber`
3. Text is cleaned and chunked at roughly `500-800` tokens with overlap
4. Each chunk stores `document_name`, `chunk_id`, `page_number`, and original text
5. Embeddings are generated through an OpenAI-compatible API
6. Vectors are stored in FAISS and metadata is saved to JSON
7. Runtime retrieval fetches top-K chunks and sends only those chunks to the LLM

## Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your PDFs to `raw/`, then run:

```bash
cd backend
PYTHONPATH=. python ingest.py
uvicorn app.main:app --reload --port 8000
```

## Frontend setup

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Set `NEXT_PUBLIC_BACKEND_URL` to your backend origin.

## Railway deployment

Create a Railway service from the `backend/` directory.

- Install command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Root directory: `backend`

Required environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` if using a non-default compatible endpoint
- `EMBEDDING_MODEL`
- `CHAT_MODEL`
- `CORS_ORIGINS`

Important:

- Run ingestion before deploying, or run it in Railway once PDFs are available in persistent storage.
- Persist the generated `faiss_index/` artifacts if you want retrieval to survive redeploys.

## Vercel deployment

Create a Vercel project from the `frontend/` directory.

- Framework preset: Next.js
- Environment variable: `NEXT_PUBLIC_BACKEND_URL=https://your-railway-backend.up.railway.app`

## API contract

`POST /api/chat`

Request:

```json
{
  "query": "What incentives are given to dealers?",
  "history": [
    {
      "role": "user",
      "content": "Previous question"
    }
  ],
  "top_k": 4
}
```

Response:

```json
{
  "answer": "...",
  "sources": [
    {
      "document": "Dealer Policy",
      "relevance": "high"
    }
  ],
  "confidence": "medium",
  "retrieved_chunks": 4
}
```

## Notes

- The system uses retrieval, not full-context stuffing.
- The LLM is instructed to answer only from retrieved chunks.
- If the answer is absent, the expected answer is `"Not present in documents"`.
