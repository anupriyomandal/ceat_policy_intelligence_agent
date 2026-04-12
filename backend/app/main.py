from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.embeddings import EmbeddingService
from app.llm import PolicyAnswerService
from app.models import ChatRequest, ChatResponse
from app.retrieval import Retriever
from app.vector_store import LocalFaissStore


class AppState:
    settings: Settings
    retriever: Retriever | None
    answer_service: PolicyAnswerService | None
    startup_error: str | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state = AppState()
    state.settings = settings
    state.retriever = None
    state.answer_service = None
    state.startup_error = None

    try:
        store = LocalFaissStore.load(settings.index_file, settings.store_file)
        embeddings = EmbeddingService(settings)
        state.retriever = Retriever(store=store, embeddings=embeddings)
        state.answer_service = PolicyAnswerService(settings)
    except FileNotFoundError as exc:
        state.startup_error = str(exc)

    app.state.services = state
    yield


app = FastAPI(
    title="Policy Intelligence Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    services: AppState = app.state.services
    if services.startup_error:
        return {"status": "degraded", "detail": services.startup_error}
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    services: AppState = app.state.services
    if services.startup_error or not services.retriever or not services.answer_service:
        raise HTTPException(
            status_code=503,
            detail=services.startup_error
            or "Retrieval service is unavailable. Run ingestion first.",
        )

    top_k = request.top_k or services.settings.top_k

    retrieved = services.retriever.retrieve(request.query, top_k=top_k)
    if not retrieved:
        return ChatResponse(
            answer="Not present in documents",
            sources=[],
            confidence="low",
            retrieved_chunks=0,
        )

    history = [message.model_dump() for message in request.history]
    return services.answer_service.answer(
        query=request.query,
        retrieved_chunks=retrieved,
        history=history,
    )
