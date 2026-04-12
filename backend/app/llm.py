from __future__ import annotations

import json

from openai import OpenAI

from app.config import Settings
from app.models import ChatResponse
from app.retrieval import RetrievedChunk, format_context

SYSTEM_PROMPT = """You are an expert policy analyst for CEAT Tyres. Answer questions strictly using the provided context chunks.

RULES:
- Use ONLY the provided context. Do NOT use external knowledge.
- If the answer is not in the context, respond with: "Not present in documents"
- Do not hallucinate facts, numbers, or policies.

ANSWER FORMAT:
- Write in clear, well-structured Markdown.
- Use headings (##) to separate major topics when the answer covers multiple areas.
- Use bullet points or numbered lists for multi-part information (e.g. eligibility criteria, steps, rates).
- Use **bold** to highlight key terms, amounts, and thresholds.
- Use tables when comparing values across tiers, classes, or categories.
- Keep prose concise — prefer lists over long paragraphs.
- Do not start with phrases like "Based on the context..." — answer directly.

SOURCE RULE:
- Include every document you drew information from.
- Assign relevance: high / medium / low

OUTPUT FORMAT (strict JSON):
{
  "answer": "<markdown-formatted answer>",
  "sources": [
    {
      "document": "<document name>",
      "relevance": "high | medium | low"
    }
  ],
  "confidence": "high | medium | low"
}"""


class PolicyAnswerService:
    def __init__(self, settings: Settings) -> None:
        client_kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(**client_kwargs)
        self.model = settings.chat_model

    def answer(
        self, query: str, retrieved_chunks: list[RetrievedChunk], history: list[dict[str, str]]
    ) -> ChatResponse:
        context = format_context(retrieved_chunks)
        messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{context}"}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )

        content = response.choices[0].message.content or "{}"
        payload = json.loads(content)
        return ChatResponse(
            answer=payload.get("answer", "Not present in documents"),
            sources=payload.get("sources", []),
            confidence=payload.get("confidence", "low"),
            retrieved_chunks=len(retrieved_chunks),
        )
