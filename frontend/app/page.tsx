"use client";

import { FormEvent, useState, useTransition } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import styles from "./page.module.css";

type Role = "user" | "assistant";

type Message = {
  role: Role;
  content: string;
  sources?: { document: string; relevance: string }[];
  confidence?: string;
};

async function askPolicyAssistant(query: string, history: Message[]) {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
  if (!backendUrl) {
    throw new Error("NEXT_PUBLIC_BACKEND_URL is not configured");
  }

  const response = await fetch(`${backendUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      history: history.map((m) => ({ role: m.role, content: m.content })),
    }),
  });

  if (!response.ok) throw new Error("Backend request failed");
  return response.json();
}


// Collapse blank lines between list items so ReactMarkdown renders tight lists
// (no <p> wrappers inside <li>), eliminating the extra vertical gaps.
function tightenLists(md: string): string {
  return md.replace(/^([ \t]*[-*+].*)\n\n(?=[ \t]*[-*+])/gm, "$1\n");
}

function ConfidenceBadge({ confidence }: { confidence?: string }) {
  if (!confidence) return null;
  return (
    <span className={styles.confidenceBadge} data-level={confidence}>
      {confidence} confidence
    </span>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  const submitQuery = (question: string) => {
    const trimmed = question.trim();
    if (!trimmed) return;

    const nextHistory = [...messages, { role: "user" as const, content: trimmed }];
    setMessages(nextHistory);
    setQuery("");
    setError(null);

    startTransition(async () => {
      try {
        const result = await askPolicyAssistant(trimmed, messages);
        setMessages([
          ...nextHistory,
          {
            role: "assistant",
            content: result.answer,
            sources: result.sources,
            confidence: result.confidence,
          },
        ]);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong.");
      }
    });
  };

  const onSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    submitQuery(query);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuery(query);
    }
  };

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <h1>CEAT Policy Intelligence Agent</h1>
      </header>

      <div className={styles.messages}>
        {messages.map((message, index) => (
          <article
            key={index}
            className={styles.message}
            data-role={message.role}
          >
            <div className={styles.messageHeader}>
              <span>{message.role === "assistant" ? "Assistant" : "You"}</span>
              <ConfidenceBadge confidence={message.confidence} />
            </div>
            <div className={styles.messageBubble}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{tightenLists(message.content)}</ReactMarkdown>
            </div>
            {message.sources && message.sources.length > 0 && (
              <div className={styles.sources}>
                {message.sources.map((source) => (
                  <div key={`${source.document}-${source.relevance}`} className={styles.source}>
                    <span>{source.document}</span>
                    <strong>{source.relevance}</strong>
                  </div>
                ))}
              </div>
            )}
          </article>
        ))}

        {isPending && (
          <article className={styles.message} data-role="assistant">
            <div className={styles.messageHeader}><span>Assistant</span></div>
            <div className={styles.messageBubble}>Thinking…</div>
          </article>
        )}
      </div>

      <form className={styles.composer} onSubmit={onSubmit}>
        <textarea
          rows={3}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask a question about CEAT policies..."
          disabled={isPending}
        />
        <div className={styles.composerFooter}>
          {error && <p className={styles.error}>{error}</p>}
          <button type="submit" disabled={isPending || !query.trim()}>
            Send
          </button>
        </div>
      </form>

      <footer className={styles.footer}>Made by Anupriyo Mandal</footer>
    </main>
  );
}
