"use client";

import { AlertCircle, Scale, User } from "lucide-react";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import type { ChatMessage } from "@/lib/types";
import { AnswerCard } from "./AnswerCard";

// Plain-language mirror of the real pipeline stages — written for the
// person asking the question (public or lawyer), not a developer. Never
// name the technique (BM25, IRAC, dense retrieval, reranking) — say what
// it's actually doing for them instead. Keep in the same order as the
// real stages in rag/main.py so this stays an honest, if simplified,
// description of what's happening, not just decoration.
const THINKING_STEPS = [
  "Understanding your question",
  "Searching Indian law for relevant sections",
  "Checking which sections actually apply",
  "Writing an answer with citations",
];

export function PendingRow() {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setStep((s) => (s + 1) % THINKING_STEPS.length), 1600);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex gap-3">
      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-brand text-brand-contrast">
        <Scale size={13} strokeWidth={2.25} />
      </span>
      <div className="flex min-w-0 flex-1 items-center gap-3 rounded-2xl rounded-tl-sm border border-border bg-bg-elevated px-4 py-3.5 shadow-[var(--shadow-sm)]">
        {/* A single measured pulse rather than three bouncing dots — reads
            as a considered research process, not a casual chat "typing…"
            cue. Reuses the same pulse-dot keyframe HealthPill's loading
            state already uses. */}
        <span className="h-1.5 w-1.5 shrink-0 animate-pulse-dot rounded-full bg-accent" />
        <span className="text-[13px] text-text-muted" aria-live="polite">
          {THINKING_STEPS[step]}
        </span>
      </div>
    </div>
  );
}

function ErrorRow({ text }: { text: string }) {
  return (
    <div className="flex gap-3">
      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-danger-soft text-danger">
        <AlertCircle size={13} strokeWidth={2.25} />
      </span>
      <div className="min-w-0 flex-1 rounded-2xl rounded-tl-sm border border-danger/20 bg-danger-soft px-4 py-3.5 text-[13.5px] text-danger shadow-[var(--shadow-sm)]">
        {text}
      </div>
    </div>
  );
}

function UserRow({ text }: { text: string }) {
  return (
    <div className="flex justify-end gap-3">
      <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-brand px-4 py-3 text-[14.5px] leading-relaxed text-brand-contrast shadow-[var(--shadow-sm)]">
        {text}
      </div>
      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-bg-subtle text-text-muted">
        <User size={13} strokeWidth={2.25} />
      </span>
    </div>
  );
}

// A single motion wrapper for every row type, so entrance + the height
// transition from a small "thinking" bubble to a full answer card (same
// message id, content swapped in place — see page.tsx) both animate through
// one consistent spring instead of snapping.
export function MessageRow({ message }: { message: ChatMessage }) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 5 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
    >
      {message.role === "user" && <UserRow text={message.text} />}
      {message.role === "pending" && <PendingRow />}
      {message.role === "error" && <ErrorRow text={message.text} />}
      {message.role === "assistant" && (
        <AnswerCard answer={message.answer} id={message.id} />
      )}
    </motion.div>
  );
}
