"use client";

import { AlertCircle, Scale, User } from "lucide-react";
import { useEffect, useState } from "react";
import type { ChatMessage } from "@/lib/types";
import { AnswerCard } from "./AnswerCard";

const THINKING_STEPS = [
  "Routing across legal domains",
  "Retrieving statutes (BM25 + dense)",
  "IRAC reranking",
  "Grounding the answer in citations",
];

export function PendingRow() {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setStep((s) => (s + 1) % THINKING_STEPS.length), 1600);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex animate-fade-up gap-3">
      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-brand text-brand-contrast">
        <Scale size={13} strokeWidth={2.25} />
      </span>
      <div className="flex min-w-0 flex-1 items-center gap-3 rounded-2xl rounded-tl-sm border border-border bg-bg-elevated px-4 py-3.5 shadow-sm">
        <span className="flex gap-1">
          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand [animation-delay:-0.3s]" />
          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand [animation-delay:-0.15s]" />
          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand" />
        </span>
        <span className="text-[13px] text-text-muted">{THINKING_STEPS[step]}</span>
      </div>
    </div>
  );
}

function ErrorRow({ text }: { text: string }) {
  return (
    <div className="flex animate-fade-up gap-3">
      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-danger-soft text-danger">
        <AlertCircle size={13} strokeWidth={2.25} />
      </span>
      <div className="min-w-0 flex-1 rounded-2xl rounded-tl-sm border border-danger/20 bg-danger-soft px-4 py-3.5 text-[13.5px] text-danger shadow-sm">
        {text}
      </div>
    </div>
  );
}

function UserRow({ text }: { text: string }) {
  return (
    <div className="flex animate-fade-up justify-end gap-3">
      <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-brand px-4 py-3 text-[14.5px] leading-relaxed text-brand-contrast shadow-sm">
        {text}
      </div>
      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-bg-subtle text-text-muted">
        <User size={13} strokeWidth={2.25} />
      </span>
    </div>
  );
}

export function MessageRow({ message }: { message: ChatMessage }) {
  switch (message.role) {
    case "user":
      return <UserRow text={message.text} />;
    case "pending":
      return <PendingRow />;
    case "error":
      return <ErrorRow text={message.text} />;
    case "assistant":
      return (
        <div className="animate-fade-up">
          <AnswerCard answer={message.answer} id={message.id} />
        </div>
      );
  }
}
