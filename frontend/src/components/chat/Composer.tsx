"use client";

import { ArrowUp } from "lucide-react";
import { useRef, useState } from "react";
import { cn } from "@/lib/utils";

export function Composer({
  onSend,
  disabled,
  activeCaseId,
}: {
  onSend: (text: string) => void;
  disabled: boolean;
  activeCaseId: string | null;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  function submit() {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
    if (ref.current) ref.current.style.height = "auto";
  }

  return (
    <div className="border-t border-border bg-bg/85 px-4 pb-4 pt-3 backdrop-blur-md sm:px-6">
      <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-border bg-bg-elevated p-2 shadow-sm transition-colors focus-within:border-brand">
        <textarea
          ref={ref}
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            e.target.style.height = "auto";
            e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={1}
          placeholder={
            activeCaseId
              ? `Ask about ${activeCaseId}, or Indian law generally…`
              : "Ask a question about Indian law…"
          }
          className="max-h-40 min-h-[40px] flex-1 resize-none bg-transparent px-2 py-2 text-[14.5px] text-text placeholder:text-text-faint outline-none"
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled || !value.trim()}
          aria-label="Send question"
          className={cn(
            "flex h-9 w-9 shrink-0 items-center justify-center rounded-xl transition-all",
            disabled || !value.trim()
              ? "bg-bg-subtle text-text-faint"
              : "bg-brand text-brand-contrast hover:opacity-90",
          )}
        >
          <ArrowUp size={16} strokeWidth={2.5} />
        </button>
      </div>
      <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-text-faint">
        Answers are generated from indexed statutes and are not a substitute for legal advice.
      </p>
    </div>
  );
}
