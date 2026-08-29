"use client";

import { ArrowUp } from "lucide-react";
import { useRef, useState } from "react";
import { cn } from "@/lib/utils";

const MAX_LENGTH = 4000; // matches the backend's QueryRequest schema

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
  const [focused, setFocused] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);

  function submit() {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
    if (ref.current) ref.current.style.height = "auto";
  }

  const nearLimit = value.length > MAX_LENGTH * 0.9;

  return (
    <div className="border-t border-border bg-bg/85 px-4 pb-4 pt-3 backdrop-blur-md sm:px-6">
      <div
        className={cn(
          "mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border bg-bg-elevated p-2 shadow-[var(--shadow-sm)] transition-all",
          focused ? "border-brand shadow-[var(--shadow-brand)]" : "border-border",
        )}
      >
        <textarea
          ref={ref}
          value={value}
          maxLength={MAX_LENGTH}
          onChange={(e) => {
            setValue(e.target.value);
            e.target.style.height = "auto";
            e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
          }}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
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
            "flex h-9 w-9 shrink-0 items-center justify-center rounded-xl transition-all active:scale-95",
            disabled || !value.trim()
              ? "bg-bg-subtle text-text-faint"
              : "bg-brand text-brand-contrast hover:opacity-90",
          )}
        >
          <ArrowUp size={16} strokeWidth={2.5} />
        </button>
      </div>
      <div className="mx-auto mt-2 flex max-w-3xl items-center justify-center gap-3 text-center text-[11px] text-text-faint">
        <p>Answers are generated from indexed statutes and are not a substitute for legal advice.</p>
        {nearLimit && (
          <span className={cn(value.length >= MAX_LENGTH && "text-warning")}>
            {value.length}/{MAX_LENGTH}
          </span>
        )}
      </div>
    </div>
  );
}
