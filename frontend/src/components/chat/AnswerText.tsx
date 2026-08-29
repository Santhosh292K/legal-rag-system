import type { ReactNode } from "react";
import type { Citation } from "@/lib/types";
import { validityTone } from "@/components/ui/Badge";

const CHIP_TONE: Record<string, string> = {
  success: "bg-success-soft text-success",
  warning: "bg-warning-soft text-warning",
  danger: "bg-danger-soft text-danger",
  neutral: "bg-bg-subtle text-text-muted",
};

export function AnswerText({
  text,
  citations,
  messageId,
}: {
  text: string;
  citations: Citation[];
  messageId: string;
}) {
  const bySection = new Map(citations.map((c) => [c.section_id.toUpperCase(), c]));
  const parts: ReactNode[] = [];
  let last = 0;
  let key = 0;

  // A fresh regex per render — a module-level `/g` regex would share
  // mutable `lastIndex` state across concurrent/interrupted renders.
  for (const match of text.matchAll(/\[([A-Za-z0-9_./-]{2,40})\]/g)) {
    const [full, token] = match;
    const citation = bySection.get(token.toUpperCase());
    if (match.index > last) parts.push(text.slice(last, match.index));

    if (citation) {
      const tone = CHIP_TONE[validityTone(citation.validity)] ?? CHIP_TONE.neutral;
      parts.push(
        <a
          key={key++}
          href={`#citation-${messageId}-${citation.section_id}`}
          className={`mx-0.5 inline-flex items-center rounded px-1.5 py-0.5 font-mono text-[11.5px] font-medium leading-none no-underline transition-opacity hover:opacity-75 ${tone}`}
        >
          {citation.section_id}
        </a>,
      );
    } else {
      parts.push(
        <span key={key++} className="font-mono text-[11.5px] text-text-faint">
          {full}
        </span>,
      );
    }
    last = match.index + full.length;
  }
  if (last < text.length) parts.push(text.slice(last));

  return (
    <p className="whitespace-pre-wrap text-[14.5px] leading-relaxed text-text">{parts}</p>
  );
}
