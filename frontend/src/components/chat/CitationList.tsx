"use client";

import { AlertTriangle, ChevronDown } from "lucide-react";
import { useState } from "react";
import type { Citation } from "@/lib/types";
import { Badge, validityTone } from "@/components/ui/Badge";
import { CopyButton } from "@/components/ui/CopyButton";
import { cn } from "@/lib/utils";

function CitationCard({ citation, id }: { citation: Citation; id: string }) {
  const [expanded, setExpanded] = useState(false);
  const long = citation.content.length > 220;

  return (
    <div
      id={id}
      className="group scroll-mt-20 rounded-xl border border-border bg-bg-elevated p-3.5 transition-shadow target:ring-2 target:ring-brand/50"
    >
      <div className="mb-1.5 flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="rounded-md bg-bg-subtle px-1.5 py-0.5 font-mono text-[12px] font-semibold text-text">
          {citation.section_id}
        </span>
        <Badge tone={validityTone(citation.validity)} dot>
          {citation.validity}
        </Badge>
        <Badge tone="neutral">{citation.category}</Badge>
        <CopyButton
          text={`${citation.section_id} — ${citation.act_name}\n${citation.content}`}
          label="Copy citation"
          className="ml-auto p-1 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
        />
      </div>
      <p className="mb-1 text-[12.5px] font-medium text-text-muted">{citation.act_name}</p>
      <p
        className={cn(
          "whitespace-pre-wrap text-[13px] leading-relaxed text-text-muted",
          !expanded && long && "line-clamp-3",
        )}
      >
        {citation.content}
      </p>
      {long && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-1 flex items-center gap-1 text-[11.5px] font-medium text-brand hover:opacity-80"
        >
          {expanded ? "Show less" : "Show more"}
          <ChevronDown size={12} className={cn("transition-transform", expanded && "rotate-180")} />
        </button>
      )}
      {citation.warning && (
        <p className="mt-2 flex items-start gap-1.5 rounded-lg bg-warning-soft px-2 py-1.5 text-[11.5px] text-warning">
          <AlertTriangle size={12} className="mt-0.5 shrink-0" />
          {citation.warning}
        </p>
      )}
    </div>
  );
}

export function CitationList({
  citations,
  messageId,
}: {
  citations: Citation[];
  messageId: string;
}) {
  if (citations.length === 0) return null;

  return (
    <div>
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-text-faint">
        Citations · {citations.length}
      </p>
      <div className="grid gap-2.5 sm:grid-cols-2">
        {citations.map((c) => (
          <CitationCard
            key={c.section_id}
            citation={c}
            id={`citation-${messageId}-${c.section_id}`}
          />
        ))}
      </div>
    </div>
  );
}
