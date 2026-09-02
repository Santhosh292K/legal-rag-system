"use client";

import { MessagesSquare, Trash2 } from "lucide-react";
import type { Conversation } from "@/lib/history";
import { groupByRecency } from "@/lib/history";
import { cn } from "@/lib/utils";

export function HistoryList({
  conversations,
  activeId,
  onSelect,
  onDelete,
}: {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  if (conversations.length === 0) {
    return (
      <p className="px-2 py-1 text-[12px] leading-relaxed text-text-faint">
        Your past questions will show up here once you ask one.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {groupByRecency(conversations).map(([bucket, items]) => (
        <div key={bucket}>
          <p className="mb-1 px-1 text-[10.5px] font-semibold uppercase tracking-wider text-text-faint">
            {bucket}
          </p>
          <div className="flex flex-col gap-0.5">
            {items.map((c) => (
              <div
                key={c.id}
                className={cn(
                  "group flex items-center gap-2 rounded-md border-l-[3px] border-transparent py-1.5 pr-2 pl-[calc(0.5rem-3px)] text-left text-[13px] transition-colors",
                  c.id === activeId
                    ? "border-l-accent bg-brand-soft font-medium text-text"
                    : "text-text-muted hover:bg-bg-subtle hover:text-text",
                )}
              >
                <button
                  type="button"
                  onClick={() => onSelect(c.id)}
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                >
                  <MessagesSquare size={13} className="shrink-0 opacity-70" />
                  <span className="truncate">{c.title}</span>
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(c.id);
                  }}
                  aria-label={`Delete "${c.title}"`}
                  title="Delete conversation"
                  className="shrink-0 rounded-md p-1 text-text-faint opacity-0 transition-opacity hover:bg-danger-soft hover:text-danger group-hover:opacity-100 focus-visible:opacity-100"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
