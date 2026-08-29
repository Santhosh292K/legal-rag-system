"use client";

import { FileText, FolderClosed, MessagesSquare, Plus, SquarePen } from "lucide-react";
import { useState } from "react";
import type { Conversation } from "@/lib/history";
import type { CaseInfo } from "@/lib/types";
import { cn, initials } from "@/lib/utils";
import { UploadDropzone } from "@/components/cases/UploadDropzone";
import { HistoryList } from "@/components/history/HistoryList";

const CASE_ID_PATTERN = /^[A-Za-z0-9_.-]{1,64}$/;

export function Sidebar({
  cases,
  activeCaseId,
  onSelectCase,
  onCreateCase,
  onUploaded,
  open,
  conversations,
  activeConversationId,
  onSelectConversation,
  onDeleteConversation,
  onNewChat,
}: {
  cases: CaseInfo[];
  activeCaseId: string | null;
  onSelectCase: (caseId: string | null) => void;
  onCreateCase: (caseId: string) => void;
  onUploaded: () => void;
  open: boolean;
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onNewChat: () => void;
}) {
  const [draft, setDraft] = useState("");
  const [formError, setFormError] = useState<string | null>(null);

  function submitNewCase(e: React.FormEvent) {
    e.preventDefault();
    const id = draft.trim();
    if (!id) return;
    if (!CASE_ID_PATTERN.test(id)) {
      setFormError("Use letters, numbers, dot, dash or underscore only.");
      return;
    }
    onCreateCase(id);
    setDraft("");
    setFormError(null);
  }

  const activeCase = cases.find((c) => c.case_id === activeCaseId) ?? null;

  return (
    <aside
      className={cn(
        "fixed inset-y-0 left-0 z-40 flex w-72 shrink-0 -translate-x-full flex-col gap-5 overflow-y-auto border-r border-border bg-bg-elevated px-4 py-5 shadow-[var(--shadow-lg)] transition-transform duration-200 lg:static lg:z-auto lg:translate-x-0 lg:shadow-none",
        open && "translate-x-0",
      )}
    >
      <button
        type="button"
        onClick={onNewChat}
        className="flex w-full items-center gap-2.5 rounded-lg border border-border px-3 py-2 text-left text-sm font-medium text-text transition-colors hover:border-border-strong hover:bg-bg-subtle"
      >
        <SquarePen size={15} />
        New chat
      </button>

      <div>
        <p className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-wider text-text-faint">
          History
        </p>
        <HistoryList
          conversations={conversations}
          activeId={activeConversationId}
          onSelect={onSelectConversation}
          onDelete={onDeleteConversation}
        />
      </div>

      <div className="border-t border-border pt-4">
        <p className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-wider text-text-faint">
          Consultation
        </p>
        <button
          type="button"
          onClick={() => onSelectCase(null)}
          className={cn(
            "flex w-full items-center gap-2.5 rounded-lg border px-3 py-2 text-left text-sm transition-colors",
            activeCaseId === null
              ? "border-brand/40 bg-brand-soft text-brand font-medium"
              : "border-transparent text-text-muted hover:bg-bg-subtle",
          )}
        >
          <MessagesSquare size={15} />
          General question
        </button>
      </div>

      <div>
        <p className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-wider text-text-faint">
          Cases
        </p>

        <form onSubmit={submitNewCase} className="mb-2.5 flex gap-1.5 px-0.5">
          <input
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value);
              setFormError(null);
            }}
            placeholder="e.g. case-2024-118"
            className="min-w-0 flex-1 rounded-lg border border-border bg-bg px-2.5 py-1.5 text-[13px] text-text placeholder:text-text-faint outline-none transition-colors focus:border-brand"
          />
          <button
            type="submit"
            aria-label="Create case"
            className="inline-flex h-[30px] w-[30px] shrink-0 items-center justify-center rounded-lg bg-brand text-brand-contrast transition-opacity hover:opacity-90 disabled:opacity-40"
            disabled={!draft.trim()}
          >
            <Plus size={15} />
          </button>
        </form>
        {formError && <p className="mb-2 px-1 text-[11px] text-danger">{formError}</p>}

        <div className="flex flex-col gap-1">
          {cases.length === 0 && (
            <p className="px-2 py-1 text-[12px] leading-relaxed text-text-faint">
              No cases yet — create one to upload FIRs, charge sheets, or evidence and
              get answers grounded in that document.
            </p>
          )}
          {cases.map((c) => (
            <button
              key={c.case_id}
              type="button"
              onClick={() => onSelectCase(c.case_id)}
              className={cn(
                "flex items-center gap-2.5 rounded-lg border px-3 py-2 text-left text-sm transition-colors",
                activeCaseId === c.case_id
                  ? "border-brand/40 bg-brand-soft text-brand font-medium"
                  : "border-transparent text-text-muted hover:bg-bg-subtle",
              )}
            >
              <span
                className={cn(
                  "flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-[10px] font-semibold",
                  activeCaseId === c.case_id
                    ? "bg-brand text-brand-contrast"
                    : "bg-bg-subtle text-text-faint",
                )}
              >
                {initials(c.case_id)}
              </span>
              <span className="min-w-0 flex-1 truncate">{c.case_id}</span>
              <span className="shrink-0 text-[10.5px] text-text-faint">
                {c.documents.length}
              </span>
            </button>
          ))}
        </div>
      </div>

      {activeCaseId && (
        <div className="border-t border-border pt-4">
          <p className="mb-2 flex items-center gap-1.5 px-1 text-[11px] font-semibold uppercase tracking-wider text-text-faint">
            <FolderClosed size={11} /> Documents in {activeCaseId}
          </p>

          {activeCase && activeCase.documents.length > 0 && (
            <ul className="mb-3 flex flex-col gap-1.5">
              {activeCase.documents.map((doc, i) => (
                <li
                  key={`${doc.filename}-${i}`}
                  className="flex items-start gap-2 rounded-lg bg-bg-subtle px-2.5 py-2 text-[11.5px]"
                >
                  <FileText size={13} className="mt-0.5 shrink-0 text-text-faint" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate font-medium text-text">
                      {doc.filename}
                    </span>
                    <span className="text-text-faint">
                      {doc.doc_type} · {doc.chunks_indexed} chunks
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          )}

          <UploadDropzone caseId={activeCaseId} onUploaded={onUploaded} />
        </div>
      )}
    </aside>
  );
}
