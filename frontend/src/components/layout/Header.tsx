import { Scale, PanelLeft, SquarePen } from "lucide-react";
import { HealthPill } from "./HealthPill";
import { ThemeToggle } from "@/components/theme/ThemeToggle";

export function Header({
  onToggleSidebar,
  onNewChat,
  hasMessages,
}: {
  onToggleSidebar: () => void;
  onNewChat: () => void;
  hasMessages: boolean;
}) {
  return (
    <header className="sticky top-0 z-30 shrink-0 bg-bg/85 backdrop-blur-md">
      <div className="flex h-16 items-center gap-3 px-4">
        <button
          type="button"
          onClick={onToggleSidebar}
          className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-text-muted transition-colors hover:bg-bg-subtle hover:text-text lg:hidden"
          aria-label="Toggle sidebar"
        >
          <PanelLeft size={17} />
        </button>

        <div className="flex min-w-0 items-center gap-2.5">
          {/* Seal mark — a thin inset brass ring around the ink badge is
              the one "institutional seal" cue, kept small and restrained
              rather than a literal emblem illustration. */}
          <span
            className="brand-glow flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-brand text-brand-contrast shadow-[var(--shadow-brand)] ring-1 ring-inset ring-accent/40"
          >
            <Scale size={16} strokeWidth={2} />
          </span>
          <div className="min-w-0 leading-tight">
            <p className="truncate font-serif text-[15px] font-semibold tracking-wide text-text">
              NYAYA
            </p>
            <p className="truncate text-[10.5px] uppercase tracking-wider text-text-faint">
              Legal Research Assistant
            </p>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-2">
          {hasMessages && (
            <button
              type="button"
              onClick={onNewChat}
              className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-border px-2.5 text-xs font-medium text-text-muted transition-colors hover:border-border-strong hover:bg-bg-subtle hover:text-text"
            >
              <SquarePen size={13} />
              <span className="hidden sm:inline">New chat</span>
            </button>
          )}
          <div className="hidden sm:block">
            <HealthPill />
          </div>
          <ThemeToggle />
        </div>
      </div>
      {/* Letterhead double rule: a hairline border plus a faint brass line
          beneath it — the recurring "printed document" motif, in place of
          a single flat border like the rest of the app's panels use. */}
      <div className="border-b border-border">
        <div className="letterhead-rule h-px" />
      </div>
    </header>
  );
}
