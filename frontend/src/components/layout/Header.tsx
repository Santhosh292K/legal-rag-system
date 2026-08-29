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
    <header className="sticky top-0 z-30 flex h-14 shrink-0 items-center gap-3 border-b border-border bg-bg/80 px-4 backdrop-blur-md">
      <button
        type="button"
        onClick={onToggleSidebar}
        className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-text-muted transition-colors hover:bg-bg-subtle hover:text-text lg:hidden"
        aria-label="Toggle sidebar"
      >
        <PanelLeft size={17} />
      </button>

      <div className="flex min-w-0 items-center gap-2.5">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-brand text-brand-contrast shadow-[var(--shadow-brand)]">
          <Scale size={16} strokeWidth={2.25} />
        </span>
        <div className="min-w-0 leading-tight">
          <p className="truncate text-sm font-semibold tracking-tight text-text">Nyaya</p>
          <p className="truncate text-[11px] text-text-faint">Legal Research Assistant</p>
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
    </header>
  );
}
