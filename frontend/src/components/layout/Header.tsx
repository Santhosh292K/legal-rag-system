import { Scale, PanelLeft } from "lucide-react";
import { HealthPill } from "./HealthPill";
import { ThemeToggle } from "@/components/theme/ThemeToggle";

export function Header({ onToggleSidebar }: { onToggleSidebar: () => void }) {
  return (
    <header className="sticky top-0 z-30 flex h-14 shrink-0 items-center gap-3 border-b border-border bg-bg/80 px-4 backdrop-blur-md">
      <button
        type="button"
        onClick={onToggleSidebar}
        className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-text-muted transition-colors hover:bg-bg-subtle hover:text-text lg:hidden"
        aria-label="Toggle sidebar"
      >
        <PanelLeft size={17} />
      </button>

      <div className="flex items-center gap-2.5">
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand text-brand-contrast shadow-sm">
          <Scale size={16} strokeWidth={2.25} />
        </span>
        <div className="leading-tight">
          <p className="text-sm font-semibold tracking-tight text-text">Nyaya</p>
          <p className="text-[11px] text-text-faint">Legal Research Assistant</p>
        </div>
      </div>

      <div className="ml-auto flex items-center gap-3">
        <HealthPill />
        <ThemeToggle />
      </div>
    </header>
  );
}
