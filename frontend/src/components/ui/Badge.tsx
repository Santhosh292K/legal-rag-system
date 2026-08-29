import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

type Tone = "neutral" | "brand" | "success" | "warning" | "danger" | "info";

const TONE_CLASSES: Record<Tone, string> = {
  neutral: "bg-bg-subtle text-text-muted border-border",
  brand: "bg-brand-soft text-brand border-transparent",
  success: "bg-success-soft text-success border-transparent",
  warning: "bg-warning-soft text-warning border-transparent",
  danger: "bg-danger-soft text-danger border-transparent",
  info: "bg-info-soft text-info border-transparent",
};

export function Badge({
  children,
  tone = "neutral",
  className,
  dot = false,
}: {
  children: ReactNode;
  tone?: Tone;
  className?: string;
  dot?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-md border px-2 py-0.5 text-[11px] font-medium leading-5 tracking-wide uppercase",
        TONE_CLASSES[tone],
        className,
      )}
    >
      {dot && (
        <span
          className={cn(
            "h-1.5 w-1.5 shrink-0 rounded-full",
            tone === "success" && "bg-success",
            tone === "warning" && "bg-warning",
            tone === "danger" && "bg-danger",
            tone === "info" && "bg-info",
            tone === "brand" && "bg-brand",
            tone === "neutral" && "bg-text-faint",
          )}
        />
      )}
      {children}
    </span>
  );
}

export function validityTone(validity: string): Tone {
  const v = validity.toLowerCase();
  if (v === "active") return "success";
  if (v === "amended") return "warning";
  if (v === "repealed" || v === "superseded") return "danger";
  return "neutral";
}

export function confidenceTone(confidence: string): Tone {
  const c = confidence.toLowerCase();
  if (c === "high") return "success";
  if (c === "medium") return "warning";
  if (c === "low") return "danger";
  return "neutral";
}
