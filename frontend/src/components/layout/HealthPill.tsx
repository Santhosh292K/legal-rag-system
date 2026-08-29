"use client";

import { useEffect, useRef, useState } from "react";
import { getHealth } from "@/lib/api";
import type { Health } from "@/lib/types";
import { cn } from "@/lib/utils";

export function HealthPill() {
  const [health, setHealth] = useState<Health | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const h = await getHealth();
        if (cancelled) return;
        setHealth(h);
        setUnreachable(false);
        // Back off once ready — no need to hammer the endpoint.
        timer.current = setTimeout(poll, h.status === "ready" ? 20000 : 2000);
      } catch {
        if (cancelled) return;
        setUnreachable(true);
        timer.current = setTimeout(poll, 4000);
      }
    }
    poll();

    return () => {
      cancelled = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const status = unreachable ? "error" : health?.status ?? "loading";
  const label = unreachable
    ? "Backend unreachable"
    : status === "ready"
      ? "Backend ready"
      : status === "error"
        ? "Pipeline error"
        : "Loading models…";

  return (
    <div
      title={unreachable ? "Could not reach the API server" : health?.detail}
      className="inline-flex items-center gap-2 rounded-full border border-border bg-bg-elevated px-3 py-1 text-xs text-text-muted"
    >
      <span className="relative flex h-2 w-2">
        <span
          className={cn(
            "absolute inline-flex h-full w-full rounded-full opacity-60",
            status === "ready" && "bg-success",
            status === "loading" && "bg-warning animate-pulse-dot",
            status === "error" && "bg-danger",
          )}
        />
        <span
          className={cn(
            "relative inline-flex h-2 w-2 rounded-full",
            status === "ready" && "bg-success",
            status === "loading" && "bg-warning",
            status === "error" && "bg-danger",
          )}
        />
      </span>
      {label}
    </div>
  );
}
