"use client";

import { useTheme } from "next-themes";
import { Toaster as Sonner } from "sonner";

/** Theme-matched wrapper around sonner — styled with this app's own design
 * tokens (via CSS vars) rather than sonner's defaults, so a toast reads as
 * part of the product, not a bolted-on library. */
export function Toaster() {
  const { resolvedTheme } = useTheme();

  return (
    <Sonner
      theme={resolvedTheme === "dark" ? "dark" : "light"}
      position="bottom-right"
      gap={10}
      toastOptions={{
        style: {
          background: "var(--bg-overlay)",
          border: "1px solid var(--border)",
          color: "var(--text)",
          borderRadius: "0.75rem",
          boxShadow: "var(--shadow-lg)",
          fontSize: "13px",
        },
        classNames: {
          title: "font-medium",
          description: "text-[var(--text-muted)]",
        },
      }}
    />
  );
}
