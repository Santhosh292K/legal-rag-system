"use client";

import { AlertTriangle, RotateCcw } from "lucide-react";
import { useEffect } from "react";

export default function ErrorPage({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-bg px-6 text-center">
      <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-danger-soft text-danger">
        <AlertTriangle size={22} strokeWidth={2.25} />
      </span>
      <div className="space-y-1.5">
        <h1 className="text-lg font-semibold text-text">Something went wrong</h1>
        <p className="max-w-sm text-[13.5px] leading-relaxed text-text-muted">
          The application hit an unexpected error. This has been logged —
          try again, or reload the page if it keeps happening.
        </p>
      </div>
      <button
        type="button"
        onClick={() => retry()}
        className="mt-2 inline-flex items-center gap-2 rounded-xl bg-brand px-4 py-2 text-sm font-medium text-brand-contrast transition-opacity hover:opacity-90"
      >
        <RotateCcw size={14} />
        Try again
      </button>
    </div>
  );
}
