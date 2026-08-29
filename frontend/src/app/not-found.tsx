import { Scale } from "lucide-react";
import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-bg px-6 text-center">
      <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-brand text-brand-contrast">
        <Scale size={22} strokeWidth={2.25} />
      </span>
      <div className="space-y-1.5">
        <h1 className="text-lg font-semibold text-text">Page not found</h1>
        <p className="max-w-sm text-[13.5px] leading-relaxed text-text-muted">
          There&rsquo;s nothing at this address. Head back to ask a legal question.
        </p>
      </div>
      <Link
        href="/"
        className="mt-2 inline-flex items-center gap-2 rounded-xl bg-brand px-4 py-2 text-sm font-medium text-brand-contrast transition-opacity hover:opacity-90"
      >
        Back to Nyaya
      </Link>
    </div>
  );
}
