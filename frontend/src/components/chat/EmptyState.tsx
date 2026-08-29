import { FileSearch, GitBranch, Scale, ShieldCheck } from "lucide-react";

const EXAMPLES = [
  "What is the punishment for hacking under the IT Act?",
  "What constitutes cheating under Section 420?",
  "Explain the ingredients of dowry death under the law.",
  "What is the procedure for filing an FIR?",
];

const FEATURES = [
  { icon: Scale, label: "IRAC-reranked retrieval" },
  { icon: GitBranch, label: "Hybrid BM25 + dense search" },
  { icon: ShieldCheck, label: "Temporal validity filtering" },
  { icon: FileSearch, label: "Citation-grounded answers" },
];

export function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="mx-auto flex max-w-2xl flex-1 flex-col items-center justify-center px-4 py-16 text-center">
      <span className="mb-5 flex h-12 w-12 items-center justify-center rounded-2xl bg-brand text-brand-contrast shadow-md shadow-brand/20">
        <Scale size={22} strokeWidth={2.25} />
      </span>
      <h1 className="text-xl font-semibold tracking-tight text-text sm:text-2xl">
        Ask a question grounded in Indian statute law
      </h1>
      <p className="mt-2 max-w-md text-[13.5px] leading-relaxed text-text-muted">
        Every answer is cited to a specific, temporally-valid section. Start a case
        in the sidebar to fuse in your own FIRs, charge sheets, or evidence.
      </p>

      <div className="mt-7 grid w-full grid-cols-1 gap-2 sm:grid-cols-2">
        {EXAMPLES.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onPick(q)}
            className="rounded-xl border border-border bg-bg-elevated px-3.5 py-2.5 text-left text-[13px] text-text-muted transition-colors hover:border-brand/40 hover:bg-brand-soft hover:text-brand"
          >
            {q}
          </button>
        ))}
      </div>

      <div className="mt-9 flex flex-wrap items-center justify-center gap-x-5 gap-y-2">
        {FEATURES.map(({ icon: Icon, label }) => (
          <span
            key={label}
            className="flex items-center gap-1.5 text-[11.5px] text-text-faint"
          >
            <Icon size={13} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
