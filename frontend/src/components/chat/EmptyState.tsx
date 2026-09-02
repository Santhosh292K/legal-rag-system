"use client";

import { BadgeCheck, Library, ScrollText, Scale } from "lucide-react";
import { motion } from "framer-motion";

const EXAMPLES = [
  "What is the punishment for hacking under the IT Act?",
  "What constitutes cheating under Section 420?",
  "Explain the ingredients of dowry death under the law.",
  "What is the procedure for filing an FIR?",
];

// Plain-language, not implementation jargon — this is read by the public
// and by lawyers, not by developers. Each one names a benefit someone
// actually cares about, not the technique behind it (no "BM25", "IRAC",
// "hybrid retrieval", etc. — see MessageRow.tsx's THINKING_STEPS for the
// same rule applied to the in-progress state).
const FEATURES = [
  { icon: Library, label: "Searches the full text of Indian law" },
  { icon: BadgeCheck, label: "Flags outdated or repealed sections" },
  { icon: ScrollText, label: "Every answer names its source section" },
  { icon: Scale, label: "No guessing — only what the law actually says" },
];

export function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="mx-auto flex max-w-2xl flex-1 flex-col items-center justify-center px-4 py-16 text-center">
      <motion.span
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.35, ease: "easeOut" }}
        className="brand-glow mb-5 flex h-12 w-12 items-center justify-center rounded-2xl bg-brand text-brand-contrast shadow-[var(--shadow-brand)] ring-1 ring-inset ring-accent/40"
      >
        <Scale size={20} strokeWidth={2} />
      </motion.span>
      <motion.h1
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.05, ease: "easeOut" }}
        className="font-serif text-2xl font-semibold tracking-tight text-text sm:text-[28px]"
      >
        Ask a question grounded in Indian statute law
      </motion.h1>
      <motion.p
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.1, ease: "easeOut" }}
        className="mt-2.5 max-w-md text-[13.5px] leading-relaxed text-text-muted"
      >
        Every answer names the exact section of law it comes from, and warns
        you if that section has since been amended or repealed. Open a case
        in the sidebar to also ask questions about your own FIR, charge
        sheet, or other evidence.
      </motion.p>

      <div className="mt-7 grid w-full grid-cols-1 gap-2 sm:grid-cols-2">
        {EXAMPLES.map((q, i) => (
          <motion.button
            key={q}
            type="button"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.15 + i * 0.04, ease: "easeOut" }}
            whileTap={{ scale: 0.99 }}
            onClick={() => onPick(q)}
            className="rounded-lg border border-border bg-bg-elevated px-3.5 py-2.5 text-left text-[13px] text-text-muted shadow-[var(--shadow-sm)] transition-colors hover:border-accent/50 hover:bg-accent-soft hover:text-text"
          >
            {q}
          </motion.button>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.35, delay: 0.35 }}
        className="mt-9 flex flex-wrap items-center justify-center gap-x-5 gap-y-2"
      >
        {FEATURES.map(({ icon: Icon, label }) => (
          <span
            key={label}
            className="flex items-center gap-1.5 text-[11.5px] text-text-faint"
          >
            <Icon size={13} />
            {label}
          </span>
        ))}
      </motion.div>
    </div>
  );
}
