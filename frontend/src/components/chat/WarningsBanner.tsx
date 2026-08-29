import { AlertTriangle } from "lucide-react";

export function WarningsBanner({ warnings }: { warnings: string[] }) {
  const unique = Array.from(new Set(warnings)).filter(Boolean);
  if (unique.length === 0) return null;

  return (
    <div className="flex flex-col gap-1.5 rounded-xl border border-warning/20 bg-warning-soft px-3.5 py-3">
      {unique.map((w, i) => (
        <p key={i} className="flex items-start gap-2 text-[12.5px] leading-relaxed text-warning">
          <AlertTriangle size={13} className="mt-0.5 shrink-0" />
          <span>{w}</span>
        </p>
      ))}
    </div>
  );
}
