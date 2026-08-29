const LABELS: Record<string, string> = {
  issue: "Issue",
  rule: "Rule",
  application: "Application",
  conclusion: "Conclusion",
};

export function IracBars({ summary }: { summary: Record<string, number> }) {
  const entries = Object.entries(summary);
  if (entries.length === 0) return null;

  return (
    <div className="grid grid-cols-2 gap-x-5 gap-y-2 sm:grid-cols-4">
      {entries.map(([key, value]) => {
        const pct = Math.max(0, Math.min(1, value)) * 100;
        return (
          <div key={key}>
            <div className="mb-1 flex items-baseline justify-between text-[10.5px]">
              <span className="font-medium uppercase tracking-wide text-text-faint">
                {LABELS[key] ?? key}
              </span>
              <span className="font-mono text-text-muted">{value.toFixed(2)}</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-bg-subtle">
              <div
                className="h-full rounded-full bg-brand transition-[width] duration-500"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
