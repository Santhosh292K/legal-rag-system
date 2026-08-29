import { Scale } from "lucide-react";
import type { QueryResponse } from "@/lib/types";
import { Badge, confidenceTone } from "@/components/ui/Badge";
import { AnswerText } from "./AnswerText";
import { CitationList } from "./CitationList";
import { IracBars } from "./IracBars";
import { WarningsBanner } from "./WarningsBanner";
import { formatMs } from "@/lib/utils";

export function AnswerCard({ answer, id }: { answer: QueryResponse; id: string }) {
  const hasIrac = Object.keys(answer.irac_summary ?? {}).length > 0;

  return (
    <div className="flex gap-3">
      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-brand text-brand-contrast">
        <Scale size={13} strokeWidth={2.25} />
      </span>

      <div className="min-w-0 flex-1 space-y-4 rounded-2xl rounded-tl-sm border border-border bg-bg-elevated p-4 shadow-sm sm:p-5">
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge tone={confidenceTone(answer.confidence)} dot>
            {answer.confidence} confidence
          </Badge>
          {answer.intent && <Badge tone="brand">{answer.intent}</Badge>}
          {answer.case_id && <Badge tone="info">case: {answer.case_id}</Badge>}
          <span className="ml-auto text-[11px] text-text-faint">
            {formatMs(answer.elapsed_ms)}
          </span>
        </div>

        <AnswerText text={answer.answer} citations={answer.citations} messageId={id} />

        <WarningsBanner warnings={answer.warnings} />

        {hasIrac && (
          <div className="border-t border-border pt-3.5">
            <IracBars summary={answer.irac_summary} />
          </div>
        )}

        {answer.citations.length > 0 && (
          <div className="border-t border-border pt-3.5">
            <CitationList citations={answer.citations} messageId={id} />
          </div>
        )}
      </div>
    </div>
  );
}
