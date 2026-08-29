"use client";

import { CloudUpload, FileWarning, Loader2 } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";
import { ApiError, uploadDocument } from "@/lib/api";
import { cn } from "@/lib/utils";

const ACCEPT = ".pdf,.jpg,.jpeg,.png,.tif,.tiff,.bmp";

export function UploadDropzone({
  caseId,
  onUploaded,
}: {
  caseId: string;
  onUploaded: () => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const result = await uploadDocument(caseId, file);
      onUploaded();
      toast.success(`${result.doc_type} indexed`, {
        description: `${result.chunks_indexed} chunk${result.chunks_indexed === 1 ? "" : "s"} added to ${caseId} · ${result.elapsed_s}s`,
      });
    } catch (e) {
      const message = e instanceof ApiError ? e.message : "Upload failed.";
      setError(message);
      toast.error("Upload failed", { description: message });
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div>
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void handleFiles(e.dataTransfer.files);
        }}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-lg border border-dashed px-3 py-4 text-center transition-colors",
          dragging
            ? "border-brand bg-brand-soft"
            : "border-border-strong hover:border-brand hover:bg-bg-subtle",
          busy && "pointer-events-none opacity-60",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          onChange={(e) => void handleFiles(e.target.files)}
          disabled={busy}
        />
        {busy ? (
          <Loader2 size={16} className="animate-spin text-brand" />
        ) : (
          <CloudUpload size={16} className="text-text-faint" />
        )}
        <span className="text-[11.5px] leading-snug text-text-muted">
          {busy ? "Ingesting document…" : "Drop a FIR, charge sheet, or evidence file"}
        </span>
        <span className="text-[10.5px] text-text-faint">PDF, JPG, PNG, TIFF</span>
      </label>
      {error && (
        <p className="mt-1.5 flex items-start gap-1.5 text-[11px] text-danger">
          <FileWarning size={12} className="mt-0.5 shrink-0" />
          {error}
        </p>
      )}
    </div>
  );
}
