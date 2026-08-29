export interface Citation {
  section_id: string;
  act_name: string;
  category: string;
  content: string;
  validity: string; // "active" | "amended" | "repealed" | ...
  warning: string;
}

export interface QueryResponse {
  query: string;
  answer: string;
  intent: string;
  confidence: string; // "high" | "medium" | "low"
  citations: Citation[];
  warnings: string[];
  irac_summary: Record<string, number>;
  retrieved_section_ids: string[];
  case_id: string | null;
  elapsed_ms: number;
}

export interface UploadResponse {
  filename: string;
  case_id: string;
  document_id: string;
  doc_type: string;
  confidence: number;
  chunks_indexed: number;
  used_ocr: boolean;
  warnings: string[];
  elapsed_s: number;
}

export interface CaseDocument {
  filename: string;
  doc_type: string;
  confidence: number;
  chunks_indexed: number;
  uploaded_at: string;
}

export interface CaseInfo {
  case_id: string;
  documents: CaseDocument[];
}

export type HealthStatus = "loading" | "ready" | "error";

export interface Health {
  status: HealthStatus;
  detail: string;
}

export type ChatMessage =
  | { id: string; role: "user"; text: string; caseId: string | null }
  | { id: string; role: "assistant"; answer: QueryResponse }
  | { id: string; role: "error"; text: string }
  | { id: string; role: "pending"; text: string };
