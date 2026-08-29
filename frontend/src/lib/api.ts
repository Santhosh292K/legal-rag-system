import type { CaseInfo, Health, QueryResponse, UploadResponse } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      "Can't reach the backend. Is it running at " + API_BASE + "?",
      0,
    );
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail =
        typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail ?? body);
    } catch {
      // body wasn't JSON — fall back to statusText
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export function getHealth(): Promise<Health> {
  return request<Health>("/api/health");
}

export function postQuery(query: string, caseId: string | null): Promise<QueryResponse> {
  return request<QueryResponse>("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, case_id: caseId }),
  });
}

export function uploadDocument(caseId: string, file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<UploadResponse>(`/api/cases/${encodeURIComponent(caseId)}/upload`, {
    method: "POST",
    body: form,
  });
}

export function listCases(): Promise<CaseInfo[]> {
  return request<CaseInfo[]>("/api/cases");
}
