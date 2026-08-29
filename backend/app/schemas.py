"""backend/app/schemas.py — API request/response models."""
from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    case_id: Optional[str] = Field(
        default=None,
        description="Active case to consult alongside (or instead of) the "
                    "statute corpus. Omit for a general legal question.",
    )


class CitationOut(BaseModel):
    section_id: str
    act_name: str
    category: str
    content: str
    validity: str
    warning: str = ""


class QueryResponse(BaseModel):
    query: str
    answer: str
    intent: str
    confidence: str
    citations: list[CitationOut] = []
    warnings: list[str] = []
    irac_summary: dict[str, float] = {}
    retrieved_section_ids: list[str] = []
    case_id: Optional[str] = None
    elapsed_ms: int = 0


class UploadResponse(BaseModel):
    filename: str
    case_id: str
    document_id: str = ""
    doc_type: str
    confidence: float
    chunks_indexed: int
    used_ocr: bool
    warnings: list[str] = []
    elapsed_s: float


class CaseDocument(BaseModel):
    filename: str
    doc_type: str
    confidence: float
    chunks_indexed: int
    uploaded_at: str


class CaseInfo(BaseModel):
    case_id: str
    documents: list[CaseDocument] = []


class HealthResponse(BaseModel):
    status: str  # "loading" | "ready" | "error"
    detail: str = ""
