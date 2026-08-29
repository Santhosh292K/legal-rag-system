import re
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool

from app import state
from app.config import MAX_UPLOAD_MB
from app.schemas import UploadResponse, CaseInfo

router = APIRouter(prefix="/api", tags=["cases"])

# Mirrors pipeline/ocr_extractor.py's DocumentExtractor support (PDF + common
# scanned-image formats — it OCRs images and image-only PDF pages via Tesseract).
ALLOWED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def _run_upload(file_path: str, case_id: str) -> dict:
    system = state.get_system()
    with state.get_lock():
        return system.upload_document(file_path, case_id=case_id)


@router.post("/cases/{case_id}/upload", response_model=UploadResponse)
async def upload_document(case_id: str, file: UploadFile = File(...)):
    if not _SAFE_CASE_ID.match(case_id):
        raise HTTPException(status_code=400, detail="Invalid case_id.")
    if state.get_status()["status"] != "ready":
        raise HTTPException(status_code=503, detail=state.get_status())

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    body = await file.read()
    if len(body) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit.")

    # Absolute temp path — deliberate, since the RAG system's CWD is pinned
    # to rag/ (see app/config.py:bootstrap_rag), and a relative path here
    # would resolve against that instead of the actual upload.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(body)
        tmp_path = tmp.name

    t0 = time.time()
    try:
        result = await run_in_threadpool(_run_upload, tmp_path, case_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    summary = {
        "filename": file.filename,
        "doc_type": result["doc_type"],
        "confidence": result["confidence"],
        "chunks_indexed": result["chunks_indexed"],
    }
    state.record_upload(case_id, summary)

    return UploadResponse(
        filename=file.filename or "",
        case_id=case_id,
        doc_type=result["doc_type"],
        confidence=result["confidence"],
        chunks_indexed=result["chunks_indexed"],
        used_ocr=result["used_ocr"],
        warnings=list(result.get("warnings") or []),
        elapsed_s=result.get("elapsed_s", round(time.time() - t0, 2)),
    )


@router.get("/cases", response_model=list[CaseInfo])
def get_cases():
    return state.list_cases()


@router.get("/cases/{case_id}", response_model=CaseInfo)
def get_case(case_id: str):
    docs = state.get_case(case_id)
    if docs is None:
        raise HTTPException(status_code=404, detail="Unknown case_id.")
    return CaseInfo(case_id=case_id, documents=docs)
