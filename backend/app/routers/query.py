import time

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app import state
from app.schemas import QueryRequest, QueryResponse, CitationOut

router = APIRouter(prefix="/api", tags=["query"])


def _run_query(query: str, case_id: str | None):
    system = state.get_system()
    with state.get_lock():
        return system.ask(query, case_id=case_id)


@router.post("/query", response_model=QueryResponse)
async def ask_query(req: QueryRequest):
    if state.get_status()["status"] != "ready":
        raise HTTPException(status_code=503, detail=state.get_status())

    t0 = time.time()
    try:
        answer = await run_in_threadpool(_run_query, req.query, req.case_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        query=answer.query,
        answer=answer.answer,
        intent=answer.intent,
        confidence=answer.confidence,
        citations=[
            CitationOut(
                section_id=c.section_id, act_name=c.act_name, category=c.category,
                content=c.content, validity=c.validity, warning=c.warning or "",
            )
            for c in answer.citations
        ],
        warnings=list(answer.warnings or []),
        irac_summary=dict(answer.irac_summary or {}),
        retrieved_section_ids=list(answer.retrieved_section_ids or []),
        case_id=req.case_id,
        elapsed_ms=int((time.time() - t0) * 1000),
    )
