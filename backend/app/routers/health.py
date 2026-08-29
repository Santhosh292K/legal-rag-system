from fastapi import APIRouter

from app.schemas import HealthResponse
from app import state

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(**state.get_status())
