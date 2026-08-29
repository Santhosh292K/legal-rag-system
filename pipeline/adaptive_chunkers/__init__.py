"""
pipeline/adaptive_chunkers/__init__.py
Phase 2 — registry mapping doc_type -> chunker instance.
"""
from .base_chunker import Chunk, BaseChunker
from .fir_chunker import FIRChunker
from .chargesheet_chunker import ChargesheetChunker
from .medical_chunker import MedicalReportChunker
from .generic_chunker import GenericChunker

_REGISTRY = {
    "FIR":            FIRChunker(),
    "Charge Sheet":    ChargesheetChunker(),
    "Medical Report":   MedicalReportChunker(),
}


def get_chunker(doc_type: str) -> BaseChunker:
    """Every doc_type not in _REGISTRY gets a GenericChunker tagged with
    its own doc_type, so the chunk metadata still reflects the real type."""
    return _REGISTRY.get(doc_type, GenericChunker(doc_type=doc_type))


__all__ = ["Chunk", "BaseChunker", "get_chunker"]
