"""
pipeline — the Track A (statute) and Track B (case document) components.

Deliberately does NOT eagerly import its submodules.

It used to import all eleven of them at package-import time, which meant
that touching anything under `pipeline.` — including leaf utilities with
no model dependencies at all, like pipeline.section_store — pulled in
ollama, sentence-transformers, torch, qdrant-client and networkx. That
made the light modules untestable without the full stack, forced a slow
import on every CLI invocation, and hid genuinely optional dependencies
behind an unrelated ImportError.

Import what you need directly:

    from pipeline.hybrid_retriever import HybridRetriever

The names below are still resolvable as attributes (``pipeline.IRACReranker``)
via the lazy __getattr__ hook, so existing call sites keep working, but
nothing is loaded until it is actually referenced.

pipeline.scenario_rewriter is gone: pipeline.universal_translator
superseded it (always-on translation instead of conditional rewriting) and
nothing in the pipeline had called it since. It remains in git history.
"""
import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "IntentClassifier":    "pipeline.intent_classifier",
    "QueryIntent":         "pipeline.intent_classifier",
    "QueryExpander":       "pipeline.query_expander",
    "HybridRetriever":     "pipeline.hybrid_retriever",
    "RetrievedChunk":      "pipeline.hybrid_retriever",
    "TemporalFilter":      "pipeline.temporal_filter",
    "ValidatedChunk":      "pipeline.temporal_filter",
    "ChunkStructurer":     "pipeline.chunk_structurer",
    "StructuredChunk":     "pipeline.chunk_structurer",
    "IRACReranker":        "pipeline.irac_reranker",
    "RankedChunk":         "pipeline.irac_reranker",
    "AnswerGenerator":     "pipeline.answer_generator",
    "LegalAnswer":         "pipeline.answer_generator",
    "DomainRouter":        "pipeline.domain_router",
    "RoutingResult":       "pipeline.domain_router",
    "UniversalTranslator": "pipeline.universal_translator",
    "TranslationResult":   "pipeline.universal_translator",
    "SectionPinner":       "pipeline.section_pinner",
    "PinResult":           "pipeline.section_pinner",
    "PIN_EXPLANATION":     "pipeline.section_pinner",
    "SectionStore":        "pipeline.section_store",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
