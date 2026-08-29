"""
legal_rag_system.py
Top-level entry point — the single place that owns both tracks and
routes between them. main.py (Track A alone) still works unchanged and
is used internally here; this file doesn't replace it, it wraps it.

Usage:
    python legal_rag_system.py
        query> What is the punishment for hacking under IT Act?
        query> /case case-1
        query> /upload path/to/fir.pdf
        query> What sections apply to the accused in this case?
        query> quit
"""
import time

from qdrant_client import QdrantClient

from config import QDRANT_PATH
from main import LegalRAGPipeline
from data.case_indexer import CaseIndexer
from pipeline.document_pipeline import DocumentIngestionPipeline
from pipeline.query_router import QueryRouter
from pipeline.fusion import CaseStatuteFusion
from pipeline.alea import ALEA, evidence_coverage_report
from pipeline.reasoning_graph import build_reasoning_graph


_UNANSWERED_PHRASES = (
    "do not specify", "does not specify", "not specify the exact",
    "not mentioned in", "no matching case document",
    "not explicitly outline", "not explicitly state",
    "not contain", "cannot be found in the document",
)


def _looks_unanswered(answer_text: str) -> bool:
    """Cheap textual check for a document-only answer that dodged the
    question instead of answering it — cheaper than a second LLM call,
    and good enough to trigger the hybrid retry above."""
    lowered = answer_text.lower()
    return any(phrase in lowered for phrase in _UNANSWERED_PHRASES)


class LegalRAGSystem:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

        # One shared client for the whole process. legal_sections and
        # case_documents both live in the same ./qdrant_db folder — Qdrant's
        # embedded/local mode file-locks that folder to a single client, so
        # every component that needs Qdrant here must share this one
        # instance rather than opening its own.
        self._log("Opening shared Qdrant client...")
        self.qdrant_client = QdrantClient(path=QDRANT_PATH)

        self._log("Loading statute pipeline (Track A)...")
        self.statute_pipeline = LegalRAGPipeline(verbose=verbose, qdrant_client=self.qdrant_client)

        self._log("Loading case indexer + document ingestion (Track B)...")
        self.case_indexer = CaseIndexer(client=self.qdrant_client)
        self.ingestion     = DocumentIngestionPipeline()

        # QueryRouter and ALEA both reuse CaseIndexer's already-loaded
        # embedding model rather than each loading a separate copy — same
        # bge-large instance, just wrapped to match each one's expected
        # embed_fn(list[str]) -> vectors signature.
        shared_embed_fn = lambda texts: self.case_indexer.embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False)

        self.router = QueryRouter(embed_fn=shared_embed_fn)
        self.alea = ALEA(embed_fn=shared_embed_fn)

        self.fusion = CaseStatuteFusion(self.statute_pipeline, self.case_indexer, alea=self.alea)
        self._last_fused = None
        self._last_graph = None
        self._log("System ready.")

    def _log(self, msg):
        if self.verbose:
            print(f"[LegalRAGSystem] {msg}")

    def upload_document(self, file_path: str, case_id: str) -> dict:
        """Runs the full Phase 1 chain and indexes the result. Returns a
        small summary rather than the full IngestedDocument, since the
        REPL just needs to confirm what happened."""
        t0 = time.time()
        ingested = self.ingestion.ingest(file_path, case_id=case_id)
        n_indexed = self.case_indexer.index_chunks(ingested.chunks)

        return {
            "doc_type":   ingested.classification.doc_type,
            "confidence": ingested.classification.confidence,
            "chunks_indexed": n_indexed,
            "used_ocr":   ingested.used_ocr,
            "warnings":   ingested.warnings,
            "elapsed_s":  round(time.time() - t0, 2),
        }

    def ask(self, query: str, case_id: str | None = None):
        """Single entry point for a question, with or without an active
        case. Returns a LegalAnswer either way, so callers don't need to
        branch on route type themselves."""
        decision = self.router.route(query, case_id)
        self._log(f"Route: {decision.route} (conf={decision.confidence:.2f}) — {decision.reasoning}")

        self._last_graph = None   # reset; only routes that touch the statute pipeline build one

        if decision.route == "general" or not case_id:
            answer = self.statute_pipeline.query(query)
            self._last_graph = build_reasoning_graph(query, statute_answer=answer)
            return answer

        if decision.route == "document":
            fused = self.fusion.answer_document_only(query, case_id=case_id)
            # No statute pipeline involved for a document-only route, so
            # there's no statute/legal-element chain to graph — the answer
            # is grounded purely in the uploaded document's own text.
            answer = self.statute_pipeline.generator.generate_document_only(
                query=query, case_chunks=fused.case_chunks,
            )
            # Safety net: a "document" route by design never consults the
            # statute corpus. If the case chunks didn't actually contain
            # what was asked (misroute of an abstract legal-rule question,
            # e.g. "what's the penalty for X"), don't hand back a dead end —
            # retry as hybrid so the statute pipeline gets a chance too.
            if not fused.case_chunks or _looks_unanswered(answer.answer):
                self._log("Document route came back empty/unfound — retrying as hybrid.")
                fused = self.fusion.answer(query, case_id=case_id)
                answer = self.statute_pipeline.generator.generate_fused(
                    query=query, case_chunks=fused.case_chunks,
                    statute_answer=fused.statute_answer,
                )
                self._last_fused = fused
                self._last_graph = build_reasoning_graph(
                    query, statute_answer=fused.statute_answer, alea_scores=fused.alea_scores,
                )
            return answer

        # hybrid
        fused = self.fusion.answer(query, case_id=case_id)
        answer = self.statute_pipeline.generator.generate_fused(
            query=query,
            case_chunks=fused.case_chunks,
            statute_answer=fused.statute_answer,
        )
        self._last_fused = fused   # so the REPL can print the coverage table alongside the answer
        self._last_graph = build_reasoning_graph(
            query, statute_answer=fused.statute_answer, alea_scores=fused.alea_scores,
        )
        return answer




def _format_upload_result(result: dict) -> str:
    lines = [
        f"doc_type    : {result['doc_type']} ({result['confidence']:.2f})",
        f"chunks      : {result['chunks_indexed']} indexed",
        f"used_ocr    : {result['used_ocr']}",
        f"elapsed     : {result['elapsed_s']}s",
    ]
    if result["warnings"]:
        lines.append(f"warnings    : {result['warnings']}")
    return "\n".join(lines)


if __name__ == "__main__":
    system = LegalRAGSystem(verbose=True)
    active_case_id = None

    print("\n[LegalRAGSystem] Ready.")
    print("  Plain text            -> ask a question")
    print("  /case <case_id>        -> switch active case (blank = none)")
    print("  /upload <file_path>     -> ingest a document into the active case")
    print("  /graph                  -> print the reasoning graph for the last answer")
    print("  /graph save <path.html> -> save the last reasoning graph as an HTML file")
    print("  quit / exit              -> stop\n")

    while True:
        try:
            line = input(f"query{f'[{active_case_id}]' if active_case_id else ''}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[LegalRAGSystem] Exiting.")
            break

        if not line:
            continue
        if line.lower() in {"quit", "exit", "q"}:
            print("[LegalRAGSystem] Exiting.")
            break

        if line.startswith("/case"):
            parts = line.split(maxsplit=1)
            active_case_id = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
            print(f"[LegalRAGSystem] Active case: {active_case_id or '(none)'}")
            continue

        if line.startswith("/upload"):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /upload path/to/file.pdf")
                continue
            if not active_case_id:
                print("No active case — set one first with /case <case_id>")
                continue
            try:
                result = system.upload_document(parts[1].strip(), case_id=active_case_id)
                print(_format_upload_result(result))
            except Exception as e:
                print(f"[LegalRAGSystem] Upload failed: {e}")
            continue

        if line.startswith("/graph"):
            if system._last_graph is None:
                print("[LegalRAGSystem] No reasoning graph available — ask a question first "
                      "(document-only routes don't produce one).")
                continue
            parts = line.split(maxsplit=2)
            if len(parts) >= 2 and parts[1] == "save":
                out_path = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "reasoning_graph.html"
                system._last_graph.save_html(out_path)
                print(f"[LegalRAGSystem] Saved reasoning graph -> {out_path}")
            else:
                print(system._last_graph.to_ascii())
                print("\n(mermaid source — paste into https://mermaid.live or use `/graph save <path.html>`)")
                print(system._last_graph.to_mermaid())
            continue

        try:
            answer = system.ask(line, case_id=active_case_id)
            print(system.statute_pipeline.format_output(answer))

            fused = getattr(system, "_last_fused", None)
            if fused is not None and fused.alea_scores:
                print("\nEVIDENCE COVERAGE ANALYSIS (Phase 3/4 — ALEA):")
                print(evidence_coverage_report(fused.alea_scores))
                system._last_fused = None   # don't reprint for the next, unrelated query
        except Exception as e:
            print(f"\n[LegalRAGSystem] Error while answering that query: {e}\n")