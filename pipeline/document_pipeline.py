"""
pipeline/document_pipeline.py
Phase 1 orchestrator — the single entry point Track B exposes to the
rest of the system, mirroring how main.py's LegalRAGPipeline is the
entry point for Track A.

    Uploaded PDF -> OCR -> Document Classifier -> Entity Extraction
                 -> Timeline Extraction -> Adaptive Chunking

Case Vector Store indexing (data/case_indexer.py) is called separately
so this class stays testable without a live Qdrant instance.
"""
import hashlib
from dataclasses import dataclass, field

from pipeline.ocr_extractor import DocumentExtractor
from pipeline.document_classifier import DocumentClassifier, DocumentClassification
from pipeline.entity_timeline_extractor import extract_entities, extract_timeline, ExtractedEntities, TimelineEvent
from pipeline.adaptive_chunkers import get_chunker, Chunk


@dataclass
class IngestedDocument:
    document_id:    str
    case_id:        str
    file_path:      str
    classification: DocumentClassification
    entities:       ExtractedEntities
    timeline:       list[TimelineEvent]
    chunks:         list[Chunk]
    used_ocr:       bool
    warnings:       list[str] = field(default_factory=list)


class DocumentIngestionPipeline:
    def __init__(self):
        self.extractor  = DocumentExtractor()
        self.classifier = DocumentClassifier()

    def ingest(self, file_path: str, case_id: str, document_id: str | None = None) -> IngestedDocument:
        # Deterministic by default: hash the file's own bytes, so uploading
        # the exact same file twice (same run or a later session) produces
        # the SAME document_id instead of a random one each time — that's
        # what lets CaseIndexer detect and replace a duplicate instead of
        # silently accumulating copies of the same chunks forever.
        if document_id is None:
            with open(file_path, "rb") as f:
                document_id = hashlib.sha256(f.read()).hexdigest()[:16]
        warnings: list[str] = []

        # Stage 1 — OCR / text extraction
        extracted = self.extractor.extract(file_path)
        if extracted.error:
            warnings.append(extracted.error)
        text = extracted.full_text

        if not text.strip():
            # Nothing to work with — still return a valid (empty) result
            # rather than raising, so a bad upload doesn't take down a batch job.
            warnings.append("No text could be extracted from this document.")
            classification = DocumentClassification(doc_type="Other", confidence=0.0)
            return IngestedDocument(
                document_id=document_id, case_id=case_id, file_path=file_path,
                classification=classification,
                entities=ExtractedEntities(), timeline=[], chunks=[],
                used_ocr=extracted.used_ocr, warnings=warnings,
            )

        # Stage 2 — Document Classifier
        classification = self.classifier.classify(text)

        # Stage 3 — Entity Extraction
        entities = extract_entities(text)

        # Stage 4 — Timeline Extraction
        timeline = extract_timeline(text)

        # Stage 5 — Adaptive Chunking
        chunker = get_chunker(classification.doc_type)
        chunks = chunker.chunk(
            text, document_id=document_id, case_id=case_id,
            entities=entities.__dict__,
        )

        return IngestedDocument(
            document_id=document_id,
            case_id=case_id,
            file_path=file_path,
            classification=classification,
            entities=entities,
            timeline=timeline,
            chunks=chunks,
            used_ocr=extracted.used_ocr,
            warnings=warnings,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("--case_id", default="demo-case")
    args = parser.parse_args()

    result = DocumentIngestionPipeline().ingest(args.file_path, case_id=args.case_id)
    print(f"doc_type   : {result.classification.doc_type} ({result.classification.confidence:.2f})")
    print(f"used_ocr   : {result.used_ocr}")
    print(f"entities   : {result.entities}")
    print(f"timeline   : {len(result.timeline)} dated events")
    print(f"chunks     : {len(result.chunks)}")
    for ch in result.chunks[:5]:
        print(f"  [{ch.chunk_role}] {ch.text[:60]!r}")
    if result.warnings:
        print("warnings   :", result.warnings)
