"""
pipeline/ocr_extractor.py
Text extraction from uploaded evidence documents.

Digital PDFs are read directly (PyMuPDF). Pages with little or no
extractable text are treated as scans and OCR'd (Tesseract). Plain
image uploads (jpg/png/tiff) always go through OCR.

Install:
    pip install pymupdf pytesseract pillow --break-system-packages
    # Tesseract binary must also be installed on the host (apt/brew).
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import OCR_MIN_CHARS_PER_PAGE, OCR_LANG

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class ExtractedPage:
    page_number: int
    text:        str
    used_ocr:    bool


@dataclass
class ExtractedDocument:
    file_path:      str
    pages:          list[ExtractedPage] = field(default_factory=list)
    used_ocr:        bool = False   # True if ANY page needed OCR
    error:           str | None = None

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def page_count(self) -> int:
        return len(self.pages)


def _ocr_image(image) -> str:
    import pytesseract
    return pytesseract.image_to_string(image, lang=OCR_LANG)


def _extract_pdf(file_path: str) -> ExtractedDocument:
    import fitz  # PyMuPDF

    doc = ExtractedDocument(file_path=file_path)
    pdf = fitz.open(file_path)

    try:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text") or ""

            if len(text.strip()) >= OCR_MIN_CHARS_PER_PAGE:
                doc.pages.append(ExtractedPage(page_number=i, text=text, used_ocr=False))
                continue

            # Sparse/no text → likely a scanned page, rasterize and OCR it.
            try:
                from PIL import Image
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                ocr_text = _ocr_image(img)
                doc.pages.append(ExtractedPage(page_number=i, text=ocr_text, used_ocr=True))
                doc.used_ocr = True
            except Exception as e:
                # Keep whatever thin text was extracted rather than losing the page.
                doc.pages.append(ExtractedPage(page_number=i, text=text, used_ocr=False))
                doc.error = f"OCR failed on page {i}: {e}"
    finally:
        pdf.close()

    return doc


def _extract_image(file_path: str) -> ExtractedDocument:
    from PIL import Image

    doc = ExtractedDocument(file_path=file_path, used_ocr=True)
    try:
        img = Image.open(file_path)
        text = _ocr_image(img)
        doc.pages.append(ExtractedPage(page_number=1, text=text, used_ocr=True))
    except Exception as e:
        doc.error = f"OCR failed: {e}"
    return doc


class DocumentExtractor:
    """Entry point: extract(file_path) -> ExtractedDocument."""

    def extract(self, file_path: str) -> ExtractedDocument:
        suffix = Path(file_path).suffix.lower()

        try:
            if suffix == ".pdf":
                return _extract_pdf(file_path)
            elif suffix in IMAGE_SUFFIXES:
                return _extract_image(file_path)
            else:
                return ExtractedDocument(
                    file_path=file_path,
                    error=f"Unsupported file type: {suffix}",
                )
        except Exception as e:
            return ExtractedDocument(file_path=file_path, error=str(e))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    args = parser.parse_args()

    result = DocumentExtractor().extract(args.file_path)
    print(f"pages={result.page_count} used_ocr={result.used_ocr} error={result.error}")
    print(result.full_text[:500])
