"""Read PDF pages and normalize text while retaining source traceability."""
import logging
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

logger = logging.getLogger(__name__)


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    *,
    page_numbers: Iterable[int] | None = None,
) -> list[DocumentPage]:
    """Read selected 1-based physical PDF pages, or all pages if omitted.

    Pass page numbers from the manifest to restrict the corpus. Selected pages
    retain their original numbers and input order; duplicates are rejected.
    Empty pages are returned with ``text_empty=True`` and logged for review.
    Image-only pages need a separate OCR step; this function does not run OCR.
    Extraction errors are logged and raised rather than silently skipping pages.
    """
    pages: list[DocumentPage] = []
    with Path(path).open('rb') as pdf_file:
        reader = PdfReader(pdf_file)
        total_pages = len(reader.pages)
        selected = list(range(1, total_pages + 1)) if page_numbers is None else list(page_numbers)
        if not selected:
            raise ValueError('Select at least one PDF page.')
        if any(type(number) is not int or not 1 <= number <= total_pages for number in selected):
            raise ValueError(f'Page numbers must be integers from 1 to {total_pages}.')
        if len(selected) != len(set(selected)):
            raise ValueError('Page numbers must not contain duplicates.')

        for page_number in selected:
            try:
                raw_text = reader.pages[page_number - 1].extract_text() or ''
                text = normalize_text(raw_text)
            except Exception:
                logger.exception('Failed to extract PDF %s, page %d', path, page_number)
                raise

            if not text:
                logger.warning(
                    'No extractable text in PDF %s, page %d; check for a blank page or OCR needs.',
                    path, page_number,
                )
            pages.append(DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={
                    'text_empty': not bool(text),
                    'normalized_char_count': len(text),
                },
            ))
    return pages


def normalize_text(text: str) -> str:
    """Follow manifest normalization, preserving lines and paragraph boundaries.

    Keep headers, footers and punctuation so benchmark evidence stays traceable.
    This does not reconstruct paragraphs lost during PDF text extraction.
    """
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [re.sub(r'[^\S\n]+', ' ', line).strip() for line in text.split('\n')]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()
