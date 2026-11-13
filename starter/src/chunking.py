"""Chunk normalized text without crossing physical pages or annotated sections."""
import hashlib
import json
import re
from dataclasses import replace

from .models import Chunk, DocumentPage

# Numeric section headings in the selected textbook; not figure/table captions.
_SECTION_HEADING = re.compile(r'^(\d+\.\d+) [A-Z][^\n]+$', re.MULTILINE)


def annotate_sections(
    pages: list[DocumentPage], *, initial_section: str | None = None,
) -> list[DocumentPage]:
    """Return copies with section spans measured in original normalized text.

    Supply consecutive pages from one document in reading order. initial_section
    must come from source inspection if the selection starts inside a section.
    Unknown leading content retains section=None rather than inventing metadata.
    PDF text reading order can differ from visual order; review floating boxes.
    """
    annotated = []
    current_section = initial_section
    previous = None
    for page in pages:
        if previous is not None and (
            page.document_id != previous.document_id
            or page.source != previous.source
            or page.page != previous.page + 1
        ):
            raise ValueError('Annotate one document with consecutive pages in ascending order.')
        spans = []
        start = 0
        for heading in _SECTION_HEADING.finditer(page.text):
            if heading.start() > start:
                spans.append({'start': start, 'end': heading.start(), 'section': current_section})
            start = heading.start()
            current_section = heading.group(1)
        if start < len(page.text):
            spans.append({'start': start, 'end': len(page.text), 'section': current_section})
        metadata = {**page.metadata, 'section_spans': spans}
        metadata.pop('section', None)  # A physical page can contain multiple sections.
        annotated.append(replace(page, metadata=metadata))
        previous = page
    return annotated


def _section_spans(page: DocumentPage) -> list[dict]:
    spans = page.metadata.get('section_spans')
    if spans is None:
        # Generic callers may already have a single-section page.
        return [{'start': 0, 'end': len(page.text), 'section': page.metadata.get('section')}]
    end = 0
    for span in spans:
        start, stop = span['start'], span['end']
        if (type(start) is not int or type(stop) is not int
                or start != end or not start < stop <= len(page.text)):
            raise ValueError('Section spans must cover page text consecutively without overlap.')
        end = stop
    if end != len(page.text):
        raise ValueError('Section spans must cover all page text.')
    return spans


def _make_chunk(page, start, end, section, method, parameters):
    # Trim boundary whitespace while retaining exact source offsets and text.
    while start < end and page.text[start].isspace():
        start += 1
    while end > start and page.text[end - 1].isspace():
        end -= 1
    if start == end:
        return None
    text = page.text[start:end]
    identity = json.dumps(
        [page.document_id, page.source, page.page, method, parameters, section, start, end, text],
        ensure_ascii=False, sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()
    metadata = {key: value for key, value in page.metadata.items() if key != 'section_spans'}
    if 'normalized_char_count' in metadata:
        metadata['page_normalized_char_count'] = metadata['normalized_char_count']
    metadata.update({
        'section': section, 'start_char': start, 'end_char': end,
        'normalized_char_count': len(text), 'chunking_method': method,
        'chunking_parameters': dict(parameters),
    })
    return Chunk(
        chunk_id=digest, document_id=page.document_id, source=page.source,
        page=page.page, text=text, metadata=metadata,
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Create character windows within each section; require 0 <= overlap < size."""
    if type(size) is not int or size <= 0:
        raise ValueError('size must be a positive integer.')
    if type(overlap) is not int or not 0 <= overlap < size:
        raise ValueError('overlap must be an integer from 0 to size - 1.')
    chunks = []
    for span in _section_spans(page):
        start = span['start']
        while start < span['end']:
            end = min(start + size, span['end'])
            chunk = _make_chunk(page, start, end, span['section'], 'fixed', {'size': size, 'overlap': overlap})
            if chunk is not None:
                chunks.append(chunk)
            if end == span['end']:
                break  # Do not create an extra chunk contained in the last one.
            start += size - overlap
    return chunks


def _paragraph_ranges(text: str) -> list[tuple[int, int]]:
    """Detect blank lines, or sentence-ending line breaks in extracted PDFs.

    The latter is a heuristic when PDF extraction loses blank paragraph lines.
    Ordinary wrapped lines stay together. Never rewrite the extracted text.
    """
    boundaries = {0, len(text)}
    for match in re.finditer(r'\n[ \t]*\n+', text):
        boundaries.add(match.end())
    for match in re.finditer(r'''[.!?][\d”"')]*\n(?=[A-Z•])''', text):
        boundaries.add(match.end())
    ordered = sorted(boundaries)
    return list(zip(ordered, ordered[1:]))


def _split_oversized(text: str, start: int, end: int, max_size: int):
    """Prefer sentence, then word boundaries; hard-split a token if necessary."""
    while end - start > max_size:
        limit = start + max_size
        window = text[start:limit]
        sentences = list(re.finditer(r'[.!?][\d”"\')]*(?=\s)', window))
        if sentences:
            stop = start + sentences[-1].end()
        else:
            spaces = list(re.finditer(r'\s+', window))
            stop = start + spaces[-1].end() if spaces else limit
        yield start, stop
        start = stop
    if start < end:
        yield start, end


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group adjacent paragraph units up to max_size, within a page/section.

    Oversized paragraphs are split at sentence/word boundaries where possible.
    No overlap is added. B and C should reuse these same chunks and vectors.
    """
    if type(max_size) is not int or max_size <= 0:
        raise ValueError('max_size must be a positive integer.')
    chunks = []
    for span in _section_spans(page):
        pending = None

        def emit(start, end):
            chunk = _make_chunk(page, start, end, span['section'], 'paragraph', {'max_size': max_size})
            if chunk is not None:
                chunks.append(chunk)

        section_text = page.text[span['start']:span['end']]
        for local_start, local_end in _paragraph_ranges(section_text):
            start, end = span['start'] + local_start, span['start'] + local_end
            if not page.text[start:end].strip():
                continue
            # Separator whitespace should not force an otherwise fitting paragraph to split.
            while start < end and page.text[start].isspace():
                start += 1
            while end > start and page.text[end - 1].isspace():
                end -= 1
            for part_start, part_end in _split_oversized(page.text, start, end, max_size):
                if pending is not None and part_end - pending[0] > max_size:
                    emit(*pending)
                    pending = None
                pending = (part_start if pending is None else pending[0], part_end)
        if pending is not None:
            emit(*pending)
    return chunks
