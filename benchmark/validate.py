"""Validate benchmark labels against the selected local PDF; no retrieval/API calls."""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pypdf
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def normalize(text: str) -> str:
    """Follow character_count_method in the corpus manifest, retaining newlines."""
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n')
    lines = [re.sub(r'[^\S\n]+', ' ', line).strip() for line in text.split('\n')]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    benchmark_path = ROOT / 'benchmark/questions.json'
    manifest = json.loads((ROOT / 'starter/data/manifest.json').read_text(encoding='utf-8'))
    doc = manifest['documents'][0]
    pdf_path = ROOT / doc['local_path']
    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    require(pdf_hash == doc['snapshot']['sha256'], 'Source PDF differs from manifest snapshot')
    reader = PdfReader(pdf_path)
    require(len(reader.pages) == doc['physical_page_count'], 'PDF page count differs from manifest')
    pages, sections = {}, {}
    current_section = '1.1'  # Selected range begins inside section 1.1.
    for record in doc['pages']:
        n = record['pdf_page']
        text = normalize(reader.pages[n - 1].extract_text() or '')
        require(len(text) == record['normalized_char_count'], f'Character count differs on page {n}')
        pages[n] = text
        starts = [(0, current_section)]
        for heading in re.finditer(r'^(1\.[1-8]) [A-Z][^\n]+', text, flags=re.MULTILINE):
            starts.append((heading.start(), heading.group(1)))
            current_section = heading.group(1)
        sections[n] = starts

    questions = json.loads(benchmark_path.read_text(encoding='utf-8'))
    ids = [q['id'] for q in questions]
    require(len(ids) == len(set(ids)), 'Duplicate question IDs')
    require(len({q['question'] for q in questions}) == len(questions), 'Duplicate questions')
    splits = Counter(q['split'] for q in questions)
    require(set(splits) <= {'dev', 'test'}, 'Unknown split')
    require(splits['dev'] >= 15 and splits['test'] >= 5, 'Insufficient dev/test questions')
    require(len(questions) >= 20, 'Fewer than 20 questions')
    filtered = Counter(q['split'] for q in questions if q['filter'])
    require(sum(filtered.values()) >= 3 and filtered['test'] >= 1, 'Insufficient filter cases')
    audit = []
    for q in questions:
        for field in ['id', 'split', 'type', 'question', 'expected_answer', 'gold_source_id', 'gold_page', 'evidence_quote']:
            require(bool(q.get(field)), f"{q['id']}: missing {field}")
        require(q['gold_page_id'] == f"{doc['source_id']}_p{q['gold_page']:03d}", f"{q['id']}: incorrect page ID")
        require(q['gold_printed_page'] == q['gold_page'] - 12, f"{q['id']}: incorrect printed page")
        if q['filter']:
            require(q['filter'] == {'section': q['gold_section']}, f"{q['id']}: unexpected filter")
            require(f"mục {q['filter']['section']}" in q['question'], f"{q['id']}: filter scope absent from question")
            require(bool(q['filter_origin']), f"{q['id']}: undocumented filter origin")
        else:
            require(q['filter'] is None, f"{q['id']}: use null for no filter")
        alternatives = [q] + q['alternative_gold']
        for index, gold in enumerate(alternatives):
            n = gold['gold_page']
            require(gold['gold_source_id'] == doc['source_id'], f"{q['id']}: unknown source")
            require(n in pages, f"{q['id']}: page outside manifest selection")
            quote = gold['evidence_quote']
            require(quote and quote in pages[n], f"{q['id']}: evidence missing on page {n}")
            offset = pages[n].index(quote)
            actual_section = [section for start, section in sections[n] if start <= offset][-1]
            require(gold['gold_section'] == actual_section, f"{q['id']}: section differs from PDF heading")
            if q['filter']:
                require(q['filter']['section'] == actual_section, f"{q['id']}: gold violates filter")
            audit.append({
                'question_id': q['id'], 'gold_kind': 'primary' if index == 0 else 'alternative',
                'pdf_page': n, 'section': actual_section,
                'evidence_start_char': offset, 'evidence_end_char_exclusive': offset + len(quote),
                'evidence_char_count': len(quote), 'exact_normalized_substring': True,
            })
    report = {
        'status': 'passed', 'pdf_sha256': pdf_hash,
        'questions_sha256': hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        'extractor': f'pypdf {pypdf.__version__}',
        'question_count': len(questions), 'split_counts': dict(splits),
        'filter_counts': dict(filtered), 'type_counts': dict(Counter(q['type'] for q in questions)),
        'selected_page_count': len(pages), 'verified_evidence_count': len(audit),
        'limits': 'Checks source identity, structure, page/section scope and exact evidence presence. Does not measure retrieval quality or prove semantic completeness of all gold alternatives. No embedding or retrieval run.',
        'evidence_checks': audit,
    }
    (ROOT / 'benchmark/evidence_validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"PASS: {len(questions)} questions; {splits['dev']} dev / {splits['test']} test; "
          f"{sum(filtered.values())} filtered; {len(audit)} verified evidence spans.")


if __name__ == '__main__':
    main()
