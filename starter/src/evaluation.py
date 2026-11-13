"""Conservative evidence matching and retrieval metrics, separate from human labels."""
import re
import unicodedata
from collections import defaultdict
from copy import deepcopy
from typing import Any


def _comparable(text: str) -> str:
    # Chunking preserves PDF line breaks; whitespace layout is not evidence content.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def matches_gold(hit: dict[str, Any], gold: dict[str, Any]) -> bool:
    """Require the document ID, physical page and complete evidence in ONE hit."""
    metadata = hit.get('metadata', {})
    document_id = hit.get('document_id', metadata.get('document_id'))
    page = hit.get('page', metadata.get('page'))
    quote = _comparable(gold['evidence_quote'])
    return bool(quote and document_id == gold['gold_source_id']
                and page == gold['gold_page'] and quote in _comparable(hit['text']))


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Score top-3 with complete literal evidence, accepting any alternative gold.

    This conservative automatic rule needs review for semantically equivalent
    evidence. It never invents coherence, grounding or metadata correctness labels.
    """
    gold = [{key: question[key] for key in ('gold_source_id', 'gold_page', 'evidence_quote')}]
    gold.extend(deepcopy(question.get('alternative_gold', [])))
    ranked = []
    for rank, hit in enumerate(hits[:3], 1):
        record = deepcopy(hit)
        record['rank'] = rank
        record['matched_gold_indices'] = [i for i, label in enumerate(gold) if matches_gold(hit, label)]
        record['evidence_hit'] = bool(record['matched_gold_indices'])
        ranked.append(record)
    return {
        'question_id': question['id'], 'split': question['split'],
        'type': question.get('type'), 'question': question['question'],
        'expected_answer': question['expected_answer'],
        'requested_filter': deepcopy(question.get('filter')),
        'gold': gold, 'hits': ranked,
        'hit_at_1': bool(ranked and ranked[0]['evidence_hit']),
        'evidence_hit_at_3': any(hit['evidence_hit'] for hit in ranked),
        'evaluation_method': 'full_evidence_literal_match_after_NFC_and_whitespace_normalization',
        'needs_equivalent_evidence_review': not any(hit['evidence_hit'] for hit in ranked),
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate numerator/denominator and ratios by strategy and split."""
    groups = defaultdict(list)
    seen = set()
    for trace in traces:
        key = (trace['strategy'], trace['split'], trace['question_id'])
        if key in seen:
            raise ValueError('Duplicate question within a strategy/split would bias metrics.')
        seen.add(key)
        groups[key[:2]].append(trace)
    metrics = []
    for (strategy, split), records in sorted(groups.items()):
        total = len(records)
        first = sum(record['hit_at_1'] for record in records)
        third = sum(record['evidence_hit_at_3'] for record in records)
        filtered = [record for record in records if record['requested_filter']]
        # Empty results provide no evidence of filter usefulness; count as failure here.
        compliant = sum(bool(record['hits']) and all(
            all(hit['metadata'].get(key) == value for key, value in record['requested_filter'].items())
            for hit in record['hits']
        ) for record in filtered)
        metrics.append({
            'strategy': strategy, 'split': split, 'question_count': total,
            'hit_at_1': {'numerator': first, 'denominator': total, 'rate': first / total},
            'evidence_hit_at_3': {'numerator': third, 'denominator': total, 'rate': third / total},
            'filter_compliance': {
                'numerator': compliant, 'denominator': len(filtered),
                'rate': compliant / len(filtered) if filtered else None,
                'method': 'nonempty_top3_all_match_requested_filter; independent_of_applied_filter',
            },
        })
    return {'metrics': metrics, 'manual_assessment_complete': False,
            'limitations': 'Literal evidence matching is conservative; review equivalent evidence. '
                           'No automatic coherence, grounding, or metadata correctness scores.'}
