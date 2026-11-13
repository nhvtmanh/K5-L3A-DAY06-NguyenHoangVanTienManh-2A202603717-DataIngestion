"""Run the real ingestion/retrieval experiment; dev by default, test explicitly."""
import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import statistics
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

from .chunking import annotate_sections, by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, matches_gold, summarize
from .ingestion import load_pdf
from .store import PgVectorStore

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_jsonl(path, records):
    with path.open('w', encoding='utf-8') as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def cached_embeddings(embedder, texts, kind, cache_dir, model_config):
    """Cache by exact texts, role and pinned model config, not benchmark labels."""
    key = fingerprint({'model': model_config, 'kind': kind, 'texts': texts})
    path = cache_dir / f'{key}.json'
    if path.exists():
        vectors = json.loads(path.read_text(encoding='utf-8'))
        if (len(vectors) != len(texts) or any(
            not isinstance(vector, list) or len(vector) != embedder.dimensions
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)
            or not math.isclose(sum(v*v for v in vector), 1.0, abs_tol=1e-4)
            for vector in vectors
        )):
            raise ValueError('Invalid embedding cache; remove the specific cache file and retry.')
        return vectors, True
    vectors = []
    if kind == 'document':
        for start in range(0, len(texts), embedder.batch_size):
            vectors.extend(embedder.embed_documents(texts[start:start + embedder.batch_size]))
            if start == 0 or (start // embedder.batch_size) % 5 == 0 or len(vectors) == len(texts):
                LOGGER.info('Embedded document chunks %d/%d', len(vectors), len(texts))
    else:
        vectors = [embedder.embed_query(text) for text in texts]
    write_json(path, vectors)
    return vectors, False


def load_inputs(manifest_path, benchmark_path, initial_sections):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    questions = json.loads(benchmark_path.read_text(encoding='utf-8'))
    if (len(questions) < 20 or sum(q['split'] == 'dev' for q in questions) < 15
            or sum(q['split'] == 'test' for q in questions) < 5):
        raise ValueError('Benchmark requires at least 20 questions: 15 dev and 5 test.')
    if len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Duplicate benchmark question IDs.')
    filtered = [q for q in questions if q.get('filter')]
    if len(filtered) < 3 or not any(q['split'] == 'test' for q in filtered):
        raise ValueError('Benchmark requires >=3 filter questions, including >=1 test.')
    pages, sources = [], {}
    for document in manifest['documents']:
        source_id = document['source_id']
        if source_id in sources:
            raise ValueError('Duplicate manifest source_id.')
        path = ROOT / document['local_path']
        if sha256(path) != document['snapshot']['sha256']:
            raise ValueError(f'PDF snapshot differs from manifest for {source_id}.')
        selected = load_pdf(path, source_id, document['source_path_or_url'],
                            page_numbers=[p['pdf_page'] for p in document['pages']])
        if len(selected) != document['selected_page_count']:
            raise ValueError('Selected page count differs from manifest.')
        for page, record in zip(selected, document['pages']):
            if not page.text or len(page.text) != record['normalized_char_count']:
                raise ValueError(f'Empty text or character count mismatch: {source_id}, page {page.page}.')
        pages.extend(annotate_sections(selected, initial_section=initial_sections.get(source_id)))
        sources[source_id] = document['snapshot']['sha256']
    if len(pages) < 30:
        raise ValueError('Corpus requires at least 30 selected content pages.')
    lookup = {(p.document_id, p.page): p for p in pages}
    for q in questions:
        for gold in [q] + q.get('alternative_gold', []):
            page = lookup.get((gold['gold_source_id'], gold['gold_page']))
            if page is None or not gold['evidence_quote'] or gold['evidence_quote'] not in page.text:
                raise ValueError(f"Evidence no longer matches source: {q['id']}.")
    return pages, questions, sources


def coherence_samples(pages, strategies):
    """Choose the first nonspace character on the first 10 sorted source pages."""
    samples = []
    for page in sorted(pages, key=lambda p: (p.document_id, p.page))[:10]:
        anchor = next(i for i, char in enumerate(page.text) if not char.isspace())
        for strategy, chunks in strategies.items():
            covering = [c for c in chunks if c.document_id == page.document_id and c.page == page.page
                        and c.metadata['start_char'] <= anchor < c.metadata['end_char']]
            samples.append({
                'strategy': strategy, 'document_id': page.document_id,
                'page': page.page, 'anchor_char': anchor,
                'chunks_covering_anchor': [asdict(c) for c in covering],
                'coherence_score_0_to_2': None, 'reason': None,
            })
    return samples


def run(args, output_dir):
    load_dotenv(ROOT / 'starter/.env')
    if not args.allow_download:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if args.split == 'test' and args.config is None:
        raise ValueError('For test, pass --config <dev-run>/run_config.json after fixing the configuration.')
    implementation_hash = fingerprint({name: sha256(ROOT / 'starter/src' / name)
        for name in ('run.py', 'evaluation.py', 'ingestion.py', 'chunking.py', 'embeddings.py', 'store.py', 'models.py')})
    if args.config:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        if config.get('format_version') != 1 or config.get('implementation_sha256') != implementation_hash:
            raise ValueError('Pipeline code differs from the frozen dev run; rerun dev before test.')
        if any(value is not None for value in (args.chunk_size, args.overlap, args.paragraph_size)):
            raise ValueError('Do not combine a frozen --config with chunk parameter overrides.')
        if config['inputs']['manifest_sha256'] != sha256(args.manifest) or config['inputs']['benchmark_sha256'] != sha256(args.benchmark):
            raise ValueError('Manifest/benchmark differs from the frozen run configuration.')
    else:
        config = {
            'format_version': 1, 'implementation_sha256': implementation_hash,
            'chunking': {'size': args.chunk_size if args.chunk_size is not None else 500,
                         'overlap': args.overlap if args.overlap is not None else 50,
                         'paragraph_max_size': args.paragraph_size if args.paragraph_size is not None else 500},
            'initial_sections': {'openstax_business_2018': '1.1'},
            'inputs': {'manifest_sha256': sha256(args.manifest), 'benchmark_sha256': sha256(args.benchmark)},
            'model': {'name': 'google/embeddinggemma-2', 'dimensions': 768,
                      'document_prompt': 'Document', 'query_prompt': 'SearchQuery',
                      'normalize_embeddings': True, 'dtype': 'float32'},
            'selection_criteria': 'dev Evidence Hit@3, then dev Hit@1, then fewer chunks',
        }
    params = config['chunking']
    if (type(params['size']) is not int or params['size'] <= 0
            or type(params['overlap']) is not int or not 0 <= params['overlap'] < params['size']
            or type(params['paragraph_max_size']) is not int or params['paragraph_max_size'] <= 0):
        raise ValueError('Invalid chunk parameters.')
    started = time.perf_counter()
    pages, questions, sources = load_inputs(args.manifest, args.benchmark, config['initial_sections'])
    chunks_a = [c for p in pages for c in fixed_size(p, params['size'], params['overlap'])]
    chunks_b = [c for p in pages for c in by_paragraph(p, params['paragraph_max_size'])]
    strategies = {'A': chunks_a, 'B': chunks_b, 'C': chunks_b}
    if any(c.metadata.get('section') is None for c in chunks_a + chunks_b):
        raise ValueError('Unknown section metadata; inspect source and configure the initial section.')
    prepare_seconds = time.perf_counter() - started
    # Fixed before retrieval scores are generated, with identical source anchors.
    write_json(output_dir / 'manual_coherence.json', coherence_samples(pages, strategies))
    write_jsonl(output_dir / 'chunks_A.jsonl', (asdict(c) for c in chunks_a))
    write_jsonl(output_dir / 'chunks_B.jsonl', (asdict(c) for c in chunks_b))

    # Resolve an actual cached/downloaded commit so cache and frozen test runs
    # cannot accidentally use a different model behind the mutable main branch.
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(config['model']['name'],
                    revision=config['model'].get('revision'), local_files_only=not args.allow_download))
    revision = snapshot.name
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Cannot resolve a model commit hash for reproducible embeddings.')
    config['model']['revision'] = revision
    embedder = MyEmbedder(revision=revision, batch_size=args.batch_size,
                          local_files_only=not args.allow_download)
    store = PgVectorStore(os.environ['DATABASE_URL'], embedder.dimensions)
    store.create_schema()
    dataset_id = fingerprint({'model': config['model'], 'chunking': params,
                              'sources': sources, 'manifest': config['inputs']['manifest_sha256']})[:20]
    stored_strategies = {s: f'{dataset_id}_{s}' for s in ('A', 'B')}
    config['storage'] = {'table': 'public.ingestion_chunks', 'strategies': stored_strategies,
                         'C_reuses': 'B', 'search': 'exact_cosine', 'top_k': 3}
    write_json(output_dir / 'run_config.json', config)
    write_json(output_dir / 'environment.json', {
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'split': args.split,
        'python': sys.version, 'device': 'cpu', 'batch_size': args.batch_size,
        'packages': {p: version(p) for p in ['pypdf', 'psycopg', 'torch', 'torchvision', 'Pillow', 'transformers', 'sentence-transformers']},
        'model_config': config['model'],
        'api_cost_usd': 0, 'cost_scope': 'Local inference; excludes hardware/electricity.',
    })
    cache_dir = ROOT / 'starter/.embedding_cache'
    cache_dir.mkdir(exist_ok=True)
    resources = {}
    for name, chunks in [('A', chunks_a), ('B', chunks_b)]:
        LOGGER.info('Strategy %s: %d chunks', name, len(chunks))
        begin = time.perf_counter()
        vectors, cached = cached_embeddings(embedder, [c.text for c in chunks], 'document', cache_dir, config['model'])
        embedding_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        store.upsert(stored_strategies[name], chunks, vectors, document_ids=list(sources))
        store_seconds = time.perf_counter() - begin
        before = store.count_chunks(stored_strategies[name])
        store.upsert(stored_strategies[name], chunks, vectors, document_ids=list(sources))
        after = store.count_chunks(stored_strategies[name])
        if before != len(chunks) or after != before:
            raise RuntimeError('Re-ingestion changed chunk count unexpectedly.')
        lengths = [len(c.text) for c in chunks]
        tokenized = embedder._model.tokenizer(
            [embedder._model.prompts['Document'] + c.text for c in chunks],
            truncation=False, padding=False, return_attention_mask=False)['input_ids']
        resources[name] = {
            'chunk_count': len(chunks), 'total_characters': sum(lengths),
            'length_min': min(lengths), 'length_max': max(lengths),
            'length_mean': statistics.mean(lengths), 'length_median': statistics.median(lengths),
            'total_input_tokens_with_prompt': sum(map(len, tokenized)),
            'shared_prepare_seconds': prepare_seconds, 'embedding_seconds': embedding_seconds,
            'embedding_cache_hit': cached, 'store_seconds': store_seconds,
            'ingest_seconds_excluding_reingest_check': prepare_seconds + embedding_seconds + store_seconds,
            'rows_before_reingest': before, 'rows_after_reingest': after, 'api_cost_usd': 0,
        }
    resources['C'] = {**resources['B'], 'reuses_vectors_and_storage_from': 'B', 'additional_document_embedding_seconds': 0}
    write_json(output_dir / 'resources.json', resources)
    selected = [q for q in questions if q['split'] == args.split]
    traces, manual = [], []
    with (output_dir / 'traces.jsonl').open('w', encoding='utf-8') as trace_file:
        for q in selected:
            begin = time.perf_counter()
            query_vectors, query_cached = cached_embeddings(embedder, [q['question']], 'query', cache_dir, config['model'])
            query_seconds = time.perf_counter() - begin
            for name in ('A', 'B', 'C'):
                applied_filter = q.get('filter') if name == 'C' else None
                begin = time.perf_counter()
                hits = store.search(stored_strategies['A' if name == 'A' else 'B'], query_vectors[0], filters=applied_filter)
                search_seconds = time.perf_counter() - begin
                result = evaluate_query(q, hits)
                result.update({'strategy': name, 'applied_filter': applied_filter,
                               'query_embedding_seconds': query_seconds, 'query_cache_hit': query_cached,
                               'search_seconds': search_seconds})
                gold = result['gold']
                result['evidence_exists_in_single_corpus_chunk'] = any(
                    matches_gold(asdict(c), label) for c in strategies[name] for label in gold)
                traces.append(result)
                trace_file.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                trace_file.flush()
                manual.append({
                    'question_id': q['id'], 'strategy': name, 'split': args.split,
                    'question': q['question'], 'gold': gold, 'requested_filter': q.get('filter'),
                    'hits': [{**hit, 'metadata_complete_and_correct': None,
                              'metadata_audit_reason': None} for hit in hits],
                    'grounding_score_0_to_2': None, 'grounding_reason': None,
                    'equivalent_evidence_review': None,
                })
            LOGGER.info('Retrieved %s (%s)', q['id'], args.split)
    summary = summarize(traces)
    write_json(output_dir / 'summary.json', summary)
    write_json(output_dir / 'manual_retrieval_review.json', manual)
    with (output_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(['strategy', 'split', 'questions', 'hit1_count', 'hit1_rate', 'hit3_count', 'hit3_rate'])
        for metric in summary['metrics']:
            writer.writerow([metric['strategy'], metric['split'], metric['question_count'],
                             metric['hit_at_1']['numerator'], metric['hit_at_1']['rate'],
                             metric['evidence_hit_at_3']['numerator'], metric['evidence_hit_at_3']['rate']])
            LOGGER.info('%s %s: Hit@1 %d/%d; Evidence Hit@3 %d/%d', metric['strategy'], args.split,
                        metric['hit_at_1']['numerator'], metric['question_count'],
                        metric['evidence_hit_at_3']['numerator'], metric['question_count'])
    LOGGER.info('Finished; results: %s', output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--config', type=Path, help='Frozen run_config.json from the chosen dev run; required for test.')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'starter/data/manifest.json')
    parser.add_argument('--benchmark', type=Path, default=ROOT / 'benchmark/questions.json')
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--paragraph-size', type=int)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--allow-download', action='store_true', help='Allow missing model files to download instead of requiring local cache.')
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output_dir = ROOT / 'starter/results' / f'{args.split}_{stamp}_{uuid.uuid4().hex[:6]}'
    output_dir.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s', handlers=[
        logging.StreamHandler(), logging.FileHandler(output_dir / 'run.log', encoding='utf-8'),
    ])
    try:
        run(args, output_dir)
    except Exception as error:
        # Never serialize .env or the database URL into run artifacts.
        message = str(error).replace(os.environ.get('DATABASE_URL') or '<no-database-url>', '[DATABASE_URL]')
        write_json(output_dir / 'error.json', {'error_type': type(error).__name__, 'message': message})
        LOGGER.error('%s: %s', type(error).__name__, message)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
