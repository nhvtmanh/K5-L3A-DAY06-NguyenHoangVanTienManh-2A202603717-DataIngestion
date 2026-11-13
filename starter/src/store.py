"""Persistent pgvector storage, exact cosine search and typed metadata filters."""
import json
import math
from numbers import Real
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from .models import Chunk

FILTER_FIELDS = frozenset({
    'section', 'year', 'version', 'category', 'audience', 'language',
    'effective_date', 'document_type',
})


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int, *, schema: str = 'public') -> None:
        """Retain settings. A table has one dimension and must use one model/config."""
        self._check_text(database_url, 'database_url')
        self._check_text(schema, 'schema')
        if type(dimensions) is not int or not 1 <= dimensions <= 16000:
            raise ValueError('dimensions must be an integer from 1 to 16000.')
        self._database_url = database_url
        self.dimensions = dimensions
        self.schema = schema
        self._table = sql.Identifier(schema, 'ingestion_chunks')

    def _connect(self):
        return psycopg.connect(self._database_url, connect_timeout=5, row_factory=dict_row)

    @staticmethod
    def _check_text(value, name):
        if not isinstance(value, str) or not value.strip() or '\x00' in value:
            raise ValueError(f'{name} must be a non-empty string without NUL characters.')

    def _vector(self, vector):
        if not isinstance(vector, (list, tuple)) or len(vector) != self.dimensions:
            raise ValueError(f'Vector must contain exactly {self.dimensions} numbers.')
        if any(isinstance(v, bool) or not isinstance(v, Real) for v in vector):
            raise ValueError('Vector values must be real numbers.')
        values = [float(v) for v in vector]
        if not all(math.isfinite(v) for v in values) or not any(v != 0 for v in values):
            raise ValueError('Cosine search requires finite, non-zero vectors.')
        # Bind this string as a value and cast in SQL; no Python pgvector adapter needed.
        return '[' + ','.join(repr(v) for v in values) + ']'

    def create_schema(self) -> None:
        """Create storage; require pgvector and reject incompatible existing dimensions.

        No approximate vector index is created: search is exact for this corpus.
        """
        with self._connect() as connection:
            if not connection.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone():
                raise RuntimeError('pgvector extension is missing; run database setup first.')
            connection.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(self.schema)))
            connection.execute(sql.SQL('''
                CREATE TABLE IF NOT EXISTS {} (
                    strategy text NOT NULL, chunk_id text NOT NULL,
                    document_id text NOT NULL, source text NOT NULL,
                    page integer NOT NULL CHECK (page > 0),
                    text text NOT NULL CHECK (length(btrim(text)) > 0),
                    embedding vector({}) NOT NULL,
                    metadata jsonb NOT NULL CHECK (jsonb_typeof(metadata) = 'object'),
                    PRIMARY KEY (strategy, document_id, chunk_id)
                )
            ''').format(self._table, sql.Literal(self.dimensions)))
            column = connection.execute('''
                SELECT format_type(a.atttypid, a.atttypmod) AS type
                FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = %s AND c.relname = 'ingestion_chunks'
                    AND a.attname = 'embedding' AND NOT a.attisdropped
            ''', (self.schema,)).fetchone()
            if column is None or column['type'] != f'vector({self.dimensions})':
                raise RuntimeError('Existing vector dimensions differ; use a separate schema.')
            connection.execute(sql.SQL('CREATE INDEX IF NOT EXISTS {} ON {} USING gin (metadata jsonb_path_ops)').format(
                sql.Identifier('ingestion_chunks_metadata_idx'), self._table,
            ))

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]],
               *, document_ids: list[str] | None = None) -> None:
        """Replace the COMPLETE chunk snapshot of each supplied document atomically.

        Pass all current chunks per document, not incremental embedding batches.
        Delete obsolete chunks only for these documents/this strategy. To clear
        a document now yielding no chunks, explicitly pass its ID in document_ids.
        Empty input without document_ids is a no-op. Other documents are retained.
        """
        self._check_text(strategy, 'strategy')
        if len(chunks) != len(vectors):
            raise ValueError('There must be exactly one vector per chunk.')
        if document_ids is not None and not isinstance(document_ids, list):
            raise ValueError('document_ids must be a list.')
        scope = set()
        for doc_id in document_ids or []:
            self._check_text(doc_id, 'document_id')
            scope.add(doc_id)
        rows, seen, current_ids = [], set(), {}
        for chunk, vector in zip(chunks, vectors):
            for field in ('chunk_id', 'document_id', 'source', 'text'):
                self._check_text(getattr(chunk, field), field)
            if type(chunk.page) is not int or chunk.page <= 0:
                raise ValueError('Chunk page must be a positive integer.')
            if not isinstance(chunk.metadata, dict):
                raise ValueError('Chunk metadata must be a JSON object.')
            for field in ('document_id', 'source', 'page'):
                if field in chunk.metadata and chunk.metadata[field] != getattr(chunk, field):
                    raise ValueError(f'Metadata contradicts chunk {field}.')
            if document_ids is not None and chunk.document_id not in scope:
                raise ValueError('Chunk is outside the explicit document replacement scope.')
            key = (chunk.document_id, chunk.chunk_id)
            if key in seen:
                raise ValueError('Duplicate chunk ID within a document.')
            seen.add(key)
            scope.add(chunk.document_id)
            current_ids.setdefault(chunk.document_id, []).append(chunk.chunk_id)
            rows.append((strategy, chunk.chunk_id, chunk.document_id, chunk.source,
                         chunk.page, chunk.text, self._vector(vector),
                         json.dumps(chunk.metadata, ensure_ascii=False, allow_nan=False)))
        if not scope:
            return
        # The connection context commits on success and rolls back on exception.
        with self._connect() as connection:
            connection.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))',
                               (f'{self.schema}.ingestion_chunks:{strategy}',))
            if rows:
                with connection.cursor() as cursor:
                    cursor.executemany(sql.SQL('''
                        INSERT INTO {} (strategy, chunk_id, document_id, source, page, text, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb)
                        ON CONFLICT (strategy, document_id, chunk_id) DO UPDATE SET
                            source = EXCLUDED.source, page = EXCLUDED.page,
                            text = EXCLUDED.text, embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata
                    ''').format(self._table), rows)
            for doc_id in sorted(scope):
                connection.execute(sql.SQL('''
                    DELETE FROM {} WHERE strategy = %s AND document_id = %s
                        AND NOT (chunk_id = ANY(%s::text[]))
                ''').format(self._table), (strategy, doc_id, current_ids.get(doc_id, [])))

    def count_chunks(self, strategy: str) -> int:
        """Count a strategy snapshot without retrieving text or running similarity."""
        self._check_text(strategy, 'strategy')
        with self._connect() as connection:
            return connection.execute(sql.SQL('SELECT count(*) AS count FROM {} WHERE strategy = %s').format(
                self._table), (strategy,)).fetchone()['count']

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter by typed metadata equality and rank by exact cosine distance.

        score = 1 - cosine_distance (larger is better). Return fewer than top_k
        when insufficient rows match; never retry without the requested filter.
        """
        self._check_text(strategy, 'strategy')
        vector = self._vector(query_vector)
        if type(top_k) is not int or top_k <= 0:
            raise ValueError('top_k must be a positive integer.')
        if filters is None:
            filters = {}
        if not isinstance(filters, dict) or not set(filters) <= FILTER_FIELDS:
            raise ValueError(f'Only these metadata filters are allowed: {sorted(FILTER_FIELDS)}')
        for key, value in filters.items():
            if value is None or not isinstance(value, (str, int, float, bool)):
                raise ValueError(f'Filter {key} requires a non-null scalar value.')
        filter_json = json.dumps(filters, ensure_ascii=False, allow_nan=False)
        with self._connect() as connection:
            rows = connection.execute(sql.SQL('''
                SELECT chunk_id, document_id, source, page, text, metadata,
                       embedding <=> %s::vector AS distance
                FROM {} WHERE strategy = %s AND metadata @> %s::jsonb
                ORDER BY distance ASC, document_id ASC, chunk_id ASC LIMIT %s
            ''').format(self._table), (vector, strategy, filter_json, top_k)).fetchall()
        for row in rows:
            row['score'] = 1.0 - row['distance']
            row['metadata'] = {**row['metadata'], 'document_id': row['document_id'],
                               'source': row['source'], 'page': row['page']}
        return rows
