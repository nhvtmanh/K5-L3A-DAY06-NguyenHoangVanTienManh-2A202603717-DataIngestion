"""Local EmbeddingGemma 2 adapter for document and query embeddings."""
import math
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Load one real model and reuse it across batches; failures propagate.

    Install requirements-local.txt first. Documents and queries use the model's
    respective retrieval prompts, float32 inference and L2-normalized vectors.
    No API key is needed. local_files_only=True requires a complete local cache.
    """

    def __init__(
        self,
        model_name: str = 'google/embeddinggemma-2',
        *,
        batch_size: int = 8,
        device: str = 'cpu',
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        if model_name != 'google/embeddinggemma-2':
            raise ValueError('This adapter supports google/embeddinggemma-2 only.')
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer.')

        # Keep optional ML dependencies out of imports for other pipeline modules.
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(
            model_name,
            device=device,
            revision=revision,
            local_files_only=local_files_only,
            model_kwargs={'torch_dtype': torch.float32},
            config_kwargs={'vision_config': None, 'audio_config': None},
        )
        self.dimensions = self._model.get_embedding_dimension()
        if self.dimensions != 768:
            raise RuntimeError(f'Expected 768-dimensional embeddings, got {self.dimensions}.')
        configured_limit = self._model.max_seq_length
        if type(configured_limit) is not int or configured_limit <= 0:
            raise RuntimeError('The model did not expose a valid input token limit.')
        # Some tokenizer configs expose a huge sentinel, not a real context limit.
        # EmbeddingGemma 2's published text context is 8,192 tokens.
        self.max_tokens = min(8192, configured_limit)
        self._model.max_seq_length = self.max_tokens
        for prompt in ('Document', 'SearchQuery'):
            if prompt not in self._model.prompts:
                raise RuntimeError(f'Model configuration is missing prompt {prompt}.')

        self.revision = getattr(self._model[0].auto_model.config, '_commit_hash', None) or revision

    def _encode(self, texts: list[str], prompt_name: str) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError('texts must be a list of strings.')
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('Every input must be a non-empty string.')
        if not texts:
            return []

        # Refuse overlong input instead of silently truncating document evidence.
        prefix = self._model.prompts[prompt_name]
        tokens = self._model.tokenizer(
            [prefix + text for text in texts],
            add_special_tokens=True, truncation=False, padding=False,
            return_attention_mask=False,
        )['input_ids']
        for index, ids in enumerate(tokens):
            if len(ids) > self.max_tokens:
                raise ValueError(
                    f'Input {index} has {len(ids)} tokens including prompt/special tokens; '
                    f'model limit is {self.max_tokens}. Split it into smaller chunks.'
                )

        encoded = self._model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = encoded.tolist()
        if len(vectors) != len(texts):
            raise RuntimeError('Embedding count does not match input count.')
        for index, vector in enumerate(vectors):
            if len(vector) != self.dimensions:
                raise RuntimeError(f'Vector {index} has an unexpected dimension.')
            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError(f'Vector {index} contains NaN or infinity.')
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                raise RuntimeError(f'Vector {index} is not L2-normalized (norm={norm}).')
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one 768-dimensional document vector per text, in input order."""
        return self._encode(texts, 'Document')

    def embed_query(self, question: str) -> list[float]:
        """Return a compatible 768-dimensional vector for one search query."""
        return self._encode([question], 'SearchQuery')[0]
