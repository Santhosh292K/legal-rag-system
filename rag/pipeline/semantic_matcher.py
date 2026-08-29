"""
pipeline/semantic_matcher.py
Shared embedding-based label matcher used by intent_classifier.py,
domain_router.py, query_expander.py, and scenario_rewriter.py.

Given a dict of {label: [canonical example phrasings]}, this embeds every
example once at construction time and scores a new query against each
label by cosine similarity to that label's closest examples (a small
soft k-NN average, same shape as pipeline/query_router.py's SemanticRouter
so behaviour is consistent across the codebase).

embed_fn follows the convention used everywhere else in this codebase
(SectionPinner, SemanticRouter, main.py's wiring): embed_fn(list[str]) ->
vectors, already L2-normalized (normalize_embeddings=True), so cosine
similarity reduces to a dot product. This class still defensively
re-normalizes so it behaves correctly even if a caller's embed_fn isn't
pre-normalized.

embed_fn is optional. Every call site already guards on `self.embed_fn`
before calling `.match()`, so when embed_fn is None this class computes no
embeddings at construction time and `match()` simply returns [].

examples_path is an optional JSON file of extra {label: [examples]} to
merge on top of the code-defined label_examples, so new phrasings can be
added without a code change. It's silently skipped if the file doesn't
exist or fails to parse — an optional extras file must never break the
matcher.
"""
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# How many nearest examples per label to average over — mirrors
# query_router.py's SemanticRouter so a label's score isn't just "did it
# happen to be close to one single example" but stays robust to any one
# oddly-phrased example while still not being averaged away by a large set.
TOP_K_EXAMPLES_PER_LABEL = 3


class SemanticMatcher:
    def __init__(
        self,
        label_examples: dict[str, list[str]],
        embed_fn: Optional[Callable[[list[str]], object]] = None,
        examples_path: Optional[str] = None,
    ):
        self.embed_fn = embed_fn
        self.label_examples = self._merge_examples(label_examples, examples_path)

        self._labels: list[str] = []
        self._example_texts: list[str] = []
        for label, examples in self.label_examples.items():
            for ex in examples:
                self._labels.append(label)
                self._example_texts.append(ex)

        self._label_indices: dict[str, np.ndarray] = {
            label: np.array([i for i, l in enumerate(self._labels) if l == label])
            for label in self.label_examples
        }

        # Only pay the embedding cost if there's actually an embed_fn and
        # something to embed — construction must stay cheap/safe when this
        # matcher is built without one (the common "no embed_fn wired up
        # yet" path exercised by every module's __main__ block).
        self._example_vectors: Optional[np.ndarray] = None
        if self.embed_fn and self._example_texts:
            vectors = self.embed_fn(self._example_texts)
            self._example_vectors = self._normalize(np.asarray(vectors, dtype=np.float32))

    @staticmethod
    def _merge_examples(
        label_examples: dict[str, list[str]], examples_path: Optional[str]
    ) -> dict[str, list[str]]:
        merged = {label: list(examples) for label, examples in label_examples.items()}
        if not examples_path or not Path(examples_path).exists():
            return merged
        try:
            with open(examples_path, "r", encoding="utf-8") as f:
                extra = json.load(f)
        except (OSError, json.JSONDecodeError):
            # A missing/malformed extras file should never break the
            # matcher — fall back to the code-defined examples only.
            return merged
        for label, examples in extra.items():
            existing = merged.setdefault(label, [])
            existing.extend(e for e in examples if e not in existing)
        return merged

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8  # guard a degenerate all-zero embedding
        return vectors / norms

    def match(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Return up to top_k (label, score) pairs, sorted by descending
        score. score is the cosine similarity between the query and each
        label's closest examples, averaged over up to
        TOP_K_EXAMPLES_PER_LABEL nearest examples for that label. Returns
        [] if no embed_fn is wired up or there are no examples to match."""
        if self.embed_fn is None or self._example_vectors is None:
            return []

        q_vec = self._normalize(np.asarray(self.embed_fn([query]), dtype=np.float32))[0]
        sims = self._example_vectors @ q_vec  # cosine sim, both sides normalized

        label_scores: dict[str, float] = {}
        for label, idx in self._label_indices.items():
            label_sims = sims[idx]
            k = min(TOP_K_EXAMPLES_PER_LABEL, len(label_sims))
            top_local = np.argsort(label_sims)[-k:]
            label_scores[label] = float(np.mean(label_sims[top_local]))

        ranked = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]