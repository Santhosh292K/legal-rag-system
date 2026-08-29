"""
pipeline/query_router.py
Orchestration layer — classifies each incoming query as general | document |
hybrid, so legal_rag_system.py knows which track(s) to call.

Routing is done by semantic similarity against a small labeled set of
example queries per route (pipeline/route_examples.json), instead of
hand-written regexes. A new way of phrasing "what applies here" is handled
by adding one example sentence to that file — the routing code itself
doesn't need to change, and it isn't limited to phrasings someone thought
to write a pattern for.

Falls back to an LLM call only when the semantic match is genuinely
ambiguous (the best route isn't clearly ahead of the others), same
rule-based-first / LLM-fallback shape as intent_classifier.py.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from config import OLLAMA_FAST_MODEL, EMBEDDING_MODEL
import ollama


EXAMPLES_PATH = Path(__file__).parent / "route_examples.json"
ROUTES = ("general", "document", "hybrid")

# How many nearest examples per route to average over — a small soft k-NN.
# With ~20 examples per route this stays robust to any single oddly-phrased
# example without letting the whole class average it away.
TOP_K = 3

# Softmax temperature applied to per-route similarity scores to get a
# confidence. Cosine similarities between short sentences from the same
# embedding model sit in a fairly narrow band (~0.3-0.9), so a small
# temperature is needed to turn a real but modest gap (e.g. 0.62 vs 0.54)
# into a confidence gap that's actually useful for the fallback gate below.
SOFTMAX_TEMPERATURE = 0.07

# Below this raw top-route similarity, don't trust the match at all even if
# it's the best of a bad set — let the LLM decide instead.
MIN_TOP_SIMILARITY = 0.35

# Confidence needed to skip the LLM fallback and trust the semantic match.
CONFIDENCE_GATE = 0.6


@dataclass
class RouteDecision:
    route:      str    # "general" | "document" | "hybrid"
    confidence: float
    reasoning:  str = ""


class SemanticRouter:
    """
    Embeds the labeled example set once, then routes new queries by
    similarity to those examples. Reuses an already-loaded embedding model
    when one is handed in (e.g. CaseIndexer's bge-large instance, wired up
    in legal_rag_system.py) instead of loading a second copy — the same
    embed_fn-sharing pattern pipeline/alea.py uses.
    """

    def __init__(self, embed_fn: Optional[Callable[[list], np.ndarray]] = None):
        self.embed_fn = embed_fn or self._default_embed_fn()

        with open(EXAMPLES_PATH) as f:
            self.examples: dict = json.load(f)

        missing = set(ROUTES) - set(self.examples)
        if missing:
            raise ValueError(f"route_examples.json is missing routes: {missing}")

        self._example_texts: list = []
        self._route_of_example: list = []
        for route in ROUTES:
            for text in self.examples[route]:
                self._example_texts.append(text)
                self._route_of_example.append(route)

        vectors = self.embed_fn(self._example_texts)
        self._example_vectors = np.asarray(vectors, dtype=np.float32)

        # Precompute per-route index arrays once so route() doesn't rebuild
        # them on every call.
        self._route_indices = {
            route: np.array([i for i, r in enumerate(self._route_of_example) if r == route])
            for route in ROUTES
        }

    @staticmethod
    def _default_embed_fn():
        """Only used if no shared model is passed in (e.g. running this
        module standalone). legal_rag_system.py normally passes CaseIndexer's
        already-loaded model instead."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        return lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def route(self, query: str) -> RouteDecision:
        q_vec = np.asarray(self.embed_fn([query]), dtype=np.float32)
        sims = (q_vec @ self._example_vectors.T)[0]  # cosine sim, both sides normalized

        route_scores = {}
        route_best_idx = {}
        for route in ROUTES:
            idx = self._route_indices[route]
            route_sims = sims[idx]
            k = min(TOP_K, len(route_sims))
            top_local = np.argsort(route_sims)[-k:]
            route_scores[route] = float(np.mean(route_sims[top_local]))
            route_best_idx[route] = idx[top_local[-1]]  # single closest example, for the explanation

        best_route = max(route_scores, key=route_scores.get)
        top_sim = route_scores[best_route]

        arr = np.array([route_scores[r] for r in ROUTES]) / SOFTMAX_TEMPERATURE
        arr = arr - arr.max()
        probs = np.exp(arr) / np.exp(arr).sum()
        confidence = float(probs[ROUTES.index(best_route)])

        if top_sim < MIN_TOP_SIMILARITY:
            # Best of a bad set — none of the routes actually resembled this
            # query. Force it below the gate so the LLM fallback decides.
            confidence = min(confidence, 0.3)

        example = self._example_texts[route_best_idx[best_route]]
        reasoning = (f"Closest in meaning to a '{best_route}'-style example "
                     f"(\"{example}\"), sim={top_sim:.2f}.")
        return RouteDecision(route=best_route, confidence=confidence, reasoning=reasoning)


ROUTE_PROMPT = """A user has an active case with uploaded legal documents (FIR,
chargesheet, evidence, etc.) and is asking a question. Decide whether
answering it needs:
- "general"  — an abstract legal-rule question (definitions, punishments,
               elements of an offence, procedure) asked in the abstract.
               A question naming an offence (e.g. "punishment for kidnapping
               for ransom") is "general" UNLESS it also references this
               specific case ("this FIR", "this case", "the accused here").
- "document" — needs facts recorded in the uploaded case documents
               specifically (names, dates, what a report/statement says
               happened) — NOT a request to look up what the law itself says.
- "hybrid"   — needs BOTH the case facts AND the general law (e.g. "what
               sections apply to what's described in this FIR?")

IMPORTANT: A query asking what the law/penalty/punishment/definition IS,
with no reference to "this case"/"this FIR"/"the accused" and no case-specific
detail, is "general" even when a case is currently loaded — the case being
open does not make every question about it. Do not route to "document"
just because documents happen to be uploaded.

Return ONLY JSON: {{"route": "...", "confidence": <0-1 float>, "reasoning": "<one sentence>"}}

Query: {query}"""


def llm_route(query: str) -> RouteDecision:
    response = ollama.chat(
        model=OLLAMA_FAST_MODEL,
        format="json",
        # Determinism — see irac_reranker.py's llm_irac_score for why.
        options={"temperature": 0},
        messages=[{"role": "user", "content": ROUTE_PROMPT.format(query=query)}],
    )
    data = json.loads(response["message"]["content"])
    route = data.get("route", "hybrid")
    if route not in {"general", "document", "hybrid"}:
        route = "hybrid"
    return RouteDecision(
        route=route,
        confidence=float(data.get("confidence", 0.5)),
        reasoning=data.get("reasoning", ""),
    )


class QueryRouter:
    def __init__(self, embed_fn: Optional[Callable[[list], np.ndarray]] = None):
        self.semantic = SemanticRouter(embed_fn=embed_fn)

    def route(self, query: str, case_id: Optional[str]) -> RouteDecision:
        if not case_id:
            # No case attached at all — there's nothing for a document track
            # to search, regardless of phrasing. Cheap and deterministic, so
            # no need to spend an embedding call on it.
            return RouteDecision(route="general", confidence=1.0,
                                  reasoning="No case_id in context.")

        fast = self.semantic.route(query)
        if fast.confidence >= CONFIDENCE_GATE:
            return fast
        try:
            return llm_route(query)
        except Exception:
            return fast


if __name__ == "__main__":
    router = QueryRouter()
    examples = [
        ("What is the punishment for hacking under IT Act?", None),
        ("What sections apply to the accused in this FIR?", "case-1"),
        ("Is there enough evidence for murder?", "case-1"),
        ("What is the punishment for hacking under IT Act?", "case-1"),
        ("Under what sections does the accused gets punished?", "case-1"),
        ("Under what sections does this case fall under?", "case-1"),
    ]
    for q, cid in examples:
        r = router.route(q, cid)
        print(f"[{r.route:9s} {r.confidence:.2f}] {q!r} (case_id={cid}) — {r.reasoning}")