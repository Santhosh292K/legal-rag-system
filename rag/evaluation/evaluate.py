"""
evaluation/evaluate.py
Full evaluation framework for the Legal RAG research paper.

Metrics (paper-grade):
  Retrieval   : Recall@K (K=1,3,5,10), Precision@K, Hit-Rate@K, MAP, MRR, NDCG@K
  Generation  : ROUGE-1, ROUGE-2, ROUGE-L, Token-F1, Exact Match, Semantic Similarity
  Citation    : Citation Precision, Citation Recall, Citation F1
  Legal       : Hallucination Rate, Answer Coverage Score
  System      : Latency P50/P95/P99, QPS, Confidence Calibration
  Breakdown   : Per-category (domain) metric table
  Ablation    : 7 component variants (A–G)

NOTE on evaluation methodology: this file evaluates against whatever
benchmark JSON path it is given. If you are also tuning hyperparameters
(e.g. evaluation/tune_pin_threshold.py) against the SAME file you report
final numbers on, that is train/test leakage — use
evaluation/split_benchmark.py to create a held-out dev/test split and
tune only against the dev file.
"""
import json
import math
import re
import sys
import time
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from main import LegalRAGPipeline
from evaluation.legal_metrics import answer_coverage_score, semantic_similarity, LegalMetrics
# NOTE: judge_faithfulness / context_from_citations are imported lazily
# inside evaluate() (see _load_faithfulness_judge below), not here.
# faithfulness_judge.py is only needed when --llm-judge is passed; a
# module-level import meant ANY run of evaluate.py — including the
# default path with no LLM-judge involved — hard-crashed if that file
# was ever missing, renamed, or briefly broken mid-edit. That already
# happened once (the file was absent from a shipped build); importing
# lazily contains the blast radius to the one flag that actually needs it.


# ═══════════════════════════════════════════════════════════════════════════════
# Metric data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalMetrics:
    recall_at_1:  float = 0.0
    recall_at_3:  float = 0.0
    recall_at_5:  float = 0.0
    recall_at_10: float = 0.0
    precision_at_1:  float = 0.0
    precision_at_5:  float = 0.0
    precision_at_10: float = 0.0
    hit_rate_at_1:  float = 0.0    # ≥1 gold in top-1
    hit_rate_at_3:  float = 0.0    # ≥1 gold in top-3
    hit_rate_at_5:  float = 0.0    # ≥1 gold in top-5
    map_score:    float = 0.0      # Mean Average Precision
    mrr:          float = 0.0      # Mean Reciprocal Rank
    ndcg_at_5:    float = 0.0
    ndcg_at_10:   float = 0.0


@dataclass
class GenerationMetrics:
    rouge_1:     float = 0.0
    rouge_2:     float = 0.0
    rouge_l:     float = 0.0
    token_f1:    float = 0.0
    exact_match: float = 0.0
    semantic_similarity: float = 0.0


@dataclass
class CitationMetrics:
    """Legal-specific: how accurately does the LLM cite retrieved sections."""
    citation_precision: float = 0.0   # cited ∩ gold / cited
    citation_recall:    float = 0.0   # cited ∩ gold / gold
    citation_f1:        float = 0.0
    hallucination_rate: float = 0.0   # cited sections NOT in gold AND NOT in retrieved


@dataclass
class SystemMetrics:
    latency_p50:  float = 0.0
    latency_p95:  float = 0.0
    latency_p99:  float = 0.0
    avg_latency:  float = 0.0
    qps:          float = 0.0


@dataclass
class ConfidenceCalibration:
    """At each confidence band, what fraction of answers have correct gold retrieval."""
    high_precision:   float = 0.0
    medium_precision: float = 0.0
    low_precision:    float = 0.0
    high_count:       int   = 0
    medium_count:     int   = 0
    low_count:        int   = 0


@dataclass
class EvalResult:
    system_name:  str
    retrieval:    RetrievalMetrics   = field(default_factory=RetrievalMetrics)
    generation:   GenerationMetrics  = field(default_factory=GenerationMetrics)
    citation:     CitationMetrics    = field(default_factory=CitationMetrics)
    legal:        LegalMetrics       = field(default_factory=LegalMetrics)
    system:       SystemMetrics      = field(default_factory=SystemMetrics)
    calibration:  ConfidenceCalibration = field(default_factory=ConfidenceCalibration)
    # domain → {metric_name: value}
    domain_breakdown: dict = field(default_factory=dict)
    n_queries:    int   = 0
    n_failed:     int   = 0
    # raw per-query scores, keyed by metric name → list[float], same order as
    # the benchmark items that succeeded. Needed for paired significance
    # testing between two systems (see evaluation/significance_test.py) —
    # the averages alone throw away everything needed for that.
    raw_scores:   dict  = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval metrics
# ═══════════════════════════════════════════════════════════════════════════════

def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold: return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not retrieved or k == 0: return 0.0
    top_k = retrieved[:k]
    return len(set(top_k) & set(gold)) / k


def hit_rate_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """1 if at least one gold section is in top-K, else 0."""
    if not gold: return 0.0
    return float(bool(set(retrieved[:k]) & set(gold)))


def average_precision(retrieved: list[str], gold: list[str]) -> float:
    """AP for a single query — area under the precision-recall curve."""
    gold_set = set(gold)
    if not gold_set: return 0.0
    hits = 0
    ap   = 0.0
    for rank, sid in enumerate(retrieved, start=1):
        if sid in gold_set:
            hits += 1
            ap   += hits / rank
    return ap / len(gold_set)


def mean_reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for rank, sid in enumerate(retrieved, start=1):
        if sid in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int = 10) -> float:
    gold_set = set(gold)
    dcg  = sum(1.0 / math.log2(i + 1)
                for i, sid in enumerate(retrieved[:k], start=1)
                if sid in gold_set)
    idcg = sum(1.0 / math.log2(i + 1)
               for i in range(1, min(len(gold_set), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Generation metrics (ROUGE-1, ROUGE-2, ROUGE-L, Token-F1, EM)
# ═══════════════════════════════════════════════════════════════════════════════

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, punctuation-stripped tokenization shared by all lexical
    metrics below. The previous version used raw `.split()`, so
    'IPC 302.' and 'IPC 302' or 'death,' and 'death' counted as different
    tokens — this was silently deflating ROUGE/F1/EM on well-punctuated
    generations. Still not a linguistic tokenizer (no stemming), but this
    removes the punctuation-sensitivity bug without adding a dependency."""
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> dict:
    counts: dict = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i+n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def rouge_n(prediction: str, gold: str, n: int) -> float:
    pred_toks = _tokenize(prediction)
    gold_toks = _tokenize(gold)
    pred_grams = _ngrams(pred_toks, n)
    gold_grams = _ngrams(gold_toks, n)
    overlap    = sum(min(pred_grams.get(g, 0), gold_grams[g]) for g in gold_grams)
    denom_p    = sum(pred_grams.values())
    denom_g    = sum(gold_grams.values())
    if denom_p == 0 or denom_g == 0: return 0.0
    prec = overlap / denom_p
    rec  = overlap / denom_g
    if prec + rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec)


def rouge_l(prediction: str, gold: str) -> float:
    pred_toks = _tokenize(prediction)
    gold_toks = _tokenize(gold)
    m, n = len(pred_toks), len(gold_toks)
    if m == 0 or n == 0: return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_toks[i-1] == gold_toks[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / m
    rec  = lcs / n
    if prec + rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec)


def token_f1(prediction: str, gold: str) -> float:
    pred_toks = set(_tokenize(prediction))
    gold_toks = set(_tokenize(gold))
    if not pred_toks or not gold_toks: return 0.0
    common    = pred_toks & gold_toks
    prec = len(common) / len(pred_toks)
    rec  = len(common) / len(gold_toks)
    if prec + rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec)


def exact_match(prediction: str, gold: str) -> float:
    return float(prediction.strip().lower() == gold.strip().lower())


# ═══════════════════════════════════════════════════════════════════════════════
# Citation metrics (legal-specific)
# ═══════════════════════════════════════════════════════════════════════════════

def citation_metrics(
    llm_cited: list[str],
    retrieved:  list[str],
    gold:       list[str],
) -> tuple[float, float, float, float]:
    """
    Returns (citation_precision, citation_recall, citation_f1, hallucination_rate).

    citation_precision  = |cited ∩ gold| / |cited|
    citation_recall     = |cited ∩ gold| / |gold|
    hallucination_rate  = |cited - gold - retrieved| / |cited|
                          (cited something that was NEITHER in gold NOR retrieved)
    """
    cited_set     = set(llm_cited)
    gold_set      = set(gold)
    retrieved_set = set(retrieved)

    if not cited_set:
        return 0.0, 0.0, 0.0, 0.0

    intersection = cited_set & gold_set
    prec = len(intersection) / len(cited_set)
    rec  = len(intersection) / len(gold_set) if gold_set else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    hall = len(cited_set - gold_set - retrieved_set) / len(cited_set)
    return prec, rec, f1, hall


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluator
# ═══════════════════════════════════════════════════════════════════════════════

class LegalRAGEvaluator:
    """
    Benchmark format (benchmark_scenarios.json):
    [
      {
        "query":         "...",
        "gold_sections": ["IPC_302", ...],
        "gold_answer":   "...",
        "category":      "criminal"           # optional, for domain breakdown
      }, ...
    ]
    """

    def __init__(self, pipeline: LegalRAGPipeline):
        self.pipeline = pipeline
        # Reuse the pipeline's already-loaded bge-large model for semantic
        # similarity instead of loading a second copy — see legal_metrics.py.
        try:
            embed_model = self.pipeline.retriever.embed_model
            self._embed_fn = lambda texts: embed_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False)
        except AttributeError:
            self._embed_fn = None
        # Lazily populated by _load_faithfulness_judge() the first time
        # evaluate() is actually called with use_llm_judge=True.
        self._judge_faithfulness = None
        self._context_from_citations = None

    @staticmethod
    def _load_faithfulness_judge():
        """Imports evaluation/faithfulness_judge.py on first use only.
        Raises a clear, actionable error (instead of the raw
        ModuleNotFoundError a top-level import would give at CLI startup)
        if the module is missing or fails to import — and does so without
        affecting any evaluate() call that doesn't pass --llm-judge."""
        try:
            from evaluation.faithfulness_judge import judge_faithfulness, context_from_citations
            return judge_faithfulness, context_from_citations
        except ImportError as e:
            raise ImportError(
                "--llm-judge requires evaluation/faithfulness_judge.py, which "
                f"failed to import ({e}). Either add/fix that module, or drop "
                "--llm-judge to run the rest of the evaluation without "
                "faithfulness scoring."
            ) from e

    def evaluate(
        self,
        benchmark_path: str,
        system_name:    str       = "LegalRAG-Full",
        max_queries:    int | None = None,
        use_llm_judge:  bool      = False,
    ) -> EvalResult:

        with open(benchmark_path) as f:
            benchmark = json.load(f)
        if max_queries:
            benchmark = benchmark[:max_queries]

        result = EvalResult(system_name=system_name, n_queries=len(benchmark))

        # Retrieval
        r1_l,r3_l,r5_l,r10_l = [],[],[],[]
        p1_l,p5_l,p10_l      = [],[],[]
        hr1_l,hr3_l,hr5_l    = [],[],[]
        ap_l,mrr_l            = [],[]
        ndcg5_l,ndcg10_l     = [],[]
        # Generation
        rou1_l,rou2_l,rl_l,tf1_l,em_l = [],[],[],[],[]
        # Citation
        cp_l,cr_l,cf1_l,hall_l = [],[],[],[]
        # Legal (answer coverage / semantic similarity)
        cov_l, sem_l = [], []
        faith_l, contra_l, n_claims_l = [], [], []
        # System
        latency_l = []
        # Confidence calibration
        conf_buckets: dict[str, list[float]] = {"high": [], "medium": [], "low": []}
        # Domain tracking
        domain_collectors: dict[str, dict] = defaultdict(lambda: {
            "r5":[], "r10":[], "ndcg10":[], "mrr":[], "rouge_l":[], "f1":[]
        })

        if use_llm_judge and self._judge_faithfulness is None:
            self._judge_faithfulness, self._context_from_citations = self._load_faithfulness_judge()

        for item in tqdm(benchmark, desc=f"[{system_name}]"):
            query       = item["query"]
            gold_secs   = item.get("gold_sections", [])
            gold_ans    = item.get("gold_answer", "")
            category    = item.get("category", "unknown")

            try:
                t0     = time.time()
                answer = self.pipeline.query(query)
                elapsed = time.time() - t0
                latency_l.append(elapsed)

                # Retrieve section list from reranker (not LLM citations)
                retrieved_ids = (
                    answer.retrieved_section_ids
                    if answer.retrieved_section_ids
                    else [c.section_id for c in answer.citations]
                )
                llm_cited_ids = [c.section_id for c in answer.citations]

                # ── Retrieval metrics ─────────────────────────────────────────
                if gold_secs:
                    r1_l.append(recall_at_k(retrieved_ids, gold_secs, 1))
                    r3_l.append(recall_at_k(retrieved_ids, gold_secs, 3))
                    r5_l.append(recall_at_k(retrieved_ids, gold_secs, 5))
                    r10_l.append(recall_at_k(retrieved_ids, gold_secs, 10))
                    p1_l.append(precision_at_k(retrieved_ids, gold_secs, 1))
                    p5_l.append(precision_at_k(retrieved_ids, gold_secs, 5))
                    p10_l.append(precision_at_k(retrieved_ids, gold_secs, 10))
                    hr1_l.append(hit_rate_at_k(retrieved_ids, gold_secs, 1))
                    hr3_l.append(hit_rate_at_k(retrieved_ids, gold_secs, 3))
                    hr5_l.append(hit_rate_at_k(retrieved_ids, gold_secs, 5))
                    ap_l.append(average_precision(retrieved_ids, gold_secs))
                    mrr_l.append(mean_reciprocal_rank(retrieved_ids, gold_secs))
                    ndcg5_l.append(ndcg_at_k(retrieved_ids, gold_secs, 5))
                    ndcg10_l.append(ndcg_at_k(retrieved_ids, gold_secs, 10))

                    # Domain breakdown (retrieval)
                    domain_collectors[category]["r5"].append(recall_at_k(retrieved_ids, gold_secs, 5))
                    domain_collectors[category]["r10"].append(recall_at_k(retrieved_ids, gold_secs, 10))
                    domain_collectors[category]["ndcg10"].append(ndcg_at_k(retrieved_ids, gold_secs, 10))
                    domain_collectors[category]["mrr"].append(mean_reciprocal_rank(retrieved_ids, gold_secs))

                # ── Generation metrics ────────────────────────────────────────
                if gold_ans:
                    rou1_l.append(rouge_n(answer.answer, gold_ans, 1))
                    rou2_l.append(rouge_n(answer.answer, gold_ans, 2))
                    rl_l.append(rouge_l(answer.answer, gold_ans))
                    tf1_l.append(token_f1(answer.answer, gold_ans))
                    em_l.append(exact_match(answer.answer, gold_ans))
                    sem_l.append(semantic_similarity(answer.answer, gold_ans, self._embed_fn))
                    # Domain breakdown (generation)
                    domain_collectors[category]["rouge_l"].append(rouge_l(answer.answer, gold_ans))
                    domain_collectors[category]["f1"].append(token_f1(answer.answer, gold_ans))

                # ── Citation metrics ──────────────────────────────────────────
                if gold_secs:
                    cp, cr, cf1, hall = citation_metrics(llm_cited_ids, retrieved_ids, gold_secs)
                    cp_l.append(cp); cr_l.append(cr)
                    cf1_l.append(cf1); hall_l.append(hall)

                # ── Legal: answer coverage (does the PROSE discuss each gold
                # section, independent of the parsed citation list) ─────────
                if gold_secs:
                    cov_l.append(answer_coverage_score(answer.answer, gold_secs))

                # ── Legal: LLM-judge faithfulness (does the PROSE's claims
                # actually follow from the retrieved section content?) ──────
                # Opt-in: one extra LLM call per query. Judges against
                # answer.citations content — NOT gold_ans — see
                # faithfulness_judge.py docstring for why.
                if use_llm_judge and answer.citations:
                    context = self._context_from_citations(answer.citations)
                    fr = self._judge_faithfulness(answer.answer, context)
                    if not fr.judge_failed:
                        faith_l.append(fr.faithfulness_score)
                        contra_l.append(fr.contradiction_rate)
                        n_claims_l.append(fr.n_claims)

                # ── Confidence calibration ────────────────────────────────────
                # A confidence band is "correct" if it has ≥1 gold section retrieved
                is_correct = float(bool(set(retrieved_ids[:5]) & set(gold_secs))) if gold_secs else 0.0
                band = getattr(answer, "confidence", "low")
                if band in conf_buckets:
                    conf_buckets[band].append(is_correct)

            except Exception as e:
                result.n_failed += 1
                print(f"  ✗ Failed: {query[:55]}... | {e}")

        # ── Aggregate ─────────────────────────────────────────────────────────
        def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0
        def pct(lst, p): return round(statistics.quantiles(lst, n=100)[p-1], 3) if len(lst) >= 2 else 0.0

        result.retrieval = RetrievalMetrics(
            recall_at_1    = avg(r1_l),
            recall_at_3    = avg(r3_l),
            recall_at_5    = avg(r5_l),
            recall_at_10   = avg(r10_l),
            precision_at_1 = avg(p1_l),
            precision_at_5 = avg(p5_l),
            precision_at_10= avg(p10_l),
            hit_rate_at_1  = avg(hr1_l),
            hit_rate_at_3  = avg(hr3_l),
            hit_rate_at_5  = avg(hr5_l),
            map_score      = avg(ap_l),
            mrr            = avg(mrr_l),
            ndcg_at_5      = avg(ndcg5_l),
            ndcg_at_10     = avg(ndcg10_l),
        )
        result.generation = GenerationMetrics(
            rouge_1     = avg(rou1_l),
            rouge_2     = avg(rou2_l),
            rouge_l     = avg(rl_l),
            token_f1    = avg(tf1_l),
            exact_match = avg(em_l),
            semantic_similarity = avg(sem_l),
        )
        result.citation = CitationMetrics(
            citation_precision = avg(cp_l),
            citation_recall    = avg(cr_l),
            citation_f1        = avg(cf1_l),
            hallucination_rate = avg(hall_l),
        )
        result.legal = LegalMetrics(
            answer_coverage      = avg(cov_l),
            semantic_similarity  = avg(sem_l),
            semantic_similarity_is_fallback = self._embed_fn is None,
            faithfulness_score    = avg(faith_l),
            contradiction_rate    = avg(contra_l),
            faithfulness_n_claims = avg(n_claims_l),
            faithfulness_judged   = bool(faith_l),
        )

        lat_sorted = sorted(latency_l)
        total_time = sum(latency_l) if latency_l else 1.0
        result.system = SystemMetrics(
            latency_p50 = pct(lat_sorted, 50),
            latency_p95 = pct(lat_sorted, 95),
            latency_p99 = pct(lat_sorted, 99),
            avg_latency = avg(latency_l),
            qps         = round(len(latency_l) / total_time, 2),
        )

        result.calibration = ConfidenceCalibration(
            high_precision   = avg(conf_buckets["high"]),
            medium_precision = avg(conf_buckets["medium"]),
            low_precision    = avg(conf_buckets["low"]),
            high_count       = len(conf_buckets["high"]),
            medium_count     = len(conf_buckets["medium"]),
            low_count        = len(conf_buckets["low"]),
        )

        # Domain breakdown aggregate
        result.domain_breakdown = {
            cat: {k: avg(v) for k, v in metrics.items()}
            for cat, metrics in domain_collectors.items()
        }

        # Raw per-query scores — for paired significance testing between
        # two systems' saved results (evaluation/significance_test.py).
        result.raw_scores = {
            "recall_at_5":  r5_l,
            "recall_at_10": r10_l,
            "ndcg_at_10":   ndcg10_l,
            "mrr":          mrr_l,
            "rouge_l":      rl_l,
            "token_f1":     tf1_l,
            "semantic_similarity": sem_l,
            "answer_coverage": cov_l,
            "citation_f1":  cf1_l,
            # Caveat for significance_test.py: faithfulness_score is only
            # appended when use_llm_judge=True AND the answer had citations
            # AND the judge call succeeded — unlike the other raw_scores
            # lists, its length/order can differ between two systems (e.g.
            # if one system fails to cite more often). Don't paired-bootstrap
            # this one across systems unless you've confirmed equal length.
            "faithfulness_score": faith_l,
        }

        return result

    # ── Print helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def print_results(results: list[EvalResult]):
        if not results: return

        # ── Table 1: Retrieval ─────────────────────────────────────────────────
        print("\n" + "═"*110)
        print("TABLE 1 — Retrieval Metrics")
        print("─"*110)
        hdr = (f"{'System':<28} {'R@1':>5} {'R@3':>5} {'R@5':>5} {'R@10':>6} "
               f"{'P@5':>5} {'P@10':>6} {'HR@5':>6} "
               f"{'MAP':>6} {'MRR':>6} {'NDCG@5':>7} {'NDCG@10':>8}")
        print(hdr)
        print("─"*110)
        for r in results:
            rt = r.retrieval
            print(f"{r.system_name:<28} "
                  f"{rt.recall_at_1:>5.3f} {rt.recall_at_3:>5.3f} "
                  f"{rt.recall_at_5:>5.3f} {rt.recall_at_10:>6.3f} "
                  f"{rt.precision_at_5:>5.3f} {rt.precision_at_10:>6.3f} "
                  f"{rt.hit_rate_at_5:>6.3f} "
                  f"{rt.map_score:>6.3f} {rt.mrr:>6.3f} "
                  f"{rt.ndcg_at_5:>7.3f} {rt.ndcg_at_10:>8.3f}")
        print("═"*110)

        # ── Table 2: Generation ────────────────────────────────────────────────
        print("\nTABLE 2 — Generation Metrics")
        print("─"*75)
        print(f"{'System':<28} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'Token-F1':>9} {'EM':>5}")
        print("─"*75)
        for r in results:
            gm = r.generation
            print(f"{r.system_name:<28} "
                  f"{gm.rouge_1:>8.3f} {gm.rouge_2:>8.3f} "
                  f"{gm.rouge_l:>8.3f} {gm.token_f1:>9.3f} {gm.exact_match:>5.3f}")
        print("═"*75)

        # ── Table 3: Citation metrics ──────────────────────────────────────────
        print("\nTABLE 3 — Citation Metrics (Legal-Specific)")
        print("─"*70)
        print(f"{'System':<28} {'Cite-P':>7} {'Cite-R':>7} {'Cite-F1':>8} {'Halluc %':>9}")
        print("─"*70)
        for r in results:
            cm = r.citation
            print(f"{r.system_name:<28} "
                  f"{cm.citation_precision:>7.3f} {cm.citation_recall:>7.3f} "
                  f"{cm.citation_f1:>8.3f} {cm.hallucination_rate*100:>8.1f}%")
        print("═"*70)

        # ── Table 3b: Legal metrics (answer coverage / semantic similarity /
        # LLM-judge faithfulness) ──────────────────────────────────────────
        print("\nTABLE 3b — Legal Metrics")
        print("─"*90)
        fallback_note = "*"
        print(f"{'System':<28} {'Ans-Coverage':>13} {'Semantic-Sim':>13} "
              f"{'Faithfulness':>13} {'Contra %':>9}")
        print("─"*90)
        for r in results:
            lm = r.legal
            mark = fallback_note if lm.semantic_similarity_is_fallback else ""
            faith_str = f"{lm.faithfulness_score:>13.3f}" if lm.faithfulness_judged else f"{'n/a':>13}"
            contra_str = f"{lm.contradiction_rate*100:>8.1f}%" if lm.faithfulness_judged else f"{'n/a':>9}"
            print(f"{r.system_name:<28} {lm.answer_coverage:>13.3f} "
                  f"{lm.semantic_similarity:>12.3f}{mark} {faith_str} {contra_str}")
        if any(r.legal.semantic_similarity_is_fallback for r in results):
            print("* = no embed_fn available; semantic-sim used bag-of-words "
                  "cosine fallback, not real embeddings — treat as indicative only.")
        if not any(r.legal.faithfulness_judged for r in results):
            print("  Faithfulness = n/a: run with --llm-judge to enable "
                  "(costs one extra LLM call per query, see faithfulness_judge.py)")
        print("═"*90)

        # ── Table 4: System metrics ────────────────────────────────────────────
        print("\nTABLE 4 — System / Latency Metrics")
        print("─"*70)
        print(f"{'System':<28} {'P50(s)':>7} {'P95(s)':>7} {'P99(s)':>7} {'Avg(s)':>7} {'QPS':>6}")
        print("─"*70)
        for r in results:
            sm = r.system
            print(f"{r.system_name:<28} "
                  f"{sm.latency_p50:>7.2f} {sm.latency_p95:>7.2f} "
                  f"{sm.latency_p99:>7.2f} {sm.avg_latency:>7.2f} {sm.qps:>6.2f}")
        print("═"*70)

        # ── Table 5: Confidence calibration ───────────────────────────────────
        print("\nTABLE 5 — Confidence Calibration (Precision@5 within band)")
        print("─"*60)
        print(f"{'System':<28} {'High (n)':>10} {'Medium (n)':>12} {'Low (n)':>9}")
        print("─"*60)
        for r in results:
            cc = r.calibration
            print(f"{r.system_name:<28} "
                  f"{cc.high_precision:.3f} ({cc.high_count:>3}) "
                  f"{cc.medium_precision:.3f} ({cc.medium_count:>3}) "
                  f"{cc.low_precision:.3f} ({cc.low_count:>3})")
        print("═"*60)

        # ── Table 6: Domain breakdown (first result only) ─────────────────────
        if results and results[0].domain_breakdown:
            full = results[0]
            print(f"\nTABLE 6 — Domain Breakdown ({full.system_name})")
            print("─"*70)
            print(f"{'Category':<22} {'R@5':>5} {'R@10':>6} {'NDCG@10':>8} {'MRR':>6} {'ROUGE-L':>8} {'F1':>6}")
            print("─"*70)
            for cat, m in sorted(full.domain_breakdown.items()):
                print(f"{cat:<22} "
                      f"{m.get('r5', 0):>5.3f} {m.get('r10', 0):>6.3f} "
                      f"{m.get('ndcg10', 0):>8.3f} {m.get('mrr', 0):>6.3f} "
                      f"{m.get('rouge_l', 0):>8.3f} {m.get('f1', 0):>6.3f}")
            print("═"*70)

    def ablation_study(
        self, benchmark_path: str, use_llm_judge: bool = False, qdrant_client=None,
    ) -> list[EvalResult]:
        """
        7 ablation variants (A–G) toggling individual pipeline components.

          A: Full pipeline (all components ON)
          B: No IRAC reranker
          C: No hierarchy enrichment
          D: No temporal filter
          E: No section pinner
          F: Baseline — dense-only, no reranking, no hierarchy, no pinner

        BUGFIX: this used to call `LegalRAGPipeline(verbose=False, **flags)`
        fresh inside the loop for every variant, with no `qdrant_client`
        passed in. Qdrant's embedded/local mode file-locks ./qdrant_db to a
        single open client — HybridRetriever already has a `client` param
        specifically to allow callers to share one connection (see its
        docstring), but this loop never used it, so each variant tried to
        open a second, independent lock on the same storage folder and
        crashed with "already accessed by another instance" — including
        variant A, since the *outer* `evaluate.py` __main__ already opens
        one full LegalRAGPipeline (and its own QdrantClient) before this
        method is ever called. Every downstream step that depended on
        results_ablation.json (all `significance_test.py` comparisons)
        then failed too, since the file never got real per-variant data.
        Fix: open exactly one QdrantClient up front and hand it to every
        variant's LegalRAGPipeline, matching the sharing pattern the
        codebase already uses internally (retriever → pinner → structurer
        → KG all share `self.retriever.client`). If the caller already has
        an open client (e.g. __main__ below, which needs one anyway to
        build self.pipeline before calling this), pass it in via
        `qdrant_client` so we reuse it instead of opening a second,
        conflicting one — a fresh client is only opened here as a fallback
        for callers (tests, notebooks) that invoke this method standalone.
        """
        from qdrant_client import QdrantClient
        from config import QDRANT_PATH

        print("\nRunning ablation study — 7 variants × N queries...")
        ablation_configs = [
            ("A: Full Pipeline",         dict(use_irac=True,  use_hierarchy=True,  use_temporal=True,  use_pinner=True)),
            ("B: No IRAC Reranker",      dict(use_irac=False, use_hierarchy=True,  use_temporal=True,  use_pinner=True)),
            ("C: No Hierarchy Chunking", dict(use_irac=True,  use_hierarchy=False, use_temporal=True,  use_pinner=True)),
            ("D: No Temporal Filter",    dict(use_irac=True,  use_hierarchy=True,  use_temporal=False, use_pinner=True)),
            ("E: No Section Pinner",     dict(use_irac=True,  use_hierarchy=True,  use_temporal=True,  use_pinner=False)),
            ("F: Baseline Naive RAG",    dict(use_irac=False, use_hierarchy=False, use_temporal=False, use_pinner=False, use_kg=False)),
            ("G: No Knowledge Graph",    dict(use_irac=True,  use_hierarchy=True,  use_temporal=True,  use_pinner=True,  use_kg=False)),
        ]

        owns_client   = qdrant_client is None
        shared_client = qdrant_client or QdrantClient(path=QDRANT_PATH)

        # BUGFIX: sharing the Qdrant client (above) wasn't enough on its
        # own — HybridRetriever also loads its own SentenceTransformer, and
        # every one of these 7 LegalRAGPipeline builds used to load a brand
        # new copy of bge-large on top of self.pipeline's already-loaded
        # copy. Two+ full copies of the embedding model don't fit in 8GB of
        # VRAM, so every variant (including A, no different from the
        # baseline) CUDA OOM'd at a trivial 20MiB allocation. Reuse
        # self.pipeline's already-loaded model for every variant instead.
        shared_embed_model = self.pipeline.retriever.embed_model
        # BUGFIX: embed_model alone wasn't the whole story — IRACReranker
        # loads its own bge-reranker-large CrossEncoder independently, and
        # 5 of these 7 variants (A, C, D, E, G) have use_irac=True, so each
        # was still loading a second/third/... copy of that model too.
        # self.pipeline was built with the default use_irac=True, so its
        # reranker (if the cross-encoder loaded successfully) is available
        # to reuse; None otherwise, and each variant falls back to loading
        # its own — no worse than before, just no longer the default path.
        shared_cross_encoder = (
            self.pipeline.reranker.cross_encoder
            if self.pipeline.reranker is not None else None
        )

        results = []
        try:
            for name, flags in ablation_configs:
                print(f"\n  ▶ {name}")
                try:
                    pipe = LegalRAGPipeline(verbose=False, qdrant_client=shared_client,
                                             embed_model=shared_embed_model,
                                             cross_encoder=shared_cross_encoder, **flags)
                    evl  = LegalRAGEvaluator(pipe)
                    results.append(evl.evaluate(benchmark_path, system_name=name, use_llm_judge=use_llm_judge))
                except Exception as e:
                    print(f"  ✗ Variant failed: {e}")
        finally:
            if owns_client:
                shared_client.close()

        # BUGFIX: this loop used to swallow per-variant exceptions and just
        # keep going, so a total ablation failure (all 7 variants OOM'ing,
        # as above) still returned `results = []`, got saved to
        # results_ablation.json as an empty list, and the *script* still
        # exited 0 — which made run_all_evaluations.py report this step as
        # PASS despite there being no data in it at all. Every downstream
        # significance_test.py call then failed with "system not found",
        # which looked like a significance_test.py bug but was actually
        # this step lying about having succeeded. Fail loudly instead.
        if len(results) < len(ablation_configs):
            print(f"\n[evaluate] ERROR: only {len(results)}/{len(ablation_configs)} "
                  f"ablation variants produced results — refusing to report success.")
            sys.exit(1)

        return results

    def save_results_json(self, results: list[EvalResult], output_path: str):
        """Save all results to JSON for LaTeX table generation."""
        import dataclasses
        data = [dataclasses.asdict(r) for r in results]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[evaluate] Results saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _default_benchmark_path() -> str:
    """Prefers the held-out test split (see split_benchmark.py) so final
    reported numbers are, by default, not the same file
    tune_pin_threshold.py tuned MIN_PIN_SIMILARITY against. Falls back to
    the full benchmark_scenarios.json with a warning if no split has been
    generated yet — old behavior, just no longer silent about the tradeoff."""
    test_path = Path("./evaluation/benchmark_scenarios_test.json")
    if test_path.exists():
        return str(test_path)
    print("[evaluate] WARNING: no benchmark_scenarios_test.json found — run "
          "`python3 evaluation/split_benchmark.py` first for a leakage-free "
          "held-out test set. Falling back to the full benchmark_scenarios.json.")
    return "./evaluation/benchmark_scenarios.json"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Legal RAG Evaluation Framework")
    parser.add_argument("benchmark", nargs="?",
                        default=None,
                        help="Path to benchmark JSON file (default: held-out "
                             "benchmark_scenarios_test.json if it exists)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study (7 variants)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to this JSON file path")
    parser.add_argument("--max", type=int, default=None,
                        help="Limit evaluation to first N queries")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Enable LLM-judge faithfulness scoring (one extra "
                             "local ollama call per query — see faithfulness_judge.py)")
    args = parser.parse_args()
    args.benchmark = args.benchmark or _default_benchmark_path()

    # BUGFIX: previously this always built `pipeline` here, then — when
    # --ablation was passed — ablation_study() built 7 MORE pipelines on
    # top of it, each opening its own QdrantClient against the same
    # file-locked ./qdrant_db folder. That crashed every single variant
    # (including the first) with "already accessed by another instance",
    # which then cascaded into every significance_test.py comparison
    # failing since results_ablation.json never had real data. Open one
    # QdrantClient for the whole run and thread it through everything.
    from qdrant_client import QdrantClient
    from config import QDRANT_PATH

    shared_client = QdrantClient(path=QDRANT_PATH)
    try:
        pipeline  = LegalRAGPipeline(verbose=False, qdrant_client=shared_client)
        evaluator = LegalRAGEvaluator(pipeline)

        if args.ablation:
            results = evaluator.ablation_study(
                args.benchmark, use_llm_judge=args.llm_judge, qdrant_client=shared_client,
            )
        else:
            result  = evaluator.evaluate(args.benchmark,
                                         system_name="LegalRAG-Full",
                                         max_queries=args.max,
                                         use_llm_judge=args.llm_judge)
            results = [result]

        evaluator.print_results(results)
    finally:
        shared_client.close()

    if args.save:
        evaluator.save_results_json(results, args.save)