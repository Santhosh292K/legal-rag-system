"""
pipeline/domain_router.py
Universal domain router — maps ANY legal query to relevant act clusters.

CHANGE FROM PREVIOUS VERSION:
The old router activated domains purely by counting regex hits from a large
per-domain signal list (9-19 patterns each). Adding a new legal domain (tax,
labor, consumer protection, a second jurisdiction...) meant hand-writing a
whole new regex block and hand-tuning its `priority` against every existing
domain's priority so it wouldn't get starved or dominate — a process with
no LLM/embedding escape hatch at all, unlike the other three components.

This version scores domains primarily via SemanticMatcher over a SMALL set
of canonical example queries per domain (5-8 examples, not 9-19 regex
patterns). Embedding similarity generalises to phrasings the example list
never explicitly enumerated. Adding a new domain going forward is a data
change (a short `examples` list, optionally in DOMAIN_EXAMPLES_PATH), not a
regex-authoring exercise.

The ORIGINAL regex `signals` lists are kept as a safety-net fallback tier —
used automatically whenever no embed_fn is supplied (so this file still
works unmodified as a drop-in replacement) and blended in as a secondary
signal when it is. This means existing behaviour is preserved by default;
the embedding tier is what you get once you wire in an embed_fn (see the
note in __main__ and the class docstring below).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from pipeline.semantic_matcher import SemanticMatcher


# ── All acts in your Qdrant index ─────────────────────────────────────────────

ALL_ACTS = {
    "IPC", "BNS",           # Indian Penal Code + Bharatiya Nyaya Sanhita (replacement)
    "CRPC", "BNSS",         # Criminal Procedure Code + Bharatiya Nagarik Suraksha Sanhita
    "IEA", "BSA",           # Indian Evidence Act + Bharatiya Sakshya Adhiniyam
    "ICA",                  # Indian Contract Act
    "CPC",                  # Code of Civil Procedure
    "SRA",                  # Specific Relief Act
    "TPA",                  # Transfer of Property Act
    "COI",                  # Constitution of India
    "ITA",                  # Information Technology Act
    "NDPS",                 # Narcotic Drugs and Psychotropic Substances Act
    "PCA",                  # Prevention of Corruption Act
    "POCSO",                # Protection of Children from Sexual Offences
    "SCST",                 # SC/ST Prevention of Atrocities Act
    "UAPA",                 # Unlawful Activities Prevention Act
    "LA",                   # Limitation Act
}

# Acts NOT in your dataset — surface clear warnings
MISSING_ACTS = {
    "NI":    "Negotiable Instruments Act (cheque bounce / Section 138)",
    "MCA":   "Companies Act / MCA (corporate law)",
    "IT":    "Income Tax Act",
    "GST":   "GST Act",
    "RTI":   "Right to Information Act",
    "CPA":   "Consumer Protection Act",
    "MVA":   "Motor Vehicles Act",
    "RERA":  "Real Estate Regulation Act",
    "EPF":   "Employees Provident Fund Act",
    "ID":    "Industrial Disputes Act",
    # Gap: queries about child labour were silently answered from adjacent
    # child-protection sections (kidnapping-for-begging, prostitution,
    # abandonment) with no signal to the user that the actual governing
    # statute for child *employment* isn't in this corpus at all. Surfacing
    # it here means the "⚠ NOT IN DATASET" warning fires like it already
    # does for NI/MCA/GST, instead of the gap being invisible.
    "CLPRA": "Child Labour (Prohibition and Regulation) Act, 1986",
    "FA":    "Factories Act, 1948 (working conditions, hazardous processes)",
}


# ── Domain definitions ────────────────────────────────────────────────────────

@dataclass
class Domain:
    name:     str
    acts:     list[str]
    examples: list[str] = field(default_factory=list)   # NEW: canonical queries (embedding tier)
    signals:  list[str] = field(default_factory=list)   # legacy regex (fallback tier)
    priority: int = 5


DOMAINS = [
    Domain(
        name="criminal",
        acts=["IPC", "BNS", "NDPS", "POCSO", "UAPA", "PCA", "SCST"],
        priority=9,
        examples=[
            "someone was murdered", "a robbery took place", "he was assaulted",
            "is this a crime", "what punishment applies for this offence",
            "a person was cheated of money", "bribery by a public official",
            "drugs were found on someone", "a minor was sexually abused",
            "caste-based discrimination or violence",
            # Gap fix: nothing in this domain's examples previously described
            # child LABOUR/employment/trafficking-for-work — only sexual
            # abuse, so a "child labour" query's nearest semantic neighbours
            # in this list were abuse-flavoured, pulling retrieval toward
            # kidnapping/prostitution/abandonment sections instead of the
            # trafficking-for-labour and forced-labour ones that actually
            # answer it.
            "a child was made to work in a factory or mine",
            "a trafficked child was employed for labour",
            "someone was forced to work against their will",
            "a child was used for begging",
        ],
        signals=[
            r"\b(murder|kill\w*|homicide|culpable)\b",
            r"\b(theft|rob\w*|dacoity|extortion|kidnap\w*)\b",
            r"\b(rape|sexual\s+assault|outrag\w*\s+modesty)\b",
            r"\b(assault|hurt|grievous|injur\w*)\b",
            r"\b(cheat\w*|fraud\w*|forge\w*|personat\w*|impersonat\w*)\b",
            r"\b(bribe\w*|corrupt\w*|gratif\w*)\b",
            r"\b(drug\w*|narcotic\w*|ndps|ganja|cocaine|heroin)\b",
            r"\b(terror\w*|uapa|unlawful\s+activity)\b",
            r"\b(minor|child)\s+(abuse|assault|molest\w*|sexu\w+)\b",
            r"\b(caste|atrocity|dalit|scheduled\s+(caste|tribe))\b",
            r"\b(child\s+labou?r|bonded\s+labou?r|forced\s+labou?r|"
            r"traffick\w*.{0,25}(child|labou?r)|employ\w*.{0,25}traffick\w*)\b",
            r"\b(crime|criminal|offence|offender|accused|punish\w*)\b",
            r"\b(section\s+\d+\s+ipc|ipc\s+\d+|ipc\s+section)\b",
            r"\bipc\b",
            r"\b(negligent\w*|rash\s+act|304a)\b",
            r"\b(died|death|killed|dead)\b",
            r"\b(fake|bogus|forged?)\s+(doctor|degree|certificate|professional)\b",
        ],
    ),
    Domain(
        name="procedure_criminal",
        acts=["CRPC", "BNSS", "IEA", "BSA"],
        priority=7,
        examples=[
            "how do I file an FIR", "applying for bail", "the police arrested someone",
            "is this evidence admissible in court", "steps in a criminal trial",
            "the investigation procedure by police",
        ],
        signals=[
            r"\b(fir|first\s+information\s+report)\b",
            r"\b(arrest\w*|remand|custody)\b",
            r"\b(bail|anticipatory\s+bail|surety)\b",
            r"\b(charge\s*sheet|chargesheet)\b",
            r"\b(trial|sessions\s+court|magistrate)\b",
            r"\b(evidence|witness|confession|statement|admissib\w*)\b",
            r"\b(crpc|bnss|criminal\s+procedure)\b",
            r"\b(investigation|cognizable|non.?cognizable)\b",
            r"\b(habeas\s+corpus|writ)\b",
            r"\b(police\s+(officer|station|complain\w*))\b",
        ],
    ),
    Domain(
        name="civil_contract",
        acts=["ICA", "SRA", "CPC", "LA"],
        priority=7,
        examples=[
            "a contract was breached", "can I cancel this agreement",
            "suing someone for damages", "specific performance of a contract",
            "was I misled into signing an agreement",
            "time limit to file a civil case",
        ],
        signals=[
            r"\b(contract|agreement|breach|performance)\b",
            r"\b(consideration|offer|acceptance|void\w*)\b",
            r"\b(damages|compensation|remedy|enforce\w*)\b",
            r"\b(specific\s+performance|injunction)\b",
            r"\b(coercion|misrepresent\w*|undue\s+influence|fraud\w*\s+contract)\b",
            r"\b(ica|indian\s+contract\s+act)\b",
            r"\b(sue|civil\s+suit|civil\s+action|file\s+a\s+case)\b",
            r"\b(refund|money\s+back|recover\w*\s+money)\b",
            r"\b(limitation|time\s+limit\s+to\s+sue)\b",
        ],
    ),
    Domain(
        name="civil_property",
        acts=["TPA", "SRA", "CPC", "IPC"],
        priority=7,
        examples=[
            "a property or land dispute", "my landlord is evicting me illegally",
            "who inherits this property", "someone trespassed on my land",
            "a sale deed for a house",
        ],
        signals=[
            r"\b(property|land|house|flat|plot|immovable)\b",
            r"\b(sale\s+deed|gift\s+deed|mortgage|lease|rent\w*)\b",
            r"\b(evict\w*|dispossess\w*|trespass\w*)\b",
            r"\b(partition|inherit\w*|succession|last\s+will|will\s+and\s+testament|testament)\b",
            r"\b(tpa|transfer\s+of\s+property)\b",
            r"\b(encroach\w*|occupy|possession)\b",
            r"\b(registr\w*|stamp\s+duty)\b",
            r"\b(landlord|tenant|lessor|lessee)\b",
            r"\b(benami|fraudulent\s+transfer)\b",
        ],
    ),
    Domain(
        name="constitutional",
        acts=["COI", "IPC", "CRPC", "BNSS"],
        priority=8,
        examples=[
            "is this a violation of my fundamental rights",
            "can the government do this under the constitution",
            "filing a writ petition", "freedom of speech question",
            "right to privacy issue",
            # Gap fix: Article 24 (child labour) and Article 21A (compulsory
            # education) are fundamental-rights provisions but the domain had
            # no example anywhere near this phrasing, so a "child labour"
            # query scored weakly here despite COI being the one act with a
            # direct, on-point hit.
            "child labour and the constitution", "children employed in factories or mines",
        ],
        signals=[
            r"\b(constitution\w*|fundamental\s+rights?|coi)\b",
            r"\b(article\s+\d+|right\s+to\s+life|right\s+to\s+equality)\b",
            r"\b(child\s+labou?r|article\s+24|article\s+21.?a)\b",
            r"\b(freedom\s+of\s+speech|right\s+to\s+education|right\s+to\s+privacy)\b",
            r"\b(directive\s+principles?|dpsp)\b",
            r"\b(parliament\w*|legislature|government|state\s+power)\b",
            r"\b(writ|mandamus|certiorari|prohibition|quo\s+warranto)\b",
            r"\b(supreme\s+court|high\s+court|jurisdiction)\b",
            r"\b(unconstitutional|void\s+law|ultra\s+vires)\b",
            r"\b(citizenship|election|voting|reservation|quota)\b",
        ],
    ),
    Domain(
        name="digital_cyber",
        acts=["ITA", "IPC", "BNS"],
        priority=8,
        examples=[
            "someone hacked my account", "online fraud through a website",
            "my data was stolen digitally", "a phishing scam",
            "morphed photos shared online without consent",
        ],
        signals=[
            r"\b(hack\w*|cyber\w*|digital\w*|online\s+fraud)\b",
            r"\b(computer|internet|website|app|software)\b",
            r"\b(data\s+(theft|breach|leak)|identity\s+theft)\b",
            r"\b(phish\w*|ransomware|malware|virus)\b",
            r"\b(social\s+media|whatsapp|instagram|facebook)\b",
            r"\b(it\s+act|ita|information\s+technology\s+act)\b",
            r"\b(electronic\s+(record|signature|message))\b",
            r"\b(otp\s+fraud|sim\s+swap|cyber\s+crime)\b",
            r"\b(obscene\s+(material|image|video|content)|revenge\s+porn)\b",
        ],
    ),
    Domain(
        name="family_domestic",
        acts=["IPC", "BNS", "POCSO", "CRPC", "BNSS"],
        priority=8,
        examples=[
            "domestic violence from a spouse", "dowry harassment",
            "child custody dispute", "cruelty by a husband or in-laws",
            "a minor was abused within the family",
        ],
        signals=[
            r"\b(husband|wife|spouse|marriage|divorce|matrimon\w*)\b",
            r"\b(dowry|498a|domestic\s+violence)\b",
            r"\b(child\w*|minor|custody|adoption|guardian)\b",
            r"\b(maintenance|alimony|cruelty\s+by)\b",
            r"\b(family|parent\w*|in.?law\w*)\b",
            r"\b(sexual\s+(abuse|assault)\s+(of\s+)?(child|minor))\b",
        ],
    ),
]

# ── Per-act disambiguation ──────────────────────────────────────────────────
# A Domain's score decides which CLUSTER is relevant (e.g. "criminal" covers
# IPC, BNS, NDPS, POCSO, UAPA, PCA, SCST together) but says nothing about
# which ACT within that cluster the query actually points at. Two tiers:
#   Tier 1 (literal)   — small, closed-vocabulary abbreviation/exact-name
#                         regex. Genuinely a fixed set, no fuzzy matching needed.
#   Tier 1.5 (semantic) — canonical behavioural examples per act, matched via
#                         embedding. Replaces most of what used to be large
#                         per-act regex lists (drug/bribe/hack/etc keyword
#                         enumerations) with a handful of example scenarios.
# An act with no Tier 1/1.5 evidence falls back to domain-priority ranking —
# correct default for IPC/BNS, the general-purpose codes.
ACT_ABBREVIATION_SIGNALS: dict[str, list[str]] = {
    "NDPS":  [r"\bndps\b"], "POCSO": [r"\bpocso\b"], "UAPA": [r"\buapa\b"],
    "PCA":   [r"\bpca\b"], "SCST": [r"\bscst\b"], "ITA": [r"\bit\s+act\b", r"\bita\b"],
    "TPA":   [r"\btpa\b"], "ICA": [r"\bica\b"], "SRA": [r"\bsra\b"],
    "CPC":   [r"\bcpc\b"], "COI": [r"\bcoi\b"], "IEA": [r"\biea\b"],
    "BSA":   [r"\bbsa\b|bharatiya\s+sakshya"], "CRPC": [r"\bcrpc\b"],
    "BNSS":  [r"\bbnss\b|bharatiya\s+nagarik"], "LA": [r"\blimitation\s+act\b"],
    "IPC":   [r"\bipc\b|indian\s+penal\s+code"], "BNS": [r"\bbns\b|bharatiya\s+nyaya"],
}

ACT_BEHAVIOURAL_EXAMPLES: dict[str, list[str]] = {
    "NDPS":  ["someone was caught with drugs", "narcotics trafficking or possession"],
    "POCSO": ["a minor was sexually abused", "child sexual assault case"],
    "UAPA":  ["terrorism or an unlawful militant activity"],
    "PCA":   ["a public official took a bribe", "corruption by a government servant"],
    "SCST":  ["caste-based discrimination or an atrocity against a scheduled caste/tribe"],
    "ITA":   ["hacking or unauthorized computer access", "an online data breach"],
    "TPA":   ["a property sale, mortgage or lease dispute"],
    "ICA":   ["breach of a signed contract or agreement"],
    "SRA":   ["asking a court to enforce a contract specifically", "seeking an injunction"],
    "CPC":   ["a civil suit or decree execution"],
    "COI":   ["a fundamental constitutional rights question"],
    "IEA":   ["whether evidence is admissible in court"],
    "BSA":   ["evidence law under the Bharatiya Sakshya Adhiniyam"],
    "CRPC":  ["filing an FIR, arrest, or bail under criminal procedure"],
    "BNSS":  ["criminal procedure under the Bharatiya Nagarik Suraksha Sanhita"],
    "LA":    ["whether a claim is time-barred under the limitation period"],
}

GENERAL_CODE_BONUS = {"IPC": 0.5, "BNS": 0.4}
DOMAIN_EXAMPLES_PATH = str(Path(__file__).parent.parent / "data" / "domain_examples.json")
ACT_EXAMPLES_PATH = str(Path(__file__).parent.parent / "data" / "act_behavioural_examples.json")


# ── Router ────────────────────────────────────────────────────────────────────

@dataclass
class RoutingResult:
    query:          str
    domains:        list[str]
    acts:           list[str]
    primary_acts:   list[str]
    missing_acts:   list[str]
    query_type:     str
    # True only when primary_acts[0] is backed by literal/semantic evidence
    # (Tier 1 or 1.5) pointing at that SPECIFIC act — e.g. the query says
    # "IPC" or matches an act's behavioural examples. False when it only
    # won via Tier 2 domain-priority fallback (which just reflects how the
    # whole domain's acts are weighted, e.g. IPC's GENERAL_CODE_BONUS — not
    # evidence the query is actually about that one act over its siblings
    # in the same domain). Callers should only hard-lock retrieval to a
    # single act when this is True; locking on fallback-only evidence is
    # how a correct domain (e.g. "criminal") silently loses a correct but
    # less-common act (e.g. SCST) to a more common one (e.g. IPC).
    primary_act_has_direct_evidence: bool = False
    # False when the query shows no real signal of being about Indian law
    # at all — see route()'s "plausibly-legal gate" for exactly what that
    # means. Callers should treat False as a reason to reject the query
    # before paying for translation/retrieval/generation, not as a reason
    # to search less broadly (that's what the "activate everything"
    # fallback below is already for, when this is True but ambiguous).
    plausibly_legal: bool = True


# Query TYPE detection is structural/syntactic ("what does section X say"),
# not open-ended vocabulary — regex is the right tool here and this is left
# unchanged; it doesn't have the same scaling problem as domain/act matching.
QUERY_TYPE_PATTERNS = {
    "direct_section": [
        r"\b(what\s+(does|is|says?|states?)\s+)(section\s+\d+|ipc\s+\d+|article\s+\d+)",
        r"\b(section\s+\d+|ipc\s+\d+|ita\s+\d+|article\s+\d+)\s+(of|under)\b",
        r"\b(define|explain|describe)\s+(section|article|clause)\s+\d+",
    ],
    "describe_law": [
        r"\b(there\s+is\s+a\s+law|is\s+there\s+a\s+(law|section|provision))\b",
        r"\b(what\s+(section|law|act|provision)\b.{0,40}(deals?|covers?|applies?))\b",
        r"\b(find|tell\s+me).{0,30}(section|law|provision|act)\b",
        r"\b(law\s+(about|on|for|regarding|governing))\b",
    ],
    "comparative": [
        r"\b(difference|distinguish|compare|vs\.?|versus|between)\b",
        r"\b(how\s+is.{0,30}different\s+from)\b",
        r"\b(same\s+as|similar\s+to)\b",
    ],
    "procedural": [
        r"\b(how\s+to|procedure|process|steps?|what\s+is\s+the\s+procedure)\b",
        r"\b(how\s+do\s+I|how\s+can\s+I|how\s+can\s+one)\b",
        r"\b(file|register|appeal|apply\s+for|obtain)\b",
    ],
    "constitutional": [
        r"\b(fundamental\s+right|constitution|article\s+\d+|supreme\s+court)\b",
        r"\b(right\s+to\s+(life|speech|education|equality|privacy))\b",
    ],
    "scenario": [
        r"\b(is\s+it\s+a\s+crime|what\s+crime|what\s+offence|what\s+(section|law)\s+applies)\b",
        r"\b(punish\w*|liable|guilty|charged?|arrested?)\s+(for|under)\b",
        r"\b(what\s+(happens?|would\s+happen)|can\s+(he|she|they)\s+be)\b",
        r"\b(scenario|situation|case)\b",
    ],
}

# Small, closed set — not the scaling target of this refactor, left as-is.
MISSING_ACT_SIGNALS = {
    # BUGFIX: "cheque\s+bounce" only matched the bare base form — a real
    # query almost always inflects it ("bounced", "bouncing"), which this
    # missed entirely (\b requires a word boundary right after "bounce",
    # but "bounced" has no boundary between "bounce" and "d"). \w* fixes it
    # the same way "dishonour\w*" already handles its own inflections.
    "NI":   [r"\b(cheque\s+bounce\w*|dishonour\w*\s+cheque|ni\s+act|section\s+138)\b"],
    "MCA":  [r"\b(company\s+law|mca|companies\s+act|director\s+(fraud|liability))\b"],
    "IT":   [r"\b(income\s+tax|tds|tax\s+evad\w*|it\s+department)\b"],
    "GST":  [r"\b(gst|goods\s+and\s+services\s+tax|input\s+tax\s+credit)\b"],
    "RTI":  [r"\b(rti|right\s+to\s+information|public\s+information\s+officer)\b"],
    "CPA":  [r"\b(consumer\s+(protection|forum|court)|defective\s+product)\b"],
    "MVA":  [r"\b(motor\s+vehicles?\s+act|driving\s+licence|mva|road\s+accident\s+compensation)\b"],
    "RERA": [r"\b(rera|real\s+estate\s+regul\w*|builder\s+delay|flat\s+possession)\b"],
    # Catches "child labour" queries themselves plus the fact-pattern
    # phrasing ("employed a child in a factory", "minor working in a
    # hazardous job") so the gap warning fires on lay language too, not
    # just the act name.
    "CLPRA": [r"\bchild\s+labou?r\b",
              r"\b(employ\w*|hir\w*|engag\w*|mak\w*|forc\w*)\b.{0,40}"
              r"\b(child|children|minor|underage)\b.{0,40}"
              r"\b(work|job|labou?r|factory|mine|hazardous)\b",
              r"\bworking\s+children\b|\bchild\s+worker\b"],
    # BUGFIX: MISSING_ACTS has had an "FA" entry (Factories Act, 1948) since
    # CLPRA was added, but no corresponding signal here — detect_missing_acts()
    # only iterates this dict's keys, so "FA" could never fire and the
    # Factories Act warning was structurally dead code, unlike every other
    # entry in MISSING_ACTS. Distinct from CLPRA: covers adult/general working
    # conditions and workplace safety, not specifically child labour (that
    # stays on CLPRA's patterns above, which fire independently and can both
    # match the same query — e.g. "child forced to work in a factory").
    "FA":    [r"\bfactories\s+act\b",
              r"\b(working\s+conditions?|workplace\s+safety|industrial\s+safety)\b",
              r"\b(hazardous\s+process|occupational\s+(health|hazard|injury)|workplace\s+injury)\b",
              r"\b(factory\s+(worker|inspector|licen[cs]e)|working\s+hours\s+in\s+a\s+factory)\b"],
}


def detect_query_type(query: str) -> str:
    q = query.lower()
    for qtype, patterns in QUERY_TYPE_PATTERNS.items():
        for p in patterns:
            if re.search(p, q):
                return qtype
    return "scenario"


def detect_missing_acts(query: str) -> list[str]:
    q = query.lower()
    warnings = []
    for code, patterns in MISSING_ACT_SIGNALS.items():
        for p in patterns:
            if re.search(p, q):
                name = MISSING_ACTS.get(code, code)
                warnings.append(f"{name} is not indexed in this dataset.")
                break
    return warnings


class DomainRouter:
    """
    Routes any legal query to the relevant act cluster.

    Key principle: NEVER lock retrieval to a single act.
    Return a cluster of 2-6 acts so retrieval stays broad enough
    to find the right sections even when the query uses lay language.

    embed_fn is optional. Pass the shared embedding model's encode function
    to enable the embedding-first scoring path — without it, this falls
    back to the original regex `signals` lists (unchanged behaviour, still
    works standalone). Adding a NEW domain going forward only requires an
    `examples` list; `signals` is optional and only used as a fallback.
    """

    def __init__(self, embed_fn: Optional[Callable[[list[str]], object]] = None):
        self.embed_fn = embed_fn
        self._domain_matcher = SemanticMatcher(
            label_examples={d.name: d.examples for d in DOMAINS if d.examples},
            embed_fn=embed_fn,
            examples_path=DOMAIN_EXAMPLES_PATH,
        )
        self._act_matcher = SemanticMatcher(
            label_examples=ACT_BEHAVIOURAL_EXAMPLES,
            embed_fn=embed_fn,
            examples_path=ACT_EXAMPLES_PATH,
        )

    def _score_domains_semantic(self, query: str) -> dict[str, float]:
        matches = self._domain_matcher.match(query, top_k=len(DOMAINS))
        return dict(matches)

    def _score_domains_regex(self, q_lower: str) -> dict[str, int]:
        scores: dict[str, int] = {}
        for domain in DOMAINS:
            score = sum(1 for sig in domain.signals if re.search(sig, q_lower))
            if score > 0:
                scores[domain.name] = score
        return scores

    def route(self, query: str) -> RoutingResult:
        q_lower = query.lower()

        query_type = detect_query_type(query)
        missing = detect_missing_acts(query)

        # Regex signal across all 6 domains' ~100+ hand-written patterns
        # (murder/theft/contract/property/constitution/cyber/family — see
        # DOMAINS above) is cheap regardless of embed_fn, and doubles below
        # as the keyword safety net for the plausibly-legal gate: a query
        # that hits ANY of these is unambiguously legal-flavored even if it
        # scores oddly against the (much smaller) curated example set.
        regex_scores = self._score_domains_regex(q_lower)

        # ── Domain scoring: embedding-first, regex fallback ────────────────
        if self.embed_fn:
            semantic_scores = self._score_domains_semantic(query)
            max_semantic = max(semantic_scores.values()) if semantic_scores else 0.0
            # Only keep domains with a plausible match — a low top score
            # across the board means the query doesn't clearly belong
            # anywhere, which is exactly the "activate everything" case.
            SEMANTIC_FLOOR = 0.35
            domain_scores = {k: v for k, v in semantic_scores.items() if v >= SEMANTIC_FLOOR}
        else:
            domain_scores = regex_scores
            max_semantic = 0.0

        # ── Plausibly-legal gate ─────────────────────────────────────────
        # Runs the ENTIRE rest of the pipeline (translation, retrieval,
        # IRAC reranking, a final 14B-model generation call — 5+ sequential
        # LLM round-trips) for a query with literally zero legal signal
        # ("what's my name", small talk, an unrelated topic) both wastes
        # that latency and forces the answer generator to confabulate
        # something from whatever tangentially-scored sections it's handed,
        # since it has no real "this isn't a legal question at all" escape
        # hatch today. Reject here instead, before any of that runs.
        #
        # Deliberately conservative — three independent ways a query counts
        # as legal, ANY one is enough, so this only fires on a query with
        # NONE of: (a) a regex hit against the broad domain pattern set,
        # (b) a hit against MISSING_ACT_SIGNALS (still legal, just an act
        # this dataset doesn't have — that deserves the existing "not
        # indexed" messaging, not a blanket rejection), (c) semantic
        # similarity to ANY domain's curated examples clearing a bar well
        # below SEMANTIC_FLOOR (0.20 vs 0.35 — a real but modest similarity,
        # not "confidently in this domain").
        REJECT_SEMANTIC_FLOOR = 0.20
        plausibly_legal = bool(regex_scores) or bool(missing) or (
            self.embed_fn is not None and max_semantic >= REJECT_SEMANTIC_FLOOR
        )

        if not domain_scores:
            activated_names = [d.name for d in DOMAINS]
            all_acts = list(ALL_ACTS)
        else:
            max_score = max(domain_scores.values())
            # Embedding scores are continuous cosine similarities (0-1);
            # regex scores are integer hit counts — use a proportional
            # band in both cases so the threshold behaves sensibly either way.
            threshold = max_score * 0.85 if self.embed_fn else max(1, max_score - 1)
            activated_names = [
                name for name, score in domain_scores.items() if score >= threshold
            ]
            act_set: set[str] = set()
            for name in activated_names:
                domain = next(d for d in DOMAINS if d.name == name)
                act_set.update(domain.acts)
            all_acts = list(act_set)

        # ── Primary act disambiguation ──────────────────────────────────────
        # Tier 1 — literal abbreviation/exact-name evidence (closed vocabulary).
        act_specific_hits: dict[str, int] = {}
        for act, patterns in ACT_ABBREVIATION_SIGNALS.items():
            hits = sum(1 for p in patterns if re.search(p, q_lower))
            if hits:
                act_specific_hits[act] = hits

        # Tier 1.5 — semantic behavioural evidence (embedding), only used
        # for acts that didn't already get literal evidence, and only when
        # an embed_fn is available.
        act_semantic_hits: dict[str, float] = {}
        if self.embed_fn:
            for act, score in self._act_matcher.match(query, top_k=len(ACT_BEHAVIOURAL_EXAMPLES)):
                if act not in act_specific_hits and score >= 0.5:
                    act_semantic_hits[act] = score

        # Tier 2 — domain-priority fallback for acts with no direct evidence.
        act_domain_fallback: dict[str, float] = {}
        for name in activated_names:
            domain = next(d for d in DOMAINS if d.name == name)
            score = domain_scores.get(name, 1)
            for act in domain.acts:
                act_domain_fallback[act] = act_domain_fallback.get(act, 0) + domain.priority * score

        candidate_acts = set(act_domain_fallback) | set(act_specific_hits) | set(act_semantic_hits)
        act_priority = {
            act: act_specific_hits.get(act, 0) * 1000
                 + act_semantic_hits.get(act, 0.0) * 100
                 + act_domain_fallback.get(act, 0)
                 + GENERAL_CODE_BONUS.get(act, 0)
            for act in candidate_acts
        }

        primary_acts = sorted(act_priority, key=lambda a: (-act_priority[a], a))[:4]

        direct_evidence = bool(primary_acts) and (
            primary_acts[0] in act_specific_hits or primary_acts[0] in act_semantic_hits
        )

        return RoutingResult(
            query        = query,
            domains      = activated_names,
            acts         = all_acts,
            primary_acts = primary_acts,
            missing_acts = missing,
            query_type   = query_type,
            primary_act_has_direct_evidence = direct_evidence,
            plausibly_legal = plausibly_legal,
        )


if __name__ == "__main__":
    router = DomainRouter()   # no embed_fn here — see main.py wiring note
    tests = [
        "What does IPC 302 say?",
        "Ramesh used a fake degree and practiced medicine, patient died",
        "A police officer took bribe to drop FIR",
        "I signed a contract under threat and want to cancel it",
        "Can the government detain someone without trial under Article 22?",
        "My cheque for 5 lakhs bounced, what can I do?",
        "How do I file a bail application?",
        "What is the difference between culpable homicide and murder?",
        "A Dalit woman was denied entry to a temple by upper caste people",
        "A hacker stole my OTP and transferred money from my bank",
        "Contractor defrauded a tribal family and seized their land",
    ]
    for q in tests:
        r = router.route(q)
        print(f"\nQ: {q[:65]}")
        print(f"  type={r.query_type:18s} domains={r.domains}")
        print(f"  primary_acts={r.primary_acts}  all_acts({len(r.acts)})={sorted(r.acts)[:6]}...")
        if r.missing_acts:
            print(f"  ⚠ {r.missing_acts}")