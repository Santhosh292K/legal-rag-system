"""
pipeline/scenario_rewriter.py
Novel component #0 — Scenario-to-Legal Query Rewriter

NOTE ON STATUS: `pipeline/universal_translator.py` already supersedes most
of this file's job in the live pipeline (main.py imports UniversalTranslator,
not ScenarioRewriter) — its own docstring says so explicitly ("Replaces the
scenario_rewriter's conditional approach with an always-on LLM-powered
translation"). This file is still exported from pipeline/__init__.py and may
be used standalone/in tests, so it's fixed here too rather than left with
the same hardcoding problem; but if it's genuinely unused elsewhere, the
lower-effort fix is deleting it and only maintaining universal_translator.py.

CHANGE FROM PREVIOUS VERSION:
The old version had two hardcoded, hand-maintained blocks doing the same
job: an ~80-entry regex→legal-term substitution table (LEGAL_TERM_MAP) for
the rule-based path, AND a ~15-line "role-to-legal-term reference" table
plus 3 fixed worked examples baked directly into the LLM prompt string
(REWRITE_PROMPT) for the LLM path. Both required someone to notice a new
common scenario pattern and hand-write another line, in two different
places, in two different formats.

This version keeps ONE small canonical scenario bank (SCENARIO_BANK) —
a handful of representative example phrasings per scenario category, each
tagged with the legal terms/charges it maps to. That single bank now
drives BOTH paths:
  - Rule-based path:  embedding match against the bank (SemanticMatcher)
                       instead of iterating ~80 regexes.
  - LLM path:          retrieval-augmented few-shot — only the bank entries
                       closest to THIS query are inserted into the prompt,
                       instead of one fixed static block covering every
                       category regardless of relevance.
New scenario categories are added by appending to SCENARIO_BANK (or, in
production, to SCENARIO_BANK_PATH as a JSON file) — one place, one format.
"""
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.semantic_matcher import SemanticMatcher
from config import OLLAMA_FAST_MODEL
import ollama


# ── Scenario detection ────────────────────────────────────────────────────────

# Structural signal words ("did", "was a", "under") — kept as regex since
# this is a cheap, syntactic pre-filter, not the open-ended vocabulary this
# refactor targets. The SemanticMatcher score below is the real detector.
SCENARIO_SIGNALS = [
    r"\b(did|does|has|had|got|used|took|gave|made|said|told|went|came|caused)\b",
    r"\b(is a|was a|works as|worked as|acts as|posed as|pretended to be)\b",
    r"\b(patient|victim|accused|person|man|woman|he|she|they|who)\b",
    r"\b(died|death|killed|injured|hurt|suffered|lost|stolen|cheated|deceived)\b",
    r"\b(under|with|using|by|through|via|because of|as a result of)\b",
    r"\b(what is|what are|what would|what crime|what offence|what punishment)\b",
    r"\b(crime|criminal|liable|guilty|punishable|charged|arrested|convicted)\b",
    r"\b(fake|forged|false|fraudulent|illegal|unlawful|unauthorized|without)\b",
]
SCENARIO_PATTERN = re.compile("|".join(SCENARIO_SIGNALS), re.IGNORECASE)
SCENARIO_REGEX_THRESHOLD = 3
SCENARIO_SEMANTIC_THRESHOLD = 0.6


# ── Canonical scenario bank ────────────────────────────────────────────────
# Replaces LEGAL_TERM_MAP + the LLM prompt's static reference table. Each
# entry: a few example phrasings (what this scenario sounds like) + the
# legal terms/charges it maps to. Matched via embedding, not regex, so
# paraphrases of these examples are caught without new entries.
SCENARIO_BANK: dict[str, dict] = {
    "fake_professional": {
        "examples": ["a fake doctor treated patients", "practiced medicine with a forged degree",
                     "someone posed as a professional they aren't"],
        "legal_terms": "personation cheating forged certificate false document IPC 416 419 468",
        "charges": ["IPC_416", "IPC_419", "IPC_468", "IPC_420"],
    },
    "negligent_death": {
        "examples": ["a patient died due to negligence", "medical negligence caused death",
                     "someone died because of carelessness"],
        "legal_terms": "death caused by negligence rash act culpable homicide IPC 304A",
        "charges": ["IPC_304A"],
    },
    "assault": {
        "examples": ["someone was beaten or assaulted", "he hit her causing injury",
                     "physical attack causing hurt"],
        "legal_terms": "hurt grievous hurt assault IPC 319 323 325",
        "charges": ["IPC_323", "IPC_325"],
    },
    "theft": {
        "examples": ["something was stolen", "a robbery took place", "property was taken without consent"],
        "legal_terms": "theft robbery stolen property IPC 378 379 390 392",
        "charges": ["IPC_378", "IPC_379"],
    },
    "murder": {
        "examples": ["someone was killed intentionally", "a murder took place"],
        "legal_terms": "murder culpable homicide death IPC 302 300",
        "charges": ["IPC_302"],
    },
    "sexual_assault": {
        "examples": ["a sexual assault occurred", "someone was raped",
                     "non-consensual sexual act"],
        "legal_terms": "rape sexual assault outraging modesty IPC 376 354",
        "charges": ["IPC_376"],
    },
    "hacking": {
        "examples": ["someone hacked a computer system", "unauthorized digital access",
                     "a cyberattack on an account"],
        "legal_terms": "unauthorised access computer system cybercrime ITA 43 66",
        "charges": ["ITA_066"],
    },
    "bribery": {
        "examples": ["a public official took a bribe", "corruption by a government servant"],
        "legal_terms": "corruption public servant gratification PCA 13 IPC 161",
        "charges": ["PCA_013"],
    },
    "cheating_fraud": {
        "examples": ["someone was cheated or scammed for money", "a person was defrauded"],
        "legal_terms": "cheating dishonestly deceived induced delivery property IPC 415 420",
        "charges": ["IPC_420"],
    },
    "extortion_blackmail": {
        "examples": ["someone was blackmailed", "threatened for money"],
        "legal_terms": "extortion threat wrongful gain IPC 383 385",
        "charges": ["IPC_383"],
    },
    "kidnapping": {
        "examples": ["a person was kidnapped or abducted"],
        "legal_terms": "abduction kidnapping wrongful confinement IPC 359 363",
        "charges": ["IPC_363"],
    },
    "domestic_violence": {
        "examples": ["a husband abused or beat his wife", "domestic violence at home",
                     "cruelty by in-laws"],
        "legal_terms": "cruelty husband wife domestic violence IPC 498A",
        "charges": ["IPC_498A"],
    },
    "dowry": {
        "examples": ["dowry was demanded", "harassment for dowry", "a dowry death"],
        "legal_terms": "dowry death cruelty husband IPC 304B 498A",
        "charges": ["IPC_304B", "IPC_498A"],
    },
    "minor_abuse": {
        "examples": ["a child or minor was sexually abused", "a student was abused by a teacher"],
        "legal_terms": "sexual abuse of minor child POCSO penetrative assault IPC 376",
        "charges": ["IPC_376"],
    },
    "narcotics": {
        "examples": ["drugs were found on someone", "narcotics trafficking or possession"],
        "legal_terms": "narcotic substance NDPS possession trafficking",
        "charges": [],
    },
    "police_misconduct": {
        "examples": ["police beat someone in custody", "a wrongful arrest by police",
                     "police took a bribe to drop a case", "police refused to register an FIR",
                     "a custodial death in police custody"],
        "legal_terms": ("public servant wrongful confinement wrongful arrest disobeying law "
                         "failure to investigate custodial death IPC 166 166A 330 342"),
        "charges": ["IPC_166", "IPC_166A", "IPC_330", "IPC_342"],
    },
    "cheque_bounce": {
        "examples": ["a cheque bounced or was dishonoured"],
        "legal_terms": ("cheating dishonestly inducing delivery of property IPC 420 — "
                         "NOTE: Negotiable Instruments Act Section 138 applies but is not indexed"),
        "charges": ["IPC_420"],
    },
    "embezzlement": {
        "examples": ["an employee stole from their employer", "funds were embezzled",
                     "criminal breach of trust"],
        "legal_terms": "criminal breach of trust misappropriation IPC 406 408 409",
        "charges": ["IPC_406", "IPC_409"],
    },
    "wage_theft": {
        "examples": ["an employer did not pay wages", "forced or unpaid labour"],
        "legal_terms": "wrongful withholding of wages forced labour IPC 374",
        "charges": ["IPC_374"],
    },
    "workplace_harassment": {
        "examples": ["sexual harassment at the workplace"],
        "legal_terms": "sexual harassment at workplace outraging modesty IPC 354A",
        "charges": ["IPC_354A"],
    },
    "wrongful_eviction": {
        "examples": ["a landlord illegally evicted a tenant", "a tenant was forced out",
                     "someone trespassed on land"],
        "legal_terms": "wrongful eviction criminal trespass IPC 441 447",
        "charges": ["IPC_441"],
    },
    "property_fraud": {
        "examples": ["property was transferred fraudulently", "a fake sale deed",
                     "benami property fraud"],
        "legal_terms": "fraudulent transfer of property forgery TPA IPC 420 463",
        "charges": ["IPC_420", "IPC_463"],
    },
    "riot_public_order": {
        "examples": ["a riot or mob attack occurred", "hate speech inciting violence",
                     "sedition against the state"],
        "legal_terms": "rioting unlawful assembly promoting enmity IPC 146 147 153A",
        "charges": ["IPC_147", "IPC_153A"],
    },
    "defamation": {
        "examples": ["someone made a false accusation", "defamatory statements were made"],
        "legal_terms": "defamation false charge IPC 499 500",
        "charges": ["IPC_499"],
    },
    "scst_atrocity": {
        "examples": ["caste-based discrimination or abuse", "an atrocity against a scheduled caste/tribe person"],
        "legal_terms": "atrocity scheduled caste tribe SCST prevention atrocities act",
        "charges": [],
    },
    "civil_contract_breach": {
        "examples": ["a contract was breached", "an agreement was not honoured",
                     "seeking damages for breach of contract"],
        "legal_terms": "breach of contract damages compensation ICA 73 74",
        "charges": [],
    },
    "civil_void_agreement": {
        "examples": ["is this agreement legally valid", "the contract may be void or unenforceable"],
        "legal_terms": "void agreement voidable contract ICA 10 19 20",
        "charges": [],
    },
    "civil_specific_performance": {
        "examples": ["asking a court to enforce a contract", "seeking an injunction"],
        "legal_terms": "specific performance injunction SRA 10 36 37",
        "charges": [],
    },
    "civil_property_dispute": {
        "examples": ["a property partition dispute", "an inheritance or succession dispute",
                     "a mortgage or lease dispute"],
        "legal_terms": "partition inheritance succession mortgage TPA CPC",
        "charges": [],
    },
}

SCENARIO_BANK_PATH = str(Path(__file__).parent.parent / "data" / "scenario_bank.json")


def _build_scenario_matcher(embed_fn: Optional[Callable] = None) -> SemanticMatcher:
    return SemanticMatcher(
        label_examples={k: v["examples"] for k, v in SCENARIO_BANK.items()},
        embed_fn=embed_fn,
        examples_path=SCENARIO_BANK_PATH,
    )


def is_scenario_query(query: str, matcher: Optional[SemanticMatcher] = None) -> bool:
    """Returns True if the query appears to be a factual scenario. Regex
    signal count is the fast primary check; if that's inconclusive and an
    embedding matcher is available, a strong semantic match against the
    scenario bank also counts (catches scenarios phrased without the
    generic signal words above)."""
    regex_hits = len(SCENARIO_PATTERN.findall(query))
    if regex_hits >= SCENARIO_REGEX_THRESHOLD:
        return True
    if matcher and matcher.embed_fn:
        matches = matcher.match(query, top_k=1)
        if matches and matches[0][1] >= SCENARIO_SEMANTIC_THRESHOLD:
            return True
    return False


# ── Rule-based fallback rewriter (embedding-driven) ─────────────────────────

def semantic_rewrite(query: str, matcher: SemanticMatcher, threshold: float = 0.5,
                      max_matches: int = 3) -> tuple[str, list[str]]:
    """Embedding replacement for the old regex-substitution rule_rewrite().
    Returns (expanded_query_text, matched_charge_ids)."""
    if not matcher.embed_fn:
        return query, []
    matches = matcher.match(query, top_k=max_matches)
    appended_terms, charges = [], []
    for label, score in matches:
        if score < threshold:
            continue
        entry = SCENARIO_BANK.get(label, {})
        if entry.get("legal_terms"):
            appended_terms.append(entry["legal_terms"])
        charges.extend(entry.get("charges", []))
    expanded = query
    if appended_terms:
        expanded = f"{query} " + " ".join(appended_terms)
    return expanded.strip(), list(dict.fromkeys(charges))


def rule_rewrite(query: str, matcher: Optional[SemanticMatcher] = None) -> str:
    """Backward-compatible signature — falls back to a no-op expansion
    (original query unchanged) if no matcher/embed_fn is supplied, since
    the old giant regex table has been retired in favour of semantic_rewrite."""
    if matcher and matcher.embed_fn:
        expanded, _ = semantic_rewrite(query, matcher)
        return expanded
    return query


# ── LLM rewriter — now retrieval-augmented instead of statically prompted ──

REWRITE_PROMPT_TEMPLATE = """You are a legal query rewriter for Indian law.

A user has described a factual scenario. Your job is to rewrite it into legal terminology so that the correct Indian Penal Code (IPC), CrPC, Prevention of Corruption Act (PCA), IT Act, POCSO, NDPS, or other relevant sections can be found.

RULES:
- Output ONLY a JSON object, nothing else.
- Identify every legal element in the scenario.
- Map everyday actions to precise Indian legal terms.
- Mention the most relevant Act(s).
- Do NOT invent case citations. Do NOT answer the question. Just rewrite it.

The following reference mappings are the ones most relevant to THIS query
(retrieved from a larger bank — not a fixed list, so trust them but you are
not limited to only these):
{retrieved_examples}

Now rewrite this:
Input: "{query}"
Output (JSON only):"""


def _format_retrieved_examples(query: str, matcher: SemanticMatcher, k: int = 5) -> str:
    """Retrieval-augmented few-shot: only pull the bank entries closest to
    THIS query into the prompt, instead of one fixed static block that
    included every category regardless of relevance."""
    matches = matcher.match(query, top_k=k) if matcher.embed_fn else []
    if not matches:
        # No embed_fn available — fall back to a small fixed sample so the
        # LLM still has *some* grounding, just not query-tailored.
        sample = list(SCENARIO_BANK.items())[:3]
    else:
        sample = [(label, SCENARIO_BANK[label]) for label, _ in matches if label in SCENARIO_BANK]

    lines = []
    for label, entry in sample:
        lines.append(f'  "{label}" → "{entry["legal_terms"]}"')
    return "\n".join(lines) if lines else "  (no close matches — rely on general legal knowledge)"


@dataclass
class RewrittenQuery:
    original:       str
    rewritten:      str
    acts:           list[str]
    legal_elements: list[str]
    charges:        list[str]
    used_llm:       bool = False
    is_civil:       bool = False
    dataset_gaps:   list[str] = None

    def __post_init__(self):
        if self.dataset_gaps is None:
            self.dataset_gaps = []


MISSING_FROM_DATASET = {
    "NI":  "Negotiable Instruments Act (cheque dishonour — Section 138). "
           "Add NI Act sections to your dataset to answer cheque-bounce queries.",
}


def _detect_dataset_gaps(acts: list[str], rewritten: str) -> list[str]:
    gaps = []
    for code, message in MISSING_FROM_DATASET.items():
        if code in acts or code in rewritten.upper():
            gaps.append(message)
    return gaps


def llm_rewrite(query: str, matcher: SemanticMatcher) -> RewrittenQuery:
    """Uses the LLM to rewrite the scenario into legal terms, grounded in
    the top-k retrieved scenario-bank entries for THIS query rather than a
    fixed static reference table."""
    prompt = REWRITE_PROMPT_TEMPLATE.format(
        retrieved_examples=_format_retrieved_examples(query, matcher),
        query=query,
    )
    response = ollama.chat(
        model=OLLAMA_FAST_MODEL,
        format="json",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response["message"]["content"].strip()
    data = json.loads(raw)

    acts = data.get("acts", [])
    rewritten = data.get("rewritten", query)
    return RewrittenQuery(
        original=query,
        rewritten=rewritten,
        acts=acts,
        legal_elements=data.get("legal_elements", []),
        charges=data.get("charges", []),
        used_llm=True,
        dataset_gaps=_detect_dataset_gaps(acts, rewritten),
    )


# ── Civil intent detector (kept regex — small, stable, structural) ─────────

CIVIL_SIGNALS = [
    r"\b(contract|agreement|breach|damages|compensation|refund)\b",
    r"\b(property|land|house|flat|plot|sale|transfer|mortgage|rent)\b",
    r"\b(injunction|specific\s+performance|possession|evict)\b",
    r"\b(sue|civil\s+suit|civil\s+case|file\s+a\s+case|take\s+to\s+court)\b",
    r"\b(partition|will|inheritance|succession|gift\s+deed)\b",
    r"\b(tenant|landlord|lessor|lessee|rent\s+agreement)\b",
]
CIVIL_PATTERN = re.compile("|".join(CIVIL_SIGNALS), re.IGNORECASE)

CRIMINAL_OVERRIDE = re.compile(
    r"\b(crime|criminal|punish|imprisonment|jail|police|FIR|arrest|murder|"
    r"rape|theft|robbery|fraud|cheat|bribe|corrupt)\b",
    re.IGNORECASE,
)


def is_civil_query(query: str) -> bool:
    civil_hits = len(CIVIL_PATTERN.findall(query))
    criminal_hits = len(CRIMINAL_OVERRIDE.findall(query))
    return civil_hits >= 2 and criminal_hits < 2


CIVIL_ACT_PRIORITY = {
    "contract":   ["ICA", "SRA"],
    "property":   ["TPA", "SRA", "CPC"],
    "eviction":   ["SRA", "TPA", "CPC"],
    "injunction": ["SRA", "CPC"],
    "partition":  ["TPA", "CPC"],
    "will":       ["TPA", "IPC"],
    "rent":       ["TPA", "CPC"],
    "specific performance": ["SRA"],
}


def _civil_acts_for_query(query: str) -> list[str]:
    q_lower = query.lower()
    for keyword, acts in CIVIL_ACT_PRIORITY.items():
        if keyword in q_lower:
            return acts
    return ["ICA", "SRA", "TPA", "CPC"]


# ── Main class ────────────────────────────────────────────────────────────────

class ScenarioRewriter:
    """
    Detects scenario-based queries and rewrites them into legal language
    BEFORE they reach BM25/dense retrieval.

    embed_fn is optional. Pass the shared embedding model's encode function
    to enable the embedding-driven scenario matching and retrieval-augmented
    LLM prompting described above. Without it, this degrades to: regex
    scenario detection + LLM rewrite with a small fixed few-shot sample
    (no semantic rewrite tier) — never crashes, just less precise.
    """

    def __init__(self, embed_fn: Optional[Callable[[list[str]], object]] = None):
        self.embed_fn = embed_fn
        self._matcher = _build_scenario_matcher(embed_fn)

    def rewrite(self, query: str) -> RewrittenQuery:
        if not is_scenario_query(query, self._matcher):
            return RewrittenQuery(
                original=query, rewritten=query, acts=[],
                legal_elements=[], charges=[], used_llm=False,
            )

        civil = is_civil_query(query)
        rule_expanded = rule_rewrite(query, self._matcher)
        rule_gaps = _detect_dataset_gaps([], rule_expanded)

        try:
            llm_result = llm_rewrite(query, self._matcher)
            merged = f"{query} {rule_expanded} {llm_result.rewritten}"
            llm_result.rewritten = merged.strip()
            llm_result.is_civil = civil
            llm_result.dataset_gaps = list(set(rule_gaps + llm_result.dataset_gaps))
            if civil:
                civil_acts = _civil_acts_for_query(query)
                for a in civil_acts:
                    if a not in llm_result.acts:
                        llm_result.acts.append(a)
            return llm_result
        except Exception:
            acts = _civil_acts_for_query(query) if civil else ["IPC"]
            return RewrittenQuery(
                original=query, rewritten=rule_expanded, acts=acts,
                legal_elements=[], charges=[], used_llm=False,
                is_civil=civil, dataset_gaps=rule_gaps,
            )


if __name__ == "__main__":
    rewriter = ScenarioRewriter()  # no embed_fn here — see main.py wiring note
    test_cases = [
        "Ramesh is a doctor who got a fake degree and practiced medicine, but a patient died under his medication",
        "A police officer arrested Ravi without reason and kept him in lockup for 5 days without informing his family",
        "A sub-inspector took 50,000 rupees bribe from a suspect to drop the case",
        "A husband beat his wife daily and his parents demanded more dowry",
        "A software engineer hacked into a bank and transferred 10 lakhs to his account",
        "An employer did not pay wages for 6 months and threatened to fire if anyone complained",
        "What is the punishment for hacking under the IT Act?",   # NOT a scenario
        "A landlord cut off water and electricity to force the tenant to vacate",
        "A teacher sexually abused a 14-year-old student after school",
        "A company accountant embezzled 20 lakhs from company funds over 2 years",
    ]
    for q in test_cases:
        r = rewriter.rewrite(q)
        is_sc = is_scenario_query(q, rewriter._matcher)
        print(f"\nOriginal   : {r.original}")
        print(f"Is scenario: {is_sc}")
        if is_sc:
            print(f"Rewritten  : {r.rewritten[:130]}...")
            if r.charges:
                print(f"Charges    : {r.charges}")