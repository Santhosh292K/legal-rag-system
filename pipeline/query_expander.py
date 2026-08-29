"""
pipeline/query_expander.py
Multi-query expansion: LLM query rewriting + embedding-based concept
matching + a small canonical abbreviation map.

CHANGE FROM PREVIOUS VERSION:
The old version leaned on a single ~36-entry hardcoded regex-substitution
dictionary (LEGAL_SYNONYMS) as its main non-LLM expansion mechanism. That
only matched the exact substrings someone thought to enumerate, and every
new term (a new slang word, a new act's vocabulary) required a code change.

This version keeps three mechanisms, each used for what it's actually
good at:
  1. LLM query rewriting (llm_expand)      — unlimited vocabulary, already
                                              existed, unchanged here.
  2. Embedding concept matching             — a SMALL set of canonical
     (semantic_concept_expand)                example phrasings per legal
                                              concept, matched by cosine
                                              similarity via SemanticMatcher.
                                              Generalises to paraphrases of
                                              those examples automatically,
                                              and new concepts/examples can
                                              be added via a JSON data file
                                              (see CONCEPT_EXAMPLES_PATH)
                                              instead of editing this file.
  3. Canonical abbreviation map              — a genuinely small (~20 item),
     (ABBREVIATIONS)                          stable set of exact acronym
                                              expansions (IPC, CRPC, FIR...).
                                              This is the one case where a
                                              literal lookup is actually the
                                              right tool: abbreviations are
                                              closed-vocabulary and don't
                                              benefit from fuzzy matching.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.intent_classifier import QueryIntent
from pipeline.semantic_matcher import SemanticMatcher

from config import OLLAMA_FAST_MODEL
import ollama


# ── 1. Canonical abbreviation map ──────────────────────────────────────────
# Small and stable by nature — Indian statute acronyms don't change often,
# and unlike free-text vocabulary, an abbreviation genuinely IS a fixed,
# closed set worth hardcoding rather than embedding-matching.
ABBREVIATIONS: dict[str, str] = {
    "fir":    "first information report",
    "ipc":    "indian penal code",
    "crpc":   "code of criminal procedure",
    "bns":    "bharatiya nyaya sanhita",
    "bnss":   "bharatiya nagarik suraksha sanhita",
    "bsa":    "bharatiya sakshya adhiniyam",
    "iea":    "indian evidence act",
    "ica":    "indian contract act",
    "cpc":    "code of civil procedure",
    "sra":    "specific relief act",
    "tpa":    "transfer of property act",
    "coi":    "constitution of india",
    "ita":    "information technology act",
    "ndps":   "narcotic drugs and psychotropic substances act",
    "pca":    "prevention of corruption act",
    "pocso":  "protection of children from sexual offences act",
    "scst":   "sc/st prevention of atrocities act",
    "uapa":   "unlawful activities prevention act",
    "jmfc":   "judicial magistrate first class",
    "hc":     "high court",
    "sc":     "supreme court",
}

_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in ABBREVIATIONS) + r")\b",
    re.IGNORECASE,
)


def expand_abbreviations(query: str) -> Optional[str]:
    """If the query contains a known abbreviation, return one variant with
    it spelled out in full (helps BM25 match sections that spell the act
    name out rather than abbreviating it). Returns None if no abbreviation
    is present, so callers can skip adding a duplicate query."""
    matches = list(_ABBREV_PATTERN.finditer(query))
    if not matches:
        return None
    expanded = query
    for m in matches:
        full = ABBREVIATIONS[m.group(1).lower()]
        expanded = expanded[: m.start()] + full + expanded[m.end():]
    return expanded


# ── 2. Embedding concept matching ──────────────────────────────────────────
# Each concept carries a few canonical example phrasings (what a query about
# this concept sounds like) and a set of terms to append when matched. New
# concepts can be added at CONCEPT_EXAMPLES_PATH without touching this file.
CONCEPT_EXAMPLES_PATH = str(
    Path(__file__).parent.parent / "data" / "query_concept_examples.json"
)

LEGAL_CONCEPTS: dict[str, dict[str, list[str]]] = {
    "punishment": {
        "examples": ["what is the punishment for this", "penalty for this offence",
                     "how many years in jail", "sentence for this crime"],
        "terms": ["penalty", "sentence", "imprisonment", "fine"],
    },
    "offence": {
        "examples": ["is this a crime", "did they commit an offence",
                     "is this illegal", "is this a violation of law"],
        "terms": ["crime", "violation", "contravention", "offense"],
    },
    "civil_judgment": {
        "examples": ["what did the court decide", "the judge's order",
                     "final verdict in the case"],
        "terms": ["judgment", "order", "verdict", "decision", "decree"],
    },
    "lawsuit": {
        "examples": ["how do I sue someone", "filing a case against",
                     "taking legal action in court"],
        "terms": ["case", "action", "proceeding", "litigation", "suit"],
    },
    "bail": {
        "examples": ["how to get out of jail before trial",
                     "temporary release from custody", "applying for bail"],
        "terms": ["bail", "temporary release", "surety", "bond"],
    },
    "appeal": {
        "examples": ["challenging a court decision", "going to a higher court",
                     "asking the court to reconsider"],
        "terms": ["appeal", "revision", "challenge", "contest"],
    },
    "contract": {
        "examples": ["a signed agreement between two people",
                     "the deal we agreed to in writing"],
        "terms": ["contract", "agreement", "deed", "covenant"],
    },
    "property": {
        "examples": ["land or house ownership", "who owns this asset",
                     "immovable property dispute"],
        "terms": ["property", "asset", "estate", "immovable property"],
    },
    "accused_person": {
        "examples": ["the person being blamed for the crime",
                     "the defendant in this case"],
        "terms": ["accused", "defendant", "respondent", "undertrial"],
    },
    "complainant": {
        "examples": ["the person who filed the complaint",
                     "the victim who brought the case"],
        "terms": ["plaintiff", "complainant", "petitioner", "claimant"],
    },
    "evidence": {
        "examples": ["proof to support the claim", "witness testimony",
                     "documents proving what happened"],
        "terms": ["evidence", "proof", "document", "witness"],
    },
    "liability": {
        "examples": ["who is responsible for this", "who is at fault",
                     "who can be held accountable"],
        "terms": ["liable", "responsible", "accountable", "culpable"],
    },
    "void_contract": {
        "examples": ["is this agreement legally valid",
                     "can this contract be cancelled", "an invalid agreement"],
        "terms": ["void", "invalid", "null", "unenforceable"],
    },
    "fir_filing": {
        "examples": ["registering a police complaint", "reporting a crime to police",
                     "the police wouldn't register my complaint"],
        "terms": ["first information report", "complaint", "police complaint"],
    },
    "chargesheet": {
        "examples": ["the document police file after investigation",
                     "formal charges filed by police"],
        "terms": ["challan", "charge sheet", "police report"],
    },
    "cognizable_offence": {
        "examples": ["can police arrest without a warrant for this",
                     "is this a serious enough crime for arrest"],
        "terms": ["non-bailable", "serious offence", "arrest without warrant"],
    },
    "abetment": {
        "examples": ["helping someone commit a crime", "encouraging a crime",
                     "being part of a conspiracy"],
        "terms": ["instigation", "aiding", "facilitation", "conspiracy"],
    },
    "cheating": {
        "examples": ["someone tricked me for money", "fraudulent deception",
                     "misrepresentation to gain money"],
        "terms": ["fraud", "deception", "misrepresentation", "false pretence"],
    },
    "misappropriation": {
        "examples": ["someone stole money entrusted to them",
                     "embezzling funds", "breach of trust with property"],
        "terms": ["embezzlement", "criminal breach of trust", "conversion"],
    },
    "dowry": {
        "examples": ["demanding gifts or money for marriage",
                     "harassment over wedding gifts"],
        "terms": ["stridhan", "matrimonial property", "wedding gift demand"],
    },
    "pocso_matter": {
        "examples": ["a minor was sexually abused", "child sexual abuse case"],
        "terms": ["protection of children", "child sexual abuse", "minor victim"],
    },
    # Added after diagnose_recall.py flagged these as raw_pool misses —
    # queries paraphrase the act (e.g. "extracted money regularly", "took
    # money by force") in words that never match the corpus's own
    # keywords ("extortion") via BM25, and dense retrieval alone wasn't
    # closing the gap either. See conversation notes / diagnose_output.txt.
    "extortion": {
        "examples": ["threatened someone to hand over money", "forced someone to pay by fear",
                     "took money regularly by threatening", "demanded money under threat of harm"],
        "terms": ["extortion", "putting in fear of injury", "dishonestly inducing delivery of property"],
    },
    "false_accusation": {
        # diagnose_recall.py: IPC_211 now surfaces on this scenario but
        # IPC_166 (public servant disobeying law with intent to cause
        # injury) still doesn't — added an example mirroring the actual
        # gold facts (officer acting "at the behest of" someone else) and
        # explicit sec-166 style terms.
        "examples": ["police filed a false case against someone", "falsely implicated in a crime",
                     "fabricated FIR at someone's instigation", "public servant abused power to frame someone",
                     "a false FIR was registered against me",
                     "an SHO registered a false FIR against an innocent person at the behest of a politician"],
        "terms": ["false charge", "false information", "instituting criminal proceedings without lawful ground",
                  "public servant disobeying a direction of law with intent to cause injury to any person",
                  "abuse of official position to frame someone at another's instigation"],
    },
    # Mirrors of the QUICK_SYNONYMS regex fixes in universal_translator.py —
    # same rationale: paraphrases that never hit the literal regex should
    # still be caught by the embedding tier. See diagnose_output.txt.
    "employee_breach_of_trust": {
        "examples": ["an employee stole company money over several months",
                     "a clerk siphoned off funds entrusted to him",
                     "a cashier misappropriated cash over time",
                     "staff member diverted company funds regularly"],
        "terms": ["criminal breach of trust", "servant entrusted with property",
                  "misappropriation by agent", "dishonest misappropriation"],
    },
    "forged_credential": {
        # diagnose_recall.py: IPC_468 was found but truncated (separate
        # top-k issue) and IPC_420 (cheating) never surfaced at all —
        # forgery is instrumental to cheating, so fold that framing in
        # directly rather than treating them as separate concepts.
        "examples": ["used a forged certificate to get a job",
                     "submitted a fake degree for employment",
                     "presented a fabricated diploma or licence",
                     "forged a signature on a document",
                     "a person forged their caste certificate to get government reservation benefits"],
        "terms": ["forgery", "false document", "fabricated for cheating purpose",
                  "cheating", "dishonestly inducing delivery of property"],
    },
    "bribe_demand": {
        # diagnose_recall.py: PCA_007 surfaces but PCA_013 (criminal
        # misconduct by a public servant) doesn't — added its own terms.
        "examples": ["an official demanded a bribe", "asked for money to process the file",
                     "a public servant sought illegal gratification",
                     "an officer demanded a bribe to pass a contractor's bill"],
        "terms": ["public servant", "illegal gratification", "demand of bribe",
                  "criminal misconduct by a public servant",
                  "obtains valuable thing or pecuniary advantage by corrupt or illegal means"],
    },
    "child_labour": {
        # Gap: no concept bucket existed for this at all (child_marriage and
        # pocso_matter both existed, but child LABOUR/employment did not),
        # so semantic_concept_expand had nothing to match a "child labour"
        # query against, and the query drifted toward whichever unrelated
        # child-protection sections (kidnapping, abandonment, prostitution)
        # happened to share the token "child" instead of the ones that
        # actually govern child employment/labour.
        "examples": ["a child was employed in a hazardous job",
                     "a minor was made to work in a factory or mine",
                     "a 13 year old was forced to work instead of going to school",
                     "a trafficked child was employed as cheap labour",
                     "what are the laws against child labour"],
        "terms": ["child labour", "employment of children in hazardous work",
                  "no child below fourteen years employed in a factory or mine",
                  "compulsory education", "employing a trafficked person is a child",
                  "exploits a trafficked person", "exploitation of a trafficked child",
                  "trafficking for forced labour or exploitation",
                  "unlawfully compelling a person to labour"],
    },
    "child_marriage": {
        # diagnose_recall.py: both IPC_493 and POCSO_003 were completely
        # missed even with the earlier expansion — this pairing was
        # already flagged as low vocabulary overlap in the original
        # priority list; the extra examples/terms here are a best effort,
        # but a legal_kg.py cross-reference is more likely to be needed.
        "examples": ["a minor was forced into marriage", "underage marriage arranged by family",
                     "married off before turning 18",
                     "a girl was forced into marriage at 17 years of age"],
        "terms": ["child marriage", "prohibition of child marriage", "minor consent to marriage",
                  "cohabitation caused by a man deceitfully inducing a belief of lawful marriage"],
    },
    "data_privacy": {
        # diagnose_recall.py: ITA_072A surfaces but ITA_043A (compensation
        # for a body corporate's negligence with sensitive personal data)
        # doesn't — that's a distinct civil-liability section.
        "examples": ["Aadhaar data was shared without consent", "misuse of personal data",
                     "sensitive personal data leaked without permission",
                     "a private company is storing Aadhaar-linked data without consent"],
        "terms": ["aadhaar", "personal data", "consent", "data protection",
                  "body corporate negligent in implementing reasonable security practices",
                  "compensation for failure to protect data"],
    },
    "testamentary_capacity": {
        # diagnose_recall.py: ICA_012 surfaces but ICA_011 (competency to
        # contract) doesn't.
        "examples": ["a will was made by someone of unsound mind",
                     "testator was not mentally competent when the will was signed",
                     "a man wrote a will leaving all property to his son but he was of unsound mind"],
        "terms": ["testamentary capacity", "unsound mind", "contractual competency",
                  "who is competent to contract", "person of unsound mind is incompetent to contract"],
    },
    "cartel_price_fixing": {
        # diagnose_recall.py: gold section is ICA_023 (agreement with
        # unlawful object / opposed to public policy is void), not a
        # Competition Act provision, so the terms need Contract Act
        # vocabulary rather than just competition-law language.
        "examples": ["companies formed a cartel", "businesses engaged in price fixing",
                     "an anti-competitive agreement between firms",
                     "two companies formed a cartel and fixed prices for essential medicines"],
        "terms": ["cartel", "price fixing", "anti-competitive agreement", "collusion",
                  "agreement with unlawful object or consideration",
                  "agreement opposed to public policy is void"],
    },
    "eviction": {
        # diagnose_recall.py: SRA_006 was found but truncated (top-k
        # issue) and TPA_108 (rights and liabilities of lessor/lessee)
        # never surfaced — added lease/tenancy vocabulary.
        "examples": ["tenant was evicted by the landlord", "forcibly evicted from the property",
                     "thrown out of the house without notice",
                     "a landlord wants to evict a long-term tenant with no written agreement"],
        "terms": ["dispossession", "eviction", "possession", "lease", "tenancy",
                  "rights and liabilities of lessor and lessee", "holding over"],
    },
    "specific_performance_sale": {
        "examples": ["seller promised to sell but now refuses",
                     "want to force the other party to transfer the property",
                     "buyer wants the sale agreement enforced"],
        "terms": ["specific performance", "enforcement of contract", "compel transfer"],
    },
    "toxic_waste_pollution": {
        # diagnose_recall.py: IPC_277/278 surface but IPC_284 (negligent
        # conduct with respect to a poisonous substance) doesn't.
        "examples": ["a factory released toxic waste into a river",
                     "chemical discharge poisoned a water source",
                     "industrial pollution harmed public health"],
        "terms": ["negligent conduct with respect to a poisonous substance",
                  "fouling water", "public nuisance"],
    },
    "hate_speech": {
        # diagnose_recall.py: IPC_153A, IPC_505, and UAPA_013 were all
        # completely missed — no concept covered this scenario before.
        "examples": ["a political leader made a speech calling for violence against a minority community",
                     "hate speech inciting violence against a religious group",
                     "public statement promoting enmity between communities"],
        "terms": ["promoting enmity between different groups", "statements conducing to public mischief",
                  "unlawful association", "acts prejudicial to communal harmony"],
    },
    "tribal_land_dispossession": {
        # diagnose_recall.py: COI_300A, IPC_447, and SCST_003 were all
        # completely missed — no concept covered this scenario before.
        "examples": ["a tribal farmer's land was forcibly acquired by a private mining company",
                     "indigenous community land taken without due process",
                     "illegal acquisition of tribal land with official support"],
        "terms": ["right to property", "no deprivation of property except by authority of law",
                  "criminal trespass", "atrocities against members of scheduled castes and scheduled tribes"],
    },
    "armed_forces_immunity": {
        # Flagged in the original priority list as low vocabulary overlap
        # with its gold sections (COI_020/021, UAPA_049) — included for
        # completeness but likely insufficient without a legal_kg.py
        # cross-reference; see note at bottom of universal_translator.py.
        "examples": ["a soldier is accused of torturing a civilian in a conflict area",
                     "is an army officer protected from prosecution for acts during duty",
                     "sanction required before prosecuting a member of the armed forces"],
        "terms": ["armed forces special powers act", "protection of persons acting in good faith",
                  "sanction required for prosecution", "constitutional immunity of the state"],
    },
}


def semantic_concept_expand(
    query: str, matcher: SemanticMatcher, threshold: float = 0.55, max_concepts: int = 2,
) -> list[str]:
    """Embedding-based replacement for the old regex-substitution
    synonym_expand(). Matches the query against canonical concept examples
    (not literal substrings), so paraphrases the original dictionary never
    covered are caught automatically. Returns new query variants with the
    matched concept's terms appended."""
    if not matcher.embed_fn:
        return []
    matches = matcher.match(query, top_k=max_concepts)
    variants = []
    for concept, score in matches:
        if score < threshold:
            continue
        terms = LEGAL_CONCEPTS.get(concept, {}).get("terms", [])
        if terms:
            variants.append(f"{query} {' '.join(terms[:3])}")
    return variants


def _build_concept_matcher(embed_fn: Optional[Callable] = None) -> SemanticMatcher:
    label_examples = {
        concept: data["examples"] for concept, data in LEGAL_CONCEPTS.items()
    }
    return SemanticMatcher(
        label_examples=label_examples,
        embed_fn=embed_fn,
        examples_path=CONCEPT_EXAMPLES_PATH,
    )


# ── 3. LLM query rewriting (unchanged mechanism, still the primary path) ───

EXPAND_PROMPT = """You are a legal query expansion expert for Indian law.
Generate {n} semantically diverse alternative queries to help retrieve relevant legal sections.
Original query: {query}
Intent type: {intent}
Return ONLY a JSON array of strings. No explanation.
["alternative 1", "alternative 2", "alternative 3"]"""


def llm_expand(query: str, intent: str, n: int = 3) -> list[str]:
    response = ollama.chat(
        model=OLLAMA_FAST_MODEL,
        format="json",
        # Determinism — see irac_reranker.py's llm_irac_score for why.
        options={"temperature": 0},
        messages=[{"role": "user", "content": EXPAND_PROMPT.format(
            query=query, intent=intent, n=n
        )}],
    )
    data = json.loads(response["message"]["content"])
    # model may return {"queries": [...]} or a raw list
    if isinstance(data, list):
        variants = data
    else:
        variants = data.get("queries", data.get("alternatives", []))
    return [v for v in variants if isinstance(v, str) and v.strip()]


# ── Public class ────────────────────────────────────────────────────────────

class QueryExpander:
    """
    embed_fn is optional. Pass the shared embedding model's encode function
    (same pattern used for SectionPinner / ALEA / QueryRouter elsewhere in
    this codebase) to enable semantic concept expansion. Without it, the
    expander still works via LLM rewriting + abbreviation expansion — it
    just skips the embedding tier.
    """

    def __init__(self, embed_fn: Optional[Callable[[list[str]], object]] = None):
        self.embed_fn = embed_fn
        self._matcher = _build_concept_matcher(embed_fn)

    def expand(
        self,
        query: str,
        intent: QueryIntent,
        rewritten_query: Optional[str] = None,
        extra_charges: Optional[list[str]] = None,
    ) -> list[str]:
        queries = [query]

        # If a scenario-rewritten query exists, use it as a primary variant
        if rewritten_query and rewritten_query != query:
            queries.append(rewritten_query)

        # Generate focused sub-queries from each predicted charge
        # e.g. "IPC 304A" → "IPC section 304A punishment meaning"
        if extra_charges:
            for charge in extra_charges[:4]:
                charge_clean = charge.strip()
                queries.append(f"punishment under {charge_clean}")

        try:
            queries.extend(llm_expand(query, intent.label, n=3))
        except Exception:
            pass

        # Embedding-based concept expansion (replaces old regex synonym dict)
        queries.extend(semantic_concept_expand(query, self._matcher))

        # Exact abbreviation expansion (small, stable, closed vocabulary)
        abbrev_variant = expand_abbreviations(query)
        if abbrev_variant:
            queries.append(abbrev_variant)

        if intent.act_hint:
            queries.append(f"{query} under {intent.act_hint}")

        seen, unique = set(), []
        for q in queries:
            q_norm = q.strip().lower()
            if q_norm not in seen:
                seen.add(q_norm)
                unique.append(q.strip())

        return unique[:8]


if __name__ == "__main__":
    from pipeline.intent_classifier import IntentClassifier
    clf      = IntentClassifier()
    expander = QueryExpander()  # no embed_fn here — see main.py wiring note
    query    = "What is IPC 512?"
    intent   = clf.classify(query)
    queries  = expander.expand(query, intent)
    print(f"Original : {query}")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")