"""
pipeline/universal_translator.py
Universal legal translator — runs on EVERY query, not just scenarios.

Replaces the scenario_rewriter's conditional approach with an always-on
LLM-powered translation that handles all 7 query types:

  1. direct_section   "What does IPC 302 say?"
  2. describe_law     "Which law covers wrongful arrest?"
  3. scenario         "Ramesh faked his degree, patient died"
  4. civil_scenario   "Contract signed under coercion"
  5. constitutional   "Can the government detain without trial?"
  6. procedural       "How to file a bail application?"
  7. comparative      "Difference between 302 and 304?"
  8. multi_act        "Dalit woman cheated and land seized"

The output is a set of search-optimised sub-queries that maximise recall
from both BM25 (keyword) and dense (semantic) retrieval.
"""
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import OLLAMA_FAST_MODEL
import ollama


# ── Output structure ──────────────────────────────────────────────────────────

@dataclass
class TranslationResult:
    original:        str
    search_queries:  list[str]     # All queries to send to retrieval (BM25 + dense)
    primary_query:   str           # Single most important rewritten query
    legal_elements:  list[str]     # Legal concepts extracted
    predicted_sections: list[str]  # Specific section IDs predicted (e.g. IPC_304A)
    domain:          str           # criminal / civil / constitutional / procedural etc.
    dataset_gaps:    list[str]     # Acts referenced but not in dataset


# ── Hardcoded synonym rules (fast, offline, no LLM needed) ───────────────────
# These run first and always — covering the most common vocabulary mismatches.

QUICK_SYNONYMS: list[tuple[str, str]] = [
    # Deaths & injuries
    (r"\b(patient|person|victim)\s+died\b",
     "death caused negligence rash act culpable homicide IPC 304A"),
    (r"\bdeath\s+by\s+(accident|negligence|rash\s+driving)\b",
     "rash negligent act causing death IPC 304A"),
    # Extortion — checked before the generic beat/assault pattern below so
    # "beaten ... extracted money regularly" style facts still surface
    # extortion terms and aren't pushed toward plain hurt/assault alone.
    (r"\b(extort\w*|extract\w*\s+money|protection\s+money|"
     r"regular\s+payment\w*\s+under\s+threat|pay\w*\s+or\s+(else|face))\b",
     "extortion putting person in fear of injury dishonestly inducing "
     "delivery of property IPC 383 384 385 386 387"),
    (r"\b(beat|hit|assault\w*|attack\w*|stab\w*)\w*\b",
     "voluntarily causing hurt grievous hurt assault IPC 323 324 325"),
    (r"\bkilled?\b",
     "murder culpable homicide IPC 300 302 304"),

    # Fake credentials / impersonation / forged documents — merged so that
    # "forged certificate", "fake deed", "falsified diploma" etc. all match,
    # not just the two disjoint original noun sets.
    # diagnose_recall.py showed IPC_420 (cheating) still missing on top of
    # 463/468 when the forgery is used to obtain some benefit — forging is
    # essentially always instrumental to a cheating offence, so pull in
    # 415/420 alongside the forgery sections rather than relying on a
    # separate "obtained benefit" trigger.
    (r"\b(fake|bogus|forged?|fabricated?|falsified?)\s+(\w+\s+){0,2}"
     r"(degree|certificate|diploma|licen[cs]e|document|signature|will|deed|record)\b",
     "forged false document forgery for the purpose of cheating "
     "dishonestly inducing delivery of property IPC 463 464 468 415 420"),
    (r"\b(fake|bogus|false)\s+(doctor|physician|engineer|lawyer|officer|professional)\b",
     "cheating personation IPC 415 416 419"),
    (r"\bpracticed?\s+(medicine|law|engineering)\s+without\s+(licence|degree|registration)\b",
     "personation cheating false pretence IPC 416 419 420"),

    # Police / public servant
    (r"\bpolice\s+(beat|tortured?|assaulted?|thrashed?)\b",
     "public servant causing hurt wrongful confinement IPC 330 342 166"),
    (r"\bwrongful\s+(arrest|detention|confinement)\b",
     "wrongful confinement illegal arrest IPC 340 342 344 166"),
    (r"\b(police|officer)\s+(bribed?|took\s+money|corrupt)\b",
     "public servant gratification bribery PCA 13 IPC 161"),
    # diagnose_recall.py showed PCA_013 (criminal misconduct) still missing
    # next to PCA_007 — a demanded bribe is both "obtaining gratification"
    # (7) and "criminal misconduct ... by corrupt or illegal means" (13),
    # so spell out the section-13 language explicitly rather than relying
    # on the number alone to carry the match.
    (r"\bdemand\w*\s+(a\s+)?bribe\b",
     "public servant gratification bribery demand criminal misconduct "
     "obtains valuable thing pecuniary advantage by corrupt or illegal "
     "means PCA 7 13 IPC 161"),
    # diagnose_recall.py: IPC_211 now surfaces but IPC_166 (public servant
    # disobeying law with intent to cause injury) still never does, even
    # with "166" spelled out numerically — the section is about the
    # officer's misconduct, not the false charge itself, so give that
    # framing its own vocabulary instead of leaning on the number.
    (r"\b(fake|false)\s+(fir|case|complaint)\b",
     "false charge instituting false criminal proceedings public servant "
     "disobeying law with intent to cause injury to a person abuse of "
     "official position at the behest of a third party IPC 211 182 166"),
    (r"\b(police|officer)\s+(refused?|failed?|didn.?t)\s+(register|investigate|take\s+fir)\b",
     "public servant disobeying law failure investigate IPC 166A"),

    # Theft / fraud / cheating
    (r"\b(stole|stolen|theft|rob\w*)\b",
     "theft robbery stolen property IPC 378 379 390 392"),
    (r"\b(cheated?|scammed?|defrauded?|conned?)\b",
     "cheating dishonestly IPC 415 417 420"),
    (r"\bembezzl\w*\b",
     "criminal breach of trust misappropriation IPC 405 406 408 409"),
    # Breach of trust by an employee/servant who appropriates property over
    # time — catches "employee stole ... over months" style facts that used
    # to match only the plain theft pattern above (via "stole") because they
    # never use the literal word "embezzle".
    (r"(?=.*\b(employee|clerk|servant|cashier|accountant|manager|staff|agent)\b)"
     r"(?=.*\b(stole|stealing|took|misappropriat\w*|siphon\w*|diverted?)\b)"
     r"(?=.*\b(over\s+(the\s+)?(past\s+)?(several\s+)?(\d+\s+)?(months|weeks|years)|"
     r"repeatedly|regularly|multiple\s+times|for\s+months|ongoing|over\s+time)\b)",
     "criminal breach of trust by servant/agent entrusted with property "
     "IPC 405 406 408 409"),

    # Domestic / family
    (r"\bhusband\s+(beat\w*|abused?|tortured?|harass\w*)\s+(wife|her)\b",
     "cruelty husband wife IPC 498A domestic violence"),
    (r"\bdowry\s+(demand\w*|harass\w*|death)\b",
     "dowry death cruelty IPC 304B 498A"),
    (r"\bchild\s+(sexual\s+abuse|molest\w*|assault)\b",
     "POCSO sexual assault child minor penetrative assault"),

    # Cyber
    (r"\bhack\w+\b",
     "unauthorised computer access cybercrime ITA 66"),
    (r"\b(otp\s+fraud|sim\s+swap|phish\w*)\b",
     "identity theft impersonation electronic ITA 66C 66D"),
    (r"\b(revenge\s+porn|morphed?\s+photo|intimate\s+image)\b",
     "obscene publication without consent ITA 66E 67A"),

    # Contract / civil
    (r"\bcontract\s+(under\s+threat|under\s+coercion|forced?\s+to\s+sign)\b",
     "contract coercion voidable ICA 15 19"),
    (r"\b(breach\s+of\s+contract|not\s+performing|refused?\s+to\s+perform)\b",
     "breach of contract damages ICA 37 39 73 74"),
    (r"\bspecific\s+performance\b",
     "specific performance enforcement contract SRA 10"),
    (r"\bpromised?\s+to\s+(sell|transfer)\b.{0,60}\brefus\w*\b|"
     r"\b(force|compel)\w*\s+(him|her|them)\s+to\s+(sell|transfer)\b",
     "specific performance enforcement contract SRA 10"),

    # Property
    # diagnose_recall.py: TPA_108 (rights and liabilities of lessor and
    # lessee) was completely missing next to SRA_006 for a tenant with no
    # written agreement — add lease/tenancy vocabulary so long-tenancy /
    # no-written-agreement facts also reach the Transfer of Property Act.
    (r"\bevict\w*\b",
     "dispossession eviction possession lease lessor lessee tenancy "
     "holding over rights and liabilities of lessor and lessee SRA 6 "
     "TPA 108 IPC 441"),
    (r"\bproperty\s+(fraud|forged?\s+(deed|document|transfer))\b",
     "cheating forgery property TPA IPC 420 463"),

    # Constitutional
    (r"\bfundamental\s+rights?\b",
     "fundamental rights constitution article COI"),
    (r"\barticle\s+(21|22|19|14|32)\b",
     "fundamental rights constitutional provision COI"),
    (r"\b(detain\w*|arrest\w*)\s+without\s+trial\b",
     "preventive detention constitutional right article 22 COI"),

    # SC/ST
    (r"\b(caste\s+(abuse|insult|atrocity|discrimination)|untouchab\w*)\b",
     "atrocity scheduled caste tribe SCST prevention atrocities"),

    # Child labour — GAP: no rule existed here at all before this fix, so
    # this query type had zero deterministic recall safety net (unlike
    # every other scenario in this list) and depended entirely on the local
    # LLM's own legal recall for both search_queries AND predicted_sections.
    # That's why COI_024 (famous, textbook-level — Article 24) came through
    # but IPC_370A (employing a trafficked child, the section that most
    # directly punishes child-labour EMPLOYMENT) was consistently missed:
    # the LLM has no few-shot example anywhere in this file demonstrating a
    # labour/employment/trafficking scenario, only sexual-abuse and
    # kidnapping-for-begging ones, so its own associations pulled toward
    # those adjacent-but-wrong sections instead.
    (r"\bchild\s+labou?r\b|"
     r"\b(employ\w*|hir\w*|engag\w*|mak\w*|forc\w*)\b.{0,40}"
     r"\b(child|children|minor|underage)\b.{0,40}"
     r"\b(work|job|labou?r|factory|mine|hazardous)\b|"
     # numeric age phrasing without an explicit "child/minor" token, e.g.
     # "a 12 year old was made to work" — capped at 1-17 (under 18), same
     # convention the child-marriage rule above uses for the same reason,
     # so this doesn't misfire on an adult's age being mentioned near "work".
     # The \b after the digit group is required (not decorative): without
     # it, [0-7] alone would match just the leading digit of an unrelated
     # two-digit number like "45" and let the trailing .{0,50} swallow the
     # rest — silently firing on any adult age that happens to start with
     # 0-7 (40s, 50s, 60s, 70s).
     r"\b(1[0-7]|[1-9])\b\s*years?\s+old\b.{0,50}\b(work\w*|job|employ\w*|labou?r)\b|"
     r"\bchild\s+worker\b|\bworking\s+children\b",
     "child employed hazardous work factory mine constitutional prohibition "
     "COI 24 compulsory education COI 21A employing a trafficked child "
     "forced labour exploitation trafficking of minor for labour "
     "IPC 370 370A 374 BNS 143 144 146 "
     "[NOTE: Child Labour (Prohibition and Regulation) Act 1986 and "
     "Factories Act 1948]"),

    # Child marriage
    (r"\b(child|minor|underage)\s+marriage\b|\bmarri\w*.{0,25}\bunder\s+18\b|"
     r"\bmarried?\s+off\b.{0,25}\b(minor|underage|child)\b|"
     r"\bforce\w*\s+(her|him|them)?\s*(into\s+)?marriage\b|"
     r"\bmarri\w*.{0,20}\bat\s+1[0-7]\s+years?\b",
     "child marriage prohibition minor consent marriage POCSO"),

    # Aadhaar / data privacy
    # diagnose_recall.py: ITA_072A was surfacing but ITA_043A (compensation
    # for a body corporate's negligence in handling sensitive personal
    # data) wasn't — that's a distinct civil-liability provision from the
    # 72A penalty section, so it needs its own vocabulary.
    (r"\b(aadhaar|adhaar)\b|\bpersonal\s+data\b.{0,25}\bconsent\b|"
     r"\bsensitive\s+personal\s+data\b",
     "aadhaar data protection privacy consent personal data body corporate "
     "negligent in implementing reasonable security practices compensation "
     "for failure to protect data ITA 43A 72A"),

    # Will made when of unsound mind — diagnose_recall.py: ICA_012 surfaces
    # but ICA_011 (who is competent to contract) doesn't; state the
    # competency-to-contract framing directly instead of only "unsound mind".
    (r"\b(will|testament\w*)\b.{0,90}\bunsound\s+mind\b|"
     r"\bunsound\s+mind\b.{0,90}\b(will|testament\w*)\b",
     "testamentary capacity person of unsound mind is incompetent to "
     "contract who is competent to contract contractual competency ICA 11 12"),

    # Cartel / price-fixing — diagnose_recall.py showed the gold section is
    # ICA_023 (agreements with an unlawful object / opposed to public
    # policy are void), not a Competition Act section, so the expansion
    # needs Contract Act vocabulary, not just "cartel"/"anti-competitive".
    (r"\bcartel\w*\b|\bprice[- ]fix\w*\b|\banti[- ]?competitive\b",
     "cartel price fixing anti-competitive agreement collusion unlawful "
     "object or consideration agreement opposed to public policy void "
     "ICA 23"),

    # Toxic waste / pollution causing harm — diagnose_recall.py showed
    # IPC_284 (negligent conduct with respect to a poisonous substance)
    # missing alongside 277/278.
    (r"\b(toxic|poisonous|hazardous|chemical)\s+(waste|substance)\b|"
     r"\bpollut\w*\b.{0,40}\b(river|water|air|soil)\b",
     "negligent conduct with respect to poisonous substance fouling water "
     "public nuisance IPC 277 278 284"),

    # Hate speech / incitement against a community — diagnose_recall.py
    # showed IPC_153A, IPC_505, and UAPA_013 all completely missing (no
    # pattern existed for this scenario at all).
    (r"\b(hate\s+speech|incit\w*\s+(violence|hatred)|promot\w*\s+enmity|"
     r"speech\b.{0,40}\b(violence|hatred)\b.{0,40}\b(minority|community|group)\b|"
     r"\bcall\w*\s+for\s+violence\b.{0,40}\b(minority|community|group)\b)\b",
     "promoting enmity between different groups statements conducing to "
     "public mischief unlawful association IPC 153A 505 UAPA 13"),

    # Tribal land forcibly taken — diagnose_recall.py showed COI_300A,
    # IPC_447, and SCST_003 all completely missing (no pattern existed).
    (r"\btribal\b.{0,30}\bland\b|\b(forcibly?|illegal\w*)\s+acquir\w*\b.{0,30}\bland\b",
     "right to property no deprivation except by authority of law criminal "
     "trespass atrocities against members of scheduled castes and "
     "scheduled tribes COI 300A IPC 447 SCST"),

    # Armed-forces immunity from prosecution — flagged in the original
    # priority list as low vocabulary overlap with its gold sections
    # (COI_020/021, UAPA_049). This expansion is included for completeness
    # but is unlikely to be sufficient on its own; see note at the bottom
    # of this file.
    (r"\b(army|soldier|armed\s+forces)\b.{0,60}\b(immun\w*|protect\w*\s+from\s+"
     r"prosecution|sanction\s+for\s+prosecution|special\s+powers)\b",
     "armed forces special powers act protection of persons acting in good "
     "faith sanction required for prosecution constitutional immunity of "
     "the state COI 20 21 UAPA 49"),

    # Drugs
    (r"\b(drug\s+peddl\w*|narcotic\s+traffick\w*|possess\w*\s+(drugs|narcotics))\b",
     "NDPS narcotic substance possession trafficking"),

    # NI Act gap
    (r"\bcheque\s+(bounce|dishonour\w*|returned?\s+unpaid)\b",
     "cheating dishonestly IPC 420 "
     "[NOTE: Negotiable Instruments Act Section 138 (cheque dishonour)]"),
]


def quick_expand(query: str) -> str:
    """Append relevant legal synonyms to the query based on quick rules."""
    extras = []
    for pattern, legal_terms in QUICK_SYNONYMS:
        if re.search(pattern, query, re.IGNORECASE):
            extras.append(legal_terms)
    if extras:
        return query + " " + " ".join(extras)
    return query


# ── LLM universal translation prompt ─────────────────────────────────────────

TRANSLATE_PROMPT = """You are a legal query translator for Indian law.
Your job: take ANY user query and produce search sub-queries that will retrieve
the right sections from a legal database containing:
  IPC, BNS, CRPC, BNSS, IEA, BSA, ICA, CPC, SRA, TPA, COI, ITA, NDPS, PCA,
  POCSO, SCST, UAPA, LA (18 acts total).

QUERY TYPES AND HOW TO HANDLE EACH:

1. direct_section — User cites a section directly.
   → Keep the section reference + add the section's topic in plain legal terms
   → E.g. "IPC 302" → also add "murder intention to cause death punishment life"

2. describe_law — User describes a legal concept/situation, wants the section.
   → Rewrite as legal terminology that appears in the actual statute
   → E.g. "law about wrongful arrest" → "wrongful confinement illegal restraint IPC 340 342 166"

3. scenario — User describes facts, wants legal analysis.
   → Extract every criminal/civil element from the facts
   → Map each to a legal term + likely section IDs
   → E.g. "police beat someone in lockup" → "public servant causing hurt wrongful confinement IPC 330 342 166 PCA 13"

4. procedural — User asks how to do something legally.
   → Add CRPC/BNSS/CPC procedural terminology
   → E.g. "how to get bail" → "bail application CRPC 436 437 439 sessions court magistrate"

5. comparative — User asks difference between two things.
   → Include both concepts with their sections
   → E.g. "IPC 302 vs 304" → "murder intention IPC 302 culpable homicide not amounting to murder IPC 304"

6. constitutional — User asks about rights, government powers.
   → Include article numbers, COI terminology
   → E.g. "can government detain without trial" → "preventive detention article 22 COI personal liberty article 21"

7. multi_act — Facts touch multiple domains (e.g. tribal woman cheated + land seized).
   → Surface ALL relevant acts: IPC for fraud + SCST for atrocity + TPA for property

OUTPUT FORMAT — JSON only, no markdown, no explanation:
{{
  "primary_query": "<most important single search query in legal terms>",
  "search_queries": ["<query1>", "<query2>", "<query3>", ...],
  "legal_elements": ["<element1>", "<element2>", ...],
  "predicted_sections": ["IPC_304A", "IPC_416", ...],
  "domain": "<criminal|civil|constitutional|procedural|comparative|multi_act>",
  "gaps": ["<act name not in dataset, if any>"]
}}

Rules:
- search_queries: 3–6 short queries of 4–10 words each, varied vocabulary
- predicted_sections: use format ACTCODE_NUMBER (e.g. IPC_302, ICA_073, COI_021)
- gaps: only if the query requires an act NOT in the 18 listed (e.g. NI Act, Consumer Protection)
- NEVER fabricate section numbers you are not confident about

Examples:

Input: "What does IPC 302 say?"
Output: {{"primary_query": "murder punishment death IPC 302", "search_queries": ["IPC section 302 murder punishment", "intention cause death life imprisonment", "culpable homicide amounts to murder"], "legal_elements": ["murder", "intention to cause death", "punishment"], "predicted_sections": ["IPC_302", "IPC_300"], "domain": "criminal", "gaps": []}}

Input: "A police officer arrested Ravi without warrant and beat him in custody"
Output: {{"primary_query": "wrongful arrest public servant causing hurt IPC 342 330", "search_queries": ["wrongful confinement illegal arrest IPC 342 344", "public servant causing hurt IPC 330", "police officer abuse of power IPC 166", "custodial violence public servant"], "legal_elements": ["wrongful confinement", "causing hurt in custody", "abuse of power by public servant"], "predicted_sections": ["IPC_342", "IPC_330", "IPC_166", "IPC_344"], "domain": "criminal", "gaps": []}}

Input: "Can I cancel a contract I signed under threat?"
Output: {{"primary_query": "contract coercion voidable ICA 15 19", "search_queries": ["voidable contract coercion ICA 19", "contract formed under threat force", "rescission of contract ICA 64 65", "consent vitiated coercion undue influence ICA 15 16"], "legal_elements": ["coercion", "voidable contract", "rescission"], "predicted_sections": ["ICA_015", "ICA_019", "ICA_064"], "domain": "civil", "gaps": []}}

Input: "My cheque for 2 lakhs bounced, what can I do?"
Output: {{"primary_query": "cheque dishonour NI Act 138 cheating IPC 420", "search_queries": ["cheque dishonour punishment", "dishonestly inducing delivery of property IPC 420", "cheating by false representation"], "legal_elements": ["cheque dishonour", "cheating"], "predicted_sections": ["IPC_420"], "domain": "civil", "gaps": ["Negotiable Instruments Act Section 138 is not indexed in this dataset"]}}

Now translate:
Input: "{query}"
Output (JSON only):"""


def llm_translate(query: str) -> TranslationResult:
    response = ollama.chat(
        model=OLLAMA_FAST_MODEL,
        format="json",
        # Determinism — see irac_reranker.py's llm_irac_score for why.
        # This call decides search_queries, i.e. what reaches retrieval
        # at all, so it's the other high-leverage place to pin.
        options={"temperature": 0},
        messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(query=query)}],
    )
    raw  = response["message"]["content"].strip()
    data = json.loads(raw)

    primary   = data.get("primary_query", query)
    queries   = data.get("search_queries", [primary])
    elements  = data.get("legal_elements", [])
    sections  = data.get("predicted_sections", [])
    domain    = data.get("domain", "criminal")
    gaps      = data.get("gaps", [])

    return TranslationResult(
        original=query,
        search_queries=[query] + [q for q in queries if q != query],
        primary_query=primary,
        legal_elements=elements,
        predicted_sections=sections,
        domain=domain,
        dataset_gaps=gaps,
    )


# ── Main class ────────────────────────────────────────────────────────────────

class UniversalTranslator:
    """
    Translates any legal query into a set of BM25+dense-friendly search queries.

    Always runs — not conditional on query type.
    LLM is primary (rich, handles any domain).
    Quick-synonym rules are always applied as additional recall boosters.
    Falls back to rule-only if LLM is unavailable.
    """

    def translate(self, query: str) -> TranslationResult:
        # Always apply quick synonym expansion (offline, instant)
        rule_expanded = quick_expand(query)

        # Detect dataset-gap NOTEs from rules immediately.
        # BUGFIX: this used to only recognise the single literal NI Act
        # string, so any OTHER QUICK_SYNONYMS entry that embedded a
        # "[NOTE: ...]" gap marker (e.g. the child-labour rule added below,
        # which flags that the Child Labour Act / Factories Act aren't
        # indexed) would leave that bracketed text sitting inside
        # rule_expanded, unstripped — polluting the actual BM25/dense
        # search query with literal "[NOTE: ...]" text instead of being
        # surfaced as a warning. Generalised to strip and collect ANY
        # "[NOTE: ...]" marker, not just NI Act's.
        ni_gap = []
        for note_match in re.findall(r"\[NOTE:\s*(.*?)\]", rule_expanded):
            ni_gap.append(f"{note_match} is not indexed in this dataset.")
        rule_expanded = re.sub(r"\s*\[NOTE:\s*.*?\]", "", rule_expanded).strip()

        try:
            result = llm_translate(query)
            result.search_queries = list(dict.fromkeys(result.search_queries))
            # BUGFIX: this used to append rule_expanded and THEN slice to
            # [:8] — so whenever the LLM returned close to its max of ~6
            # sub-queries (giving [query] + LLM queries = 7-8 entries
            # already), rule_expanded landed past the cap and was silently
            # dropped. diagnose_recall.py showed several sections
            # (IPC_420, IPC_166, PCA_013, etc.) "never surfaced at any
            # stage" even though the QUICK_SYNONYMS regex demonstrably
            # matched and produced the right terms — this truncation-
            # before-append ordering is why. Trim the LLM's own queries
            # first, then guarantee rule_expanded a slot as the 8th entry
            # so the offline regex tier — the recall safety net — always
            # reaches retrieval.
            if rule_expanded != query:
                result.search_queries = result.search_queries[:7]
                if rule_expanded not in result.search_queries:
                    result.search_queries.append(rule_expanded)
            else:
                result.search_queries = result.search_queries[:8]
            result.dataset_gaps   = list(set(result.dataset_gaps + ni_gap))
            return result

        except Exception:
            # LLM unavailable — rule-only fallback
            queries = [query]
            if rule_expanded != query:
                queries.append(rule_expanded)
            # Add section-based sub-queries from any section IDs in rule expansion
            section_refs = re.findall(r'\b(IPC|ICA|SRA|TPA|COI|ITA|NDPS|PCA|POCSO|SCST|CRPC|BNSS|BNS|CPC|IEA|BSA|UAPA)\s+(\d+[A-Z]?)\b',
                                      rule_expanded, re.IGNORECASE)
            for act, num in section_refs[:4]:
                queries.append(f"section {num} {act.upper()} punishment")

            return TranslationResult(
                original=query,
                search_queries=queries,
                primary_query=rule_expanded,
                legal_elements=[],
                predicted_sections=[],
                domain="criminal",
                dataset_gaps=ni_gap,
            )


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    translator = UniversalTranslator()
    tests = [
        "What does IPC 302 say?",
        "Ramesh used a fake degree and practiced medicine, patient died",
        "A police officer arrested Ravi without reason and beat him in custody",
        "I signed a contract under threat and now the other party refuses to perform",
        "Can the government detain someone without trial under Article 22?",
        "My cheque for 5 lakhs bounced, what can I do?",
        "How do I file a bail application in a sessions court?",
        "What is the difference between culpable homicide and murder?",
        "A Dalit woman was defrauded by a contractor who also seized her land",
        "A hacker stole my OTP and transferred money from my bank",
        "Which section deals with wrongful confinement?",
        "An employer refused to pay wages for 3 months",
    ]
    for q in tests:
        r = translator.translate(q)
        print(f"\nQ: {q[:65]}")
        print(f"  domain   : {r.domain}")
        print(f"  primary  : {r.primary_query[:80]}")
        print(f"  queries  : {len(r.search_queries)} variants")
        print(f"  sections : {r.predicted_sections[:6]}")
        if r.dataset_gaps:
            print(f"  ⚠ GAP    : {r.dataset_gaps}")