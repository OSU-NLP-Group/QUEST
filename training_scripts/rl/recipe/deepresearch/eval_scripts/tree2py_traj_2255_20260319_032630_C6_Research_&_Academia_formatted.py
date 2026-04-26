import asyncio
import logging
import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ai_venues_selection_2025"
TASK_DESCRIPTION = """A researcher affiliated with a university ranked in the top 5 of the Times Higher Education World University Rankings 2026 is preparing to submit an artificial intelligence research paper in May 2025. The researcher needs to identify suitable publication venues that meet the following requirements:

1. Journal requirement: Identify one peer-reviewed academic journal that:
   - Has an impact factor of 40 or higher (according to the most recent Journal Citation Reports)
   - Accepts AI, machine learning, or related multidisciplinary research
   - Supports open access publication with Article Processing Charges (APCs) under $3,000

2. Conference requirement: Identify one academic conference that:
   - Is in the field of artificial intelligence or machine learning
   - Has a full paper submission deadline between June 1, 2025 and December 31, 2025
   - Supports open access publication with registration/publication fees under $3,000
   - Provide the conference location and dates

3. University context: State which top-5 ranked university the researcher is affiliated with, according to THE World University Rankings 2026.

For each venue (journal and conference), provide:
- The complete name of the venue
- Specific metric values (impact factor for journal, submission deadline for conference)
- Evidence of open access support and fee information
- Reference URLs that verify all claimed information
"""

DEADLINE_RANGE_START = date(2025, 6, 1)
DEADLINE_RANGE_END = date(2025, 12, 31)


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class JournalExtraction(BaseModel):
    journal_name: Optional[str] = None

    # Peer-review evidence
    peer_review_url: Optional[str] = None

    # Scope evidence (AI/ML or multidisciplinary scope)
    scope_url: Optional[str] = None

    # Impact factor value and JCR URL
    impact_factor_value: Optional[str] = None
    jcr_if_url: Optional[str] = None

    # Open access and APC
    open_access_supported_text: Optional[str] = None
    apc_value: Optional[str] = None
    oa_apc_url: Optional[str] = None


class ConferenceExtraction(BaseModel):
    conference_name: Optional[str] = None

    # Field evidence (AI/ML)
    field_url: Optional[str] = None

    # Submission deadline
    submission_deadline: Optional[str] = None
    deadline_url: Optional[str] = None

    # OA/proceedings and fees
    open_access_supported_text: Optional[str] = None
    fees_amount: Optional[str] = None
    oa_fees_url: Optional[str] = None

    # Location and dates
    location: Optional[str] = None
    dates: Optional[str] = None
    location_dates_url: Optional[str] = None


class UniversityExtraction(BaseModel):
    university_name: Optional[str] = None
    ranking_url: Optional[str] = None


class VenuesExtraction(BaseModel):
    journal: Optional[JournalExtraction] = None
    conference: Optional[ConferenceExtraction] = None
    university: Optional[UniversityExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract exactly one candidate journal, exactly one candidate conference, and exactly one university affiliation as presented in the answer. Populate the following JSON fields using only what is explicitly stated in the answer. If a field is not present in the answer, return null for that field (or an empty string is not allowed; use null).

Return a JSON object with this structure:

{
  "journal": {
    "journal_name": string | null,
    "peer_review_url": string | null,
    "scope_url": string | null,
    "impact_factor_value": string | null,   // e.g., "69.5" or "69.5 (2023 JCR)"
    "jcr_if_url": string | null,            // a URL that specifically supports the stated impact factor from the most recent JCR
    "open_access_supported_text": string | null, // any phrase in the answer indicating OA support (if any)
    "apc_value": string | null,             // e.g., "$2,900", "USD 2900 (excluding tax)"
    "oa_apc_url": string | null             // a URL that supports OA option and/or APC value
  },
  "conference": {
    "conference_name": string | null,
    "field_url": string | null,             // a URL showing the conference is AI/ML
    "submission_deadline": string | null,   // full paper submission deadline as written (e.g., "October 15, 2025")
    "deadline_url": string | null,          // URL confirming the submission deadline
    "open_access_supported_text": string | null, // any phrase in the answer indicating OA/proceedings availability
    "fees_amount": string | null,           // registration or publication fees mentioned (e.g., "$1,200 early, $1,500 regular")
    "oa_fees_url": string | null,           // URL confirming OA/proceedings availability and/or fee schedule
    "location": string | null,              // e.g., "Vancouver, Canada"
    "dates": string | null,                 // e.g., "Dec 7–13, 2025"
    "location_dates_url": string | null     // URL confirming location and dates
  },
  "university": {
    "university_name": string | null,
    "ranking_url": string | null            // URL confirming the university is top-5 in THE World University Rankings 2026
  }
}

Rules:
- Extract only URLs explicitly shown in the answer.
- Do not fabricate URLs. If a URL is missing in the answer, set the corresponding field to null.
- Preserve textual values exactly as they appear (even if approximate or with additional context).
- If the answer lists multiple options, pick the first one for journal, the first one for conference, and the first university mentioned that is top-5 (per the answer).
"""


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
def pick_first_url(*urls: Optional[str]) -> Optional[str]:
    for u in urls:
        if u and isinstance(u, str) and u.strip():
            return u.strip()
    return None


def extract_numbers(text: Optional[str]) -> List[float]:
    if not text:
        return []
    # Find numbers like 2,900 or 2900.50 or 1 200 or 2.900
    cleaned = text.replace("\u00A0", " ")  # non-breaking space
    pattern = r"(?<!\w)(\d{1,3}(?:[,\s.]\d{3})+|\d+)(?:\.\d+)?(?!\w)"
    nums = []
    for m in re.finditer(pattern, cleaned):
        s = m.group(0)
        # Normalize thousands separators
        s_norm = re.sub(r"[,\s\.](?=\d{3}\b)", "", s)  # remove thousands separators before groups of 3
        try:
            nums.append(float(s_norm))
        except Exception:
            continue
    return nums


def extract_first_float(text: Optional[str]) -> Optional[float]:
    nums = extract_numbers(text)
    return nums[0] if nums else None


def parse_date_safely(text_date: Optional[str]) -> Optional[date]:
    if not text_date or not isinstance(text_date, str):
        return None

    s = text_date.strip()
    # Try common formats
    fmts = [
        "%B %d, %Y",   # October 15, 2025
        "%b %d, %Y",   # Oct 15, 2025
        "%Y-%m-%d",    # 2025-10-15
        "%d %B %Y",    # 15 October 2025
        "%d %b %Y",    # 15 Oct 2025
        "%m/%d/%Y",    # 10/15/2025
        "%d/%m/%Y",    # 15/10/2025
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass

    # Try a flexible parse if dateutil is available
    try:
        import dateutil.parser as dparser  # type: ignore
        try:
            dt = dparser.parse(s, fuzzy=True, dayfirst=False)
            return dt.date()
        except Exception:
            # Try dayfirst True
            dt = dparser.parse(s, fuzzy=True, dayfirst=True)
            return dt.date()
    except Exception:
        pass

    return None


def in_deadline_range(d: Optional[date]) -> bool:
    if d is None:
        return False
    return DEADLINE_RANGE_START <= d <= DEADLINE_RANGE_END


def make_missing_url_instruction() -> str:
    return ("Important: This verification REQUIRES evidence from the provided URL. "
            "No URL was supplied or the URL is empty. Mark the claim as NOT SUPPORTED.")


# --------------------------------------------------------------------------- #
# Verification Subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_journal(evaluator: Evaluator, parent, journal: Optional[JournalExtraction]) -> None:
    node = evaluator.add_parallel(
        id="journal_venue",
        desc="Identify one peer-reviewed academic journal meeting all stated requirements, including metric values and URLs verifying required claims.",
        parent=parent,
        critical=True
    )

    jname = (journal.journal_name.strip() if journal and journal.journal_name else None)

    # journal_name (existence)
    evaluator.add_custom_node(
        result=bool(jname),
        id="journal_name",
        desc="Provide the complete name of the journal.",
        parent=node,
        critical=True
    )

    # journal_peer_reviewed (URL-based)
    peer_url = pick_first_url(journal.peer_review_url if journal else None,
                              journal.scope_url if journal else None,
                              journal.oa_apc_url if journal else None)
    peer_leaf = evaluator.add_leaf(
        id="journal_peer_reviewed",
        desc="Provide evidence (via a reference URL) that the journal is peer-reviewed.",
        parent=node,
        critical=True
    )
    claim_peer = f"The journal '{jname or 'the selected journal'}' is a peer-reviewed academic journal."
    await evaluator.verify(
        claim=claim_peer,
        node=peer_leaf,
        sources=peer_url,
        additional_instruction=make_missing_url_instruction() if not peer_url else
        "Verify that the page explicitly indicates the journal uses peer review (e.g., 'peer-reviewed', editorial review process, or standard scholarly journal peer-review policy)."
    )

    # journal_scope_ai_ml (URL-based)
    scope_url = pick_first_url(journal.scope_url if journal else None)
    scope_leaf = evaluator.add_leaf(
        id="journal_scope_ai_ml",
        desc="Provide evidence (via a reference URL) that the journal accepts AI, machine learning, or related multidisciplinary research.",
        parent=node,
        critical=True
    )
    claim_scope = ("The journal accepts submissions in artificial intelligence, machine learning, "
                   "or closely related multidisciplinary research areas.")
    await evaluator.verify(
        claim=claim_scope,
        node=scope_leaf,
        sources=scope_url,
        additional_instruction=make_missing_url_instruction() if not scope_url else
        "Check the journal aims/scope, topics, or instructions for authors to confirm AI/ML or related areas are within scope."
    )

    # journal_impact_factor (group)
    if_node = evaluator.add_parallel(
        id="journal_impact_factor",
        desc="State an impact factor value and verify it is ≥ 40 according to the most recent Journal Citation Reports (JCR), including a supporting URL.",
        parent=node,
        critical=True
    )

    # impact_factor_value_stated_and_threshold_met (custom check)
    if_value = extract_first_float(journal.impact_factor_value if journal else None)
    evaluator.add_custom_node(
        result=(if_value is not None and if_value >= 40.0),
        id="impact_factor_value_stated_and_threshold_met",
        desc="Impact factor value is stated and is ≥ 40.",
        parent=if_node,
        critical=True
    )

    # impact_factor_jcr_url_provided (URL-based)
    jcr_url = pick_first_url(journal.jcr_if_url if journal else None)
    if_jcr_leaf = evaluator.add_leaf(
        id="impact_factor_jcr_url_provided",
        desc="Provide a reference URL that supports the impact factor specifically via the most recent JCR.",
        parent=if_node,
        critical=True
    )
    claim_jcr = (f"The most recent Journal Citation Reports lists '{jname or 'the journal'}' "
                 f"with an Impact Factor around {if_value if if_value is not None else '[value stated in answer]'} "
                 f"(≥ 40).")
    await evaluator.verify(
        claim=claim_jcr,
        node=if_jcr_leaf,
        sources=jcr_url,
        additional_instruction=make_missing_url_instruction() if not jcr_url else
        "Confirm that this page is from Journal Citation Reports (or an official rehost summarizing JCR) and supports the stated Impact Factor for the journal."
    )

    # journal_open_access_apc (group)
    oa_node = evaluator.add_parallel(
        id="journal_open_access_apc",
        desc="Verify the journal supports open access publication and that the APC is under $3,000, including a supporting URL.",
        parent=node,
        critical=True
    )

    oa_apc_url = pick_first_url(journal.oa_apc_url if journal else None)

    # open_access_supported (URL-based)
    journal_oa_leaf = evaluator.add_leaf(
        id="journal_open_access_supported",
        desc="Evidence that the journal supports open access publication.",
        parent=oa_node,
        critical=True
    )
    claim_journal_oa = (f"The journal '{jname or 'the selected journal'}' offers an open access option "
                        f"(fully OA or hybrid) for publishing articles.")
    await evaluator.verify(
        claim=claim_journal_oa,
        node=journal_oa_leaf,
        sources=oa_apc_url,
        additional_instruction=make_missing_url_instruction() if not oa_apc_url else
        "Look for explicit statements about open access options (hybrid or gold OA), or APC information indicating OA availability."
    )

    # apc_value_stated_and_under_3000 (custom check)
    apc_val_num_list = extract_numbers(journal.apc_value if journal else None)
    apc_under_3000 = any(v < 3000 for v in apc_val_num_list) if apc_val_num_list else False
    evaluator.add_custom_node(
        result=apc_under_3000,
        id="journal_apc_value_under_3000",
        desc="APC amount is stated and is < $3,000.",
        parent=oa_node,
        critical=True
    )

    # oa_apc_url_provided (URL-based)
    apc_url_leaf = evaluator.add_leaf(
        id="journal_oa_apc_url_provided",
        desc="Provide a reference URL that verifies both the OA option and the APC amount (or fee schedule) for the journal.",
        parent=oa_node,
        critical=True
    )
    claim_apc = (f"The open access Article Processing Charge (APC) for '{jname or 'the journal'}' is "
                 f"approximately {journal.apc_value if journal and journal.apc_value else '[APC value from answer]'}, "
                 f"which is under $3,000.")
    await evaluator.verify(
        claim=claim_apc,
        node=apc_url_leaf,
        sources=oa_apc_url,
        additional_instruction=make_missing_url_instruction() if not oa_apc_url else
        "Verify that the page explicitly mentions the APC amount (or fee schedule) and that it is under $3,000 USD (or clearly below $3,000 if currency is different)."
    )


async def verify_conference(evaluator: Evaluator, parent, conf: Optional[ConferenceExtraction]) -> None:
    node = evaluator.add_parallel(
        id="conference_venue",
        desc="Identify one AI/ML conference meeting all stated requirements, including deadline, fees, location/dates, and URLs verifying required claims.",
        parent=parent,
        critical=True
    )

    cname = (conf.conference_name.strip() if conf and conf.conference_name else None)

    # conference_name (existence)
    evaluator.add_custom_node(
        result=bool(cname),
        id="conference_name",
        desc="Provide the complete name of the conference.",
        parent=node,
        critical=True
    )

    # conference_field_ai_ml (URL-based)
    field_url = pick_first_url(conf.field_url if conf else None)
    field_leaf = evaluator.add_leaf(
        id="conference_field_ai_ml",
        desc="Provide evidence (via a reference URL) that the conference is in artificial intelligence or machine learning.",
        parent=node,
        critical=True
    )
    claim_field = (f"The conference '{cname or 'the selected conference'}' is a conference in artificial intelligence "
                   f"or machine learning.")
    await evaluator.verify(
        claim=claim_field,
        node=field_leaf,
        sources=field_url,
        additional_instruction=make_missing_url_instruction() if not field_url else
        "Check the conference scope, tracks, or description to confirm it is an AI/ML conference."
    )

    # conference_submission_deadline (group)
    dl_node = evaluator.add_parallel(
        id="conference_submission_deadline",
        desc="State and verify (via a reference URL) a full paper submission deadline between June 1, 2025 and December 31, 2025.",
        parent=node,
        critical=True
    )

    # deadline_stated_and_in_range (custom)
    d_parsed = parse_date_safely(conf.submission_deadline if conf else None)
    evaluator.add_custom_node(
        result=(in_deadline_range(d_parsed)),
        id="conference_deadline_in_range",
        desc="Full paper submission deadline is stated and falls within June 1, 2025–Dec 31, 2025.",
        parent=dl_node,
        critical=True
    )

    # deadline_url_provided (URL-based)
    d_url = pick_first_url(conf.deadline_url if conf else None)
    d_leaf = evaluator.add_leaf(
        id="conference_deadline_url_provided",
        desc="Provide a reference URL that confirms the full paper submission deadline.",
        parent=dl_node,
        critical=True
    )
    claim_deadline = (f"The full paper submission deadline for '{cname or 'the conference'}' is "
                      f"{conf.submission_deadline if conf and conf.submission_deadline else '[deadline from answer]'}, "
                      f"which falls between June 1, 2025 and December 31, 2025.")
    await evaluator.verify(
        claim=claim_deadline,
        node=d_leaf,
        sources=d_url,
        additional_instruction=make_missing_url_instruction() if not d_url else
        "Verify the posted full paper submission deadline (not abstract-only) and ensure it matches the stated date."
    )

    # conference_open_access_and_fees (group)
    cof_node = evaluator.add_parallel(
        id="conference_open_access_and_fees",
        desc="Verify the conference supports open access publication (or proceedings are freely available) and fees are under $3,000, including a supporting URL.",
        parent=node,
        critical=True
    )

    oa_fees_url = pick_first_url(conf.oa_fees_url if conf else None)

    # open_access_supported (URL-based)
    conf_oa_leaf = evaluator.add_leaf(
        id="conference_open_access_supported",
        desc="Evidence that the conference supports open access publication or makes proceedings freely available.",
        parent=cof_node,
        critical=True
    )
    claim_conf_oa = (f"The conference '{cname or 'the selected conference'}' makes its proceedings openly accessible "
                     f"(e.g., in an open-access proceedings series) or otherwise supports open access publication.")
    await evaluator.verify(
        claim=claim_conf_oa,
        node=conf_oa_leaf,
        sources=oa_fees_url,
        additional_instruction=make_missing_url_instruction() if not oa_fees_url else
        "Check whether proceedings are published in an open-access venue (e.g., PMLR, ACM Open, IEEE Open) or are freely available."
    )

    # fees_value_stated_and_under_3000 (custom)
    fees_vals = extract_numbers(conf.fees_amount if conf else None)
    fees_under_3000 = any(v < 3000 for v in fees_vals) if fees_vals else False
    evaluator.add_custom_node(
        result=fees_under_3000,
        id="conference_fees_under_3000",
        desc="Registration/publication fee amount is stated and is < $3,000.",
        parent=cof_node,
        critical=True
    )

    # fees_url_provided (URL-based)
    fees_leaf = evaluator.add_leaf(
        id="conference_fees_url_provided",
        desc="Provide a reference URL that verifies open access/proceedings availability and the fee amount (or fee schedule).",
        parent=cof_node,
        critical=True
    )
    claim_fees = (f"The registration/publication fees for '{cname or 'the conference'}' are "
                  f"{conf.fees_amount if conf and conf.fees_amount else '[fees from answer]'}, "
                  f"and are under $3,000.")
    await evaluator.verify(
        claim=claim_fees,
        node=fees_leaf,
        sources=oa_fees_url,
        additional_instruction=make_missing_url_instruction() if not oa_fees_url else
        "Verify that the page provides fee information (registration or publication) and that the typical required amount is under $3,000."
    )

    # conference_location_and_dates (group)
    loc_node = evaluator.add_parallel(
        id="conference_location_and_dates",
        desc="Provide the conference location and dates and verify them via a reference URL.",
        parent=node,
        critical=True
    )

    # location_stated (custom)
    evaluator.add_custom_node(
        result=bool(conf and conf.location and conf.location.strip()),
        id="conference_location_stated",
        desc="Conference location is stated.",
        parent=loc_node,
        critical=True
    )

    # dates_stated (custom)
    evaluator.add_custom_node(
        result=bool(conf and conf.dates and conf.dates.strip()),
        id="conference_dates_stated",
        desc="Conference dates are stated.",
        parent=loc_node,
        critical=True
    )

    # location_dates_url_provided (URL-based)
    ld_url = pick_first_url(conf.location_dates_url if conf else None)
    ld_leaf = evaluator.add_leaf(
        id="conference_location_dates_url_provided",
        desc="Provide a reference URL that confirms the conference location and dates.",
        parent=loc_node,
        critical=True
    )
    claim_ld = (f"The conference '{cname or 'the selected conference'}' will take place in "
                f"{conf.location if conf and conf.location else '[location from answer]'} "
                f"on {conf.dates if conf and conf.dates else '[dates from answer]'}.")
    await evaluator.verify(
        claim=claim_ld,
        node=ld_leaf,
        sources=ld_url,
        additional_instruction=make_missing_url_instruction() if not ld_url else
        "Verify both the city/country (or venue) and the date range on the official site or credible listing."
    )


async def verify_university(evaluator: Evaluator, parent, uni: Optional[UniversityExtraction]) -> None:
    node = evaluator.add_parallel(
        id="university_affiliation_context",
        desc="State and verify the researcher’s affiliation with a university ranked in the top 5 of THE World University Rankings 2026, including a verifying URL.",
        parent=parent,
        critical=True
    )

    uname = (uni.university_name.strip() if uni and uni.university_name else None)

    # university_name_stated (custom)
    evaluator.add_custom_node(
        result=bool(uname),
        id="university_name_stated",
        desc="University name is stated.",
        parent=node,
        critical=True
    )

    # university_top5_verified (URL-based)
    rk_url = pick_first_url(uni.ranking_url if uni else None)
    top5_leaf = evaluator.add_leaf(
        id="university_top5_verified",
        desc="University is verified (via evidence) to be ranked in the top 5 of THE World University Rankings 2026.",
        parent=node,
        critical=True
    )
    claim_top5 = (f"According to THE World University Rankings 2026, "
                  f"'{uname or 'the stated university'}' is ranked within the top 5.")
    await evaluator.verify(
        claim=claim_top5,
        node=top5_leaf,
        sources=rk_url,
        additional_instruction=make_missing_url_instruction() if not rk_url else
        "Verify on an official Times Higher Education (THE) ranking page for World University Rankings 2026 that the university is listed in the top 5 overall."
    )

    # university_ranking_url_provided (URL-based)
    rk2_leaf = evaluator.add_leaf(
        id="university_ranking_url_provided",
        desc="Provide a reference URL that confirms the university’s top-5 ranking in THE WUR 2026.",
        parent=node,
        critical=True
    )
    claim_ranking_page = "This page is a THE World University Rankings 2026 ranking page listing the top universities."
    await evaluator.verify(
        claim=claim_ranking_page,
        node=rk2_leaf,
        sources=rk_url,
        additional_instruction=make_missing_url_instruction() if not rk_url else
        "Confirm that the page is specifically about THE World University Rankings 2026 and shows the top positions."
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Add a critical task root (since Evaluator root is non-critical by default)
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Identify one qualifying journal, one qualifying conference, and a qualifying top-5 THE WUR 2026 university affiliation; provide required metric values, open-access/fee evidence, and reference URLs verifying each required claim.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Build and verify subtrees
    await verify_journal(evaluator, task_root, extracted.journal or JournalExtraction())
    await verify_conference(evaluator, task_root, extracted.conference or ConferenceExtraction())
    await verify_university(evaluator, task_root, extracted.university or UniversityExtraction())

    # Return evaluation summary
    return evaluator.get_summary()