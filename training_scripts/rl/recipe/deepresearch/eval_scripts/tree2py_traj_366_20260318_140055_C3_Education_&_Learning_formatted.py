import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dean_chancellor_2026_identity"
TASK_DESCRIPTION = (
    "An individual was appointed to serve as president or chancellor of a major U.S. university, "
    "with the appointment becoming effective in 2026. Prior to this appointment, the individual served "
    "as chancellor of a different university starting in August 2022. Before becoming a chancellor, this "
    "individual served as the dean of a professional school at a major research university for at least 7 years, "
    "with their deanship ending in 2022. Identify this individual and provide the following information about their "
    "position as dean: (1) The name of the university where they served as dean, (2) The complete official name of "
    "the professional school they led, (3) The month and year they began serving as dean, (4) The month and year "
    "their tenure as dean ended, (5) The name of the university where they were appointed as president or chancellor "
    "in 2026, (6) The exact date (month, day, and year) when they officially began their new role as president or chancellor."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CareerExtraction(BaseModel):
    # Identity
    name: Optional[str] = None

    # Dean role information
    dean_university: Optional[str] = None
    professional_school_official_name: Optional[str] = None

    dean_start_month: Optional[str] = None         # e.g., "July"
    dean_start_year: Optional[str] = None          # e.g., "2015"
    dean_end_month: Optional[str] = None           # e.g., "June"
    dean_end_year: Optional[str] = None            # e.g., "2022"

    # Chancellor role starting Aug 2022 (different university)
    chancellor_university: Optional[str] = None
    chancellor_start_month: Optional[str] = None   # should be "August"
    chancellor_start_year: Optional[str] = None    # should be "2022"

    # New appointment (effective 2026)
    new_university_name: Optional[str] = None
    new_role_start_date_full: Optional[str] = None   # e.g., "July 1, 2026"
    new_role_start_month: Optional[str] = None
    new_role_start_day: Optional[str] = None
    new_role_start_year: Optional[str] = None

    # Per-claim source URLs extracted from the answer text
    sources_dean_university: List[str] = Field(default_factory=list)
    sources_school_name: List[str] = Field(default_factory=list)
    sources_dean_start: List[str] = Field(default_factory=list)
    sources_dean_end: List[str] = Field(default_factory=list)
    sources_aug_2022_chancellor: List[str] = Field(default_factory=list)
    sources_new_university: List[str] = Field(default_factory=list)
    sources_new_role_start_date: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_career_info() -> str:
    return """
Extract the individual and all requested career details from the answer text. Return JSON with the following fields (use null for any missing field; for sources, return an empty array if none were cited). Extract URLs exactly as presented in the answer (plain links or markdown). Do not invent information.

Required fields:
- name: The individual's full name.

Dean role (prior to the chancellorship):
- dean_university: The university where they served as dean.
- professional_school_official_name: The complete official name of the professional school they led (e.g., 'School of Law', 'Graduate School of Education', 'School of Public Policy', 'College of Engineering', 'School of Public Health', etc.).
- dean_start_month: The month they began serving as dean (e.g., 'July', allow abbreviations like 'Jul.' but keep as in the answer).
- dean_start_year: The year they began serving as dean (four digits if available).
- dean_end_month: The month their tenure as dean ended.
- dean_end_year: The year their tenure as dean ended.

Chancellor role (starting in August 2022 at a different university):
- chancellor_university: The name of the university at which they served as chancellor starting in August 2022.
- chancellor_start_month: The start month (should be 'August' if given).
- chancellor_start_year: The start year (should be '2022').

New appointment effective in 2026:
- new_university_name: The university where they were appointed to serve as president or chancellor, effective in 2026.
- new_role_start_date_full: The exact official start date in 'Month Day, Year' format if present in the answer (e.g., 'July 1, 2026'; keep exact text as in the answer).
- new_role_start_month: The month component of the official start date, if available.
- new_role_start_day: The day component (numeric), if available.
- new_role_start_year: The year component (numeric), if available.

Per-claim source URLs (extract exactly as listed in the answer; include duplicates across multiple fields if the same URL supports multiple claims):
- sources_dean_university: URLs supporting that the individual served as dean at the specified university (any official university site or reputable news).
- sources_school_name: URLs supporting the complete official name of the professional school they led and their role as dean there.
- sources_dean_start: URLs supporting the dean start month/year.
- sources_dean_end: URLs supporting the dean end month/year.
- sources_aug_2022_chancellor: URLs supporting that the individual became chancellor starting in August 2022 at the stated university.
- sources_new_university: URLs supporting that the individual was appointed as president/chancellor at the new university effective in 2026.
- sources_new_role_start_date: URLs supporting the exact official start date (month, day, year) for the new role.

Rules:
- If the answer mentions a general 'Sources' section covering multiple claims, include those URLs in all relevant 'sources_*' arrays.
- Only extract URLs explicitly present in the answer text.
- Keep all strings exactly as written in the answer; do not normalize months or dates.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
_MONTH_PATTERN = r"(?:Jan(?:\.|uary)?|Feb(?:\.|ruary)?|Mar(?:\.|ch)?|Apr(?:\.|il)?|" \
                 r"May|Jun(?:\.|e)?|Jul(?:\.|y)?|Aug(?:\.|ust)?|Sep(?:\.|t(?:\.|ember)?)|" \
                 r"Oct(?:\.|ober)?|Nov(?:\.|ember)?|Dec(?:\.|ember)?)"


def is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def ensure_list(urls: Optional[List[str] | str]) -> List[str]:
    if urls is None:
        return []
    if isinstance(urls, list):
        return [u for u in urls if is_nonempty(u)]
    return [urls] if is_nonempty(urls) else []


def has_full_mdy_date(s: Optional[str]) -> bool:
    if not is_nonempty(s):
        return False
    text = s.strip()
    # Accept forms like "July 1, 2026" or "Jul. 1, 2026" (comma optional)
    pat1 = re.compile(rf"\b{_MONTH_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+20\d{{2}}\b", re.IGNORECASE)
    # Accept ISO-style "2026-07-01"
    pat2 = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
    # Accept "1 July 2026" style
    pat3 = re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_PATTERN}\s+20\d{{2}}\b", re.IGNORECASE)

    return bool(pat1.search(text) or pat2.search(text) or pat3.search(text))


def extract_year_from_text(s: Optional[str]) -> Optional[str]:
    if not is_nonempty(s):
        return None
    m = re.search(r"\b(20\d{2})\b", s)
    return m.group(1) if m else None


def pick_year_for_new_role(info: CareerExtraction) -> Optional[str]:
    # Prefer explicit field; fallback to parsing the full date
    if is_nonempty(info.new_role_start_year):
        return str(info.new_role_start_year).strip()
    return extract_year_from_text(info.new_role_start_date_full)


def pick_year_for_dean_end(info: CareerExtraction) -> Optional[str]:
    if is_nonempty(info.dean_end_year):
        return str(info.dean_end_year).strip()
    # Try to derive from a combined field if present
    # Here we don't have a combined field; attempt from month or name (unlikely)
    return None


def person_or_placeholder(name: Optional[str]) -> str:
    return name if is_nonempty(name) else "the identified individual"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_identify_individual(evaluator: Evaluator, parent) -> None:
    """
    Build and verify the 'identify_individual' subtree.
    """
    node = evaluator.add_parallel(
        id="identify_individual",
        desc="Correctly identify the individual who matches the specified career progression constraints",
        parent=parent,
        critical=True,
    )

    # Load extracted data (latest extraction result should be first/only entry of this type)
    # We'll retrieve from evaluator's recorded extractions by passing around the object instead:
    # Instead, we'll find it via a closure variable bound in the outer scope if needed.
    # Here, we assume caller passed the info in evaluator.add_custom_info or closure.
    # To keep it explicit, we'll attach info to node via evaluator.add_custom_info earlier in flow.
    # For simplicity, we will access it from the outer scope through a passed argument in actual calls.
    # In this function, we will re-extract it from the caller via a closure. To keep the signature simple,
    # we instead expect the caller to set evaluator._latest_info before calling this function. To avoid such
    # hacks, we pass info explicitly. So implement an overload below.
    pass  # This will be replaced by the overloaded version below


async def verify_identify_individual_with_info(evaluator: Evaluator, parent, info: CareerExtraction) -> None:
    identify = evaluator.add_parallel(
        id="identify_individual",
        desc="Correctly identify the individual who matches the specified career progression constraints",
        parent=parent,
        critical=True,
    )

    # 1) individual_named (existence check)
    evaluator.add_custom_node(
        result=is_nonempty(info.name),
        id="individual_named",
        desc="Provides the individual's full name",
        parent=identify,
        critical=True,
    )

    # 2) served_as_dean_professional_school_major_research_university
    # We primarily check that the school name indicates a professional school.
    dean_prof_leaf = evaluator.add_leaf(
        id="served_as_dean_professional_school_major_research_university",
        desc="The identified individual served as dean of a professional school at a major U.S. research university",
        parent=identify,
        critical=True,
    )
    school_name = info.professional_school_official_name or ""
    dean_univ = info.dean_university or ""
    claim_prof_school = (
        f"The school name '{school_name}' denotes a professional school (e.g., law, business, medicine, "
        f"public policy, public health, engineering, education, information, architecture, social work, "
        f"journalism, pharmacy, nursing, veterinary medicine, dentistry, public affairs)."
    )
    await evaluator.verify(
        claim=claim_prof_school,
        node=dean_prof_leaf,
        additional_instruction="Base your judgment only on the school name string. Answer Correct if it clearly represents a professional school.",
    )

    # 3) served_as_chancellor_starting_aug_2022
    ch_leaf = evaluator.add_leaf(
        id="served_as_chancellor_starting_aug_2022",
        desc="The identified individual served as chancellor of a university starting in August 2022",
        parent=identify,
        critical=True,
    )
    ch_start_m = info.chancellor_start_month or ""
    ch_start_y = info.chancellor_start_year or ""
    ch_claim = (
        f"The extracted chancellor start month/year indicates August 2022. "
        f"Month='{ch_start_m}', Year='{ch_start_y}'."
    )
    await evaluator.verify(
        claim=ch_claim,
        node=ch_leaf,
        additional_instruction="Return Correct only if the month is August (case-insensitive, allowing abbreviations like 'Aug.') and the year is 2022.",
    )

    # 4) appointed_president_or_chancellor_different_university
    diff_leaf = evaluator.add_leaf(
        id="appointed_president_or_chancellor_different_university",
        desc="The identified individual was appointed to serve as president or chancellor at a different university (i.e., not the Aug 2022 chancellor institution)",
        parent=identify,
        critical=True,
    )
    new_univ = info.new_university_name or ""
    ch_univ = info.chancellor_university or ""
    diff_claim = (
        f"The new appointment university '{new_univ}' is different from the "
        f"Aug 2022 chancellor's university '{ch_univ}'. Consider minor variations "
        f"in naming (e.g., including or excluding 'The', 'University of', etc.) as the same."
    )
    await evaluator.verify(
        claim=diff_claim,
        node=diff_leaf,
        additional_instruction="Return Correct only if these refer to different institutions; normalize trivial prefixes/suffixes.",
    )


async def verify_dean_details(evaluator: Evaluator, parent, info: CareerExtraction) -> None:
    dean = evaluator.add_parallel(
        id="dean_position_details",
        desc="Provide the required information about the individual's deanship",
        parent=parent,
        critical=True,
    )

    # Presence checks (critical)
    evaluator.add_custom_node(
        result=is_nonempty(info.dean_university),
        id="dean_university_name",
        desc="Provides the name of the university where they served as dean",
        parent=dean,
        critical=True,
    )

    evaluator.add_custom_node(
        result=is_nonempty(info.professional_school_official_name),
        id="professional_school_official_name",
        desc="Provides the complete official name of the professional school they led as dean",
        parent=dean,
        critical=True,
    )

    evaluator.add_custom_node(
        result=is_nonempty(info.dean_start_month) and is_nonempty(info.dean_start_year),
        id="dean_start_month_year",
        desc="Provides the month and year they began serving as dean",
        parent=dean,
        critical=True,
    )

    evaluator.add_custom_node(
        result=is_nonempty(info.dean_end_month) and is_nonempty(info.dean_end_year),
        id="dean_end_month_year",
        desc="Provides the month and year their tenure as dean ended",
        parent=dean,
        critical=True,
    )

    # Tenure at least 7 years (LLM logic check)
    tenure_leaf = evaluator.add_leaf(
        id="dean_tenure_at_least_7_years",
        desc="The deanship duration was at least 7 years",
        parent=dean,
        critical=True,
    )
    s_m = info.dean_start_month or ""
    s_y = info.dean_start_year or ""
    e_m = info.dean_end_month or ""
    e_y = info.dean_end_year or ""
    tenure_claim = (
        f"From '{s_m} {s_y}' to '{e_m} {e_y}', the elapsed time is at least 7 years "
        f"(allowing minor rounding if within ~1 month)."
    )
    await evaluator.verify(
        claim=tenure_claim,
        node=tenure_leaf,
        additional_instruction="Compute approximate duration between the two given dates. Consider 6 years 11 months as acceptable rounding to 7 years.",
    )

    # Deanship ended in 2022 (LLM logic check)
    end_2022_leaf = evaluator.add_leaf(
        id="deanship_ended_in_2022",
        desc="The individual's deanship ended in 2022",
        parent=dean,
        critical=True,
    )
    year_end = info.dean_end_year or ""
    ended_claim = f"The dean tenure ended in the calendar year 2022. Extracted end year: '{year_end}'."
    await evaluator.verify(
        claim=ended_claim,
        node=end_2022_leaf,
        additional_instruction="Return Correct only if the year is 2022.",
    )


async def verify_new_appointment(evaluator: Evaluator, parent, info: CareerExtraction) -> None:
    newapp = evaluator.add_parallel(
        id="new_appointment_details",
        desc="Provide the required information about the individual's 2026-effective president/chancellor appointment",
        parent=parent,
        critical=True,
    )

    # Presence of new university name
    evaluator.add_custom_node(
        result=is_nonempty(info.new_university_name),
        id="new_university_name",
        desc="Provides the name of the university where they were appointed as president or chancellor (the new institution)",
        parent=newapp,
        critical=True,
    )

    # Exact official start date presence (month-day-year)
    full_date_ok = has_full_mdy_date(info.new_role_start_date_full) or (
        is_nonempty(info.new_role_start_month)
        and is_nonempty(info.new_role_start_day)
        and is_nonempty(info.new_role_start_year)
    )
    evaluator.add_custom_node(
        result=bool(full_date_ok),
        id="official_start_date_exact",
        desc="Provides the exact official start date (month, day, and year) for the new president/chancellor role",
        parent=newapp,
        critical=True,
    )

    # Year must be 2026 (LLM logic check)
    start_yr_leaf = evaluator.add_leaf(
        id="start_date_year_is_2026",
        desc="The provided official start date is in the year 2026",
        parent=newapp,
        critical=True,
    )
    new_year = pick_year_for_new_role(info) or ""
    yr_claim = (
        f"The official start date for the new role is in 2026. "
        f"Extracted components: Month='{info.new_role_start_month or ''}', "
        f"Day='{info.new_role_start_day or ''}', Year='{new_year}', "
        f"Full='{info.new_role_start_date_full or ''}'."
    )
    await evaluator.verify(
        claim=yr_claim,
        node=start_yr_leaf,
        additional_instruction="Return Correct only if the year is 2026. If components are missing, judge using the provided full date string.",
    )


async def verify_sources(evaluator: Evaluator, parent, info: CareerExtraction) -> None:
    """
    Build and verify the 'source_verifiability' subtree.
    For each item, if no URLs were cited, immediately fail that leaf (critical).
    Otherwise, verify the claim against the cited URLs.
    """
    srcnode = evaluator.add_parallel(
        id="source_verifiability",
        desc="All provided information is verifiable via official university sources or reputable news outlets",
        parent=parent,
        critical=True,
    )

    # Helper to create a source-backed verification leaf or immediate fail if no URLs
    claims_and_sources: List[tuple[str, List[str], Any, Optional[str]]] = []

    def add_source_check(node_id: str, desc: str, claim: str, urls: List[str], add_ins: str) -> None:
        urls = ensure_list(urls)
        if len(urls) == 0:
            evaluator.add_custom_node(
                result=False,
                id=node_id,
                desc=desc,
                parent=srcnode,
                critical=True,
            )
        else:
            leaf = evaluator.add_leaf(
                id=node_id,
                desc=desc,
                parent=srcnode,
                critical=True,
            )
            claims_and_sources.append((claim, urls, leaf, add_ins))

    person = person_or_placeholder(info.name)

    # 1) Dean university name support
    add_source_check(
        node_id="sources_support_dean_university_name",
        desc="Cites at least one official university source or reputable news outlet supporting the university where the individual served as dean",
        claim=f"{person} served as dean at {info.dean_university or ''}.",
        urls=info.sources_dean_university,
        add_ins="Look for explicit mentions that the person held the title 'dean' at the specified university.",
    )

    # 2) Professional school name support
    add_source_check(
        node_id="sources_support_professional_school_name",
        desc="Cites at least one official university source or reputable news outlet supporting the official name of the professional school the individual led as dean",
        claim=f"{person} served as dean of the '{info.professional_school_official_name or ''}' at {info.dean_university or ''}.",
        urls=info.sources_school_name,
        add_ins="Check that the exact or clearly equivalent school name and the role of dean are stated.",
    )

    # 3) Dean start date support
    add_source_check(
        node_id="sources_support_dean_start_date",
        desc="Cites at least one official university source or reputable news outlet supporting the month and year the individual began serving as dean",
        claim=f"{person} began serving as dean in {info.dean_start_month or ''} {info.dean_start_year or ''}.",
        urls=info.sources_dean_start,
        add_ins="Confirm the starting month and year; statements like 'effective July 2015' count.",
    )

    # 4) Dean end date support
    add_source_check(
        node_id="sources_support_dean_end_date",
        desc="Cites at least one official university source or reputable news outlet supporting the month and year the individual's tenure as dean ended",
        claim=f"{person}'s tenure as dean ended in {info.dean_end_month or ''} {info.dean_end_year or ''}.",
        urls=info.sources_dean_end,
        add_ins="Look for language such as 'stepped down', 'ended', 'served until', or 'through June 2022'.",
    )

    # 5) August 2022 chancellor start support
    add_source_check(
        node_id="sources_support_aug_2022_chancellor_start",
        desc="Cites at least one official university source or reputable news outlet supporting that the individual became chancellor starting in August 2022",
        claim=f"{person} began serving as chancellor at {info.chancellor_university or ''} in August 2022.",
        urls=info.sources_aug_2022_chancellor,
        add_ins="Look for phrases like 'named chancellor', 'assumed office', 'effective August 2022', etc.",
    )

    # 6) New appointment university (effective 2026)
    add_source_check(
        node_id="sources_support_new_appointment_university",
        desc="Cites at least one official university source or reputable news outlet supporting the university where the individual was appointed president/chancellor effective in 2026",
        claim=f"In 2026, {person} was appointed to serve as president or chancellor at {info.new_university_name or ''}.",
        urls=info.sources_new_university,
        add_ins="Accept press releases or reputable media that clearly state the appointment and the new university.",
    )

    # 7) Exact start date for new role
    start_full = info.new_role_start_date_full or ""
    add_source_check(
        node_id="sources_support_new_role_start_date",
        desc="Cites at least one official university source or reputable news outlet supporting the exact official start date (month, day, year) of the new president/chancellor role",
        claim=f"{person} officially began the new role on {start_full}.",
        urls=info.sources_new_role_start_date,
        add_ins="The date must match exactly or be a clear equivalent (e.g., 'effective July 1, 2026').",
    )

    # Execute all source verifications in parallel
    if claims_and_sources:
        await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the dean/chancellor/2026-appointment identification task.
    """
    # Initialize evaluator with a sequential root, to mirror the rubric's top-level sequencing
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # 1) Extraction
    info: CareerExtraction = await evaluator.extract(
        prompt=prompt_extract_career_info(),
        template_class=CareerExtraction,
        extraction_name="career_info",
    )

    # 2) Verification subtrees in rubric order
    await verify_identify_individual_with_info(evaluator, root, info)
    await verify_dean_details(evaluator, root, info)
    await verify_new_appointment(evaluator, root, info)
    await verify_sources(evaluator, root, info)

    # 3) Return summary
    return evaluator.get_summary()