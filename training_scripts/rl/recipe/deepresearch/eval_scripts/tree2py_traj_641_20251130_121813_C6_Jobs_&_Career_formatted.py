import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import re

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "power_conf_ad_openings_2025"
TASK_DESCRIPTION = (
    "Research the athletic director job market for Power conference schools in 2025. Identify TWO different Power "
    "conference schools (from Big Ten, SEC, ACC, Big 12, or Pac-12 conferences) that experienced athletic director "
    "openings or vacancies between June 2025 and November 2025. For each of the two schools, provide comprehensive "
    "documentation including: (1) the school's name, conference affiliation, and the exact date the position became "
    "vacant, (2) the name of the departing athletic director and the stated reason for departure (e.g., retired, "
    "resigned, dismissed, stepped down), (3) the typical minimum education and experience requirements for Power "
    "conference athletic director positions based on industry standards, and (4) URL references verifying all factual "
    "claims. All information must be supported by verifiable sources with URL references provided."
)

ALLOWED_CONFERENCES = {"Big Ten", "SEC", "ACC", "Big 12", "Pac-12"}
DATE_RANGE_START = datetime(2025, 6, 1)
DATE_RANGE_END = datetime(2025, 11, 30)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SchoolEntry(BaseModel):
    school_name: Optional[str] = None
    conference_affiliation: Optional[str] = None
    vacancy_date: Optional[str] = None  # Keep as string to be flexible (e.g., "June 12, 2025")
    departing_ad_name: Optional[str] = None
    departure_reason: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class SchoolsExtraction(BaseModel):
    schools: List[SchoolEntry] = Field(default_factory=list)


class QualificationExtraction(BaseModel):
    bachelors_required_claim: Optional[str] = None
    masters_preferred_with_80_percent_claim: Optional[str] = None
    min_experience_3_4_years_claim: Optional[str] = None
    typical_experience_6_10_years_claim: Optional[str] = None
    qualification_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_schools() -> str:
    return (
        "Extract ALL schools the answer claims had an athletic director opening or vacancy. For each school, return:\n"
        "- school_name: The institution's name as written in the answer.\n"
        "- conference_affiliation: The conference stated (e.g., Big Ten, SEC, ACC, Big 12, Pac-12).\n"
        "- vacancy_date: The exact date the position became vacant (effective or announcement date as claimed), as written.\n"
        "- departing_ad_name: The name of the departing athletic director.\n"
        "- departure_reason: A short phrase summarizing the stated reason (e.g., retired, resigned, dismissed, stepped down).\n"
        "- source_urls: All URLs that the answer cites for this school's vacancy date, AD name, and departure reason. "
        "Extract actual URLs (plain URLs or markdown links).\n"
        "Important:\n"
        "- Extract exactly what the answer states (do not infer or add).\n"
        "- Include all mentioned sources for each school; return full URLs.\n"
        "- If any field is missing, set it to null. If no sources are provided, return an empty list."
    )


def prompt_extract_qualifications() -> str:
    return (
        "Extract the industry-standard qualification claims for Power-conference athletic director positions as stated "
        "in the answer. Return:\n"
        "- bachelors_required_claim: The statement about a bachelor's degree being required.\n"
        "- masters_preferred_with_80_percent_claim: The statement that a master's degree is strongly preferred AND that "
        "approximately 80% of Division I ADs hold advanced degrees.\n"
        "- min_experience_3_4_years_claim: The statement that ~3–4 years of athletic administration experience is a "
        "minimum requirement for college-level positions.\n"
        "- typical_experience_6_10_years_claim: The statement that ~6–10 years of experience is typical for Power "
        "conference AD positions.\n"
        "- qualification_urls: All URLs cited to support these qualification claims. Extract actual URLs.\n"
        "If any claim is missing, set it to null. If no URLs are provided, return an empty list."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_school_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return re.sub(r"[\s\W_]+", "", name.strip().lower())


def _parse_date_flexible(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    s = date_str.strip()
    fmts = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # Try to extract month name and day, year using regex and attempt normalization
    try:
        # Handle forms like "June 2025" (no day): default to first of month to be lenient
        m = re.match(r"^\s*([A-Za-z]+)\s+(\d{4})\s*$", s)
        if m:
            month_name, year = m.group(1), int(m.group(2))
            dt = datetime.strptime(f"{month_name} 1, {year}", "%B %d, %Y")
            return dt
    except Exception:
        pass
    return None


def _date_in_required_range(dt: Optional[datetime]) -> bool:
    if dt is None:
        return False
    return DATE_RANGE_START <= dt <= DATE_RANGE_END


def _reason_matches_allowed(reason: Optional[str]) -> bool:
    if not reason or not reason.strip():
        return False
    r = reason.strip().lower()
    # Allow core categories plus common synonyms
    keywords = [
        "retired", "retire",
        "resigned", "resign",
        "dismissed", "dismiss",
        "stepped down", "step down", "stepping down",
        "fired", "termination", "terminated",
        "parted ways",
    ]
    return any(k in r for k in keywords)


def _has_valid_sources(urls: List[str]) -> bool:
    if not urls:
        return False
    return any(u.strip().startswith(("http://", "https://")) for u in urls)


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_school_subtree(
    evaluator: Evaluator,
    parent_node,
    school: SchoolEntry,
    idx: int,
) -> None:
    """
    Build verification nodes for one school entry, following rubric structure.
    """
    school_node = evaluator.add_parallel(
        id=f"school_{idx+1}",
        desc=f"{'First' if idx == 0 else 'Second'} Power conference school with an AD opening in the specified timeframe",
        parent=parent_node,
        critical=False,  # Non-critical at school-level; critical checks inside
    )

    # Opening core (critical, parallel)
    opening_core = evaluator.add_parallel(
        id=f"school_{idx+1}_opening_core",
        desc=f"Core identification and opening/vacancy facts for school {idx+1}",
        parent=school_node,
        critical=True,
    )

    # School name provided (existence)
    evaluator.add_custom_node(
        result=bool(school.school_name and school.school_name.strip()),
        id=f"school_{idx+1}_school_name",
        desc=f"Provides the school name for school {idx+1}",
        parent=opening_core,
        critical=True,
    )

    # Conference affiliation provided and in allowed set
    conf_ok = bool(school.conference_affiliation and school.conference_affiliation.strip())
    conf_in_allowed = school.conference_affiliation.strip() in ALLOWED_CONFERENCES if conf_ok else False
    evaluator.add_custom_node(
        result=conf_ok and conf_in_allowed,
        id=f"school_{idx+1}_conference_affiliation",
        desc=f"Provides conference affiliation for school {idx+1}, and it is one of Big Ten, SEC, ACC, Big 12, or Pac-12",
        parent=opening_core,
        critical=True,
    )

    # Vacancy date exact and in range
    vac_dt = _parse_date_flexible(school.vacancy_date)
    evaluator.add_custom_node(
        result=bool(school.vacancy_date and school.vacancy_date.strip()) and _date_in_required_range(vac_dt),
        id=f"school_{idx+1}_vacancy_date_exact_and_in_range",
        desc=f"Provides the exact vacancy/opening date for school {idx+1}, and it falls between June 2025 and November 2025 (inclusive)",
        parent=opening_core,
        critical=True,
    )

    # Departing AD info (critical, parallel)
    departing_ad = evaluator.add_parallel(
        id=f"school_{idx+1}_departing_ad",
        desc=f"Departing athletic director information for school {idx+1}",
        parent=school_node,
        critical=True,
    )

    # Previous AD name provided (existence)
    evaluator.add_custom_node(
        result=bool(school.departing_ad_name and school.departing_ad_name.strip()),
        id=f"school_{idx+1}_previous_ad_name",
        desc=f"Provides the name of the departing/previous athletic director for school {idx+1}",
        parent=departing_ad,
        critical=True,
    )

    # Departure reason provided and matches allowed categories
    evaluator.add_custom_node(
        result=_reason_matches_allowed(school.departure_reason),
        id=f"school_{idx+1}_departure_reason",
        desc=f"Provides the stated reason for departure for school {idx+1} (e.g., retired, resigned, dismissed, stepped down)",
        parent=departing_ad,
        critical=True,
    )

    # Sources group (critical, parallel) — restructure to ensure single-step verifications
    sources_group = evaluator.add_parallel(
        id=f"school_{idx+1}_sources",
        desc=f"Provides verifiable URL references that collectively support the vacancy date, departing AD name, and departure reason claims for school {idx+1}",
        parent=school_node,
        critical=True,
    )

    # Sources presence check (critical)
    sources_present_node = evaluator.add_custom_node(
        result=_has_valid_sources(school.source_urls),
        id=f"school_{idx+1}_sources_present",
        desc=f"At least one verifiable URL reference is provided for school {idx+1}",
        parent=sources_group,
        critical=True,
    )

    # Sources support: vacancy date
    vac_support_leaf = evaluator.add_leaf(
        id=f"school_{idx+1}_sources_support_vacancy_date",
        desc=f"URL references support the claimed vacancy date for school {idx+1}",
        parent=sources_group,
        critical=True,
    )
    vac_claim_school = school.school_name or ""
    vac_claim_date = school.vacancy_date or ""
    vac_claim = (
        f"The athletic director position at {vac_claim_school} became vacant (opening) on {vac_claim_date}."
    )
    await evaluator.verify(
        claim=vac_claim,
        node=vac_support_leaf,
        sources=school.source_urls,
        additional_instruction=(
            "Verify the effective vacancy/opening date from the sources. Consider phrases like 'stepped down effective', "
            "'resignation effective', 'vacancy occurred', or 'announcement date' if explicitly tied to the vacancy. "
            "Minor date formatting differences are acceptable as long as the date itself matches."
        ),
        extra_prerequisites=[sources_present_node, evaluator.find_node(f"school_{idx+1}_vacancy_date_exact_and_in_range")],
    )

    # Sources support: departing AD name
    ad_name_support_leaf = evaluator.add_leaf(
        id=f"school_{idx+1}_sources_support_departing_ad_name",
        desc=f"URL references support the departing athletic director name for school {idx+1}",
        parent=sources_group,
        critical=True,
    )
    ad_claim_name = school.departing_ad_name or ""
    ad_claim = f"The departing athletic director for {vac_claim_school} was {ad_claim_name}."
    await evaluator.verify(
        claim=ad_claim,
        node=ad_name_support_leaf,
        sources=school.source_urls,
        additional_instruction=(
            "Confirm that the named individual was the athletic director who departed/stepped down/was dismissed. "
            "Allow minor name variations (e.g., middle initials, casing)."
        ),
        extra_prerequisites=[sources_present_node, evaluator.find_node(f"school_{idx+1}_previous_ad_name")],
    )

    # Sources support: departure reason
    reason_support_leaf = evaluator.add_leaf(
        id=f"school_{idx+1}_sources_support_departure_reason",
        desc=f"URL references support the stated departure reason for school {idx+1}",
        parent=sources_group,
        critical=True,
    )
    reason_text = school.departure_reason or ""
    reason_claim = (
        f"The departure reason for the athletic director {ad_claim_name} at {vac_claim_school} was '{reason_text}'."
    )
    await evaluator.verify(
        claim=reason_claim,
        node=reason_support_leaf,
        sources=school.source_urls,
        additional_instruction=(
            "Verify the stated departure reason (e.g., retired, resigned, dismissed, stepped down). "
            "Synonyms or paraphrases are acceptable if they clearly correspond to the stated category."
        ),
        extra_prerequisites=[sources_present_node, evaluator.find_node(f"school_{idx+1}_departure_reason")],
    )


async def build_industry_qualifications_subtree(
    evaluator: Evaluator,
    parent_node,
    qual: QualificationExtraction,
) -> None:
    """
    Build verification nodes for industry-standard qualifications with sources.
    """
    qual_node = evaluator.add_parallel(
        id="industry_standard_qualifications",
        desc="Typical minimum education and experience requirements for Power-conference AD positions (industry standards), with sources",
        parent=parent_node,
        critical=True,
    )

    # URLs presence as a critical gate
    urls_present_node = evaluator.add_custom_node(
        result=_has_valid_sources(qual.qualification_urls),
        id="qualification_standards_urls",
        desc="Provides verifiable URL reference(s) supporting the industry qualification claims",
        parent=qual_node,
        critical=True,
    )

    # Bachelor's degree required
    bachelors_leaf = evaluator.add_leaf(
        id="bachelors_required_claim",
        desc="States that a bachelor's degree is required for Power conference AD positions",
        parent=qual_node,
        critical=True,
    )
    bachelors_claim = (
        qual.bachelors_required_claim
        if qual.bachelors_required_claim
        else "A bachelor's degree is required for Power conference athletic director positions."
    )
    await evaluator.verify(
        claim=bachelors_claim,
        node=bachelors_leaf,
        sources=qual.qualification_urls,
        additional_instruction=(
            "Evaluate whether the provided sources support that a bachelor’s degree is a typical minimum requirement "
            "for athletic director roles at Power-conference institutions."
        ),
        extra_prerequisites=[urls_present_node],
    )

    # Master's preferred with ~80% advanced degrees claim
    masters_leaf = evaluator.add_leaf(
        id="masters_preferred_with_80_percent_claim",
        desc="States that a master's degree is strongly preferred AND includes the ~80% of Division I ADs holding advanced degrees claim",
        parent=qual_node,
        critical=True,
    )
    masters_claim = (
        qual.masters_preferred_with_80_percent_claim
        if qual.masters_preferred_with_80_percent_claim
        else "A master’s degree is strongly preferred for Power-conference AD positions, and approximately 80% of Division I ADs hold advanced degrees."
    )
    await evaluator.verify(
        claim=masters_claim,
        node=masters_leaf,
        sources=qual.qualification_urls,
        additional_instruction=(
            "Check that sources indicate both (1) strong preference for a master's degree, and "
            "(2) approximately 80% of Division I athletic directors hold advanced degrees (master's or higher). "
            "Reasonable approximations (e.g., ~78–82%) count as ~80%."
        ),
        extra_prerequisites=[urls_present_node],
    )

    # Minimum experience ~3–4 years
    min_exp_leaf = evaluator.add_leaf(
        id="min_experience_3_4_years_claim",
        desc="States that ~3–4 years of athletic administration experience is a minimum requirement for college-level positions",
        parent=qual_node,
        critical=True,
    )
    min_exp_claim = (
        qual.min_experience_3_4_years_claim
        if qual.min_experience_3_4_years_claim
        else "Around 3–4 years of athletic administration experience is a minimum requirement for college-level positions."
    )
    await evaluator.verify(
        claim=min_exp_claim,
        node=min_exp_leaf,
        sources=qual.qualification_urls,
        additional_instruction=(
            "Verify that sources support ~3–4 years as a minimum baseline experience for college-level athletics administration roles."
        ),
        extra_prerequisites=[urls_present_node],
    )

    # Typical experience ~6–10 years
    typical_exp_leaf = evaluator.add_leaf(
        id="typical_experience_6_10_years_claim",
        desc="States that ~6–10 years of experience is typical for Power conference AD positions",
        parent=qual_node,
        critical=True,
    )
    typical_exp_claim = (
        qual.typical_experience_6_10_years_claim
        if qual.typical_experience_6_10_years_claim
        else "Approximately 6–10 years of experience is typical for Power-conference athletic director positions."
    )
    await evaluator.verify(
        claim=typical_exp_claim,
        node=typical_exp_leaf,
        sources=qual.qualification_urls,
        additional_instruction=(
            "Verify that sources support ~6–10 years of experience as typical for athletic director positions at Power-conference schools."
        ),
        extra_prerequisites=[urls_present_node],
    )


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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate the agent's answer for Power conference AD openings (2025).
    """
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
        default_model=model,
    )

    # Extract school and qualification info from the answer
    schools_extracted = await evaluator.extract(
        prompt=prompt_extract_schools(),
        template_class=SchoolsExtraction,
        extraction_name="schools_extraction",
    )
    quals_extracted = await evaluator.extract(
        prompt=prompt_extract_qualifications(),
        template_class=QualificationExtraction,
        extraction_name="qualifications_extraction",
    )

    # Add ground truth info for constraints (for transparency)
    evaluator.add_ground_truth({
        "allowed_conferences": list(ALLOWED_CONFERENCES),
        "date_range": {
            "start_inclusive": DATE_RANGE_START.strftime("%Y-%m-%d"),
            "end_inclusive": DATE_RANGE_END.strftime("%Y-%m-%d"),
        }
    }, gt_type="constraints_context")

    # Global constraints node (critical)
    global_constraints = evaluator.add_parallel(
        id="global_constraints",
        desc="Global task constraints about the set of schools provided",
        parent=root,
        critical=True,
    )

    # Exactly two schools (critical)
    exactly_two = evaluator.add_custom_node(
        result=len(schools_extracted.schools) == 2,
        id="exactly_two_schools",
        desc="Response identifies exactly two schools (no more, no fewer)",
        parent=global_constraints,
        critical=True,
    )

    # Distinct schools (critical) — compare the first two names if two provided
    if len(schools_extracted.schools) >= 2:
        s1 = _normalize_school_name(schools_extracted.schools[0].school_name)
        s2 = _normalize_school_name(schools_extracted.schools[1].school_name)
        are_distinct = bool(s1 and s2 and s1 != s2)
    else:
        are_distinct = False
    evaluator.add_custom_node(
        result=are_distinct,
        id="schools_are_distinct",
        desc="The two identified schools are different institutions",
        parent=global_constraints,
        critical=True,
    )

    # For detailed verification, consider only first two schools (padding with blanks if fewer)
    schools_for_check: List[SchoolEntry] = list(schools_extracted.schools[:2])
    while len(schools_for_check) < 2:
        schools_for_check.append(SchoolEntry())

    # Build per-school verification subtrees
    for idx, school in enumerate(schools_for_check):
        await build_school_subtree(evaluator, root, school, idx)

    # Industry-standard qualifications subtree
    await build_industry_qualifications_subtree(evaluator, root, quals_extracted)

    # Return structured summary
    return evaluator.get_summary()