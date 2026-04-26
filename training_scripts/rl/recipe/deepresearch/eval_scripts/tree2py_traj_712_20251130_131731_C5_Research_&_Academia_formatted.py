import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "schmidt_ai_science_host_selection"
TASK_DESCRIPTION = (
    "A postdoctoral researcher who received their PhD in molecular biology in June 2025 is seeking to apply for the "
    "Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship to pursue research at the intersection of AI and genomics. "
    "For family reasons, they wish to remain in either Canada or the northeastern United States. Identify one host institution "
    "of the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship that is located in either Ontario, Canada OR New York State, USA. "
    "For your identified institution, provide the following information with supporting reference URLs from official program sources: "
    "(1) The exact application deadline for the 2026 cohort (including date, time, and timezone), "
    "(2) The specific city and province/state where the institution is located, "
    "(3) The duration of the fellowship in years, "
    "(4) The PhD timing eligibility requirement (how recently must applicants have obtained their PhD), "
    "(5) The field requirement (which academic fields are eligible), and "
    "(6) Whether prior AI experience is required for applicants."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class InstitutionInfo(BaseModel):
    institution_name: Optional[str] = None
    city: Optional[str] = None
    province_or_state: Optional[str] = None
    country: Optional[str] = None
    program_affiliation_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)


class DeadlineInfo(BaseModel):
    deadline_date: Optional[str] = None
    deadline_time: Optional[str] = None
    deadline_timezone: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class FellowshipInfo(BaseModel):
    duration_years: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PhDEligibility(BaseModel):
    timing_requirement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class FieldEligibility(BaseModel):
    field_requirement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AIExperienceReq(BaseModel):
    requirement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class InstitutionPackage(BaseModel):
    institution: Optional[InstitutionInfo] = None
    deadline: Optional[DeadlineInfo] = None
    duration: Optional[FellowshipInfo] = None
    phd_timing: Optional[PhDEligibility] = None
    field: Optional[FieldEligibility] = None
    ai_experience: Optional[AIExperienceReq] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_institution_package() -> str:
    return """
    Extract details for one host institution of the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship as presented in the answer.
    You must extract EXACTLY as stated in the answer text and return null for any missing fields.

    1) institution:
       - institution_name: The name of the host institution selected.
       - city: The city of the host institution.
       - province_or_state: The province/state of the host institution.
       - country: The country of the host institution.
       - program_affiliation_urls: All URLs cited in the answer that directly confirm this institution is a host of the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship.
       - location_urls: All URLs cited in the answer that support the city/province/state location of the institution (can be official institution or program pages).

    2) deadline (for the 2026 cohort at the chosen institution):
       - deadline_date: The exact date provided in the answer.
       - deadline_time: The exact time provided in the answer (e.g., "11:59 PM").
       - deadline_timezone: The timezone provided in the answer (e.g., "ET", "Eastern Time", "EST/EDT", etc.).
       - urls: All URLs cited in the answer that support this exact deadline (date, time, and timezone).

    3) duration (fellowship duration at the chosen institution):
       - duration_years: The fellowship duration as stated in the answer (e.g., "2 years" or "2").
       - urls: All URLs cited in the answer supporting the duration.

    4) phd_timing (eligibility timing at the chosen institution):
       - timing_requirement: The PhD timing requirement stated in the answer (e.g., "within 3 years of obtaining PhD").
       - urls: All URLs cited in the answer supporting the timing requirement.

    5) field (eligible fields at the chosen institution):
       - field_requirement: The answer's statement about eligible fields (e.g., non-computer science areas such as natural sciences, engineering, mathematics).
       - urls: All URLs cited supporting the field eligibility requirement.

    6) ai_experience (requirement related to prior AI experience at the chosen institution):
       - requirement: The answer's statement (e.g., "prior AI experience is not required; applicants must demonstrate a desire to learn AI").
       - urls: All URLs cited supporting this requirement.

    IMPORTANT:
    - Extract only URLs explicitly present in the answer. Do not invent or infer any URLs.
    - If any field above is not mentioned in the answer, set it to null (or empty list for URL arrays).
    - Do not normalize or rewrite values; keep them as the answer states (even if approximate).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_region(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    mapping = {
        "ontario": "ontario",
        "on": "ontario",
        "new york": "new york",
        "ny": "new york",
        "n.y.": "new york"
    }
    return mapping.get(v, v)


def union_sources(pkg: InstitutionPackage) -> List[str]:
    urls: List[str] = []
    if pkg.institution:
        urls.extend(pkg.institution.program_affiliation_urls or [])
        urls.extend(pkg.institution.location_urls or [])
    if pkg.deadline:
        urls.extend(pkg.deadline.urls or [])
    if pkg.duration:
        urls.extend(pkg.duration.urls or [])
    if pkg.phd_timing:
        urls.extend(pkg.phd_timing.urls or [])
    if pkg.field:
        urls.extend(pkg.field.urls or [])
    if pkg.ai_experience:
        urls.extend(pkg.ai_experience.urls or [])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_institution_selection(evaluator: Evaluator, parent_node, pkg: InstitutionPackage) -> None:
    """
    Build and verify the Institution_Selection node:
      - Program_Affiliation (critical, verified via official URLs)
      - Geographic_Constraint (critical, custom check: Ontario (Canada) OR New York (USA))
    """
    inst = pkg.institution or InstitutionInfo()
    inst_node = evaluator.add_parallel(
        id="Institution_Selection",
        desc="Select one host institution that satisfies program-host and geography constraints.",
        parent=parent_node,
        critical=True
    )

    # Program_Affiliation
    prog_leaf = evaluator.add_leaf(
        id="Program_Affiliation",
        desc="Institution is a confirmed host of the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship program.",
        parent=inst_node,
        critical=True
    )
    claim_affil = f"{inst.institution_name or 'The institution'} is a host institution of the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship."
    await evaluator.verify(
        claim=claim_affil,
        node=prog_leaf,
        sources=inst.program_affiliation_urls,
        additional_instruction=(
            "Use only official program or host institution sources to confirm host status. Accept pages on the official "
            "Schmidt AI in Science program site or the host institution's official pages for the program. "
            "Ignore third-party news or blogs."
        ),
    )

    # Geographic_Constraint (Ontario, Canada OR New York State, USA)
    province_norm = normalize_region(inst.province_or_state)
    country_norm = (inst.country or "").strip().lower() if inst.country else ""
    is_ontario_canada = (province_norm == "ontario") and (country_norm in ("canada", "ca"))
    is_new_york_usa = (province_norm == "new york") and (country_norm in ("united states", "usa", "us", "u.s.", "u.s.a."))
    geo_ok = is_ontario_canada or is_new_york_usa

    evaluator.add_custom_node(
        result=geo_ok,
        id="Geographic_Constraint",
        desc="Institution is located in either Ontario, Canada OR New York State, USA.",
        parent=inst_node,
        critical=True
    )


async def verify_required_details(evaluator: Evaluator, parent_node, pkg: InstitutionPackage) -> None:
    """
    Build and verify Required_Details_For_Chosen_Institution node with its children.
    For each criterion, verify using official URLs attached to that claim when available.
    """
    inst = pkg.institution or InstitutionInfo()
    deadline = pkg.deadline or DeadlineInfo()
    duration = pkg.duration or FellowshipInfo()
    phd = pkg.phd_timing or PhDEligibility()
    field = pkg.field or FieldEligibility()
    ai = pkg.ai_experience or AIExperienceReq()

    details_node = evaluator.add_parallel(
        id="Required_Details_For_Chosen_Institution",
        desc="Provide all required details for the identified institution/program.",
        parent=parent_node,
        critical=False
    )

    # Application_Deadline_2026_Cohort: add a component existence check and the actual verification
    components_ok = bool((deadline.deadline_date or "").strip()) and bool((deadline.deadline_time or "").strip()) and bool((deadline.deadline_timezone or "").strip())
    evaluator.add_custom_node(
        result=components_ok,
        id="Application_Deadline_2026_Components_Provided",
        desc="Exact deadline components (date, time, timezone) are provided in the answer.",
        parent=details_node,
        critical=True
    )

    deadline_leaf = evaluator.add_leaf(
        id="Application_Deadline_2026_Cohort",
        desc="Provide the exact application deadline for the 2026 cohort, including date, time, and timezone.",
        parent=details_node,
        critical=True
    )
    claim_deadline = (
        f"The application deadline for the 2026 cohort at {inst.institution_name or 'the institution'} is "
        f"{deadline.deadline_date or '[missing date]'} at {deadline.deadline_time or '[missing time]'} "
        f"{deadline.deadline_timezone or '[missing timezone]'}."
    )
    await evaluator.verify(
        claim=claim_deadline,
        node=deadline_leaf,
        sources=deadline.urls,
        additional_instruction=(
            "Strictly check that the cited official program page(s) explicitly state the deadline date, time, and timezone "
            "for the 2026 cohort at the chosen institution. If any component is missing or the sources are non-official, fail."
        ),
    )

    # Institution_Location: verify city and province/state via official sources
    # First, ensure city/state present
    location_present = bool((inst.city or "").strip()) and bool((inst.province_or_state or "").strip())
    evaluator.add_custom_node(
        result=location_present,
        id="Institution_Location_Provided",
        desc="City and province/state are provided in the answer.",
        parent=details_node,
        critical=True
    )

    loc_leaf = evaluator.add_leaf(
        id="Institution_Location",
        desc="Provide the specific city and province/state where the host institution is located.",
        parent=details_node,
        critical=True
    )
    loc_sources = inst.location_urls if inst.location_urls else inst.program_affiliation_urls
    claim_loc = (
        f"The host institution {inst.institution_name or 'the institution'} is located in {inst.city or '[missing city]'}, "
        f"{inst.province_or_state or '[missing province/state]'}."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=loc_sources,
        additional_instruction=(
            "Confirm the city and province/state using official institution or official program pages. "
            "Ignore non-official sources."
        ),
    )

    # Fellowship_Duration: Split into value check and source support
    duration_value_leaf = evaluator.add_leaf(
        id="Fellowship_Duration_Value",
        desc="Answer states fellowship duration equals 2 years.",
        parent=details_node,
        critical=True
    )
    claim_duration_value = (
        f"The stated fellowship duration '{(duration.duration_years or '').strip()}' indicates that the program is 2 years."
    )
    await evaluator.verify(
        claim=claim_duration_value,
        node=duration_value_leaf,
        additional_instruction=(
            "Judge whether the provided text indicates a 2-year duration. Accept reasonable variants like '2 years', "
            "'two years', '2-year', 'two-year fellowship'."
        ),
    )

    duration_leaf = evaluator.add_leaf(
        id="Fellowship_Duration",
        desc="Correctly identify that the fellowship duration is 2 years.",
        parent=details_node,
        critical=True
    )
    claim_duration = "The fellowship duration is 2 years."
    await evaluator.verify(
        claim=claim_duration,
        node=duration_leaf,
        sources=duration.urls,
        additional_instruction=(
            "Verify on official program pages (host institution or central program) that the fellowship duration is explicitly 2 years."
        ),
    )

    # PhD_Timing_Eligibility: value check + supported by sources
    phd_value_leaf = evaluator.add_leaf(
        id="PhD_Timing_Eligibility_Value",
        desc="Answer states applicants must be within 3 years of obtaining PhD.",
        parent=details_node,
        critical=True
    )
    claim_phd_value = (
        f"The stated PhD timing eligibility '{(phd.timing_requirement or '').strip()}' indicates applicants must be within 3 years of obtaining their PhD."
    )
    await evaluator.verify(
        claim=claim_phd_value,
        node=phd_value_leaf,
        additional_instruction=(
            "Judge whether the text conveys 'within 3 years of PhD'. Accept minor phrasing variants like "
            "'no more than three years since PhD', 'PhD obtained within past three (3) years'."
        ),
    )

    phd_leaf = evaluator.add_leaf(
        id="PhD_Timing_Eligibility",
        desc="Correctly state that applicants must be within 3 years of obtaining their PhD degree.",
        parent=details_node,
        critical=True
    )
    claim_phd = "Applicants must be within 3 years of obtaining their PhD degree."
    await evaluator.verify(
        claim=claim_phd,
        node=phd_leaf,
        sources=phd.urls,
        additional_instruction=(
            "Verify on official program pages that the eligibility requires applicants to be within 3 years of receiving their PhD."
        ),
    )

    # Field_Requirement: value check + supported by sources
    field_value_leaf = evaluator.add_leaf(
        id="Field_Requirement_Value",
        desc="Answer states eligible PhDs are in non-computer science areas (e.g., natural sciences, engineering, mathematics).",
        parent=details_node,
        critical=True
    )
    claim_field_value = (
        f"The field eligibility statement '{(field.field_requirement or '').strip()}' indicates that applicants' PhDs must be in non-computer science areas, "
        "such as natural sciences, engineering, or mathematics."
    )
    await evaluator.verify(
        claim=claim_field_value,
        node=field_value_leaf,
        additional_instruction=(
            "Judge whether the text clearly excludes computer science and includes non-CS scientific fields like natural sciences, engineering, or mathematics."
        ),
    )

    field_leaf = evaluator.add_leaf(
        id="Field_Requirement",
        desc="Correctly state that eligible applicants’ PhDs must be in non-computer science areas (e.g., natural sciences, engineering, or mathematics).",
        parent=details_node,
        critical=True
    )
    claim_field = (
        "Eligible applicants’ PhDs must be in non-computer science areas (for example, natural sciences, engineering, or mathematics)."
    )
    await evaluator.verify(
        claim=claim_field,
        node=field_leaf,
        sources=field.urls,
        additional_instruction=(
            "Verify on official program pages that eligible fields exclude computer science and emphasize non-CS scientific disciplines."
        ),
    )

    # AI_Experience_Requirement: value check + supported by sources
    ai_value_leaf = evaluator.add_leaf(
        id="AI_Experience_Requirement_Value",
        desc="Answer states prior AI experience is not required but applicants must show desire to learn AI methods.",
        parent=details_node,
        critical=True
    )
    claim_ai_value = (
        f"The statement '{(ai.requirement or '').strip()}' indicates that prior AI experience is not required, but a desire to learn AI methods is required."
    )
    await evaluator.verify(
        claim=claim_ai_value,
        node=ai_value_leaf,
        additional_instruction=(
            "Judge whether the text conveys that prior AI experience is NOT required and applicants should demonstrate willingness to learn AI."
        ),
    )

    ai_leaf = evaluator.add_leaf(
        id="AI_Experience_Requirement",
        desc="Correctly state that prior AI experience is not required, but applicants must demonstrate a desire to learn AI methods.",
        parent=details_node,
        critical=True
    )
    claim_ai = "Prior AI experience is not required, but applicants must demonstrate a desire to learn AI methods."
    await evaluator.verify(
        claim=claim_ai,
        node=ai_leaf,
        sources=ai.urls,
        additional_instruction=(
            "Verify on official program pages that no prior AI experience is required and that applicants should show intent to learn AI methods."
        ),
    )


async def verify_official_sources(evaluator: Evaluator, parent_node, pkg: InstitutionPackage) -> None:
    """
    Add a critical check that all provided claims have at least one supporting URL from official program sources.
    We enforce existence by checking non-empty URL lists for each claim group.
    Officialness is further enforced via additional_instruction in each per-claim verification.
    """
    inst = pkg.institution or InstitutionInfo()
    deadline = pkg.deadline or DeadlineInfo()
    duration = pkg.duration or FellowshipInfo()
    phd = pkg.phd_timing or PhDEligibility()
    field = pkg.field or FieldEligibility()
    ai = pkg.ai_experience or AIExperienceReq()

    all_claims_have_urls = all([
        bool(inst.program_affiliation_urls),
        bool(inst.location_urls) or bool(inst.program_affiliation_urls),
        bool(deadline.urls),
        bool(duration.urls),
        bool(phd.urls),
        bool(field.urls),
        bool(ai.urls),
    ])

    evaluator.add_custom_node(
        result=all_claims_have_urls,
        id="Official_Source_URLs",
        desc="All provided claims (host status, deadline, location, duration, PhD timing eligibility, field eligibility, and AI-experience requirement) are supported by reference URL(s) from official program sources.",
        parent=parent_node,
        critical=True
    )

    # Record a summary of URLs checked for transparency
    evaluator.add_custom_info(
        info={
            "total_urls": len(union_sources(pkg)),
            "program_affiliation_urls": inst.program_affiliation_urls,
            "location_urls": inst.location_urls,
            "deadline_urls": deadline.urls,
            "duration_urls": duration.urls,
            "phd_timing_urls": phd.urls,
            "field_urls": field.urls,
            "ai_experience_urls": ai.urls,
        },
        info_type="url_summary",
        info_name="official_urls_summary"
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
) -> Dict:
    """
    Evaluate an answer for selecting a Schmidt AI in Science host institution in Ontario (Canada) or New York (USA)
    and verifying required program details for the 2026 cohort with official sources.
    """
    # Initialize evaluator
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
        default_model=model
    )

    # Extract the institution package from the answer
    pkg: InstitutionPackage = await evaluator.extract(
        prompt=prompt_extract_institution_package(),
        template_class=InstitutionPackage,
        extraction_name="institution_package"
    )

    # Build root-level node (non-critical) descriptive
    complete_node = evaluator.add_sequential(
        id="Complete_Task",
        desc="Identify an eligible host institution in the specified region and provide all required program details for the 2026 cohort.",
        parent=root,
        critical=False
    )

    # 1) Institution selection (critical parallel group)
    await verify_institution_selection(evaluator, complete_node, pkg)

    # 2) Required details (non-critical parallel group)
    await verify_required_details(evaluator, complete_node, pkg)

    # 3) Official source URLs (critical single check)
    await verify_official_sources(evaluator, complete_node, pkg)

    # Return evaluation summary
    return evaluator.get_summary()