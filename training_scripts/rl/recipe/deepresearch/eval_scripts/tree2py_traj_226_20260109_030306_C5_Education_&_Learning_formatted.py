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
TASK_ID = "cs_phd_funding_requirements_2025"
TASK_DESCRIPTION = """A U.S. citizen who completed their bachelor's degree in computer science in 2023 and has worked in the software industry for 3 years is planning to pursue a Ph.D. in computer science starting in Fall 2025. They are exploring funding opportunities and program requirements. Please provide the following information:

1. Are they eligible for the NSF Graduate Research Fellowship Program (GRFP)? What is the annual stipend amount and total duration of support provided by this fellowship?

2. Are they eligible for the Fulbright U.S. Student Program? What types of grants are available through this program?

3. What are the minimum curriculum requirements (in semester credit hours) for ABET-accredited computer science programs, specifically for mathematics/statistics/science courses and fundamental computing topics?

4. What is the typical range of minimum annual stipends for graduate assistantship positions at U.S. universities for the 2024-2025 academic year, and what are the standard enrollment and work hour requirements for these positions?
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NSFSection(BaseModel):
    # Eligibility checks (answer explicitly addresses + correct per scenario)
    citizenship_requirement_addressed_and_met: Optional[bool] = None
    degree_intention_requirement_addressed_and_met: Optional[bool] = None
    eligibility_conclusion_explicit: Optional[bool] = None  # did the answer explicitly say "eligible" or "not eligible"
    # Fellowship details
    annual_stipend: Optional[str] = None
    support_duration: Optional[str] = None  # e.g., "3 years of support within a 5-year fellowship period"
    sources: List[str] = Field(default_factory=list)


class FulbrightSection(BaseModel):
    # Eligibility checks (answer explicitly addresses + correct per scenario)
    citizenship_requirement_addressed_and_met: Optional[bool] = None
    degree_phd_status_requirement_addressed_and_met: Optional[bool] = None
    professional_experience_requirement_addressed_and_met: Optional[bool] = None
    eligibility_conclusion_explicit: Optional[bool] = None
    # Grant types
    grant_types: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class ABETSection(BaseModel):
    math_stats_science_min_hours: Optional[str] = None
    fundamental_computing_min_hours: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class GASection(BaseModel):
    stipend_range_text: Optional[str] = None          # e.g., "$20,000–$35,000"
    min_enrollment_credits: Optional[str] = None      # e.g., "9 credits"
    work_hours_per_week: Optional[str] = None         # e.g., "20 hours/week"
    sources: List[str] = Field(default_factory=list)


class AllExtraction(BaseModel):
    nsf: Optional[NSFSection] = None
    fulbright: Optional[FulbrightSection] = None
    abet: Optional[ABETSection] = None
    ga: Optional[GASection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the requested information exactly as presented in the answer. Return a single JSON object with the following structure:

{
  "nsf": {
    "citizenship_requirement_addressed_and_met": boolean or null,
    "degree_intention_requirement_addressed_and_met": boolean or null,
    "eligibility_conclusion_explicit": boolean or null,
    "annual_stipend": string or null,
    "support_duration": string or null,
    "sources": [array of URLs explicitly cited for NSF GRFP details]
  },
  "fulbright": {
    "citizenship_requirement_addressed_and_met": boolean or null,
    "degree_phd_status_requirement_addressed_and_met": boolean or null,
    "professional_experience_requirement_addressed_and_met": boolean or null,
    "eligibility_conclusion_explicit": boolean or null,
    "grant_types": [array of strings, the grant type names listed in the answer],
    "sources": [array of URLs explicitly cited for Fulbright info]
  },
  "abet": {
    "math_stats_science_min_hours": string or null,
    "fundamental_computing_min_hours": string or null,
    "sources": [array of URLs explicitly cited for ABET requirements]
  },
  "ga": {
    "stipend_range_text": string or null,
    "min_enrollment_credits": string or null,
    "work_hours_per_week": string or null,
    "sources": [array of URLs explicitly cited for graduate assistantship standards]
  }
}

Important instructions:
- Extract only what the answer actually states. Do not infer or invent values.
- For any item not mentioned in the answer, set the field to null (or an empty array for lists).
- For each "sources" field, include only valid URLs that the answer explicitly cites for that category. If none are cited, return an empty array.
- Preserve units and formatting present in the answer (e.g., "$37,000", "3 years of support within a 5-year fellowship period", "20 hours/week").
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _clean_sources(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    cleaned = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    return cleaned if cleaned else None


def _join_quoted(items: List[str]) -> str:
    if not items:
        return ""
    return ", ".join([f"'{i}'" for i in items])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_nsf_checks(evaluator: Evaluator, root_node, data: AllExtraction) -> None:
    nsf_node = evaluator.add_sequential(
        id="NSF_GRFP",
        desc="NSF GRFP: determine eligibility and provide fellowship stipend/duration details",
        parent=root_node,
        critical=True
    )

    # Eligibility group (parallel)
    nsf_elig_node = evaluator.add_parallel(
        id="NSF_Eligibility",
        desc="Determine and state NSF GRFP eligibility for the applicant",
        parent=nsf_node,
        critical=True
    )

    # Citizenship requirement
    leaf_nsf_cit = evaluator.add_leaf(
        id="NSF_Citizenship_Requirement",
        desc="Check applicant meets NSF GRFP citizenship/PR requirement (U.S. citizen, national, or permanent resident)",
        parent=nsf_elig_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly addresses the NSF GRFP citizenship/permanent resident requirement and, given the task description that the applicant is a U.S. citizen, correctly states that this requirement is met.",
        node=leaf_nsf_cit,
        additional_instruction="Judge using only the task description and the answer content; confirm the answer clearly checks this requirement and reaches the correct conclusion."
    )

    # Degree intention requirement
    leaf_nsf_deg = evaluator.add_leaf(
        id="NSF_Degree_Intention_Requirement",
        desc="Check applicant intends to enroll in a research-based STEM master's/doctoral program (as required for NSF GRFP)",
        parent=nsf_elig_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly checks the NSF GRFP degree intention requirement (enrollment in a research-based STEM master's/doctoral program) and, given the task description that the applicant plans a research Ph.D. in Fall 2025, correctly states that this requirement is met.",
        node=leaf_nsf_deg,
        additional_instruction="Confirm the answer explicitly addresses this requirement and the conclusion matches the scenario."
    )

    # Eligibility conclusion (explicit statement)
    leaf_nsf_conc = evaluator.add_leaf(
        id="NSF_Eligibility_Conclusion",
        desc="Explicitly state whether the applicant is eligible or not eligible for NSF GRFP",
        parent=nsf_elig_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states whether the applicant is eligible or not eligible for the NSF GRFP using clear, unambiguous language.",
        node=leaf_nsf_conc,
        additional_instruction="Look for explicit phrases such as 'eligible', 'not eligible', 'meets eligibility', or 'does not meet eligibility'."
    )

    # Fellowship details (parallel)
    nsf_details_node = evaluator.add_parallel(
        id="NSF_Fellowship_Details",
        desc="Provide NSF GRFP support details requested in the question",
        parent=nsf_node,
        critical=True
    )

    nsf_info = data.nsf or NSFSection()
    nsf_sources = _clean_sources(nsf_info.sources)

    # Annual stipend
    leaf_nsf_stip = evaluator.add_leaf(
        id="NSF_Annual_Stipend",
        desc="Provide the annual NSF GRFP stipend amount",
        parent=nsf_details_node,
        critical=True
    )
    stipend_text = nsf_info.annual_stipend or ""
    await evaluator.verify(
        claim=f"The NSF GRFP annual stipend amount is {stipend_text}.",
        node=leaf_nsf_stip,
        sources=nsf_sources,
        additional_instruction="Verify that the stated annual stipend matches the official NSF GRFP information on the cited page(s)."
    )

    # Support duration
    leaf_nsf_duration = evaluator.add_leaf(
        id="NSF_Support_Duration",
        desc="Provide the total duration of support for NSF GRFP (support years and overall fellowship period, per constraints)",
        parent=nsf_details_node,
        critical=True
    )
    duration_text = nsf_info.support_duration or ""
    await evaluator.verify(
        claim=f"The NSF GRFP provides {duration_text}.",
        node=leaf_nsf_duration,
        sources=nsf_sources,
        additional_instruction="Confirm that the cited page(s) explicitly state the support years and overall fellowship period (e.g., years of support within a fellowship period)."
    )


async def build_fulbright_checks(evaluator: Evaluator, root_node, data: AllExtraction) -> None:
    fb_node = evaluator.add_sequential(
        id="Fulbright_US_Student",
        desc="Fulbright U.S. Student Program: determine eligibility and list available grant types",
        parent=root_node,
        critical=True
    )

    # Eligibility group
    fb_elig = evaluator.add_parallel(
        id="Fulbright_Eligibility",
        desc="Determine and state Fulbright U.S. Student Program eligibility for the applicant",
        parent=fb_node,
        critical=True
    )

    # Citizenship requirement
    leaf_fb_cit = evaluator.add_leaf(
        id="Fulbright_Citizenship_Requirement",
        desc="Check applicant meets Fulbright citizenship requirement (U.S. citizen or U.S. national)",
        parent=fb_elig,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly checks the Fulbright U.S. Student Program citizenship requirement and, given the task description that the applicant is a U.S. citizen, correctly states that this requirement is met.",
        node=leaf_fb_cit,
        additional_instruction="Judge using only the task description and the answer content; confirm explicit checking and correct conclusion."
    )

    # Degree/PhD status requirement
    leaf_fb_deg = evaluator.add_leaf(
        id="Fulbright_Degree_PhD_Status_Requirement",
        desc="Check applicant holds a bachelor's degree by award start date and does not already have a Ph.D.",
        parent=fb_elig,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly checks that the applicant will have a bachelor's degree by the award start date and does not already hold a Ph.D., and correctly states that this requirement is met given the task description.",
        node=leaf_fb_deg,
        additional_instruction="Confirm explicit mention and correct conclusion given the scenario facts (BS in 2023; planning Ph.D. start in Fall 2025; no existing Ph.D.)."
    )

    # Professional experience requirement (limited experience)
    leaf_fb_prof = evaluator.add_leaf(
        id="Fulbright_Professional_Experience_Requirement",
        desc="Check applicant meets the 'limited professional experience' criterion (typically ≤7 years for non-enrolled applicants), per constraints",
        parent=fb_elig,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly addresses the Fulbright U.S. Student Program's 'limited professional experience' criterion and correctly states that the applicant meets it given they have 3 years of professional experience.",
        node=leaf_fb_prof,
        additional_instruction="Focus on whether the answer explicitly evaluated this criterion and concluded correctly based on the scenario (3 years experience)."
    )

    # Eligibility conclusion (explicit statement)
    leaf_fb_conc = evaluator.add_leaf(
        id="Fulbright_Eligibility_Conclusion",
        desc="Explicitly state whether the applicant is eligible or not eligible for the Fulbright U.S. Student Program",
        parent=fb_elig,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states whether the applicant is eligible or not eligible for the Fulbright U.S. Student Program using clear, unambiguous language.",
        node=leaf_fb_conc,
        additional_instruction="Look for explicit eligibility statements; not just descriptions of requirements."
    )

    # Grant types (single verification leaf as per rubric)
    fb_sources = _clean_sources((data.fulbright or FulbrightSection()).sources)
    grant_types = (data.fulbright or FulbrightSection()).grant_types or []
    leaf_fb_types = evaluator.add_leaf(
        id="Fulbright_Grant_Types",
        desc="Identify the types of grants available through the Fulbright U.S. Student Program (per constraints)",
        parent=fb_node,
        critical=True
    )
    grant_types_str = _join_quoted(grant_types)
    await evaluator.verify(
        claim=f"The Fulbright U.S. Student Program offers the following grant types: {grant_types_str}.",
        node=leaf_fb_types,
        sources=fb_sources,
        additional_instruction="Verify that the cited page(s) list these grant types (e.g., Study/Research Awards, English Teaching Assistant Awards, and other categories the answer lists)."
    )


async def build_abet_checks(evaluator: Evaluator, root_node, data: AllExtraction) -> None:
    abet_node = evaluator.add_parallel(
        id="ABET_CS_Curriculum_Minima",
        desc="ABET-accredited CS curriculum: provide minimum semester credit hour requirements requested",
        parent=root_node,
        critical=True
    )

    abet_info = data.abet or ABETSection()
    abet_sources = _clean_sources(abet_info.sources)

    # Math/Stats/Science minimum hours
    leaf_abet_mss = evaluator.add_leaf(
        id="ABET_Math_Stats_Science_Min_Hours",
        desc="Provide the minimum required semester credit hours for mathematics/statistics/science courses",
        parent=abet_node,
        critical=True
    )
    mss_text = abet_info.math_stats_science_min_hours or ""
    await evaluator.verify(
        claim=f"ABET-accredited computer science programs require at least {mss_text} in mathematics/statistics/science (semester credit hours).",
        node=leaf_abet_mss,
        sources=abet_sources,
        additional_instruction="Confirm the minimum credit hours from the cited ABET or authoritative specification page(s)."
    )

    # Fundamental computing topics minimum hours
    leaf_abet_comp = evaluator.add_leaf(
        id="ABET_Fundamental_Computing_Min_Hours",
        desc="Provide the minimum required semester credit hours for fundamental computing topics",
        parent=abet_node,
        critical=True
    )
    comp_text = abet_info.fundamental_computing_min_hours or ""
    await evaluator.verify(
        claim=f"ABET-accredited computer science programs require at least {comp_text} in fundamental computing topics (semester credit hours).",
        node=leaf_abet_comp,
        sources=abet_sources,
        additional_instruction="Verify the minimum credit hours for computer science/fundamental computing topics on the cited page(s)."
    )


async def build_ga_checks(evaluator: Evaluator, root_node, data: AllExtraction) -> None:
    ga_node = evaluator.add_parallel(
        id="Graduate_Assistantship_2024_2025",
        desc="Graduate assistantship standards (U.S., 2024–2025): stipend range, enrollment, and work hours",
        parent=root_node,
        critical=True
    )

    ga_info = data.ga or GASection()
    ga_sources = _clean_sources(ga_info.sources)

    # Stipend range
    leaf_ga_stip = evaluator.add_leaf(
        id="GA_Minimum_Annual_Stipend_Range",
        desc="Provide the typical range of minimum annual stipends for graduate assistantships for 2024–2025",
        parent=ga_node,
        critical=True
    )
    range_text = ga_info.stipend_range_text or ""
    await evaluator.verify(
        claim=f"For the 2024–2025 academic year in the U.S., graduate assistantship positions typically have minimum annual stipends in the range of {range_text}.",
        node=leaf_ga_stip,
        sources=ga_sources,
        additional_instruction="Verify that at least one of the cited policy pages or authoritative sources supports this typical minimum range statement."
    )

    # Enrollment requirement
    leaf_ga_enroll = evaluator.add_leaf(
        id="GA_Minimum_Enrollment_Requirement",
        desc="Provide the standard minimum enrollment requirement (credit hours) for graduate assistantships",
        parent=ga_node,
        critical=True
    )
    enroll_text = ga_info.min_enrollment_credits or ""
    await evaluator.verify(
        claim=f"Graduate assistants are typically required to enroll in at least {enroll_text}.",
        node=leaf_ga_enroll,
        sources=ga_sources,
        additional_instruction="Confirm the minimum enrollment (credit hours) requirement on the cited page(s). Accept reasonable phrasing variants like 'full-time enrollment of X credits' or equivalent."
    )

    # Work hours requirement
    leaf_ga_hours = evaluator.add_leaf(
        id="GA_Work_Hours_Requirement",
        desc="Provide the standard work-hour expectation (hours/week) for graduate assistantships",
        parent=ga_node,
        critical=True
    )
    hours_text = ga_info.work_hours_per_week or ""
    await evaluator.verify(
        claim=f"The standard graduate assistant workload is approximately {hours_text} during the academic term.",
        node=leaf_ga_hours,
        sources=ga_sources,
        additional_instruction="Verify that the cited page(s) support this weekly work-hours expectation (e.g., ~20 hours/week for a 50% appointment)."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for: funding eligibility and program/assistantship requirements
    for the described CS Ph.D. applicant.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates in parallel across subtopics
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Provide all requested funding eligibility details and program/assistantship requirements for the described CS Ph.D. applicant",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extraction
    extracted: AllExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllExtraction,
        extraction_name="structured_answer_extraction",
    )

    # Build verification subtrees according to rubric
    await build_nsf_checks(evaluator, root, extracted)
    await build_fulbright_checks(evaluator, root, extracted)
    await build_abet_checks(evaluator, root, extracted)
    await build_ga_checks(evaluator, root, extracted)

    # Return structured summary with verification tree and score
    return evaluator.get_summary()