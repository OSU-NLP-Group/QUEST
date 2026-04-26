import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

TASK_ID = "fortune500_ldp_rotational_2026"
TASK_DESCRIPTION = "Identify four distinct Fortune 500 leadership development/rotational programs meeting all stated constraints, and provide all requested supporting details for each program."

ALLOWED_LOCATIONS = [
    "New York City",
    "San Francisco Bay Area",
    "Chicago",
    "Boston",
]

YEAR_2026 = 2026


class ProgramEntry(BaseModel):
    company_legal_name: Optional[str] = None
    fortune500_sources: List[str] = Field(default_factory=list)
    program_official_name: Optional[str] = None
    program_url: Optional[str] = None
    program_type_evidence_text: Optional[str] = None
    qualifying_locations: List[str] = Field(default_factory=list)
    location_evidence_urls: List[str] = Field(default_factory=list)
    duration_text: Optional[str] = None
    rotations_functions: List[str] = Field(default_factory=list)
    rotations_count_text: Optional[str] = None
    bachelors_requirement_text: Optional[str] = None
    bachelors_requirement_sources: List[str] = Field(default_factory=list)
    cohort_2026_timing_text: Optional[str] = None
    cohort_2026_timing_sources: List[str] = Field(default_factory=list)
    application_status_2026_text: Optional[str] = None
    application_status_2026_sources: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramEntry] = Field(default_factory=list)


def prompt_extract_programs() -> str:
    return """
    Extract up to six distinct leadership development or rotational programs mentioned in the answer. For each program, return a JSON object with the following fields:

    - company_legal_name: The full legal name of the company offering the program (e.g., "International Business Machines Corporation" or "IBM" — extract what the answer uses).
    - fortune500_sources: An array of URLs explicitly cited that verify the company is ranked in the Fortune 500 (e.g., fortune.com or other credible sources); extract only URLs present in the answer.
    - program_official_name: The official program name as stated in the answer.
    - program_url: The URL to the official program page on the company's careers website (extract exactly the URL(s) provided in the answer; prefer company-owned domains, not third-party aggregators).
    - program_type_evidence_text: Any brief text from the answer that indicates this is a formal leadership development or rotational program.
    - qualifying_locations: Extract an array of locations from the following list ONLY if the answer explicitly mentions the program operating in them: ["New York City","San Francisco Bay Area","Chicago","Boston"]. Use exactly these canonical names if present; otherwise leave empty.
    - location_evidence_urls: URLs in the answer that specifically show positions available or location details for the program (e.g., job postings pages or location-specific program pages).
    - duration_text: The program duration as stated (e.g., "24 months", "2 years", "18 months").
    - rotations_functions: An array of business functions or departments the rotations cover (e.g., "Finance","Operations","Marketing") if stated.
    - rotations_count_text: The rotations count as described (e.g., "2 rotations","three rotations").
    - bachelors_requirement_text: The qualification requirement text for the program (e.g., "Bachelor's degree required"; if MBA is 'preferred' but not required, include that text).
    - bachelors_requirement_sources: URLs from the answer that indicate the minimum qualification requirements for this program.
    - cohort_2026_timing_text: Any text indicating a defined cohort start date/timing in 2026 (e.g., "Fall 2026", "Start date: July 2026").
    - cohort_2026_timing_sources: URLs from the answer that indicate 2026 cohort timing.
    - application_status_2026_text: The application status for 2026 cohorts (e.g., "applications open", "applications upcoming", "applications closed").
    - application_status_2026_sources: URLs from the answer that indicate 2026 application status for the program.

    Rules:
    - Extract only what is explicitly present in the answer. If a field is missing, set it to null or an empty array as appropriate.
    - For URLs, extract real, valid URLs mentioned in the answer (plain or markdown). Do not fabricate URLs.
    - For qualifying_locations, include only canonical names from the list above.
    - Keep each program separate; do not merge data across programs.
    """


def _choose_qualifying_location(entry: ProgramEntry) -> Optional[str]:
    for loc in entry.qualifying_locations:
        if loc in ALLOWED_LOCATIONS:
            return loc
    return None


def _merge_sources(*lists: List[str]) -> Optional[List[str]]:
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in merged:
                merged.append(url)
    if merged:
        return merged
    return None


async def verify_program(evaluator: Evaluator, parent_node, entry: ProgramEntry, idx: int) -> None:
    program_node = evaluator.add_parallel(
        id=f"program_{idx+1}",
        desc=f"{idx+1}st qualifying program" if idx == 0 else (f"{idx+1}nd qualifying program" if idx == 1 else (f"{idx+1}rd qualifying program" if idx == 2 else f"{idx+1}th qualifying program")),
        parent=parent_node,
        critical=False,
    )

    core_exists = evaluator.add_custom_node(
        result=bool(entry.company_legal_name and entry.program_official_name and entry.program_url),
        id=f"program_{idx+1}_core_exists",
        desc="Core fields present: company legal name, program official name, and official program URL",
        parent=program_node,
        critical=True,
    )

    # company_legal_name
    leaf_company = evaluator.add_leaf(
        id=f"program_{idx+1}_company_legal_name",
        desc="Provide the full legal name of the company offering the program.",
        parent=program_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The company offering the program on this page is '{entry.company_legal_name}'.",
        node=leaf_company,
        sources=entry.program_url,
        extra_prerequisites=[core_exists],
        additional_instruction="Check the page branding and company identifiers. Allow legal suffix variations (Inc., Corp., LLC) and common brand vs. legal name equivalence.",
    )

    # fortune500_verification
    leaf_f500 = evaluator.add_leaf(
        id=f"program_{idx+1}_fortune500_verification",
        desc="Provide verification that the company is ranked in the Fortune 500 (citation or link).",
        parent=program_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The company '{entry.company_legal_name}' is ranked in the Fortune 500.",
        node=leaf_f500,
        sources=entry.fortune500_sources if entry.fortune500_sources else None,
        extra_prerequisites=[core_exists],
        additional_instruction="Prefer Fortune.com 'Fortune 500' ranking pages. Membership for recent years (e.g., 2024/2025) is acceptable. The source must explicitly indicate Fortune 500 membership.",
    )

    # program_official_name
    leaf_prog_name = evaluator.add_leaf(
        id=f"program_{idx+1}_program_official_name",
        desc="Provide the official name of the leadership development / rotational program.",
        parent=program_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official program name on the page is '{entry.program_official_name}'.",
        node=leaf_prog_name,
        sources=entry.program_url,
        extra_prerequisites=[core_exists],
        additional_instruction="Allow minor formatting or casing variations. Match the program's name as shown on the official page.",
    )

    # official_careers_url
    leaf_official_url = evaluator.add_leaf(
        id=f"program_{idx+1}_official_careers_url",
        desc="Provide a URL to the official program page on the company's careers website.",
        parent=program_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This URL is the official program page on the company's careers website (company-owned domain; careers/jobs/early-careers section).",
        node=leaf_official_url,
        sources=entry.program_url,
        extra_prerequisites=[core_exists],
        additional_instruction="Confirm the URL is hosted on the company's domain and represents an official careers page (not third-party aggregators).",
    )

    # program_type_ldp_or_rotational
    leaf_type = evaluator.add_leaf(
        id=f"program_{idx+1}_program_type_ldp_or_rotational",
        desc="Provide evidence the offering is a formal leadership development program or rotational program.",
        parent=program_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This offering is a formal leadership development program or rotational program.",
        node=leaf_type,
        sources=entry.program_url,
        extra_prerequisites=[core_exists],
        additional_instruction="Look for explicit indicators like 'Leadership Development Program', 'Rotational Program', multi-rotation structure, and early-career leadership focus.",
    )

    # qualifying_location
    chosen_location = _choose_qualifying_location(entry)
    loc_exists = evaluator.add_custom_node(
        result=bool(chosen_location),
        id=f"program_{idx+1}_location_provided",
        desc="At least one qualifying location is provided",
        parent=program_node,
        critical=True,
    )
    leaf_location = evaluator.add_leaf(
        id=f"program_{idx+1}_qualifying_location",
        desc="Identify at least one qualifying location where the program operates: New York City, San Francisco Bay Area, Chicago, or Boston.",
        parent=program_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The program operates in {chosen_location}.",
        node=leaf_location,
        sources=_merge_sources([entry.program_url] if entry.program_url else [], entry.location_evidence_urls),
        extra_prerequisites=[core_exists, loc_exists],
        additional_instruction="Allow reasonable location name variants (NYC for New York City; San Francisco or Bay Area for San Francisco Bay Area). Confirm the program is actually offered in that metro area.",
    )

    # location_availability_evidence
    leaf_loc_avail = evaluator.add_leaf(
        id=f"program_{idx+1}_location_availability_evidence",
        desc="Provide evidence from official sources that positions are available in the specified qualifying location(s).",
        parent=program_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"There are positions available in {chosen_location} for this program.",
        node=leaf_loc_avail,
        sources=entry.location_evidence_urls if entry.location_evidence_urls else (entry.program_url or None),
        extra_prerequisites=[core_exists, loc_exists],
        additional_instruction="Job posting pages, location-specific program pages, or official listings indicating openings in the target location count as evidence.",
    )

    # duration_requirement
    dur_exists = evaluator.add_custom_node(
        result=bool(entry.duration_text),
        id=f"program_{idx+1}_duration_field_provided",
        desc="Program duration is provided in the answer",
        parent=program_node,
        critical=True,
    )
    leaf_duration = evaluator.add_leaf(
        id=f"program_{idx+1}_duration_requirement",
        desc="State the program duration and verify it lasts at least 18 months.",
        parent=program_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The program duration is at least 18 months.",
        node=leaf_duration,
        sources=entry.program_url,
        extra_prerequisites=[core_exists, dur_exists],
        additional_instruction="Accept durations like '18 months', '24 months', or '2 years' as meeting the requirement.",
    )

    # rotations_across_distinct_functions
    rotations_provided = evaluator.add_custom_node(
        result=bool(entry.rotations_count_text or (entry.rotations_functions and len(entry.rotations_functions) >= 2)),
        id=f"program_{idx+1}_rotations_info_provided",
        desc="Rotations / functions info is provided in the answer",
        parent=program_node,
        critical=True,
    )
    leaf_rotations = evaluator.add_leaf(
        id=f"program_{idx+1}_rotations_across_distinct_functions",
        desc="Verify the program includes at least 2 distinct rotations across different business functions/departments.",
        parent=program_node,
        critical=True,
    )
    rotations_desc = ", ".join(entry.rotations_functions) if entry.rotations_functions else "not specified"
    await evaluator.verify(
        claim=f"The program includes at least two distinct rotations across different business functions or departments. Rotations mentioned: {rotations_desc}.",
        node=leaf_rotations,
        sources=entry.program_url,
        extra_prerequisites=[core_exists, rotations_provided],
        additional_instruction="Look for explicit 'rotations' and distinct functional areas (e.g., Finance vs Operations). Two or more rotations are required.",
    )

    # bachelors_minimum_no_mba_required
    bachelors_exists = evaluator.add_custom_node(
        result=bool(entry.bachelors_requirement_text),
        id=f"program_{idx+1}_bachelors_req_provided",
        desc="Bachelor's degree requirement info is provided in the answer",
        parent=program_node,
        critical=True,
    )
    leaf_bachelors = evaluator.add_leaf(
        id=f"program_{idx+1}_bachelors_minimum_no_mba_required",
        desc="Provide evidence that a bachelor's degree is the minimum required qualification and that an MBA is not required.",
        parent=program_node,
        critical=True,
    )
    await evaluator.verify(
        claim="A bachelor's degree is the minimum required qualification for this program, and an MBA is not required.",
        node=leaf_bachelors,
        sources=_merge_sources([entry.program_url] if entry.program_url else [], entry.bachelors_requirement_sources),
        extra_prerequisites=[core_exists, bachelors_exists],
        additional_instruction="If the page says 'MBA preferred' but not required, this still meets the requirement. The claim should fail if MBA is required.",
    )

    # provide_2026_cohort_timing
    cohort_exists = evaluator.add_custom_node(
        result=bool(entry.cohort_2026_timing_text),
        id=f"program_{idx+1}_cohort_2026_timing_provided",
        desc="2026 cohort timing is provided in the answer",
        parent=program_node,
        critical=True,
    )
    leaf_cohort = evaluator.add_leaf(
        id=f"program_{idx+1}_provide_2026_cohort_timing",
        desc="Provide the program start date or cohort timing for 2026.",
        parent=program_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The program has a cohort start date or timing in {YEAR_2026}: {entry.cohort_2026_timing_text}.",
        node=leaf_cohort,
        sources=_merge_sources([entry.program_url] if entry.program_url else [], entry.cohort_2026_timing_sources),
        extra_prerequisites=[core_exists, cohort_exists],
        additional_instruction=f"Accept expressions like 'Fall {YEAR_2026}', 'Start date July {YEAR_2026}', or 'Class of {YEAR_2026}'.",
    )

    # provide_2026_application_status
    app_exists = evaluator.add_custom_node(
        result=bool(entry.application_status_2026_text),
        id=f"program_{idx+1}_application_status_2026_provided",
        desc="2026 application status is provided in the answer",
        parent=program_node,
        critical=True,
    )
    leaf_app_status = evaluator.add_leaf(
        id=f"program_{idx+1}_provide_2026_application_status",
        desc="State the application status for 2026 cohorts.",
        parent=program_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The application status for {YEAR_2026} cohorts is: {entry.application_status_2026_text}.",
        node=leaf_app_status,
        sources=_merge_sources([entry.program_url] if entry.program_url else [], entry.application_status_2026_sources),
        extra_prerequisites=[core_exists, app_exists],
        additional_instruction=f"Accept statuses like 'applications open', 'upcoming', or 'closed' for {YEAR_2026}.",
    )

    # meets_2026_or_constraint
    has_2026_evidence = evaluator.add_custom_node(
        result=bool(entry.cohort_2026_timing_text or entry.application_status_2026_text),
        id=f"program_{idx+1}_has_2026_evidence",
        desc="Has 2026 timing or 2026 applications evidence",
        parent=program_node,
        critical=True,
    )
    leaf_meets_2026 = evaluator.add_leaf(
        id=f"program_{idx+1}_meets_2026_or_constraint",
        desc="Either defined cohort timing in 2026 OR accepting applications for 2026 cohorts.",
        parent=program_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The program meets the 2026 requirement (either defined {YEAR_2026} cohort timing or accepting applications for {YEAR_2026} cohorts).",
        node=leaf_meets_2026,
        sources=_merge_sources(
            [entry.program_url] if entry.program_url else [],
            entry.cohort_2026_timing_sources,
            entry.application_status_2026_sources,
        ),
        extra_prerequisites=[core_exists, has_2026_evidence],
        additional_instruction="Verify that at least one of the two is true based on the provided sources.",
    )


async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction",
    )

    # Keep first four programs for evaluation
    programs = (extracted.programs or [])[:4]
    while len(programs) < 4:
        programs.append(ProgramEntry())

    # Set-level requirements (critical)
    set_level = evaluator.add_parallel(
        id="set_level_requirements",
        desc="Set-level requirements about the overall list of programs.",
        parent=root,
        critical=True,
    )

    # provide_four_programs
    leaf_four = evaluator.add_custom_node(
        result=(len(extracted.programs) == 4),
        id="provide_four_programs",
        desc="Response provides exactly four programs (i.e., four company+program entries).",
        parent=set_level,
        critical=True,
    )

    # programs_are_distinct
    seen_pairs = set()
    distinct_count = 0
    for i, p in enumerate(programs):
        comp = (p.company_legal_name or "").strip().lower()
        prog = (p.program_official_name or "").strip().lower()
        key = (comp, prog)
        if key not in seen_pairs and comp and prog:
            seen_pairs.add(key)
            distinct_count += 1
    leaf_distinct = evaluator.add_custom_node(
        result=(distinct_count == 4),
        id="programs_are_distinct",
        desc="Verify the four listed programs are distinct (no duplicate company+program pairs).",
        parent=set_level,
        critical=True,
    )

    # Program-level verifications
    for idx in range(4):
        await verify_program(evaluator, root, programs[idx], idx)

    # Add custom info for transparency
    evaluator.add_custom_info(
        {"allowed_locations": ALLOWED_LOCATIONS, "target_year": YEAR_2026},
        info_type="config",
        info_name="evaluation_constraints",
    )

    return evaluator.get_summary()