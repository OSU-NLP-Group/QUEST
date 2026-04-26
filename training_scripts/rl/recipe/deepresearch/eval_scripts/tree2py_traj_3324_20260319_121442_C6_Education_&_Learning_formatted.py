import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "superintendent_endowment_2026"
TASK_DESCRIPTION = """
Among the current superintendents of the 10 largest school districts in the United States (as of March 2026), identify the one who holds a doctoral degree from a university with an endowment exceeding $1 billion as of fiscal year 2024. For this superintendent, provide the following information: (1) Full name, (2) School district name and state, (3) Year appointed to current position, (4) Specific type of doctoral degree held, (5) Name of the university that granted the doctorate, (6) The university's total endowment value as of FY 2024, and (7) The superintendent's prior position or role immediately before the current appointment.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SuperintendentExtraction(BaseModel):
    # Required core facts
    full_name: Optional[str] = None
    district_name: Optional[str] = None
    district_state: Optional[str] = None  # Accept state full name or abbreviation
    appointment_year: Optional[str] = None  # Keep as string to be robust (e.g., "2022", "2021–2022")
    degree_type: Optional[str] = None       # e.g., "Ed.D.", "Ph.D.", "DBA"
    granting_university: Optional[str] = None
    endowment_value_fy2024: Optional[str] = None  # as presented in the answer (string)
    prior_position: Optional[str] = None

    # Source URLs by facet
    top10_list_urls: List[str] = Field(default_factory=list)        # URLs evidencing "top 10 largest" membership
    current_role_urls: List[str] = Field(default_factory=list)      # URLs evidencing they are currently serving as superintendent
    degree_urls: List[str] = Field(default_factory=list)            # URLs evidencing doctoral degree and (ideally) awarding university
    university_urls: List[str] = Field(default_factory=list)        # URLs specifically about the university and degree (optional)
    endowment_urls: List[str] = Field(default_factory=list)         # URLs evidencing FY 2024 endowment value
    appointment_urls: List[str] = Field(default_factory=list)       # URLs evidencing appointment year
    district_urls: List[str] = Field(default_factory=list)          # URLs about the district (homepage, board, leadership page)
    prior_position_urls: List[str] = Field(default_factory=list)    # URLs evidencing prior role immediately before appointment
    name_urls: List[str] = Field(default_factory=list)              # URLs supporting the full legal name (often overlaps with district/current role)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent() -> str:
    return """
    From the provided answer, extract exactly ONE superintendent record that the answer claims satisfies ALL of the following:
    - Leads one of the 10 largest U.S. school districts by student enrollment.
    - Is currently serving in the role as of March 2026.
    - Holds a doctoral degree (e.g., Ph.D., Ed.D., DBA, etc.).
    - The awarding university for that doctoral degree has an endowment exceeding $1 billion in fiscal year 2024.

    Return a single JSON object with these fields:
    1) full_name: The superintendent's complete name, as written in the answer.
    2) district_name: The school district name (e.g., "Los Angeles Unified School District").
    3) district_state: The U.S. state (full name or two-letter abbreviation).
    4) appointment_year: Year appointed to current superintendent role (string, keep as written in the answer, e.g., "2023" or "2022–2023" if that’s how it appears).
    5) degree_type: The specific doctoral degree type (e.g., "Ed.D.", "Ph.D.", etc.).
    6) granting_university: The name of the university that granted the doctoral degree.
    7) endowment_value_fy2024: The FY 2024 total endowment amount as presented in the answer (string, include currency and commas if present).
    8) prior_position: The immediate prior position/role before the current appointment (as described in the answer).

    Also extract any URL sources explicitly cited in the answer that directly support each facet:
    - top10_list_urls: URLs explicitly supporting that the district is among the 10 largest U.S. school districts by enrollment.
    - current_role_urls: URLs supporting that the person is currently serving as superintendent (as of March 2026).
    - degree_urls: URLs supporting that the person holds a doctoral degree and (ideally) that the degree was awarded by the specified university.
    - university_urls: URLs supporting the granting university information (optional if covered by degree_urls).
    - endowment_urls: URLs supporting the FY 2024 endowment amount for the university (NACUBO reports, official financials, etc.).
    - appointment_urls: URLs supporting the appointment year to current role (e.g., board press releases).
    - district_urls: URLs describing the district and/or its leadership.
    - prior_position_urls: URLs supporting the immediate prior role before current appointment.
    - name_urls: URLs supporting the full, official name spelling.

    Extraction rules:
    - Extract ONLY what is explicitly present in the answer. Do NOT invent or infer missing values.
    - If a value is not present, set it to null. If a URL category has no sources, return an empty list for that category.
    - For URLs, extract the actual links (plain URLs or links from markdown). Do not include non-link citations.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*url_lists: Optional[List[str]]) -> List[str]:
    """Flatten, deduplicate while preserving order."""
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u or not isinstance(u, str):
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_superintendent_identification(
    evaluator: Evaluator,
    parent_node,
    ext: SuperintendentExtraction,
) -> None:
    """
    Build and verify the 'SuperintendentIdentification' subtree:
    - Top 10 district criterion
    - Currently serving criterion
    - Doctoral degree from the specified university
    - University FY2024 endowment > $1B
    """
    sup_node = evaluator.add_parallel(
        id="superintendent_identification",
        desc="Correctly identify the superintendent who meets all specified criteria",
        parent=parent_node,
        critical=False,  # Keep container non-critical to allow mixed children
    )

    # 1) Top 10 largest district criterion
    n_top10 = evaluator.add_leaf(
        id="top10_district",
        desc="The superintendent leads one of the 10 largest school districts in the United States by enrollment",
        parent=sup_node,
        critical=True,
    )
    district_name = ext.district_name or ""
    claim_top10 = f"{district_name} is one of the 10 largest school districts in the United States by student enrollment."
    await evaluator.verify(
        claim=claim_top10,
        node=n_top10,
        sources=ext.top10_list_urls,
        additional_instruction=(
            "Verify that the provided page(s) show the district among the top 10 U.S. school districts by enrollment. "
            "Allow naming variants (e.g., 'NYC Public Schools' vs. 'New York City Department of Education'). "
            "The evidence should explicitly indicate top-10 membership."
        ),
    )

    # 2) Currently serving criterion (as of March 2026)
    n_current = evaluator.add_leaf(
        id="current_serving",
        desc="The superintendent is serving in the position as of March 2026",
        parent=sup_node,
        critical=True,
    )
    full_name = ext.full_name or ""
    state = ext.district_state or ""
    claim_current = f"As of March 2026, {full_name} is serving as the superintendent of {district_name} in {state}."
    current_sources = combine_sources(ext.current_role_urls, ext.district_urls)
    await evaluator.verify(
        claim=claim_current,
        node=n_current,
        sources=current_sources,
        additional_instruction=(
            "The page(s) should indicate the person currently holds the superintendent role (or equivalent title) "
            "and reflect timely/updated information around March 2026 (e.g., an updated leadership page or recent press release)."
        ),
    )

    # 3) Doctoral degree from the specified university
    n_degree = evaluator.add_leaf(
        id="doctoral_degree",
        desc="The superintendent holds a doctoral degree",
        parent=sup_node,
        critical=True,
    )
    granting_univ = ext.granting_university or ""
    claim_degree = f"{full_name} holds a doctoral degree from {granting_univ}."
    await evaluator.verify(
        claim=claim_degree,
        node=n_degree,
        sources=ext.degree_urls,
        additional_instruction=(
            "Confirm that the person holds a doctoral-level degree (e.g., Ph.D., Ed.D., DBA) and that "
            f"the awarding institution is {granting_univ}. Allow formatting variants like 'Ed.D.' vs 'EdD'."
        ),
    )

    # 4) University endowment > $1B in FY 2024
    n_endow_criterion = evaluator.add_leaf(
        id="university_endowment_criterion",
        desc="The university that granted the doctorate has an endowment exceeding $1 billion as of FY 2024",
        parent=sup_node,
        critical=True,
    )
    claim_endow_criterion = f"In fiscal year 2024, {granting_univ} had an endowment exceeding $1 billion."
    await evaluator.verify(
        claim=claim_endow_criterion,
        node=n_endow_criterion,
        sources=ext.endowment_urls,
        additional_instruction=(
            "Use FY 2024 figures (NACUBO 2024 report or official FY 2024 financials). "
            "If the exact dollar value is shown, it should be greater than $1,000,000,000. "
            "If a page is about a different fiscal year, treat as not supported."
        ),
    )


async def verify_required_information(
    evaluator: Evaluator,
    parent_node,
    ext: SuperintendentExtraction,
) -> None:
    """
    Build and verify the 'RequiredInformation' subtree:
    For each information item, add:
    - A correctness verification leaf (critical).
    - A reference existence check (non-critical, custom node).
    """
    req_node = evaluator.add_parallel(
        id="required_information",
        desc="Provide all required information about the identified superintendent with proper verification",
        parent=parent_node,
        critical=False,
    )

    # ---- Full Name ----
    full_name_node = evaluator.add_sequential(
        id="full_name",
        desc="Provide the superintendent's full name",
        parent=req_node,
        critical=False,
    )
    ln = evaluator.add_leaf(
        id="full_name_correctness",
        desc="The full name is accurate and complete",
        parent=full_name_node,
        critical=True,
    )
    name_sources = combine_sources(ext.name_urls, ext.current_role_urls, ext.district_urls)
    await evaluator.verify(
        claim=f"The superintendent's full name is '{ext.full_name or ''}'.",
        node=ln,
        sources=name_sources,
        additional_instruction=(
            "Confirm the official full name. Allow reasonable formatting differences (e.g., middle initials, suffixes). "
            "Prefer the district's official page or authoritative bios."
        ),
    )
    evaluator.add_custom_node(
        result=len(name_sources) > 0,
        id="full_name_reference",
        desc="URL reference supporting the full name",
        parent=full_name_node,
        critical=False,
    )

    # ---- District Name + State ----
    district_node = evaluator.add_sequential(
        id="district_name",
        desc="Provide the school district name and state location",
        parent=req_node,
        critical=False,
    )
    ld = evaluator.add_leaf(
        id="district_name_correctness",
        desc="The district name and state are accurate",
        parent=district_node,
        critical=True,
    )
    district_sources = combine_sources(ext.district_urls, ext.current_role_urls)
    district_state = ext.district_state or ""
    await evaluator.verify(
        claim=f"The superintendent leads the '{ext.district_name or ''}' in {district_state}.",
        node=ld,
        sources=district_sources,
        additional_instruction="Confirm both the district name and its state location as provided.",
    )
    evaluator.add_custom_node(
        result=len(district_sources) > 0,
        id="district_name_reference",
        desc="URL reference supporting the district name and location",
        parent=district_node,
        critical=False,
    )

    # ---- Appointment Year ----
    appt_node = evaluator.add_sequential(
        id="appointment_year",
        desc="Provide the year the superintendent was appointed to current position",
        parent=req_node,
        critical=False,
    )
    la = evaluator.add_leaf(
        id="appointment_year_correctness",
        desc="The appointment year is accurate",
        parent=appt_node,
        critical=True,
    )
    appt_sources = combine_sources(ext.appointment_urls, ext.current_role_urls, ext.district_urls)
    await evaluator.verify(
        claim=f"{ext.full_name or ''} was appointed to the superintendent role in {ext.appointment_year or ''}.",
        node=la,
        sources=appt_sources,
        additional_instruction=(
            "Confirm the appointment year (not necessarily the start date if those differ slightly). "
            "Press releases, board announcements, or official bios are acceptable sources."
        ),
    )
    evaluator.add_custom_node(
        result=len(appt_sources) > 0,
        id="appointment_year_reference",
        desc="URL reference supporting the appointment year",
        parent=appt_node,
        critical=False,
    )

    # ---- Doctoral Degree Type ----
    degtype_node = evaluator.add_sequential(
        id="degree_type",
        desc="Provide the specific type of doctoral degree",
        parent=req_node,
        critical=False,
    )
    ldt = evaluator.add_leaf(
        id="degree_type_correctness",
        desc="The doctoral degree type is accurate",
        parent=degtype_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{ext.full_name or ''} holds a {ext.degree_type or ''} doctoral degree.",
        node=ldt,
        sources=ext.degree_urls,
        additional_instruction="Verify the precise doctoral degree type (e.g., Ed.D., Ph.D.). Allow standard formatting variants.",
    )
    evaluator.add_custom_node(
        result=len(ext.degree_urls) > 0,
        id="degree_type_reference",
        desc="URL reference supporting the degree type",
        parent=degtype_node,
        critical=False,
    )

    # ---- Granting University ----
    gu_node = evaluator.add_sequential(
        id="granting_university",
        desc="Provide the name of the university that granted the doctorate",
        parent=req_node,
        critical=False,
    )
    lgu = evaluator.add_leaf(
        id="university_name_correctness",
        desc="The university name is accurate",
        parent=gu_node,
        critical=True,
    )
    uni_sources = combine_sources(ext.degree_urls, ext.university_urls)
    await evaluator.verify(
        claim=f"{ext.full_name or ''} received the doctoral degree from {ext.granting_university or ''}.",
        node=lgu,
        sources=uni_sources,
        additional_instruction="Verify that the specified university actually awarded the person's doctoral degree.",
    )
    evaluator.add_custom_node(
        result=len(uni_sources) > 0,
        id="university_name_reference",
        desc="URL reference supporting the granting university",
        parent=gu_node,
        critical=False,
    )

    # ---- University Endowment (FY 2024) ----
    endow_node = evaluator.add_sequential(
        id="university_endowment",
        desc="Provide the university's FY 2024 endowment value",
        parent=req_node,
        critical=False,
    )
    lec = evaluator.add_leaf(
        id="endowment_value_correctness",
        desc="The FY 2024 endowment value is accurate and exceeds $1 billion",
        parent=endow_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"As of fiscal year 2024, {ext.granting_university or ''} had an endowment of "
            f"{ext.endowment_value_fy2024 or ''}, and this amount exceeds $1 billion."
        ),
        node=lec,
        sources=ext.endowment_urls,
        additional_instruction=(
            "Verify using FY 2024 sources (e.g., NACUBO 2024 report or the university's FY 2024 financial statements). "
            "Confirm the figure corresponds to FY 2024 and is > $1,000,000,000. "
            "If the extracted numeric value conflicts with the page, mark incorrect."
        ),
    )
    evaluator.add_custom_node(
        result=len(ext.endowment_urls) > 0,
        id="endowment_value_reference",
        desc="URL reference supporting the endowment value",
        parent=endow_node,
        critical=False,
    )

    # ---- Prior Position ----
    prior_node = evaluator.add_sequential(
        id="prior_position",
        desc="Provide the superintendent's prior position or role before the current appointment",
        parent=req_node,
        critical=False,
    )
    lpp = evaluator.add_leaf(
        id="prior_position_correctness",
        desc="The prior position information is accurate",
        parent=prior_node,
        critical=True,
    )
    prior_sources = combine_sources(ext.prior_position_urls, ext.appointment_urls)
    await evaluator.verify(
        claim=(
            f"Immediately prior to being appointed superintendent of {ext.district_name or ''}, "
            f"{ext.full_name or ''} served as {ext.prior_position or ''}."
        ),
        node=lpp,
        sources=prior_sources,
        additional_instruction=(
            "Confirm the role immediately preceding the current superintendent appointment. "
            "Press releases or bios that explicitly state 'prior to this role' are preferred."
        ),
    )
    evaluator.add_custom_node(
        result=len(prior_sources) > 0,
        id="prior_position_reference",
        desc="URL reference supporting the prior position",
        parent=prior_node,
        critical=False,
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
    Evaluate an answer for the superintendent endowment (2026) task.
    Returns a structured summary including the verification tree and final score.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Evaluate identification first, then detailed fields
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_superintendent(),
        template_class=SuperintendentExtraction,
        extraction_name="superintendent_extraction",
    )

    # Build and verify tree
    await verify_superintendent_identification(evaluator, root, extracted)
    await verify_required_information(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()