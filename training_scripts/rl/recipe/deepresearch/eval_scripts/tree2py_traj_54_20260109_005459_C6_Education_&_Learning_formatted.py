import asyncio
import logging
from typing import Optional, List, Dict, Any

from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "online_mba_aacsb_public_states"
TASK_DESCRIPTION = (
    "You are researching affordable, accessible online MBA programs for working professionals. "
    "Identify one AACSB-accredited online MBA program from a public university located in California, Texas, or Florida that meets ALL requirements "
    "across Accreditation, Location, Program Structure, Admissions, Cost, and Delivery Format, and provide all requested attributes with official URL citations."
)

ALLOWED_STATES = {"California", "Texas", "Florida"}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramIdentity(BaseModel):
    university_name: Optional[str] = None
    program_title: Optional[str] = None
    state: Optional[str] = None
    program_urls: List[str] = Field(default_factory=list)          # Official program/college pages
    state_urls: List[str] = Field(default_factory=list)            # Pages confirming location/state
    public_status_urls: List[str] = Field(default_factory=list)    # Pages confirming public university status


class AccreditationInfo(BaseModel):
    aacsb_status: Optional[str] = None
    aacsb_urls: List[str] = Field(default_factory=list)            # AACSB directory or official university accreditation page


class StructureInfo(BaseModel):
    credit_hours: Optional[str] = None
    credit_urls: List[str] = Field(default_factory=list)

    completion_time: Optional[str] = None
    completion_urls: List[str] = Field(default_factory=list)

    core_and_electives: Optional[str] = None
    core_urls: List[str] = Field(default_factory=list)

    specialization: Optional[str] = None
    specialization_urls: List[str] = Field(default_factory=list)

    capstone: Optional[str] = None
    capstone_urls: List[str] = Field(default_factory=list)


class AdmissionsInfo(BaseModel):
    bachelors_degree_required: Optional[str] = None
    degree_urls: List[str] = Field(default_factory=list)

    work_experience_minimum: Optional[str] = None
    experience_urls: List[str] = Field(default_factory=list)

    no_gmat_gre_required: Optional[str] = None
    gmat_urls: List[str] = Field(default_factory=list)


class CostInfo(BaseModel):
    total_cost: Optional[str] = None
    cost_urls: List[str] = Field(default_factory=list)


class DeliveryInfo(BaseModel):
    fully_online_confirmation: Optional[str] = None
    online_urls: List[str] = Field(default_factory=list)

    residency_policy: Optional[str] = None
    residency_urls: List[str] = Field(default_factory=list)


class ProgramExtraction(BaseModel):
    identity: Optional[ProgramIdentity] = None
    accreditation: Optional[AccreditationInfo] = None
    structure: Optional[StructureInfo] = None
    admissions: Optional[AdmissionsInfo] = None
    cost: Optional[CostInfo] = None
    delivery: Optional[DeliveryInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program() -> str:
    return (
        "Extract exactly one online MBA program described in the answer. The program must be from a public (state) university "
        "in California, Texas, or Florida and AACSB-accredited, meeting all constraints. If multiple programs are mentioned, "
        "extract the primary one used in the answer. Return the following fields:\n\n"
        "identity:\n"
        "  - university_name: full university name\n"
        "  - program_title: exact program title (e.g., 'Online MBA')\n"
        "  - state: the state where the university is located (California/Texas/Florida)\n"
        "  - program_urls: list of official university program pages (URLs)\n"
        "  - state_urls: list of official pages confirming university location/state\n"
        "  - public_status_urls: list of official pages confirming the university is public (e.g., state system membership)\n\n"
        "accreditation:\n"
        "  - aacsb_status: statement of AACSB accreditation\n"
        "  - aacsb_urls: list of official URLs confirming AACSB accreditation (AACSB directory or official university accreditation page)\n\n"
        "structure:\n"
        "  - credit_hours: total credits required (string as presented)\n"
        "  - credit_urls: official URLs showing total credit hours\n"
        "  - completion_time: typical/maximum time to complete (string, e.g., '18 months', '2 years', etc.)\n"
        "  - completion_urls: official URLs showing completion time\n"
        "  - core_and_electives: statement confirming the program includes required core courses and electives\n"
        "  - core_urls: official URLs showing curriculum structure (core + electives)\n"
        "  - specialization: at least one specialization/concentration name\n"
        "  - specialization_urls: official URLs listing specializations\n"
        "  - capstone: statement confirming capstone or culminating experience is required\n"
        "  - capstone_urls: official URLs confirming the capstone requirement\n\n"
        "admissions:\n"
        "  - bachelors_degree_required: statement confirming bachelor's degree required from accredited institution\n"
        "  - degree_urls: official URLs confirming this\n"
        "  - work_experience_minimum: statement of minimum experience requirement (e.g., '2 years')\n"
        "  - experience_urls: official URLs confirming this\n"
        "  - no_gmat_gre_required: statement about GMAT/GRE policy (e.g., 'not required', 'waiver available')\n"
        "  - gmat_urls: official URLs confirming GMAT/GRE policy\n\n"
        "cost:\n"
        "  - total_cost: total program cost (tuition + required fees) as stated (string)\n"
        "  - cost_urls: official URLs confirming total program cost or tuition+fees\n\n"
        "delivery:\n"
        "  - fully_online_confirmation: statement confirming 100% online delivery\n"
        "  - online_urls: official URLs confirming fully online\n"
        "  - residency_policy: statement confirming no mandatory on-campus residencies\n"
        "  - residency_urls: official URLs confirming no mandatory residency\n\n"
        "RULES:\n"
        "1) Extract only information explicitly present in the answer; do not invent data. Use strings as presented.\n"
        "2) For each field requesting URLs, extract official URLs (university .edu pages, AACSB directory). "
        "Avoid third-party aggregators when official sources are present. If the answer provides no URL for a field, return an empty list.\n"
        "3) If any field is missing in the answer, set it to null (or empty list for URLs).\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def choose_sources(*candidates: List[List[str]]) -> Optional[List[str]]:
    """Return the first non-empty URL list; otherwise None."""
    for lst in candidates:
        if lst and len(lst) > 0:
            return lst
    return None


def collect_all_urls(info: ProgramExtraction) -> List[str]:
    """Collect all URLs from the extraction results."""
    urls: List[str] = []

    def extend(lst: Optional[List[str]]):
        if lst:
            urls.extend([u for u in lst if isinstance(u, str) and u.strip()])

    if info.identity:
        extend(info.identity.program_urls)
        extend(info.identity.state_urls)
        extend(info.identity.public_status_urls)
    if info.accreditation:
        extend(info.accreditation.aacsb_urls)
    if info.structure:
        extend(info.structure.credit_urls)
        extend(info.structure.completion_urls)
        extend(info.structure.core_urls)
        extend(info.structure.specialization_urls)
        extend(info.structure.capstone_urls)
    if info.admissions:
        extend(info.admissions.degree_urls)
        extend(info.admissions.experience_urls)
        extend(info.admissions.gmat_urls)
    if info.cost:
        extend(info.cost.cost_urls)
    if info.delivery:
        extend(info.delivery.online_urls)
        extend(info.delivery.residency_urls)

    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


def is_official_url(url: str) -> bool:
    """Heuristic: official if domain ends with .edu or is AACSB."""
    try:
        netloc = urlsplit(url).netloc.lower()
        if not netloc:
            return False
        # strip common www.
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return (
            netloc.endswith(".edu")
            or netloc == "aacsb.edu"
            or netloc.endswith(".aacsb.edu")
            or netloc == "aacsb.org"  # allow org too
        )
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_program_identity_and_location(evaluator: Evaluator, parent_node, info: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="program_identity_and_location",
        desc="Program identity, public status, and allowed-state location are correctly provided",
        parent=parent_node,
        critical=True,
    )

    # University and Program Title
    n1 = evaluator.add_leaf(
        id="university_and_program_title",
        desc="Provide the university name and the specific MBA program title",
        parent=node,
        critical=True,
    )
    uni = info.identity.university_name if info.identity else None
    title = info.identity.program_title if info.identity else None
    sources = choose_sources(
        info.identity.program_urls if info.identity else [],
        info.structure.core_urls if info.structure else [],
        info.structure.credit_urls if info.structure else [],
    )
    claim = f"The program titled '{title}' is offered by {uni}."
    add_ins = (
        "Verify the page shows the exact or an equivalent program title and the university name. "
        "Allow minor formatting differences (e.g., Online MBA vs MBA Online)."
        if sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim, node=n1, sources=sources, additional_instruction=add_ins)

    # State Location
    n2 = evaluator.add_leaf(
        id="state_location",
        desc="State where the university is located must be California, Texas, or Florida (and the stated state matches the university)",
        parent=node,
        critical=True,
    )
    state = info.identity.state if info.identity else None
    state_sources = choose_sources(
        info.identity.state_urls if info.identity else [],
        info.identity.program_urls if info.identity else [],
    )
    state_claim = f"The university is located in {state}, which is one of California, Texas, or Florida."
    state_add_ins = (
        "Confirm the location/state from the official page. If the institution name implies a state (e.g., University of Florida), this is acceptable when the page corroborates it."
        if state_sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=state_claim, node=n2, sources=state_sources, additional_instruction=state_add_ins)

    # Public University
    n3 = evaluator.add_leaf(
        id="public_university",
        desc="University is a public (state) university (not private)",
        parent=node,
        critical=True,
    )
    pub_sources = choose_sources(
        info.identity.public_status_urls if info.identity else [],
        info.identity.program_urls if info.identity else [],
    )
    pub_claim = "This university is a public (state) university."
    pub_add_ins = (
        "Confirm public status from an official source (e.g., membership in a state system like UC, CSU, or UT System)."
        if pub_sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=pub_claim, node=n3, sources=pub_sources, additional_instruction=pub_add_ins)


async def verify_aacsb_accreditation(evaluator: Evaluator, parent_node, info: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="aacsb_accreditation",
        desc="Program/school holds AACSB accreditation and it is verified via an official URL",
        parent=parent_node,
        critical=True,
    )

    # AACSB status
    n1 = evaluator.add_leaf(
        id="aacsb_status",
        desc="Confirm AACSB accreditation status for the business school/program",
        parent=node,
        critical=True,
    )
    sources = choose_sources(info.accreditation.aacsb_urls if info.accreditation else [])
    claim = "This business school is AACSB-accredited."
    add_ins = (
        "Use AACSB directory or an official university accreditation page that explicitly states AACSB accreditation."
        if sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim, node=n1, sources=sources, additional_instruction=add_ins)

    # AACSB official URL citation
    n2 = evaluator.add_leaf(
        id="aacsb_official_url_citation",
        desc="Cite an official URL that supports AACSB accreditation (e.g., AACSB listing or official university page)",
        parent=node,
        critical=True,
    )
    claim2 = "This URL is an official AACSB listing or an official university page that confirms AACSB accreditation."
    add_ins2 = (
        "Confirm that the page itself is authoritative (AACSB domain or .edu) and explicitly confirms accreditation."
        if sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim2, node=n2, sources=sources, additional_instruction=add_ins2)


async def verify_program_structure(evaluator: Evaluator, parent_node, info: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="program_structure",
        desc="Program structure constraints are satisfied",
        parent=parent_node,
        critical=True,
    )

    # Credit hours range 36-48
    n1 = evaluator.add_leaf(
        id="credit_hours_range",
        desc="Total credit hours are stated and are between 36 and 48 inclusive, supported by an official URL",
        parent=node,
        critical=True,
    )
    credits = info.structure.credit_hours if info.structure else None
    credits_sources = choose_sources(info.structure.credit_urls if info.structure else [])
    claim1 = f"The program requires {credits} total credit hours, and the total lies between 36 and 48 inclusive."
    add_ins1 = (
        "Confirm total credits from the official curriculum/overview page. If a range is given, ensure it falls within 36–48."
        if credits_sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim1, node=n1, sources=credits_sources, additional_instruction=add_ins1)

    # Completion time ≤ 24 months
    n2 = evaluator.add_leaf(
        id="completion_time_limit",
        desc="Completion time is stated and is 24 months or less (online/part-time format), supported by an official URL",
        parent=node,
        critical=True,
    )
    comp = info.structure.completion_time if info.structure else None
    comp_sources = choose_sources(info.structure.completion_urls if info.structure else [])
    claim2 = f"The program can be completed in {comp}, which is 24 months or less, in an online/part-time format."
    add_ins2 = (
        "Confirm typical or maximum completion time (e.g., 12–24 months) from official pages. Allow minor phrasing variations."
        if comp_sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim2, node=n2, sources=comp_sources, additional_instruction=add_ins2)

    # Core and electives included
    n3 = evaluator.add_leaf(
        id="core_and_electives",
        desc="Program includes required core courses and elective options, supported by an official URL",
        parent=node,
        critical=True,
    )
    core_sources = choose_sources(
        info.structure.core_urls if info.structure else [],
        info.identity.program_urls if info.identity else [],
    )
    claim3 = "The program includes required core courses and elective options."
    add_ins3 = (
        "Confirm the curriculum presents core requirements and separate elective choices."
        if core_sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim3, node=n3, sources=core_sources, additional_instruction=add_ins3)

    # Specialization
    n4 = evaluator.add_leaf(
        id="specialization",
        desc="Provide at least one specialization/concentration option offered by the program, supported by an official URL",
        parent=node,
        critical=True,
    )
    spec = info.structure.specialization if info.structure else None
    spec_sources = choose_sources(info.structure.specialization_urls if info.structure else [])
    claim4 = f"The program offers at least one specialization/concentration, such as '{spec}'."
    add_ins4 = (
        "Confirm that specializations/concentrations are listed on the official page."
        if spec_sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim4, node=n4, sources=spec_sources, additional_instruction=add_ins4)

    # Capstone requirement
    n5 = evaluator.add_leaf(
        id="capstone",
        desc="Confirm a capstone project or culminating experience is required, supported by an official URL",
        parent=node,
        critical=True,
    )
    cap_sources = choose_sources(info.structure.capstone_urls if info.structure else [])
    claim5 = "A capstone project or culminating experience is required to graduate."
    add_ins5 = (
        "Verify that the curriculum explicitly requires a capstone, culminating project, or equivalent."
        if cap_sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim5, node=n5, sources=cap_sources, additional_instruction=add_ins5)


async def verify_admissions(evaluator: Evaluator, parent_node, info: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="admissions",
        desc="Admission requirements constraints are satisfied",
        parent=parent_node,
        critical=True,
    )

    # Bachelor's degree required
    n1 = evaluator.add_leaf(
        id="bachelors_degree_required",
        desc="Bachelor’s degree from an accredited institution is required, supported by an official URL",
        parent=node,
        critical=True,
    )
    deg_sources = choose_sources(info.admissions.degree_urls if info.admissions else [])
    claim1 = "Admission requires a bachelor's degree from an accredited institution."
    add_ins1 = (
        "Confirm the requirement from the official admissions or program page."
        if deg_sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim1, node=n1, sources=deg_sources, additional_instruction=add_ins1)

    # Work experience minimum at least 2 years
    n2 = evaluator.add_leaf(
        id="work_experience_minimum",
        desc="Minimum work experience is stated and is at least 2 years, supported by an official URL",
        parent=node,
        critical=True,
    )
    exp_text = info.admissions.work_experience_minimum if info.admissions else None
    exp_sources = choose_sources(info.admissions.experience_urls if info.admissions else [])
    claim2 = f"Admission requires at least 2 years of professional work experience (policy states: '{exp_text}')."
    add_ins2 = (
        "Confirm the minimum experience requirement is two or more years."
        if exp_sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim2, node=n2, sources=exp_sources, additional_instruction=add_ins2)

    # GMAT/GRE not required (or waiver available)
    n3 = evaluator.add_leaf(
        id="no_gmat_gre_required",
        desc="GMAT/GRE is not required (or waiver available as allowed), supported by an official URL",
        parent=node,
        critical=True,
    )
    gmat_text = info.admissions.no_gmat_gre_required if info.admissions else None
    gmat_sources = choose_sources(info.admissions.gmat_urls if info.admissions else [])
    claim3 = "GMAT/GRE scores are not required to apply (either test optional or waivers available such that the test is not mandatory)."
    add_ins3 = (
        "Confirm the policy states test not required or waivable such that it is not mandatory."
        if gmat_sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim3, node=n3, sources=gmat_sources, additional_instruction=add_ins3)


async def verify_cost(evaluator: Evaluator, parent_node, info: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="cost",
        desc="Cost constraint is satisfied",
        parent=parent_node,
        critical=True,
    )

    # Total cost under $70,000
    n1 = evaluator.add_leaf(
        id="total_cost_under_70000",
        desc="Total program cost (tuition + required fees) is stated and is under $70,000, supported by an official URL",
        parent=node,
        critical=True,
    )
    total_cost_text = info.cost.total_cost if info.cost else None
    cost_sources = choose_sources(info.cost.cost_urls if info.cost else [])
    claim1 = f"Total program cost (tuition + required fees) is under $70,000 (stated total: '{total_cost_text}')."
    add_ins1 = (
        "Confirm that the total cost or an official tuition+fees calculation supports the claim being under $70,000."
        if cost_sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim1, node=n1, sources=cost_sources, additional_instruction=add_ins1)


async def verify_delivery(evaluator: Evaluator, parent_node, info: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="delivery_format",
        desc="Delivery format constraints are satisfied",
        parent=parent_node,
        critical=True,
    )

    # Fully online
    n1 = evaluator.add_leaf(
        id="fully_online",
        desc="Program is 100% online (can be completed entirely online), supported by an official URL",
        parent=node,
        critical=True,
    )
    online_sources = choose_sources(info.delivery.online_urls if info.delivery else [])
    claim1 = "The program is 100% online and can be completed entirely online."
    add_ins1 = (
        "Confirm that the official page explicitly states fully online delivery; hybrid or residency required should fail."
        if online_sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim1, node=n1, sources=online_sources, additional_instruction=add_ins1)

    # No mandatory residency
    n2 = evaluator.add_leaf(
        id="no_mandatory_residency",
        desc="No mandatory on-campus residency visits are required, supported by an official URL",
        parent=node,
        critical=True,
    )
    resid_sources = choose_sources(info.delivery.residency_urls if info.delivery else [])
    claim2 = "No mandatory on-campus residency visits are required for this program."
    add_ins2 = (
        "Confirm that the official page indicates no required residencies; optional immersions are allowed."
        if resid_sources else
        "No URLs are provided; treat the claim as unsupported."
    )
    await evaluator.verify(claim=claim2, node=n2, sources=resid_sources, additional_instruction=add_ins2)


async def verify_official_sources_requirement(evaluator: Evaluator, parent_node, info: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="official_sources_requirement",
        desc="All requested output fields are supported with specific URL references from official sources",
        parent=parent_node,
        critical=True,
    )

    # Heuristic check that all provided URLs are official (edu or AACSB)
    all_urls = collect_all_urls(info)
    official_flags = [is_official_url(u) for u in all_urls]
    official_count = sum(1 for f in official_flags if f)
    total_count = len(all_urls)
    all_official = total_count > 0 and official_count == total_count

    evaluator.add_custom_info(
        info={
            "total_urls_collected": total_count,
            "official_urls_count": official_count,
            "non_official_urls": [u for u, f in zip(all_urls, official_flags) if not f]
        },
        info_type="url_quality",
        info_name="official_sources_check"
    )

    evaluator.add_custom_node(
        result=all_official,
        id="urls_are_official",
        desc="Provided URLs are official sources (university/college pages, AACSB directory, or similarly authoritative official pages) and correspond to the claims they support",
        parent=node,
        critical=True
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
    Evaluate an answer for the online MBA accreditation/location/structure/admissions/cost/delivery constraints.
    """
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
        default_model=model,
    )

    # Extract structured program info
    extracted: ProgramExtraction = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_info",
    )

    # Build verification tree according to rubric
    # Root is critical; all children must be critical too per framework rule
    # 1. Program identity and location
    await verify_program_identity_and_location(evaluator, root, extracted)
    # 2. AACSB accreditation
    await verify_aacsb_accreditation(evaluator, root, extracted)
    # 3. Program structure
    await verify_program_structure(evaluator, root, extracted)
    # 4. Admissions
    await verify_admissions(evaluator, root, extracted)
    # 5. Cost
    await verify_cost(evaluator, root, extracted)
    # 6. Delivery format
    await verify_delivery(evaluator, root, extracted)
    # 7. Official sources requirement
    await verify_official_sources_requirement(evaluator, root, extracted)

    # Return standard summary
    return evaluator.get_summary()