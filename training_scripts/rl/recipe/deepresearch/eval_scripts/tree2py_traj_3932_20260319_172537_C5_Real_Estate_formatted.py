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
TASK_ID = "abq_mixed_use_2026"
TASK_DESCRIPTION = (
    "Identify a specific mixed-use development project in Albuquerque, New Mexico that was completed between 2024-2026, is currently under construction, or in advanced planning phase as of March 2026, and meets the following requirements: "
    "(1) The project must be a mixed-use development that includes both a residential housing component and a commercial office/business space component; "
    "(2) The commercial office or business space component must be at least 15,000 square feet; "
    "(3) The project must include parking facilities, with the total number of parking spaces specified; "
    "(4) For the office/business space component, the parking provision should reasonably align with Albuquerque's Integrated Development Ordinance guideline of approximately 1 parking space per 500 square feet of office space (considering that mixed-use developments may share parking among components); "
    "(5) The project must have secured necessary development approvals and be actively moving toward construction or already under construction; "
    "(6) The specific project name, location address or area, development timeline, and all component details must be verifiable through reliable sources. "
    "Provide the project name, specific location, residential unit count, office/business space size and type, total parking spaces, parking type, development status and timeline, developer information, and reference URLs supporting all information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProjectSources(BaseModel):
    sources_identity_location: List[str] = Field(default_factory=list)
    sources_status_approvals: List[str] = Field(default_factory=list)
    sources_residential: List[str] = Field(default_factory=list)
    sources_office: List[str] = Field(default_factory=list)
    sources_parking: List[str] = Field(default_factory=list)
    sources_developer: List[str] = Field(default_factory=list)


class ProjectExtraction(BaseModel):
    # Identification / Location
    project_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    address_or_area: Optional[str] = None

    # Status / timeline / approvals
    status: Optional[str] = None  # e.g., "under construction", "advanced planning", "completed"
    status_as_of: Optional[str] = None  # e.g., "March 2026"
    completion_year: Optional[str] = None  # e.g., "2025" if completed
    groundbreaking_date: Optional[str] = None
    est_completion: Optional[str] = None
    approvals_or_permits: Optional[str] = None  # textual description of approvals/permitting milestones
    ido_compliance_mentioned: Optional[str] = None  # "yes" / "no" / "unclear"
    actively_moving_description: Optional[str] = None  # e.g., "site work ongoing"

    # Mixed-use components
    residential_unit_count: Optional[str] = None
    office_business_sqft: Optional[str] = None
    office_business_type: Optional[str] = None  # e.g., "office", "coworking", "business space"

    # Parking
    total_parking_spaces: Optional[str] = None
    parking_type: Optional[str] = None  # e.g., "structured garage", "surface", "underground", "mixed"

    # Developer
    developer_name: Optional[str] = None

    # Sources (category-specific)
    sources: ProjectSources = Field(default_factory=ProjectSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project() -> str:
    return """
Extract details for exactly one specific mixed-use development project in Albuquerque, New Mexico described in the answer (choose the first or primary one if multiple are mentioned). Return a single JSON object with the following fields:

IDENTITY & LOCATION
- project_name: The project's name or designation (string).
- city: City (string).
- state: State (string).
- address_or_area: Specific street address OR a clearly identifiable area/district or intersection (string).

STATUS / TIMELINE / APPROVALS
- status: Current status (e.g., "under construction", "advanced planning", "completed"; use exact phrasing from the answer if present).
- status_as_of: Status as-of date if given (e.g., "March 2026").
- completion_year: If completed, include the completion year (string like "2024", "2025", or "2026"; otherwise null).
- groundbreaking_date: Groundbreaking date if mentioned (string or null).
- est_completion: Estimated completion timeframe if mentioned (string or null).
- approvals_or_permits: Summary text of key approvals/permits/entitlements (e.g., DRB/EPC approvals, building permits).
- ido_compliance_mentioned: "yes" / "no" / "unclear" depending on whether the answer states or implies IDO compliance/approval.
- actively_moving_description: Short text confirming activity toward construction (e.g., "site work ongoing", "contracting", "scheduled groundbreaking").

MIXED-USE COMPONENTS
- residential_unit_count: Exact unit count number as presented (string; keep formatting as in answer, e.g., "210").
- office_business_sqft: Stated square footage of the commercial office/business space (string; keep digits/commas, e.g., "20,000").
- office_business_type: Type/description of the office/business space (e.g., "office", "coworking", "micro-office", "business space").

PARKING
- total_parking_spaces: The total number of parking spaces (string; keep as provided, e.g., "275").
- parking_type: Parking type (e.g., "structured garage", "underground", "surface", "mixed").

DEVELOPER
- developer_name: Primary developer/development entity (string).

SOURCES (URL lists – must be explicitly present in the answer)
- sources: {
    sources_identity_location: [URLs that mention the project identity/name and its Albuquerque location/address/area],
    sources_status_approvals: [URLs that mention status/timeline/approvals/IDO compliance],
    sources_residential: [URLs that mention the residential unit count],
    sources_office: [URLs that mention office/business space size and/or type],
    sources_parking: [URLs that mention total parking spaces and/or parking type],
    sources_developer: [URLs that name the developer]
}

RULES:
- Extract only what is explicitly present in the answer text. Do not invent or infer.
- For each URL list, include only valid URLs that appear in the answer. If none are present for a category, return an empty list for that category.
- Keep numbers as strings exactly as written (e.g., "15,000").
- If any field is missing from the answer, set it to null.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def safe_str(v: Optional[str]) -> str:
    return v if (v is not None) else ""


def parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # extract first number with optional commas
    m = re.search(r"(\d[\d,\.]*)", text)
    if not m:
        return None
    digits = m.group(1)
    # strip non-digits except dot/comma, then remove commas
    digits = digits.replace(",", "")
    try:
        # Prefer integer
        if "." in digits:
            return int(float(digits))
        return int(digits)
    except Exception:
        return None


def uniq_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Main evaluation logic                                                       #
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

    # Extract structured project info from the answer
    proj: ProjectExtraction = await evaluator.extract(
        prompt=prompt_extract_project(),
        template_class=ProjectExtraction,
        extraction_name="project_extraction",
    )

    # Convenience variables
    name = safe_str(proj.project_name)
    city = safe_str(proj.city)
    state = safe_str(proj.state)
    addr = safe_str(proj.address_or_area)

    status = safe_str(proj.status)
    status_as_of = safe_str(proj.status_as_of)
    completion_year = safe_str(proj.completion_year)
    groundbreaking = safe_str(proj.groundbreaking_date)
    est_completion = safe_str(proj.est_completion)
    approvals_text = safe_str(proj.approvals_or_permits)
    ido_flag = safe_str(proj.ido_compliance_mentioned)
    moving_text = safe_str(proj.actively_moving_description)

    res_units = safe_str(proj.residential_unit_count)
    office_sqft_text = safe_str(proj.office_business_sqft)
    office_type = safe_str(proj.office_business_type)
    parking_spaces_text = safe_str(proj.total_parking_spaces)
    parking_type = safe_str(proj.parking_type)
    developer = safe_str(proj.developer_name)

    # Numeric parsing for reasoning
    office_sqft_int = parse_int_from_text(office_sqft_text)
    parking_spaces_int = parse_int_from_text(parking_spaces_text)

    # Prepare sources
    src_id_loc = (proj.sources.sources_identity_location if proj.sources else []) or []
    src_status = (proj.sources.sources_status_approvals if proj.sources else []) or []
    src_res = (proj.sources.sources_residential if proj.sources else []) or []
    src_office = (proj.sources.sources_office if proj.sources else []) or []
    src_park = (proj.sources.sources_parking if proj.sources else []) or []
    src_dev = (proj.sources.sources_developer if proj.sources else []) or []

    # Create the top-level critical node mirroring the rubric "root"
    top = evaluator.add_parallel(
        id="albuquerque_mixed_use_development",
        desc="Evaluate whether the identified Albuquerque mixed-use development project (as of March 2026) satisfies all stated constraints and provides all required verifiable details.",
        parent=root,
        critical=True,
    )

    # ---------------- Project identification & location -------------------
    proj_loc = evaluator.add_parallel(
        id="project_identification_and_location",
        desc="A single, specific project is clearly identified and located in Albuquerque, NM, with a specific location.",
        parent=top,
        critical=True,
    )

    # project_name_provided (existence check)
    evaluator.add_custom_node(
        result=(len(name.strip()) > 0),
        id="project_name_provided",
        desc="Project name/designation is provided.",
        parent=proj_loc,
        critical=True,
    )

    # located_in_albuquerque_nm (URL‑verified)
    node_located = evaluator.add_leaf(
        id="located_in_albuquerque_nm",
        desc="Project is located in Albuquerque, New Mexico.",
        parent=proj_loc,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The project named '{name}' is located in Albuquerque, New Mexico.",
        node=node_located,
        sources=src_id_loc,
        additional_instruction="Verify that the page identifies the project's location in Albuquerque, NM. Minor naming variations are acceptable.",
    )

    # specific_address_or_area_provided (URL‑verified)
    node_specific_loc = evaluator.add_leaf(
        id="specific_address_or_area_provided",
        desc="Specific address or clearly identifiable location area (e.g., intersection/district/site) is provided.",
        parent=proj_loc,
        critical=True,
    )
    addr_claim = (
        f"The project has a specific address or clearly identifiable area in Albuquerque: '{addr}'. "
        "A page that states a specific street address, an intersection, a district name, or a well-defined site also satisfies this."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=node_specific_loc,
        sources=src_id_loc,
        additional_instruction="Accept either a specific street address or an unambiguous identifiable area/district/intersection within Albuquerque.",
    )

    # --------------- Development status, timeline, approvals --------------
    dev_stat = evaluator.add_parallel(
        id="development_status_timeline_and_approvals",
        desc="The project meets the timing/status requirement and has approvals/IDO compliance indicating it is moving toward or in construction.",
        parent=top,
        critical=True,
    )

    # status_within_required_time_window (URL‑verified)
    node_timewin = evaluator.add_leaf(
        id="status_within_required_time_window",
        desc="Project is completed between 2024–2026 OR under construction OR in advanced planning phase as of March 2026.",
        parent=dev_stat,
        critical=True,
    )
    timewin_claim = (
        "As of March 2026, the project is either: (a) completed between 2024 and 2026 (inclusive), "
        "(b) under construction, or (c) in an advanced planning phase. "
        f"Extracted status: '{status}' {('(as of ' + status_as_of + ')') if status_as_of else ''}; "
        f"extracted completion year: '{completion_year}'."
    )
    await evaluator.verify(
        claim=timewin_claim,
        node=node_timewin,
        sources=src_status,
        additional_instruction="Accept synonyms for 'advanced planning' (e.g., final design, pre-construction) and 'under construction' (e.g., site work ongoing). If the source clearly shows completion in 2024-2026, it qualifies.",
    )

    # approvals_or_permits_secured (URL‑verified)
    node_approvals = evaluator.add_leaf(
        id="approvals_or_permits_secured",
        desc="Evidence the project has secured necessary development approvals/permits to proceed with construction.",
        parent=dev_stat,
        critical=True,
    )
    approvals_claim = (
        "The sources show that the project has secured key development approvals or permits to proceed "
        "(e.g., DRB/EPC approvals, site plan approval, building permits, entitlements)."
    )
    await evaluator.verify(
        claim=approvals_claim,
        node=node_approvals,
        sources=src_status,
        additional_instruction="Look for explicit mentions like 'approved', 'DRB', 'EPC', 'site plan approval', 'building permit', 'entitlement approved', etc.",
    )

    # ido_compliance_evidence (URL‑verified)
    node_ido = evaluator.add_leaf(
        id="ido_compliance_evidence",
        desc="Evidence the project complies with Albuquerque's Integrated Development Ordinance (IDO) requirements applicable to the development (e.g., IDO-approved/IDO-compliant per credible documentation).",
        parent=dev_stat,
        critical=True,
    )
    ido_claim = (
        "The sources provide evidence that the project complies with or was reviewed/approved under Albuquerque's Integrated Development Ordinance (IDO), "
        "such as 'IDO-compliant', 'IDO approval', DRB/EPC references under IDO, or equivalent language."
    )
    await evaluator.verify(
        claim=ido_claim,
        node=node_ido,
        sources=src_status,
        additional_instruction="Prefer explicit 'IDO' mentions; acceptance is possible if an official DRB/EPC action under IDO is clearly documented.",
    )

    # actively_moving_toward_construction_or_under_construction (URL‑verified)
    node_moving = evaluator.add_leaf(
        id="actively_moving_toward_construction_or_under_construction",
        desc="Evidence the project is actively moving toward construction (e.g., scheduled groundbreaking/contracting/site work) or already under construction.",
        parent=dev_stat,
        critical=True,
    )
    moving_claim = (
        "The sources indicate the project is actively moving toward construction or already under construction "
        "(e.g., scheduled groundbreaking, contracting, procurement, site work started)."
    )
    await evaluator.verify(
        claim=moving_claim,
        node=node_moving,
        sources=src_status,
        additional_instruction="Look for phrases confirming activity: 'groundbreaking', 'site work', 'under construction', 'construction start', 'contract awarded', etc.",
    )

    # timeline_details_provided (URL‑verified)
    node_timeline = evaluator.add_leaf(
        id="timeline_details_provided",
        desc="Development timeline details are provided (e.g., approval date(s), groundbreaking and/or estimated completion timeframe).",
        parent=dev_stat,
        critical=True,
    )
    timeline_claim = (
        "The sources provide at least one timeline detail such as an approval date, groundbreaking date, "
        "or an estimated completion timeframe for the project."
    )
    await evaluator.verify(
        claim=timeline_claim,
        node=node_timeline,
        sources=src_status,
        additional_instruction="Any explicit timeline markers (dates or timeframes) satisfy this requirement.",
    )

    # ---------------- Mixed-use components and sizes ----------------------
    mix = evaluator.add_parallel(
        id="mixed_use_components_and_sizes",
        desc="The project is mixed-use with required residential and office/business components, and required component details are provided.",
        parent=top,
        critical=True,
    )

    # mixed_use_includes_residential_and_office_business (URL‑verified)
    node_both = evaluator.add_leaf(
        id="mixed_use_includes_residential_and_office_business",
        desc="Project includes BOTH (a) a residential housing component and (b) a commercial office/business space component.",
        parent=mix,
        critical=True,
    )
    both_claim = (
        "This project includes both a residential housing component and a distinct commercial office/business space component."
    )
    await evaluator.verify(
        claim=both_claim,
        node=node_both,
        sources=uniq_urls(src_res, src_office, src_id_loc),
        additional_instruction="The evidence can come from one page mentioning both components, or separate credible pages where one states residential units and another states office/business space; either is acceptable for this verification.",
    )

    # residential_unit_count_provided (URL‑verified)
    node_res_units = evaluator.add_leaf(
        id="residential_unit_count_provided",
        desc="Residential unit count is specified.",
        parent=mix,
        critical=True,
    )
    res_units_claim = f"The project includes {res_units} residential housing units."
    await evaluator.verify(
        claim=res_units_claim,
        node=node_res_units,
        sources=src_res,
        additional_instruction="Verify that the number of residential units is stated or can be clearly inferred from the page.",
    )

    # office_or_business_space_size_meets_minimum (URL‑verified)
    node_office_size = evaluator.add_leaf(
        id="office_or_business_space_size_meets_minimum",
        desc="Office/business space square footage is specified and is at least 15,000 sq ft.",
        parent=mix,
        critical=True,
    )
    office_size_claim = (
        f"The project includes approximately {office_sqft_text} square feet of office/business space, "
        "which is at least 15,000 square feet."
    )
    await evaluator.verify(
        claim=office_size_claim,
        node=node_office_size,
        sources=src_office,
        additional_instruction="Confirm both that office/business space is specified and that the stated amount is ≥ 15,000 sq ft.",
    )

    # office_or_business_space_type_provided (URL‑verified)
    node_office_type = evaluator.add_leaf(
        id="office_or_business_space_type_provided",
        desc="Type of office/business space is described (e.g., office, coworking, micro-office, business space classification).",
        parent=mix,
        critical=True,
    )
    office_type_claim = f"The project includes an office/business space component described as '{office_type}'."
    await evaluator.verify(
        claim=office_type_claim,
        node=node_office_type,
        sources=src_office,
        additional_instruction="Accept reasonable synonyms for type (e.g., 'office', 'business space', 'coworking', 'micro-office').",
    )

    # -------------------- Parking requirements ----------------------------
    park = evaluator.add_parallel(
        id="parking_requirements",
        desc="Parking is provided with required details and is reasonably consistent with the stated office-parking guideline (allowing shared parking considerations).",
        parent=top,
        critical=True,
    )

    # total_parking_spaces_specified (URL‑verified)
    node_park_spaces = evaluator.add_leaf(
        id="total_parking_spaces_specified",
        desc="Total number of parking spaces is specified.",
        parent=park,
        critical=True,
    )
    park_spaces_claim = f"The project provides a total of {parking_spaces_text} parking spaces."
    await evaluator.verify(
        claim=park_spaces_claim,
        node=node_park_spaces,
        sources=src_park,
        additional_instruction="Confirm the page states the total number of parking spaces (combined across all parking types if applicable).",
    )

    # parking_type_specified (URL‑verified)
    node_park_type = evaluator.add_leaf(
        id="parking_type_specified",
        desc="Parking type is specified (e.g., underground, structured garage, surface, mixed).",
        parent=park,
        critical=True,
    )
    park_type_claim = f"The parking type for the project is '{parking_type}' (e.g., structured garage, underground, surface, or mixed)."
    await evaluator.verify(
        claim=park_type_claim,
        node=node_park_type,
        sources=src_park,
        additional_instruction="Any clear description of the parking facility type(s) is acceptable.",
    )

    # parking_guideline_reasonable_alignment (Reasoning check; depends on office size + total parking)
    node_park_align = evaluator.add_leaf(
        id="parking_guideline_reasonable_alignment",
        desc="Using the provided office/business sq ft and total parking spaces (and any stated allocation/shared-parking explanation), the parking provision is not clearly inconsistent with the IDO guideline of ~1 space per 500 sq ft of office space, considering mixed-use shared parking; any deviation is supported by an explanation or credible evidence (e.g., shared-parking plan, waiver, or documented compliance).",
        parent=park,
        critical=True,
    )

    # Prepare reasoning instruction with computed numbers if available
    ratio_info = ""
    if office_sqft_int and parking_spaces_int and office_sqft_int > 0:
        spaces_per_sqft = parking_spaces_int / float(office_sqft_int)
        sqft_per_space = office_sqft_int / float(parking_spaces_int) if parking_spaces_int > 0 else None
        ratio_info = (
            f"Numbers provided: office_sqft={office_sqft_int}, total_parking_spaces={parking_spaces_int}. "
            f"Computed: spaces_per_sqft={spaces_per_sqft:.6f} (target ≈ 0.002), "
            f"sqft_per_space={sqft_per_space:.2f} (target ≈ 500). "
            "Treat values roughly between 1 space per 350–800 sqft as reasonably aligned, "
            "especially if shared parking among mixed-use components is indicated."
        )
    else:
        ratio_info = (
            "Insufficient numeric data parsed from the extracted values to compute a ratio. "
            "If either office_sqft or total_parking_spaces is missing or not supported by sources, judge this as not aligned."
        )

    park_align_claim = (
        f"Given office/business area ≈ '{office_sqft_text}' sq ft and total parking ≈ '{parking_spaces_text}' spaces, "
        "the parking provision is reasonably aligned with Albuquerque’s IDO guideline of about 1 space per 500 sq ft for office, "
        "taking into account shared parking in mixed-use developments. "
        "Deviations are acceptable if not grossly inconsistent or if supported by a documented shared-parking approach."
    )
    await evaluator.verify(
        claim=park_align_claim,
        node=node_park_align,
        # Use simple reasoning; rely on previously verified leaves for numeric facts
        sources=None,
        additional_instruction=ratio_info,
        extra_prerequisites=[node_office_size, node_park_spaces],
    )

    # ---------------------- Developer information -------------------------
    dev = evaluator.add_parallel(
        id="developer_information",
        desc="Developer information is provided.",
        parent=top,
        critical=True,
    )

    node_dev_name = evaluator.add_leaf(
        id="developer_name_provided",
        desc="Primary developer/development entity name is provided.",
        parent=dev,
        critical=True,
    )
    dev_claim = f"The project's primary developer/development entity is '{developer}'."
    await evaluator.verify(
        claim=dev_claim,
        node=node_dev_name,
        sources=src_dev,
        additional_instruction="Verify that the page names the developer/development entity for this project.",
    )

    # ----------------------- Source verifiability -------------------------
    srcv = evaluator.add_parallel(
        id="source_verifiability",
        desc="All required details are supported by reference URLs from credible sources (e.g., city/government, official development documentation, reputable local news).",
        parent=top,
        critical=True,
    )

    # Helper for conditional URL verification (fail fast if no URLs)
    async def verify_sources_leaf(node_id: str, desc: str, claim_text: str, urls: List[str]):
        if not urls:
            evaluator.add_custom_node(
                result=False,
                id=node_id,
                desc=desc,
                parent=srcv,
                critical=True,
            )
            return
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=srcv,
            critical=True,
        )
        await evaluator.verify(
            claim=claim_text,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Judge credibility: prioritize city/government (.gov), official planning/board portals, "
                "developer/owner releases, or reputable local/regional news and business journals. "
                "The page must directly support the stated item."
            ),
        )

    # sources_provided_for_identity_and_location
    await verify_sources_leaf(
        node_id="sources_provided_for_identity_and_location",
        desc="At least one credible reference URL supports the project identity (name) and Albuquerque location (address/area).",
        claim_text=(
            f"The page explicitly references the project '{name}' and its location in Albuquerque, New Mexico, "
            f"including the specific address or identifiable area if available: '{addr}'."
        ),
        urls=src_id_loc,
    )

    # sources_provided_for_status_timeline_and_approvals
    await verify_sources_leaf(
        node_id="sources_provided_for_status_timeline_and_approvals",
        desc="At least one credible reference URL supports the development status/time-window fit and the approvals/permit/IDO-compliance claims.",
        claim_text=(
            "The page provides at least one of the following for the project: "
            "(a) current status within the 2024–2026 window (completed/under construction/advanced planning), "
            "(b) approvals/permits milestones (e.g., DRB/EPC/building permits), or "
            "(c) evidence of IDO compliance/approval."
        ),
        urls=src_status,
    )

    # sources_provided_for_residential_details
    await verify_sources_leaf(
        node_id="sources_provided_for_residential_details",
        desc="At least one credible reference URL supports the residential unit count.",
        claim_text=f"The page states the number of residential units for the project (e.g., '{res_units}' units).",
        urls=src_res,
    )

    # sources_provided_for_office_business_details
    await verify_sources_leaf(
        node_id="sources_provided_for_office_business_details",
        desc="At least one credible reference URL supports the office/business component size (sq ft) and type.",
        claim_text=(
            f"The page states the office/business component details for the project, such as area (~'{office_sqft_text}' sq ft) "
            f"and/or type ('{office_type}')."
        ),
        urls=src_office,
    )

    # sources_provided_for_parking_details
    await verify_sources_leaf(
        node_id="sources_provided_for_parking_details",
        desc="At least one credible reference URL supports the total parking spaces and parking type.",
        claim_text=(
            f"The page states the project's parking details, including total spaces (~'{parking_spaces_text}') "
            f"and/or parking type ('{parking_type}')."
        ),
        urls=src_park,
    )

    # sources_provided_for_developer
    await verify_sources_leaf(
        node_id="sources_provided_for_developer",
        desc="At least one credible reference URL supports the developer/development entity identification.",
        claim_text=f"The page names the developer/development entity for the project (e.g., '{developer}').",
        urls=src_dev,
    )

    # Return standard summary
    return evaluator.get_summary()