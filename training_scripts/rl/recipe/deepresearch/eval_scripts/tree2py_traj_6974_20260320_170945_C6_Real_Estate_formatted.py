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
TASK_ID = "ca_class_a_leed_warehouse_2024_2025"
TASK_DESCRIPTION = """
Identify a Class A industrial warehouse development project in California that was completed between January 2024 and December 2025, and achieved LEED Gold or Platinum certification. The project must meet all of the following requirements:

Building Physical Specifications:
- Minimum clear height of 32 feet
- Column spacing of at least 40 feet by 40 feet
- Total building size of at least 100,000 square feet
- Loading dock doors in dock-high configuration (elevated approximately 48-52 inches)
- Automatic fire sprinkler system installed per NFPA 13 standards

Sustainability & Energy Efficiency:
- LEED Gold (60-79 points) or LEED Platinum (80+ points) certification achieved
- Meets the LEED prerequisite for indoor water use reduction
- Complies with California Title 24 Energy Code requirements
- HVAC systems meet or exceed ASHRAE 90.1 energy efficiency standards

Provide the project name, location (city), developer, total square footage, clear height, column spacing (if available), LEED certification level achieved, and reference URLs for all key specifications.
"""
ALLOWED_LEED_LEVELS = {"gold", "platinum"}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProjectExtraction(BaseModel):
    # Core identity and requested output fields
    project_name: Optional[str] = None
    city: Optional[str] = None
    developer: Optional[str] = None
    total_sqft: Optional[str] = None
    clear_height: Optional[str] = None
    column_spacing: Optional[str] = None
    column_spacing_declared_unavailable: Optional[bool] = False
    leed_level: Optional[str] = None
    completion_date: Optional[str] = None  # e.g., "Q2 2025", "June 2024", "2025"
    class_designation: Optional[str] = None  # e.g., "Class A"
    building_type: Optional[str] = None      # e.g., "industrial warehouse", "distribution"

    # Building specs evidence text snippets (if present in the answer text)
    dock_high: Optional[str] = None
    loading_docks_note: Optional[str] = None
    sprinkler_nfpa13: Optional[str] = None

    # Sustainability / energy code evidence text snippets (if present)
    indoor_water_use_reduction: Optional[str] = None
    title24: Optional[str] = None
    ashrae_90_1: Optional[str] = None

    # URL sources grouped by claim/topic (must be extracted from the answer text)
    urls_location: List[str] = Field(default_factory=list)
    urls_completion: List[str] = Field(default_factory=list)
    urls_class: List[str] = Field(default_factory=list)
    urls_size: List[str] = Field(default_factory=list)
    urls_clear_height: List[str] = Field(default_factory=list)
    urls_column_spacing: List[str] = Field(default_factory=list)
    urls_loading: List[str] = Field(default_factory=list)      # dock-high / (cross-dock and/or door counts)
    urls_sprinkler: List[str] = Field(default_factory=list)
    urls_leed_cert: List[str] = Field(default_factory=list)
    urls_leed_prereq: List[str] = Field(default_factory=list)
    urls_title24: List[str] = Field(default_factory=list)
    urls_ashrae: List[str] = Field(default_factory=list)
    urls_misc: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project() -> str:
    return """
Extract details for the SINGLE primary project the answer proposes (if multiple projects are mentioned, pick the first one presented as the main candidate). Extract EXACTLY what the answer states; do not invent or infer.

Return a JSON object with these fields:

Identity and requested output fields:
- project_name: string | null
- city: string | null
- developer: string | null
- total_sqft: string | null         // as written (e.g., "1,000,000 SF", "1M SF")
- clear_height: string | null       // as written (e.g., "36'", "40 feet", "36-40 ft")
- column_spacing: string | null     // as written; if NOT provided, set to null
- column_spacing_declared_unavailable: boolean // true ONLY if the answer explicitly says column spacing is not available / N/A / not specified
- leed_level: string | null         // e.g., "LEED Gold", "LEED Platinum" as written
- completion_date: string | null    // completion/delivery/CO date as given (e.g., "June 2024", "Q1 2025")
- class_designation: string | null  // e.g., "Class A" if explicitly stated
- building_type: string | null      // e.g., "industrial warehouse", "distribution"

Evidence phrases as written in the answer (optional):
- dock_high: string | null                       // text stating dock-high doors (~48–52 in)
- loading_docks_note: string | null              // text about number/configuration (e.g., "cross-dock", "X dock-high doors")
- sprinkler_nfpa13: string | null                // text like "NFPA 13" or "ESFR (per NFPA 13)"
- indoor_water_use_reduction: string | null      // text stating LEED indoor water use reduction (if present)
- title24: string | null                         // text stating Title 24 compliance (if present)
- ashrae_90_1: string | null                     // text stating ASHRAE 90.1 (if present)

URL sources for each claim/topic (MUST be URLs explicitly present in the answer):
- urls_location: string[]          // URLs supporting California location/city
- urls_completion: string[]        // URLs supporting completion between Jan 2024 – Dec 2025
- urls_class: string[]             // URLs supporting Class A AND industrial/distribution facility designation
- urls_size: string[]              // URLs supporting total building size (>= 100,000 SF)
- urls_clear_height: string[]      // URLs supporting clear height (>= 32 ft)
- urls_column_spacing: string[]    // URLs supporting column spacing (>= 40x40), if provided
- urls_loading: string[]           // URLs supporting dock-high doors (~48–52 in) and/or cross-dock / sufficient doors
- urls_sprinkler: string[]         // URLs supporting NFPA 13 sprinkler system
- urls_leed_cert: string[]         // URLs supporting LEED certification (Gold or Platinum)
- urls_leed_prereq: string[]       // URLs supporting indoor water use reduction prerequisite (can be certification/scorecard page if it implies prerequisites met)
- urls_title24: string[]           // URLs supporting California Title 24 compliance
- urls_ashrae: string[]            // URLs supporting ASHRAE 90.1 compliance
- urls_misc: string[]              // Any other relevant URLs cited in the answer

Special rules:
- Extract ONLY URLs explicitly included in the answer text (including markdown links). Do not infer or add new URLs.
- Normalize URLs if missing protocol by prepending http://
- If a URL group is not present, return an empty array (not null).
- Set column_spacing_declared_unavailable to true ONLY if the answer explicitly states it's unavailable (N/A / not specified).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls or []:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _coalesce_urls(*groups: List[str]) -> List[str]:
    all_urls: List[str] = []
    for g in groups:
        all_urls.extend(g or [])
    return _dedup_urls(all_urls)


def _has_digits(s: Optional[str]) -> bool:
    if not s:
        return False
    return any(ch.isdigit() for ch in s)


def _normalize_level(level: Optional[str]) -> Optional[str]:
    if not level:
        return None
    return level.strip().lower()


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def _verify_claim_with_sources(
    evaluator: Evaluator,
    parent,
    node_id_base: str,
    main_desc: str,
    claim: str,
    sources: List[str],
    *,
    critical: bool = True,
    additional_instruction: str = "None",
) -> None:
    """
    Build a small sequential group that first checks sources are provided, then verifies the claim by the URLs.
    This ensures source-grounding for every factual verification.
    """
    group = evaluator.add_sequential(
        id=node_id_base,
        desc=main_desc,
        parent=parent,
        critical=critical
    )

    # 1) Source existence (critical)
    src_exist = evaluator.add_custom_node(
        result=len(sources) > 0,
        id=f"{node_id_base}_sources_provided",
        desc=f"{main_desc} — supporting URL(s) are provided in the answer",
        parent=group,
        critical=True
    )

    # 2) Verify claim by URL(s) (critical)
    leaf = evaluator.add_leaf(
        id=f"{node_id_base}_supported",
        desc=main_desc,
        parent=group,
        critical=True
    )
    # This call will automatically skip if the prior source-existence node failed in this sequential group.
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction
    )


async def _build_step_1_constraints(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    """
    Step 1: Project satisfies every stated constraint with URL support (all critical).
    Organized as a parallel node; each constraint is a small sequential sub-group enforcing 'source exists' then 'verify by URL'.
    """
    step1 = evaluator.add_parallel(
        id="Step_1_Project_Meets_All_Constraints_With_Citations",
        desc="Chosen project satisfies every stated constraint, and each constraint is supported by at least one reference URL provided in the response.",
        parent=parent,
        critical=True
    )

    # Located in California
    loc_sources = _dedup_urls(data.urls_location)
    loc_city = data.city or "the stated city"
    await _verify_claim_with_sources(
        evaluator, step1, "Located_In_California",
        "Project is located in California AND at least one provided URL supports this.",
        claim=f"The project '{data.project_name or 'the project'}' is located in {loc_city}, California (CA).",
        sources=loc_sources,
        additional_instruction="Pass only if the page clearly places the project in a California city or states 'CA' for the location."
    )

    # Completed between Jan 2024 and Dec 2025
    comp_sources = _dedup_urls(data.urls_completion)
    await _verify_claim_with_sources(
        evaluator, step1, "Completed_Between_Jan_2024_And_Dec_2025",
        "Project completion date is between January 2024 and December 2025 AND at least one provided URL supports this.",
        claim=f"The project was completed (delivered/CO issued/substantially complete) between January 2024 and December 2025 (reported completion: {data.completion_date or 'unspecified in answer'}).",
        sources=comp_sources,
        additional_instruction="Accept synonyms such as 'delivered', 'CO issued', 'substantial completion'. Confirm that the date falls within Jan 1, 2024 to Dec 31, 2025."
    )

    # Class A industrial warehouse
    class_sources = _dedup_urls(data.urls_class)
    await _verify_claim_with_sources(
        evaluator, step1, "Class_A_Industrial_Warehouse",
        "Facility is designated/labeled as a Class A industrial warehouse (or equivalent explicit Class A industrial/distribution designation) AND at least one provided URL supports this.",
        claim="This facility is explicitly designated as a Class A industrial warehouse/distribution building.",
        sources=class_sources,
        additional_instruction="Page should explicitly indicate 'Class A' and an industrial/warehouse/distribution use."
    )

    # Clear height >= 32 ft
    ch_sources = _dedup_urls(data.urls_clear_height)
    await _verify_claim_with_sources(
        evaluator, step1, "Clear_Height_Min_32ft",
        "Clear height is at least 32 feet AND at least one provided URL supports the clear-height value/claim.",
        claim=f"The building has a clear height of at least 32 feet (value cited in answer: {data.clear_height or 'unspecified'}).",
        sources=ch_sources,
        additional_instruction="Accept formats such as 32', 36', 40 feet, or ranges (e.g., 36'–40'). Confirm the page states a number >= 32 feet."
    )

    # Column spacing >= 40x40
    cs_sources = _dedup_urls(data.urls_column_spacing)
    await _verify_claim_with_sources(
        evaluator, step1, "Column_Spacing_Min_40x40",
        "Column spacing is at least 40 ft by 40 ft AND at least one provided URL supports the column-spacing value/claim.",
        claim=f"The building's structural column spacing is at least 40 ft by 40 ft (value cited in answer: {data.column_spacing or 'unspecified'}).",
        sources=cs_sources,
        additional_instruction="Page should state column spacing. Accept >= 40' x 40' (e.g., 50' x 50'). If not stated on the page, do not pass."
    )

    # Building size >= 100,000 SF
    size_sources = _dedup_urls(data.urls_size)
    await _verify_claim_with_sources(
        evaluator, step1, "Building_Size_Min_100k_SF",
        "Total building size is at least 100,000 square feet AND at least one provided URL supports the building-size value/claim.",
        claim=f"The total building area is at least 100,000 square feet (value cited in answer: {data.total_sqft or 'unspecified'}).",
        sources=size_sources,
        additional_instruction="Confirm the page states a building size >= 100,000 SF. Accept formats like 'sf', 'square feet', or 'MSF'."
    )

    # Dock-high loading doors (48–52 inches) - explicit dock-high evidence
    load_sources = _dedup_urls(data.urls_loading)
    await _verify_claim_with_sources(
        evaluator, step1, "Loading_Docks_Dock_High_48_52in",
        "Loading dock doors are dock-high (approximately 48–52 inches, or explicitly stated dock-high) AND at least one provided URL supports this.",
        claim="The building provides dock-high loading doors (approximately 48–52 inches deck height) or explicitly states 'dock-high' loading.",
        sources=load_sources,
        additional_instruction="Pass if page states 'dock-high' or clearly indicates 48–52 inch dock height."
    )

    # Sufficient loading / cross-dock configuration (or similar adequacy)
    await _verify_claim_with_sources(
        evaluator, step1, "Loading_Docks_Sufficient_And_Appropriate_For_Size_Cross_Dock",
        "Evidence indicates the building has sufficient loading dock doors appropriate for its size and cross-dock configuration (as stated in the constraints) AND at least one provided URL supports this claim.",
        claim=f"The facility is cross-dock or otherwise provides an appropriate/sufficient number of dock-high doors for a Class A warehouse of this size ({data.total_sqft or 'size unspecified'}).",
        sources=load_sources,
        additional_instruction="Pass if the page states 'cross-dock', 'two-sided loading', or lists a substantial dock door count appropriate for a large Class A facility."
    )

    # Sprinkler system per NFPA 13
    spr_sources = _dedup_urls(data.urls_sprinkler)
    await _verify_claim_with_sources(
        evaluator, step1, "Sprinkler_System_NFPA_13",
        "Automatic fire sprinkler system is installed per NFPA 13 standards AND at least one provided URL supports this.",
        claim="The building has an automatic fire sprinkler system installed per NFPA 13 (ESFR per NFPA 13 also qualifies).",
        sources=spr_sources,
        additional_instruction="Look for 'NFPA 13', 'ESFR (per NFPA 13)'. If not explicitly stated, do not pass."
    )

    # LEED Gold or Platinum certification
    leed_sources = _dedup_urls(data.urls_leed_cert)
    norm_level = _normalize_level(data.leed_level)
    level_for_claim = data.leed_level or "LEED Gold/Platinum"
    await _verify_claim_with_sources(
        evaluator, step1, "LEED_Gold_Or_Platinum",
        "Project achieved LEED Gold or LEED Platinum certification AND at least one provided URL supports the certification level.",
        claim=f"The project achieved {level_for_claim}, which is within Gold or Platinum tiers.",
        sources=leed_sources,
        additional_instruction="The page should explicitly show LEED Gold or LEED Platinum (or USGBC/Green Building Registry listing)."
    )

    # LEED prerequisite: indoor water use reduction
    # It is acceptable to ground via the certification page if it clearly implies prerequisites are met for Gold/Platinum.
    leed_prereq_sources = _dedup_urls(_coalesce_urls(data.urls_leed_prereq, data.urls_leed_cert))
    await _verify_claim_with_sources(
        evaluator, step1, "LEED_Indoor_Water_Use_Reduction_Prereq",
        "Project meets the LEED prerequisite for indoor water use reduction AND at least one provided URL supports this.",
        claim="By achieving LEED Gold/Platinum, the project satisfied all LEED prerequisites, including the Indoor Water Use Reduction prerequisite.",
        sources=leed_prereq_sources,
        additional_instruction="Prefer a LEED scorecard/certification page. If not explicitly listing the prerequisite, it is acceptable to conclude prerequisites were met if the page confirms Gold/Platinum certification (since all prerequisites are mandatory)."
    )

    # California Title 24 compliance
    t24_sources = _dedup_urls(data.urls_title24)
    await _verify_claim_with_sources(
        evaluator, step1, "California_Title_24_Compliance",
        "Project complies with California Title 24 Energy Code requirements applicable at time of construction AND at least one provided URL supports this.",
        claim="The project complies with California Title 24 Energy Code (e.g., 2019 or 2022 standards) as applicable during construction.",
        sources=t24_sources,
        additional_instruction="Pass only if the page clearly mentions 'Title 24' compliance or equivalent language."
    )

    # HVAC meets/exceeds ASHRAE 90.1
    ash_sources = _dedup_urls(data.urls_ashrae)
    await _verify_claim_with_sources(
        evaluator, step1, "HVAC_Meets_ASHRAE_90_1",
        "HVAC systems meet or exceed ASHRAE 90.1 energy efficiency standards AND at least one provided URL supports this.",
        claim="The building's HVAC systems meet or exceed ASHRAE 90.1 standards.",
        sources=ash_sources,
        additional_instruction="Look for explicit mention of 'ASHRAE 90.1' or equivalent compliance language for mechanical systems."
    )


def _add_step_2_field_checks(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    """
    Step 2: Response includes all requested output fields (existence/format checks).
    All are critical under this node (as per rubric).
    """
    step2 = evaluator.add_parallel(
        id="Step_2_Response_Contains_Requested_Fields",
        desc="Response includes all requested output fields for the identified project.",
        parent=parent,
        critical=True
    )

    # Project Name
    evaluator.add_custom_node(
        result=bool(data.project_name and data.project_name.strip()),
        id="Project_Name_Provided",
        desc="Response provides the project name.",
        parent=step2,
        critical=True
    )

    # City Location
    evaluator.add_custom_node(
        result=bool(data.city and data.city.strip()),
        id="City_Location_Provided",
        desc="Response provides the project location city.",
        parent=step2,
        critical=True
    )

    # Developer
    evaluator.add_custom_node(
        result=bool(data.developer and data.developer.strip()),
        id="Developer_Provided",
        desc="Response provides the developer name.",
        parent=step2,
        critical=True
    )

    # Total Square Footage (numeric-like)
    evaluator.add_custom_node(
        result=bool(_has_digits(data.total_sqft)),
        id="Total_Square_Footage_Provided",
        desc="Response provides total square footage (numeric value).",
        parent=step2,
        critical=True
    )

    # Clear Height (numeric-like)
    evaluator.add_custom_node(
        result=bool(_has_digits(data.clear_height)),
        id="Clear_Height_Provided",
        desc="Response provides clear height (numeric value, in feet).",
        parent=step2,
        critical=True
    )

    # Column Spacing if available: either provided with a supporting URL, or explicitly noted unavailable
    column_spacing_ok = (
        (bool(data.column_spacing and data.column_spacing.strip()) and len(_dedup_urls(data.urls_column_spacing)) > 0)
        or (not (data.column_spacing and data.column_spacing.strip()) and bool(data.column_spacing_declared_unavailable))
    )
    evaluator.add_custom_node(
        result=column_spacing_ok,
        id="Column_Spacing_If_Available_Handled_Correctly",
        desc="Response provides column spacing if supported by sources; otherwise explicitly states it is not available (and does not fabricate a value).",
        parent=step2,
        critical=True
    )

    # LEED level provided and valid (Gold or Platinum)
    leed_norm = _normalize_level(data.leed_level)
    evaluator.add_custom_node(
        result=bool(leed_norm in ALLOWED_LEED_LEVELS),
        id="LEED_Level_Provided",
        desc="Response states the achieved LEED certification level (Gold or Platinum).",
        parent=step2,
        critical=True
    )

    # Reference URLs provided (for key specs/claims)
    all_key_urls = _coalesce_urls(
        data.urls_location, data.urls_completion, data.urls_class, data.urls_size,
        data.urls_clear_height, data.urls_column_spacing, data.urls_loading,
        data.urls_sprinkler, data.urls_leed_cert, data.urls_leed_prereq,
        data.urls_title24, data.urls_ashrae, data.urls_misc
    )
    evaluator.add_custom_node(
        result=len(all_key_urls) > 0,
        id="Reference_URLs_Provided",
        desc="Response provides reference URL(s) for the key specifications/claims it makes (not missing entirely).",
        parent=step2,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluation entry point for:
    Identify one qualifying Class A industrial warehouse project in California (completed Jan 2024–Dec 2025),
    meeting all building and sustainability constraints, and provide requested details with supporting URLs.
    """
    # Initialize evaluator (root uses SEQUENTIAL so Step 2 will be skipped if Step 1 fails)
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_project(),
        template_class=ProjectExtraction,
        extraction_name="project_extraction"
    )

    # Build step 1 constraints checks (all critical)
    await _build_step_1_constraints(evaluator, root, extracted)

    # Build step 2 field presence checks (all critical)
    _add_step_2_field_checks(evaluator, root, extracted)

    # Optional: record quick custom info
    evaluator.add_custom_info(
        {
            "project_name": extracted.project_name,
            "city": extracted.city,
            "developer": extracted.developer,
            "leed_level": extracted.leed_level,
            "completion_date": extracted.completion_date,
        },
        info_type="extracted_summary",
        info_name="extracted_project_summary"
    )

    # Return summary
    return evaluator.get_summary()