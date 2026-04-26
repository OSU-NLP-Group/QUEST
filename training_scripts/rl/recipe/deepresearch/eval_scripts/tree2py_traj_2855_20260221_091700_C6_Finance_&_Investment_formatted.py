import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# ------------------------------------------------------------------------------------
# Task Constants
# ------------------------------------------------------------------------------------
TASK_ID = "park_ave_hq_2025"
TASK_DESCRIPTION = (
    "Identify the major U.S. financial institution that recently opened its new global headquarters on Park Avenue "
    "in Midtown Manhattan in 2025, where the headquarters building occupies an entire city block. This institution must "
    "have an Asset & Wealth Management division that manages over $75 billion in real estate assets globally, and the "
    "institution's total assets under management must exceed $3.5 trillion. Provide the complete headquarters address "
    "including ZIP code, confirm the building is bounded by 47th Street, 48th Street, Madison Avenue, and Park Avenue, "
    "and verify these figures using official sources from the institution's investor relations materials or press releases "
    "dated from 2024 or early 2025."
)


# ------------------------------------------------------------------------------------
# Extraction Models
# ------------------------------------------------------------------------------------
class InstitutionExtraction(BaseModel):
    # Institution identification
    institution_name: Optional[str] = None
    identification_sources: List[str] = Field(default_factory=list)

    # Headquarters address
    hq_address: Optional[str] = None  # Full address string as given (should include ZIP)
    hq_zip: Optional[str] = None      # ZIP extracted if separately stated in the answer
    hq_address_sources: List[str] = Field(default_factory=list)

    # Building characteristics / boundaries
    boundaries_statement: Optional[str] = None  # Free text claim from the answer (e.g., "bounded by 47th, 48th, Madison, Park")
    building_sources: List[str] = Field(default_factory=list)

    # Opening timeline
    opening_year: Optional[str] = None
    opening_sources: List[str] = Field(default_factory=list)

    # Asset management scale
    real_estate_aum: Optional[str] = None
    real_estate_aum_sources: List[str] = Field(default_factory=list)

    total_aum: Optional[str] = None
    total_aum_sources: List[str] = Field(default_factory=list)

    # Organizational structure
    division_name: Optional[str] = None  # e.g., "Asset & Wealth Management" or similar
    division_sources: List[str] = Field(default_factory=list)

    segments_list: List[str] = Field(default_factory=list)  # names of operating segments if provided
    segments_sources: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------------
# Extraction Prompt
# ------------------------------------------------------------------------------------
def prompt_extract_institution_info() -> str:
    return """
Extract the following structured information from the answer text. Return null for any missing item and an empty list for missing URLs.

Fields to extract:
1) institution_name: The complete official name of the identified U.S. financial institution.
2) identification_sources: All URLs (official sources only if provided) used to identify/confirm the institution’s identity (e.g., investor relations, press releases, corporate pages).

3) hq_address: The complete headquarters street address on Park Avenue in New York, NY, including the ZIP code if provided.
4) hq_zip: The ZIP code for the headquarters address, extracted explicitly if present; otherwise null.
5) hq_address_sources: All URLs (prefer official sources) that confirm the headquarters address.

6) boundaries_statement: The exact phrasing from the answer that claims the new HQ building's boundaries (e.g., “bounded by 47th Street, 48th Street, Madison Avenue, and Park Avenue”).
7) building_sources: All URLs (prefer official sources) used to support the building’s characteristics/boundaries.

8) opening_year: The year in which the new HQ officially opened (should be 2025 if stated).
9) opening_sources: All URLs (prefer official sources) used to support the opening year.

10) real_estate_aum: The stated real estate assets under management figure for the institution’s Asset & Wealth Management (or equivalent) division (e.g., "$100+ billion").
11) real_estate_aum_sources: All URLs (prefer investor relations materials or press releases) that substantiate the real estate AUM figure. Prefer sources dated 2024 or early 2025 if given.

12) total_aum: The institution’s total assets under management figure (e.g., "$3.8 trillion").
13) total_aum_sources: All URLs (prefer official sources) that substantiate total AUM. Prefer sources dated 2024 or early 2025 if provided.

14) division_name: The name of the Asset & Wealth Management division (or equivalent).
15) division_sources: All URLs (prefer official sources) documenting that the institution has an Asset & Wealth Management division (or equivalent) as a primary operating segment.

16) segments_list: The list of distinct operating segments if the answer provides them (e.g., “Consumer & Community Banking”, “Corporate & Investment Bank”, “Asset & Wealth Management”, etc.).
17) segments_sources: URLs (prefer official sources) documenting the institution’s operating segment structure.

Rules:
- Extract only what is explicitly present in the answer.
- Include URLs as complete URLs (http/https). Do not invent any URLs.
- Keep numbers as strings exactly as written (e.g., "$3.8 trillion", "$110B").
"""


# ------------------------------------------------------------------------------------
# Verification Helpers
# ------------------------------------------------------------------------------------
def _non_empty_list(lst: Optional[List[str]]) -> bool:
    return bool(lst) and len(lst) > 0


async def _verify_institution_identification(evaluator: Evaluator, parent, data: InstitutionExtraction):
    node = evaluator.add_parallel(
        id="institution_identification",
        desc="Identify the financial institution that meets all specified criteria",
        parent=parent,
        critical=True,
    )

    # Existence: institution name provided
    evaluator.add_custom_node(
        result=(data.institution_name is not None and data.institution_name.strip() != ""),
        id="institution_name_provided",
        desc="Provide the complete official name of the financial institution",
        parent=node,
        critical=True,
    )

    # Ensure we have at least one identification source URL
    evaluator.add_custom_node(
        result=_non_empty_list(data.identification_sources),
        id="identification_source_provided",
        desc="At least one official URL is provided to confirm the institution’s identity",
        parent=node,
        critical=True,
    )

    # Verify the identification source is an official institutional page confirming identity
    id_leaf = evaluator.add_leaf(
        id="identification_source",
        desc="Provide URL from official institutional sources confirming its identity",
        parent=node,
        critical=True,
    )
    inst_name = data.institution_name or ""
    await evaluator.verify(
        claim=(
            f"The provided webpage is an official page from '{inst_name}' (e.g., investor relations, press release, or corporate site) "
            f"that clearly confirms the institution's identity and name."
        ),
        node=id_leaf,
        sources=data.identification_sources,
        additional_instruction=(
            "Confirm that the page belongs to the identified institution (look for brand, logo, footer, or ownership) "
            "and that it clearly identifies the institution by name."
        ),
    )


async def _verify_headquarters_location(evaluator: Evaluator, parent, data: InstitutionExtraction):
    node = evaluator.add_parallel(
        id="headquarters_location",
        desc="Verify headquarters location on Park Avenue in Midtown Manhattan",
        parent=parent,
        critical=True,
    )

    # ---------------- Complete Address ----------------
    addr_node = evaluator.add_parallel(
        id="complete_address",
        desc="Verify the complete headquarters address on Park Avenue in New York, NY",
        parent=node,
        critical=True,
    )

    # Existence checks: address and source presence
    evaluator.add_custom_node(
        result=(data.hq_address is not None and data.hq_address.strip() != ""),
        id="address_value_provided",
        desc="Headquarters address is provided in the answer",
        parent=addr_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_list(data.hq_address_sources),
        id="address_source_provided",
        desc="At least one official URL is provided confirming the headquarters address",
        parent=addr_node,
        critical=True,
    )

    # Verify address components (Park Avenue, New York, NY, ZIP code)
    addr_components_leaf = evaluator.add_leaf(
        id="address_components_correct",
        desc="Confirm address includes Park Avenue, New York, NY, and correct ZIP code",
        parent=addr_node,
        critical=True,
    )
    hq_addr = data.hq_address or ""
    hq_zip = data.hq_zip or ""
    await evaluator.verify(
        claim=(
            f"The official headquarters address is '{hq_addr}', located on Park Avenue in New York, NY, and includes a valid ZIP code "
            f"{('('+hq_zip+')') if hq_zip else '(ZIP code included in the address)'}."
        ),
        node=addr_components_leaf,
        sources=data.hq_address_sources,
        additional_instruction=(
            "Verify the page explicitly lists the full street address on Park Avenue in New York, NY and shows a valid ZIP code "
            "(e.g., 100xx). Fuzzy match is allowed for formatting (‘NY’ vs ‘New York, NY’)."
        ),
    )

    # Verify the source is official
    addr_source_leaf = evaluator.add_leaf(
        id="address_source",
        desc="Provide URL from official sources confirming the headquarters address",
        parent=addr_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The provided webpage is an official institutional source (e.g., corporate site, investor relations, "
            "or press release) that confirms the headquarters address."
        ),
        node=addr_source_leaf,
        sources=data.hq_address_sources,
        additional_instruction="Confirm institutional ownership via branding/footer and explicit address mention.",
    )

    # ---------------- Building Characteristics ----------------
    bldg_node = evaluator.add_parallel(
        id="building_characteristics",
        desc="Verify the headquarters building occupies a full city block",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty_list(data.building_sources),
        id="building_source_provided",
        desc="At least one official URL is provided confirming the building’s specifications",
        parent=bldg_node,
        critical=True,
    )

    boundaries_leaf = evaluator.add_leaf(
        id="city_block_boundaries",
        desc="Confirm building is bounded by 47th Street, 48th Street, Madison Avenue, and Park Avenue",
        parent=bldg_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The new headquarters building occupies a full city block bounded by 47th Street, 48th Street, "
            "Madison Avenue, and Park Avenue."
        ),
        node=boundaries_leaf,
        sources=data.building_sources,
        additional_instruction="Verify the page explicitly states these four boundaries, implying a full-block footprint.",
    )

    bldg_source_leaf = evaluator.add_leaf(
        id="building_source",
        desc="Provide URL confirming the building's specifications",
        parent=bldg_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The provided webpage confirms the building’s footprint/specifications and is an official institutional page.",
        node=bldg_source_leaf,
        sources=data.building_sources,
        additional_instruction="Prefer official press releases or corporate pages describing the new HQ project.",
    )

    # ---------------- Opening Timeline ----------------
    open_node = evaluator.add_parallel(
        id="opening_timeline",
        desc="Verify the new headquarters opened in 2025",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty_list(data.opening_sources),
        id="opening_source_provided",
        desc="At least one official URL is provided confirming the 2025 opening date",
        parent=open_node,
        critical=True,
    )

    opened_leaf = evaluator.add_leaf(
        id="opened_in_2025",
        desc="Confirm the headquarters officially opened in 2025",
        parent=open_node,
        critical=True,
    )
    open_year = data.opening_year or "2025"
    await evaluator.verify(
        claim=f"The new global headquarters officially opened in {open_year}.",
        node=opened_leaf,
        sources=data.opening_sources,
        additional_instruction="Verify the page states the opening occurred in calendar year 2025.",
    )

    opening_src_leaf = evaluator.add_leaf(
        id="opening_source",
        desc="Provide URL confirming the 2025 opening date",
        parent=open_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The provided webpage is an official institutional source confirming the 2025 opening.",
        node=opening_src_leaf,
        sources=data.opening_sources,
        additional_instruction=(
            "Prefer investor relations or press releases dated around 2024/2025. Confirm 2025 opening is stated."
        ),
    )


async def _verify_asset_management_scale(evaluator: Evaluator, parent, data: InstitutionExtraction):
    node = evaluator.add_parallel(
        id="asset_management_scale",
        desc="Verify asset management scale meets required thresholds",
        parent=parent,
        critical=True,
    )

    # ---------------- Real Estate AUM ----------------
    rea_node = evaluator.add_parallel(
        id="real_estate_assets",
        desc="Verify real estate asset management exceeds $75 billion as of 2024/early 2025",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(data.real_estate_aum is not None and data.real_estate_aum.strip() != ""),
        id="real_estate_value_provided",
        desc="Real estate AUM value is provided in the answer",
        parent=rea_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_list(data.real_estate_aum_sources),
        id="real_estate_source_provided",
        desc="At least one official URL is provided documenting real estate AUM",
        parent=rea_node,
        critical=True,
    )

    rea_thresh_leaf = evaluator.add_leaf(
        id="real_estate_threshold_met",
        desc="Provide real estate AUM figure and confirm it exceeds $75 billion",
        parent=rea_node,
        critical=True,
    )
    rea_val = data.real_estate_aum or ""
    await evaluator.verify(
        claim=f"The institution's real estate AUM is '{rea_val}' and exceeds $75 billion.",
        node=rea_thresh_leaf,
        sources=data.real_estate_aum_sources,
        additional_instruction=(
            "Verify the page quotes the real estate AUM figure and that it is greater than $75B. "
            "Prefer investor relations or official press releases dated in 2024 or early 2025."
        ),
    )

    rea_src_leaf = evaluator.add_leaf(
        id="real_estate_source",
        desc="Provide URL from official investor materials documenting real estate AUM",
        parent=rea_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The provided webpage is an official investor relations or press release page of the institution, "
            "dated in 2024 or early 2025, that documents the real estate AUM figure."
        ),
        node=rea_src_leaf,
        sources=data.real_estate_aum_sources,
        additional_instruction=(
            "Check that the page clearly belongs to the institution and displays a date in 2024 or early 2025."
        ),
    )

    # ---------------- Total AUM ----------------
    taum_node = evaluator.add_parallel(
        id="total_aum",
        desc="Verify total assets under management exceed $3.5 trillion as of 2024/early 2025",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(data.total_aum is not None and data.total_aum.strip() != ""),
        id="total_aum_value_provided",
        desc="Total AUM value is provided in the answer",
        parent=taum_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_list(data.total_aum_sources),
        id="total_aum_source_provided",
        desc="At least one official URL is provided confirming total AUM",
        parent=taum_node,
        critical=True,
    )

    taum_thresh_leaf = evaluator.add_leaf(
        id="total_aum_threshold_met",
        desc="Provide total AUM figure and confirm it exceeds $3.5 trillion",
        parent=taum_node,
        critical=True,
    )
    taum_val = data.total_aum or ""
    await evaluator.verify(
        claim=f"The institution's total AUM is '{taum_val}' and exceeds $3.5 trillion.",
        node=taum_thresh_leaf,
        sources=data.total_aum_sources,
        additional_instruction=(
            "Verify the page states the total AUM and that it is greater than $3.5T. "
            "Prefer investor relations or official press releases dated in 2024 or early 2025."
        ),
    )

    taum_src_leaf = evaluator.add_leaf(
        id="total_aum_source",
        desc="Provide URL from official sources confirming total AUM",
        parent=taum_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The provided webpage is an official institutional page (investor relations or press release) "
            "dated in 2024 or early 2025 confirming the total AUM figure."
        ),
        node=taum_src_leaf,
        sources=data.total_aum_sources,
        additional_instruction="Confirm the page is official and dated 2024 or early 2025.",
    )


async def _verify_organizational_structure(evaluator: Evaluator, parent, data: InstitutionExtraction):
    node = evaluator.add_parallel(
        id="organizational_structure",
        desc="Verify the institution has required organizational structure including dedicated Asset & Wealth Management division",
        parent=parent,
        critical=True,
    )

    # Ensure division source exists
    evaluator.add_custom_node(
        result=_non_empty_list(data.division_sources),
        id="division_source_provided",
        desc="At least one official URL is provided for the Asset & Wealth Management division",
        parent=node,
        critical=True,
    )

    # Asset & Wealth Management division verified
    div_leaf = evaluator.add_leaf(
        id="asset_management_division_verified",
        desc="Confirm institution has Asset & Wealth Management division as primary operating segment and provide URL documenting this structure",
        parent=node,
        critical=True,
    )
    div_name = data.division_name or "Asset & Wealth Management"
    await evaluator.verify(
        claim=(
            f"The institution has a division named (or equivalent to) '{div_name}', representing an Asset & Wealth Management "
            f"operating segment."
        ),
        node=div_leaf,
        sources=data.division_sources,
        additional_instruction=(
            "Allow minor naming variants (e.g., 'Asset Management' & 'Wealth Management' combined). Verify it is an official operating segment."
        ),
    )

    # Multiple business segments (mark critical True to satisfy framework constraint for children of critical parent)
    # Add a source existence gate for segments if provided
    evaluator.add_custom_node(
        result=_non_empty_list(data.segments_sources),
        id="segments_source_provided",
        desc="At least one official URL is provided documenting multiple operating segments (if claimed)",
        parent=node,
        critical=True,
    )

    seg_leaf = evaluator.add_leaf(
        id="multiple_business_segments",
        desc="Confirm institution has multiple distinct operating segments covering banking and investment activities and provide URL documenting segment structure",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The institution has multiple distinct operating segments covering both banking and investment activities "
            "(e.g., Consumer/Community Banking, Corporate & Investment Bank, Asset & Wealth Management, etc.)."
        ),
        node=seg_leaf,
        sources=data.segments_sources,
        additional_instruction="Verify the official segmentation structure on investor relations or annual report pages.",
    )


# ------------------------------------------------------------------------------------
# Main Evaluation Function
# ------------------------------------------------------------------------------------
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
        strategy=AggregationStrategy.SEQUENTIAL,  # root node follows sequential per rubric
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_institution_info(),
        template_class=InstitutionExtraction,
        extraction_name="institution_extraction",
    )

    # Build top-level structure to match rubric tree
    # 1) Institution Identification (critical)
    await _verify_institution_identification(evaluator, root, extracted)

    # 2) Property Verification (critical, parallel)
    prop_node = evaluator.add_parallel(
        id="property_verification",
        desc="Verify all required properties of the identified institution",
        parent=root,
        critical=True,
    )

    # 2.1) Headquarters location checks
    await _verify_headquarters_location(evaluator, prop_node, extracted)

    # 2.2) Asset management scale checks
    await _verify_asset_management_scale(evaluator, prop_node, extracted)

    # 2.3) Organizational structure checks
    await _verify_organizational_structure(evaluator, prop_node, extracted)

    # Return summary
    return evaluator.get_summary()