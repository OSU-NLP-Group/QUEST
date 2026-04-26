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
TASK_ID = "denver_class_a_office_2026"
TASK_DESCRIPTION = """
Identify a Class A commercial office building in Denver, Colorado that meets all of the following criteria:
(1) minimum gross floor area of 50,000 square feet,
(2) LEED certified at Silver level or higher,
(3) professionally managed by a recognized commercial real estate firm,
(4) has office space available for lease in 2026, and
(5) offers lease terms of at least 5 years.
Provide the building name and address.
"""

RECOGNIZED_FIRMS_EXAMPLES = [
    "CBRE", "JLL", "Cushman & Wakefield", "Colliers", "Hines", "Brookfield Properties",
    "Transwestern", "Lincoln Property Company", "Prologis", "Newmark", "COPT", "Skanska",
    "RMR", "Equity Office", "Kilroy Realty", "Hudson Pacific", "Tishman Speyer"
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DenverOfficeExtraction(BaseModel):
    """Structured extraction of the building and criterion-specific evidence from the answer."""
    building_name: Optional[str] = None
    building_address: Optional[str] = None

    # Class A evidence
    class_a_urls: List[str] = Field(default_factory=list)

    # Denver location evidence
    location_urls: List[str] = Field(default_factory=list)

    # Gross floor area evidence
    gfa_text: Optional[str] = None
    gfa_urls: List[str] = Field(default_factory=list)

    # LEED certification evidence
    leed_level_text: Optional[str] = None  # e.g., "LEED Silver", "LEED Gold", "LEED Platinum"
    leed_urls: List[str] = Field(default_factory=list)

    # Professional management evidence
    management_firm_name: Optional[str] = None
    management_urls: List[str] = Field(default_factory=list)

    # 2026 availability evidence
    lease_availability_2026_text: Optional[str] = None  # e.g., "Available in Q3 2026"
    lease_availability_urls: List[str] = Field(default_factory=list)

    # Lease terms evidence
    lease_term_min_text: Optional[str] = None  # e.g., "Minimum 5-year term"
    lease_terms_urls: List[str] = Field(default_factory=list)

    # Permanent location evidence (LEED MPR #1)
    permanent_location_urls: List[str] = Field(default_factory=list)

    # Optional primary property URL to serve as general fallback if provided in the answer
    primary_property_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_building_info() -> str:
    return """
    You must extract structured information about a single building referenced in the answer that the answer claims meets all criteria.
    If multiple buildings are mentioned, select the first one the answer presents as satisfying the criteria.

    Extract the following fields exactly as they appear in the answer text:

    1) building_name: The building name (string). If not provided, return null.
    2) building_address: The building street address including city and state if present (string). If not provided, return null.

    For each criterion below, extract both any textual value/statement and all source URLs explicitly provided in the answer that support that criterion. Return empty lists for URLs if none are provided. Do not invent URLs.

    3) class_a_urls: Array of URLs that support the building being Class A.
    4) location_urls: Array of URLs that support the building being located within Denver, Colorado city limits.
    5) gfa_text: The gross floor area statement as text (e.g., "200,000 sq ft" or "approx. 60,000 sf"). If missing, return null.
    6) gfa_urls: Array of URLs that support the gross floor area.
    7) leed_level_text: The LEED certification level as text (e.g., "LEED Silver", "LEED Gold"). If missing, return null.
    8) leed_urls: Array of URLs that support the LEED certification claim.
    9) management_firm_name: The name of the professional management firm (string). If missing, return null.
    10) management_urls: Array of URLs that support the professional management claim.
    11) lease_availability_2026_text: Any direct statement indicating office space is available for lease in 2026 (string). If missing, return null.
    12) lease_availability_urls: Array of URLs that support 2026 availability.
    13) lease_term_min_text: Any direct statement about lease term length (e.g., "min 5-year term") (string). If missing, return null.
    14) lease_terms_urls: Array of URLs that support lease term length of at least 5 years.
    15) permanent_location_urls: Array of URLs that support the building being a permanent location on existing land (LEED MPR #1). Often this can be supported by a property website, a city record, or a listing that shows a physical address.

    16) primary_property_url: If the answer presents a single main property page URL (e.g., the building's official page), extract it here. If not provided, return null.

    URL extraction rules:
    - Extract only URLs that are explicitly present in the answer (plain URLs or URLs inside markdown links). Do not infer or construct URLs.
    - Return complete URLs. If a URL is missing a protocol (http/https), prepend http://.
    - If a criterion has no URLs cited in the answer, return an empty array for that criterion.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _policy_instruction_base() -> str:
    return (
        "Source-grounding policy: Only judge as 'supported' if the provided webpages explicitly support the claim. "
        "If no URLs are provided, or if the URLs are irrelevant/inaccessible, treat the claim as NOT SUPPORTED."
    )


def _merge_instruction(base: str) -> str:
    """Append the source-grounding policy to a base instruction."""
    return f"{base}\n\n{_policy_instruction_base()}"


def _fallback_sources(primary_list: List[str], fallback: Optional[str]) -> List[str] | None:
    """Return primary_list if non-empty; otherwise return [fallback] if provided; otherwise None."""
    if primary_list and len(primary_list) > 0:
        return primary_list
    if fallback and fallback.strip():
        return [fallback]
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_required_fields_subtree(
    evaluator: Evaluator,
    parent_node,
    ext: DenverOfficeExtraction,
) -> None:
    """Add required fields existence checks (critical)."""
    rf_node = evaluator.add_parallel(
        id="Response_Contains_Required_Fields",
        desc="Response must provide the requested identifying information.",
        parent=parent_node,
        critical=True,
    )

    # Building name provided (critical)
    evaluator.add_custom_node(
        result=(ext.building_name is not None and ext.building_name.strip() != ""),
        id="Building_Name_Provided",
        desc="Provide the building name.",
        parent=rf_node,
        critical=True,
    )

    # Building address provided (critical)
    evaluator.add_custom_node(
        result=(ext.building_address is not None and ext.building_address.strip() != ""),
        id="Building_Address_Provided",
        desc="Provide the building address.",
        parent=rf_node,
        critical=True,
    )


async def build_constraints_subtree(
    evaluator: Evaluator,
    parent_node,
    ext: DenverOfficeExtraction,
) -> None:
    """Add all constraint verification leaves (critical) and perform batch verification."""
    cons_node = evaluator.add_parallel(
        id="Building_Meets_All_Stated_Criteria_And_Constraints",
        desc="Chosen building satisfies all criteria in the proposed question and all listed constraints.",
        parent=parent_node,
        critical=True,
    )

    # Prepare leaf nodes
    class_a_node = evaluator.add_leaf(
        id="Class_A_Commercial_Office_Building",
        desc="The property is a Class A commercial office building (highest quality category).",
        parent=cons_node,
        critical=True,
    )

    location_node = evaluator.add_leaf(
        id="Located_Within_Denver_City_Limits",
        desc="The building is physically located within Denver, Colorado city limits.",
        parent=cons_node,
        critical=True,
    )

    gfa_node = evaluator.add_leaf(
        id="Minimum_Gross_Floor_Area_50k",
        desc="Gross floor area is at least 50,000 square feet.",
        parent=cons_node,
        critical=True,
    )

    leed_node = evaluator.add_leaf(
        id="LEED_Silver_Or_Higher",
        desc="Building holds LEED certification at Silver level or higher (Silver/Gold/Platinum).",
        parent=cons_node,
        critical=True,
    )

    mgmt_node = evaluator.add_leaf(
        id="Professionally_Managed_By_Recognized_Firm",
        desc="Building is professionally managed by a recognized commercial real estate management firm.",
        parent=cons_node,
        critical=True,
    )

    avail2026_node = evaluator.add_leaf(
        id="Office_Space_Available_For_Lease_In_2026",
        desc="Building has office space available for lease in 2026 (currently available or becoming available in 2026).",
        parent=cons_node,
        critical=True,
    )

    lease5_node = evaluator.add_leaf(
        id="Lease_Terms_At_Least_5_Years",
        desc="Building offers lease terms of at least 5 years.",
        parent=cons_node,
        critical=True,
    )

    permanent_node = evaluator.add_leaf(
        id="Permanent_Location_On_Existing_Land",
        desc="The building is on a permanent location on existing land (LEED Minimum Program Requirement #1).",
        parent=cons_node,
        critical=True,
    )

    # Build claims and sources
    name = (ext.building_name or "the building").strip()
    address = (ext.building_address or "").strip()

    # Class A claim
    class_a_claim = f"The building '{name}' is a Class A commercial office building."
    class_a_add_ins = _merge_instruction(
        "Verify that the provided webpages explicitly state the building is 'Class A'. "
        "Allow reasonable synonyms like 'Class A office tower' or 'trophy office' when clearly equivalent. "
        "Do not accept 'Class B' or 'Class C'."
    )
    class_a_sources = _fallback_sources(ext.class_a_urls, ext.primary_property_url)

    # Location (Denver city limits) claim
    if address:
        location_claim = f"The address '{address}' is within the city limits of Denver, Colorado."
    else:
        location_claim = "The building is located within the city limits of Denver, Colorado."
    location_add_ins = _merge_instruction(
        "Confirm the webpages explicitly show the building's address in 'Denver, CO' or otherwise clearly state "
        "the building is in Denver city limits. "
        "Neighborhood labels are acceptable only if they are within Denver. If the page indicates another city or "
        "an unincorporated area, treat as NOT SUPPORTED."
    )
    location_sources = _fallback_sources(ext.location_urls, ext.primary_property_url)

    # Gross floor area >= 50,000 sf claim
    gfa_claim = "The building has a gross floor area of at least 50,000 square feet."
    gfa_add_ins = _merge_instruction(
        "Check the webpages for square footage. Accept >= 50,000 sf. "
        "Allow reasonable formatting variants like 'sq ft', 'SF', or comma separators, and accept approximate statements "
        "that clearly exceed 50,000 sf."
    )
    gfa_sources = _fallback_sources(ext.gfa_urls, ext.primary_property_url)

    # LEED Silver or higher claim
    if ext.leed_level_text and ext.leed_level_text.strip():
        leed_claim = f"The building holds LEED {ext.leed_level_text.strip()} certification, which is Silver level or higher."
    else:
        leed_claim = "The building holds LEED certification at Silver level or higher (Silver, Gold, or Platinum)."
    leed_add_ins = _merge_instruction(
        "Confirm that the webpages explicitly state the building's LEED certification level and that it is Silver or higher. "
        "Accept 'LEED Silver', 'LEED Gold', or 'LEED Platinum' (including versioned labels like 'LEED v4 Gold')."
    )
    leed_sources = _fallback_sources(ext.leed_urls, ext.primary_property_url)

    # Professionally managed by recognized firm claim
    if ext.management_firm_name and ext.management_firm_name.strip():
        mgmt_claim = (
            f"The building is professionally managed by '{ext.management_firm_name.strip()}', "
            f"which is a recognized commercial real estate management firm."
        )
    else:
        mgmt_claim = "The building is professionally managed by a recognized commercial real estate management firm."
    mgmt_add_ins = _merge_instruction(
        "Verify the webpages explicitly identify the building's professional property management firm. "
        "Consider 'recognized' satisfied when the named firm is a widely known regional/national CRE manager (e.g., "
        + ", ".join(RECOGNIZED_FIRMS_EXAMPLES)
        + "). If the management is self-managed by a tenant/owner with no evidence of recognized CRE management, treat as NOT SUPPORTED."
    )
    mgmt_sources = _fallback_sources(ext.management_urls, ext.primary_property_url)

    # Office space available for lease in 2026 claim
    avail2026_claim = "The building has office space available for lease in 2026."
    avail2026_add_ins = _merge_instruction(
        "Confirm the webpages indicate availability in 2026 (e.g., 'Available 2026', 'Delivering 2026', "
        "'Expected availability in 2026', or listing availability dates that include the year 2026). "
        "If the pages only mention past or different years, treat as NOT SUPPORTED."
    )
    avail2026_sources = _fallback_sources(ext.lease_availability_urls, ext.primary_property_url)

    # Lease terms at least 5 years claim
    lease5_claim = "The building offers lease terms of at least 5 years."
    lease5_add_ins = _merge_instruction(
        "Check the webpages for lease term language such as 'minimum 5-year term', '5 years or longer', "
        "'60 months', or similar. If no term information is provided, treat as NOT SUPPORTED."
    )
    lease5_sources = _fallback_sources(ext.lease_terms_urls, ext.primary_property_url)

    # Permanent location on existing land (LEED MPR #1) claim
    permanent_claim = "The building is a permanent location on existing land (LEED Minimum Program Requirement #1)."
    permanent_add_ins = _merge_instruction(
        "Use property webpages, city records, or listings to confirm the building exists at a fixed physical address. "
        "If the pages suggest a temporary/relocatable structure or do not indicate a physical building on existing land, "
        "treat as NOT SUPPORTED."
    )
    permanent_sources = _fallback_sources(ext.permanent_location_urls, ext.primary_property_url)

    claims_and_sources = [
        (class_a_claim, class_a_sources, class_a_node, class_a_add_ins),
        (location_claim, location_sources, location_node, location_add_ins),
        (gfa_claim, gfa_sources, gfa_node, gfa_add_ins),
        (leed_claim, leed_sources, leed_node, leed_add_ins),
        (mgmt_claim, mgmt_sources, mgmt_node, mgmt_add_ins),
        (avail2026_claim, avail2026_sources, avail2026_node, avail2026_add_ins),
        (lease5_claim, lease5_sources, lease5_node, lease5_add_ins),
        (permanent_claim, permanent_sources, permanent_node, permanent_add_ins),
    ]

    # Execute verifications in parallel
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
    Evaluate an answer for the Denver Class A office building task.
    """
    # Initialize evaluator (root is non-critical; we add a critical top-level node below)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Criteria can be checked independently
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_building_info(),
        template_class=DenverOfficeExtraction,
        extraction_name="building_candidate",
    )

    # Top-level critical node for the overall identification task
    top_node = evaluator.add_parallel(
        id="Denver_Class_A_Office_Building_Identification",
        desc="Identify a Class A commercial office building in Denver, Colorado that satisfies all stated criteria/constraints, and provide the building name and address.",
        parent=root,
        critical=True,
    )

    # Subtree: Required fields
    await build_required_fields_subtree(evaluator, top_node, extraction)

    # Subtree: Constraints and criteria
    await build_constraints_subtree(evaluator, top_node, extraction)

    # Optional: record some custom info handy for debugging
    evaluator.add_custom_info(
        info={
            "extracted_name": extraction.building_name,
            "extracted_address": extraction.building_address,
            "primary_property_url": extraction.primary_property_url,
            "url_counts": {
                "class_a": len(extraction.class_a_urls),
                "location": len(extraction.location_urls),
                "gfa": len(extraction.gfa_urls),
                "leed": len(extraction.leed_urls),
                "management": len(extraction.management_urls),
                "availability_2026": len(extraction.lease_availability_urls),
                "lease_terms": len(extraction.lease_terms_urls),
                "permanent_location": len(extraction.permanent_location_urls),
            },
        },
        info_type="debug",
        info_name="extraction_debug_summary",
    )

    # Return structured summary
    return evaluator.get_summary()