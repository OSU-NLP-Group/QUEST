import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sg_reit_malls"
TASK_DESCRIPTION = """
Identify 4 retail shopping malls in Singapore that meet ALL of the following criteria:

1. REIT Ownership: The mall must be owned by a Singapore-listed Real Estate Investment Trust (REIT).

2. Property Type and Location: The property must be a retail shopping mall located in Singapore.

3. Minimum Size Requirements:
   - The mall must have a Retail Gross Floor Area (GFA) of at least 7,000 square meters.
   - The mall must have a Net Lettable Area (NLA) of at least 4,600 square meters.

4. Physical Structure: The mall must have at least 3 storeys (levels) of retail space.

5. Tenant Requirements:
   - The mall must have at least 100 retail stores/units.
   - The mall must have at least one anchor tenant, which can be a supermarket, hypermarket, department store, or cinema.

6. Green Building Certification: The mall must hold BCA Green Mark certification at any level (Certified, Gold, GoldPlus, or Platinum).

7. Public Transport Connectivity: The mall must be located within 400 meters walking distance of an MRT (Mass Rapid Transit) station.

8. Regulatory Compliance:
   - The mall must comply with SCDF (Singapore Civil Defence Force) fire safety requirements, including automatic sprinkler systems where required based on building size thresholds.
   - The mall must comply with BCA Code on Accessibility in the Built Environment, including wheelchair-accessible entrances and routes.

For each of the 4 malls you identify, provide:
- The name of the mall
- The name of the REIT that owns it
- Supporting information demonstrating that it meets each of the specified criteria
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MallItem(BaseModel):
    # Basic identification
    name: Optional[str] = None
    reit: Optional[str] = None
    property_type: Optional[str] = None
    location: Optional[str] = None  # e.g., "Singapore", "Jurong, Singapore", etc.

    # Size and structure
    gfa_sqm: Optional[str] = None  # Retail GFA; string to allow ranges or text
    nla_sqm: Optional[str] = None  # NLA; string to allow ranges or text
    storeys: Optional[str] = None  # number of retail levels as string

    # Tenants
    num_stores: Optional[str] = None  # count or textual
    anchor_tenants: List[str] = Field(default_factory=list)
    anchor_categories: List[str] = Field(default_factory=list)  # supermarket/hypermarket/department store/cinema

    # Green building
    green_mark_level: Optional[str] = None  # Certified/Gold/GoldPlus/Platinum

    # Connectivity
    mrt_station: Optional[str] = None
    mrt_distance_m: Optional[str] = None  # walking distance in meters (string)
    google_map_url: Optional[str] = None  # optional Google Maps walking route URL

    # Safety & Accessibility
    scdf_fire_safety: Optional[str] = None  # textual affirmation or certificate reference
    sprinkler: Optional[str] = None  # textual affirmation of sprinkler system presence (if stated)
    bca_accessibility: Optional[str] = None  # textual affirmation of accessibility compliance

    # Grouped evidence URLs
    ownership_urls: List[str] = Field(default_factory=list)
    size_urls: List[str] = Field(default_factory=list)
    tenant_urls: List[str] = Field(default_factory=list)
    green_mark_urls: List[str] = Field(default_factory=list)
    connectivity_urls: List[str] = Field(default_factory=list)
    safety_urls: List[str] = Field(default_factory=list)


class MallExtraction(BaseModel):
    malls: List[MallItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_malls() -> str:
    return """
    Extract up to 6 retail shopping malls in Singapore (limit to those the answer mentions) with the following fields for each mall. Return them in an array `malls`. If the answer mentions more than 4, include all mentioned but we will only evaluate the first 4 later. If any field is missing, set it to null or an empty array as appropriate.

    For each mall, extract:
    - name: The mall name
    - reit: The name of the Singapore-listed REIT that owns the mall
    - property_type: The property type (should be "retail mall" or equivalent wording)
    - location: The location text (must indicate it is in Singapore)
    - gfa_sqm: Retail Gross Floor Area value (text as presented, e.g., "8,500 sqm" or "approx. 10,000 sq m")
    - nla_sqm: Net Lettable Area value (text as presented)
    - storeys: Number of retail storeys/levels
    - num_stores: Number of retail stores/units (text as presented)
    - anchor_tenants: A list of anchor tenant names (if mentioned)
    - anchor_categories: A list of anchor categories present (e.g., ["supermarket", "cinema", "department store", "hypermarket"])
    - green_mark_level: The BCA Green Mark level (Certified, Gold, GoldPlus, or Platinum)
    - mrt_station: The nearest MRT station name (if mentioned)
    - mrt_distance_m: The walking distance in meters to the MRT station (text as presented; e.g., "300m")
    - google_map_url: A Google Maps route or place URL (if the answer includes one for walking distance verification)
    - scdf_fire_safety: Text indicating SCDF fire safety compliance (e.g., "SCDF Fire Certificate" or "complies with SCDF fire safety")
    - sprinkler: Text indicating automatic sprinkler systems presence (if stated)
    - bca_accessibility: Text indicating BCA accessibility compliance (e.g., "complies with BCA Code on Accessibility", "wheelchair accessible")

    Also extract grouped evidence URLs for each set of claims (include all URLs the answer cites for that aspect):
    - ownership_urls: URLs specifically supporting REIT ownership and property details
    - size_urls: URLs supporting GFA/NLA/storeys size specifications
    - tenant_urls: URLs supporting tenant info (number of stores, anchor tenants)
    - green_mark_urls: URLs supporting Green Mark certification
    - connectivity_urls: URLs supporting MRT connectivity and walking distance
    - safety_urls: URLs supporting SCDF fire safety and BCA accessibility compliance

    IMPORTANT:
    - Only include URLs that are explicitly present in the answer (plain URLs or markdown links). Do not invent or infer URLs.
    - If a URL is missing protocol, prepend http://
    - If a specific group has no URLs in the answer, return an empty array for that group.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: List[str], single_url: Optional[str] = None) -> List[str]:
    """Combine multiple URL lists and an optional single URL, deduplicate, preserve order."""
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in lst:
            if not isinstance(u, str):
                continue
            uu = u.strip()
            if not uu:
                continue
            if uu not in seen:
                seen.add(uu)
                out.append(uu)
    if single_url and isinstance(single_url, str) and single_url.strip():
        uu = single_url.strip()
        if uu not in seen:
            out.append(uu)
    return out


def _safe_name(mall: MallItem) -> str:
    return mall.name or "the mall"


def _safe_reit(mall: MallItem) -> str:
    return mall.reit or "the REIT"


# --------------------------------------------------------------------------- #
# Verification subtrees per mall                                              #
# --------------------------------------------------------------------------- #
async def _verify_ownership_and_type(evaluator: Evaluator, parent, mall: MallItem, idx: int):
    group = evaluator.add_parallel(
        id=f"Mall_{idx+1}_Ownership_and_Type",
        desc="Verify ownership by Singapore REIT and property type",
        parent=parent,
        critical=True
    )

    # Leaf: REIT Ownership (including SG-listing requirement)
    reit_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_REIT_Ownership",
        desc="The property is owned by a Singapore-listed REIT",
        parent=group,
        critical=True
    )
    claim_reit = (
        f"The property '{_safe_name(mall)}' is owned by the REIT '{_safe_reit(mall)}', "
        f"and '{_safe_reit(mall)}' is listed on the Singapore Exchange (SGX)."
    )
    await evaluator.verify(
        claim=claim_reit,
        node=reit_leaf,
        sources=mall.ownership_urls,
        additional_instruction="Use only the provided URLs to confirm both ownership and SGX listing. "
                               "If URLs are missing or do not explicitly support ownership and SGX listing, mark as not supported."
    )

    # Leaf: Property Type
    type_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Property_Type",
        desc="The property is a retail shopping mall",
        parent=group,
        critical=True
    )
    claim_type = f"The property '{_safe_name(mall)}' is a retail shopping mall."
    await evaluator.verify(
        claim=claim_type,
        node=type_leaf,
        sources=mall.ownership_urls,
        additional_instruction="Allow reasonable synonyms like 'retail mall' or 'shopping centre'. "
                               "If sources do not clearly indicate it is a retail shopping mall, mark incorrect."
    )

    # Leaf: Singapore Location
    loc_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Singapore_Location",
        desc="The property is located in Singapore",
        parent=group,
        critical=True
    )
    claim_loc = f"The property '{_safe_name(mall)}' is located in Singapore."
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=_combine_sources(mall.ownership_urls, mall.connectivity_urls),
        additional_instruction="Confirm from provided sources. If location is ambiguous or not in Singapore, mark incorrect."
    )

    # Leaf: Ownership Reference (explicit support)
    ref_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Ownership_Reference",
        desc="URL reference supporting REIT ownership and property details",
        parent=group,
        critical=True
    )
    claim_ref = (
        f"The provided sources explicitly state that '{_safe_name(mall)}' is owned by '{_safe_reit(mall)}' "
        f"and include property details (name, type, location)."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=mall.ownership_urls,
        additional_instruction="At least one source must directly and explicitly state REIT ownership and basic property details. "
                               "If no such URL is provided, mark incorrect."
    )


async def _verify_size_requirements(evaluator: Evaluator, parent, mall: MallItem, idx: int):
    group = evaluator.add_parallel(
        id=f"Mall_{idx+1}_Size_Requirements",
        desc="Verify minimum size requirements are met",
        parent=parent,
        critical=True
    )

    # Leaf: Retail GFA >= 7000 sqm
    gfa_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Retail_GFA",
        desc="Retail Gross Floor Area is at least 7,000 square meters",
        parent=group,
        critical=True
    )
    claim_gfa = f"The mall '{_safe_name(mall)}' has a Retail Gross Floor Area (GFA) of at least 7,000 square meters."
    await evaluator.verify(
        claim=claim_gfa,
        node=gfa_leaf,
        sources=mall.size_urls,
        additional_instruction="Look for GFA numbers or statements in the provided URLs. "
                               "Allow minor rounding. If no explicit GFA is provided, mark incorrect."
    )

    # Leaf: NLA >= 4600 sqm
    nla_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_NLA",
        desc="Net Lettable Area is at least 4,600 square meters",
        parent=group,
        critical=True
    )
    claim_nla = f"The mall '{_safe_name(mall)}' has a Net Lettable Area (NLA) of at least 4,600 square meters."
    await evaluator.verify(
        claim=claim_nla,
        node=nla_leaf,
        sources=mall.size_urls,
        additional_instruction="Look for NLA numbers or statements in the provided URLs. "
                               "Allow minor rounding. If no explicit NLA is provided, mark incorrect."
    )

    # Leaf: Multi-storey >= 3
    storey_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Multi_Storey",
        desc="The mall has at least 3 storeys of retail space",
        parent=group,
        critical=True
    )
    claim_storeys = f"The mall '{_safe_name(mall)}' has at least 3 storeys (levels) of retail space."
    await evaluator.verify(
        claim=claim_storeys,
        node=storey_leaf,
        sources=mall.size_urls,
        additional_instruction="Confirm number of storeys from provided URLs. If unclear or fewer than 3, mark incorrect."
    )

    # Leaf: Size Reference
    size_ref_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Size_Reference",
        desc="URL reference supporting size specifications",
        parent=group,
        critical=True
    )
    claim_size_ref = f"The provided sources explicitly support the mall size specifications (GFA/NLA/storeys) for '{_safe_name(mall)}'."
    await evaluator.verify(
        claim=claim_size_ref,
        node=size_ref_leaf,
        sources=mall.size_urls,
        additional_instruction="At least one source must explicitly state the size specs. If no such URL is provided, mark incorrect."
    )


async def _verify_tenant_requirements(evaluator: Evaluator, parent, mall: MallItem, idx: int):
    group = evaluator.add_parallel(
        id=f"Mall_{idx+1}_Tenant_Requirements",
        desc="Verify tenant-related requirements",
        parent=parent,
        critical=True
    )

    # Leaf: Minimum stores >= 100
    stores_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Minimum_Stores",
        desc="The mall has at least 100 retail stores/units",
        parent=group,
        critical=True
    )
    claim_stores = f"The mall '{_safe_name(mall)}' has at least 100 retail stores or units."
    await evaluator.verify(
        claim=claim_stores,
        node=stores_leaf,
        sources=mall.tenant_urls,
        additional_instruction="Confirm explicit store/unit count from provided URLs. If not clearly ≥100, mark incorrect."
    )

    # Leaf: Anchor tenant present (category requirement)
    anchor_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Anchor_Tenant",
        desc="The mall has at least one anchor tenant (supermarket, hypermarket, department store, or cinema)",
        parent=group,
        critical=True
    )
    categories = ", ".join(mall.anchor_categories) if mall.anchor_categories else "none"
    tenants = ", ".join(mall.anchor_tenants) if mall.anchor_tenants else "none"
    claim_anchor = (
        f"The mall '{_safe_name(mall)}' has at least one anchor tenant; "
        f"qualifying anchor categories include supermarket, hypermarket, department store, or cinema. "
        f"Named anchors: {tenants}. Categories mentioned: {categories}."
    )
    await evaluator.verify(
        claim=claim_anchor,
        node=anchor_leaf,
        sources=mall.tenant_urls,
        additional_instruction="Verify that at least one anchor is present and is one of the allowed categories. "
                               "If anchors are not clearly stated or not of allowed types, mark incorrect."
    )

    # Leaf: Tenant Reference
    tenant_ref_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Tenant_Reference",
        desc="URL reference supporting tenant information",
        parent=group,
        critical=True
    )
    claim_tenant_ref = f"The provided sources explicitly support tenant information (store count and anchor tenants) for '{_safe_name(mall)}'."
    await evaluator.verify(
        claim=claim_tenant_ref,
        node=tenant_ref_leaf,
        sources=mall.tenant_urls,
        additional_instruction="At least one source must explicitly state store count and/or anchors. If none, mark incorrect."
    )


async def _verify_green_building(evaluator: Evaluator, parent, mall: MallItem, idx: int):
    group = evaluator.add_parallel(
        id=f"Mall_{idx+1}_Green_Building",
        desc="Verify BCA Green Mark certification",
        parent=parent,
        critical=True
    )

    # Leaf: Green Mark Certification
    gm_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Green_Mark_Certification",
        desc="The property holds BCA Green Mark certification (Certified, Gold, GoldPlus, or Platinum)",
        parent=group,
        critical=True
    )
    level_text = mall.green_mark_level or "unspecified level"
    claim_gm = f"The mall '{_safe_name(mall)}' holds BCA Green Mark certification ({level_text})."
    await evaluator.verify(
        claim=claim_gm,
        node=gm_leaf,
        sources=mall.green_mark_urls,
        additional_instruction="Confirm any Green Mark level (Certified/Gold/GoldPlus/Platinum) from provided URLs. "
                               "If certification is not explicitly stated, mark incorrect."
    )

    # Leaf: Green Mark Reference
    gm_ref_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Green_Mark_Reference",
        desc="URL reference supporting Green Mark certification",
        parent=group,
        critical=True
    )
    claim_gm_ref = f"The provided sources explicitly support the BCA Green Mark certification for '{_safe_name(mall)}'."
    await evaluator.verify(
        claim=claim_gm_ref,
        node=gm_ref_leaf,
        sources=mall.green_mark_urls,
        additional_instruction="At least one source must directly state Green Mark certification. If none, mark incorrect."
    )


async def _verify_connectivity(evaluator: Evaluator, parent, mall: MallItem, idx: int):
    group = evaluator.add_parallel(
        id=f"Mall_{idx+1}_Connectivity",
        desc="Verify MRT connectivity",
        parent=parent,
        critical=True
    )

    # Leaf: MRT within 400m
    mrt_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_MRT_Access",
        desc="The mall is located within 400 meters walking distance of an MRT station",
        parent=group,
        critical=True
    )
    sources = _combine_sources(mall.connectivity_urls, single_url=mall.google_map_url)
    station_text = mall.mrt_station or "an MRT station"
    claim_mrt = f"The mall '{_safe_name(mall)}' is within 400 meters walking distance of {station_text}."
    await evaluator.verify(
        claim=claim_mrt,
        node=mrt_leaf,
        sources=sources,
        additional_instruction="Use only provided URLs (including any Google Maps route) to confirm walking distance ≤ 400 meters. "
                               "If distance cannot be confirmed, mark incorrect."
    )

    # Leaf: Connectivity Reference
    conn_ref_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Connectivity_Reference",
        desc="URL reference supporting MRT connectivity information",
        parent=group,
        critical=True
    )
    claim_conn_ref = f"The provided sources explicitly support MRT connectivity and walking distance for '{_safe_name(mall)}'."
    await evaluator.verify(
        claim=claim_conn_ref,
        node=conn_ref_leaf,
        sources=sources,
        additional_instruction="At least one source must explicitly indicate proximity to an MRT station with walking distance. If none, mark incorrect."
    )


async def _verify_safety_accessibility(evaluator: Evaluator, parent, mall: MallItem, idx: int):
    group = evaluator.add_parallel(
        id=f"Mall_{idx+1}_Safety_and_Accessibility",
        desc="Verify regulatory compliance for safety and accessibility",
        parent=parent,
        critical=True
    )

    # Leaf: Fire Safety (SCDF + sprinklers where required)
    fire_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Fire_Safety",
        desc="The mall complies with SCDF fire safety requirements including automatic sprinkler systems where required",
        parent=group,
        critical=True
    )
    claim_fire = (
        f"The mall '{_safe_name(mall)}' complies with SCDF fire safety requirements, including automatic sprinkler "
        f"systems where required based on building size thresholds."
    )
    await evaluator.verify(
        claim=claim_fire,
        node=fire_leaf,
        sources=mall.safety_urls,
        additional_instruction="Look for explicit references such as SCDF Fire Certificate, fire safety compliance statements, "
                               "or specific mention of sprinkler systems. If not explicitly supported by provided URLs, mark incorrect."
    )

    # Leaf: Accessibility (BCA Code on Accessibility)
    access_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Accessibility",
        desc="The mall complies with BCA Code on Accessibility including wheelchair-accessible entrances and routes",
        parent=group,
        critical=True
    )
    claim_access = (
        f"The mall '{_safe_name(mall)}' complies with the BCA Code on Accessibility in the Built Environment, including "
        f"wheelchair-accessible entrances and routes."
    )
    await evaluator.verify(
        claim=claim_access,
        node=access_leaf,
        sources=mall.safety_urls,
        additional_instruction="Look for accessibility statements, compliance notes, or certifications in provided URLs. "
                               "If not explicitly supported, mark incorrect."
    )

    # Leaf: Safety Reference
    safety_ref_leaf = evaluator.add_leaf(
        id=f"Mall_{idx+1}_Safety_Reference",
        desc="URL reference supporting safety and accessibility compliance",
        parent=group,
        critical=True
    )
    claim_safety_ref = f"The provided sources explicitly support SCDF fire safety and BCA accessibility compliance for '{_safe_name(mall)}'."
    await evaluator.verify(
        claim=claim_safety_ref,
        node=safety_ref_leaf,
        sources=mall.safety_urls,
        additional_instruction="At least one source must directly indicate compliance (fire safety/accessibility). If none, mark incorrect."
    )


async def verify_single_mall(evaluator: Evaluator, root_mall_node, mall: MallItem, idx: int):
    """
    Build and verify the full criteria subtree for a single mall.
    """
    # Criteria verification block (critical)
    criteria_node = evaluator.add_parallel(
        id=f"Mall_{idx+1}_Criteria_Verification",
        desc=f"Verify that the {'first' if idx==0 else ('second' if idx==1 else ('third' if idx==2 else 'fourth'))} mall meets all required criteria",
        parent=root_mall_node,
        critical=True
    )

    # Ownership and type
    await _verify_ownership_and_type(evaluator, criteria_node, mall, idx)

    # Size requirements
    await _verify_size_requirements(evaluator, criteria_node, mall, idx)

    # Tenant requirements
    await _verify_tenant_requirements(evaluator, criteria_node, mall, idx)

    # Green building certification
    await _verify_green_building(evaluator, criteria_node, mall, idx)

    # Connectivity
    await _verify_connectivity(evaluator, criteria_node, mall, idx)

    # Safety and accessibility
    await _verify_safety_accessibility(evaluator, criteria_node, mall, idx)


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
    Evaluate an answer for the Singapore REIT malls task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # The 4 malls are evaluated independently
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

    # Extract malls data from the answer
    mall_extraction: MallExtraction = await evaluator.extract(
        prompt=prompt_extract_malls(),
        template_class=MallExtraction,
        extraction_name="mall_extraction"
    )

    # Limit to first 4 malls; pad if fewer
    malls = mall_extraction.malls[:4]
    while len(malls) < 4:
        malls.append(MallItem())

    # Add a custom info block for transparency
    evaluator.add_custom_info(
        info={
            "total_malls_in_answer": len(mall_extraction.malls),
            "evaluated_malls": [m.name for m in malls]
        },
        info_type="extraction_stats",
        info_name="extraction_summary"
    )

    # Build mall-level parallel nodes under root and verify each mall
    for i, mall in enumerate(malls):
        mall_node = evaluator.add_parallel(
            id=f"Mall_{i+1}",
            desc=f"{'First' if i==0 else ('Second' if i==1 else ('Third' if i==2 else 'Fourth'))} retail mall meeting all specified criteria",
            parent=root,
            critical=False  # Allow partial credit across different malls
        )
        await verify_single_mall(evaluator, mall_node, mall, i)

    return evaluator.get_summary()