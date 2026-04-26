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
TASK_ID = "ncaa_d3_aquatic_centers_3"
TASK_DESCRIPTION = (
    "Identify three NCAA Division III colleges or universities that have aquatic centers meeting ALL of the following "
    "specifications: (1) The institution must compete in NCAA Division III athletics; (2) The swimming pool must be either "
    "50 meters in length OR have the capability to configure to 50 meters using movable bulkheads; (3) The pool must have "
    "a minimum of 8 competition lanes; (4) The facility must include dedicated diving equipment with at least two springboards "
    "(either 1-meter, 3-meter, or both); (5) The diving well or diving area must meet NCAA depth requirements (minimum 11.2 feet "
    "for 1-meter springboards, 12.1 feet for 3-meter springboards); (6) The facility must have hosted or be certified to host NCAA "
    "Division III championship meets or conference championship meets. For each of the three facilities you identify, provide: the name "
    "of the institution, the name of the aquatic facility/natatorium, the exact pool dimensions (length × width), the number of lanes "
    "available for competition, the specific diving equipment present, confirmation that it meets NCAA Division III competition standards, "
    "and a reference URL that verifies these specifications."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PoolInfo(BaseModel):
    length: Optional[str] = None  # e.g., "50 m", "50 meters", "50m", "25-yard"
    width: Optional[str] = None   # e.g., "25 m", "25 yards"
    fifty_meter_capability: Optional[str] = None  # e.g., "Configurable to 50 meters with movable bulkheads", "Olympic-size 50m"


class DivingInfo(BaseModel):
    springboards_description: Optional[str] = None  # e.g., "two 1-meter and two 3-meter springboards"
    springboard_types: Optional[str] = None         # e.g., "1m, 3m"
    springboard_count: Optional[str] = None         # e.g., "2", "two", "at least two"
    depth: Optional[str] = None                     # e.g., "12'6\"", "13 ft", "12.5 feet"
    notes: Optional[str] = None


class FacilityItem(BaseModel):
    institution: Optional[str] = None
    facility_name: Optional[str] = None
    pool: PoolInfo = Field(default_factory=PoolInfo)
    lanes: Optional[str] = None                     # e.g., "8 lanes", "10 lanes"
    diving: DivingInfo = Field(default_factory=DivingInfo)
    championships: Optional[str] = None            # hosted/certified for championships (DIII or conference)
    standards_confirmation: Optional[str] = None   # explicit statement of meeting NCAA DIII standards, certification, etc.
    reference_urls: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    facilities: List[FacilityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
    Extract up to five facilities proposed in the answer that are intended to satisfy the task requirements.
    For each facility, extract the following fields exactly as they appear in the answer (do not infer):
    - institution: The college/university name.
    - facility_name: The aquatic center or natatorium name.
    - pool.length: The pool length value with units if present (e.g., "50 m", "50 meters", "50m", "25 yards").
    - pool.width: The pool width value with units if present.
    - pool.fifty_meter_capability: If the answer states the pool is 50 meters in length, or that it can be configured to 50 meters using movable bulkheads, capture that statement here (e.g., "Olympic-size 50m", "configurable to 50m with movable bulkheads"). If not specified, return null.
    - lanes: The number of competition lanes stated (text as given, e.g., "8 lanes", "10 lanes", "8-10 lanes").
    - diving.springboards_description: The description of springboards present (e.g., "two 1-meter and two 3-meter springboards").
    - diving.springboard_types: The types present (e.g., "1m, 3m"), if stated.
    - diving.springboard_count: The quantity of springboards if stated (text as given).
    - diving.depth: The diving well/area depth with units (e.g., "12'6\"", "13 ft", "12.5 feet"), if stated.
    - diving.notes: Any additional diving-related details, if present.
    - championships: Any explicit statement that the facility has hosted or is certified to host NCAA Division III or conference championship meets (verbatim text as presented).
    - standards_confirmation: Any explicit statement claiming the facility meets NCAA Division III competition standards or is NCAA-certified, if present (verbatim).
    - reference_urls: The list of all URLs cited in the answer for this facility (include both official institutional pages and any other credible sources). 
      Extract only URLs explicitly present in the answer. If none are provided, return an empty list.

    If any field is missing for a facility, set it to null (or empty list for URLs).
    Place the results in a JSON object with a top-level key "facilities", which is an array of facility objects in the same order as presented in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _text_or_unknown(s: Optional[str]) -> str:
    return s if (s and s.strip()) else "unknown"


# --------------------------------------------------------------------------- #
# Facility verification builder                                               #
# --------------------------------------------------------------------------- #
async def verify_facility(
    evaluator: Evaluator,
    parent_node,
    item: FacilityItem,
    index: int,
) -> None:
    """
    Build and execute verification sub-tree for a single facility.
    """
    i = index + 1
    fac_node = evaluator.add_parallel(
        id=f"facility_{i}",
        desc=f"Facility #{i}: meets all constraints and includes all requested information with verifying references",
        parent=parent_node,
        critical=False  # allow partial across facilities
    )

    urls = _safe_urls(item.reference_urls)
    institution = _text_or_unknown(item.institution)
    facility_name = _text_or_unknown(item.facility_name)

    # 1) Institution name provided (critical leaf)
    evaluator.add_custom_node(
        result=(item.institution is not None and item.institution.strip() != ""),
        id=f"facility_{i}_institution_name_provided",
        desc="Institution name is provided",
        parent=fac_node,
        critical=True
    )

    # 2) Facility name provided (critical leaf)
    evaluator.add_custom_node(
        result=(item.facility_name is not None and item.facility_name.strip() != ""),
        id=f"facility_{i}_facility_name_provided",
        desc="Aquatic facility/natatorium name is provided",
        parent=fac_node,
        critical=True
    )

    # 3) NCAA Division III institution (critical leaf)
    node_d3 = evaluator.add_leaf(
        id=f"facility_{i}_ncaa_diii_institution",
        desc="Institution competes in NCAA Division III athletics",
        parent=fac_node,
        critical=True
    )
    claim_d3 = f"The institution {institution} competes in NCAA Division III athletics."
    await evaluator.verify(
        claim=claim_d3,
        node=node_d3,
        sources=urls,
        additional_instruction=(
            "Look for explicit indications such as 'NCAA Division III', 'DIII', or 'Div. III' on official institutional or athletics pages, "
            "or clear membership in a known Division III conference (e.g., NESCAC, UAA, SCIAC, etc.). "
            "If no URLs are provided, judge as Incorrect."
        ),
    )

    # 4) Pool dimensions provided (critical parent with 2 critical children)
    dims_parent = evaluator.add_parallel(
        id=f"facility_{i}_pool_dimensions_provided",
        desc="Exact pool dimensions (length and width) are provided",
        parent=fac_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(item.pool.length is not None and item.pool.length.strip() != ""),
        id=f"facility_{i}_pool_length_value_provided",
        desc="Pool length value is explicitly stated",
        parent=dims_parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=(item.pool.width is not None and item.pool.width.strip() != ""),
        id=f"facility_{i}_pool_width_value_provided",
        desc="Pool width value is explicitly stated",
        parent=dims_parent,
        critical=True
    )

    # 5) 50m compliance (critical leaf)
    node_50m = evaluator.add_leaf(
        id=f"facility_{i}_pool_50m_compliance",
        desc="Pool is 50 meters OR is configurable to 50 meters using movable bulkheads",
        parent=fac_node,
        critical=True
    )
    claim_50m = (
        f"The swimming pool at {institution} ({facility_name}) is 50 meters in length OR can be configured to 50 meters using movable bulkheads."
    )
    await evaluator.verify(
        claim=claim_50m,
        node=node_50m,
        sources=urls,
        additional_instruction=(
            "Accept terms such as 'Olympic-size (50m)', '50-meter pool', or 'movable bulkheads enable 50m configuration'. "
            f"Extracted hints: length='{_text_or_unknown(item.pool.length)}', fifty_meter_capability='{_text_or_unknown(item.pool.fifty_meter_capability)}'. "
            "If the pages only mention 25 yards/meters without 50m configurability, judge as Not Supported."
        ),
    )

    # 6) Lane count check (critical sequential parent: first provided, then >= 8)
    lanes_parent = evaluator.add_sequential(
        id=f"facility_{i}_lane_count_check",
        desc="Competition lane count is provided and meets the minimum requirement",
        parent=fac_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(item.lanes is not None and item.lanes.strip() != ""),
        id=f"facility_{i}_lane_count_provided",
        desc="Number of competition lanes is explicitly stated",
        parent=lanes_parent,
        critical=True
    )
    node_lanes8 = evaluator.add_leaf(
        id=f"facility_{i}_lane_count_atleast_8",
        desc="Competition lane count is at least 8",
        parent=lanes_parent,
        critical=True
    )
    claim_lanes8 = (
        f"The pool at {institution} ({facility_name}) provides at least 8 lanes for competition."
    )
    await evaluator.verify(
        claim=claim_lanes8,
        node=node_lanes8,
        sources=urls,
        additional_instruction=(
            f"Check for phrases like '8 lanes', '10 lanes', or '8-10 lanes'. Extracted lane text: '{_text_or_unknown(item.lanes)}'. "
            "If the pages indicate fewer than 8 competition lanes, judge as Not Supported."
        ),
    )

    # 7) Diving equipment check (critical parallel: springboards + depth compliance)
    diving_parent = evaluator.add_parallel(
        id=f"facility_{i}_diving_equipment_check",
        desc="Diving equipment requirements are satisfied and documented",
        parent=fac_node,
        critical=True
    )

    node_sb_two = evaluator.add_leaf(
        id=f"facility_{i}_springboards_at_least_two",
        desc="Facility has at least two springboards (1-meter and/or 3-meter), with types/quantities documented",
        parent=diving_parent,
        critical=True
    )
    claim_sb_two = (
        f"The facility at {institution} ({facility_name}) has at least two diving springboards (either 1-meter, 3-meter, or both)."
    )
    await evaluator.verify(
        claim=claim_sb_two,
        node=node_sb_two,
        sources=urls,
        additional_instruction=(
            "Look for explicit counts like 'two 1-meter boards', 'two 3-meter boards', or a combination (e.g., 'one 1m and one 3m'). "
            f"Extracted: springboards_description='{_text_or_unknown(item.diving.springboards_description)}', "
            f"springboard_types='{_text_or_unknown(item.diving.springboard_types)}', "
            f"springboard_count='{_text_or_unknown(item.diving.springboard_count)}'."
        ),
    )

    node_depth = evaluator.add_leaf(
        id=f"facility_{i}_diving_depth_compliance",
        desc=("Diving area depth meets NCAA minimums for the installed springboards: "
              "if any 1m springboard is present depth ≥ 11.2 ft; if any 3m springboard is present depth ≥ 12.1 ft"),
        parent=diving_parent,
        critical=True
    )
    claim_depth = (
        "The diving area depth satisfies NCAA minimums: at least 11.2 feet where a 1-meter springboard is present and at least 12.1 feet where a 3-meter springboard is present, "
        f"for the facility at {institution} ({facility_name})."
    )
    await evaluator.verify(
        claim=claim_depth,
        node=node_depth,
        sources=urls,
        additional_instruction=(
            f"Use the pages to confirm depth relative to installed boards. Extracted: depth='{_text_or_unknown(item.diving.depth)}', "
            f"springboard_types='{_text_or_unknown(item.diving.springboard_types)}', "
            f"springboards_description='{_text_or_unknown(item.diving.springboards_description)}'. "
            "If both 1m and 3m are present, require ≥12.1 ft. If only 1m is present, require ≥11.2 ft."
        ),
    )

    # 8) Hosted or certified championship meets (critical leaf)
    node_host = evaluator.add_leaf(
        id=f"facility_{i}_hosted_or_certified_championship_meets",
        desc="Facility has hosted or is certified to host NCAA Division III championship meets or conference championship meets",
        parent=fac_node,
        critical=True
    )
    claim_host = (
        f"The facility at {institution} ({facility_name}) has hosted or is certified to host NCAA Division III championship meets or conference championship meets."
    )
    await evaluator.verify(
        claim=claim_host,
        node=node_host,
        sources=urls,
        additional_instruction=(
            "Accept explicit mentions of hosting NCAA Division III championships or hosting conference championship meets (e.g., NESCAC, UAA, SCIAC, etc.), "
            f"or official certifications stating championship hosting capability. Extracted: '{_text_or_unknown(item.championships)}'."
        ),
    )

    # 9) Explicit NCAA DIII competition standards confirmation (critical leaf)
    node_std = evaluator.add_leaf(
        id=f"facility_{i}_ncaa_diii_competition_standards_confirmation",
        desc="An explicit confirmation (or clearly cited certification/standard) is provided that the facility meets NCAA Division III competition standards",
        parent=fac_node,
        critical=True
    )
    claim_std = (
        f"The facility at {institution} ({facility_name}) meets NCAA Division III competition standards (explicit confirmation or certification)."
    )
    await evaluator.verify(
        claim=claim_std,
        node=node_std,
        sources=urls,
        additional_instruction=(
            "Look for phrases such as 'NCAA certified', 'meets NCAA standards', 'NCAA regulation facility', "
            "or equivalent explicit confirmation on official sources. "
            f"Extracted: '{_text_or_unknown(item.standards_confirmation)}'. "
            "If no such explicit confirmation exists but the facility has officially hosted NCAA Division III or conference championships on record, consider that as strong evidence of compliance."
        ),
    )

    # 10) References check (critical parallel group)
    refs_parent = evaluator.add_parallel(
        id=f"facility_{i}_references_check",
        desc="References are provided and substantiate the claimed specifications",
        parent=fac_node,
        critical=True
    )

    # 10.a) At least one official institutional source present (critical leaf)
    node_official = evaluator.add_leaf(
        id=f"facility_{i}_official_institution_source_present",
        desc="At least one reference URL is from an official institutional source (e.g., the institution/athletics/facilities site)",
        parent=refs_parent,
        critical=True
    )
    claim_official = (
        f"This webpage is an official institutional source for {institution} (on the institution's own domain such as .edu, its athletics subdomain, or official facilities/rec pages)."
    )
    await evaluator.verify(
        claim=claim_official,
        node=node_official,
        sources=urls,
        additional_instruction=(
            "Judge each URL independently. Consider domains owned by the institution (e.g., *.edu or recognized institutional/athletics subdomains) as official. "
            "If there are no URLs provided at all, judge as Incorrect."
        ),
    )

    # 10.b) References verify core specs (expanded into individual critical leaves under a parallel subgroup)
    specs_parent = evaluator.add_parallel(
        id=f"facility_{i}_references_verify_core_specs",
        desc="Provided reference URL(s) verify the key required specs",
        parent=refs_parent,
        critical=True
    )

    # Each spec check is a dedicated leaf verified against the provided URLs
    node_ref_50m = evaluator.add_leaf(
        id=f"facility_{i}_ref_verifies_50m",
        desc="References verify: 50m length or 50m-configurable with movable bulkheads",
        parent=specs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The provided reference page supports that the pool at {institution} ({facility_name}) is 50 meters in length OR configurable to 50 meters via movable bulkheads."
        ),
        node=node_ref_50m,
        sources=urls,
        additional_instruction=(
            f"Use the reference page content. Hints: length='{_text_or_unknown(item.pool.length)}', "
            f"fifty_meter_capability='{_text_or_unknown(item.pool.fifty_meter_capability)}'."
        ),
    )

    node_ref_lanes = evaluator.add_leaf(
        id=f"facility_{i}_ref_verifies_8_lanes",
        desc="References verify: competition lane count ≥ 8",
        parent=specs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The provided reference page supports that the competition lane count at {institution} ({facility_name}) is at least 8."
        ),
        node=node_ref_lanes,
        sources=urls,
        additional_instruction=(
            f"Accept '8 lanes', '10 lanes', or similar. Extracted lanes text: '{_text_or_unknown(item.lanes)}'."
        ),
    )

    node_ref_sb = evaluator.add_leaf(
        id=f"facility_{i}_ref_verifies_springboards",
        desc="References verify: at least two springboards are present (1m and/or 3m)",
        parent=specs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The provided reference page supports that at least two diving springboards (1m and/or 3m) are present at {institution} ({facility_name})."
        ),
        node=node_ref_sb,
        sources=urls,
        additional_instruction=(
            f"Extracted: springboards_description='{_text_or_unknown(item.diving.springboards_description)}', "
            f"springboard_types='{_text_or_unknown(item.diving.springboard_types)}', "
            f"springboard_count='{_text_or_unknown(item.diving.springboard_count)}'."
        ),
    )

    node_ref_depth = evaluator.add_leaf(
        id=f"facility_{i}_ref_verifies_diving_depth",
        desc="References verify: diving depth meets NCAA minimums for installed boards",
        parent=specs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The provided reference page supports that the diving area depth at {institution} ({facility_name}) meets NCAA minimums (≥11.2 ft for 1m, ≥12.1 ft for 3m) given installed boards."
        ),
        node=node_ref_depth,
        sources=urls,
        additional_instruction=(
            f"Extracted depth: '{_text_or_unknown(item.diving.depth)}'; springboard_types='{_text_or_unknown(item.diving.springboard_types)}'. "
            "If only 1m boards are present, require ≥11.2 ft; if any 3m present, require ≥12.1 ft."
        ),
    )

    node_ref_host = evaluator.add_leaf(
        id=f"facility_{i}_ref_verifies_hosting",
        desc="References verify: hosted or certified to host NCAA DIII or conference championships",
        parent=specs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The provided reference page supports that {institution} ({facility_name}) has hosted or is certified to host NCAA Division III championship meets or conference championship meets."
        ),
        node=node_ref_host,
        sources=urls,
        additional_instruction=(
            f"Extracted: '{_text_or_unknown(item.championships)}'. Accept explicit hosting history or official statements of certification."
        ),
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
    Evaluate an answer for three NCAA Division III aquatic facilities that meet all specified constraints.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall: facilities are independent; allow partial credit
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

    # 1) Extract structured facility data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction"
    )

    facilities = list(extracted.facilities or [])
    # Keep only the first 3 facilities (pad with empty ones if fewer)
    if len(facilities) < 3:
        facilities = facilities + [FacilityItem() for _ in range(3 - len(facilities))]
    else:
        facilities = facilities[:3]

    # 2) Build Task_Completion node (non-critical to allow partial score across facilities)
    # Note: To allow partial credit per facility, we set this node as non-critical.
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify three NCAA Division III aquatic facilities that satisfy all stated facility constraints and provide all requested fields with verifying references",
        parent=root,
        critical=False
    )

    # 3) Verify each facility subtree
    for idx, item in enumerate(facilities):
        await verify_facility(evaluator, task_node, item, idx)

    # 4) Return evaluation summary
    return evaluator.get_summary()