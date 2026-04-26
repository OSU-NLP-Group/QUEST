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
TASK_ID = "chicago_office_buildings_two"
TASK_DESCRIPTION = """A technology company is planning to relocate its headquarters to Chicago, Illinois, and needs to identify suitable office buildings for its team of 50 employees. The company has specific requirements to ensure sustainability, accessibility, and adequate workspace.

Find two Class A office buildings in Chicago that meet all of the following criteria:

1. Location: The building must be located within Chicago, Illinois.

2. Building Classification: The building must be designated as Class A office space, indicating high-quality construction, premium amenities, and professional management.

3. Sustainability Certification: The building must hold at least one of the following environmental certifications:
   - LEED Gold certification (60-79 points) OR LEED Platinum certification (80+ points), OR
   - ENERGY STAR certification with a score of 75 or higher

4. ADA Accessibility: The building must meet ADA accessibility standards, specifically providing doorways with a clear width of at least 32 inches when the door is open at 90 degrees.

5. Space Capacity: The building must have sufficient available office space to accommodate 50 employees, meeting the industry standard of at least 150 square feet per employee (minimum 7,500 square feet total).

For each building, provide the following information:
- Building name and complete address
- The specific sustainability certification held (LEED Gold, LEED Platinum, or ENERGY STAR with score)
- Confirmation that the building meets Class A standards and ADA accessibility requirements
- Available office space capacity
- A reference URL where this information can be verified
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BuildingInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    class_a_claim: Optional[str] = None  # raw text indicating Class A
    leed_level: Optional[str] = None     # e.g., "LEED Gold", "LEED Platinum"
    energy_star_score: Optional[str] = None  # e.g., "82"
    ada_door_width: Optional[str] = None     # e.g., "32 inches clear width"
    available_space: Optional[str] = None    # e.g., "10,000 SF"
    source_urls: List[str] = Field(default_factory=list)  # all supporting URLs


class BuildingsExtraction(BaseModel):
    buildings: List[BuildingInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_buildings() -> str:
    return """
Extract up to two qualifying office buildings described in the answer. For each building, extract exactly the following fields:

- name: The building's name as stated.
- address: The complete mailing address as stated (include city and state if present).
- class_a_claim: The exact phrasing that indicates the building is classified as "Class A" (if present).
- leed_level: One of "LEED Gold" or "LEED Platinum" if explicitly stated (otherwise null).
- energy_star_score: The ENERGY STAR score as a number or string if explicitly stated (otherwise null).
- ada_door_width: Any explicit statement about ADA doorway clear width (e.g., "32 inches clear width") or equivalent ADA doorway compliance text (otherwise null).
- available_space: The stated available office space (e.g., "7,500 SF", "10,000 sq ft"); if multiple suites are listed, you may provide the largest single suite or a total if the answer provides it. If missing, null.
- source_urls: A list of all URLs the answer provides for verifying this building’s information. This should include building pages, leasing brochures, or official certification pages (USGBC, ENERGY STAR), etc. Only include URLs explicitly present in the answer.

Rules:
- Do not infer values; only extract what is explicitly present.
- If a field is missing, set it to null (or an empty list for source_urls).
- Return a JSON object with a key 'buildings' which is an array of up to two BuildingInfo objects in the same order as presented in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        # Basic normalization: ensure URL has protocol (per extraction rules, they may add http://)
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        cleaned.append(u)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _two_buildings(extraction: BuildingsExtraction) -> List[BuildingInfo]:
    blist = extraction.buildings[:2] if extraction and extraction.buildings else []
    if len(blist) < 2:
        blist = blist + [BuildingInfo()] * (2 - len(blist))
    return blist


# --------------------------------------------------------------------------- #
# Verification for one building                                               #
# --------------------------------------------------------------------------- #
async def verify_building(
    evaluator: Evaluator,
    parent_node,
    building: BuildingInfo,
    index: int,
) -> None:
    """
    Build the verification subtree for a single building and run all checks.
    We follow the rubric intent; some structural adjustments are made so that
    the critical OR condition for sustainability is enforced robustly.
    """

    b_label = f"building_{index + 1}"
    b_name = building.name or "the building"
    b_addr = building.address or "(address not provided)"
    urls = _valid_urls(building.source_urls)

    # Top-level node for this building
    b_node = evaluator.add_parallel(
        id=b_label,
        desc="First qualifying office building" if index == 0 else "Second qualifying office building",
        parent=parent_node,
        critical=False  # Keep non-critical so partial credit across buildings is possible
    )

    # Space and Reference parent (critical as per rubric)
    sr_node = evaluator.add_parallel(
        id=f"{b_label}_space_and_reference",
        desc="Building meets space requirements and provides documentation",
        parent=b_node,
        critical=True
    )

    # Reference URL provided (critical)
    ref_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"{b_label}_reference_url_provided",
        desc="Valid reference URL is provided for the building information",
        parent=sr_node,
        critical=True
    )

    # Capacity for 50 employees (critical)
    cap_leaf = evaluator.add_leaf(
        id=f"{b_label}_capacity_for_50_employees",
        desc="Building can accommodate at least 50 employees at minimum 150 square feet per employee (7,500 sq ft total)",
        parent=sr_node,
        critical=True
    )
    cap_claim = (
        "This building has at least 7,500 square feet of available office space suitable for office use "
        "(either a single suite or a combination of available suites)."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=urls,
        additional_instruction=(
            "Verify that the referenced page(s) explicitly indicate available office space of 7,500 square feet or more. "
            "It's acceptable if multiple available suites can be combined to meet or exceed 7,500 sq ft. "
            "Do not count total building area unless it is explicitly stated as available for lease/occupancy."
        ),
        extra_prerequisites=[ref_node]
    )

    # Location in Chicago (critical)
    loc_leaf = evaluator.add_leaf(
        id=f"{b_label}_location_in_chicago",
        desc="Building is located in Chicago, Illinois",
        parent=b_node,
        critical=True
    )
    loc_claim = f"The building '{b_name}' with address '{b_addr}' is located in Chicago, Illinois."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=urls,
        additional_instruction=(
            "Pass if the address or location clearly indicates the city is Chicago and the state is Illinois (IL). "
            "If the page lists a different city, or is unclear/ambiguous, fail."
        ),
        extra_prerequisites=[ref_node]
    )

    # Class A Office Space (critical)
    class_leaf = evaluator.add_leaf(
        id=f"{b_label}_class_a_office_space",
        desc="Building is classified as Class A office space",
        parent=b_node,
        critical=True
    )
    class_claim = "This building is classified as Class A office space."
    await evaluator.verify(
        claim=class_claim,
        node=class_leaf,
        sources=urls,
        additional_instruction=(
            "Look for explicit language such as 'Class A', 'Class A office', or equivalent. "
            "Marketing or broker pages that describe the building as 'Class A' are acceptable."
        ),
        extra_prerequisites=[ref_node]
    )

    # Sustainability requirement (critical OR across LEED/ENERGY STAR)
    sust_req_leaf = evaluator.add_leaf(
        id=f"{b_label}_sustainability_certification_requirement",
        desc="Building holds either LEED Gold/Platinum or ENERGY STAR certification",
        parent=b_node,
        critical=True
    )
    sust_req_claim = (
        "This building has at least one of the following: LEED Gold certification, LEED Platinum certification, "
        "or ENERGY STAR certification with a score of 75 or higher."
    )
    await evaluator.verify(
        claim=sust_req_claim,
        node=sust_req_leaf,
        sources=urls,
        additional_instruction=(
            "Pass if the evidence clearly shows LEED Gold or LEED Platinum certification, "
            "or indicates the building has an ENERGY STAR certification with a score of at least 75. "
            "Accept variations such as 'LEED v4 Gold' or 'LEED BD+C Gold'. "
            "For ENERGY STAR, any listed score ≥ 75 (for any recent year) is acceptable."
        ),
        extra_prerequisites=[ref_node]
    )

    # Optional detailed evidence checks for sustainability (non-critical, informational)
    sust_ev_node = evaluator.add_parallel(
        id=f"{b_label}_sustainability_certification_evidence",
        desc="Detailed sustainability evidence checks (non-critical)",
        parent=b_node,
        critical=False
    )

    # LEED Gold or Platinum (non-critical evidence leaf)
    leed_leaf = evaluator.add_leaf(
        id=f"{b_label}_leed_gold_or_platinum",
        desc="Building has LEED Gold (60-79 points) or LEED Platinum (80+ points) certification",
        parent=sust_ev_node,
        critical=False
    )
    # Use extracted level if available to make the claim more specific
    if (building.leed_level or "").strip():
        leed_claim = f"This building has {building.leed_level.strip()} certification."
    else:
        leed_claim = "This building has LEED Gold or LEED Platinum certification."
    await evaluator.verify(
        claim=leed_claim,
        node=leed_leaf,
        sources=urls,
        additional_instruction=(
            "Pass only if the page explicitly mentions LEED Gold or LEED Platinum for this building. "
            "USGBC listings or building/broker pages explicitly stating the LEED level are acceptable."
        ),
        extra_prerequisites=[ref_node]
    )

    # ENERGY STAR 75+ (non-critical evidence leaf)
    es_leaf = evaluator.add_leaf(
        id=f"{b_label}_energy_star_score_75_plus",
        desc="Building has ENERGY STAR certification with score of 75 or higher",
        parent=sust_ev_node,
        critical=False
    )
    if (building.energy_star_score or "").strip():
        es_claim = (
            f"This building has an ENERGY STAR score of {building.energy_star_score.strip()}, which is at least 75."
        )
    else:
        es_claim = "This building has an ENERGY STAR certification with a score of 75 or higher."
    await evaluator.verify(
        claim=es_claim,
        node=es_leaf,
        sources=urls,
        additional_instruction=(
            "Pass only if the page shows an ENERGY STAR certification with a score ≥ 75 for the building. "
            "Include ENERGY STAR Portfolio Manager listings or building pages that state the score."
        ),
        extra_prerequisites=[ref_node]
    )

    # ADA compliant doorway width (critical)
    ada_leaf = evaluator.add_leaf(
        id=f"{b_label}_ada_compliant_doorwidth",
        desc="Building meets ADA requirement of at least 32 inches clear door width when open at 90 degrees",
        parent=b_node,
        critical=True
    )
    ada_claim = (
        "This building provides doorways with a clear width of at least 32 inches when the door is open at 90 degrees, "
        "meeting ADA requirements."
    )
    await evaluator.verify(
        claim=ada_claim,
        node=ada_leaf,
        sources=urls,
        additional_instruction=(
            "Pass only if the source explicitly indicates compliance with ADA doorway clearance requirements "
            "(e.g., '32-inch clear width' at 90 degrees) or clearly states doorway specifications meeting this requirement. "
            "Generic statements like 'ADA compliant' without mentioning doorways/clear widths should not be considered sufficient."
        ),
        extra_prerequisites=[ref_node]
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
    Evaluate an answer for the Chicago Class A buildings task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Buildings are evaluated independently
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

    # Extract up to 2 buildings
    extraction = await evaluator.extract(
        prompt=prompt_extract_buildings(),
        template_class=BuildingsExtraction,
        extraction_name="extracted_buildings"
    )

    buildings = _two_buildings(extraction)

    # Build and verify each building
    # Root node is non-critical to allow partial credit if only one building qualifies
    tasks = []
    for i in range(2):
        # For clarity and ID stability, we build each building subtree sequentially
        await verify_building(evaluator, root, buildings[i], i)

    # Return evaluation summary
    return evaluator.get_summary()