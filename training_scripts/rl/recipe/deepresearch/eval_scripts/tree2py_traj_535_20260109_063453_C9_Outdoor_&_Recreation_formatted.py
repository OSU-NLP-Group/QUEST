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
TASK_ID = "pnw_usfs_campgrounds"
TASK_DESCRIPTION = """I'm planning a camping trip in the Pacific Northwest and need to find three developed campgrounds that are managed by the US Forest Service. The campgrounds must be located in National Forests in Washington, Oregon, or California, and must meet all of the following specific requirements:

1. Be accessible by vehicle (no backpacking or hiking required to reach the campground)
2. Have at least 15 individual campsites
3. Include basic site amenities: picnic tables and fire rings/grates at each campsite
4. Provide toilet facilities (either vault toilets or flush toilets)
5. Provide potable drinking water available on-site
6. NOT have electric hookups at individual campsites
7. NOT have hot shower facilities
8. Be available for reservation through Recreation.gov

For each of the three campgrounds, please provide:
- The campground name
- The specific National Forest where it's located
- The state (Washington, Oregon, or California)
- The total number of individual campsites
- Confirmation of which amenities are present (toilets, water, picnic tables, fire rings)
- Explicit confirmation that electric hookups are NOT available
- Explicit confirmation that hot showers are NOT available
- The official Recreation.gov URL for the campground
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Campground(BaseModel):
    name: Optional[str] = None
    national_forest: Optional[str] = None
    state: Optional[str] = None  # Accept forms like "WA", "Washington", etc.
    total_campsites: Optional[str] = None  # Keep as string for flexibility
    vehicle_accessible: Optional[bool] = None
    developed_campground: Optional[bool] = None
    managed_by_usfs: Optional[bool] = None
    reservable_on_recreation_gov: Optional[bool] = None

    picnic_tables_present: Optional[bool] = None
    fire_rings_or_grates_present: Optional[bool] = None
    toilets_available: Optional[bool] = None
    potable_water_available: Optional[bool] = None

    electric_hookups_available: Optional[bool] = None
    hot_showers_available: Optional[bool] = None

    recreation_gov_url: Optional[str] = None
    other_source_urls: List[str] = Field(default_factory=list)


class CampgroundsExtraction(BaseModel):
    campgrounds: List[Campground] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract up to three developed campgrounds that the answer proposes. For each campground, return a JSON object with the following fields:

    Identification fields (strings):
    - name
    - national_forest  (the specific National Forest name)
    - state            (the U.S. state for the campground; accept forms like "WA"/"Washington", "OR"/"Oregon", "CA"/"California")
    - total_campsites  (number of individual campsites; keep as a string exactly as written)

    Eligibility flags (booleans if explicitly stated; otherwise null):
    - vehicle_accessible
    - developed_campground
    - managed_by_usfs
    - reservable_on_recreation_gov

    Amenities present (booleans if explicitly stated; otherwise null):
    - picnic_tables_present
    - fire_rings_or_grates_present
    - toilets_available
    - potable_water_available

    Amenities absent (booleans if explicitly stated; otherwise null):
    - electric_hookups_available          (true if hookups exist; false if they do NOT exist; null if unknown)
    - hot_showers_available               (true if hot showers exist; false if NOT available; null if unknown)

    Source URLs:
    - recreation_gov_url   (Official Recreation.gov URL for the campground; if missing, return null)
    - other_source_urls    (Additional official or relevant URLs explicitly mentioned in the answer, such as USFS pages; return a list; empty if none)

    IMPORTANT RULES:
    - Do not infer any information; only extract what is explicitly stated in the answer.
    - If a field is not mentioned, set it to null (for booleans) or null (for strings).
    - Only include valid URLs that are explicitly provided in the answer. If URLs are missing a protocol, prepend "http://".
    - If the answer lists more than three campgrounds, extract only the first three.
    - If fewer than three campgrounds are provided, return whatever is available.

    Return a JSON with a single top-level key `campgrounds` which is an array of campground objects.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
ALLOWED_STATES = {
    "wa": "WA", "washington": "WA",
    "or": "OR", "oregon": "OR",
    "ca": "CA", "california": "CA",
}


def normalize_state(state_str: Optional[str]) -> Optional[str]:
    if not state_str:
        return None
    s = state_str.strip().lower()
    return ALLOWED_STATES.get(s)


def combine_sources(cg: Campground) -> List[str]:
    urls: List[str] = []
    if cg.recreation_gov_url and cg.recreation_gov_url.strip():
        urls.append(cg.recreation_gov_url.strip())
    for u in cg.other_source_urls:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    uniq_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq_urls.append(u)
    return uniq_urls


def has_official_source(cg: Campground) -> bool:
    urls = combine_sources(cg)
    for u in urls:
        low = u.lower()
        if "recreation.gov" in low or "fs.usda.gov" in low or "usda.gov" in low:
            return True
    return False


def campground_label(idx: int) -> str:
    return ["Campground_1", "Campground_2", "Campground_3"][idx]


def id_prefix(idx: int) -> str:
    return ["C1", "C2", "C3"][idx]


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_campground(
    evaluator: Evaluator,
    parent_node,
    cg: Campground,
    idx: int,
) -> None:
    camp_id = campground_label(idx)
    prefix = id_prefix(idx)

    # Create main campground node (non-critical, parallel aggregation)
    camp_node = evaluator.add_parallel(
        id=camp_id,
        desc=f"{idx + 1}st campground entry (eligible + required fields provided)" if idx == 0 else (
            f"{idx + 1}nd campground entry (eligible + required fields provided)" if idx == 1 else
            f"{idx + 1}rd campground entry (eligible + required fields provided)"
        ),
        parent=parent_node,
        critical=False
    )

    # 1) Required Identification Fields (critical)
    req_node = evaluator.add_parallel(
        id=f"{prefix}_Required_Identification_Fields",
        desc="Provide the required identifying fields for this campground",
        parent=camp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(cg.name is not None and cg.name.strip() != ""),
        id=f"{prefix}_Name_Provided",
        desc="Campground name is provided",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(cg.national_forest is not None and cg.national_forest.strip() != ""),
        id=f"{prefix}_National_Forest_Name_Provided",
        desc="Specific National Forest name is provided",
        parent=req_node,
        critical=True
    )

    normalized_state = normalize_state(cg.state)
    evaluator.add_custom_node(
        result=(normalized_state in {"WA", "OR", "CA"} if normalized_state else False),
        id=f"{prefix}_State_Is_WA_OR_OR_CA",
        desc="State is Washington, Oregon, or California",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(cg.recreation_gov_url is not None and "recreation.gov" in (cg.recreation_gov_url or "").lower()),
        id=f"{prefix}_RecreationGov_URL_Provided",
        desc="Official Recreation.gov URL for the campground is provided",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(cg.total_campsites is not None and str(cg.total_campsites).strip() != ""),
        id=f"{prefix}_Total_Campsites_Provided",
        desc="Total number of individual campsites is provided",
        parent=req_node,
        critical=True
    )

    # 2) Eligibility Constraints (critical)
    elig_node = evaluator.add_parallel(
        id=f"{prefix}_Eligibility_Constraints",
        desc="Meets all eligibility constraints for a qualifying campground",
        parent=camp_node,
        critical=True
    )

    # Gather sources once
    sources_list = combine_sources(cg)
    cg_name = cg.name or "the campground"
    nf_name = cg.national_forest or "a National Forest"

    # Located within a National Forest
    located_leaf = evaluator.add_leaf(
        id=f"{prefix}_Located_In_National_Forest",
        desc="Located within a National Forest",
        parent=elig_node,
        critical=True
    )
    claim_located = f"{cg_name} is located within {nf_name}, which is a U.S. National Forest."
    await evaluator.verify(
        claim=claim_located,
        node=located_leaf,
        sources=sources_list,
        additional_instruction="Verify on the official pages (Recreation.gov or USFS) that the facility belongs to a U.S. National Forest. Look for mentions like 'National Forest', 'Forest', or USFS branding."
    )

    # Managed by USFS
    managed_leaf = evaluator.add_leaf(
        id=f"{prefix}_Managed_By_USFS",
        desc="Managed by the US Forest Service (not state parks or NPS)",
        parent=elig_node,
        critical=True
    )
    claim_managed = f"{cg_name} is managed by the U.S. Forest Service."
    await evaluator.verify(
        claim=claim_managed,
        node=managed_leaf,
        sources=sources_list,
        additional_instruction="Check for 'Managed by: U.S. Forest Service' on Recreation.gov or verify that the source is an official USFS page. Fail if managed by a state parks agency or National Park Service."
    )

    # Developed/established campground
    developed_leaf = evaluator.add_leaf(
        id=f"{prefix}_Is_Developed_Campground",
        desc="Is a developed/established campground with designated campsites (not dispersed camping)",
        parent=elig_node,
        critical=True
    )
    claim_developed = f"{cg_name} is a developed campground with designated campsites (not dispersed camping)."
    await evaluator.verify(
        claim=claim_developed,
        node=developed_leaf,
        sources=sources_list,
        additional_instruction="A listing on Recreation.gov typically indicates a developed campground with designated sites. Look for evidence such as campsite listings, facility descriptions, or USFS facility pages confirming developed status."
    )

    # Vehicle accessible
    vehicle_leaf = evaluator.add_leaf(
        id=f"{prefix}_Vehicle_Accessible",
        desc="Accessible by vehicle (no backpacking/hiking required to reach the campground)",
        parent=elig_node,
        critical=True
    )
    claim_vehicle = f"{cg_name} is accessible by vehicle without backpacking or hiking required."
    await evaluator.verify(
        claim=claim_vehicle,
        node=vehicle_leaf,
        sources=sources_list,
        additional_instruction="Look for wording such as 'drive-in', 'car camping', RV access, or typical family campground language. If the facility is exclusively hike-in, boat-in, or walk-in, then this should fail."
    )

    # At least 15 individual campsites
    sites_leaf = evaluator.add_leaf(
        id=f"{prefix}_At_Least_15_Campsites",
        desc="Has at least 15 individual campsites",
        parent=elig_node,
        critical=True
    )
    claim_sites = f"{cg_name} has at least 15 individual campsites."
    await evaluator.verify(
        claim=claim_sites,
        node=sites_leaf,
        sources=sources_list,
        additional_instruction="Confirm the number of individual campsites or capacity on the page. If only loops or group sites are shown, ensure total individual campsites is >= 15."
    )

    # Reservable on Recreation.gov
    reservable_leaf = evaluator.add_leaf(
        id=f"{prefix}_Reservable_On_RecreationGov",
        desc="Available for reservation through Recreation.gov",
        parent=elig_node,
        critical=True
    )
    claim_reservable = f"Campsites at {cg_name} are reservable through Recreation.gov."
    await evaluator.verify(
        claim=claim_reservable,
        node=reservable_leaf,
        sources=cg.recreation_gov_url or sources_list,
        additional_instruction="Check if the Recreation.gov page supports reservations (e.g., 'Reserve a campsite' button, availability calendar). If the page indicates 'first come, first served' only and no reservations, then fail."
    )

    # 3) Amenities Present (critical)
    amenities_present_node = evaluator.add_parallel(
        id=f"{prefix}_Amenities_Present",
        desc="Required amenities are present",
        parent=camp_node,
        critical=True
    )

    # Picnic tables present
    pic_leaf = evaluator.add_leaf(
        id=f"{prefix}_Picnic_Tables_Present",
        desc="Picnic tables are present at campsites",
        parent=amenities_present_node,
        critical=True
    )
    claim_picnic = f"Picnic tables are provided at campsites at {cg_name}."
    await evaluator.verify(
        claim=claim_picnic,
        node=pic_leaf,
        sources=sources_list,
        additional_instruction="Look for amenity lists mentioning 'picnic table' or equivalent wording on the official page."
    )

    # Fire rings/grates present
    fire_leaf = evaluator.add_leaf(
        id=f"{prefix}_Fire_Rings_Or_Grates_Present",
        desc="Fire rings/grates are present at campsites",
        parent=amenities_present_node,
        critical=True
    )
    claim_fire = f"Fire rings or grills are provided at campsites at {cg_name}."
    await evaluator.verify(
        claim=claim_fire,
        node=fire_leaf,
        sources=sources_list,
        additional_instruction="Check for phrases like 'fire ring', 'campfire ring', 'grill' in the amenities."
    )

    # Toilets available
    toilets_leaf = evaluator.add_leaf(
        id=f"{prefix}_Toilets_Available",
        desc="Toilet facilities available (vault or flush)",
        parent=amenities_present_node,
        critical=True
    )
    claim_toilets = f"Toilet facilities (vault or flush) are available at {cg_name}."
    await evaluator.verify(
        claim=claim_toilets,
        node=toilets_leaf,
        sources=sources_list,
        additional_instruction="Look for 'vault toilet', 'flush toilet', or general restroom availability."
    )

    # Potable water available
    water_leaf = evaluator.add_leaf(
        id=f"{prefix}_Potable_Water_Available",
        desc="Potable drinking water is available on-site",
        parent=amenities_present_node,
        critical=True
    )
    claim_water = f"Potable drinking water is available on-site at {cg_name}."
    await evaluator.verify(
        claim=claim_water,
        node=water_leaf,
        sources=sources_list,
        additional_instruction="Verify 'drinking water', 'potable water', or equivalent phrasing on the official pages."
    )

    # 4) Amenities Absent (critical)
    amenities_absent_node = evaluator.add_parallel(
        id=f"{prefix}_Amenities_Absent",
        desc="Prohibited amenities are explicitly stated as not available",
        parent=camp_node,
        critical=True
    )

    # No electric hookups
    no_hookups_leaf = evaluator.add_leaf(
        id=f"{prefix}_No_Electric_Hookups_Explicit",
        desc="Explicitly confirms no electric hookups at individual campsites",
        parent=amenities_absent_node,
        critical=True
    )
    claim_no_electric = f"Individual campsites at {cg_name} do not have electric hookups."
    await evaluator.verify(
        claim=claim_no_electric,
        node=no_hookups_leaf,
        sources=sources_list,
        additional_instruction="Look for 'no hookups', 'electric hookups: none', or explicit absence of electricity at campsites."
    )

    # No hot showers
    no_showers_leaf = evaluator.add_leaf(
        id=f"{prefix}_No_Hot_Showers_Explicit",
        desc="Explicitly confirms no hot shower facilities",
        parent=amenities_absent_node,
        critical=True
    )
    claim_no_showers = f"There are no hot shower facilities at {cg_name}."
    await evaluator.verify(
        claim=claim_no_showers,
        node=no_showers_leaf,
        sources=sources_list,
        additional_instruction="Check for 'no showers', or absence of shower facilities in amenities. If showers are present, fail."
    )

    # 5) Official Source Evidence (critical leaf)
    evaluator.add_custom_node(
        result=has_official_source(cg),
        id=f"{prefix}_Official_Source_Evidence",
        desc="Provides official source link(s) (Recreation.gov and/or USFS) sufficient to verify the stated attributes for this campground",
        parent=camp_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    # Initialize evaluator with parallel aggregation at root
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
        default_model=model
    )

    # Extract structured campground info
    extraction = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds_extraction"
    )

    # Build top-level "Task_Completion" node (non-critical, parallel)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Provide three USFS-developed National Forest campgrounds (WA/OR/CA) meeting all specified criteria, with required details and official Recreation.gov links",
        parent=root,
        critical=False
    )

    # Prepare up to 3 campgrounds (pad if fewer)
    items = extraction.campgrounds[:3]
    while len(items) < 3:
        items.append(Campground())

    # Verify each campground subtree
    for i, cg in enumerate(items):
        await verify_campground(evaluator, task_node, cg, i)

    # Return evaluation summary
    return evaluator.get_summary()