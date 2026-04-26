import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Set

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "accessible_state_park_campgrounds"
TASK_DESCRIPTION = """
I am planning a camping trip and need accessible campgrounds that meet specific ADA requirements. Please identify at least three state park campgrounds from at least three different U.S. states that meet all of the following criteria:

- Each campground must be located within an official state park (designated and managed by state authorities)
- Each accessible campsite must have wheelchair-accessible picnic tables
- Each accessible campsite must have paved or hard-surface paths from the campsite to restroom/bathhouse facilities
- Each campground must have accessible shower facilities (not just restrooms)
- Each accessible campsite must provide electric hookups with at least 20 amp service
- Each campground must have an online reservation system

For each campground, please provide:
- The official state park name
- The state location
- A direct link to the campground's page on the state's official website or reservation system
- Specific details about the accessible campsite features available
"""

# Canonical U.S. states mapping for normalization
US_STATE_ABBR_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia"
}
US_STATE_NAME_TO_NAME = {v.lower(): v for v in US_STATE_ABBR_TO_NAME.values()}

def normalize_state_name(state_raw: Optional[str]) -> Optional[str]:
    if not state_raw:
        return None
    s = state_raw.strip()
    if len(s) == 2:
        abbr = s.upper()
        return US_STATE_ABBR_TO_NAME.get(abbr, abbr)
    # try full name case-insensitive
    return US_STATE_NAME_TO_NAME.get(s.lower(), s.title())


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    state_park_name: Optional[str] = None
    state: Optional[str] = None
    official_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)
    accessible_feature_details: Optional[str] = None


class CampgroundList(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
Extract all campground entries provided in the answer that the answer claims meet the stated accessibility and facility requirements.

For each campground entry, extract the following fields:
- state_park_name: The official state park name where the campground is located (e.g., "Fort Yargo State Park"). This must be explicitly mentioned in the answer. If unclear, return null.
- state: The U.S. state of the park (full name preferred; 2-letter abbreviation acceptable). If missing, return null.
- official_url: A direct link (URL) to the campground's page on the state's official website or on an official state reservation system (e.g., ReserveAmerica, ReserveCalifornia, ReserveSAS, etc.) if such a link is given in the answer. If multiple are listed, pick the most specific campground or park page. If none, return null.
- additional_urls: Any other URLs in the answer that pertain to this campground and may support accessibility features (e.g., accessibility PDFs, facility details, park brochures). If none, return an empty list.
- accessible_feature_details: Specific textual details the answer provides about accessibility features for this campground (e.g., mentions of "wheelchair-accessible picnic tables", "paved path to restroom", "accessible showers", "20/30/50 amp electric", "online reservation"). Capture the text snippets or summary as provided. If only vague claims are present and nothing specific is stated, return null.

Return a JSON object with a single key 'campgrounds' which is an array of these campground objects. Include all entries mentioned in the answer in the order they appear. Do not fabricate or infer information that is not explicitly present in the answer text.
"""


# --------------------------------------------------------------------------- #
# Helper functions for node construction and verification                     #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _has_specific_accessibility_details(text: Optional[str]) -> bool:
    if not _nonempty(text):
        return False
    t = text.lower()
    keywords = [
        "wheelchair", "accessible picnic", "picnic table", "paved", "hard-surface", "hard surface",
        "path", "sidewalk", "bathhouse", "restroom", "accessible shower", "shower", "ada",
        "electric", "amp", "20a", "30a", "50a", "reservation", "book", "reserve"
    ]
    if any(k in t for k in keywords):
        return True
    # also accept reasonably descriptive longer text
    return len(t) >= 40


def _merge_sources(official_url: Optional[str], additional_urls: List[str]) -> List[str]:
    urls: List[str] = []
    if _nonempty(official_url):
        urls.append(official_url.strip())  # type: ignore
    for u in additional_urls or []:
        if _nonempty(u):
            urls.append(u.strip())
    # de-duplicate while preserving order
    seen: Set[str] = set()
    merged: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            merged.append(u)
    return merged


# Handles for later global checks
class CampgroundHandles(BaseModel):
    index: int
    entry_node_id: str
    state_value: Optional[str] = None

    # Critical leaf IDs for qualification check
    state_park_status_id: str
    doc_leaf_ids: List[str] = Field(default_factory=list)
    ada_leaf_ids: List[str] = Field(default_factory=list)


def _entry_qualifies(evaluator: Evaluator, handles: CampgroundHandles) -> bool:
    def passed(node_id: str) -> bool:
        node = evaluator.find_node(node_id)
        return bool(node and node.status == "passed")

    if not passed(handles.state_park_status_id):
        return False
    if not all(passed(lid) for lid in handles.doc_leaf_ids):
        return False
    if not all(passed(lid) for lid in handles.ada_leaf_ids):
        return False
    return True


async def verify_one_campground(
    evaluator: Evaluator,
    parent_node_id_prefix: str,
    parent_node_desc: str,
    item: CampgroundItem,
    idx: int,
    parent_node
) -> CampgroundHandles:
    """
    Build the subtree for a single campground and run verifications.
    """
    # Create Campground Entry node (parallel, non-critical to allow per-entry evaluation)
    entry_node = evaluator.add_parallel(
        id=f"Campground_Entry_{idx+1}",
        desc=f"Campground entry {idx+1} must meet all per-campground constraints to count as qualifying.",
        parent=parent_node,
        critical=False
    )

    # 1) State Park Status (critical leaf)
    cg_state_park_status = evaluator.add_leaf(
        id=f"CG{idx+1}_State_Park_Status",
        desc="Campground is located within an official U.S. state park designated/managed by state authorities.",
        parent=entry_node,
        critical=True
    )

    normalized_state = normalize_state_name(item.state)
    park_name_for_claim = item.state_park_name or "the listed park"
    state_for_claim = normalized_state or (item.state or "the listed state")
    sources = _merge_sources(item.official_url, item.additional_urls)

    state_park_claim = (
        f"The campground referenced is located within {park_name_for_claim}, which is an official state park in "
        f"{state_for_claim} designated/managed by state authorities (e.g., the state's parks department). "
        f"The referenced page(s) are official state park or official state-run reservation system sources."
    )
    await evaluator.verify(
        claim=state_park_claim,
        node=cg_state_park_status,
        sources=sources,
        additional_instruction=(
            "Accept evidence if the page is clearly an official state domain (e.g., *.state.xx.us, state parks domain) "
            "or an official state-run reservation portal (e.g., ReserveAmerica/ReserveCalifornia operated for the state). "
            "The page should indicate it is a state park and that the campground is located within this park."
        )
    )

    # 2) Documentation Fields group (parallel, critical)
    doc_group = evaluator.add_parallel(
        id=f"CG{idx+1}_Documentation_Fields",
        desc="Required documentation is present for this campground entry.",
        parent=entry_node,
        critical=True
    )

    # 2a) Park name provided
    leaf_name_provided = evaluator.add_custom_node(
        result=_nonempty(item.state_park_name),
        id=f"CG{idx+1}_Official_State_Park_Name_Provided",
        desc="Official state park name is provided.",
        parent=doc_group,
        critical=True
    )

    # 2b) State provided
    leaf_state_provided = evaluator.add_custom_node(
        result=_nonempty(item.state),
        id=f"CG{idx+1}_State_Location_Provided",
        desc="U.S. state location is provided.",
        parent=doc_group,
        critical=True
    )

    # 2c) Direct official/reservation URL provided
    leaf_url_provided = evaluator.add_custom_node(
        result=_nonempty(item.official_url),
        id=f"CG{idx+1}_Direct_Official_URL_Provided",
        desc="Direct link to the campground page on the state's official website or official reservation system is provided.",
        parent=doc_group,
        critical=True
    )

    # 2d) Specific accessible feature details provided
    leaf_feature_details_provided = evaluator.add_custom_node(
        result=_has_specific_accessibility_details(item.accessible_feature_details),
        id=f"CG{idx+1}_Accessible_Feature_Details_Provided",
        desc="Specific details about the accessible campsite features are provided (not merely a generic claim of accessibility).",
        parent=doc_group,
        critical=True
    )

    # 3) ADA/Facility Constraints group (parallel, critical)
    ada_group = evaluator.add_parallel(
        id=f"CG{idx+1}_ADA_And_Facility_Constraints",
        desc="All ADA/accessibility/infrastructure/reservation constraints are satisfied for this campground entry.",
        parent=entry_node,
        critical=True
    )

    # ADA leaves
    leaf_picnic = evaluator.add_leaf(
        id=f"CG{idx+1}_Wheelchair_Accessible_Picnic_Tables",
        desc="Accessible campsite(s) have wheelchair-accessible picnic tables.",
        parent=ada_group,
        critical=True
    )
    leaf_paths = evaluator.add_leaf(
        id=f"CG{idx+1}_Hard_Surface_Paths_To_Restrooms",
        desc="Accessible campsite(s) have paved or hard-surface paths from campsite to restroom/bathhouse facilities.",
        parent=ada_group,
        critical=True
    )
    leaf_showers = evaluator.add_leaf(
        id=f"CG{idx+1}_Accessible_Showers",
        desc="Campground has accessible shower facilities (not just restrooms).",
        parent=ada_group,
        critical=True
    )
    leaf_electric = evaluator.add_leaf(
        id=f"CG{idx+1}_Electric_Hookups_Min_20A",
        desc="Accessible campsite(s) provide electric hookups with at least 20 amp service.",
        parent=ada_group,
        critical=True
    )
    leaf_reserve = evaluator.add_leaf(
        id=f"CG{idx+1}_Online_Reservation_System",
        desc="Campground has an online reservation system allowing advance bookings.",
        parent=ada_group,
        critical=True
    )

    # Build ADA claims
    park_ctx = park_name_for_claim
    ada_claims: List[Tuple[str, List[str], Any, str]] = [
        (
            f"Accessible campsite(s) at {park_ctx} provide wheelchair-accessible picnic table(s).",
            sources,
            leaf_picnic,
            "Look for terms like 'wheelchair-accessible picnic tables', 'accessible picnic tables', or ADA-compliant picnic tables."
        ),
        (
            f"Accessible campsite(s) at {park_ctx} have paved or hard-surface paths connecting the site to restrooms/bathhouses.",
            sources,
            leaf_paths,
            "Accept wording such as paved path, hard-surface, firm and stable surface, ADA path, or accessible route between campsite and restrooms/bathhouse."
        ),
        (
            f"{park_ctx} campground provides accessible shower facilities (not just accessible restrooms).",
            sources,
            leaf_showers,
            "Confirm that showers are accessible (roll-in, accessible stalls, or similar). Do not accept only 'accessible restrooms' without showers."
        ),
        (
            f"Accessible campsite(s) at {park_ctx} include electric hookups of at least 20 amps (20A or higher, e.g., 30A or 50A).",
            sources,
            leaf_electric,
            "Consider 20A, 30A, or 50A electrical service as satisfying 'at least 20 amp'. Verify that electric service is part of the accessible site features."
        ),
        (
            f"{park_ctx} campground supports online reservations (an official state-run reservation system or official park portal allowing booking).",
            sources,
            leaf_reserve,
            "Accept state official reservation portals (e.g., ReserveAmerica, ReserveCalifornia, ReserveSAS, or a state-run booking site) and pages with a working 'reserve/book online' functionality."
        ),
    ]

    # Batch verify the five ADA claims (and state park status already verified above)
    await evaluator.batch_verify(ada_claims)

    # Prepare handles for global checks
    handles = CampgroundHandles(
        index=idx,
        entry_node_id=entry_node.id,
        state_value=normalized_state,
        state_park_status_id=cg_state_park_status.id,
        doc_leaf_ids=[
            leaf_name_provided.id,
            leaf_state_provided.id,
            leaf_url_provided.id,
            leaf_feature_details_provided.id
        ],
        ada_leaf_ids=[
            leaf_picnic.id,
            leaf_paths.id,
            leaf_showers.id,
            leaf_electric.id,
            leaf_reserve.id
        ]
    )
    return handles


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
    Evaluate an answer for the accessible state park campgrounds task.
    """
    # Initialize evaluator (root is non-critical, parallel to avoid unintended short-circuiting)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundList,
        extraction_name="campgrounds_extraction"
    )

    # Keep first three entries (pad if needed)
    items: List[CampgroundItem] = list(extracted.campgrounds[:3])
    while len(items) < 3:
        items.append(CampgroundItem())

    # Build main subtrees
    # 1) Evaluate Campground Entries (non-critical, parallel)
    eval_entries_node = evaluator.add_parallel(
        id="Evaluate_Campground_Entries",
        desc="Evaluate the three campground entries provided to satisfy the 'at least three' requirement. (If more than three are listed, evaluate any three.)",
        parent=root,
        critical=False
    )

    # For each of the 3 entries, construct and verify
    handles_list: List[CampgroundHandles] = []
    for i in range(3):
        h = await verify_one_campground(
            evaluator=evaluator,
            parent_node_id_prefix=f"Campground_Entry_{i+1}",
            parent_node_desc=f"Campground entry {i+1} must meet all per-campground constraints to count as qualifying.",
            item=items[i],
            idx=i,
            parent_node=eval_entries_node
        )
        handles_list.append(h)

    # 2) Global List Requirements (critical, parallel)
    global_node = evaluator.add_parallel(
        id="Global_List_Requirements",
        desc="Set-level requirements that must be satisfied by the overall response (considering qualifying campground entries).",
        parent=root,
        critical=True
    )

    # Compute global validations based on verified leaf statuses
    qualifies_flags: List[bool] = [_entry_qualifies(evaluator, h) for h in handles_list]
    qualifies_count = sum(1 for q in qualifies_flags if q)

    # Unique states among qualifying entries
    qualified_states: Set[str] = set()
    for ok, h, item in zip(qualifies_flags, handles_list, items):
        if ok:
            norm_state = normalize_state_name(item.state)
            if norm_state:
                qualified_states.add(norm_state)

    # Add global custom nodes
    leaf_at_least_three = evaluator.add_custom_node(
        result=(qualifies_count >= 3),
        id="At_Least_Three_Qualifying_Campgrounds",
        desc="At least three distinct campground entries satisfy all per-campground constraints (state-park status + all ADA/facility constraints + required documentation).",
        parent=global_node,
        critical=True
    )

    leaf_three_states = evaluator.add_custom_node(
        result=(len(qualified_states) >= 3),
        id="At_Least_Three_Different_States",
        desc="The qualifying campgrounds are located across at least three different U.S. states.",
        parent=global_node,
        critical=True
    )

    # Record useful custom info
    evaluator.add_custom_info(
        info={
            "qualifies_count": qualifies_count,
            "qualified_states": sorted(list(qualified_states)),
            "per_entry_qualifies": qualifies_flags
        },
        info_type="global_computed_stats",
        info_name="global_requirements_computation"
    )

    # Return the evaluation summary
    return evaluator.get_summary()