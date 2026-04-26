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
TASK_ID = "rv_lakefront_campgrounds"
TASK_DESCRIPTION = (
    "I am planning an extended RV camping trip and need to identify suitable lakefront state park campgrounds that meet specific accessibility and facility requirements. "
    "Find four official state park campgrounds located across at least three different U.S. states. Each campground must meet ALL of the following requirements:\n\n"
    "1. Waterfront Location: The campground must have campsites located directly on a lake shoreline with direct lake access from the campsite (not just within walking distance).\n"
    "2. Hookup Type: The campground must offer either full hookup campsites (water, electric, and sewer connections) or electric hookup campsites (water and electric connections).\n"
    "3. Modern Restroom Facilities: The campground must provide modern shower houses or bathhouses with hot water and flush toilets.\n"
    "4. RV Dump Station: The campground must have an on-site dump station for RV waste disposal.\n"
    "5. ADA Accessibility: The campground must have designated ADA-accessible campsites with accessible features such as paved pathways, accessible picnic tables, or accessible restrooms.\n"
    "6. Pet-Friendly Policy: The campground must explicitly allow pets in designated campsites, with clearly stated pet rules (such as leash requirements and vaccination requirements).\n"
    "7. Maximum Stay Policy: The campground must have a clearly stated maximum stay limit policy (typically 14 consecutive nights within a 30-day period).\n"
    "8. Reservation System: The campground must offer an online reservation system that allows bookings at least 6 months in advance.\n\n"
    "For each campground, provide:\n"
    "- The official campground name\n"
    "- The state where it is located\n"
    "- The name of the lake\n"
    "- Confirmation that it is an official state park facility\n"
    "- Documentation of hookup types available (full or electric)\n"
    "- Confirmation of modern shower/bathhouse facilities with hot water and flush toilets\n"
    "- Confirmation of on-site dump station availability\n"
    "- Documentation of ADA-accessible campsites and features\n"
    "- Documentation of the pet-friendly policy with specific rules\n"
    "- The maximum stay limit policy\n"
    "- Information about the online reservation system and advance booking window\n"
    "- Reference URLs to official state park websites or reservation pages documenting each of these features"
)

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    official_name: Optional[str] = None
    state: Optional[str] = None
    lake_name: Optional[str] = None

    # Stated confirmations (as described in the answer; strings preferred for flexibility)
    is_state_park: Optional[str] = None

    hookup_type: Optional[str] = None  # e.g., "full hookups", "electric hookups", "both"
    shower_house: Optional[str] = None  # wording indicating showers with hot water
    flush_toilets: Optional[str] = None
    dump_station: Optional[str] = None
    ada_accessible_sites: Optional[str] = None
    accessible_features: Optional[str] = None  # optional details about features
    pets_allowed: Optional[str] = None
    pet_rules: Optional[str] = None  # optional details
    max_stay_policy: Optional[str] = None
    online_reservation: Optional[str] = None
    booking_window: Optional[str] = None  # e.g., "6 months", "180 days"

    # Source URLs grouped by feature; must be URLs explicitly present in the answer
    identification_urls: List[str] = Field(default_factory=list)
    waterfront_urls: List[str] = Field(default_factory=list)
    hookup_urls: List[str] = Field(default_factory=list)
    restroom_urls: List[str] = Field(default_factory=list)
    dump_urls: List[str] = Field(default_factory=list)
    ada_urls: List[str] = Field(default_factory=list)
    pet_urls: List[str] = Field(default_factory=list)
    stay_urls: List[str] = Field(default_factory=list)
    reservation_urls: List[str] = Field(default_factory=list)


class CampgroundExtraction(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return (
        "Extract up to four official state park campgrounds described in the answer. "
        "For each campground, return an object with the following fields exactly as stated in the answer (do not invent):\n"
        "- official_name: The official name of the campground.\n"
        "- state: The U.S. state where the campground is located.\n"
        "- lake_name: The name of the lake that the campground is directly on.\n"
        "- is_state_park: A phrase confirming it is an official state park facility (if stated).\n"
        "- hookup_type: Text indicating the hookup type offered (e.g., 'full hookups', 'electric hookups').\n"
        "- shower_house: Text confirming presence of modern shower houses/bathhouses with hot water.\n"
        "- flush_toilets: Text confirming presence of flush toilets.\n"
        "- dump_station: Text confirming an on-site RV dump station is available.\n"
        "- ada_accessible_sites: Text confirming designated ADA-accessible campsites exist.\n"
        "- accessible_features: Optional text describing accessible features (e.g., paved pathways, accessible tables, accessible restrooms).\n"
        "- pets_allowed: Text confirming pets are explicitly allowed in designated campsites.\n"
        "- pet_rules: Optional text with specific pet rules (e.g., leash, vaccinations).\n"
        "- max_stay_policy: Text stating the maximum stay limit policy.\n"
        "- online_reservation: Text confirming an online reservation system exists for this campground.\n"
        "- booking_window: Text indicating the advance booking window (e.g., 'at least 6 months', '180 days').\n"
        "- identification_urls: Array of official state park or reservation page URLs used to identify/confirm state park status.\n"
        "- waterfront_urls: Array of URLs documenting lakefront sites with direct lake access and the lake's name.\n"
        "- hookup_urls: Array of URLs documenting hookup types.\n"
        "- restroom_urls: Array of URLs documenting shower houses with hot water and flush toilets.\n"
        "- dump_urls: Array of URLs documenting the on-site RV dump station.\n"
        "- ada_urls: Array of URLs documenting ADA-accessible campsites and accessibility features.\n"
        "- pet_urls: Array of URLs documenting the pet policy and rules.\n"
        "- stay_urls: Array of URLs documenting the maximum stay limit policy.\n"
        "- reservation_urls: Array of URLs documenting the online reservation system and booking window.\n\n"
        "Rules:\n"
        "1) Extract only from the answer text; do not guess. If a field is not present, set it to null (for strings) or [] (for URLs).\n"
        "2) For URL fields, extract only actual URLs explicitly present in the answer (plain URL or markdown link). Do not invent URLs.\n"
        "3) If the answer lists more than four campgrounds, include only the first four. If fewer than four, include all provided.\n"
        "4) Use full URLs including protocol (http/https).\n"
        "Return a JSON object with one key: 'campgrounds', which is an array of campground objects as described."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(index_zero_based: int) -> str:
    mapping = ["First", "Second", "Third", "Fourth"]
    return mapping[index_zero_based] if 0 <= index_zero_based < 4 else f"#{index_zero_based + 1}"


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _urls_present(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification logic per campground                                           #
# --------------------------------------------------------------------------- #
async def verify_campground(
    evaluator: Evaluator,
    root_node,
    cg: CampgroundItem,
    idx: int,
) -> None:
    cg_idx = idx + 1
    ordinal = _ordinal(idx)

    # Campground container node (non-critical; parent root handles overall)
    cg_node = evaluator.add_parallel(
        id=f"campground_{cg_idx}",
        desc=f"{ordinal} qualifying lakefront state park campground",
        parent=root_node,
        critical=False,
    )

    # 1) Identification
    ident_node = evaluator.add_parallel(
        id=f"campground_{cg_idx}_identification",
        desc="Campground is correctly identified with official name and state",
        parent=cg_node,
        critical=True,
    )
    # 1.1 Official name provided
    evaluator.add_custom_node(
        result=_nonempty(cg.official_name),
        id=f"campground_{cg_idx}_official_name",
        desc="The campground's official name is provided",
        parent=ident_node,
        critical=True,
    )
    # 1.2 State provided
    evaluator.add_custom_node(
        result=_nonempty(cg.state),
        id=f"campground_{cg_idx}_state_location",
        desc="The state where the campground is located is provided",
        parent=ident_node,
        critical=True,
    )
    # 1.3 Identification URL provided
    ident_url_exist = evaluator.add_custom_node(
        result=_urls_present(cg.identification_urls),
        id=f"campground_{cg_idx}_identification_url",
        desc="A reference URL to the official state park website or reservation page is provided",
        parent=ident_node,
        critical=True,
    )
    # 1.4 State park status verification
    sp_status_leaf = evaluator.add_leaf(
        id=f"campground_{cg_idx}_state_park_status",
        desc="The campground is confirmed to be an official state park facility managed by a state agency",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This campground is an official state park facility managed by a state agency.",
        node=sp_status_leaf,
        sources=cg.identification_urls,
        additional_instruction=(
            "Verify that the provided source(s) clearly indicate the campground is part of a state park system or managed by a state agency "
            "(e.g., state parks department, DNR). Accept official park pages or official reservation pages that explicitly indicate state park affiliation."
        ),
        extra_prerequisites=[ident_url_exist],
    )

    # 2) Waterfront location
    waterfront_node = evaluator.add_parallel(
        id=f"campground_{cg_idx}_waterfront_location",
        desc="Campground provides lakefront campsites with direct lake access",
        parent=cg_node,
        critical=True,
    )
    wf_url_exist = evaluator.add_custom_node(
        result=_urls_present(cg.waterfront_urls),
        id=f"campground_{cg_idx}_waterfront_url",
        desc="A reference URL documenting the lakefront location feature is provided",
        parent=waterfront_node,
        critical=True,
    )
    # 2.1 Lakefront sites verification
    lakefront_leaf = evaluator.add_leaf(
        id=f"campground_{cg_idx}_lakefront_sites",
        desc="The campground has campsites located directly on a lake shoreline with direct water access",
        parent=waterfront_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The campground has campsites located directly on a lake shoreline with direct lake access from the campsite.",
        node=lakefront_leaf,
        sources=cg.waterfront_urls,
        additional_instruction=(
            "Confirm that sites are directly on the lake shoreline and provide immediate lake access from the campsite itself, "
            "not merely nearby or within walking distance. Look for explicit phrases like 'lakefront sites', 'waterfront sites', or maps showing sites on the shoreline."
        ),
        extra_prerequisites=[wf_url_exist],
    )
    # 2.2 Lake name provided (existence check)
    evaluator.add_custom_node(
        result=_nonempty(cg.lake_name),
        id=f"campground_{cg_idx}_lake_name",
        desc="The name of the lake where the campground is located is provided",
        parent=waterfront_node,
        critical=True,
    )

    # 3) Hookup type
    hookup_node = evaluator.add_parallel(
        id=f"campground_{cg_idx}_hookup_type",
        desc="Campground provides either full hookups or electric hookups",
        parent=cg_node,
        critical=True,
    )
    hookup_url_exist = evaluator.add_custom_node(
        result=_urls_present(cg.hookup_urls),
        id=f"campground_{cg_idx}_hookup_url",
        desc="A reference URL documenting the hookup types available is provided",
        parent=hookup_node,
        critical=True,
    )
    hookup_leaf = evaluator.add_leaf(
        id=f"campground_{cg_idx}_hookup_availability",
        desc="The campground offers either full hookup sites (water/electric/sewer) or electric hookup sites (water/electric)",
        parent=hookup_node,
        critical=True,
    )
    # Form claim based on extracted type if available
    if _nonempty(cg.hookup_type):
        claim_hookup = f"The campground offers {cg.hookup_type.strip()}."
    else:
        claim_hookup = "The campground offers either full hookups (water, electric, sewer) or electric hookups (water and electric)."
    await evaluator.verify(
        claim=claim_hookup,
        node=hookup_leaf,
        sources=cg.hookup_urls,
        additional_instruction=(
            "Confirm hookup availability. Accept phrases like 'full hookups', 'W/E/S', 'electric and water', 'W/E'. "
            "For 'full hookups', sewer must be included. For 'electric hookups', electric and water must be present."
        ),
        extra_prerequisites=[hookup_url_exist],
    )

    # 4) Restroom facilities
    restroom_node = evaluator.add_parallel(
        id=f"campground_{cg_idx}_restroom_facilities",
        desc="Campground provides modern restroom facilities with showers",
        parent=cg_node,
        critical=True,
    )
    rr_url_exist = evaluator.add_custom_node(
        result=_urls_present(cg.restroom_urls),
        id=f"campground_{cg_idx}_restroom_url",
        desc="A reference URL documenting the restroom facilities is provided",
        parent=restroom_node,
        critical=True,
    )
    shower_leaf = evaluator.add_leaf(
        id=f"campground_{cg_idx}_shower_house",
        desc="The campground has modern shower houses or bathhouses with hot water",
        parent=restroom_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The campground provides modern shower houses or bathhouses with hot water.",
        node=shower_leaf,
        sources=cg.restroom_urls,
        additional_instruction="Look for 'shower house', 'bathhouse', or 'showers' with hot water explicitly mentioned.",
        extra_prerequisites=[rr_url_exist],
    )
    flush_leaf = evaluator.add_leaf(
        id=f"campground_{cg_idx}_flush_toilets",
        desc="The campground provides flush toilets in the restroom facilities",
        parent=restroom_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The campground's restroom facilities include flush toilets.",
        node=flush_leaf,
        sources=cg.restroom_urls,
        additional_instruction="Verify that restrooms specify 'flush toilets' or equivalent terminology.",
        extra_prerequisites=[rr_url_exist],
    )

    # 5) Dump station
    dump_node = evaluator.add_parallel(
        id=f"campground_{cg_idx}_dump_station",
        desc="Campground has an on-site RV dump station",
        parent=cg_node,
        critical=True,
    )
    dump_url_exist = evaluator.add_custom_node(
        result=_urls_present(cg.dump_urls),
        id=f"campground_{cg_idx}_dump_url",
        desc="A reference URL documenting the dump station availability is provided",
        parent=dump_node,
        critical=True,
    )
    dump_leaf = evaluator.add_leaf(
        id=f"campground_{cg_idx}_dump_availability",
        desc="An on-site dump station for RV waste disposal is available at the campground",
        parent=dump_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The campground has an on-site RV dump station.",
        node=dump_leaf,
        sources=cg.dump_urls,
        additional_instruction="Confirm that an RV dump station is present on site (not off-site).",
        extra_prerequisites=[dump_url_exist],
    )

    # 6) ADA accessibility
    ada_node = evaluator.add_parallel(
        id=f"campground_{cg_idx}_ada_accessibility",
        desc="Campground provides ADA-accessible facilities",
        parent=cg_node,
        critical=True,
    )
    ada_url_exist = evaluator.add_custom_node(
        result=_urls_present(cg.ada_urls),
        id=f"campground_{cg_idx}_ada_url",
        desc="A reference URL documenting the ADA accessibility features is provided",
        parent=ada_node,
        critical=True,
    )
    ada_sites_leaf = evaluator.add_leaf(
        id=f"campground_{cg_idx}_accessible_sites",
        desc="The campground has designated ADA-accessible campsites",
        parent=ada_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The campground has designated ADA-accessible campsites.",
        node=ada_sites_leaf,
        sources=cg.ada_urls,
        additional_instruction="Look for 'ADA-accessible campsites', 'accessible campsite', or similar explicit designation.",
        extra_prerequisites=[ada_url_exist],
    )
    # Note: Optional accessible features are not enforced here to keep parent-child critical consistency.

    # 7) Pet policy
    pet_node = evaluator.add_parallel(
        id=f"campground_{cg_idx}_pet_policy",
        desc="Campground has a clearly stated pet-friendly policy",
        parent=cg_node,
        critical=True,
    )
    pet_url_exist = evaluator.add_custom_node(
        result=_urls_present(cg.pet_urls),
        id=f"campground_{cg_idx}_pet_url",
        desc="A reference URL documenting the pet policy is provided",
        parent=pet_node,
        critical=True,
    )
    pets_allowed_leaf = evaluator.add_leaf(
        id=f"campground_{cg_idx}_pets_allowed",
        desc="Pets are explicitly allowed in designated campsites at the campground",
        parent=pet_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Pets are explicitly allowed in designated campsites at the campground.",
        node=pets_allowed_leaf,
        sources=cg.pet_urls,
        additional_instruction="Verify that pets are allowed and that the policy explicitly permits pets at campsites. Pet rules may be listed; presence is helpful but not strictly required for this leaf.",
        extra_prerequisites=[pet_url_exist],
    )

    # 8) Maximum stay limit
    stay_node = evaluator.add_parallel(
        id=f"campground_{cg_idx}_stay_limit",
        desc="Campground has a stated maximum stay limit policy",
        parent=cg_node,
        critical=True,
    )
    stay_url_exist = evaluator.add_custom_node(
        result=_urls_present(cg.stay_urls),
        id=f"campground_{cg_idx}_stay_url",
        desc="A reference URL documenting the maximum stay policy is provided",
        parent=stay_node,
        critical=True,
    )
    max_stay_leaf = evaluator.add_leaf(
        id=f"campground_{cg_idx}_max_stay",
        desc="A maximum stay limit policy is clearly stated (typically 14 consecutive nights within a 30-day period)",
        parent=stay_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The campground has a clearly stated maximum stay limit policy (e.g., 14 consecutive nights within a 30-day period).",
        node=max_stay_leaf,
        sources=cg.stay_urls,
        additional_instruction="Confirm that an explicit maximum stay policy is stated. Common phrasing is '14 nights in a 30-day period'.",
        extra_prerequisites=[stay_url_exist],
    )

    # 9) Reservation system
    res_node = evaluator.add_parallel(
        id=f"campground_{cg_idx}_reservation_system",
        desc="Campground offers online reservations with at least 6-month advance booking",
        parent=cg_node,
        critical=True,
    )
    res_url_exist = evaluator.add_custom_node(
        result=_urls_present(cg.reservation_urls),
        id=f"campground_{cg_idx}_reservation_url",
        desc="A direct link to the reservation page or reservation system information is provided",
        parent=res_node,
        critical=True,
    )
    online_res_leaf = evaluator.add_leaf(
        id=f"campground_{cg_idx}_online_reservations",
        desc="The campground has an online reservation system available",
        parent=res_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The campground offers an online reservation system.",
        node=online_res_leaf,
        sources=cg.reservation_urls,
        additional_instruction="Verify that reservations can be made online via an official portal (state parks system or official reservation partner).",
        extra_prerequisites=[res_url_exist],
    )
    booking_window_leaf = evaluator.add_leaf(
        id=f"campground_{cg_idx}_booking_window",
        desc="Reservations can be made at least 6 months in advance",
        parent=res_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Reservations can be made at least 6 months in advance.",
        node=booking_window_leaf,
        sources=cg.reservation_urls,
        additional_instruction="Accept equivalents like '180 days'. If the window is longer (e.g., 11 months), it still satisfies 'at least 6 months'.",
        extra_prerequisites=[res_url_exist],
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator (root node kept non-critical to allow mix of critical/non-critical children)
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

    # Extract structured information
    extraction = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundExtraction,
        extraction_name="campground_extraction",
    )

    # Prepare exactly four campgrounds (pad with empty if fewer, trim if more)
    campgrounds: List[CampgroundItem] = list(extraction.campgrounds[:4])
    while len(campgrounds) < 4:
        campgrounds.append(CampgroundItem())

    # Verify each campground
    for idx, cg in enumerate(campgrounds):
        await verify_campground(evaluator, root, cg, idx)

    # Geographic diversity check: at least three distinct states across the four campgrounds
    unique_states = sorted({(cg.state or "").strip() for cg in campgrounds if _nonempty(cg.state)})
    evaluator.add_custom_node(
        result=len(unique_states) >= 3,
        id="geographic_diversity",
        desc="The four campgrounds span at least three different U.S. states",
        parent=root,
        critical=True,
    )
    evaluator.add_custom_info(
        info={"extracted_states": unique_states, "distinct_state_count": len(unique_states)},
        info_type="geographic_diversity_info",
    )

    # Return the evaluation summary
    return evaluator.get_summary()