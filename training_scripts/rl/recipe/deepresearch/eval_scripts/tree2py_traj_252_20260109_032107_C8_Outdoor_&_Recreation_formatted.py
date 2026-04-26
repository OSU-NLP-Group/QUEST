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
TASK_ID = "tx_group_camping_2026"
TASK_DESCRIPTION = (
    "My family is planning a large reunion camping trip in Texas for summer 2026. We need to identify 4 different Texas state park campgrounds that meet all of the following requirements for our group of 50+ people:\n\n"
    "1. Group Camping: Must offer group camping sites that can accommodate at least 20 people per site\n"
    "2. RV Accommodation: Must have RV campsites that accept recreational vehicles up to at least 35 feet in length\n"
    "3. Electrical Hookups: RV sites must provide electrical hookups (either 30-amp or 50-amp service)\n"
    "4. Accessibility: Must have at least 2 ADA-accessible campsites with proper accessible facilities\n"
    "5. Pet Policy: Must allow pets with standard leash requirements (maximum 6-foot leash length)\n"
    "6. Shower Facilities: Must provide shower facilities with hot water\n"
    "7. Potable Water: Must provide access to potable drinking water either at campsites or at central locations\n"
    "8. RV Dump Station: Must have an RV dump station for wastewater disposal\n"
    "9. Stay Duration: Must allow camping stays of at least 14 consecutive days\n"
    "10. Reservation System: Must use an online reservation system that allows booking at least 6 months in advance\n"
    "11. Campfire Facilities: Must provide designated fire rings or fire pits at the campsites\n"
    "12. Picnic Amenities: Campsites must include picnic tables\n\n"
    "For each of the 4 campgrounds, provide the campground name, the specific Texas state park where it's located, and a reference URL from the official Texas State Parks website or the Texas State Parks reservation system that confirms these amenities and policies."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampgroundEntry(BaseModel):
    campground_name: Optional[str] = None
    park_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CampgroundExtraction(BaseModel):
    campgrounds: List[CampgroundEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return (
        "Extract all Texas state park campground entries mentioned in the answer. For each campground, return:\n"
        "1. campground_name: The specific campground name (e.g., 'Lakeview Campground', or a general label if only one campground area exists)\n"
        "2. park_name: The Texas State Park name where the campground is located (e.g., 'Garner State Park')\n"
        "3. reference_urls: A list of URLs cited in the answer that are official Texas State Parks (TPWD) webpages "
        "   (typically tpwd.texas.gov/state-parks/...) or the Texas State Parks reservation system pages (ReserveAmerica domain).\n"
        "   If non-official URLs are present in the answer, include them as well—but prefer the official ones when extracting.\n"
        "Return a JSON object with a 'campgrounds' array of objects containing these fields. If any field is missing for an item, set it to null or an empty list accordingly."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize(s: Optional[str]) -> str:
    return (s or "").strip()


def _ordinal(idx: int) -> str:
    return ["First", "Second", "Third", "Fourth"][idx] if 0 <= idx < 4 else f"#{idx+1}"


def _distinct_pairs(entries: List[CampgroundEntry]) -> bool:
    seen = set()
    for e in entries:
        key = (_normalize(e.campground_name).lower(), _normalize(e.park_name).lower())
        if key in seen:
            return False
        seen.add(key)
    return True


# --------------------------------------------------------------------------- #
# Verification per campground                                                 #
# --------------------------------------------------------------------------- #
async def verify_campground(
    evaluator: Evaluator,
    parent_node,
    cg: CampgroundEntry,
    index: int,
) -> None:
    """
    Build verification subtree for a single campground and run verifications.
    All children are critical under the campground node (since the overall task requires
    each campground to meet ALL constraints). The parent campground node is also critical
    to satisfy framework constraints when the top-level 'complete set' node is critical.
    """
    ordinal = _ordinal(index)
    cg_node = evaluator.add_parallel(
        id=f"Campground_{index+1}",
        desc=f"{ordinal} campground entry.",
        parent=parent_node,
        critical=True  # Must be critical due to critical parent requirements and task strictness
    )

    # Existence checks (critical; gate subsequent verifications via auto preconditions)
    name_exists = bool(_normalize(cg.campground_name))
    park_exists = bool(_normalize(cg.park_name))

    evaluator.add_custom_node(
        result=name_exists,
        id=f"camp_{index+1}_name_provided",
        desc="Provides the campground name.",
        parent=cg_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=park_exists,
        id=f"camp_{index+1}_park_name_provided",
        desc="Provides the specific Texas state park name where the campground is located.",
        parent=cg_node,
        critical=True
    )

    # Official reference URL check (verification leaf; ensures at least one official TPWD or reservation system page)
    official_ref_leaf = evaluator.add_leaf(
        id=f"camp_{index+1}_official_reference_url",
        desc="Provides at least one official Texas State Parks (TPWD) or Texas State Parks reservation/ReserveAmerica URL for this campground/park that supports the listed amenities/policies.",
        parent=cg_node,
        critical=True
    )
    claim_official = (
        "This webpage is an official Texas State Parks (TPWD) page or the Texas State Parks reservation system page "
        "for the specified park/camping area."
    )
    await evaluator.verify(
        claim=claim_official,
        node=official_ref_leaf,
        sources=cg.reference_urls,
        additional_instruction=(
            "Pass if any URL is clearly an official TPWD page (e.g., domain 'tpwd.texas.gov', 'tpwd.texas.gov/state-parks/...') "
            "or an official Texas State Parks reservation system page (e.g., 'texasstateparks.reserveamerica.com' or ReserveAmerica "
            "pages explicitly labeled for Texas State Parks). The content should pertain to the stated park or its camping facilities."
        )
    )

    # Location in Texas State Park (TPWD)
    located_leaf = evaluator.add_leaf(
        id=f"camp_{index+1}_located_in_tpwd",
        desc="The campground is confirmed to be within a Texas State Park (TPWD).",
        parent=cg_node,
        critical=True
    )
    cg_name_for_claim = _normalize(cg.campground_name) or "the campground"
    park_name_for_claim = _normalize(cg.park_name) or "the park"
    claim_located = (
        f"The campground '{cg_name_for_claim}' is located within Texas State Park '{park_name_for_claim}', "
        "operated by Texas Parks & Wildlife Department (TPWD)."
    )
    await evaluator.verify(
        claim=claim_located,
        node=located_leaf,
        sources=cg.reference_urls,
        additional_instruction=(
            "Look for explicit park identification on the page indicating it belongs to the Texas State Parks system "
            "(TPWD) and that the described campground or camping area is within that park."
        )
    )

    # Constraint leaves (all critical)
    def add_and_verify_req(node_id_suffix: str, desc: str, claim_text: str, add_ins: str):
        leaf = evaluator.add_leaf(
            id=f"camp_{index+1}_{node_id_suffix}",
            desc=desc,
            parent=cg_node,
            critical=True
        )
        return evaluator.verify(
            claim=claim_text,
            node=leaf,
            sources=cg.reference_urls,
            additional_instruction=add_ins
        )

    # 1. Group Camping Capacity >= 20 per site
    await add_and_verify_req(
        "group_camping_capacity",
        "Offers group camping sites that accommodate at least 20 people per site.",
        "This park offers group camping sites that accommodate at least 20 people per site.",
        "Check for phrases like 'group camp', 'group campsite', 'group area', and capacity information (e.g., 'up to 20 people', '20+ persons'). "
        "If multiple group sites exist, it's acceptable if any one site accommodates at least 20 people."
    )

    # 2. RV Max Length >= 35 ft
    await add_and_verify_req(
        "rv_max_length_at_least_35ft",
        "Has RV campsites that accept RVs up to at least 35 feet in length.",
        "RV campsites at this park accept recreational vehicles up to at least 35 feet long.",
        "Look for 'maximum RV length', 'max trailer length', or related specs. Accept values of 35 feet or greater."
    )

    # 3. Electrical Hookups (30 or 50 amp)
    await add_and_verify_req(
        "electrical_hookups_30_or_50a",
        "RV sites provide electrical hookups (30-amp or 50-amp service).",
        "RV campsites provide electrical hookups of 30-amp or 50-amp service.",
        "Look for 'electric', 'electricity', '30 amp', '50 amp', 'hookups', or 'full hookups' on the campsite page."
    )

    # 4. Accessibility: at least 2 ADA-accessible campsites
    await add_and_verify_req(
        "at_least_2_ada_campsites",
        "Has at least 2 ADA-accessible campsites with proper accessible facilities.",
        "There are at least two ADA-accessible campsites with appropriate accessible facilities.",
        "Look for 'ADA accessible sites', 'accessible campsite', 'wheelchair accessible', or explicit counts; "
        "if phrased as 'several accessible campsites', consider whether that implies at least 2."
    )

    # 5. Pets allowed; leash <= 6 feet
    await add_and_verify_req(
        "pets_allowed_6ft_leash",
        "Allows pets with a maximum 6-foot leash requirement.",
        "Pets are allowed and must be kept on a leash no longer than six feet.",
        "Look for pet policy statements such as 'pets allowed' and 'leash length 6 feet' (or equivalent phrasing)."
    )

    # 6. Hot showers available
    await add_and_verify_req(
        "hot_showers_available",
        "Provides shower facilities with hot water.",
        "The park provides shower facilities with hot water.",
        "Look for 'restrooms with showers', 'hot showers', 'bathhouse', or 'showers available'."
    )

    # 7. Potable water access
    await add_and_verify_req(
        "potable_water_access",
        "Provides access to potable drinking water at campsites or at central locations.",
        "Potable drinking water is available either at the campsites or at central locations within the park.",
        "Look for 'potable water', 'drinking water', 'water available', 'spigots', or taps near campsites."
    )

    # 8. RV dump station
    await add_and_verify_req(
        "rv_dump_station_available",
        "Has an RV dump station for wastewater disposal.",
        "An RV dump station is available for wastewater disposal.",
        "Search for 'dump station' or 'sewage disposal' amenities."
    )

    # 9. Allows at least 14 consecutive days
    await add_and_verify_req(
        "allows_14_consecutive_days",
        "Allows camping stays of at least 14 consecutive days.",
        "Camping stays of at least 14 consecutive days are permitted.",
        "Look for stay limits like 'up to 14 days' or 'maximum of 14 consecutive nights'."
    )

    # 10. Reservation system allows booking >= 6 months in advance
    await add_and_verify_req(
        "reserveamerica_6_months_ahead",
        "Uses the Texas State Parks online reservation system (ReserveAmerica) and allows booking at least 6 months in advance.",
        "The online reservation system allows booking at least six months in advance.",
        "Verify via the official reservation system page whether bookings are open six months or more in advance; "
        "phrasing like 'book up to 6 months ahead' or any policy stating a 6+ months window should pass."
    )

    # 11. Fire ring or fire pit provided
    await add_and_verify_req(
        "fire_ring_or_fire_pit",
        "Provides designated fire rings or fire pits at campsites.",
        "Campsites provide designated fire rings or fire pits.",
        "Look for mentions of 'fire ring', 'fire pit', or 'designated fire area' at campsites."
    )

    # 12. Picnic table included
    await add_and_verify_req(
        "picnic_table_included",
        "Campsites include picnic tables.",
        "Campsites include picnic tables.",
        "Look for 'picnic table' or similar phrasing in campsite amenities."
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
    Evaluate the answer for the Texas group camping criteria task.
    """
    # Initialize evaluator
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

    # Extract campground entries from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundExtraction,
        extraction_name="campgrounds_extraction",
    )

    # Record some custom info
    evaluator.add_custom_info(
        info={"total_campgrounds_found_in_answer": len(extracted.campgrounds)},
        info_type="extraction_stats",
        info_name="campground_count"
    )

    # Build the top-level critical node (complete set)
    complete_set_node = evaluator.add_parallel(
        id="Complete_Set_of_Qualifying_Campgrounds",
        desc="The solution provides exactly four distinct Texas state park campgrounds, each satisfying all constraints with official references.",
        parent=root,
        critical=True
    )

    # Exactly 4 distinct campgrounds check (critical leaf via custom node)
    # Check strictly that the answer lists exactly 4, and they are distinct.
    first_four = extracted.campgrounds[:4]
    all_count_is_4 = len(extracted.campgrounds) == 4
    distinct_in_first_four = _distinct_pairs(first_four) if len(first_four) == 4 else False

    evaluator.add_custom_node(
        result=(all_count_is_4 and distinct_in_first_four),
        id="Exactly_4_Distinct_Campgrounds",
        desc="The response lists exactly 4 campgrounds and they are all different (no duplicates, no extra campgrounds).",
        parent=complete_set_node,
        critical=True
    )

    # Prepare four campground entries for verification (pad if fewer than 4 in answer)
    while len(first_four) < 4:
        first_four.append(CampgroundEntry())

    # Verify each campground subtree (all constraints)
    for idx in range(4):
        await verify_campground(
            evaluator=evaluator,
            parent_node=complete_set_node,
            cg=first_four[idx],
            index=idx
        )

    # Return evaluation summary
    return evaluator.get_summary()