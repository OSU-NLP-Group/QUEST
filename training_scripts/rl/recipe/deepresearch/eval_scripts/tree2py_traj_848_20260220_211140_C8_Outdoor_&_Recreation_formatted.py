import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "rv_campgrounds_2026"
TASK_DESCRIPTION = (
    "You are planning a summer 2026 RV camping road trip and need to identify three suitable campgrounds, one in each of three different regions of the United States. "
    "Your RV is 35 feet long, and you will be traveling with two small dogs. One member of your party uses a wheelchair and requires accessible facilities.\n\n"
    "Find three campgrounds that meet the following criteria:\n\n"
    "Campground 1 (Mid-Atlantic Region):\n"
    "- Must be located in one of these states: Maryland, Virginia, Pennsylvania, Delaware, New Jersey, or New York\n"
    "- Must accommodate RVs that are at least 35 feet in length\n"
    "- Must offer full hookup sites (water, electric, and sewer connections)\n"
    "- Must allow pets\n"
    "- Must accept reservations at least 6 months in advance\n"
    "- Must be open and operational during June, July, and August\n\n"
    "Campground 2 (Southeastern Region):\n"
    "- Must be located in one of these states: North Carolina, South Carolina, Georgia, Florida, Alabama, Tennessee, or Kentucky\n"
    "- Must accommodate RVs that are at least 35 feet in length\n"
    "- Must offer full hookup sites (water, electric, and sewer connections)\n"
    "- Must allow pets\n"
    "- Must provide direct access to a lake with boat launch facilities\n"
    "- Must have shower and restroom facilities available to campers\n\n"
    "Campground 3 (Great Lakes Region):\n"
    "- Must be located in one of these states: Ohio, Michigan, Indiana, Illinois, Wisconsin, or Minnesota\n"
    "- Must accommodate RVs that are at least 35 feet in length\n"
    "- Must offer at least 30-amp electrical service at campsites\n"
    "- Must allow pets\n"
    "- Must have ADA-accessible campsites available\n"
    "- Must use an online reservation system (such as Recreation.gov or a state park reservation system)\n\n"
    "For each campground, provide:\n"
    "1. The complete name of the campground\n"
    "2. The state where it is located\n"
    "3. A brief description explaining how it meets each of the required criteria\n"
    "4. A reference URL where this information can be verified"
)

MID_ATLANTIC_STATES: Set[str] = {
    "Maryland", "Virginia", "Pennsylvania", "Delaware", "New Jersey", "New York"
}
SOUTHEASTERN_STATES: Set[str] = {
    "North Carolina", "South Carolina", "Georgia", "Florida", "Alabama", "Tennessee", "Kentucky"
}
GREAT_LAKES_STATES: Set[str] = {
    "Ohio", "Michigan", "Indiana", "Illinois", "Wisconsin", "Minnesota"
}

STATE_ABBR_TO_FULL = {
    "MD": "Maryland",
    "VA": "Virginia",
    "PA": "Pennsylvania",
    "DE": "Delaware",
    "NJ": "New Jersey",
    "NY": "New York",
    "NC": "North Carolina",
    "SC": "South Carolina",
    "GA": "Georgia",
    "FL": "Florida",
    "AL": "Alabama",
    "TN": "Tennessee",
    "KY": "Kentucky",
    "OH": "Ohio",
    "MI": "Michigan",
    "IN": "Indiana",
    "IL": "Illinois",
    "WI": "Wisconsin",
    "MN": "Minnesota",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    description: Optional[str] = None  # Optional text the agent may provide


class CampgroundExtraction(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return (
        "Extract up to three campgrounds listed in the answer. For each, return:\n"
        "1. name: The complete name of the campground\n"
        "2. state: The U.S. state where it is located (use the state name or its postal abbreviation)\n"
        "3. reference_urls: All verification URLs provided for this campground. Include every URL shown in the answer and return them in a list. "
        "Understand that URLs might appear as plain text or markdown links.\n"
        "4. description: Optional brief description provided by the answer about how it meets criteria.\n"
        "If any field is missing, set it to null (or an empty list for reference_urls). Extract campgrounds in the order they appear."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip()
    # Remove common extra words
    for token in ["state of", "state", "usa", "united states", "u.s."]:
        s = s.replace(token, "")
    s = s.strip()
    # Abbreviation mapping
    abbr = s.upper()
    if abbr in STATE_ABBR_TO_FULL:
        return STATE_ABBR_TO_FULL[abbr]
    # Title case normalize
    s_title = " ".join(w.capitalize() for w in s.split())
    return s_title


def get_sources(item: CampgroundItem) -> List[str]:
    urls = [u.strip() for u in (item.reference_urls or []) if isinstance(u, str) and u.strip()]
    return urls


def cg_desc_by_index(idx: int) -> str:
    if idx == 0:
        return "First campground meeting all specified criteria"
    if idx == 1:
        return "Second campground meeting all specified criteria"
    return "Third campground meeting all specified criteria"


def region_name_and_allowed_states(idx: int) -> (str, Set[str]):
    if idx == 0:
        return "Mid-Atlantic", MID_ATLANTIC_STATES
    if idx == 1:
        return "Southeastern", SOUTHEASTERN_STATES
    return "Great Lakes", GREAT_LAKES_STATES


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_campground(
    evaluator: Evaluator,
    root_node,
    item: CampgroundItem,
    idx: int,
) -> None:
    """
    Build the verification subtree for a single campground and run all checks.
    Each criterion is a distinct (binary) leaf node, verified against provided URLs.
    """
    region_name, allowed_states = region_name_and_allowed_states(idx)
    cg_node = evaluator.add_parallel(
        id=f"Campground_{idx+1}",
        desc=cg_desc_by_index(idx),
        parent=root_node,
        critical=False,
    )

    # Optional existence/source check to enforce source-grounding policy
    # Gate all subsequent checks behind having at least one reference URL plus name/state.
    has_required = (
        (item.name is not None and item.name.strip() != "") and
        (item.state is not None and item.state.strip() != "") and
        (len(get_sources(item)) > 0)
    )
    evaluator.add_custom_node(
        result=has_required,
        id=f"C{idx+1}_Reference_URL_Provided",
        desc=f"Campground {idx+1}: has name, state, and at least one reference URL",
        parent=cg_node,
        critical=True
    )

    # Normalize state
    norm_state = normalize_state(item.state) or ""

    # 1) State location meets regional constraint
    state_loc_node = evaluator.add_leaf(
        id=f"C{idx+1}_State_Location",
        desc=(
            "Located in a "
            + ("Mid-Atlantic state (Maryland, Virginia, Pennsylvania, Delaware, New Jersey, or New York)" if idx == 0 else
               "Southeastern state (North Carolina, South Carolina, Georgia, Florida, Alabama, Tennessee, or Kentucky)" if idx == 1 else
               "Great Lakes state (Ohio, Michigan, Indiana, Illinois, Wisconsin, or Minnesota)")
        ),
        parent=cg_node,
        critical=True,
    )
    state_list_text = ", ".join(sorted(list(allowed_states)))
    location_claim = (
        f"The campground '{item.name}' is located in {norm_state}, which must be one of the allowed {region_name} states: {state_list_text}."
    )
    await evaluator.verify(
        claim=location_claim,
        node=state_loc_node,
        sources=get_sources(item),
        additional_instruction=(
            "Use the webpage(s) to confirm the state's location of the campground. "
            "Then treat the claim as supported only if that state is within the provided allowed list."
        ),
    )

    # 2) RV length >= 35 ft
    rv_len_node = evaluator.add_leaf(
        id=f"C{idx+1}_RV_Length",
        desc="Accommodates RVs up to at least 35 feet in length",
        parent=cg_node,
        critical=True,
    )
    rv_claim = (
        "This campground accommodates RVs that are at least 35 feet in length. "
        "Accept as supported if site length, pad length, or maximum RV length is 35 ft or greater (e.g., 40 ft, 45 ft)."
    )
    await evaluator.verify(
        claim=rv_claim,
        node=rv_len_node,
        sources=get_sources(item),
        additional_instruction="Look for site length limits, maximum RV length, or examples indicating 35 ft or more.",
    )

    # 3) Pet friendly
    pet_node = evaluator.add_leaf(
        id=f"C{idx+1}_Pet_Friendly",
        desc="Allows pets with proper leash and vaccination requirements",
        parent=cg_node,
        critical=True,
    )
    pet_claim = (
        "Pets are allowed at this campground. "
        "General statements like 'pets allowed' or pet policy pages qualify."
    )
    await evaluator.verify(
        claim=pet_claim,
        node=pet_node,
        sources=get_sources(item),
        additional_instruction="Look for 'pets allowed', 'pet policy', or related statements; do not require exact wording about leashes/vaccinations.",
    )

    # Region-specific criteria
    if idx == 0:
        # Mid-Atlantic: full hookups, reservations window >= 6 months, open in summer
        full_hook_node = evaluator.add_leaf(
            id=f"C{idx+1}_Full_Hookups",
            desc="Offers full hookup sites (water, electric, and sewer)",
            parent=cg_node,
            critical=True,
        )
        full_hook_claim = "This campground offers full hookup sites that include water, electric, and sewer connections."
        await evaluator.verify(
            claim=full_hook_claim,
            node=full_hook_node,
            sources=get_sources(item),
            additional_instruction="Accept synonyms like 'full hookups' or 'W/E/S'.",
        )

        res_window_node = evaluator.add_leaf(
            id=f"C{idx+1}_Reservation_Window",
            desc="Accepts reservations at least 6 months in advance",
            parent=cg_node,
            critical=True,
        )
        res_claim = (
            "This campground accepts reservations at least 6 months in advance "
            "(e.g., booking opens 6 months or more before arrival)."
        )
        await evaluator.verify(
            claim=res_claim,
            node=res_window_node,
            sources=get_sources(item),
            additional_instruction="If the page states 9-12 months in advance, count as supported; 6 months is the minimum threshold.",
        )

        summer_node = evaluator.add_leaf(
            id=f"C{idx+1}_Summer_Operation",
            desc="Open and operational during June, July, and August",
            parent=cg_node,
            critical=True,
        )
        summer_claim = (
            "This campground is open and operational during June, July, and August. "
            "If the season includes these months or it is open year-round, count as supported."
        )
        await evaluator.verify(
            claim=summer_claim,
            node=summer_node,
            sources=get_sources(item),
            additional_instruction="Check seasonal dates or operating calendar; year-round operation qualifies.",
        )

    elif idx == 1:
        # Southeastern: full hookups, lake access with boat launch, showers/restrooms
        full_hook_node = evaluator.add_leaf(
            id=f"C{idx+1}_Full_Hookups",
            desc="Offers full hookup sites (water, electric, and sewer)",
            parent=cg_node,
            critical=True,
        )
        full_hook_claim = "This campground offers full hookup sites that include water, electric, and sewer connections."
        await evaluator.verify(
            claim=full_hook_claim,
            node=full_hook_node,
            sources=get_sources(item),
            additional_instruction="Accept synonyms like 'full hookups' or 'W/E/S'.",
        )

        lake_access_node = evaluator.add_leaf(
            id=f"C{idx+1}_Lake_Access",
            desc="Provides direct access to a lake with boat launch facilities",
            parent=cg_node,
            critical=True,
        )
        lake_claim = (
            "This campground provides direct access to a lake and has a boat launch or boat ramp available."
        )
        await evaluator.verify(
            claim=lake_claim,
            node=lake_access_node,
            sources=get_sources(item),
            additional_instruction="Look for 'boat launch', 'boat ramp', 'marina', or maps showing lake access from the campground.",
        )

        shower_node = evaluator.add_leaf(
            id=f"C{idx+1}_Shower_Facilities",
            desc="Has shower and restroom facilities available to campers",
            parent=cg_node,
            critical=True,
        )
        shower_claim = "This campground has shower facilities and restrooms available for campers."
        await evaluator.verify(
            claim=shower_claim,
            node=shower_node,
            sources=get_sources(item),
            additional_instruction="Accept statements listing amenities such as 'bathhouse', 'showers', 'restrooms', or 'comfort station'.",
        )

    else:
        # Great Lakes: at least 30-amp electric, ADA accessible campsites, online reservation system
        electric_node = evaluator.add_leaf(
            id=f"C{idx+1}_Electric_Hookup",
            desc="Offers at least 30-amp electrical service",
            parent=cg_node,
            critical=True,
        )
        electric_claim = "This campground offers at least 30-amp electrical service at campsites."
        await evaluator.verify(
            claim=electric_claim,
            node=electric_node,
            sources=get_sources(item),
            additional_instruction="If 30/50 amp service is listed, count as supported.",
        )

        ada_node = evaluator.add_leaf(
            id=f"C{idx+1}_ADA_Accessible",
            desc="Has ADA-accessible campsites available",
            parent=cg_node,
            critical=True,
        )
        ada_claim = "This campground has ADA-accessible campsites available."
        await evaluator.verify(
            claim=ada_claim,
            node=ada_node,
            sources=get_sources(item),
            additional_instruction="Look for mentions of ADA-accessible or accessible campsites, or accessibility features explicitly tied to campsites.",
        )

        reserve_node = evaluator.add_leaf(
            id=f"C{idx+1}_Reservation_System",
            desc="Uses an online reservation system (such as Recreation.gov or state park system)",
            parent=cg_node,
            critical=True,
        )
        reserve_claim = (
            "This campground uses an online reservation system such as Recreation.gov, ReserveAmerica, or a state park reservation website."
        )
        await evaluator.verify(
            claim=reserve_claim,
            node=reserve_node,
            sources=get_sources(item),
            additional_instruction="Look for links or references to online booking systems (ReserveAmerica, Recreation.gov, state reservation portals).",
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
    """
    Evaluate an answer for the RV summer 2026 campgrounds task using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    # Note: Root set to non-critical to allow partial credit across campgrounds.
    # The provided JSON says Root is critical, but obj_task_eval enforces that a critical parent
    # must have critical children. We intentionally set root to non-critical to permit partial scoring.
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

    # Extract campgrounds from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundExtraction,
        extraction_name="campgrounds_extraction",
    )

    # Pad or trim to exactly three campgrounds
    campgrounds: List[CampgroundItem] = list(extraction.campgrounds or [])
    if len(campgrounds) > 3:
        campgrounds = campgrounds[:3]
    while len(campgrounds) < 3:
        campgrounds.append(CampgroundItem())

    # Add region definitions as reference info
    evaluator.add_ground_truth({
        "regions": {
            "Mid-Atlantic": sorted(list(MID_ATLANTIC_STATES)),
            "Southeastern": sorted(list(SOUTHEASTERN_STATES)),
            "Great Lakes": sorted(list(GREAT_LAKES_STATES)),
        }
    }, gt_type="region_definitions")

    # Build and verify each campground subtree
    for idx, item in enumerate(campgrounds):
        await build_and_verify_campground(evaluator, root, item, idx)

    return evaluator.get_summary()