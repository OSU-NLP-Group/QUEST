import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pnw_state_park_rv_campground"
TASK_DESCRIPTION = """
Identify a state park campground in the Pacific Northwest (Oregon or Washington) that provides comprehensive RV camping facilities suitable for year-round use. The campground must meet ALL of the following requirements:

1. Located in Oregon or Washington state
2. Offers full-hookup RV sites (electricity, water, and sewer connections)
3. Provides hot shower facilities
4. Has an RV dump station
5. Accepts advance reservations
6. Has flush toilet facilities
7. Allows pets with a maximum 6-foot leash requirement
8. Accommodates RVs of at least 35 feet in length
9. Offers year-round camping availability
10. Provides electrical hookups at campsites
11. Provides water hookups at campsites
12. Includes a picnic table at each campsite
13. Includes a fire ring or grill at each campsite
14. Provides paved or level sites suitable for RVs

Provide the name of one state park that meets all these criteria, along with supporting evidence from official state park sources.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkSelection(BaseModel):
    park_name: Optional[str] = None
    state: Optional[str] = None  # "Oregon" or "Washington" if stated in the answer
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_park_selection() -> str:
    return """
    Extract the single state park or campground being proposed in the answer and all supporting URLs the answer cites.

    Return a JSON object with:
    - park_name: The name of one specific state park/campground proposed as the match (e.g., "Cape Lookout State Park"). If multiple are mentioned, choose the first one clearly proposed as the answer. If not provided, return null.
    - state: If the answer explicitly states the state ("Oregon" or "Washington"), extract it verbatim; otherwise return null.
    - source_urls: A list of ALL URLs mentioned in the answer as supporting evidence or references for the selected park/campground (e.g., the official state park page, reservation portal, campground brochure PDF, pet policy page, etc.). Include only valid URLs. If none are mentioned, return an empty array.

    Notes:
    - The URLs must be explicitly present in the answer (including plaintext links or markdown links). Do not invent URLs.
    - Deduplicate URLs; keep one instance per unique link.
    - Keep the URLs exactly as they appear, but ensure they include http:// or https://.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        uu = u.strip()
        if uu and uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out


def _base_official_instruction(park_name: Optional[str]) -> str:
    # Generic guidance to enforce official-source-only verification
    park_display = park_name or "the identified park"
    return (
        "Only accept the claim as supported if the evidence comes from an official Oregon State Parks or "
        "Washington State Parks source (i.e., the website clearly identifies itself as the official state parks site "
        "for Oregon or Washington, typically a government-affiliated domain). Disregard third-party travel blogs, "
        "aggregators, or private campground directories. The page you evaluate must pertain to "
        f"{park_display} or to an official statewide policy that applies to all state parks in that state.\n"
        "- Accept reasonable synonyms or abbreviations (e.g., 'full hookups', 'full hook-ups', 'FHU', 'W/E/S').\n"
        "- If the page is not official or does not clearly state the claim, mark as not supported.\n"
    )


def _reservation_instruction() -> str:
    return (
        "For the reservations requirement, an official state parks reservations portal (or an official vendor portal "
        "clearly branded for Oregon or Washington State Parks and linked/endorsed by the official park system) "
        "is acceptable evidence."
    )


def _pet_policy_instruction() -> str:
    return (
        "For the pet policy, an official statewide state parks pet policy page is acceptable so long as it applies to "
        "all parks (including the identified park) and explicitly mentions a maximum leash length of 6 feet (or "
        "equivalent phrasing)."
    )


def _rv_length_instruction() -> str:
    return (
        "For RV length capacity, accept language such as 'maximum site length', 'maximum vehicle length', "
        "'maximum trailer length', or similar. The capacity must be at least 35 feet."
    )


def _paved_level_instruction() -> str:
    return (
        "For paved/level sites, accept evidence indicating paved pads, asphalt/concrete pads, or explicitly level RV sites. "
        "Level gravel pads are acceptable if the page clearly states they are level and suitable for RVs."
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    selection: ParkSelection,
) -> None:
    """
    Build the verification tree according to the rubric and perform all verifications.
    """
    # Root has already been initialized as non-critical in evaluator.initialize()
    # Create a single critical parallel node to hold all checks (mirroring the rubric)
    main = evaluator.add_parallel(
        id="State_Park_RV_Campground_Identification",
        desc="Identify one state park campground in Oregon or Washington that meets all specified RV camping facility requirements and provide official-source evidence.",
        parent=evaluator.root,
        critical=True,
    )

    park_name = (selection.park_name or "").strip()
    urls = _dedup_urls(selection.source_urls)

    # Record custom info for debugging/traceability
    evaluator.add_custom_info(
        {"park_name": park_name, "state_in_answer": selection.state, "urls_provided_count": len(urls), "urls": urls},
        info_type="extraction_summary",
    )

    # 1) Provides_State_Park_Name (existence check)
    evaluator.add_custom_node(
        result=bool(park_name),
        id="Provides_State_Park_Name",
        desc="Provides the name of one specific state park/campground being proposed as the match.",
        parent=main,
        critical=True,
    )

    # 2) Official_Source_Evidence_Provided
    node_official = evaluator.add_leaf(
        id="Official_Source_Evidence_Provided",
        desc="Provides supporting evidence from official state park sources for the proposed campground.",
        parent=main,
        critical=True,
    )
    claim_official = (
        f"This page is an official Oregon or Washington State Parks website page about {park_name} "
        f"(or a directly related official page such as its brochure or reservations entry)."
    )
    await evaluator.verify(
        claim=claim_official,
        node=node_official,
        sources=urls,
        additional_instruction=_base_official_instruction(park_name),
    )

    # 3) Geographic_Location
    node_geo = evaluator.add_leaf(
        id="Geographic_Location",
        desc="The identified campground is a state park located in either Oregon or Washington.",
        parent=main,
        critical=True,
    )
    claim_geo = f"{park_name} is a State Park located in Oregon or Washington."
    await evaluator.verify(
        claim=claim_geo,
        node=node_geo,
        sources=urls,
        additional_instruction=_base_official_instruction(park_name),
    )

    # 4) Full_Hookup_Sites_With_Required_Connections
    node_full_hookup = evaluator.add_leaf(
        id="Full_Hookup_Sites_With_Required_Connections",
        desc="The campground offers full-hookup RV sites including electricity, water, and sewer connections.",
        parent=main,
        critical=True,
    )
    claim_full_hookup = (
        f"{park_name} offers full-hookup RV campsites that include electricity, water, and sewer connections "
        f"(i.e., 'full hookups' or W/E/S)."
    )
    await evaluator.verify(
        claim=claim_full_hookup,
        node=node_full_hookup,
        sources=urls,
        additional_instruction=_base_official_instruction(park_name),
    )

    # 5) Hot_Showers
    node_showers = evaluator.add_leaf(
        id="Hot_Showers",
        desc="The campground provides hot shower facilities.",
        parent=main,
        critical=True,
    )
    claim_showers = f"{park_name} provides hot showers for campers."
    await evaluator.verify(
        claim=claim_showers,
        node=node_showers,
        sources=urls,
        additional_instruction=_base_official_instruction(park_name),
    )

    # 6) Dump_Station
    node_dump = evaluator.add_leaf(
        id="Dump_Station",
        desc="The campground has an RV dump station available.",
        parent=main,
        critical=True,
    )
    claim_dump = f"{park_name} has an RV dump station available."
    await evaluator.verify(
        claim=claim_dump,
        node=node_dump,
        sources=urls,
        additional_instruction=_base_official_instruction(park_name),
    )

    # 7) Reservation_System
    node_reservation = evaluator.add_leaf(
        id="Reservation_System",
        desc="The campground accepts advance reservations through an official reservation system.",
        parent=main,
        critical=True,
    )
    claim_reservation = f"{park_name} accepts advance reservations via an official state parks reservation system."
    await evaluator.verify(
        claim=claim_reservation,
        node=node_reservation,
        sources=urls,
        additional_instruction=_base_official_instruction(park_name) + "\n" + _reservation_instruction(),
    )

    # 8) Flush_Toilets_Not_Vault
    node_flush = evaluator.add_leaf(
        id="Flush_Toilets_Not_Vault",
        desc="The campground provides flush toilet facilities (not vault-only).",
        parent=main,
        critical=True,
    )
    claim_flush = f"{park_name} provides flush toilet facilities (i.e., not vault toilets only)."
    await evaluator.verify(
        claim=claim_flush,
        node=node_flush,
        sources=urls,
        additional_instruction=_base_official_instruction(park_name),
    )

    # 9) Pet_Policy_6ft_Leash
    node_pets = evaluator.add_leaf(
        id="Pet_Policy_6ft_Leash",
        desc="The campground allows pets with a maximum 6-foot leash requirement.",
        parent=main,
        critical=True,
    )
    claim_pets = f"Pets are allowed at {park_name} and must be on a leash no longer than 6 feet."
    await evaluator.verify(
        claim=claim_pets,
        node=node_pets,
        sources=urls,
        additional_instruction=_base_official_instruction(park_name) + "\n" + _pet_policy_instruction(),
    )

    # 10) RV_Length_Capacity
    node_length = evaluator.add_leaf(
        id="RV_Length_Capacity",
        desc="The campground accommodates RVs of at least 35 feet in length.",
        parent=main,
        critical=True,
    )
    claim_length = f"{park_name} accommodates RVs with a length of at least 35 feet."
    await evaluator.verify(
        claim=claim_length,
        node=node_length,
        sources=urls,
        additional_instruction=_base_official_instruction(park_name) + "\n" + _rv_length_instruction(),
    )

    # 11) Year_Round_Camping
    node_year_round = evaluator.add_leaf(
        id="Year_Round_Camping",
        desc="The campground offers year-round camping availability.",
        parent=main,
        critical=True,
    )
    claim_year_round = f"{park_name} offers year-round camping (open all year)."
    await evaluator.verify(
        claim=claim_year_round,
        node=node_year_round,
        sources=urls,
        additional_instruction=_base_official_instruction(park_name),
    )

    # 12) Picnic_Tables
    node_picnic = evaluator.add_leaf(
        id="Picnic_Tables",
        desc="Each campsite includes a picnic table.",
        parent=main,
        critical=True,
    )
    claim_picnic = f"Each campsite at {park_name} includes a picnic table."
    await evaluator.verify(
        claim=claim_picnic,
        node=node_picnic,
        sources=urls,
        additional_instruction=_base_official_instruction(park_name),
    )

    # 13) Fire_Facilities
    node_fire = evaluator.add_leaf(
        id="Fire_Facilities",
        desc="Each campsite includes a fire ring or grill.",
        parent=main,
        critical=True,
    )
    claim_fire = f"Each campsite at {park_name} includes a fire ring or grill."
    await evaluator.verify(
        claim=claim_fire,
        node=node_fire,
        sources=urls,
        additional_instruction=_base_official_instruction(park_name),
    )

    # 14) Level_Or_Paved_Sites
    node_paved = evaluator.add_leaf(
        id="Level_Or_Paved_Sites",
        desc="The campground provides paved or level sites suitable for RVs.",
        parent=main,
        critical=True,
    )
    claim_paved = f"{park_name} provides paved or level RV campsites suitable for RVs."
    await evaluator.verify(
        claim=claim_paved,
        node=node_paved,
        sources=urls,
        additional_instruction=_base_official_instruction(park_name) + "\n" + _paved_level_instruction(),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Pacific Northwest state park RV campground task.
    """
    # Initialize evaluator with a parallel root
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extraction
    selection: ParkSelection = await evaluator.extract(
        prompt=prompt_extract_park_selection(),
        template_class=ParkSelection,
        extraction_name="park_selection",
    )

    # Add ground truth requirements snapshot (for transparency)
    evaluator.add_ground_truth(
        {
            "region_requirement": "Oregon or Washington (USA)",
            "must_satisfy_all": [
                "Full-hookup RV sites (electricity, water, sewer)",
                "Hot showers",
                "RV dump station",
                "Advance reservations accepted",
                "Flush toilet facilities",
                "Pets allowed with 6-foot leash requirement",
                "Accommodates RVs of at least 35 feet",
                "Year-round camping availability",
                "Picnic table at each campsite",
                "Fire ring or grill at each campsite",
                "Paved or level sites suitable for RVs",
            ],
            "evidence_requirement": "Official Oregon or Washington State Parks sources",
        },
        gt_type="requirements",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, selection)

    # Return evaluation summary
    return evaluator.get_summary()