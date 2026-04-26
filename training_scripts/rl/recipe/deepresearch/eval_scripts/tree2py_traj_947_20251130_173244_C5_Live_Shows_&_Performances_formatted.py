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
TASK_ID = "dua_lipa_nov2025_venues"
TASK_DESCRIPTION = (
    "For Dua Lipa's Radical Optimism Tour in November 2025, identify the concert venue in Buenos Aires, "
    "Argentina and the concert venue in Santiago, Chile. For each venue, provide: (1) the stadium name, "
    "(2) the seating capacity, and (3) the specific concert date(s) in November 2025."
)

# Expected details per rubric
EXPECTED_BA_STADIUM = "Estadio River Plate"
EXPECTED_BA_CAPACITY = "85,018"
EXPECTED_BA_DATES = ["November 7, 2025", "November 8, 2025"]
EXPECTED_SCL_STADIUM = "Estadio Nacional"
EXPECTED_SCL_CAPACITY = "48,665"
EXPECTED_SCL_DATES = ["November 11, 2025", "November 12, 2025"]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    stadium_name: Optional[str] = None
    seating_capacity: Optional[str] = None
    concert_dates: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class TourVenuesExtraction(BaseModel):
    buenos_aires: Optional[VenueInfo] = None
    santiago: Optional[VenueInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract the information the answer provides for Dua Lipa's Radical Optimism Tour in November 2025 for the Buenos Aires (Argentina) and Santiago (Chile) stops.

    For each city, return a JSON object with the following fields exactly as stated in the answer:

    - stadium_name: The stadium/venue name mentioned for the city.
    - seating_capacity: The seating capacity number for that stadium/venue (keep any commas or separators; do not normalize—extract as written).
    - concert_dates: An array of date strings for the specific November 2025 concert date(s) (e.g., ["November 7, 2025", "November 8, 2025"]). If the answer provides a range (e.g., "November 7–8, 2025"), split into individual dates if the exact dates are inferable; otherwise include the range string as a single element.
    - sources: An array of URL strings that the answer cites to substantiate the stadium name, seating capacity, or concert date(s) for that city. Only include URLs explicitly present in the answer (plain links or markdown links). If none are provided, return an empty array.

    Use the top-level keys:
    - buenos_aires
    - santiago

    If any city is missing in the answer, set that city's object to null. If a field is not mentioned, set it to null or an empty array appropriately.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def format_dates_range(dates: List[str]) -> str:
    """
    Format a concise date range string for display in claims:
    - If two dates in the same month/year: "Month D1–D2, YYYY"
    - If one date: "Month D, YYYY"
    - Else: join with ", " for readability.
    This is only for claim phrasing; verification should accept reasonable variants.
    """
    if not dates:
        return ""
    if len(dates) == 1:
        return dates[0]
    if len(dates) == 2:
        # Attempt to parse month/year sameness by simple split; fallback to join
        try:
            # naive split "Month D, YYYY"
            m1, rest1 = dates[0].split(" ", 1)
            m2, rest2 = dates[1].split(" ", 1)
            if m1 == m2:
                # Extract day and year
                d1 = rest1.split(",")[0].strip()
                year = rest1.split(",")[1].strip() if "," in rest1 else ""
                d2 = rest2.split(",")[0].strip()
                return f"{m1} {d1}–{d2}, {year}".strip().strip(", ")
        except Exception:
            pass
        return ", ".join(dates)
    return ", ".join(dates)


# --------------------------------------------------------------------------- #
# Verification subroutine                                                     #
# --------------------------------------------------------------------------- #
async def verify_city_stop(
    evaluator: Evaluator,
    parent_node,
    *,
    city_node_id: str,
    city_node_desc: str,
    stadium_leaf_id: str,
    stadium_leaf_desc: str,
    capacity_leaf_id: str,
    capacity_leaf_desc: str,
    dates_leaf_id: str,
    dates_leaf_desc: str,
    expected_stadium: str,
    expected_capacity: str,
    expected_dates: List[str],
    venue: Optional[VenueInfo],
    city_label_for_claims: str,
) -> None:
    """
    Build and verify the subtree for one city (Buenos Aires or Santiago).
    All children are critical to comply with rubric's critical requirement.
    We verify each claim against the provided sources (URLs) from the answer.
    """
    # Add the city node (critical parallel)
    city_node = evaluator.add_parallel(
        id=city_node_id,
        desc=city_node_desc,
        parent=parent_node,
        critical=True,
    )

    sources = venue.sources if (venue and venue.sources) else []

    # Stadium Name leaf (critical)
    stadium_node = evaluator.add_leaf(
        id=stadium_leaf_id,
        desc=stadium_leaf_desc,
        parent=city_node,
        critical=True,
    )
    stadium_claim = (
        f"For Dua Lipa's Radical Optimism Tour in November 2025, the {city_label_for_claims} concert venue is {expected_stadium}."
    )
    # Additional instruction: allow synonyms (common alternate official names)
    synonyms_instruction = (
        "Treat well-known official aliases as equivalent: "
        "• Buenos Aires: 'El Monumental' or 'Estadio Monumental Antonio Vespucio Liberti' are equivalent to 'Estadio River Plate'. "
        "• Santiago: 'Estadio Nacional Julio Martínez Prádanos' is equivalent to 'Estadio Nacional'. "
        "Verify that the cited sources explicitly support the venue for Dua Lipa's Radical Optimism Tour in November 2025."
    )
    await evaluator.verify(
        claim=stadium_claim,
        node=stadium_node,
        sources=sources,
        additional_instruction=synonyms_instruction,
    )

    # Seating Capacity leaf (critical)
    capacity_node = evaluator.add_leaf(
        id=capacity_leaf_id,
        desc=capacity_leaf_desc,
        parent=city_node,
        critical=True,
    )
    capacity_claim = f"The seating capacity of {expected_stadium} is {expected_capacity}."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=sources,
        additional_instruction=(
            "Confirm the official seating capacity reported by credible sources for the venue. "
            "Minor formatting variations and thousands separators are acceptable, but the numeric value should match {expected_capacity}. "
            "If multiple capacities are listed, prefer the standard/all-seater capacity; do not use temporary event capacities."
        ),
    )

    # Concert Dates leaf (critical)
    dates_node = evaluator.add_leaf(
        id=dates_leaf_id,
        desc=dates_leaf_desc,
        parent=city_node,
        critical=True,
    )
    expected_dates_range = format_dates_range(expected_dates)
    dates_claim = (
        f"The {city_label_for_claims} concert date(s) for Dua Lipa's Radical Optimism Tour are {expected_dates_range} (in November 2025)."
    )
    await evaluator.verify(
        claim=dates_claim,
        node=dates_node,
        sources=sources,
        additional_instruction=(
            "Accept reasonable formatting/locale variants (e.g., '7–8 November 2025', '7 y 8 de noviembre de 2025'). "
            "The source(s) must explicitly state these specific November 2025 dates for Dua Lipa in the specified city."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the answer for Dua Lipa's Radical Optimism Tour venues and dates in November 2025.
    """
    # Initialize evaluator with a non-critical root; we'll add a critical aggregator under it.
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

    # Add the main critical aggregator node as per rubric
    main_node = evaluator.add_parallel(
        id="Dua_Lipa_Radical_Optimism_Tour_Nov_2025_Venues",
        desc=(
            "Identify the Radical Optimism Tour venues in Buenos Aires (Argentina) and Santiago (Chile) in November 2025 "
            "and provide stadium name, seating capacity, and specific concert date(s) for each, supported by reference URLs."
        ),
        parent=root,
        critical=True,
    )

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=TourVenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth(
        {
            "expected_buenos_aires": {
                "stadium_name": EXPECTED_BA_STADIUM,
                "seating_capacity": EXPECTED_BA_CAPACITY,
                "concert_dates_range": format_dates_range(EXPECTED_BA_DATES),
                "concert_dates_list": EXPECTED_BA_DATES,
            },
            "expected_santiago": {
                "stadium_name": EXPECTED_SCL_STADIUM,
                "seating_capacity": EXPECTED_SCL_CAPACITY,
                "concert_dates_range": format_dates_range(EXPECTED_SCL_DATES),
                "concert_dates_list": EXPECTED_SCL_DATES,
            },
        }
    )

    # Build and verify Buenos Aires subtree
    await verify_city_stop(
        evaluator=evaluator,
        parent_node=main_node,
        city_node_id="Buenos_Aires_Stop",
        city_node_desc="Buenos Aires, Argentina stop details (per constraints).",
        stadium_leaf_id="BA_Stadium_Name",
        stadium_leaf_desc="Gives the Buenos Aires stadium name as Estadio River Plate.",
        capacity_leaf_id="BA_Seating_Capacity",
        capacity_leaf_desc="Gives the Buenos Aires stadium seating capacity as 85,018.",
        dates_leaf_id="BA_Concert_Dates",
        dates_leaf_desc="Gives the Buenos Aires concert date(s) as November 7–8, 2025.",
        expected_stadium=EXPECTED_BA_STADIUM,
        expected_capacity=EXPECTED_BA_CAPACITY,
        expected_dates=EXPECTED_BA_DATES,
        venue=extraction.buenos_aires or VenueInfo(),
        city_label_for_claims="Buenos Aires, Argentina",
    )

    # Build and verify Santiago subtree
    await verify_city_stop(
        evaluator=evaluator,
        parent_node=main_node,
        city_node_id="Santiago_Stop",
        city_node_desc="Santiago, Chile stop details (per constraints).",
        stadium_leaf_id="Santiago_Stadium_Name",
        stadium_leaf_desc="Gives the Santiago stadium name as Estadio Nacional.",
        capacity_leaf_id="Santiago_Seating_Capacity",
        capacity_leaf_desc="Gives the Santiago stadium seating capacity as 48,665.",
        dates_leaf_id="Santiago_Concert_Dates",
        dates_leaf_desc="Gives the Santiago concert date(s) as November 11–12, 2025.",
        expected_stadium=EXPECTED_SCL_STADIUM,
        expected_capacity=EXPECTED_SCL_CAPACITY,
        expected_dates=EXPECTED_SCL_DATES,
        venue=extraction.santiago or VenueInfo(),
        city_label_for_claims="Santiago, Chile",
    )

    # Reference URLs node (critical) – ensure both stops include at least one URL
    ba_sources = (extraction.buenos_aires.sources if extraction.buenos_aires else []) or []
    scl_sources = (extraction.santiago.sources if extraction.santiago else []) or []
    ref_urls_ok = bool(ba_sources) and bool(scl_sources)

    evaluator.add_custom_node(
        result=ref_urls_ok,
        id="Reference_URLs",
        desc=(
            "Includes reference URL(s) that substantiate the stadium name, seating capacity, and concert date(s) "
            "for both the Buenos Aires and Santiago stops."
        ),
        parent=main_node,
        critical=True,
    )

    # Record some useful custom info
    evaluator.add_custom_info(
        {
            "buenos_aires_sources_count": len(ba_sources),
            "santiago_sources_count": len(scl_sources),
            "buenos_aires_sources": ba_sources,
            "santiago_sources": scl_sources,
        },
        info_type="url_stats",
        info_name="reference_url_statistics",
    )

    # Return the evaluation summary (score, tree, and recorded info)
    return evaluator.get_summary()