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
TASK_ID = "epic_universe_hotels_guide_2025"
TASK_DESCRIPTION = (
    "Universal Epic Universe, the new theme park at Universal Orlando Resort, opened in 2025 alongside three brand-new on-site hotels. "
    "A travel planning company needs to create a detailed information guide about these hotels for their clients. Provide comprehensive information "
    "about Universal Epic Universe and its associated hotels, including: the official opening date of Universal Epic Universe; the total number of "
    "hotels built specifically for Epic Universe; the complete names of all three hotels; for Universal Helios Grand Hotel: whether it has a dedicated "
    "entrance to the park, whether it features a rooftop bar, the size of its resort-style pool in square feet, and its management company; for Universal "
    "Stella Nova Resort: its location relative to Epic Universe, whether it has a walking path to the park, and its management company; for Universal "
    "Terra Luna Resort: its resort classification category, its walking distance characteristic compared to Stella Nova, and its management company; and "
    "whether all three hotels provide Early Park Admission benefits to Universal theme parks."
)

# Ground-truth expectations (used to phrase verification claims)
GROUND_TRUTH = {
    "opening_date": "May 22, 2025",
    "hotels_count": 3,
    "hotel_names": [
        "Universal Helios Grand Hotel",
        "Universal Stella Nova Resort",
        "Universal Terra Luna Resort",
    ],
    "helios_pool_size_sqft": "8,660"
}

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class HotelsGuideExtraction(BaseModel):
    # Epic Universe global facts
    epic_opening_date: Optional[str] = None
    opening_date_url_sources: List[str] = Field(default_factory=list)

    epic_hotel_count: Optional[str] = None  # keep as string to be robust
    hotel_count_url_sources: List[str] = Field(default_factory=list)

    hotel_names: List[str] = Field(default_factory=list)
    hotel_names_url_sources: List[str] = Field(default_factory=list)

    # Universal Helios Grand Hotel
    helios_dedicated_entrance: Optional[str] = None  # "yes"/"no"/text phrasing
    helios_rooftop_bar: Optional[str] = None         # "yes"/"no"/text
    helios_pool_size_sqft: Optional[str] = None      # e.g., "8,660"
    helios_management_company: Optional[str] = None
    helios_url_sources: List[str] = Field(default_factory=list)

    # Universal Stella Nova Resort
    stella_location_relative_to_epic_universe: Optional[str] = None  # e.g., "across the street"
    stella_has_walking_path: Optional[str] = None
    stella_management_company: Optional[str] = None
    stella_url_sources: List[str] = Field(default_factory=list)

    # Universal Terra Luna Resort
    terra_classification_category: Optional[str] = None  # e.g., "value resort"
    terra_walking_distance_compared_to_stella: Optional[str] = None  # e.g., "farther than Stella Nova"
    terra_management_company: Optional[str] = None
    terra_url_sources: List[str] = Field(default_factory=list)

    # Early Park Admission benefits
    early_park_admission_all_three_hotels: Optional[str] = None  # "yes"/"no"/text
    early_park_admission_url_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels_guide() -> str:
    return """
    Extract structured information from the provided answer text about Universal Epic Universe and its three associated hotels.

    Return a JSON object containing the following fields exactly:

    1) epic_opening_date: The official opening date for Universal Epic Universe as stated in the answer (string, e.g., "May 22, 2025"). If not stated, return null.
    2) opening_date_url_sources: Array of URLs explicitly cited in the answer that support the opening date (exclude non-URL text). If none, return [].

    3) epic_hotel_count: The total number of hotels built specifically for Epic Universe as stated in the answer (string, e.g., "3"). If not stated, return null.
    4) hotel_count_url_sources: Array of URLs explicitly cited in the answer that support this count. If none, return [].

    5) hotel_names: Array of the full names of all Epic Universe hotels as stated in the answer; include up to 3 names, e.g.,
       ["Universal Helios Grand Hotel", "Universal Stella Nova Resort", "Universal Terra Luna Resort"].
       If not stated, return [].
    6) hotel_names_url_sources: Array of URLs explicitly cited in the answer that support the hotel names. If none, return [].

    7) helios_dedicated_entrance: Whether Universal Helios Grand Hotel has a dedicated/direct entrance to Epic Universe as stated in the answer.
       Prefer "yes"/"no"; if phrased text (e.g., "private entrance"), return that text. If not stated, return null.
    8) helios_rooftop_bar: Whether Universal Helios Grand Hotel features a rooftop bar as stated; prefer "yes"/"no" or text. If not stated, return null.
    9) helios_pool_size_sqft: The resort-style pool size as a string (e.g., "8,660"). If not stated, return null.
    10) helios_management_company: The management company name as stated (e.g., "Loews Hotels & Co."). If not stated, return null.
    11) helios_url_sources: Array of URLs explicitly cited for Helios details. If none, return [].

    12) stella_location_relative_to_epic_universe: The location phrasing relative to Epic Universe (e.g., "across the street"). If not stated, return null.
    13) stella_has_walking_path: Whether there is a dedicated walking path to the park; prefer "yes"/"no" or text. If not stated, return null.
    14) stella_management_company: The management company name as stated. If not stated, return null.
    15) stella_url_sources: Array of URLs explicitly cited for Stella Nova details. If none, return [].

    16) terra_classification_category: The resort classification category (e.g., "value resort"). If not stated, return null.
    17) terra_walking_distance_compared_to_stella: A phrasing of Terra Luna's walking distance characteristic compared to Stella Nova (e.g., "farther than Stella Nova", "more than one mile"). If not stated, return null.
    18) terra_management_company: The management company name as stated. If not stated, return null.
    19) terra_url_sources: Array of URLs explicitly cited for Terra Luna details. If none, return [].

    20) early_park_admission_all_three_hotels: Whether all three hotels provide Early Park Admission; prefer "yes"/"no" or text. If not stated, return null.
    21) early_park_admission_url_sources: Array of URLs explicitly cited that support Early Park Admission benefits. If none, return [].

    IMPORTANT:
    - Extract only what is explicitly present in the answer text; do not invent information.
    - For URL arrays, extract only valid URLs explicitly mentioned in the answer (including markdown links).
    - If a field is missing, return null (for strings) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(src: Optional[List[str]]) -> List[str]:
    if not src:
        return []
    # filter obvious empties or malformed entries
    return [s for s in src if isinstance(s, str) and len(s.strip()) > 0]


async def _add_and_verify_leaf(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str]],
    critical: bool,
    additional_instruction: str = "None",
) -> bool:
    """
    Add a leaf node and verify the claim. If sources are missing, mark as failed to enforce source-grounding.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    srcs = _safe_sources(sources)

    # Enforce source-grounding: fail if no sources provided for factual claims
    if len(srcs) == 0:
        leaf.score = 0.0
        leaf.status = "failed"
        evaluator.add_custom_info(
            info={"node_id": node_id, "reason": "no_sources_provided"},
            info_type="missing_sources",
            info_name=f"missing_sources_{node_id}",
        )
        return False

    return await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extraction: HotelsGuideExtraction) -> None:
    """
    Build the verification tree exactly following the rubric and run verifications.
    """
    # Create the rubric root under evaluator.root
    rubric_root = evaluator.add_parallel(
        id="Epic_Universe_Hotels_Information",
        desc="Comprehensive information about Universal Epic Universe hotels including opening details, hotel names, locations, amenities, and management",
        parent=evaluator.root,
        critical=False,
    )

    # Claims and their sources
    # Opening Date (Critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Opening_Date",
        desc="Epic Universe opening date is May 22, 2025",
        claim=f"The official opening date of Universal Epic Universe is {GROUND_TRUTH['opening_date']}.",
        sources=extraction.opening_date_url_sources,
        critical=True,
        additional_instruction="Verify the date explicitly on the provided source(s). Allow minor formatting differences (e.g., 'May 22 2025').",
    )

    # Number of Hotels (Critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Number_of_Hotels",
        desc="Exactly three hotels are associated with Epic Universe",
        claim="Exactly three hotels were built specifically for Universal Epic Universe.",
        sources=extraction.hotel_count_url_sources if extraction.hotel_count_url_sources else extraction.hotel_names_url_sources,
        critical=True,
        additional_instruction="Confirm that Helios Grand Hotel, Stella Nova Resort, and Terra Luna Resort are the three Epic Universe hotels.",
    )

    # Hotel Names (Critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Hotel_Names",
        desc="The three hotel names are Universal Helios Grand Hotel, Universal Stella Nova Resort, and Universal Terra Luna Resort",
        claim="The three hotels are Universal Helios Grand Hotel, Universal Stella Nova Resort, and Universal Terra Luna Resort.",
        sources=extraction.hotel_names_url_sources,
        critical=True,
        additional_instruction="Verify that all three names appear in the source(s). Allow minor punctuation or capitalization variations.",
    )

    # Helios: Dedicated Entrance (Non-critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Helios_Dedicated_Entrance",
        desc="Universal Helios Grand Hotel has a dedicated or direct entrance to Epic Universe",
        claim="Universal Helios Grand Hotel has a dedicated/direct private entrance to Epic Universe.",
        sources=extraction.helios_url_sources,
        critical=False,
        additional_instruction="Look for language such as 'private entrance', 'dedicated entrance', or 'direct access' between Helios and Epic Universe.",
    )

    # Helios: Rooftop Bar (Non-critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Helios_Rooftop_Bar",
        desc="Universal Helios Grand Hotel features a rooftop bar",
        claim="Universal Helios Grand Hotel features a rooftop bar.",
        sources=extraction.helios_url_sources,
        critical=False,
        additional_instruction="Confirm the presence of a rooftop bar; the name may appear (e.g., 'Solis').",
    )

    # Helios: Pool Size (Non-critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Helios_Pool_Size",
        desc="Universal Helios Grand Hotel has a resort-style pool measuring 8,660 square feet",
        claim=f"Universal Helios Grand Hotel has a resort-style pool of about {GROUND_TRUTH['helios_pool_size_sqft']} square feet.",
        sources=extraction.helios_url_sources,
        critical=False,
        additional_instruction="Allow formatting variants like '8660 sq ft', '8,660-square-foot'. The numeric value should match approximately.",
    )

    # Helios: Management (Non-critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Helios_Management",
        desc="Universal Helios Grand Hotel is managed by Loews Hotels & Co.",
        claim="Universal Helios Grand Hotel is managed by Loews Hotels & Co.",
        sources=extraction.helios_url_sources,
        critical=False,
        additional_instruction="Verify the management/operator statement. 'Co-owned and operated by Loews Hotels & Co.' or similar is acceptable.",
    )

    # Stella Nova: Location (Non-critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Stella_Nova_Location",
        desc="Universal Stella Nova Resort is located across the street from Epic Universe",
        claim="Universal Stella Nova Resort is located across the street from Universal Epic Universe.",
        sources=extraction.stella_url_sources,
        critical=False,
        additional_instruction="Look for phrasing like 'across the street' or equivalent proximity wording explicitly referencing Epic Universe.",
    )

    # Stella Nova: Walking Path (Non-critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Stella_Nova_Walking_Path",
        desc="Universal Stella Nova Resort has a dedicated walking path to Epic Universe",
        claim="Universal Stella Nova Resort has a dedicated walking path to Universal Epic Universe.",
        sources=extraction.stella_url_sources,
        critical=False,
        additional_instruction="Confirm existence of a dedicated pedestrian path or walkway linking Stella Nova to Epic Universe.",
    )

    # Stella Nova: Management (Non-critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Stella_Nova_Management",
        desc="Universal Stella Nova Resort is co-owned and operated by Loews Hotels & Co.",
        claim="Universal Stella Nova Resort is co-owned and operated by Loews Hotels & Co.",
        sources=extraction.stella_url_sources,
        critical=False,
        additional_instruction="Confirm the 'co-owned and operated by Loews Hotels & Co.' statement or equivalent wording.",
    )

    # Terra Luna: Classification (Non-critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Terra_Luna_Classification",
        desc="Universal Terra Luna Resort is classified as a value resort",
        claim="Universal Terra Luna Resort is classified as a value resort.",
        sources=extraction.terra_url_sources,
        critical=False,
        additional_instruction="Verify the resort category wording; 'value' classification is what we seek.",
    )

    # Terra Luna: Distance (Non-critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Terra_Luna_Distance",
        desc="Universal Terra Luna Resort is more than one mile from Epic Universe via walking path (farther than Stella Nova)",
        claim="Universal Terra Luna Resort is farther than Stella Nova via the walking path and is more than one mile from Epic Universe.",
        sources=extraction.terra_url_sources,
        critical=False,
        additional_instruction="Look for explicit distance phrasing indicating Terra Luna's walking path is >1 mile and farther than Stella Nova.",
    )

    # Terra Luna: Management (Non-critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Terra_Luna_Management",
        desc="Universal Terra Luna Resort is co-owned and operated by Loews Hotels & Co.",
        claim="Universal Terra Luna Resort is co-owned and operated by Loews Hotels & Co.",
        sources=extraction.terra_url_sources,
        critical=False,
        additional_instruction="Confirm the 'co-owned and operated by Loews Hotels & Co.' statement or equivalent wording.",
    )

    # Early Park Admission (Non-critical)
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="Early_Park_Admission",
        desc="All three Epic Universe hotels provide Early Park Admission benefits to Universal theme parks",
        claim="Universal Helios Grand Hotel, Universal Stella Nova Resort, and Universal Terra Luna Resort all provide Early Park Admission to Universal theme parks.",
        sources=extraction.early_park_admission_url_sources,
        critical=False,
        additional_instruction="Confirm Early Park Admission benefit applies to all three hotels mentioned.",
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
    Evaluate an answer for the Universal Epic Universe hotels information task.
    """
    # Initialize evaluator (root is non-critical parallel by default; desc uses TASK_DESCRIPTION)
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
    extraction: HotelsGuideExtraction = await evaluator.extract(
        prompt=prompt_extract_hotels_guide(),
        template_class=HotelsGuideExtraction,
        extraction_name="epic_universe_hotels_extraction",
    )

    # Add ground-truth info for reference (not used for scoring)
    evaluator.add_ground_truth(
        {
            "expected_opening_date": GROUND_TRUTH["opening_date"],
            "expected_hotels_count": GROUND_TRUTH["hotels_count"],
            "expected_hotel_names": GROUND_TRUTH["hotel_names"],
            "expected_helios_pool_size_sqft": GROUND_TRUTH["helios_pool_size_sqft"],
        },
        gt_type="expected_facts",
    )

    # Build and run verification tree
    await build_and_verify_tree(evaluator, extraction)

    # Return summary
    return evaluator.get_summary()