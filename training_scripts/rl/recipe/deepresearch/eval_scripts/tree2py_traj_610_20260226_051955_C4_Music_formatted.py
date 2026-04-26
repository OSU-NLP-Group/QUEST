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
TASK_ID = "dua_ro_na_start_venue"
TASK_DESCRIPTION = (
    "Identify the specific venue (provide the venue name) where Dua Lipa's Radical Optimism Tour North American leg began. "
    "The venue must meet all of the following criteria: (1) The first concert at this venue was on September 1, 2025; "
    "(2) Dua Lipa performed two consecutive nights at this venue; (3) The venue's concert capacity is between 19,000 and 24,000 people; "
    "(4) The venue is a major indoor arena in a North American city."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    # Core identification
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state_or_province: Optional[str] = None
    country: Optional[str] = None

    # Dates/occurrence info as claimed in the answer
    start_date: Optional[str] = None  # e.g., "September 1, 2025"
    performance_dates: List[str] = Field(default_factory=list)  # e.g., ["September 1, 2025", "September 2, 2025"]

    # Capacity/type descriptors if mentioned
    capacity_text: Optional[str] = None
    capacity_number: Optional[str] = None  # keep as string to be lenient
    venue_type: Optional[str] = None  # e.g., "indoor arena", "multi-purpose indoor arena"

    # URL sources explicitly cited in the answer
    general_sources: List[str] = Field(default_factory=list)
    tour_association_sources: List[str] = Field(default_factory=list)
    start_date_sources: List[str] = Field(default_factory=list)
    consecutive_nights_sources: List[str] = Field(default_factory=list)
    capacity_sources: List[str] = Field(default_factory=list)
    venue_type_sources: List[str] = Field(default_factory=list)
    location_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    Extract the venue information for the kickoff (beginning) of Dua Lipa's Radical Optimism Tour North American leg, as stated in the answer.

    Return a JSON object with these fields:
    - venue_name: The exact venue name provided in the answer (e.g., "Scotiabank Arena"). If not provided, return null.
    - city: The city where the venue is located (e.g., "Toronto"). If not provided, return null.
    - state_or_province: The state or province if provided (e.g., "Ontario" or "ON"). If not provided, return null.
    - country: The country if provided (e.g., "Canada", "United States", "Mexico"). If not provided, return null.

    - start_date: The first performance date at this venue for Dua Lipa's Radical Optimism Tour North American leg, as claimed in the answer (e.g., "September 1, 2025"). If not provided, return null.
    - performance_dates: A list of all specific dates at which she performed at this venue (as claimed in the answer), in the order the answer presents them. For example: ["September 1, 2025", "September 2, 2025"]. If not provided, return an empty list.

    - capacity_text: Any capacity text stated in the answer (e.g., "concert capacity ~20,000").
    - capacity_number: If the answer provides a single number for capacity, extract it exactly as written (as a string). If unclear or expressed as a range, keep this null.

    - venue_type: Any descriptor the answer gives for the venue type (e.g., "indoor arena", "multi-purpose indoor arena"). If not provided, return null.

    Additionally, extract URL sources explicitly cited in the answer. Only URLs present in the answer text:
    - general_sources: All general/source URLs cited in the answer that support the overall claim about the venue and shows.
    - tour_association_sources: URLs that specifically support that the shows at this venue are part of the Radical Optimism Tour North American leg.
    - start_date_sources: URLs that specifically support that the first show at this venue occurred on September 1, 2025 (or explicitly show that date for Dua Lipa at this venue).
    - consecutive_nights_sources: URLs that specifically support that Dua Lipa performed on two consecutive nights at this venue (ideally showing both dates).
    - capacity_sources: URLs that support the venue's concert/event capacity (prefer official venue pages or Wikipedia).
    - venue_type_sources: URLs that support that the venue is an indoor arena (or equivalent).
    - location_sources: URLs that confirm the city and country of the venue.

    IMPORTANT:
    - Extract only URLs explicitly present in the answer (including markdown links). Do not invent or infer URLs.
    - If the answer provides a single "Sources" section without per-claim attribution, copy those URLs into general_sources. If you can reasonably infer which URL supports which sub-claim from the answer context (e.g., a venue page likely supports capacity and venue type), also include them in the more specific lists in addition to general_sources.
    - If a list (like capacity_sources) has no applicable URLs in the answer, return an empty list (not null).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def pick_sources(extracted: VenueExtraction, preferred_fields: List[str]) -> List[str]:
    """
    Pick URLs for verification using a priority list of field names on VenueExtraction.
    If all preferred lists are empty, falls back to general_sources.
    Returns a de-duplicated list.
    """
    agg: List[str] = []
    for field in preferred_fields:
        lst = getattr(extracted, field, []) or []
        if lst:
            agg.extend(lst)
    if not agg:
        agg.extend(extracted.general_sources or [])
    return _dedup_urls(agg)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_venue(
    evaluator: Evaluator,
    root_node,
    extracted: VenueExtraction,
) -> None:
    """
    Build the verification tree under the critical VenueIdentification node
    and run verifications for each rubric leaf.
    """
    # Parent node per rubric (critical, parallel aggregation)
    venue_node = evaluator.add_parallel(
        id="VenueIdentification",
        desc="Correctly identify the venue where Dua Lipa's Radical Optimism Tour North American leg began",
        parent=root_node,
        critical=True,
    )

    # 1) VenueName (critical leaf)
    vn_desc = "The specific name of the venue is correctly provided"
    if not extracted.venue_name or not extracted.venue_name.strip():
        evaluator.add_custom_node(
            result=False,
            id="VenueName",
            desc=vn_desc,
            parent=venue_node,
            critical=True,
        )
    else:
        vn_node = evaluator.add_leaf(
            id="VenueName",
            desc=vn_desc,
            parent=venue_node,
            critical=True,
        )
        vn_sources = pick_sources(extracted, ["start_date_sources", "tour_association_sources"])
        if vn_sources:
            vn_claim = (
                f"The venue associated with the kickoff of Dua Lipa's Radical Optimism Tour North American leg is named "
                f"'{extracted.venue_name}'."
            )
            await evaluator.verify(
                claim=vn_claim,
                node=vn_node,
                sources=vn_sources,
                additional_instruction=(
                    "Confirm that the provided webpages name the venue exactly or with a clearly equivalent official name. "
                    "Allow minor variations in punctuation or branding (e.g., sponsored names)."
                ),
            )
        else:
            # No sources provided for this claim; fail per source-grounding policy
            evaluator.add_custom_node(
                result=False,
                id="VenueName_no_sources",
                desc="VenueName verification failed due to missing supporting URLs",
                parent=venue_node,
                critical=True,
            )

    # 2) TourAssociation (critical leaf)
    ta_desc = "The venue is part of Dua Lipa's Radical Optimism Tour North American leg"
    ta_sources = pick_sources(extracted, ["tour_association_sources"])
    if not ta_sources or not extracted.venue_name:
        evaluator.add_custom_node(
            result=False,
            id="TourAssociation",
            desc=ta_desc,
            parent=venue_node,
            critical=True,
        )
    else:
        ta_node = evaluator.add_leaf(
            id="TourAssociation",
            desc=ta_desc,
            parent=venue_node,
            critical=True,
        )
        ta_claim = (
            f"The performances at {extracted.venue_name} are part of Dua Lipa's 'Radical Optimism Tour' North American leg."
        )
        await evaluator.verify(
            claim=ta_claim,
            node=ta_node,
            sources=ta_sources,
            additional_instruction=(
                "Look for explicit mention that these shows belong to Dua Lipa's 'Radical Optimism Tour' and specifically "
                "to its North American leg, or clear placement under a North America section of the tour schedule."
            ),
        )

    # 3) StartDate (critical leaf)
    sd_desc = "The first performance at this venue occurred on September 1, 2025"
    sd_sources = pick_sources(extracted, ["start_date_sources"])
    if not sd_sources or not extracted.venue_name:
        evaluator.add_custom_node(
            result=False,
            id="StartDate",
            desc=sd_desc,
            parent=venue_node,
            critical=True,
        )
    else:
        sd_node = evaluator.add_leaf(
            id="StartDate",
            desc=sd_desc,
            parent=venue_node,
            critical=True,
        )
        sd_claim = f"Dua Lipa's first performance at {extracted.venue_name} took place on September 1, 2025."
        await evaluator.verify(
            claim=sd_claim,
            node=sd_node,
            sources=sd_sources,
            additional_instruction=(
                "Verify the event listing or announcement explicitly shows a show on September 1, 2025 at this venue. "
                "Allow for standard date formatting variations (e.g., Sep 1, 2025 or 2025-09-01)."
            ),
        )

    # 4) ConsecutiveNights (critical leaf)
    cn_desc = "Dua Lipa performed on two consecutive nights at this venue"
    cn_sources = pick_sources(extracted, ["consecutive_nights_sources", "start_date_sources"])
    if not cn_sources or not extracted.venue_name:
        evaluator.add_custom_node(
            result=False,
            id="ConsecutiveNights",
            desc=cn_desc,
            parent=venue_node,
            critical=True,
        )
    else:
        cn_node = evaluator.add_leaf(
            id="ConsecutiveNights",
            desc=cn_desc,
            parent=venue_node,
            critical=True,
        )
        # Use concrete dates in the claim to aid verification
        cn_claim = f"Dua Lipa performed on two consecutive nights at {extracted.venue_name}, on September 1 and 2, 2025."
        await evaluator.verify(
            claim=cn_claim,
            node=cn_node,
            sources=cn_sources,
            additional_instruction=(
                "Confirm there were shows on both September 1, 2025 and September 2, 2025 at the same venue, indicating consecutive nights."
            ),
        )

    # 5) CapacityRange (critical leaf)
    cr_desc = "The venue's concert capacity is between 19,000 and 24,000 people"
    cr_sources = pick_sources(extracted, ["capacity_sources", "venue_type_sources"])
    if not cr_sources or not extracted.venue_name:
        evaluator.add_custom_node(
            result=False,
            id="CapacityRange",
            desc=cr_desc,
            parent=venue_node,
            critical=True,
        )
    else:
        cr_node = evaluator.add_leaf(
            id="CapacityRange",
            desc=cr_desc,
            parent=venue_node,
            critical=True,
        )
        cr_claim = (
            f"The concert or event capacity of {extracted.venue_name} lies within the range of 19,000 to 24,000 people."
        )
        await evaluator.verify(
            claim=cr_claim,
            node=cr_node,
            sources=cr_sources,
            additional_instruction=(
                "Prefer 'concert capacity' when available; if only a general or seating capacity is shown but falls within this range, accept it. "
                "Allow phrasing like 'up to 20,000', '~20,000', or ranges that fit inside 19,000–24,000. "
                "Ignore attendance figures for specific events; focus on the venue's stated capacity."
            ),
        )

    # 6) VenueType (critical leaf)
    vt_desc = "The venue is a major indoor arena"
    vt_sources = pick_sources(extracted, ["venue_type_sources", "capacity_sources"])
    if not vt_sources or not extracted.venue_name:
        evaluator.add_custom_node(
            result=False,
            id="VenueType",
            desc=vt_desc,
            parent=venue_node,
            critical=True,
        )
    else:
        vt_node = evaluator.add_leaf(
            id="VenueType",
            desc=vt_desc,
            parent=venue_node,
            critical=True,
        )
        vt_claim = f"{extracted.venue_name} is a major indoor arena."
        await evaluator.verify(
            claim=vt_claim,
            node=vt_node,
            sources=vt_sources,
            additional_instruction=(
                "Treat descriptions like 'indoor arena', 'multi-purpose indoor arena', or being home to major league teams (NBA/NHL) "
                "as sufficient evidence of being a major indoor arena."
            ),
        )

    # 7) GeographicLocation (critical leaf)
    gl_desc = "The venue is located in a major North American city"
    gl_sources = pick_sources(extracted, ["location_sources"])
    if not gl_sources or not extracted.venue_name:
        evaluator.add_custom_node(
            result=False,
            id="GeographicLocation",
            desc=gl_desc,
            parent=venue_node,
            critical=True,
        )
    else:
        gl_node = evaluator.add_leaf(
            id="GeographicLocation",
            desc=gl_desc,
            parent=venue_node,
            critical=True,
        )
        if extracted.city:
            gl_claim = (
                f"The venue {extracted.venue_name} is located in {extracted.city}, which is a major city in North America."
            )
        else:
            gl_claim = (
                f"The venue {extracted.venue_name} is located in a major city in North America."
            )
        await evaluator.verify(
            claim=gl_claim,
            node=gl_node,
            sources=gl_sources,
            additional_instruction=(
                "Use the webpage(s) to confirm the city and that it is in North America (U.S., Canada, or Mexico). "
                "You may use general world knowledge to judge whether the city is 'major' once the city is identified."
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
    Evaluate an answer for the 'Radical Optimism Tour North American leg kickoff venue' task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Only one main subtree; parallel is fine
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_venue(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()