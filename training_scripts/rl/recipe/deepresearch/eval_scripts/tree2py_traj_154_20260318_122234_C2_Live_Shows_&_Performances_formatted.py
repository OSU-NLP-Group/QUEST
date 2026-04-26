import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "ft_lauderdale_beach_festival_2026_headliner_award"
TASK_DESCRIPTION = (
    "A three-day music festival takes place on Fort Lauderdale Beach in Florida from April 10-12, 2026. "
    "The festival features different headliners each day. On the final day of the festival (Sunday, April 12, 2026), "
    "one of the headliners is scheduled to receive a special ocean conservation award during their performance. "
    "Who is the artist headlining on Sunday, April 12, 2026, at this Fort Lauderdale beach festival, and what is the name "
    "of the specific conservation award they are receiving?"
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class FestivalExtraction(BaseModel):
    """Structured festival details extracted from the answer."""
    festival_name: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    date_range_text: Optional[str] = None
    start_date: Optional[str] = None    # Prefer strings for robustness, e.g., "2026-04-10"
    end_date: Optional[str] = None      # e.g., "2026-04-12"
    venue_name_or_type: Optional[str] = None  # e.g., "Fort Lauderdale Beach" or "Fort Lauderdale Beach Park"
    venue_address: Optional[str] = None       # e.g., "1100 Seabreeze Blvd, Fort Lauderdale, FL 33316"
    reference_urls: List[str] = Field(default_factory=list)


class PerformerExtraction(BaseModel):
    """Structured performer and award details extracted from the answer."""
    performer_name: Optional[str] = None               # The Sunday's headliner artist name
    headline_day: Optional[str] = None                 # e.g., "Sunday"
    headline_date_text: Optional[str] = None           # e.g., "April 12, 2026"
    award_name: Optional[str] = None                   # Full, specific name of the conservation award
    award_timing_text: Optional[str] = None            # e.g., "during their performance" or similar
    previous_headline_years: List[str] = Field(default_factory=list)  # years or editions previously headlined
    reference_urls: List[str] = Field(default_factory=list)           # URLs supporting performer headliner, history, and award


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_festival_details() -> str:
    return """
    Extract the festival details explicitly mentioned in the answer. Return a JSON object with:
    - festival_name: The official festival name if provided (string or null).
    - location_city: City where the festival occurs (string or null).
    - location_state: State where the festival occurs (string or null).
    - date_range_text: The exact date range text as written in the answer (e.g., "April 10-12, 2026"; string or null).
    - start_date: The festival start date if explicitly provided or clearly inferable from the same text (string "YYYY-MM-DD" if possible, else keep the original format as a string; or null).
    - end_date: The festival end date similarly (string or null).
    - venue_name_or_type: The venue description (e.g., "Fort Lauderdale Beach", "Fort Lauderdale Beach Park"; string or null).
    - venue_address: Full venue street address if present (e.g., "1100 Seabreeze Blvd, Fort Lauderdale, FL 33316"; string or null).
    - reference_urls: An array of all URLs in the answer that support the festival location, dates, venue/venue address; return [] if none.
    
    IMPORTANT:
    - Extract only what is explicitly present in the answer. Do not invent or guess.
    - For URLs, include only valid, explicit URLs present in the answer (plain links or markdown links).
    """


def prompt_extract_performer_details() -> str:
    return """
    Extract the performer and award details for the Sunday (final day) headliner as presented in the answer. Return a JSON object with:
    - performer_name: The artist headlining on Sunday, April 12, 2026 (string or null).
    - headline_day: The day label if present (e.g., "Sunday"; string or null).
    - headline_date_text: The date text for the headline day if present (e.g., "April 12, 2026"; string or null).
    - award_name: The specific, full name of the ocean conservation award the performer is receiving during their performance (string or null).
    - award_timing_text: Wording indicating the timing/context (e.g., "during their performance"; string or null).
    - previous_headline_years: A list of years or editions this same artist previously headlined this festival if mentioned (list of strings; [] if none).
    - reference_urls: An array of all URLs that the answer cites to support the performer's Sunday headline slot, their festival headliner history, and the award details; [] if none.
    
    IMPORTANT:
    - Extract only information explicitly present in the answer; do not infer.
    - Include all relevant URLs explicitly appearing in the answer (valid links only).
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _combine_sources(*url_lists: Optional[List[str]]) -> List[str]:
    """Combine and deduplicate multiple URL lists while preserving order."""
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for url in lst:
            if isinstance(url, str) and url.strip() and url not in seen:
                combined.append(url)
                seen.add(url)
    return combined


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_festival_identification_subtree(
    evaluator: Evaluator,
    parent_node,
    fest: FestivalExtraction,
) -> None:
    """
    Build and verify the 'Festival_Identification' subtree.
    Note: The rubric marked the inner 'Reference_URL' as non-critical, but since its parent is critical,
    we mark it as critical here to satisfy the framework's constraint that critical parents cannot have non-critical children.
    """
    # Create the critical parallel node for festival identification
    fest_node = evaluator.add_parallel(
        id="Festival_Identification",
        desc="Verify the festival location, timing, and venue details",
        parent=parent_node,
        critical=True
    )

    # Reference URL presence (gates other checks if missing)
    has_festival_urls = bool(fest.reference_urls)
    evaluator.add_custom_node(
        result=has_festival_urls,
        id="Festival_Reference_URL",
        desc="Provide a reference URL confirming the festival details",
        parent=fest_node,
        critical=True
    )

    # Location & Date
    loc_date_leaf = evaluator.add_leaf(
        id="Location_And_Date",
        desc="The festival takes place in Fort Lauderdale, Florida on April 10-12, 2026",
        parent=fest_node,
        critical=True
    )
    await evaluator.verify(
        claim="The festival takes place in Fort Lauderdale, Florida on April 10–12, 2026.",
        node=loc_date_leaf,
        sources=fest.reference_urls,
        additional_instruction=(
            "Accept reasonable date formatting variants (e.g., Apr 10–12, 2026). "
            "Focus on city=Fort Lauderdale, state=Florida, and the 3-day range April 10–12, 2026 on Fort Lauderdale Beach."
        ),
    )

    # Venue Type (Beach)
    venue_type_leaf = evaluator.add_leaf(
        id="Venue_Type",
        desc="The festival is held on Fort Lauderdale Beach (beach venue)",
        parent=fest_node,
        critical=True
    )
    await evaluator.verify(
        claim="The festival is held on Fort Lauderdale Beach (a beach venue).",
        node=venue_type_leaf,
        sources=fest.reference_urls,
        additional_instruction="Allow synonyms like 'Fort Lauderdale Beach Park' that clearly indicate the beach location."
    )

    # Venue Address
    venue_addr_leaf = evaluator.add_leaf(
        id="Venue_Address",
        desc="The festival venue address is 1100 Seabreeze Blvd, Fort Lauderdale, FL 33316",
        parent=fest_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official festival venue address is 1100 Seabreeze Blvd, Fort Lauderdale, FL 33316.",
        node=venue_addr_leaf,
        sources=fest.reference_urls,
        additional_instruction=(
            "Allow reasonable address formatting variants (e.g., 'Seabreeze Boulevard'). "
            "The ZIP should be 33316. If the page shows the beach park address matching this, consider it supported."
        )
    )


async def build_performer_identification_subtree(
    evaluator: Evaluator,
    parent_node,
    perf: PerformerExtraction,
    fest: FestivalExtraction,
) -> None:
    """
    Build and verify the 'Performer_Identification' subtree.
    As above, we mark the reference URL presence leaf as critical to conform with the framework's constraint.
    """
    perf_node = evaluator.add_parallel(
        id="Performer_Identification",
        desc="Identify and verify the performer's role, history, and recognition at the festival",
        parent=parent_node,
        critical=True
    )

    # Reference URL presence for performer facts (gates other checks if missing)
    has_perf_urls = bool(perf.reference_urls)
    evaluator.add_custom_node(
        result=has_perf_urls,
        id="Performer_Reference_URL",
        desc="Provide a reference URL confirming the performer's headline slot, festival history, and award",
        parent=perf_node,
        critical=True
    )

    # Prepare sources (use both performer and festival URLs if available)
    combined_sources = _combine_sources(perf.reference_urls, fest.reference_urls)

    # Headline Performance (Sunday, April 12, 2026)
    headline_leaf = evaluator.add_leaf(
        id="Headline_Performance",
        desc="The performer headlines on Sunday, April 12, 2026 (the final day of the three-day festival)",
        parent=perf_node,
        critical=True
    )
    performer_name = perf.performer_name or ""
    await evaluator.verify(
        claim=f"{performer_name} is a headliner on Sunday, April 12, 2026 at this festival.",
        node=headline_leaf,
        sources=combined_sources,
        additional_instruction=(
            "Verify the official daily lineup/schedule. "
            "Minor name variations are acceptable if they clearly refer to the same artist. "
            "Sunday is the final day of the three-day festival (Apr 10–12, 2026)."
        )
    )

    # Festival History (previously headlined multiple times)
    history_leaf = evaluator.add_leaf(
        id="Festival_History",
        desc="The performer has previously headlined this same festival multiple times (documented history)",
        parent=perf_node,
        critical=True
    )
    years_text = ", ".join(perf.previous_headline_years) if perf.previous_headline_years else "multiple prior years"
    await evaluator.verify(
        claim=f"{performer_name} has headlined this same festival in multiple prior years (e.g., {years_text}).",
        node=history_leaf,
        sources=combined_sources,
        additional_instruction=(
            "Confirm that the artist was a headliner (top billed) in at least two distinct prior editions of the same festival. "
            "Articles, lineup posters, or official historical pages count if they explicitly show headliner status."
        )
    )

    # Award Reception (special ocean conservation award at 2026 festival)
    award_reception_leaf = evaluator.add_leaf(
        id="Award_Reception",
        desc="The performer is receiving a special ocean conservation award at the 2026 festival",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"During the 2026 festival, {performer_name} is scheduled to receive a special ocean conservation award during their performance.",
        node=award_reception_leaf,
        sources=combined_sources,
        additional_instruction=(
            "Look for explicit statements that the artist will receive a conservation-related award on-stage during the 2026 festival performance. "
            "Press releases, official festival announcements, or reputable news coverage are acceptable."
        )
    )

    # Award Name (specific)
    award_name_leaf = evaluator.add_leaf(
        id="Award_Name",
        desc="The specific name of the conservation award is provided",
        parent=perf_node,
        critical=True
    )
    award_name = perf.award_name or ""
    await evaluator.verify(
        claim=f"The specific name of the conservation award that {performer_name} is receiving is '{award_name}'.",
        node=award_name_leaf,
        sources=combined_sources,
        additional_instruction=(
            "Confirm the exact award name as written in the cited source(s). "
            "Minor punctuation or apostrophe variations are acceptable if they clearly refer to the same named award."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Entry point for evaluating the agent's answer against the rubric tree.
    """
    # Initialize evaluator with a neutral root; we'll add a critical top-level node beneath it.
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

    # Parallelize extractions
    fest_task = evaluator.extract(
        prompt=prompt_extract_festival_details(),
        template_class=FestivalExtraction,
        extraction_name="festival_details"
    )
    perf_task = evaluator.extract(
        prompt=prompt_extract_performer_details(),
        template_class=PerformerExtraction,
        extraction_name="performer_details"
    )
    fest_extraction, perf_extraction = await asyncio.gather(fest_task, perf_task)

    # Top-level critical node aggregating both festival and performer identification
    top_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify the correct performer and award based on all specified constraints",
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_festival_identification_subtree(evaluator, top_node, fest_extraction)
    await build_performer_identification_subtree(evaluator, top_node, perf_extraction, fest_extraction)

    # Return summarized evaluation
    return evaluator.get_summary()