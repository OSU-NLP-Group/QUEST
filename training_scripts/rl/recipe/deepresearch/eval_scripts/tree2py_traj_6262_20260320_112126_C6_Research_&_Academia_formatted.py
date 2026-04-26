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
TASK_ID = "eclipse_2026_aas248"
TASK_DESCRIPTION = """
I am an astronomy graduate student planning to observe the total lunar eclipse on March 3, 2026, and subsequently attend the 248th American Astronomical Society (AAS) Meeting in June 2026. I need to identify suitable observation locations and gather conference information.

Please provide the following information:

1. Two Major Research-Grade Observatories:
   - Identify one major research-grade astronomical observatory in Arizona that will have visibility of the March 3, 2026 total lunar eclipse
   - Identify one major research-grade astronomical observatory in California that will have visibility of the March 3, 2026 total lunar eclipse
   - For each observatory, provide:
     * Observatory name
     * Specific location (city or mountain name) within the state
     * Confirmation that the eclipse will be visible from that location on March 3, 2026
     * Confirmation that it is a major research-grade facility
     * Reference URL(s) supporting this information

2. 248th AAS Meeting Information:
   - Provide the exact dates of the 248th AAS Meeting
   - Confirm the meeting location
   - Identify what types of presentation formats are available at AAS meetings (e.g., oral presentations, poster presentations)
   - Provide reference URL(s) for the meeting and presentation information
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ObservatoryInfo(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None  # City, site, or mountain name
    visibility_confirmation: Optional[str] = None  # Textual confirmation quoted/extracted from answer
    research_grade_confirmation: Optional[str] = None  # Textual confirmation quoted/extracted from answer
    reference_urls: List[str] = Field(default_factory=list)  # URLs explicitly cited in the answer


class ObservatoriesExtraction(BaseModel):
    arizona: Optional[ObservatoryInfo] = None
    california: Optional[ObservatoryInfo] = None


class ConferenceExtraction(BaseModel):
    dates: Optional[str] = None  # Exact dates (e.g., "June 1–6, 2026")
    location: Optional[str] = None  # City, venue, or city+state
    presentation_formats: List[str] = Field(default_factory=list)  # e.g., ["oral", "poster"]
    reference_urls: List[str] = Field(default_factory=list)  # URLs explicitly cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_observatories() -> str:
    return """
    Extract information for exactly two observatories mentioned in the answer: one in Arizona and one in California, each suitable for observing the March 3, 2026 total lunar eclipse.

    For each state (arizona and california), extract the following fields from the answer exactly as presented:
    - name: The observatory name
    - location: The specific location within the state (city, mountain/site name)
    - visibility_confirmation: The answer's statement confirming the March 3, 2026 total lunar eclipse is visible from that location
    - research_grade_confirmation: The answer's statement confirming it is a major research-grade facility
    - reference_urls: All reference URLs in the answer that support the observatory information (observatory details, research-grade status, and eclipse visibility). Return only valid complete URLs; include multiple if present.

    Rules:
    - If more than one observatory per state is listed, return only the first for that state.
    - If any field is missing for a state, set it to null (or an empty list for reference_urls).
    - Do not invent URLs; only return URLs explicitly present in the answer text (including markdown links).
    """


def prompt_extract_conference() -> str:
    return """
    Extract information about the 248th American Astronomical Society (AAS) Meeting from the answer.

    Fields to extract:
    - dates: The exact dates for the 248th AAS Meeting as written in the answer (e.g., "June 1–6, 2026"). Include the year.
    - location: The meeting location as written in the answer (e.g., city and state, or city+venue if provided).
    - presentation_formats: A list of presentation types mentioned in the answer for AAS meetings (e.g., "oral", "poster", "iPoster", "plenary", etc.). Use lowercase singular nouns where possible.
    - reference_urls: All URLs explicitly cited in the answer that support the meeting dates, location, or presentation formats. Only include valid URLs.

    Rules:
    - Do not infer or create URLs; only extract those explicitly present in the answer text (including markdown links).
    - If a field is missing, set it to null or an empty list (for presentation_formats/reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_nonempty_text(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _has_valid_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


def _fmt_list(values: List[str]) -> str:
    return ", ".join(values) if values else ""


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_observatory(
    evaluator: Evaluator,
    parent_node,
    state_id: str,           # "arizona" | "california"
    state_name: str,         # "Arizona" | "California"
    obs: Optional[ObservatoryInfo],
) -> None:
    """
    Build verification nodes for a single state's observatory.
    Leaves:
      - {state}_name (custom existence)
      - {state}_location (custom existence)
      - {state}_visibility (URL-verified)
      - {state}_research_grade (URL-verified)
      - {state}_reference (custom existence)
    """
    # Aggregator for this state
    agg = evaluator.add_parallel(
        id=f"{state_id}_observatory",
        desc=f"Identify one major research-grade observatory in {state_name} suitable for eclipse viewing",
        parent=parent_node,
        critical=False  # Allow partial credit between states
    )

    # Existence / gating checks (critical siblings)
    name_ok = _has_nonempty_text(obs.name) if obs else False
    evaluator.add_custom_node(
        result=name_ok,
        id=f"{state_id}_name",
        desc=f"Observatory name is provided",
        parent=agg,
        critical=True
    )

    location_ok = _has_nonempty_text(obs.location) if obs else False
    evaluator.add_custom_node(
        result=location_ok,
        id=f"{state_id}_location",
        desc=f"Specific location (city or mountain name) in {state_name} is provided",
        parent=agg,
        critical=True
    )

    refs_ok = _has_valid_urls(obs.reference_urls) if obs else False
    evaluator.add_custom_node(
        result=refs_ok,
        id=f"{state_id}_reference",
        desc=f"Valid reference URL(s) supporting the observatory information are provided",
        parent=agg,
        critical=True
    )

    # Visibility verification
    vis_node = evaluator.add_leaf(
        id=f"{state_id}_visibility",
        desc=f"Confirmation that the March 3, 2026 total lunar eclipse will be visible from the observatory location is provided",
        parent=agg,
        critical=True
    )
    obs_name = (obs.name if obs else "") or ""
    loc_text = (obs.location if obs else "") or ""
    claim_visibility = (
        f"The total lunar eclipse on March 3, 2026 (local date may appear as March 3–4 depending on time zone) "
        f"is visible from {obs_name or 'the selected observatory'} in {state_name}{', specifically ' + loc_text if loc_text else ''}."
    )
    await evaluator.verify(
        claim=claim_visibility,
        node=vis_node,
        sources=(obs.reference_urls if obs else []),
        additional_instruction=(
            "Use the provided URLs to confirm lunar eclipse visibility for the specified location on 2026-03-03. "
            "Accept authoritative sources (e.g., NASA, timeanddate.com, observatory or university pages) that clearly show "
            "the location lies within the visibility region for the March 3, 2026 total lunar eclipse. "
            "If the source notes local date spanning Mar 3–4, that still counts as visible on Mar 3 in UTC terms."
        ),
    )

    # Research-grade verification
    rg_node = evaluator.add_leaf(
        id=f"{state_id}_research_grade",
        desc=f"Confirmation that the observatory is a major research-grade facility is provided",
        parent=agg,
        critical=True
    )
    claim_research = (
        f"{obs_name or 'The observatory'} is a major research-grade astronomical observatory with professional research facilities."
    )
    await evaluator.verify(
        claim=claim_research,
        node=rg_node,
        sources=(obs.reference_urls if obs else []),
        additional_instruction=(
            "Check that the provided sources substantiate the observatory as a significant professional research facility "
            "(e.g., operated by a university, national observatory, or major research organization; hosts large telescopes; "
            "engages in active scientific research; recognized by the astronomical community). "
            "General tourism-only sites without research credentials are not sufficient."
        ),
    )


async def verify_conference(
    evaluator: Evaluator,
    parent_node,
    conf: ConferenceExtraction
) -> None:
    """
    Build verification nodes for the 248th AAS Meeting:
      - conference_reference (custom existence)
      - meeting_dates (URL-verified)
      - meeting_location (URL-verified)
      - presentation_formats (URL-verified)
    """
    conf_node = evaluator.add_parallel(
        id="conference_planning",
        desc="Gather information about the 248th AAS Meeting in June 2026",
        parent=parent_node,
        critical=True  # Treat conference info as essential cluster
    )

    # Reference URLs existence (critical sibling for gating)
    refs_ok = _has_valid_urls(conf.reference_urls)
    evaluator.add_custom_node(
        result=refs_ok,
        id="conference_reference",
        desc="Valid reference URL(s) for the meeting and presentation information are provided",
        parent=conf_node,
        critical=True
    )

    # Meeting dates
    dates_node = evaluator.add_leaf(
        id="meeting_dates",
        desc="The exact dates of the 248th AAS Meeting are provided",
        parent=conf_node,
        critical=True
    )
    claim_dates = f"The 248th AAS Meeting takes place on {conf.dates}."
    await evaluator.verify(
        claim=claim_dates,
        node=dates_node,
        sources=conf.reference_urls,
        additional_instruction=(
            "Verify that the cited pages explicitly state the official dates for the 248th AAS Meeting (June 2026). "
            "Allow equivalent formatting (e.g., 'June 1–6, 2026' vs '1-6 June 2026')."
        ),
    )

    # Meeting location
    loc_node = evaluator.add_leaf(
        id="meeting_location",
        desc="The meeting location is provided",
        parent=conf_node,
        critical=True
    )
    claim_location = f"The 248th AAS Meeting location is {conf.location}."
    await evaluator.verify(
        claim=claim_location,
        node=loc_node,
        sources=conf.reference_urls,
        additional_instruction=(
            "Confirm that the referenced pages explicitly indicate the host city/venue for the 248th AAS Meeting."
        ),
    )

    # Presentation formats
    pf_node = evaluator.add_leaf(
        id="presentation_formats",
        desc="Types of presentation formats available at AAS meetings are identified",
        parent=conf_node,
        critical=True
    )
    formats_text = _fmt_list(conf.presentation_formats)
    claim_formats = f"AAS meetings include the following presentation formats: {formats_text}."
    await evaluator.verify(
        claim=claim_formats,
        node=pf_node,
        sources=conf.reference_urls,
        additional_instruction=(
            "Confirm that the cited pages describe presentation formats at AAS meetings (e.g., oral talks, posters, iPosters, "
            "plenary talks). Accept reasonable synonyms (e.g., 'oral' vs 'talk')."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 2026 lunar eclipse observatories and AAS 248 meeting task.
    """
    # Initialize evaluator (root is non-critical by default to allow partial credit overall)
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

    # Extract all required structured data
    observatories = await evaluator.extract(
        prompt=prompt_extract_observatories(),
        template_class=ObservatoriesExtraction,
        extraction_name="observatories_extraction"
    )

    conference = await evaluator.extract(
        prompt=prompt_extract_conference(),
        template_class=ConferenceExtraction,
        extraction_name="conference_extraction"
    )

    # Build observatory research branch (make non-critical to permit partial scoring between AZ/CA)
    obs_parent = evaluator.add_parallel(
        id="observatory_research",
        desc="Identify two suitable astronomical observatories for viewing the March 3, 2026 total lunar eclipse: one in Arizona and one in California",
        parent=root,
        critical=False
    )

    # Arizona observatory checks
    await verify_observatory(
        evaluator=evaluator,
        parent_node=obs_parent,
        state_id="arizona",
        state_name="Arizona",
        obs=observatories.arizona if observatories else None
    )

    # California observatory checks
    await verify_observatory(
        evaluator=evaluator,
        parent_node=obs_parent,
        state_id="california",
        state_name="California",
        obs=observatories.california if observatories else None
    )

    # Conference planning branch (critical cluster as per rubric)
    await verify_conference(
        evaluator=evaluator,
        parent_node=root,
        conf=conference
    )

    # Return structured summary
    return evaluator.get_summary()