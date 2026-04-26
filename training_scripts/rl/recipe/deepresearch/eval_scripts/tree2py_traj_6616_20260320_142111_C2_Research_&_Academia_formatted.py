import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lunar_eclipse_2026_03_03"
TASK_DESCRIPTION = """
Research and document key characteristics of the total lunar eclipse that occurred on March 3, 2026 (also known as the "Blood Moon" eclipse). Using at least two different authoritative sources, verify and provide: (1) the duration of the totality phase in minutes, (2) the primary geographic regions from which the eclipse was visible, (3) the date of the next total lunar eclipse that will be visible from North America after this event. For each piece of information, cite the specific authoritative sources with direct URLs where this information can be verified.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EclipseInfoExtraction(BaseModel):
    # Event identification
    eclipse_date_text: Optional[str] = None
    eclipse_date_sources: List[str] = Field(default_factory=list)

    # Key characteristics
    totality_duration_minutes: Optional[str] = None
    totality_duration_sources: List[str] = Field(default_factory=list)

    visibility_regions: List[str] = Field(default_factory=list)
    visibility_regions_text: Optional[str] = None
    visibility_sources: List[str] = Field(default_factory=list)

    next_eclipse_na_date: Optional[str] = None
    next_eclipse_sources: List[str] = Field(default_factory=list)

    # Any additional sources the answer cites generally for this eclipse
    additional_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_eclipse_info() -> str:
    return """
    Extract the requested information exactly as presented in the answer text. Do not infer or invent details.

    Return a JSON object with the following fields:
    - eclipse_date_text: The date expression the answer uses for the total lunar eclipse (e.g., "March 3, 2026", "March 2–3, 2026", "3 March 2026"). If not explicitly stated, return null.
    - eclipse_date_sources: All URLs in the answer that directly support or are cited for the eclipse date. If none are provided, return an empty array.

    - totality_duration_minutes: The duration of the totality phase as the answer states it, preferably in minutes (e.g., "64 minutes", "65", "approximately 65 minutes", "1 hour 5 minutes"). Do not convert; copy the answer's phrasing. If not present, return null.
    - totality_duration_sources: All URLs in the answer that directly support the totality duration. If none are provided, return an empty array.

    - visibility_regions: A list of the primary geographic regions the answer claims for visibility of this eclipse (e.g., "North America", "South America", "Europe", "Africa", "Asia", "Australia/Oceania", "Pacific", "Atlantic"). Include 3–8 macro regions, not countries/cities. If the answer gives a free-form sentence, convert it to a concise list of macro-regions while preserving meaning.
    - visibility_regions_text: The exact sentence or phrase from the answer describing visibility regions, if present. Otherwise null.
    - visibility_sources: All URLs in the answer that directly support the visibility regions. If none are provided, return an empty array.

    - next_eclipse_na_date: The date the answer states for the next total lunar eclipse visible from North America after the March 3, 2026 event (e.g., "March 14, 2025"). Copy the answer's phrasing (do not normalize). If not provided, return null.
    - next_eclipse_sources: All URLs in the answer that directly support the 'next eclipse in North America' date. If none are provided, return an empty array.

    - additional_sources: Any other URLs the answer cites that are about this eclipse but not tied to a specific item above. If none, return an empty array.

    Rules:
    - Only include URLs that explicitly appear in the answer in any reasonable format (plain URL or markdown link).
    - Do not add or invent URLs.
    - If a field is missing in the answer, set it to null (for single values) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _get_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return url


def _collect_all_sources(ex: EclipseInfoExtraction) -> List[str]:
    all_urls = []
    all_urls += ex.eclipse_date_sources or []
    all_urls += ex.totality_duration_sources or []
    all_urls += ex.visibility_sources or []
    all_urls += ex.next_eclipse_sources or []
    all_urls += ex.additional_sources or []
    return _dedup_preserve_order(all_urls)


def _pick_two_distinct_domains(urls: List[str]) -> List[str]:
    picked = []
    seen_domains = set()
    for u in urls:
        d = _get_domain(u)
        if d and d not in seen_domains:
            seen_domains.add(d)
            picked.append(u)
        if len(picked) >= 2:
            break
    return picked


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_eclipse_identification(evaluator: Evaluator, parent, ex: EclipseInfoExtraction) -> None:
    """
    Build and verify the 'eclipse_identification' branch:
    - Provided/claimed date matches the expected event date (allowing March 2–3 due to time zones)
    - Date supported by cited sources
    - At least two distinct authoritative sources about this eclipse
    """
    node = evaluator.add_parallel(
        id="eclipse_identification",
        desc="Correctly identify and source the March 3, 2026 total lunar eclipse event",
        parent=parent,
        critical=True
    )

    # 1) Date provided in the answer (existence)
    evaluator.add_custom_node(
        result=bool(ex.eclipse_date_text and ex.eclipse_date_text.strip()),
        id="eclipse_date_provided",
        desc="An explicit eclipse date is provided in the answer",
        parent=node,
        critical=True
    )

    # 2) Date matches expected (allow "March 2–3, 2026" or equivalent due to time zones)
    date_match_leaf = evaluator.add_leaf(
        id="eclipse_date_match_expected",
        desc="The provided eclipse date is equivalent to 'March 3, 2026' (allowing March 2–3, 2026 for time zones)",
        parent=node,
        critical=True
    )
    provided_date = ex.eclipse_date_text or ""
    date_match_claim = (
        f"The provided date expression '{provided_date}' is equivalent to 'March 3, 2026' when considering global "
        f"time zones (i.e., 'March 2–3, 2026' or variants like '3 March 2026' are acceptable)."
    )
    await evaluator.verify(
        claim=date_match_claim,
        node=date_match_leaf,
        additional_instruction=(
            "Accept variants like '3 March 2026', '2026-03-03', 'March 2–3, 2026', or timezone-qualified dates "
            "that span March 2 and March 3. If the provided string is empty or refers to a different date, mark as incorrect."
        )
    )

    # 3) Date supported by cited sources (prefer date-specific sources; fall back to any if necessary)
    date_sources_present = bool(ex.eclipse_date_sources)
    evaluator.add_custom_node(
        result=date_sources_present,
        id="eclipse_date_sources_provided",
        desc="At least one URL is provided to support the eclipse date",
        parent=node,
        critical=True
    )

    date_supported_leaf = evaluator.add_leaf(
        id="eclipse_date_supported",
        desc="The eclipse date 'March 3, 2026' (may appear as March 2–3) is supported by cited sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="A total lunar eclipse occurred on March 3, 2026 (UTC), sometimes expressed as March 2–3, 2026 due to time zones.",
        node=date_supported_leaf,
        sources=ex.eclipse_date_sources,
        additional_instruction=(
            "Confirm the page specifies the total lunar eclipse date as March 3, 2026 (UTC) or an equivalent time-zone spanning form (March 2–3, 2026)."
        )
    )

    # 4) Two distinct authoritative sources about this eclipse
    all_sources = _collect_all_sources(ex)
    distinct_domains = {_get_domain(u) for u in all_sources}
    evaluator.add_custom_node(
        result=len([d for d in distinct_domains if d]) >= 2,
        id="at_least_two_distinct_sources",
        desc="Provide URLs to at least two different sources about this eclipse (distinct domains)",
        parent=node,
        critical=True
    )

    top_two = _pick_two_distinct_domains(all_sources)
    # If fewer than two, still create the leaves; verification will fail accordingly
    for idx in range(2):
        url = top_two[idx] if idx < len(top_two) else None
        leaf = evaluator.add_leaf(
            id=f"authoritative_source_{idx+1}",
            desc=f"Source #{idx+1} is authoritative and specifically about the March 3, 2026 total lunar eclipse",
            parent=node,
            critical=True
        )
        claim = (
            "This webpage is an authoritative, reputable source (e.g., space agency such as NASA/ESA/JAXA, "
            "recognized observatory, national met/astronomy office, or well-established astronomy reference like timeanddate.com) "
            "that specifically discusses the March 3, 2026 total lunar eclipse."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=url if url else None,
            additional_instruction=(
                "Judge 'authoritative' based on the site's institutional credibility and editorial standards. "
                "Accept: NASA, ESA, JPL, USNO, timeanddate.com, national observatories, reputable scientific institutions, "
                "or established astronomy publications. Reject: random blogs, unvetted forums, low-credibility aggregators. "
                "Additionally, the page must clearly concern the March 3, 2026 total lunar eclipse."
            )
        )


async def verify_characteristics(evaluator: Evaluator, parent, ex: EclipseInfoExtraction) -> None:
    """
    Build and verify the 'characteristic_verification' branch (all critical):
    - Totality duration (minutes) with sources
    - Visibility regions with sources
    - Next total lunar eclipse visible from North America with sources
    """
    node = evaluator.add_parallel(
        id="characteristic_verification",
        desc="Extract and verify three key characteristics of the eclipse from authoritative sources",
        parent=parent,
        critical=True
    )

    # ---- (1) Totality duration ----
    evaluator.add_custom_node(
        result=bool(ex.totality_duration_minutes and ex.totality_duration_minutes.strip()),
        id="totality_duration_provided",
        desc="Totality duration in minutes is provided in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ex.totality_duration_sources),
        id="totality_duration_sources_provided",
        desc="At least one source URL is provided for totality duration",
        parent=node,
        critical=True
    )
    leaf_duration = evaluator.add_leaf(
        id="totality_duration_supported",
        desc="Provide the duration of the totality phase in minutes with authoritative source URL",
        parent=node,
        critical=True
    )
    duration_value = ex.totality_duration_minutes or ""
    await evaluator.verify(
        claim=f"The duration of the totality phase was approximately {duration_value} minutes.",
        node=leaf_duration,
        sources=ex.totality_duration_sources,
        additional_instruction=(
            "Verify the page's 'totality' (total phase) duration. Allow reasonable rounding or phrasing differences "
            "(e.g., seconds included or ±2 minutes tolerance). Do not confuse with partial or penumbral durations."
        )
    )

    # ---- (2) Visibility regions ----
    regions_text_for_claim = (
        ", ".join(ex.visibility_regions) if ex.visibility_regions else (ex.visibility_regions_text or "")
    )
    evaluator.add_custom_node(
        result=bool(regions_text_for_claim.strip()),
        id="visibility_regions_provided",
        desc="Primary geographic visibility regions are provided in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ex.visibility_sources),
        id="visibility_regions_sources_provided",
        desc="At least one source URL is provided for visibility regions",
        parent=node,
        critical=True
    )
    leaf_visibility = evaluator.add_leaf(
        id="visibility_regions_supported",
        desc="Identify the primary geographic regions where the eclipse was visible with authoritative source URL",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The primary visibility regions for the March 3, 2026 total lunar eclipse included: {regions_text_for_claim}.",
        node=leaf_visibility,
        sources=ex.visibility_sources,
        additional_instruction=(
            "Confirm that the cited page lists substantially similar macro-regions (e.g., North America, South America, "
            "Europe, Africa, Asia, Australia/Oceania, Pacific). Allow equivalent phrasings and overlapping groupings. "
            "The statement should match the page's regional coverage at a macro level."
        )
    )

    # ---- (3) Next total lunar eclipse visible from North America ----
    evaluator.add_custom_node(
        result=bool(ex.next_eclipse_na_date and ex.next_eclipse_na_date.strip()),
        id="next_eclipse_provided",
        desc="Date of the next total lunar eclipse visible from North America is provided in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ex.next_eclipse_sources),
        id="next_eclipse_sources_provided",
        desc="At least one source URL is provided for the 'next eclipse visible from North America' date",
        parent=node,
        critical=True
    )
    leaf_next = evaluator.add_leaf(
        id="next_eclipse_supported",
        desc="Identify the date of the next total lunar eclipse visible from North America with authoritative source URL",
        parent=node,
        critical=True
    )
    next_date_value = ex.next_eclipse_na_date or ""
    await evaluator.verify(
        claim=(
            f"The next total lunar eclipse visible from North America after the March 3, 2026 event occurs on {next_date_value}."
        ),
        node=leaf_next,
        sources=ex.next_eclipse_sources,
        additional_instruction=(
            "Ensure it is a TOTAL lunar eclipse (not partial or penumbral) and that at least some portion of North America has visibility. "
            "If the page lists global visibility maps/tables, confirm North America is included. "
            "Reject if the date pertains to a non-total lunar eclipse."
        )
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
    Evaluate an answer for the March 3, 2026 total lunar eclipse task.
    """
    # Initialize evaluator with SEQUENTIAL root so identification gates characteristics
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_eclipse_info(),
        template_class=EclipseInfoExtraction,
        extraction_name="eclipse_info_extraction"
    )

    # Add helpful debug info: aggregate sources and domains
    all_sources = _collect_all_sources(extraction)
    evaluator.add_custom_info(
        info={
            "all_extracted_sources": all_sources,
            "distinct_domains": sorted({_get_domain(u) for u in all_sources if u}),
            "top_two_distinct_sources": _pick_two_distinct_domains(all_sources),
        },
        info_type="sources_summary",
        info_name="extracted_sources_overview"
    )

    # Build and run verifications according to rubric
    await verify_eclipse_identification(evaluator, root, extraction)
    await verify_characteristics(evaluator, root, extraction)

    # Return final structured evaluation summary
    return evaluator.get_summary()