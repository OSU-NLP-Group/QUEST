import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


# -----------------------------------------------------------------------------
# Task Constants
# -----------------------------------------------------------------------------
TASK_ID = "ca_eclipse_adjacent_university_observatory_2026"
TASK_DESCRIPTION = (
    "A researcher wants to observe the March 3, 2026 total lunar eclipse from a California public university with an observatory. "
    "The university must be located in a county that shares a border with Santa Barbara County, where Vandenberg Space Force Base is located. "
    "Identify: (1) the name of a county adjacent to Santa Barbara County that contains such a university, "
    "(2) the name of the public university, (3) the name of the observatory facility, and "
    "(4) the local time range of totality for the eclipse as observed from that location."
)


# -----------------------------------------------------------------------------
# Data Models for Extraction
# -----------------------------------------------------------------------------
class CountyInfo(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class UniversityInfo(BaseModel):
    name: Optional[str] = None
    county: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ObservatoryInfo(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EclipseInfo(BaseModel):
    totality_local_time_range: Optional[str] = None
    visibility_sources: List[str] = Field(default_factory=list)
    timing_sources: List[str] = Field(default_factory=list)


class ResearchExtraction(BaseModel):
    county: Optional[CountyInfo] = None
    university: Optional[UniversityInfo] = None
    observatory: Optional[ObservatoryInfo] = None
    eclipse: Optional[EclipseInfo] = None


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_research() -> str:
    return """
    Extract the following information exactly as it appears in the provided answer text. Do not infer or add anything that is not explicitly present.

    Return a JSON object with this structure:
    {
      "county": {
        "name": string or null,
        "sources": array of URL strings (can be empty)
      },
      "university": {
        "name": string or null,
        "county": string or null,   // the county explicitly claimed for the university in the answer (if mentioned)
        "sources": array of URL strings (can be empty) // URLs supporting the university's existence/public status/location
      },
      "observatory": {
        "name": string or null,
        "sources": array of URL strings (can be empty) // URLs supporting the observatory facility at the university
      },
      "eclipse": {
        "totality_local_time_range": string or null, // e.g., "10:12–10:53 PM PST" (local time range of totality at that location)
        "visibility_sources": array of URL strings (can be empty), // URLs about eclipse visibility at the location
        "timing_sources": array of URL strings (can be empty) // URLs about local timing/totality window at the location
      }
    }

    Rules:
    - "sources" and URL fields must be explicit URLs present in the answer (plain links or markdown links). If none are present, return an empty list.
    - Do not fabricate URLs or details.
    - Preserve the time range string exactly as given (including AM/PM / timezone if included).
    """


# -----------------------------------------------------------------------------
# Helper Utilities
# -----------------------------------------------------------------------------
def _dedup_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not isinstance(u, str):
                continue
            u_stripped = u.strip()
            if not u_stripped:
                continue
            if u_stripped not in seen:
                seen.add(u_stripped)
                result.append(u_stripped)
    return result


def _safe(s: Optional[str]) -> str:
    return s or ""


# -----------------------------------------------------------------------------
# Verification Subtrees
# -----------------------------------------------------------------------------
async def build_geo_and_institutional_checks(
    evaluator: Evaluator,
    parent: VerificationNode,
    ex: ResearchExtraction
) -> Dict[str, VerificationNode]:
    """
    Build and run the Geographic_and_Institutional_Identification subtree (sequential).
    Returns key leaf nodes for use as prerequisites for eclipse checks.
    """
    results: Dict[str, VerificationNode] = {}

    geo_inst_node = evaluator.add_sequential(
        id="Geographic_and_Institutional_Identification",
        desc="Identify the county, public university, and observatory facility meeting the specified geographic constraints",
        parent=parent,
        critical=False
    )

    # 1) County identification and adjacency (critical)
    county_node = evaluator.add_parallel(
        id="County_Identification_and_Adjacency",
        desc="Identify a county that shares a geographic border with Santa Barbara County, California",
        parent=geo_inst_node,
        critical=True
    )

    county_name_present = evaluator.add_custom_node(
        result=bool(ex.county and ex.county.name and ex.county.name.strip()),
        id="County_Name_Present",
        desc="County name is provided in the answer",
        parent=county_node,
        critical=True
    )

    county_ref_leaf = evaluator.add_leaf(
        id="County_Geographic_Reference",
        desc="Provide valid reference URL(s) verifying the county borders Santa Barbara County",
        parent=county_node,
        critical=True
    )
    results["county_ref_leaf"] = county_ref_leaf

    county_name = _safe(ex.county.name if ex.county else None)
    county_sources = (ex.county.sources if (ex.county and ex.county.sources) else [])

    await evaluator.verify(
        claim=f"{county_name} shares a geographic border with Santa Barbara County, California.",
        node=county_ref_leaf,
        sources=county_sources,
        additional_instruction="Verify adjacency strictly using the provided sources (e.g., official county pages, maps, or Wikipedia pages listing adjacent counties)."
    )

    # 2) Public University identification (critical)
    uni_node = evaluator.add_parallel(
        id="Public_University_Identification",
        desc="Identify a public university that exists in the identified county",
        parent=geo_inst_node,
        critical=True
    )

    uni_name_present = evaluator.add_custom_node(
        result=bool(ex.university and ex.university.name and ex.university.name.strip()),
        id="University_Name_Present",
        desc="University name is provided in the answer",
        parent=uni_node,
        critical=True
    )

    uni_sources_present = evaluator.add_custom_node(
        result=bool(ex.university and ex.university.sources and len(ex.university.sources) > 0),
        id="University_Sources_Provided",
        desc="University reference URL(s) are provided",
        parent=uni_node,
        critical=True
    )

    # Split verification into two critical leaves to avoid multi-claim aggregation:
    # 2.a) University is public
    uni_public_leaf = evaluator.add_leaf(
        id="University_Reference",
        desc="Provide valid reference URL(s) verifying the university exists and is public",
        parent=uni_node,
        critical=True
    )
    results["uni_public_leaf"] = uni_public_leaf

    uni_name = _safe(ex.university.name if ex.university else None)
    uni_sources = (ex.university.sources if (ex.university and ex.university.sources) else [])

    await evaluator.verify(
        claim=f"{uni_name} is a public university in the United States (California).",
        node=uni_public_leaf,
        sources=uni_sources,
        additional_instruction="Confirm that the institution is publicly funded (e.g., part of the UC or CSU systems) using only the provided URLs."
    )

    # 2.b) University is located in the identified county
    uni_in_county_leaf = evaluator.add_leaf(
        id="University_In_County",
        desc="Verify the identified public university is located within the identified county",
        parent=uni_node,
        critical=True
    )
    results["uni_in_county_leaf"] = uni_in_county_leaf

    uni_claim_county = _safe(ex.university.county if ex.university else None) or county_name
    await evaluator.verify(
        claim=f"The main campus of {uni_name} is located within {uni_claim_county} County, California.",
        node=uni_in_county_leaf,
        sources=_dedup_urls(uni_sources, county_sources),
        additional_instruction=(
            "Use the provided URLs to confirm the university's location within the stated county. "
            "If a page mentions the campus city, it should explicitly state or allow clear inference (from other provided sources) "
            "that the city lies within the county."
        )
    )

    # 3) Observatory facility identification (critical)
    obs_node = evaluator.add_parallel(
        id="Observatory_Facility_Identification",
        desc="Identify an observatory facility at the identified public university",
        parent=geo_inst_node,
        critical=True
    )

    obs_name_present = evaluator.add_custom_node(
        result=bool(ex.observatory and ex.observatory.name and ex.observatory.name.strip()),
        id="Observatory_Name_Present",
        desc="Observatory facility name is provided in the answer",
        parent=obs_node,
        critical=True
    )

    obs_sources_present = evaluator.add_custom_node(
        result=bool(ex.observatory and ex.observatory.sources and len(ex.observatory.sources) > 0),
        id="Observatory_Sources_Provided",
        desc="Observatory reference URL(s) are provided",
        parent=obs_node,
        critical=True
    )

    obs_ref_leaf = evaluator.add_leaf(
        id="Observatory_Reference",
        desc="Provide valid reference URL(s) verifying the observatory facility exists at the university",
        parent=obs_node,
        critical=True
    )
    results["obs_ref_leaf"] = obs_ref_leaf

    obs_name = _safe(ex.observatory.name if ex.observatory else None)
    obs_sources = (ex.observatory.sources if (ex.observatory and ex.observatory.sources) else [])

    await evaluator.verify(
        claim=f"The observatory facility named '{obs_name}' exists at or is operated by {uni_name}.",
        node=obs_ref_leaf,
        sources=obs_sources,
        additional_instruction="Accept official university/department pages or clearly authoritative sources that show the observatory is at or run by the identified university."
    )

    return results


async def build_eclipse_information_checks(
    evaluator: Evaluator,
    parent: VerificationNode,
    ex: ResearchExtraction,
    prereq_leaves: List[VerificationNode]
) -> None:
    """
    Build and run the Eclipse_Information subtree (parallel).
    Uses prior leaves as prerequisites to ensure location validity before eclipse checks.
    """
    eclipse_node = evaluator.add_parallel(
        id="Eclipse_Information",
        desc="Provide information about observing the March 3, 2026 total lunar eclipse from the identified location",
        parent=parent,
        critical=False
    )

    eclipse = ex.eclipse or EclipseInfo()
    county_name = _safe(ex.county.name if ex.county else None)
    uni_name = _safe(ex.university.name if ex.university else None)
    obs_name = _safe(ex.observatory.name if ex.observatory else None)

    vis_urls = eclipse.visibility_sources or []
    time_urls = eclipse.timing_sources or []
    all_eclipse_urls = _dedup_urls(vis_urls, time_urls)

    # 1) Eclipse visibility verification (critical)
    vis_leaf = evaluator.add_leaf(
        id="Eclipse_Visibility_Verification",
        desc="The March 3, 2026 total lunar eclipse must be visible from the identified location",
        parent=eclipse_node,
        critical=True
    )
    location_phrase = (f"{obs_name} at {uni_name}" if obs_name and uni_name else (uni_name or county_name or "the identified location"))
    await evaluator.verify(
        claim=f"The total lunar eclipse on March 3, 2026 is visible as totality from {location_phrase} in {county_name} County, California.",
        node=vis_leaf,
        sources=all_eclipse_urls,
        additional_instruction=(
            "Confirm that the March 3, 2026 total lunar eclipse is observable as totality at the stated location using the provided URLs. "
            "Rely on the sources only; relevance to California or the specific county/city is required."
        ),
        extra_prerequisites=prereq_leaves
    )

    # 2) Totality time range (non-critical)
    time_leaf = evaluator.add_leaf(
        id="Totality_Time_Range",
        desc="Provide the local time range during which totality occurs at the identified location",
        parent=eclipse_node,
        critical=False
    )
    time_range_str = _safe(eclipse.totality_local_time_range)

    await evaluator.verify(
        claim=f"At {location_phrase}, the totality of the lunar eclipse on March 3, 2026 occurs during the local time range: {time_range_str}.",
        node=time_leaf,
        sources=time_urls if time_urls else all_eclipse_urls,
        additional_instruction=(
            "Verify the local time window of totality from the provided source(s). Allow minor formatting variations and reasonable rounding. "
            "Date boundary issues (near midnight) should be treated carefully; the local timezone should correspond to California."
        ),
        extra_prerequisites=prereq_leaves
    )

    # 3) Eclipse references validity (critical)
    refs_leaf = evaluator.add_leaf(
        id="Eclipse_References",
        desc="Provide valid reference URL(s) supporting the eclipse visibility and timing information",
        parent=eclipse_node,
        critical=True
    )

    await evaluator.verify(
        claim="This source provides visibility and/or timing information for the March 3, 2026 total lunar eclipse relevant to the specified California location.",
        node=refs_leaf,
        sources=all_eclipse_urls,
        additional_instruction=(
            "Judge each provided URL for clear relevance to the March 3, 2026 total lunar eclipse and for containing visibility or timing details "
            "applicable to the identified California location."
        ),
        extra_prerequisites=prereq_leaves
    )


# -----------------------------------------------------------------------------
# Main Evaluation Entry
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the California adjacent-county university observatory eclipse task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel (two major branches)
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

    # 1) Extract structured information from the answer
    extracted: ResearchExtraction = await evaluator.extract(
        prompt=prompt_extract_research(),
        template_class=ResearchExtraction,
        extraction_name="research_extraction"
    )

    # 2) Build rubric tree nodes according to specification
    complete_task_node = evaluator.add_parallel(
        id="Complete_Research_Task",
        desc="Correctly identify all required information about a California public university with an observatory in a county adjacent to Santa Barbara County for observing the March 3, 2026 total lunar eclipse",
        parent=root,
        critical=False
    )

    # 2.a) Geographic and Institutional Identification (sequential)
    geo_institution_leaves = await build_geo_and_institutional_checks(evaluator, complete_task_node, extracted)

    # Prepare prerequisites for eclipse checks (ensure location validity first)
    prereq_for_eclipse = [
        leaf for key, leaf in geo_institution_leaves.items()
        if key in ("county_ref_leaf", "uni_public_leaf", "uni_in_county_leaf", "obs_ref_leaf")
    ]

    # 2.b) Eclipse Information (parallel)
    await build_eclipse_information_checks(evaluator, complete_task_node, extracted, prereq_for_eclipse)

    # 3) Return final structured summary
    return evaluator.get_summary()