import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "azores_trail_prc_smi_easy_trail"
TASK_DESCRIPTION = (
    "Identify a circular hiking trail located on São Miguel Island in the Azores that meets all of the following criteria: "
    "Must be classified as an 'Easy' difficulty level trail by the Regional Government of the Azores; "
    "Must be a circular trail (PRC classification, not linear PR classification); "
    "Trail distance must be less than 5 kilometers; "
    "Elevation gain must be less than 300 meters; "
    "Must have an official PRC trail code following the format PRC##SMI (where ## is a two-digit number and SMI indicates São Miguel Island). "
    "Provide the complete official PRC code and the official trail name. "
    "Include reference URLs that confirm the trail's location, classification type (circular/PRC), difficulty rating, distance, and elevation gain."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TrailExtraction(BaseModel):
    # Required identifiers
    prc_code: Optional[str] = None
    official_name: Optional[str] = None

    # Stated attributes (as strings, keep as-is from the answer)
    island: Optional[str] = None                  # e.g., "São Miguel", "Sao Miguel"
    classification_type: Optional[str] = None     # e.g., "PRC", "circular", "PR"
    difficulty: Optional[str] = None              # e.g., "Easy", "Fácil"
    distance: Optional[str] = None                # e.g., "4.2 km"
    elevation_gain: Optional[str] = None          # e.g., "250 m"
    january_accessibility_note: Optional[str] = None  # e.g., "accessible year-round", "open all year"

    # URLs grouped by what they are supposed to confirm
    urls_location: List[str] = Field(default_factory=list)
    urls_prc_classification: List[str] = Field(default_factory=list)
    urls_difficulty_official: List[str] = Field(default_factory=list)
    urls_distance: List[str] = Field(default_factory=list)
    urls_elevation_gain: List[str] = Field(default_factory=list)
    urls_official_listing: List[str] = Field(default_factory=list)  # official portal pages (if provided)
    urls_general: List[str] = Field(default_factory=list)           # any other URLs mentioned


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trail() -> str:
    return """
    Extract exactly one trail (the first or primary one if multiple are present) and its supporting references from the answer text.

    Return the following fields:
    - prc_code: the complete official PRC code exactly as shown (e.g., "PRC05SMI")
    - official_name: the official trail name exactly as shown
    - island: the island the trail is on (e.g., "São Miguel" or "Sao Miguel")
    - classification_type: the classification term used for the trail (e.g., "PRC", "circular", "PR", etc.)
    - difficulty: the difficulty rating stated in the answer (e.g., "Easy", "Fácil")
    - distance: the trail distance value as stated (keep units and punctuation, e.g., "4.8 km" or "4,8 km")
    - elevation_gain: the elevation gain as stated (keep units, e.g., "250 m")
    - january_accessibility_note: any statement about January accessibility or year-round availability; if not stated, set null

    Also extract URLs mentioned in the answer text and group them by what they are supposed to confirm:
    - urls_location: URLs that confirm the trail is on São Miguel (Azores)
    - urls_prc_classification: URLs that confirm the PRC/circular classification (not linear PR)
    - urls_difficulty_official: URLs that confirm the official difficulty rating, ideally an official Azores trails portal
    - urls_distance: URLs that confirm the distance
    - urls_elevation_gain: URLs that confirm the elevation gain
    - urls_official_listing: URLs that appear to be an official Azores trails listing page for this specific trail (if any)
    - urls_general: any other URLs referenced for this trail not already captured above

    IMPORTANT:
    - Only extract URLs explicitly present in the answer (including markdown links). Do not invent URLs.
    - If a single official trail page is cited to confirm multiple attributes (e.g., distance and difficulty), include that URL in each relevant list.
    - If any field is missing in the answer, return null (or an empty list for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _unique_nonempty_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out


def _union_urls(*lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        combined.extend(lst or [])
    return _unique_nonempty_urls(combined)


def _safe_trail_label(ex: TrailExtraction) -> str:
    if _is_nonempty_str(ex.official_name):
        return f"'{ex.official_name.strip()}'"
    if _is_nonempty_str(ex.prc_code):
        return f"with code {ex.prc_code.strip()}"
    return "the trail"


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def _build_and_verify_tree(evaluator: Evaluator, ex: TrailExtraction) -> None:
    # Top-level critical groups
    answer_ident_node = evaluator.add_parallel(
        id="answer_identification",
        desc="Provide the required trail identifiers.",
        parent=evaluator.root,
        critical=True
    )

    elig_node = evaluator.add_parallel(
        id="eligibility_constraints",
        desc="Verify the trail satisfies each stated constraint.",
        parent=evaluator.root,
        critical=True
    )

    refs_node = evaluator.add_parallel(
        id="supporting_references",
        desc="Provide reference URLs that confirm each required attribute explicitly requested for confirmation by URL.",
        parent=evaluator.root,
        critical=True
    )

    # ----------------------------
    # Answer identification (critical existence)
    # ----------------------------
    evaluator.add_custom_node(
        result=_is_nonempty_str(ex.prc_code),
        id="provide_prc_code",
        desc="Provide the complete official PRC trail code.",
        parent=answer_ident_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty_str(ex.official_name),
        id="provide_official_trail_name",
        desc="Provide the official trail name.",
        parent=answer_ident_node,
        critical=True
    )

    # ----------------------------
    # Supporting references (critical presence of URLs)
    # ----------------------------
    evaluator.add_custom_node(
        result=len(ex.urls_location) > 0,
        id="url_confirms_location",
        desc="Include at least one reference URL that confirms the trail is on São Miguel Island (Azores).",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(ex.urls_prc_classification) > 0,
        id="url_confirms_prc_circular",
        desc="Include at least one reference URL that confirms the trail is PRC/circular (not PR/linear).",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(ex.urls_difficulty_official) > 0 or len(ex.urls_official_listing) > 0),
        id="url_confirms_easy_and_official_classification",
        desc="Include at least one reference URL that confirms the trail's official 'Easy' difficulty classification by the Regional Government of the Azores.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(ex.urls_distance) > 0 or len(ex.urls_official_listing) > 0),
        id="url_confirms_distance",
        desc="Include at least one reference URL that confirms the trail distance value.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(ex.urls_elevation_gain) > 0 or len(ex.urls_official_listing) > 0),
        id="url_confirms_elevation_gain",
        desc="Include at least one reference URL that confirms the elevation gain value.",
        parent=refs_node,
        critical=True
    )

    # ----------------------------
    # Eligibility constraints (all critical verifications)
    # ----------------------------

    # Code format check PRC##SMI (two digits)
    code_ok = False
    if _is_nonempty_str(ex.prc_code):
        code_ok = re.fullmatch(r"\s*PRC\d{2}SMI\s*", ex.prc_code.strip().upper()) is not None
    evaluator.add_custom_node(
        result=code_ok,
        id="code_format_prc_hash_hash_smi",
        desc="Trail code matches the format PRC##SMI (two digits, then SMI).",
        parent=elig_node,
        critical=True
    )

    # Location on São Miguel (Azores)
    loc_leaf = evaluator.add_leaf(
        id="location_sao_miguel",
        desc="Trail is located on São Miguel Island in the Azores archipelago.",
        parent=elig_node,
        critical=True
    )
    loc_sources = _union_urls(
        ex.urls_location,
        ex.urls_official_listing,
        ex.urls_prc_classification,
        ex.urls_difficulty_official,
        ex.urls_distance,
        ex.urls_elevation_gain,
        ex.urls_general
    )
    await evaluator.verify(
        claim=f"The trail {_safe_trail_label(ex)} is located on São Miguel Island (in the Azores).",
        node=loc_leaf,
        sources=loc_sources,
        additional_instruction=(
            "Treat 'São Miguel' and 'Sao Miguel' as equivalent. Pages that show the SMI code or mention São Miguel explicitly support this."
        )
    )

    # Classification PRC (circular)
    prc_leaf = evaluator.add_leaf(
        id="classification_prc_circular",
        desc="Trail is classified as PRC (circular), not PR (linear).",
        parent=elig_node,
        critical=True
    )
    prc_sources = _union_urls(
        ex.urls_prc_classification,
        ex.urls_official_listing,
        ex.urls_difficulty_official,
        ex.urls_general
    )
    await evaluator.verify(
        claim="This trail is a PRC circular trail (not a linear PR trail).",
        node=prc_leaf,
        sources=prc_sources,
        additional_instruction=(
            "Look for 'PRC' label or text indicating 'percurso circular' (circular). "
            "If the page shows a code like PRC##SMI, that implies PRC (circular). "
            "Do not accept PR (linear) as circular."
        )
    )

    # Officially classified by the Regional Government of the Azores
    official_leaf = evaluator.add_leaf(
        id="officially_classified_by_regional_government",
        desc="Trail is officially classified by the Regional Government of the Azores.",
        parent=elig_node,
        critical=True
    )
    official_sources = _union_urls(
        ex.urls_official_listing,
        ex.urls_difficulty_official,
        ex.urls_prc_classification
    )
    await evaluator.verify(
        claim=f"This webpage is an official page by the Regional Government of the Azores that lists or classifies the trail {_safe_trail_label(ex)}.",
        node=official_leaf,
        sources=official_sources,
        additional_instruction=(
            "Accept official Azores trails portals (e.g., Visit Azores / trilhos) or other clearly official Azores government domains. "
            "Look for signs of official branding such as 'Governo dos Açores', official logos, or explicit statements indicating it is an official portal."
        )
    )

    # Difficulty Easy
    diff_leaf = evaluator.add_leaf(
        id="difficulty_easy",
        desc="Trail difficulty rating is classified as 'Easy'.",
        parent=elig_node,
        critical=True
    )
    diff_sources = _union_urls(
        ex.urls_difficulty_official,
        ex.urls_official_listing
    )
    await evaluator.verify(
        claim="The trail's difficulty is classified as 'Easy' (Portuguese: 'Fácil') by the official listing.",
        node=diff_leaf,
        sources=diff_sources,
        additional_instruction=(
            "Only accept difficulty if the page explicitly shows 'Easy' or 'Fácil' for this trail. "
            "If multiple difficulties are listed, ensure 'Easy' applies to this specific route."
        )
    )

    # Distance < 5 km
    dist_leaf = evaluator.add_leaf(
        id="distance_under_5_km",
        desc="Trail distance is less than 5 kilometers.",
        parent=elig_node,
        critical=True
    )
    dist_sources = _union_urls(
        ex.urls_distance,
        ex.urls_official_listing
    )
    await evaluator.verify(
        claim="The trail's official distance is less than 5 kilometers.",
        node=dist_leaf,
        sources=dist_sources,
        additional_instruction=(
            "Use the numeric distance value shown on the page. "
            "Accept decimal commas (e.g., 4,8 km) as 4.8 km. "
            "If it is exactly 5.0 km or more, do NOT support. "
            "Minor rounding differences are acceptable only if clearly < 5 km."
        )
    )

    # Elevation gain < 300 m
    elev_leaf = evaluator.add_leaf(
        id="elevation_gain_under_300_m",
        desc="Elevation gain is less than 300 meters.",
        parent=elig_node,
        critical=True
    )
    elev_sources = _union_urls(
        ex.urls_elevation_gain,
        ex.urls_official_listing
    )
    await evaluator.verify(
        claim="The trail's elevation gain is less than 300 meters.",
        node=elev_leaf,
        sources=elev_sources,
        additional_instruction=(
            "Use the numeric elevation gain shown on the page. "
            "If the page lists multiple elevation metrics, consider 'elevation gain' or equivalent indicator. "
            "If it is exactly 300 m or higher, do NOT support."
        )
    )

    # January accessible/hikeable
    jan_leaf = evaluator.add_leaf(
        id="january_accessible_hikeable",
        desc="Trail is accessible and hikeable during January (specifically around January 19, 2026).",
        parent=elig_node,
        critical=True
    )
    jan_sources = _union_urls(
        ex.urls_official_listing,
        ex.urls_general,
        ex.urls_prc_classification,
        ex.urls_difficulty_official
    )
    await evaluator.verify(
        claim="The trail is accessible and hikeable in January.",
        node=jan_leaf,
        sources=jan_sources,
        additional_instruction=(
            "Accept explicit statements such as 'open all year', 'year-round', or Portuguese equivalents like 'todo o ano'. "
            "If the page contains explicit seasonal closures or recommends not hiking in January, do NOT support. "
            "If no information is provided about seasonality or January accessibility, do NOT support."
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
    Evaluate an answer for the Azores São Miguel PRC trail task.
    """
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_trail(),
        template_class=TrailExtraction,
        extraction_name="trail_extraction"
    )

    # Record some custom info counts (optional, helps debugging)
    evaluator.add_custom_info(
        info={
            "prc_code": extraction.prc_code,
            "official_name": extraction.official_name,
            "url_counts": {
                "urls_location": len(extraction.urls_location),
                "urls_prc_classification": len(extraction.urls_prc_classification),
                "urls_difficulty_official": len(extraction.urls_difficulty_official),
                "urls_distance": len(extraction.urls_distance),
                "urls_elevation_gain": len(extraction.urls_elevation_gain),
                "urls_official_listing": len(extraction.urls_official_listing),
                "urls_general": len(extraction.urls_general),
            }
        },
        info_type="extraction_diagnostics",
    )

    # Build and verify the rubric tree
    await _build_and_verify_tree(evaluator, extraction)

    # Return the summarized evaluation
    return evaluator.get_summary()