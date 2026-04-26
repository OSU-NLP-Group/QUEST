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
TASK_ID = "astro_march_2026_research"
TASK_DESCRIPTION = """
A researcher specializing in astronomy is planning their schedule for March 2026 and wants to document four significant events. Identify the following four events occurring in March 2026 and provide the specified details for each:

1. Total Lunar Eclipse: Provide the exact date of the eclipse, the duration of totality in minutes, at least one geographic region where totality will be visible, and a reference URL.

2. Planetary Conjunction: Identify which two planets will appear in conjunction during March 2026, provide the exact date of this conjunction, describe where in the sky it will be visible (cardinal direction and time of day), and provide a reference URL.

3. Academic Conference on Exoplanets: Identify the name of the American Astronomical Society conference focused on exoplanet atmospheres taking place in March 2026, provide the complete date range of the conference, the city and state where it will be held, and a reference URL.

4. Vernal Equinox: Provide the exact date and time (in UTC) when the vernal (spring) equinox occurs in March 2026, and a reference URL.

For each event, all information must be verifiable through the provided reference URLs.
"""

# Ground truth expectations embedded in rubric (for transparency and partial checks)
GROUND_TRUTH = {
    "eclipse": {
        "date": "March 3, 2026",
        "totality_minutes": "58 minutes",
        "allowed_visibility_regions": [
            "eastern asia", "east asia", "australia", "pacific",
            "north america", "central america", "far western south america", "western south america"
        ],
        "not_visible_regions": ["africa", "europe"]
    },
    "conjunction": {
        "date": "March 8, 2026",
        "planets": ["venus", "saturn"],
        "viewing": {
            "direction_keyword": "west",
            "time_keywords": ["after sunset", "evening", "twilight"]
        }
    },
    "conference": {
        "name_aliases": [
            "exoplanet atmospheres 2026",
            "aastcs 11",
            "aastcs 11: exoplanet atmospheres",
            "aas topical conference series 11: exoplanet atmospheres"
        ],
        "dates": "March 16-20, 2026",
        "location": {"city": "denver", "state": "colorado"}
    },
    "equinox": {
        "date": "March 20, 2026",
        "time_utc": "14:46 UTC"
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EclipseInfo(BaseModel):
    date: Optional[str] = None
    totality_duration: Optional[str] = None  # keep as string to allow "58 min", "58 minutes"
    visibility_regions: List[str] = Field(default_factory=list)
    non_visibility_regions: List[str] = Field(default_factory=list)  # e.g., ["Africa", "Europe"]
    urls: List[str] = Field(default_factory=list)


class ConjunctionInfo(BaseModel):
    date: Optional[str] = None
    planets: List[str] = Field(default_factory=list)  # Expecting two names (order-insensitive)
    viewing_direction: Optional[str] = None          # e.g., "western sky"
    viewing_time: Optional[str] = None               # e.g., "after sunset", "evening"
    urls: List[str] = Field(default_factory=list)


class ConferenceInfo(BaseModel):
    name: Optional[str] = None
    date_range: Optional[str] = None                 # e.g., "March 16–20, 2026" (any dash)
    start_date: Optional[str] = None                 # if separately mentioned
    end_date: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class EquinoxInfo(BaseModel):
    date: Optional[str] = None                       # e.g., "March 20, 2026"
    time_utc: Optional[str] = None                   # e.g., "14:46 UTC"
    urls: List[str] = Field(default_factory=list)


class AstronomyMarch2026Extraction(BaseModel):
    eclipse: Optional[EclipseInfo] = None
    conjunction: Optional[ConjunctionInfo] = None
    conference: Optional[ConferenceInfo] = None
    equinox: Optional[EquinoxInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract from the answer the details for four March 2026 items. For each, return exactly what the answer states (do not infer).

    1) Total Lunar Eclipse (March 2026)
       - date: the eclipse date as written in the answer (string)
       - totality_duration: totality duration as written (e.g., "58 minutes", "58 min")
       - visibility_regions: list all regions the answer claims totality will be visible from (each as a short phrase, e.g., "eastern Asia", "Australia", "Pacific", "North America", "Central America", "far western South America")
       - non_visibility_regions: list all regions the answer explicitly claims the eclipse is NOT visible from (e.g., "Africa", "Europe")
       - urls: list all URLs given as references for the eclipse

    2) Planetary Conjunction (March 2026)
       - date: the conjunction date as written (string)
       - planets: list of the two planet names mentioned (e.g., ["Venus", "Saturn"])
       - viewing_direction: the directional phrasing as written (e.g., "western sky")
       - viewing_time: the time-of-day phrasing as written (e.g., "after sunset", "evening")
       - urls: list all URLs given as references for the conjunction

    3) Academic Conference on Exoplanets (March 2026)
       - name: the conference name as written (e.g., "Exoplanet Atmospheres 2026" or "AASTCS 11")
       - date_range: the full date range as written (e.g., "March 16–20, 2026" or "March 16-20, 2026")
       - start_date: start date if given separately, else null
       - end_date: end date if given separately, else null
       - city: the city as written (e.g., "Denver")
       - state: the state/region as written (e.g., "Colorado")
       - urls: list all URLs given as references for the conference

    4) Vernal Equinox (March 2026)
       - date: the date as written (string)
       - time_utc: the time in UTC as written (e.g., "14:46 UTC")
       - urls: list all URLs given as references for the equinox
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _nonempty_list(lst: Optional[List[str]]) -> bool:
    return bool(lst and any(x and str(x).strip() for x in lst))


def _filter_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if isinstance(u, str):
            u2 = u.strip()
            if u2 and ("http://" in u2 or "https://" in u2):
                out.append(u2)
    return out


def _has_allowed_visibility_region(regions: List[str]) -> Optional[str]:
    # Return the first region that matches the allowed set (normalized), else None
    allowed = set(GROUND_TRUTH["eclipse"]["allowed_visibility_regions"])
    for r in regions:
        r_norm = _normalize_text(r)
        # Handle a few common variants
        r_norm = r_norm.replace("eastern", "east").replace(" w. ", " west ").replace(" pacific ocean", " pacific")
        if r_norm in allowed:
            return r
        # Simple loosenings
        if "east asia" in r_norm:
            return r
        if "western south america" in r_norm:
            return r
        if "north america" in r_norm or "central america" in r_norm or "australia" in r_norm or "pacific" in r_norm:
            return r
    return None


def _planets_match_expected(planets: List[str]) -> bool:
    expected = set(GROUND_TRUTH["conjunction"]["planets"])
    actual = set(_normalize_text(p) for p in planets if _nonempty(p))
    # Accept "venus" and "saturn" only (order-insensitive)
    return expected.issubset(actual) and len(actual) >= 2


def _name_matches_aliases(name: Optional[str]) -> bool:
    if not _nonempty(name):
        return False
    n = _normalize_text(name)
    for alias in GROUND_TRUTH["conference"]["name_aliases"]:
        if n == alias or alias in n:
            return True
    return False


def _location_is_denver_co(city: Optional[str], state: Optional[str]) -> bool:
    return _normalize_text(city) == GROUND_TRUTH["conference"]["location"]["city"] and \
           _normalize_text(state) == GROUND_TRUTH["conference"]["location"]["state"]


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_eclipse(evaluator: Evaluator, parent_node, eclipse: Optional[EclipseInfo]) -> None:
    node = evaluator.add_parallel(
        id="Total_Lunar_Eclipse_Event",
        desc="Correct identification and details of the total lunar eclipse in March 2026",
        parent=parent_node,
        critical=False
    )

    urls = _filter_urls(eclipse.urls if eclipse else [])
    # Gate: at least one URL present (critical for event)
    evaluator.add_custom_node(
        result=_nonempty_list(urls),
        id="Eclipse_Sources_Present",
        desc="At least one reference URL is provided for the eclipse",
        parent=node,
        critical=True
    )

    # Date provided gate (critical per rubric)
    evaluator.add_custom_node(
        result=_nonempty(eclipse.date if eclipse else None),
        id="Eclipse_Date_Provided",
        desc="Eclipse date is provided in the answer",
        parent=node,
        critical=True
    )
    # Date leaf (critical)
    date_leaf = evaluator.add_leaf(
        id="Eclipse_Date",
        desc="The eclipse date is correctly identified as March 3, 2026",
        parent=node,
        critical=True
    )
    date_claim = f"The total lunar eclipse in March 2026 occurs on {eclipse.date}." if eclipse else \
        "The total lunar eclipse in March 2026 occurs on March 3, 2026."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=urls,
        additional_instruction="Verify the date on the cited page(s). Allow day/month formatting variations but the calendar day must match March 3, 2026."
    )

    # Duration provided gate (critical per rubric)
    evaluator.add_custom_node(
        result=_nonempty(eclipse.totality_duration if eclipse else None),
        id="Eclipse_Duration_Provided",
        desc="Eclipse totality duration is provided in the answer",
        parent=node,
        critical=True
    )
    # Duration leaf (critical)
    dur_leaf = evaluator.add_leaf(
        id="Totality_Duration",
        desc="The duration of totality is correctly stated as 58 minutes",
        parent=node,
        critical=True
    )
    dur_claim = f"The duration of totality for the March 2026 total lunar eclipse is {eclipse.totality_duration}." if eclipse else \
        "The duration of totality for the March 2026 total lunar eclipse is 58 minutes."
    await evaluator.verify(
        claim=dur_claim,
        node=dur_leaf,
        sources=urls,
        additional_instruction="Check the totality duration on the source page. It should match the stated number of minutes exactly or via an equivalent phrasing."
    )

    # Visibility region provided gate (critical per rubric)
    provided_region = None
    if eclipse:
        provided_region = _has_allowed_visibility_region(eclipse.visibility_regions)
    evaluator.add_custom_node(
        result=provided_region is not None,
        id="Eclipse_Visibility_Provided",
        desc="At least one correct visibility region for totality is provided in the answer",
        parent=node,
        critical=True
    )
    # Visibility leaf (critical)
    vis_leaf = evaluator.add_leaf(
        id="Visibility_Region",
        desc="At least one correct visibility region for totality is provided (eastern Asia, Australia, Pacific, North America, Central America, or far western South America)",
        parent=node,
        critical=True
    )
    region_for_claim = provided_region if provided_region else (eclipse.visibility_regions[0] if eclipse and eclipse.visibility_regions else "the stated region")
    vis_claim = f"During totality, the March 2026 total lunar eclipse is visible from {region_for_claim}."
    await evaluator.verify(
        claim=vis_claim,
        node=vis_leaf,
        sources=urls,
        additional_instruction="Confirm that the cited page lists this region among those where totality is visible."
    )

    # Non-visibility (non-critical)
    nonvis_leaf = evaluator.add_leaf(
        id="Non_Visibility_Region",
        desc="Correctly identifies that the eclipse is not visible from Africa or Europe",
        parent=node,
        critical=False
    )
    nonvis_claim = "The March 2026 total lunar eclipse is not visible from Africa or Europe."
    await evaluator.verify(
        claim=nonvis_claim,
        node=nonvis_leaf,
        sources=urls,
        additional_instruction="Verify that the page explicitly indicates non-visibility (or outside path) for Africa and Europe."
    )

    # Reference URL validity/support (critical)
    ref_leaf = evaluator.add_leaf(
        id="Eclipse_Reference_URL",
        desc="A valid reference URL supporting the eclipse information is provided",
        parent=node,
        critical=True
    )
    ref_claim = "This webpage is a relevant reference about the March 3, 2026 total lunar eclipse (date, duration of totality, or visibility)."
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=urls,
        additional_instruction="The page should clearly be about the 2026-03 total lunar eclipse and include at least one of: exact date, totality duration, or visibility information."
    )


async def verify_conjunction(evaluator: Evaluator, parent_node, conj: Optional[ConjunctionInfo]) -> None:
    node = evaluator.add_parallel(
        id="Planetary_Conjunction_Event",
        desc="Correct identification and details of the Venus-Saturn conjunction in March 2026",
        parent=parent_node,
        critical=False
    )

    urls = _filter_urls(conj.urls if conj else [])
    evaluator.add_custom_node(
        result=_nonempty_list(urls),
        id="Conjunction_Sources_Present",
        desc="At least one reference URL is provided for the conjunction",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(conj.date if conj else None),
        id="Conjunction_Date_Provided",
        desc="Conjunction date is provided in the answer",
        parent=node,
        critical=True
    )
    date_leaf = evaluator.add_leaf(
        id="Conjunction_Date",
        desc="The conjunction date is correctly identified as March 8, 2026",
        parent=node,
        critical=True
    )
    date_claim = f"The planetary conjunction in March 2026 occurs on {conj.date}." if conj else \
        "The planetary conjunction in March 2026 occurs on March 8, 2026."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=urls,
        additional_instruction="Confirm the date of the conjunction on the cited page."
    )

    evaluator.add_custom_node(
        result=_nonempty_list(conj.planets if conj else []),
        id="Planet_Names_Provided",
        desc="Planets involved are provided in the answer",
        parent=node,
        critical=True
    )
    planet_leaf = evaluator.add_leaf(
        id="Planet_Names",
        desc="Both planets (Venus and Saturn) are correctly identified",
        parent=node,
        critical=True
    )
    planets_str = ", ".join(conj.planets) if conj and conj.planets else "N/A"
    planet_claim = f"The two planets named in the answer are Venus and Saturn (order-insensitive). Provided: [{planets_str}]."
    await evaluator.verify(
        claim=planet_claim,
        node=planet_leaf,
        additional_instruction="Judge whether the provided two planet names correspond to Venus and Saturn, allowing capitalization or minor spelling variants."
    )

    # Viewing details provided gate (direction + time)
    viewing_provided = (_nonempty(conj.viewing_direction if conj else None) and _nonempty(conj.viewing_time if conj else None))
    evaluator.add_custom_node(
        result=viewing_provided,
        id="Viewing_Details_Provided",
        desc="Viewing direction and time-of-day are provided in the answer",
        parent=node,
        critical=True
    )
    viewing_leaf = evaluator.add_leaf(
        id="Viewing_Location",
        desc="Correctly states the conjunction is visible in the western sky after sunset",
        parent=node,
        critical=True
    )
    viewing_claim = "This conjunction is viewed in the western sky after sunset (evening twilight)."
    await evaluator.verify(
        claim=viewing_claim,
        node=viewing_leaf,
        sources=urls,
        additional_instruction="Confirm that guidance on the cited page indicates viewing in the western sky after sunset/evening."
    )

    ref_leaf = evaluator.add_leaf(
        id="Conjunction_Reference_URL",
        desc="A valid reference URL supporting the conjunction information is provided",
        parent=node,
        critical=True
    )
    ref_claim = "This webpage is a relevant reference about the March 2026 Venus–Saturn conjunction (date and viewing guidance)."
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=urls,
        additional_instruction="The page should clearly discuss the Venus–Saturn conjunction and include date and/or viewing details."
    )


async def verify_conference(evaluator: Evaluator, parent_node, conf: Optional[ConferenceInfo]) -> None:
    node = evaluator.add_parallel(
        id="Academic_Conference_Event",
        desc="Correct identification and details of the exoplanet atmospheres conference in March 2026",
        parent=parent_node,
        critical=False
    )

    urls = _filter_urls(conf.urls if conf else [])
    evaluator.add_custom_node(
        result=_nonempty_list(urls),
        id="Conference_Sources_Present",
        desc="At least one reference URL is provided for the conference",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(conf.name if conf else None),
        id="Conference_Name_Provided",
        desc="Conference name is provided in the answer",
        parent=node,
        critical=True
    )
    name_leaf = evaluator.add_leaf(
        id="Conference_Name",
        desc="The conference is correctly identified as 'Exoplanet Atmospheres 2026' or 'AASTCS 11'",
        parent=node,
        critical=True
    )
    name_claim = f"The conference name in the answer corresponds to 'Exoplanet Atmospheres 2026' (AASTCS 11). Provided: '{conf.name if conf else ''}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        additional_instruction="Accept if the provided name matches any alias such as 'AASTCS 11', 'AASTCS 11: Exoplanet Atmospheres', or 'Exoplanet Atmospheres 2026' (case-insensitive, minor variants allowed)."
    )

    evaluator.add_custom_node(
        result=_nonempty(conf.date_range if conf else None) or (_nonempty(conf.start_date if conf else None) and _nonempty(conf.end_date if conf else None)),
        id="Conference_Dates_Provided",
        desc="Conference date range is provided in the answer",
        parent=node,
        critical=True
    )
    dates_leaf = evaluator.add_leaf(
        id="Conference_Dates",
        desc="The conference dates are correctly stated as March 16-20, 2026",
        parent=node,
        critical=True
    )
    if conf and conf.date_range:
        dates_claim = f"The conference runs during {conf.date_range}."
    elif conf and conf.start_date and conf.end_date:
        dates_claim = f"The conference runs from {conf.start_date} to {conf.end_date}."
    else:
        dates_claim = "The conference runs during March 16-20, 2026."
    await evaluator.verify(
        claim=dates_claim,
        node=dates_leaf,
        sources=urls,
        additional_instruction="Confirm that the official conference page lists March 16–20, 2026 (allow hyphen/en-dash variants)."
    )

    evaluator.add_custom_node(
        result=_nonempty(conf.city if conf else None) and _nonempty(conf.state if conf else None),
        id="Conference_Location_Provided",
        desc="Conference location (city and state) is provided in the answer",
        parent=node,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id="Conference_Location",
        desc="The conference location is correctly identified as Denver, Colorado",
        parent=node,
        critical=True
    )
    loc_claim = f"The conference location is Denver, Colorado."  # verify against source
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=urls,
        additional_instruction="Confirm that the conference page lists Denver, Colorado as the location (city and state)."
    )

    ref_leaf = evaluator.add_leaf(
        id="Conference_Reference_URL",
        desc="A valid reference URL supporting the conference information is provided",
        parent=node,
        critical=True
    )
    ref_claim = "This webpage is an official or authoritative reference about the 'Exoplanet Atmospheres' AASTCS 11 conference, including dates and location."
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=urls,
        additional_instruction="The page should clearly be about AASTCS 11: Exoplanet Atmospheres and include date range and/or venue location."
    )


async def verify_equinox(evaluator: Evaluator, parent_node, eq: Optional[EquinoxInfo]) -> None:
    node = evaluator.add_parallel(
        id="Vernal_Equinox_Event",
        desc="Correct identification and details of the vernal equinox in March 2026",
        parent=parent_node,
        critical=False
    )

    urls = _filter_urls(eq.urls if eq else [])
    evaluator.add_custom_node(
        result=_nonempty_list(urls),
        id="Equinox_Sources_Present",
        desc="At least one reference URL is provided for the equinox",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(eq.date if eq else None),
        id="Equinox_Date_Provided",
        desc="Equinox date is provided in the answer",
        parent=node,
        critical=True
    )
    date_leaf = evaluator.add_leaf(
        id="Equinox_Date",
        desc="The vernal equinox date is correctly identified as March 20, 2026",
        parent=node,
        critical=True
    )
    date_claim = f"The March 2026 vernal equinox occurs on {eq.date}." if eq else \
        "The March 2026 vernal equinox occurs on March 20, 2026."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=urls,
        additional_instruction="Confirm the UTC calendar date of the March 2026 northern-hemisphere spring equinox."
    )

    evaluator.add_custom_node(
        result=_nonempty(eq.time_utc if eq else None),
        id="Equinox_Time_Provided",
        desc="Equinox time (UTC) is provided in the answer",
        parent=node,
        critical=True
    )
    time_leaf = evaluator.add_leaf(
        id="Equinox_Time_UTC",
        desc="The equinox time is correctly stated as 14:46 UTC",
        parent=node,
        critical=True
    )
    time_claim = f"The March 2026 vernal equinox occurs at {eq.time_utc} (UTC)." if eq else \
        "The March 2026 vernal equinox occurs at 14:46 UTC."
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        sources=urls,
        additional_instruction="Verify the exact equinox instant in UTC on the cited page. Allow 'UT' vs 'UTC' and inclusion of seconds."
    )

    ref_leaf = evaluator.add_leaf(
        id="Equinox_Reference_URL",
        desc="A valid reference URL supporting the equinox information is provided",
        parent=node,
        critical=True
    )
    ref_claim = "This webpage is a relevant reference that states the March 2026 vernal equinox date and/or UTC time."
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=urls,
        additional_instruction="The page should explicitly provide the equinox date/time in UTC for March 2026."
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root per rubric
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AstronomyMarch2026Extraction,
        extraction_name="astro_march_2026_extraction",
    )

    # Add GT info for transparency
    evaluator.add_ground_truth(
        {
            "expected": GROUND_TRUTH,
            "notes": "Expected values are taken from the rubric. Verification is primarily evidence-based using the URLs provided in the answer."
        },
        gt_type="ground_truth"
    )

    # Build verification subtrees
    await verify_eclipse(evaluator, root, extracted.eclipse if extracted else None)
    await verify_conjunction(evaluator, root, extracted.conjunction if extracted else None)
    await verify_conference(evaluator, root, extracted.conference if extracted else None)
    await verify_equinox(evaluator, root, extracted.equinox if extracted else None)

    return evaluator.get_summary()