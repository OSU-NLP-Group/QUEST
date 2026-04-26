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
TASK_ID = "astro_2026_opportunities"
TASK_DESCRIPTION = (
    "As an astronomy researcher planning your 2026 professional activities, identify five significant opportunities related to space science and astronomical observations:\n\n"
    "1. First Conference (March 2026, exoplanet atmospheres): name, exact dates, location (city, state), primary research focus area, and a reference URL.\n"
    "2. Second Conference (June 2026, AAS meeting): name, exact dates, location (city and specific venue), which AAS divisions are jointly organizing, the date when abstract submissions open, and a reference URL.\n"
    "3. First Lunar Mission (NASA, around the Moon, 2026): mission name, launch timeframe, crew size, names of all crew members, approximate mission duration, and a reference URL.\n"
    "4. Second Lunar Mission (China, mid–late 2026, lunar south pole): mission name, country, launch timeframe, target region, specific crater/feature, primary objective, and a reference URL.\n"
    "5. Major Astronomical Event (early March 2026): event type, exact date, at least three major visibility regions, significance re future total lunar eclipses, and a reference URL.\n"
    "All information must be verifiable via reliable sources (NASA, AAS, space agencies, established astronomy organizations)."
)

# Expected ground-truth-like targets per rubric
EXPECTED = {
    "conference_1": {
        "name": "AASTCS 11: Exoplanet Atmospheres 2026",
        "dates": ("March 16, 2026", "March 20, 2026"),
        "location": ("Denver", "Colorado"),
        "focus": "exoplanet atmospheres",
    },
    "conference_2": {
        "name": "248th AAS Meeting",
        "dates": ("June 14, 2026", "June 18, 2026"),
        "city_state": ("Pasadena", "California"),
        "venue": "Pasadena Convention Center",
        "divisions": ["High Energy Astrophysics Division", "Laboratory Astrophysics Division"],
        "abstract_open": "March 19, 2026",
    },
    "mission_1": {
        "name": "Artemis II",
        "launch_timeframe": "no earlier than April 2026",
        "crew_size": "four",
        "crew_names": ["Reid Wiseman", "Victor Glover", "Christina Koch", "Jeremy Hansen"],
        "duration": "approximately 10 days",
    },
    "mission_2": {
        "name": "Chang'e 7",
        "country": "China",
        "launch_timeframe": "mid to late 2026",
        "target_region": "lunar south pole",
        "specific_location": "near Shackleton Crater rim",
        "objective": "searching for water ice and volatiles",
    },
    "event": {
        "type": "total lunar eclipse",
        "date": "March 3, 2026",
        "significance": "the last total lunar eclipse for nearly 3 years",
    },
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class Conference1Extraction(BaseModel):
    name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    focus: Optional[str] = None
    url: Optional[str] = None


class Conference2Extraction(BaseModel):
    name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    venue: Optional[str] = None
    divisions: List[str] = Field(default_factory=list)
    abstract_open_date: Optional[str] = None
    url: Optional[str] = None


class Mission1Extraction(BaseModel):
    name: Optional[str] = None
    launch_timeframe: Optional[str] = None
    crew_size: Optional[str] = None
    crew_names: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    url: Optional[str] = None


class Mission2Extraction(BaseModel):
    name: Optional[str] = None
    country: Optional[str] = None
    launch_timeframe: Optional[str] = None
    target_region: Optional[str] = None
    specific_location: Optional[str] = None
    objective: Optional[str] = None
    url: Optional[str] = None


class EventExtraction(BaseModel):
    event_type: Optional[str] = None
    date: Optional[str] = None
    visibility_regions: List[str] = Field(default_factory=list)
    significance: Optional[str] = None
    url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_conference_1() -> str:
    return """
    Extract details for the major astronomy conference in March 2026 that focuses on exoplanet atmospheres, as provided in the answer.

    Required fields:
    - name: the conference name exactly as stated in the answer
    - start_date: the start date (include month, day, and year if present)
    - end_date: the end date (include month, day, and year if present)
    - location_city: the city
    - location_state: the state (or region)
    - focus: the primary research focus area (e.g., "exoplanet atmospheres")
    - url: a single reference URL explicitly mentioned in the answer for this conference (must be an actual URL or markdown link's target; if not present, set to null)

    If any field is missing in the answer, set it to null. Do not invent information.
    """


def prompt_extract_conference_2() -> str:
    return """
    Extract details for the major American Astronomical Society (AAS) meeting in June 2026, as provided in the answer.

    Required fields:
    - name: the conference name exactly as stated in the answer
    - start_date: the start date (month, day, year)
    - end_date: the end date (month, day, year)
    - city: the city
    - state: the state (or region)
    - venue: the specific venue (e.g., "Pasadena Convention Center")
    - divisions: an array of AAS divisions jointly organizing this meeting (e.g., "High Energy Astrophysics Division", "Laboratory Astrophysics Division")
    - abstract_open_date: date when abstract submissions open (month, day, year)
    - url: a single reference URL explicitly mentioned in the answer for this meeting

    If any field is missing in the answer, set it to null (or empty array for divisions).
    """


def prompt_extract_mission_1() -> str:
    return """
    Extract details for the NASA lunar mission scheduled to launch in 2026 that will carry astronauts around the Moon (Artemis II), as provided in the answer.

    Required fields:
    - name: the mission name
    - launch_timeframe: e.g., "no earlier than April 2026"
    - crew_size: the crew size as written in the answer (e.g., "four" or "4")
    - crew_names: array of the names of all crew members
    - duration: the approximate mission duration as stated (e.g., "10 days", "approximately 10 days")
    - url: a single reference URL from the answer for this mission (NASA or equivalent)

    If any field is missing in the answer, set it to null (or empty array).
    """


def prompt_extract_mission_2() -> str:
    return """
    Extract details for the Chinese lunar mission scheduled to launch in mid to late 2026 targeting the Moon's south pole (Chang'e 7), as provided in the answer.

    Required fields:
    - name: the mission name
    - country: the operating country
    - launch_timeframe: e.g., "mid to late 2026"
    - target_region: the target region on the Moon (e.g., "lunar south pole")
    - specific_location: the specific feature targeted (e.g., "near Shackleton Crater rim")
    - objective: the primary mission objective
    - url: a single reference URL from the answer for this mission

    If any field is missing in the answer, set it to null.
    """


def prompt_extract_event() -> str:
    return """
    Extract details for the rare total lunar eclipse in early March 2026, as provided in the answer.

    Required fields:
    - event_type: the event type (e.g., "total lunar eclipse")
    - date: the exact date (month, day, year)
    - visibility_regions: array listing major world regions of visibility as stated in the answer
    - significance: a statement of its significance regarding future total lunar eclipses (e.g., "last total lunar eclipse for nearly 3 years")
    - url: a single reference URL from the answer for this event

    If any field is missing in the answer, set it to null (or empty array).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def valid_url(url: Optional[str]) -> bool:
    return isinstance(url, str) and url.strip() != "" and url.strip().lower().startswith(("http://", "https://"))


def list_to_english(lst: List[str]) -> str:
    items = [s.strip() for s in lst if s and s.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_conference_1_checks(evaluator: Evaluator, root):
    node = evaluator.add_parallel(
        id="conference_1",
        desc="Identification and details of the first major astronomy conference in March 2026",
        parent=root,
        critical=False,
    )

    conf: Conference1Extraction = await evaluator.extract(
        prompt=prompt_extract_conference_1(),
        template_class=Conference1Extraction,
        extraction_name="conference_1_extraction",
    )

    # URL existence (critical)
    url_ok = valid_url(conf.url)
    url_node = evaluator.add_custom_node(
        result=url_ok,
        id="conference_1_url",
        desc="Valid reference URL for the conference information is provided",
        parent=node,
        critical=True,
    )

    # Name (critical) - identity check to expected
    name_leaf = evaluator.add_leaf(
        id="conference_1_name",
        desc="Conference name is correctly identified as AASTCS 11: Exoplanet Atmospheres 2026",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The conference name provided ('{conf.name}') refers to 'AASTCS 11: Exoplanet Atmospheres 2026'.",
        node=name_leaf,
        additional_instruction="Judge if the two names denote the same event; allow minor punctuation/casing variations.",
    )

    # Dates (critical) - verify via URL
    dates_leaf = evaluator.add_leaf(
        id="conference_1_dates",
        desc="Conference dates are correctly identified as March 16-20, 2026",
        parent=node,
        critical=True,
    )
    start, end = EXPECTED["conference_1"]["dates"]
    await evaluator.verify(
        claim=f"The conference takes place from {start} to {end}.",
        node=dates_leaf,
        sources=conf.url if url_ok else None,
        additional_instruction="Verify on the official conference page that the date range matches (format variations like en-dash vs 'to' are fine).",
    )

    # Location (critical) - verify via URL
    loc_leaf = evaluator.add_leaf(
        id="conference_1_location",
        desc="Conference location is correctly identified as Denver, Colorado",
        parent=node,
        critical=True,
    )
    city, state = EXPECTED["conference_1"]["location"]
    await evaluator.verify(
        claim=f"The conference location is {city}, {state}.",
        node=loc_leaf,
        sources=conf.url if url_ok else None,
        additional_instruction="Verify the city and state on the conference webpage.",
    )

    # Focus (critical) - verify via URL
    focus_leaf = evaluator.add_leaf(
        id="conference_1_focus",
        desc="Conference focus area is correctly identified as exoplanet atmospheres",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The primary research focus of this conference is exoplanet atmospheres.",
        node=focus_leaf,
        sources=conf.url if url_ok else None,
        additional_instruction="Check the conference description or theme on the official page.",
    )


async def build_conference_2_checks(evaluator: Evaluator, root):
    node = evaluator.add_parallel(
        id="conference_2",
        desc="Identification and details of the second major astronomy conference in June 2026",
        parent=root,
        critical=False,
    )

    conf: Conference2Extraction = await evaluator.extract(
        prompt=prompt_extract_conference_2(),
        template_class=Conference2Extraction,
        extraction_name="conference_2_extraction",
    )

    url_ok = valid_url(conf.url)
    url_node = evaluator.add_custom_node(
        result=url_ok,
        id="conference_2_url",
        desc="Valid reference URL for the conference information is provided",
        parent=node,
        critical=True,
    )

    # Name (critical) - identity vs expected
    name_leaf = evaluator.add_leaf(
        id="conference_2_name",
        desc="Conference name is correctly identified as 248th AAS Meeting",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The conference name provided ('{conf.name}') refers to '248th AAS Meeting'.",
        node=name_leaf,
        additional_instruction="Allow minor formatting differences (e.g., 'AAS 248' vs '248th AAS Meeting') if they clearly refer to the same meeting.",
    )

    # Dates (critical) - via URL
    dates_leaf = evaluator.add_leaf(
        id="conference_2_dates",
        desc="Conference dates are correctly identified as June 14-18, 2026",
        parent=node,
        critical=True,
    )
    start, end = EXPECTED["conference_2"]["dates"]
    await evaluator.verify(
        claim=f"The meeting occurs from {start} to {end}.",
        node=dates_leaf,
        sources=conf.url if url_ok else None,
        additional_instruction="Verify dates on the official AAS meeting page; tolerate formatting variations.",
    )

    # City (critical) - via URL
    city_leaf = evaluator.add_leaf(
        id="conference_2_location_city",
        desc="Conference city is correctly identified as Pasadena, California",
        parent=node,
        critical=True,
    )
    city, state = EXPECTED["conference_2"]["city_state"]
    await evaluator.verify(
        claim=f"The meeting location city is {city}, {state}.",
        node=city_leaf,
        sources=conf.url if url_ok else None,
        additional_instruction="Confirm the city/state on the AAS meeting site.",
    )

    # Venue (critical) - via URL
    venue_leaf = evaluator.add_leaf(
        id="conference_2_venue",
        desc="Conference venue is correctly identified as Pasadena Convention Center",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The specific venue is the {EXPECTED['conference_2']['venue']}.",
        node=venue_leaf,
        sources=conf.url if url_ok else None,
        additional_instruction="Look for the venue on the official meeting logistics or overview page.",
    )

    # Divisions (critical) - via URL
    divisions_leaf = evaluator.add_leaf(
        id="conference_2_divisions",
        desc="Conference joint organization with High Energy Astrophysics Division and Laboratory Astrophysics Division is identified",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This AAS meeting is jointly organized with the AAS High Energy Astrophysics Division (HEAD) and the Laboratory Astrophysics Division (LAD).",
        node=divisions_leaf,
        sources=conf.url if url_ok else None,
        additional_instruction="Confirm both divisions (HEAD and LAD) are explicitly mentioned as part of the meeting.",
    )

    # Abstract open date (critical) - via URL
    abstract_leaf = evaluator.add_leaf(
        id="conference_2_abstract_deadline",
        desc="Abstract submission opening date of March 19, 2026 is correctly identified",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Abstract submissions open on {EXPECTED['conference_2']['abstract_open']}.",
        node=abstract_leaf,
        sources=conf.url if url_ok else None,
        additional_instruction="Specifically verify the 'abstracts open' date (not the close date or deadlines).",
    )


async def build_mission_1_checks(evaluator: Evaluator, root):
    node = evaluator.add_parallel(
        id="space_mission_1",
        desc="Identification and details of the first lunar mission in 2026",
        parent=root,
        critical=False,
    )

    m1: Mission1Extraction = await evaluator.extract(
        prompt=prompt_extract_mission_1(),
        template_class=Mission1Extraction,
        extraction_name="mission_1_extraction",
    )

    url_ok = valid_url(m1.url)
    url_node = evaluator.add_custom_node(
        result=url_ok,
        id="mission_1_url",
        desc="Valid reference URL for the mission information is provided",
        parent=node,
        critical=True,
    )

    # Mission name (critical) - identity vs expected
    name_leaf = evaluator.add_leaf(
        id="mission_1_name",
        desc="Mission name is correctly identified as Artemis II",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The mission name provided ('{m1.name}') refers to 'Artemis II'.",
        node=name_leaf,
        additional_instruction="Allow minor formatting (hyphens/spacing/roman numerals), e.g., 'Artemis 2' vs 'Artemis II', if clearly the same mission.",
    )

    # Launch timeframe (critical) - via URL
    timeframe_leaf = evaluator.add_leaf(
        id="mission_1_launch_timeframe",
        desc="Launch timeframe is correctly identified as no earlier than April 2026",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Artemis II launch is scheduled no earlier than April 2026.",
        node=timeframe_leaf,
        sources=m1.url if url_ok else None,
        additional_instruction="Confirm that NASA (or authoritative source) states 'no earlier than April 2026' (NET April 2026).",
    )

    # Crew size (critical) - via URL
    crew_size_leaf = evaluator.add_leaf(
        id="mission_1_crew_size",
        desc="Crew size is correctly identified as four astronauts",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Artemis II will carry four astronauts.",
        node=crew_size_leaf,
        sources=m1.url if url_ok else None,
        additional_instruction="Verify crew count from the official mission page or press release.",
    )

    # Crew names (critical) - identity vs expected (logical check)
    crew_names_leaf = evaluator.add_leaf(
        id="mission_1_crew_names",
        desc="All four crew member names are correctly identified: Reid Wiseman, Victor Glover, Christina Koch, and Jeremy Hansen",
        parent=node,
        critical=True,
    )
    provided_names = list_to_english(m1.crew_names)
    expected_names = list_to_english(EXPECTED["mission_1"]["crew_names"])
    await evaluator.verify(
        claim=f"The set of crew members listed in the answer ({provided_names}) matches the expected set ({expected_names}).",
        node=crew_names_leaf,
        additional_instruction="Treat sets as unordered; allow minor spelling/formatting differences (e.g., middle initials). Names must correspond to the same four people.",
    )

    # Duration (critical) - via URL
    duration_leaf = evaluator.add_leaf(
        id="mission_1_duration",
        desc="Mission duration is correctly identified as approximately 10 days",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Artemis II mission duration is approximately 10 days.",
        node=duration_leaf,
        sources=m1.url if url_ok else None,
        additional_instruction="Confirm the approximate duration on the authoritative mission page.",
    )


async def build_mission_2_checks(evaluator: Evaluator, root):
    node = evaluator.add_parallel(
        id="space_mission_2",
        desc="Identification and details of the second lunar mission in 2026",
        parent=root,
        critical=False,
    )

    m2: Mission2Extraction = await evaluator.extract(
        prompt=prompt_extract_mission_2(),
        template_class=Mission2Extraction,
        extraction_name="mission_2_extraction",
    )

    url_ok = valid_url(m2.url)
    url_node = evaluator.add_custom_node(
        result=url_ok,
        id="mission_2_url",
        desc="Valid reference URL for the mission information is provided",
        parent=node,
        critical=True,
    )

    # Mission name (critical) - identity vs expected
    name_leaf = evaluator.add_leaf(
        id="mission_2_name",
        desc="Mission name is correctly identified as Chang'e 7",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The mission name provided ('{m2.name}') refers to 'Chang'e 7'.",
        node=name_leaf,
        additional_instruction="Allow minor formatting and ASCII vs Unicode apostrophes; treat as the same mission if unambiguous.",
    )

    # Country (critical) - logical or via URL; use logical identity to expected
    country_leaf = evaluator.add_leaf(
        id="mission_2_country",
        desc="Mission country is correctly identified as China",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The operating country provided ('{m2.country}') is China.",
        node=country_leaf,
        additional_instruction="Treat 'PRC' or 'People's Republic of China' as China.",
    )

    # Launch timeframe (critical) - via URL
    timeframe_leaf = evaluator.add_leaf(
        id="mission_2_launch_timeframe",
        desc="Launch timeframe is correctly identified as mid to late 2026",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Chang'e 7 is planned to launch in mid to late 2026.",
        node=timeframe_leaf,
        sources=m2.url if url_ok else None,
        additional_instruction="Verify timeframe language (e.g., 'mid- to late-2026'); small wording variations are acceptable.",
    )

    # Target region (critical) - via URL
    target_leaf = evaluator.add_leaf(
        id="mission_2_target",
        desc="Mission target is correctly identified as the lunar south pole",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Chang'e 7 targets the lunar south pole region.",
        node=target_leaf,
        sources=m2.url if url_ok else None,
        additional_instruction="Confirm that the target region is the Moon's south pole.",
    )

    # Specific location (critical) - via URL
    specific_loc_leaf = evaluator.add_leaf(
        id="mission_2_specific_location",
        desc="Specific target location is correctly identified as near Shackleton Crater rim",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The specific target area is near the rim of Shackleton Crater.",
        node=specific_loc_leaf,
        sources=m2.url if url_ok else None,
        additional_instruction="Confirm mention of Shackleton Crater rim or an equivalent phrasing.",
    )

    # Objective (critical) - via URL
    objective_leaf = evaluator.add_leaf(
        id="mission_2_primary_objective",
        desc="Primary mission objective is correctly identified as searching for water ice and volatiles",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="A primary mission objective is to search for water ice and other volatiles.",
        node=objective_leaf,
        sources=m2.url if url_ok else None,
        additional_instruction="Check mission science goals/objectives for explicit mention of water ice/volatiles.",
    )


async def build_event_checks(evaluator: Evaluator, root):
    node = evaluator.add_parallel(
        id="astronomical_event",
        desc="Identification and details of the major astronomical event in early 2026",
        parent=root,
        critical=False,
    )

    ev: EventExtraction = await evaluator.extract(
        prompt=prompt_extract_event(),
        template_class=EventExtraction,
        extraction_name="event_extraction",
    )

    url_ok = valid_url(ev.url)
    url_node = evaluator.add_custom_node(
        result=url_ok,
        id="event_url",
        desc="Valid reference URL for the event information is provided",
        parent=node,
        critical=True,
    )

    # Event type (critical) - via URL
    type_leaf = evaluator.add_leaf(
        id="event_type",
        desc="Event type is correctly identified as a total lunar eclipse",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The event is a total lunar eclipse.",
        node=type_leaf,
        sources=ev.url if url_ok else None,
        additional_instruction="Confirm the eclipse type is total (not partial/penumbral).",
    )

    # Event date (critical) - via URL
    date_leaf = evaluator.add_leaf(
        id="event_date",
        desc="Event date is correctly identified as March 3, 2026",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The total lunar eclipse occurs on {EXPECTED['event']['date']}.",
        node=date_leaf,
        sources=ev.url if url_ok else None,
        additional_instruction="Accept UT-based calendar date if consistent; focus on exact day and month/year.",
    )

    # Visibility regions (critical) - via URL
    regions_leaf = evaluator.add_leaf(
        id="event_visibility_regions",
        desc="At least three major visibility regions are correctly identified from: western North America, Australia, New Zealand, East Asia, or Pacific region",
        parent=node,
        critical=True,
    )
    regions_text = list_to_english(ev.visibility_regions[:6])  # avoid overly long claims
    claim_regions = (
        f"The eclipse will be visible from at least these major regions: {regions_text}. "
        "Verify that at least three such major regions are correctly listed per the source."
    )
    await evaluator.verify(
        claim=claim_regions,
        node=regions_leaf,
        sources=ev.url if url_ok else None,
        additional_instruction="Check the visibility map/description; ensure at least three major regions listed in the claim are indeed covered.",
    )

    # Significance (critical) - via URL
    signif_leaf = evaluator.add_leaf(
        id="event_significance",
        desc="Event significance is correctly identified as the last total lunar eclipse for nearly 3 years",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This eclipse is the last total lunar eclipse for nearly 3 years.",
        node=signif_leaf,
        sources=ev.url if url_ok else None,
        additional_instruction="Look for statements about when the next total lunar eclipse will occur and confirm the gap is ~3 years.",
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
    # Initialize evaluator with root node
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

    # Record expected info as ground truth reference
    evaluator.add_ground_truth(
        {
            "conference_1_expected": EXPECTED["conference_1"],
            "conference_2_expected": {
                "name": EXPECTED["conference_2"]["name"],
                "dates": EXPECTED["conference_2"]["dates"],
                "city_state": EXPECTED["conference_2"]["city_state"],
                "venue": EXPECTED["conference_2"]["venue"],
                "divisions": EXPECTED["conference_2"]["divisions"],
                "abstract_open": EXPECTED["conference_2"]["abstract_open"],
            },
            "mission_1_expected": EXPECTED["mission_1"],
            "mission_2_expected": EXPECTED["mission_2"],
            "event_expected": EXPECTED["event"],
        },
        gt_type="expected_targets",
    )

    # Build verification subtrees (can run parts concurrently)
    await asyncio.gather(
        build_conference_1_checks(evaluator, root),
        build_conference_2_checks(evaluator, root),
        build_mission_1_checks(evaluator, root),
        build_mission_2_checks(evaluator, root),
        build_event_checks(evaluator, root),
    )

    # Return standard summary
    return evaluator.get_summary()