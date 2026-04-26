import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs_conf_2025"
TASK_DESCRIPTION = """A researcher is planning their 2025 conference attendance and wants to identify major computer science conferences across different subfields and geographic regions. Identify four major computer science conferences in 2025 that meet the following requirements:

Conference 1: A major Artificial Intelligence conference held in the United States between February and April 2025 at a convention center venue. Provide the conference name, city, venue name, and dates.

Conference 2: A major Computer Vision conference held in the United States between May and July 2025 at a convention center or music center venue (not a hotel). Provide the conference name, city, venue name, and dates.

Conference 3: A major Software Engineering conference held in Norway in June 2025. Provide the conference name, city, venue name, and dates.

Conference 4: A major Human-Computer Interaction conference held in Asia between April and May 2025. Provide the conference name, city, venue name, and dates.

For each conference, provide the following information: full conference name, city and country, venue name, conference dates (start and end dates), and primary research topic focus.
"""

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ConferenceInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    venue_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    topic_focus: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class ConferencesExtraction(BaseModel):
    conference_1: Optional[ConferenceInfo] = None
    conference_2: Optional[ConferenceInfo] = None
    conference_3: Optional[ConferenceInfo] = None
    conference_4: Optional[ConferenceInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
    Extract up to four conferences described in the answer according to the specified constraints in the task, mapping them to Conference 1, Conference 2, Conference 3, and Conference 4 respectively.

    For each conference, extract:
    - name: Full conference name
    - city: City where it is held
    - country: Country where it is held
    - venue_name: The specific venue name
    - start_date: Conference start date as a clear date string (e.g., "March 24, 2025" or "2025-03-24")
    - end_date: Conference end date as a clear date string
    - topic_focus: The primary research topic focus (e.g., Artificial Intelligence, Computer Vision, Software Engineering, Human-Computer Interaction)
    - source_urls: All explicit URLs mentioned in the answer that are relevant to this conference (official site, venue page, Wikipedia page, etc.). If no URLs are provided, return an empty array.

    Return a JSON object with keys:
    - conference_1
    - conference_2
    - conference_3
    - conference_4

    If the answer does not clearly provide information for a given conference, set that conference object to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def is_us_country(country: Optional[str]) -> bool:
    c = _norm(country).lower()
    return c in {
        "united states", "united states of america", "usa", "us", "u.s.", "u.s.a.", "america"
    }


def is_norway_country(country: Optional[str]) -> bool:
    return _norm(country).lower() == "norway"


def asia_countries() -> set:
    return {
        "afghanistan", "armenia", "azerbaijan", "bahrain", "bangladesh", "bhutan", "brunei", "cambodia",
        "china", "cyprus", "georgia", "india", "indonesia", "iran", "iraq", "israel", "japan", "jordan",
        "kazakhstan", "kuwait", "kyrgyzstan", "laos", "lebanon", "malaysia", "maldives", "mongolia",
        "myanmar", "nepal", "north korea", "oman", "pakistan", "palestine", "philippines", "qatar",
        "saudi arabia", "singapore", "south korea", "sri lanka", "syria", "taiwan", "tajikistan",
        "thailand", "timor-leste", "turkey", "turkmenistan", "united arab emirates", "uae", "uzbekistan",
        "vietnam", "yemen", "hong kong", "macau", "macao"
    }


def is_asia_country(country: Optional[str]) -> bool:
    c = _norm(country).lower()
    return c in asia_countries()


def _strip_ordinal(day_str: str) -> str:
    # Remove st/nd/rd/th from day numbers
    import re
    return re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", day_str, flags=re.IGNORECASE)


def parse_date_str(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    txt = _strip_ordinal(_norm(s))
    # Try common patterns
    patterns = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]
    for p in patterns:
        try:
            return datetime.strptime(txt, p).date()
        except Exception:
            continue
    # Fallback: try to detect "Month Day Year" with extra commas/spaces
    try:
        # Example: "March 5 2025" without comma
        return datetime.strptime(txt, "%B %d %Y").date()
    except Exception:
        pass
    try:
        return datetime.strptime(txt, "%b %d %Y").date()
    except Exception:
        pass
    return None


def dates_in_window(start_s: Optional[str], end_s: Optional[str], win_start: date, win_end: date) -> bool:
    start_d = parse_date_str(start_s)
    end_d = parse_date_str(end_s)
    if not start_d or not end_d:
        return False
    # Inclusive window check
    return (win_start <= start_d <= win_end) and (win_start <= end_d <= win_end) and (start_d <= end_d)


def topic_matches(expected_group: str, topic: Optional[str]) -> bool:
    t = _norm(topic).lower()
    if expected_group == "AI":
        synonyms = {"artificial intelligence", "ai"}
    elif expected_group == "CV":
        synonyms = {"computer vision", "cv"}
    elif expected_group == "SE":
        synonyms = {"software engineering", "foundations of software engineering", "fse"}
    elif expected_group == "HCI":
        synonyms = {"human-computer interaction", "hci"}
    else:
        synonyms = {expected_group.lower()}
    return any(s in t for s in synonyms)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_conference_1(evaluator: Evaluator, parent_node, info: Optional[ConferenceInfo]) -> None:
    conf_node = evaluator.add_parallel(
        id="conference_1",
        desc="Conference 1 (AI, US, Feb–Apr 2025, convention center): satisfies all constraints and includes all required details",
        parent=parent_node,
        critical=False
    )

    # c1_name_provided
    evaluator.add_custom_node(
        result=bool(info and _norm(info.name)),
        id="c1_name_provided",
        desc="Provides the full conference name",
        parent=conf_node,
        critical=True
    )

    # c1_location_and_provided
    evaluator.add_custom_node(
        result=bool(info and _norm(info.city) and is_us_country(info.country)),
        id="c1_location_and_provided",
        desc="Provides city and country, and the country is the United States",
        parent=conf_node,
        critical=True
    )

    # c1_dates_and_window
    feb1 = date(2025, 2, 1)
    apr30 = date(2025, 4, 30)
    evaluator.add_custom_node(
        result=bool(info and dates_in_window(info.start_date, info.end_date, feb1, apr30)),
        id="c1_dates_and_window",
        desc="Provides start and end dates, and the dates fall between Feb 1 and Apr 30, 2025",
        parent=conf_node,
        critical=True
    )

    # c1_topic_and_stated
    evaluator.add_custom_node(
        result=bool(info and topic_matches("AI", info.topic_focus)),
        id="c1_topic_and_stated",
        desc="States the primary research topic focus, and it is Artificial Intelligence",
        parent=conf_node,
        critical=True
    )

    # c1_venue_name_provided
    evaluator.add_custom_node(
        result=bool(info and _norm(info.venue_name)),
        id="c1_venue_name_provided",
        desc="Provides the specific venue name",
        parent=conf_node,
        critical=True
    )

    # c1_venue_type_constraint (verification against sources)
    venue_type_node = evaluator.add_leaf(
        id="c1_venue_type_constraint",
        desc="Venue is a convention center (not a hotel or university campus)",
        parent=conf_node,
        critical=True
    )
    venue_name = _norm(info.venue_name) if info else ""
    claim = f"The venue '{venue_name}' is a convention center and not a hotel or a university campus."
    await evaluator.verify(
        claim=claim,
        node=venue_type_node,
        sources=(info.source_urls if info else []),
        additional_instruction="Use the provided URLs (conference site, venue site, Wikipedia, city pages) to confirm the venue classification. Accept if the venue is clearly a convention center; reject if it is a hotel or a university campus."
    )


async def verify_conference_2(evaluator: Evaluator, parent_node, info: Optional[ConferenceInfo]) -> None:
    conf_node = evaluator.add_parallel(
        id="conference_2",
        desc="Conference 2 (Computer Vision, US, May–Jul 2025, convention center or music center): satisfies all constraints and includes all required details",
        parent=parent_node,
        critical=False
    )

    # c2_name_provided
    evaluator.add_custom_node(
        result=bool(info and _norm(info.name)),
        id="c2_name_provided",
        desc="Provides the full conference name",
        parent=conf_node,
        critical=True
    )

    # c2_location_and_provided
    evaluator.add_custom_node(
        result=bool(info and _norm(info.city) and is_us_country(info.country)),
        id="c2_location_and_provided",
        desc="Provides city and country, and the country is the United States",
        parent=conf_node,
        critical=True
    )

    # c2_dates_and_window
    may1 = date(2025, 5, 1)
    jul31 = date(2025, 7, 31)
    evaluator.add_custom_node(
        result=bool(info and dates_in_window(info.start_date, info.end_date, may1, jul31)),
        id="c2_dates_and_window",
        desc="Provides start and end dates, and the dates fall between May 1 and Jul 31, 2025",
        parent=conf_node,
        critical=True
    )

    # c2_topic_and_stated
    evaluator.add_custom_node(
        result=bool(info and topic_matches("CV", info.topic_focus)),
        id="c2_topic_and_stated",
        desc="States the primary research topic focus, and it is Computer Vision",
        parent=conf_node,
        critical=True
    )

    # c2_venue_name_provided
    evaluator.add_custom_node(
        result=bool(info and _norm(info.venue_name)),
        id="c2_venue_name_provided",
        desc="Provides the specific venue name",
        parent=conf_node,
        critical=True
    )

    # c2_venue_type_constraint (verification against sources)
    venue_type_node = evaluator.add_leaf(
        id="c2_venue_type_constraint",
        desc="Venue is a convention center or music center (not a hotel)",
        parent=conf_node,
        critical=True
    )
    venue_name = _norm(info.venue_name) if info else ""
    claim = f"The venue '{venue_name}' is either a convention center or a music center, and it is not a hotel."
    await evaluator.verify(
        claim=claim,
        node=venue_type_node,
        sources=(info.source_urls if info else []),
        additional_instruction="Use the provided URLs to confirm the venue classification. Accept if the venue is a convention center or music center; reject if it is a hotel."
    )


async def verify_conference_3(evaluator: Evaluator, parent_node, info: Optional[ConferenceInfo]) -> None:
    conf_node = evaluator.add_parallel(
        id="conference_3",
        desc="Conference 3 (Software Engineering, Norway, June 2025): satisfies all constraints and includes all required details",
        parent=parent_node,
        critical=False
    )

    # c3_name_provided
    evaluator.add_custom_node(
        result=bool(info and _norm(info.name)),
        id="c3_name_provided",
        desc="Provides the full conference name",
        parent=conf_node,
        critical=True
    )

    # c3_location_and_provided
    evaluator.add_custom_node(
        result=bool(info and _norm(info.city) and is_norway_country(info.country)),
        id="c3_location_and_provided",
        desc="Provides city and country, and the country is Norway",
        parent=conf_node,
        critical=True
    )

    # c3_dates_and_window (must occur in June 2025)
    jun1 = date(2025, 6, 1)
    jun30 = date(2025, 6, 30)
    evaluator.add_custom_node(
        result=bool(info and dates_in_window(info.start_date, info.end_date, jun1, jun30)),
        id="c3_dates_and_window",
        desc="Provides start and end dates, and the conference occurs in June 2025",
        parent=conf_node,
        critical=True
    )

    # c3_topic_and_stated
    evaluator.add_custom_node(
        result=bool(info and topic_matches("SE", info.topic_focus)),
        id="c3_topic_and_stated",
        desc="States the primary research topic focus, and it is Software Engineering or Foundations of Software Engineering",
        parent=conf_node,
        critical=True
    )

    # c3_venue_name_provided
    evaluator.add_custom_node(
        result=bool(info and _norm(info.venue_name)),
        id="c3_venue_name_provided",
        desc="Provides the specific venue name",
        parent=conf_node,
        critical=True
    )


async def verify_conference_4(evaluator: Evaluator, parent_node, info: Optional[ConferenceInfo]) -> None:
    conf_node = evaluator.add_parallel(
        id="conference_4",
        desc="Conference 4 (HCI, Asia, Apr–May 2025): satisfies all constraints and includes all required details",
        parent=parent_node,
        critical=False
    )

    # c4_name_provided
    evaluator.add_custom_node(
        result=bool(info and _norm(info.name)),
        id="c4_name_provided",
        desc="Provides the full conference name",
        parent=conf_node,
        critical=True
    )

    # c4_location_and_provided (country is in Asia)
    evaluator.add_custom_node(
        result=bool(info and _norm(info.city) and is_asia_country(info.country)),
        id="c4_location_and_provided",
        desc="Provides city and country, and the country is in Asia",
        parent=conf_node,
        critical=True
    )

    # c4_dates_and_window (Apr–May 2025)
    apr1 = date(2025, 4, 1)
    may31 = date(2025, 5, 31)
    evaluator.add_custom_node(
        result=bool(info and dates_in_window(info.start_date, info.end_date, apr1, may31)),
        id="c4_dates_and_window",
        desc="Provides start and end dates, and the dates fall between Apr 1 and May 31, 2025",
        parent=conf_node,
        critical=True
    )

    # c4_topic_and_stated
    evaluator.add_custom_node(
        result=bool(info and topic_matches("HCI", info.topic_focus)),
        id="c4_topic_and_stated",
        desc="States the primary research topic focus, and it is Human-Computer Interaction (HCI)",
        parent=conf_node,
        critical=True
    )

    # c4_venue_name_provided
    evaluator.add_custom_node(
        result=bool(info and _norm(info.venue_name)),
        id="c4_venue_name_provided",
        desc="Provides the specific venue name",
        parent=conf_node,
        critical=True
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
    Evaluate an answer for the 2025 CS conferences task.
    """
    # Initialize evaluator
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

    # Extract conference information
    extracted = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferencesExtraction,
        extraction_name="conferences_2025"
    )

    # Build verification according to rubric
    await verify_conference_1(evaluator, root, extracted.conference_1)
    await verify_conference_2(evaluator, root, extracted.conference_2)
    await verify_conference_3(evaluator, root, extracted.conference_3)
    await verify_conference_4(evaluator, root, extracted.conference_4)

    # Return structured summary
    return evaluator.get_summary()