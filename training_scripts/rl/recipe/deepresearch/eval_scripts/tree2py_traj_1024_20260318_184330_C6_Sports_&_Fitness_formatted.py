import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "corporate_event_stadiums_2026"
TASK_DESCRIPTION = """
I am organizing a series of high-profile corporate events during the week of April 18-27, 2026, and need to identify three NFL stadiums in the United States that can accommodate large-scale gatherings with premium hospitality. For each stadium, please provide:

1. The stadium name and its seating capacity
2. Confirmation that the stadium has either a retractable roof OR is located in a city where the average April temperature exceeds 60°F (15.6°C)
3. Verification that the stadium offers luxury suites with a minimum capacity of 20 guests per suite and has at least 100 luxury suites available
4. Confirmation that the stadium is accessible via direct public rail transit (metro, light rail, or subway) with a station within 0.5 miles of the venue
5. Verification that the stadium meets ADA accessibility requirements, specifically providing accessible parking spaces for at least 2% of total parking capacity
6. Confirmation that the stadium will NOT be hosting the 2026 NFL Draft (Pittsburgh, April 23-25) or any other major sporting events during April 18-27, 2026
7. A direct link to the stadium's official website that verifies these specifications

The three stadiums must each have a seating capacity of at least 70,000. Exclude any stadiums located in Pittsburgh, Pennsylvania.
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class StadiumEntry(BaseModel):
    # Identification and location
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None

    # Capacity
    seating_capacity: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)

    # NFL status
    nfl_team: Optional[str] = None
    nfl_status_sources: List[str] = Field(default_factory=list)

    # Roof / Climate
    roof_type: Optional[str] = None  # e.g., retractable, fixed, open-air
    roof_sources: List[str] = Field(default_factory=list)
    april_avg_temp_f: Optional[str] = None  # string form, e.g., "62°F" or "62"
    climate_sources: List[str] = Field(default_factory=list)

    # Suites / Premium
    suite_min_capacity: Optional[str] = None  # e.g., "20", "20+"
    suite_count: Optional[str] = None  # e.g., "120"
    suites_sources: List[str] = Field(default_factory=list)

    # Transit
    transit_description: Optional[str] = None
    transit_sources: List[str] = Field(default_factory=list)

    # ADA parking
    ada_parking_pct: Optional[str] = None  # e.g., "2%", "at least 2%"
    ada_sources: List[str] = Field(default_factory=list)

    # Events schedule (no conflicts Apr 18–27, 2026)
    events_conflict_statement: Optional[str] = None  # e.g., "No major sporting events"
    events_sources: List[str] = Field(default_factory=list)

    # Official website (verification link)
    official_url: Optional[str] = None


class StadiumsExtraction(BaseModel):
    stadiums: List[StadiumEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadiums() -> str:
    return """
    Extract all distinct NFL stadium entries the answer proposes (aim for the first 5 if more are present; we will later evaluate only the first three). For each stadium, return:

    Identification and location:
    - name: Stadium name (string)
    - city: City (string or null)
    - state: State (string or null)
    - country: Country (string or null, should be "United States" if stated)

    Capacity:
    - seating_capacity: The seating capacity value mentioned (string; keep number as-is)
    - capacity_sources: Array of URLs cited for capacity

    NFL status:
    - nfl_team: The NFL team associated with this stadium (string or null)
    - nfl_status_sources: Array of URLs cited confirming it is an NFL stadium/home venue

    Roof / Climate:
    - roof_type: "retractable", "fixed", "open-air", or other description from the answer (string or null)
    - roof_sources: Array of URLs cited for roof information
    - april_avg_temp_f: The average April temperature for the city in °F as provided in the answer (string, e.g., "62" or "62°F"; null if not provided)
    - climate_sources: Array of URLs cited for climate data

    Suites / Premium:
    - suite_min_capacity: Minimum capacity per luxury suite (string or null; e.g., "20")
    - suite_count: Total number of luxury suites available (string or null; e.g., "120")
    - suites_sources: Array of URLs cited for suites info

    Transit:
    - transit_description: The claimed rail transit station and walking distance info (string or null)
    - transit_sources: Array of URLs cited for transit info

    ADA Parking:
    - ada_parking_pct: The percentage of accessible parking vs total (string or null; e.g., "2%")
    - ada_sources: Array of URLs cited for ADA parking info

    Events Schedule:
    - events_conflict_statement: A plain statement whether the stadium has NO major sporting events scheduled during April 18–27, 2026 (string or null)
    - events_sources: Array of URLs cited for event schedule

    Official website:
    - official_url: A direct URL to the stadium's official website (team/stadium operated) that helps verify the specifications (string or null)

    RULES:
    - Only extract URLs explicitly present in the answer; do not invent any link.
    - If a field is not present in the answer, set it to null (or empty array for URLs).
    - Deduplicate repeated stadiums by name (case-insensitive); keep the first occurrence.
    - Preserve the original formatting of string values; avoid normalizing numbers.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return f"{n}{'tsnrhtdd'[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4]}"  # neat ordinal trick


def pick_sources(st: StadiumEntry, kind: str) -> List[str]:
    urls: List[str] = []
    if kind == "nfl":
        urls = st.nfl_status_sources or []
    elif kind == "capacity":
        urls = st.capacity_sources or []
    elif kind == "location":
        # Prefer explicit location sources; fallback to official site
        urls = st.capacity_sources + st.nfl_status_sources + st.transit_sources + st.ada_sources + st.climate_sources
        # If answer gave specific location sources, they should be within the above collections
    elif kind == "roof_or_temp":
        urls = (st.roof_sources or []) + (st.climate_sources or [])
    elif kind == "suites":
        urls = st.suites_sources or []
    elif kind == "transit":
        urls = st.transit_sources or []
    elif kind == "ada":
        urls = st.ada_sources or []
    elif kind == "events":
        urls = st.events_sources or []
    elif kind == "official":
        urls = [st.official_url] if st.official_url else []
    # Always use official_url as fallback to help ground claims when other sources are empty
    if (not urls) and st.official_url:
        urls = [st.official_url]
    # Dedup while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u and u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Stadium verification routine                                                #
# --------------------------------------------------------------------------- #
async def verify_one_stadium(evaluator: Evaluator, parent_node, st: StadiumEntry, index: int) -> None:
    # Parent bucket for this stadium
    stadium_node = evaluator.add_parallel(
        id=f"Stadium_{index+1}",
        desc=f"{ordinal(index+1)} qualifying NFL stadium",
        parent=parent_node,
        critical=False
    )

    # Core requirement gate (all must pass)
    req_node = evaluator.add_parallel(
        id=f"Stadium_{index+1}_Meets_All_Requirements",
        desc=f"Stadium {index+1} satisfies all required constraints and required provided fields.",
        parent=stadium_node,
        critical=True
    )

    # 1) Is NFL stadium
    leaf = evaluator.add_leaf(
        id=f"Stadium_{index+1}_Is_NFL_Stadium",
        desc=f"Stadium {index+1} is an NFL stadium (used as an NFL team home stadium / NFL venue).",
        parent=req_node,
        critical=True
    )
    name_text = st.name or "the stadium"
    claim = f"{name_text} is an NFL stadium that serves as a home venue for an NFL team or is an established NFL game venue."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=pick_sources(st, "nfl"),
        additional_instruction="Confirm on the provided webpages that this venue is an NFL stadium (home stadium or primary NFL venue)."
    )

    # 2) Name & seating capacity provided (existence check)
    evaluator.add_custom_node(
        result=nonempty(st.name) and nonempty(st.seating_capacity),
        id=f"Stadium_{index+1}_Name_And_Seating_Capacity_Provided",
        desc=f"Stadium {index+1} name and a seating capacity value are provided.",
        parent=req_node,
        critical=True
    )

    # 3) Located in the United States
    leaf = evaluator.add_leaf(
        id=f"Stadium_{index+1}_Located_In_United_States",
        desc=f"Stadium {index+1} is located in the United States.",
        parent=req_node,
        critical=True
    )
    loc_phrase = ""
    if nonempty(st.city) or nonempty(st.state):
        loc_city = st.city or ""
        loc_state = st.state or ""
        loc_phrase = f" (located in {loc_city}, {loc_state})"
    claim = f"{name_text} is located in the United States{loc_phrase}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=pick_sources(st, "location"),
        additional_instruction="Use the page to confirm the city/state are in the USA. If the page clearly shows a US address or city/state, mark as supported."
    )

    # 4) Seating capacity at least 70,000
    leaf = evaluator.add_leaf(
        id=f"Stadium_{index+1}_Seating_Capacity_At_Least_70000",
        desc=f"Stadium {index+1} seating capacity is at least 70,000.",
        parent=req_node,
        critical=True
    )
    claim = f"The seating capacity of {name_text} is at least 70,000."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=pick_sources(st, "capacity"),
        additional_instruction="Check the page for capacity. Consider 'seating capacity' wording or equivalent. If multiple capacities are listed, use the standard or maximum stated by the stadium."
    )

    # 5) Not in Pittsburgh, PA
    leaf = evaluator.add_leaf(
        id=f"Stadium_{index+1}_Not_Located_In_Pittsburgh_PA",
        desc=f"Stadium {index+1} is not located in Pittsburgh, Pennsylvania.",
        parent=req_node,
        critical=True
    )
    city_state = f"{(st.city or '').strip()}, {(st.state or '').strip()}".strip(", ")
    neg_note = f" It is located in {city_state}." if city_state.strip(", ") else ""
    claim = f"{name_text} is not located in Pittsburgh, Pennsylvania.{neg_note}"
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=pick_sources(st, "location"),
        additional_instruction="Confirm the stadium's city/state. If the page indicates a city/state other than Pittsburgh, PA, consider the claim supported."
    )

    # 6) Retractable roof OR April avg temp > 60°F
    leaf = evaluator.add_leaf(
        id=f"Stadium_{index+1}_Retractable_Roof_OR_April_Avg_Temp_Above_60F",
        desc=f"Stadium {index+1} has a retractable roof OR is in a city where April average temperature exceeds 60°F.",
        parent=req_node,
        critical=True
    )
    rt = st.roof_type or "(roof type unspecified)"
    temp_info = st.april_avg_temp_f or "(temperature unspecified)"
    claim = f"At least one of the following is true for {name_text}: (1) the stadium has a retractable roof; OR (2) the city's average April temperature exceeds 60°F. Roof: {rt}. April avg temp: {temp_info}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=pick_sources(st, "roof_or_temp"),
        additional_instruction="Accept as supported if EITHER the roof is explicitly 'retractable' OR a reliable source shows the city's April average temperature > 60°F (15.6°C)."
    )

    # 7) Suites min 20 guests per suite
    leaf = evaluator.add_leaf(
        id=f"Stadium_{index+1}_Luxury_Suites_Min_20_Guests",
        desc=f"Stadium {index+1} offers luxury suites with a minimum capacity of 20 guests per suite.",
        parent=req_node,
        critical=True
    )
    claim = f"{name_text} offers luxury suites with a minimum capacity of at least 20 guests per suite."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=pick_sources(st, "suites"),
        additional_instruction="Look for premium/luxury suites pages specifying suite capacity per suite. If the minimum capacity shown is 20 or more, mark supported."
    )

    # 8) Suites count at least 100
    leaf = evaluator.add_leaf(
        id=f"Stadium_{index+1}_Luxury_Suites_At_Least_100",
        desc=f"Stadium {index+1} has at least 100 luxury suites available.",
        parent=req_node,
        critical=True
    )
    claim = f"{name_text} has at least 100 luxury suites."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=pick_sources(st, "suites"),
        additional_instruction="Verify that the total number of suites is 100 or more. Accept if the page states a count >= 100."
    )

    # 9) Direct public rail within 0.5 miles
    leaf = evaluator.add_leaf(
        id=f"Stadium_{index+1}_Public_Rail_Within_0_5_Miles",
        desc=f"Stadium {index+1} is accessible via direct public rail with a station within 0.5 miles.",
        parent=req_node,
        critical=True
    )
    tr_desc = st.transit_description or "(no distance stated)"
    claim = f"{name_text} has direct public rail access (metro, light rail, or subway) with a station within 0.5 miles of the venue. Detail: {tr_desc}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=pick_sources(st, "transit"),
        additional_instruction="Confirm an official transit guide or rail map/page shows a station within approximately 0.5 miles (about 10-minute walk). Allow small rounding differences."
    )

    # 10) ADA accessible parking >= 2% of total
    leaf = evaluator.add_leaf(
        id=f"Stadium_{index+1}_ADA_Accessible_Parking_At_Least_2pct",
        desc=f"Stadium {index+1} provides accessible parking spaces for at least 2% of total parking capacity.",
        parent=req_node,
        critical=True
    )
    ada_txt = st.ada_parking_pct or "(percentage unspecified)"
    claim = f"{name_text} provides ADA-accessible parking spaces accounting for at least 2% of total parking capacity. Extracted value: {ada_txt}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=pick_sources(st, "ada"),
        additional_instruction="Look for ADA parking policies or counts. Accept if the page states at least 2% of total parking spaces are accessible or indicates compliance that implies meeting this threshold."
    )

    # 11) No major sporting events Apr 18–27, 2026
    leaf = evaluator.add_leaf(
        id=f"Stadium_{index+1}_No_Major_Sporting_Events_Apr_18_27_2026",
        desc=f"Stadium {index+1} will NOT be hosting any major sporting events during April 18–27, 2026.",
        parent=req_node,
        critical=True
    )
    claim = f"During April 18–27, 2026 (inclusive), {name_text} does not host any major sporting events."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=pick_sources(st, "events"),
        additional_instruction="Check official event calendar(s) or announcements. Consider 'major sporting events' as professional or high-profile college sports (e.g., NFL, MLB, MLS, NCAA tournaments). Ignore concerts/non-sport events."
    )

    # 12) Official website link verifies specifications
    leaf = evaluator.add_leaf(
        id=f"Stadium_{index+1}_Official_Website_Link_Verifies_Specs",
        desc=f"A direct link to Stadium {index+1}'s official website is provided and it contains evidence supporting the stated specifications.",
        parent=req_node,
        critical=True
    )
    claim = f"The provided URL is the official website for {name_text} (team/stadium operated) and includes information corroborating key specifications (e.g., seating capacity and/or premium suites details, amenities, accessibility, or transit)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=pick_sources(st, "official"),
        additional_instruction="First, confirm the URL is an official team/stadium site. Then, verify it contains concrete info corroborating at least core specs such as capacity and/or premium suites/amenities or accessibility/transit."
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator (root should be non-critical to allow partial credit aggregation)
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

    # Extract proposed stadiums from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_stadiums(),
        template_class=StadiumsExtraction,
        extraction_name="stadiums_extraction",
    )

    # Normalize list and keep as provided
    provided = [s for s in (extraction.stadiums or []) if s and nonempty(s.name)]
    first_three: List[StadiumEntry] = provided[:3]
    provided_count = len(provided)

    # Global critical checks (under root)
    evaluator.add_custom_node(
        result=(provided_count == 3),
        id="Exactly_Three_Stadiums_Provided",
        desc="The response provides exactly three stadium entries.",
        parent=root,
        critical=True,
    )

    # Distinctness among the first three (only meaningful if we have three)
    names_lower = [s.name.strip().lower() for s in first_three if s.name]
    distinct_ok = (len(first_three) == 3) and (len(set(names_lower)) == 3)
    evaluator.add_custom_node(
        result=distinct_ok,
        id="Stadiums_Are_Distinct",
        desc="All three selected stadiums are distinct venues (no duplicates).",
        parent=root,
        critical=True,
    )

    # Build per-stadium verification subtrees (always create three buckets for transparency)
    tasks = []
    for i in range(3):
        st = first_three[i] if i < len(first_three) else StadiumEntry()
        tasks.append(verify_one_stadium(evaluator, root, st, i))
    await asyncio.gather(*tasks)

    # Return final structured summary
    return evaluator.get_summary()