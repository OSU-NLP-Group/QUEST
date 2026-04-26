import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tour_2026_comedy_country"
TASK_DESCRIPTION = (
    "Identify a comedian or country music artist who has a tour scheduled for 2026 that meets ALL of the following "
    "requirements: (1) The tour must be officially announced with specific dates and venues; "
    "(2) Include at least one performance at a major arena venue with a concert capacity of 15,000 or more; "
    "(3) Include performances in at least 3 different U.S. states; "
    "(4) Include at least one venue in a U.S. state that borders the Atlantic Ocean, Pacific Ocean, or Gulf of Mexico; "
    "(5) Include at least one performance scheduled between May 1, 2026 and August 31, 2026; "
    "(6) Include at least one performance at a historically significant venue that has been operating for at least 50 years "
    "(as of 2026, meaning opened in 1976 or earlier); "
    "(7) Include at least one performance scheduled in a U.S. state capital city; "
    "(8) Include performances in at least two different climate zones (both northern/cold-winter states and southern/warm-winter states); "
    "(9) Include at least one performance at an outdoor amphitheater or stadium (not an enclosed indoor arena); "
    "(10) Feature confirmed supporting acts, opening performers, or special guests for at least some dates; "
    "(11) All tour dates must be scheduled in the year 2026. "
    "Provide the name of the performer and their tour, along with specific evidence demonstrating how the tour satisfies each requirement."
)

# --------------------------------------------------------------------------- #
# State utilities                                                             #
# --------------------------------------------------------------------------- #
STATE_ABBR_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
}
STATE_NAME_TO_ABBR = {v.lower(): k for k, v in STATE_ABBR_TO_NAME.items()}

ATLANTIC_STATES = {"CT", "DE", "FL", "GA", "ME", "MD", "MA", "NH", "NJ", "NY", "NC", "RI", "SC", "VA"}
PACIFIC_STATES = {"CA", "OR", "WA", "AK", "HI"}
GULF_STATES = {"FL", "AL", "MS", "LA", "TX"}
COASTAL_STATES = ATLANTIC_STATES | PACIFIC_STATES | GULF_STATES

def normalize_state_to_abbr(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s_clean = s.strip()
    if not s_clean:
        return None
    s_upper = s_clean.upper()
    if s_upper in STATE_ABBR_TO_NAME:
        return s_upper
    s_lower = s_clean.lower()
    if s_lower in STATE_NAME_TO_ABBR:
        return STATE_NAME_TO_ABBR[s_lower]
    # Try to handle cases like "Washington, D.C." (not a state); return None
    return None

def full_state_name(abbr_or_name: Optional[str]) -> Optional[str]:
    if not abbr_or_name:
        return None
    abbr = normalize_state_to_abbr(abbr_or_name)
    if abbr and abbr in STATE_ABBR_TO_NAME:
        return STATE_ABBR_TO_NAME[abbr]
    # Return original cleaned string as fallback
    return abbr_or_name.strip()

def dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        u2 = u.strip()
        if not u2:
            continue
        if u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out

def first_n_unique_states(stops: List["TourStop"], n: int) -> List[Tuple[str, "TourStop"]]:
    seen = set()
    result: List[Tuple[str, TourStop]] = []
    for stop in stops:
        abbr = normalize_state_to_abbr(stop.state)
        if abbr and abbr not in seen and stop.url:
            seen.add(abbr)
            result.append((abbr, stop))
            if len(result) >= n:
                break
    return result

def find_any_coastal_stop(stops: List["TourStop"]) -> Optional["TourStop"]:
    for stop in stops:
        abbr = normalize_state_to_abbr(stop.state)
        if abbr and abbr in COASTAL_STATES and stop.url:
            return stop
    return None

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TourStop(BaseModel):
    date: Optional[str] = None
    venue: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    url: Optional[str] = None

class ArenaVenue(BaseModel):
    venue: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity_evidence_urls: List[str] = Field(default_factory=list)
    event_urls: List[str] = Field(default_factory=list)

class CoastalVenue(BaseModel):
    venue: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    water_body: Optional[str] = None  # Atlantic / Pacific / Gulf of Mexico
    evidence_urls: List[str] = Field(default_factory=list)  # may include event page(s) and state/coast references

class SummerDate(BaseModel):
    date: Optional[str] = None  # Prefer ISO like 2026-06-15 if available
    venue: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    urls: List[str] = Field(default_factory=list)

class HistoricVenue(BaseModel):
    venue: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    opening_year: Optional[str] = None  # string; should be <= 1976
    evidence_urls: List[str] = Field(default_factory=list)  # venue history page / wiki
    event_urls: List[str] = Field(default_factory=list)     # event page showing tour stop at this venue

class GenreEvidence(BaseModel):
    performer_name: Optional[str] = None
    tour_name: Optional[str] = None
    genre: Optional[str] = None  # comedy or country
    genre_source_urls: List[str] = Field(default_factory=list)  # artist page, wiki, authoritative bios
    official_tour_urls: List[str] = Field(default_factory=list)  # official tour page(s) if reiterated here

class CapitalCityVenue(BaseModel):
    venue: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    evidence_urls: List[str] = Field(default_factory=list)  # include event page(s) and capital-status reference(s)

class ClimateEntry(BaseModel):
    region_label: Optional[str] = None  # "northern"/"cold-winter" or "southern"/"warm-winter"
    city: Optional[str] = None
    state: Optional[str] = None
    urls: List[str] = Field(default_factory=list)          # event page(s)
    reference_urls: List[str] = Field(default_factory=list)  # optional general geographic/climate references

class OutdoorVenue(BaseModel):
    venue: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    evidence_urls: List[str] = Field(default_factory=list)  # venue page showing open-air amphitheater/stadium
    event_urls: List[str] = Field(default_factory=list)     # event page for this venue

class SupportingAct(BaseModel):
    act_name: Optional[str] = None
    date: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    urls: List[str] = Field(default_factory=list)  # announcement or ticketing page listing support

class Year2026Evidence(BaseModel):
    schedule_urls: List[str] = Field(default_factory=list)  # authoritative page(s) listing the full schedule

class TourExtraction(BaseModel):
    performer_name: Optional[str] = None
    tour_name: Optional[str] = None
    official_tour_urls: List[str] = Field(default_factory=list)

    tour_stops: List[TourStop] = Field(default_factory=list)

    arena_venue: Optional[ArenaVenue] = None
    coastal_venue: Optional[CoastalVenue] = None
    summer_date: Optional[SummerDate] = None
    historic_venue: Optional[HistoricVenue] = None
    genre_info: Optional[GenreEvidence] = None
    capital_city_venue: Optional[CapitalCityVenue] = None
    climate_diversity: List[ClimateEntry] = Field(default_factory=list)
    outdoor_venue: Optional[OutdoorVenue] = None
    supporting_acts: List[SupportingAct] = Field(default_factory=list)
    year_2026: Optional[Year2026Evidence] = None

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tour() -> str:
    return """
    Extract structured information about a single 2026 tour by a comedian or a country music artist from the answer.
    Only extract information explicitly present in the answer text and the URLs explicitly cited there.

    Return a JSON object matching this schema:
    - performer_name: string or null
    - tour_name: string or null
    - official_tour_urls: array of URLs explicitly cited for the official tour schedule or announcement
    - tour_stops: array of objects, each containing:
        - date: string date as written (prefer ISO like 2026-06-15 if present)
        - venue: venue name
        - city: city name
        - state: full name or 2-letter abbreviation
        - url: URL cited for that event/stop (ticketing, venue page, official schedule, etc.)
    - arena_venue: object or null with:
        - venue, city, state
        - capacity_evidence_urls: array of URLs used in the answer to document that this is a 15,000+ capacity arena
        - event_urls: array of URLs used in the answer to show the tour is scheduled at this venue
    - coastal_venue: object or null with:
        - venue, city, state
        - water_body: "Atlantic", "Pacific", or "Gulf of Mexico" if specified; else null
        - evidence_urls: array of URLs used in the answer to show this stop is in a coastal state (include event page and any coastal reference)
    - summer_date: object or null with:
        - date (should fall between 2026-05-01 and 2026-08-31 if provided in the answer)
        - venue, city, state
        - urls: array of URLs that show the scheduled performance on that date
    - historic_venue: object or null with:
        - venue, city, state
        - opening_year: string with the venue's opening year as cited (1976 or earlier qualifies)
        - evidence_urls: array of URLs proving the opening year/history
        - event_urls: array of URLs showing the tour is scheduled at this venue
    - genre_info: object or null with:
        - performer_name, tour_name, genre ("comedy" or "country" if stated)
        - genre_source_urls: array of URLs proving the performer's genre from authoritative sources
        - official_tour_urls: array of URLs (if repeated here) for the official tour schedule
    - capital_city_venue: object or null with:
        - venue, city, state
        - evidence_urls: array of URLs including the event page AND a source confirming the city is the state capital
    - climate_diversity: array with up to 2 entries (one northern/cold-winter, one southern/warm-winter), each:
        - region_label: "northern"/"cold-winter" or "southern"/"warm-winter" as stated in the answer
        - city, state
        - urls: array of event URLs for that stop
        - reference_urls: array of general reference URLs (if cited) supporting the region/climate classification
    - outdoor_venue: object or null with:
        - venue, city, state
        - evidence_urls: array of URLs proving it is an outdoor amphitheater or open-air stadium
        - event_urls: array of URLs showing the tour is scheduled there
    - supporting_acts: array of objects, each:
        - act_name
        - date, city, state (if specified)
        - urls: array of URLs showing the supporting act/special guest for that date
    - year_2026: object or null with:
        - schedule_urls: array of URLs that (according to the answer) show the schedule and that all dates are in 2026

    Rules:
    - Do NOT invent URLs. Include only URLs that are explicitly present in the answer.
    - If a field is not provided in the answer, set it to null (or an empty array for list fields).
    """

# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _official_urls(ex: TourExtraction) -> List[str]:
    urls = []
    urls.extend(ex.official_tour_urls or [])
    if ex.genre_info:
        urls.extend(ex.genre_info.official_tour_urls or [])
    return dedup_urls(urls)

def _combine(*url_lists: List[str]) -> List[str]:
    out: List[str] = []
    for lst in url_lists:
        out.extend(lst or [])
    return dedup_urls(out)

# --------------------------------------------------------------------------- #
# Verification groups                                                         #
# --------------------------------------------------------------------------- #
async def verify_tour_announcement(evaluator: Evaluator, parent, ex: TourExtraction) -> None:
    group = evaluator.add_parallel(
        id="tour_announcement_verification",
        desc="The tour has been publicly announced with specific dates and venues documented in official sources",
        parent=parent,
        critical=True
    )
    urls = _official_urls(ex)
    exists = bool((ex.performer_name and ex.performer_name.strip()) and urls)
    evaluator.add_custom_node(
        result=exists,
        id="official_tour_docs_exist",
        desc="Official tour documentation URLs provided",
        parent=group,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="official_tour_documentation",
        desc="Provide URL to official tour page, Ticketmaster listing, or verified tour announcement showing scheduled dates and venues",
        parent=group,
        critical=True
    )
    performer = ex.performer_name or "the performer"
    tour = ex.tour_name or "the 2026 tour"
    claim = (
        f"At least one cited page is an official or authoritative tour announcement/schedule for {performer}'s {tour}. "
        f"It explicitly lists specific dates and venue names for 2026."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="Accept official artist website, Ticketmaster/AXS/Live Nation, or official social media announcement that explicitly lists date(s) and venue(s). Pages with only vague 'coming soon' info should not pass."
    )

async def verify_arena_capacity(evaluator: Evaluator, parent, ex: TourExtraction) -> None:
    group = evaluator.add_parallel(
        id="arena_venue_requirement",
        desc="The tour includes at least one performance at a major arena venue with concert capacity of 15,000 or more",
        parent=parent,
        critical=True
    )
    av = ex.arena_venue
    exists = bool(av and av.venue and dedup_urls(av.capacity_evidence_urls))
    evaluator.add_custom_node(
        result=exists,
        id="arena_capacity_info_exists",
        desc="Arena venue and 15,000+ capacity evidence URLs provided",
        parent=group,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="arena_capacity_verification",
        desc="Provide venue name with documented capacity of 15,000+ and URL source confirming this capacity",
        parent=group,
        critical=True
    )
    venue_name = av.venue if av and av.venue else "the venue"
    claim = f"The venue {venue_name} has a concert/event seating capacity of at least 15,000."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=av.capacity_evidence_urls if av else [],
        additional_instruction="Use the venue's official page, Wikipedia, or credible venue databases that state capacity. Accept 'seating capacity' or 'concert capacity' statements ≥ 15,000."
    )

async def verify_multi_state_coverage(evaluator: Evaluator, parent, ex: TourExtraction) -> None:
    group = evaluator.add_parallel(
        id="multi_state_geographic_coverage",
        desc="The tour includes performances scheduled in at least 3 different U.S. states",
        parent=parent,
        critical=True
    )
    picks = first_n_unique_states(ex.tour_stops or [], 3)
    exists = bool(len(picks) >= 3 and all(stop.url for _, stop in picks))
    evaluator.add_custom_node(
        result=exists,
        id="state_count_exist",
        desc="At least 3 distinct states with event URLs were identified from the answer",
        parent=group,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="state_count_verification",
        desc="List at least 3 different U.S. states where tour performances are scheduled with URL verification",
        parent=group,
        critical=True
    )
    states = [full_state_name(abbr) or abbr for abbr, _ in picks]
    urls = [stop.url for _, stop in picks if stop.url]
    claim = (
        f"The tour includes scheduled performances in at least three different U.S. states: {', '.join(states)}. "
        f"Each cited page shows a scheduled date and venue in the stated state."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="Verify each page clearly shows an event in the specified state. If any page is unrelated or not an event page, this should fail."
    )

async def verify_coastal_state_requirement(evaluator: Evaluator, parent, ex: TourExtraction) -> None:
    group = evaluator.add_parallel(
        id="coastal_state_requirement",
        desc="At least one tour venue is located in a U.S. state that borders the Atlantic Ocean, Pacific Ocean, or Gulf of Mexico",
        parent=parent,
        critical=True
    )
    cv = ex.coastal_venue
    chosen_stop = None
    chosen_urls: List[str] = []
    body = None

    if cv and cv.state and (cv.evidence_urls or []):
        chosen_stop = TourStop(date=None, venue=cv.venue, city=cv.city, state=cv.state, url=(cv.evidence_urls[0] if cv.evidence_urls else None))
        chosen_urls = dedup_urls(cv.evidence_urls)
        body = cv.water_body
    else:
        # Fallback: find any coastal stop from provided tour stops
        guess = find_any_coastal_stop(ex.tour_stops or [])
        if guess:
            chosen_stop = guess
            chosen_urls = dedup_urls([guess.url] if guess.url else [])

    exists = bool(chosen_stop and chosen_stop.state and chosen_urls)
    evaluator.add_custom_node(
        result=exists,
        id="coastal_venue_exists",
        desc="A coastal-state tour stop and supporting URL(s) are provided",
        parent=group,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="coastal_venue_documentation",
        desc="Identify the coastal state venue with URL confirmation of its location in an ocean/Gulf-bordering state",
        parent=group,
        critical=True
    )
    state_name = full_state_name(chosen_stop.state) if chosen_stop and chosen_stop.state else "a coastal state"
    venue_name = chosen_stop.venue if chosen_stop and chosen_stop.venue else "the venue"
    city_name = chosen_stop.city if chosen_stop and chosen_stop.city else "the city"
    body_label = body if body else "an ocean or the Gulf of Mexico"
    claim = (
        f"The tour includes a performance at {venue_name} in {city_name}, {state_name}; "
        f"{state_name} borders {body_label}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=chosen_urls,
        additional_instruction="The evidence must show the event is in the specified state. If an additional cited page is present confirming the state's coastline (e.g., state page), use it as well."
    )

async def verify_summer_2026_timing(evaluator: Evaluator, parent, ex: TourExtraction) -> None:
    group = evaluator.add_parallel(
        id="summer_2026_timing",
        desc="The tour includes at least one performance scheduled between May 1, 2026 and August 31, 2026",
        parent=parent,
        critical=True
    )
    sd = ex.summer_date
    exists = bool(sd and sd.date and dedup_urls(sd.urls))
    evaluator.add_custom_node(
        result=exists,
        id="summer_date_exists",
        desc="A summer 2026 date and supporting URL(s) are provided",
        parent=group,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="summer_date_confirmation",
        desc="Provide specific date(s) between May 1-Aug 31, 2026 with URL verification of scheduled performance(s)",
        parent=group,
        critical=True
    )
    when = sd.date if sd and sd.date else "the specified date"
    claim = (
        f"The cited page(s) show a scheduled performance on {when}, and that date falls between May 1, 2026 and August 31, 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sd.urls if sd else [],
        additional_instruction="Check that the page shows the specific date and confirm it falls within the stated 2026 summer window."
    )

async def verify_historic_venue(evaluator: Evaluator, parent, ex: TourExtraction) -> None:
    group = evaluator.add_parallel(
        id="historic_venue_requirement",
        desc="At least one tour venue is a historically significant performance space that has been operating for at least 50 years as of 2026",
        parent=parent,
        critical=True
    )
    hv = ex.historic_venue
    exists = bool(hv and hv.venue and hv.opening_year and dedup_urls(hv.evidence_urls))
    evaluator.add_custom_node(
        result=exists,
        id="historic_venue_exists",
        desc="Historic venue, opening year, and evidence URL(s) are provided",
        parent=group,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="historic_venue_documentation",
        desc="Identify venue with opening year of 1976 or earlier and provide URL confirming operational history",
        parent=group,
        critical=True
    )
    venue_name = hv.venue if hv and hv.venue else "the venue"
    opening_year = hv.opening_year if hv and hv.opening_year else "the stated year"
    claim = (
        f"The venue {venue_name} has an opening year of {opening_year}, which is 1976 or earlier (i.e., at least 50 years old as of 2026)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=hv.evidence_urls if hv else [],
        additional_instruction="Verify the opening year on the cited venue history or Wikipedia page. The opening year must be 1976 or earlier."
    )

async def verify_genre_specification(evaluator: Evaluator, parent, ex: TourExtraction) -> None:
    group = evaluator.add_parallel(
        id="genre_specification",
        desc="The performer is either a comedian or a country music artist",
        parent=parent,
        critical=True
    )
    gi = ex.genre_info
    exists = bool(gi and gi.performer_name and gi.genre and dedup_urls(gi.genre_source_urls))
    evaluator.add_custom_node(
        result=exists,
        id="genre_info_exists",
        desc="Performer name, genre, and evidence URL(s) are provided",
        parent=group,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="genre_classification_verification",
        desc="Provide evidence and URL confirming performer's primary genre as comedy or country music",
        parent=group,
        critical=True
    )
    performer = gi.performer_name if gi and gi.performer_name else (ex.performer_name or "the performer")
    genre = gi.genre if gi and gi.genre else "the specified genre"
    claim = f"The performer {performer} is primarily a {genre} performer (comedy or country)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=gi.genre_source_urls if gi else [],
        additional_instruction="Use authoritative sources (official bio, Wikipedia, major publications). Ensure the genre clearly matches comedy or country."
    )

async def verify_state_capital_city(evaluator: Evaluator, parent, ex: TourExtraction) -> None:
    group = evaluator.add_parallel(
        id="state_capital_city_venue",
        desc="At least one tour performance is scheduled at a venue located in a U.S. state capital city",
        parent=parent,
        critical=True
    )
    cc = ex.capital_city_venue
    exists = bool(cc and cc.city and cc.state and dedup_urls(cc.evidence_urls))
    evaluator.add_custom_node(
        result=exists,
        id="capital_city_evidence_exists",
        desc="Capital city stop and capital-status evidence URL(s) are provided",
        parent=group,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="capital_city_confirmation",
        desc="Identify the state capital city and venue with URL verification of tour date in that capital",
        parent=group,
        critical=True
    )
    city = cc.city if cc and cc.city else "the city"
    state = full_state_name(cc.state) if cc and cc.state else "the state"
    venue = cc.venue if cc and cc.venue else "the venue"
    claim = f"There is a scheduled tour performance at {venue} in {city}, {state}, and {city} is the capital city of {state}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=cc.evidence_urls if cc else [],
        additional_instruction="The evidence should include (1) an event/venue/ticket page showing the tour stop in the city, and (2) a reliable source confirming the city is the state capital."
    )

async def verify_climate_zone_diversity(evaluator: Evaluator, parent, ex: TourExtraction) -> None:
    group = evaluator.add_parallel(
        id="climate_zone_diversity",
        desc="The tour includes performances at venues in at least two different climate zones, representing both northern/cold-winter and southern/warm-winter regions",
        parent=parent,
        critical=True
    )
    north = None
    south = None
    for entry in ex.climate_diversity or []:
        label = (entry.region_label or "").lower()
        if any(k in label for k in ["northern", "cold"]):
            if not north and entry.urls:
                north = entry
        if any(k in label for k in ["southern", "warm"]):
            if not south and entry.urls:
                south = entry
    exists = bool(north and south)
    evaluator.add_custom_node(
        result=exists,
        id="climate_entries_exist",
        desc="At least one northern/cold-winter and one southern/warm-winter stop provided with event URL(s)",
        parent=group,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="geographic_climate_verification",
        desc="Identify venues in at least one northern state and one southern state with URL documentation of tour dates",
        parent=group,
        critical=True
    )
    north_state = full_state_name(north.state) if north and north.state else "a northern/cold-winter state"
    south_state = full_state_name(south.state) if south and south.state else "a southern/warm-winter state"
    urls = _combine(north.urls if north else [], south.urls if south else [], north.reference_urls if north else [], south.reference_urls if south else [])
    claim = (
        f"The tour includes performances in at least two climate zones: one in a northern/cold-winter state ({north_state}) "
        f"and one in a southern/warm-winter state ({south_state})."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="Use the event pages to verify the states. If classification references are cited, also use them."
    )

async def verify_outdoor_venue(evaluator: Evaluator, parent, ex: TourExtraction) -> None:
    group = evaluator.add_parallel(
        id="outdoor_venue_requirement",
        desc="At least one tour performance is scheduled at an outdoor amphitheater or stadium (not an enclosed indoor arena)",
        parent=parent,
        critical=True
    )
    ov = ex.outdoor_venue
    exists = bool(ov and ov.venue and dedup_urls(ov.evidence_urls))
    evaluator.add_custom_node(
        result=exists,
        id="outdoor_venue_exists",
        desc="Outdoor amphitheater/stadium and evidence URL(s) are provided",
        parent=group,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="outdoor_venue_documentation",
        desc="Identify outdoor venue and provide URL confirming it is an open-air amphitheater or stadium",
        parent=group,
        critical=True
    )
    venue_name = ov.venue if ov and ov.venue else "the venue"
    claim = (
        f"The venue {venue_name} is an outdoor amphitheater or open-air stadium (not an enclosed indoor arena), "
        f"and the tour schedules a performance there."
    )
    urls = _combine(ov.evidence_urls if ov else [], ov.event_urls if ov else [])
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="The venue's official page or a reliable source should clearly indicate it is open-air/outdoor. Include an event page if available to confirm the tour stop."
    )

async def verify_supporting_acts(evaluator: Evaluator, parent, ex: TourExtraction) -> None:
    group = evaluator.add_parallel(
        id="supporting_acts_requirement",
        desc="The tour features supporting acts, opening performers, or special guests confirmed for at least some tour dates",
        parent=parent,
        critical=True
    )
    sa = None
    for s in ex.supporting_acts or []:
        if s and s.act_name and dedup_urls(s.urls):
            sa = s
            break
    exists = bool(sa)
    evaluator.add_custom_node(
        result=exists,
        id="supporting_act_exists",
        desc="At least one supporting act/special guest with evidence URL(s) is provided",
        parent=group,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="supporting_acts_confirmation",
        desc="Identify at least one confirmed supporting act or special guest with URL verification from tour announcement",
        parent=group,
        critical=True
    )
    act = sa.act_name if sa and sa.act_name else "the supporting act"
    date_info = f" on {sa.date}" if sa and sa.date else ""
    where_info = ""
    if sa and (sa.city or sa.state):
        city = sa.city or ""
        state = full_state_name(sa.state) or (sa.state or "")
        loc = ", ".join([p for p in [city, state] if p])
        if loc:
            where_info = f" in {loc}"
    claim = f"The tour includes {act} as a supporting act or special guest{date_info}{where_info}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sa.urls if sa else [],
        additional_instruction="The cited page should explicitly mention the opening act/supporting act/special guest for the specified date(s)."
    )

async def verify_year_2026_only(evaluator: Evaluator, parent, ex: TourExtraction) -> None:
    group = evaluator.add_parallel(
        id="year_2026_scheduling",
        desc="All identified tour dates are scheduled to occur in the year 2026",
        parent=parent,
        critical=True
    )
    y = ex.year_2026
    schedule_urls = dedup_urls((y.schedule_urls if y else []) or _official_urls(ex))
    exists = bool(schedule_urls)
    evaluator.add_custom_node(
        result=exists,
        id="schedule_urls_exist",
        desc="Official schedule/announcement URL(s) provided for verifying all dates are in 2026",
        parent=group,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="year_2026_verification",
        desc="Confirm tour dates fall within 2026 calendar year with URL showing date schedule",
        parent=group,
        critical=True
    )
    claim = "All tour dates listed on this official schedule are in the calendar year 2026 (no dates in 2025 or 2027)."
    # Prefer single authoritative schedule page for 'all' verification
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=schedule_urls[0] if schedule_urls else None,
        additional_instruction="Scan the schedule for any dates outside of 2026. If any appear (e.g., 2025 or 2027), the claim is incorrect."
    )

# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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

    # Extract structured tour info
    extraction: TourExtraction = await evaluator.extract(
        prompt=prompt_extract_tour(),
        template_class=TourExtraction,
        extraction_name="tour_extraction"
    )

    # Build verification tree according to rubric
    # 1. Tour announcement
    await verify_tour_announcement(evaluator, root, extraction)

    # 2. Arena capacity >= 15,000
    await verify_arena_capacity(evaluator, root, extraction)

    # 3. Multi-state coverage (>=3 states)
    await verify_multi_state_coverage(evaluator, root, extraction)

    # 4. Coastal state requirement
    await verify_coastal_state_requirement(evaluator, root, extraction)

    # 5. Summer 2026 timing (May 1 to Aug 31 inclusive)
    await verify_summer_2026_timing(evaluator, root, extraction)

    # 6. Historic venue (opened <= 1976)
    await verify_historic_venue(evaluator, root, extraction)

    # 7. Genre specification (comedian or country)
    await verify_genre_specification(evaluator, root, extraction)

    # 8. State capital city venue
    await verify_state_capital_city(evaluator, root, extraction)

    # 9. Climate zone diversity (northern + southern)
    await verify_climate_zone_diversity(evaluator, root, extraction)

    # 10. Outdoor amphitheater/stadium
    await verify_outdoor_venue(evaluator, root, extraction)

    # 11. All dates in 2026
    await verify_year_2026_only(evaluator, root, extraction)

    return evaluator.get_summary()