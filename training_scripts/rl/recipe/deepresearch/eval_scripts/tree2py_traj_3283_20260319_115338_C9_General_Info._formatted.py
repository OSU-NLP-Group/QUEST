import asyncio
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "entertainment_multi_domain_2026"
TASK_DESCRIPTION = """
You are conducting a comprehensive entertainment industry research report covering multiple domains in 2026. Your task is to identify and provide detailed information about the following four items:

Item 1 - Houston Rodeo Country Performer:
Identify a country music performer who took the stage at the Houston Livestock Show and Rodeo during the March 2-22, 2026 period. Provide:
- The performer's full name
- The exact date (month and day) of their performance
- The name of the venue where RodeoHouston 2026 performances took place
- The city and state location of this venue
- Confirmation that this performer is classified as a country music artist

Item 2 - International Tour Venue:
Identify a specific arena or concert venue outside the United States where Hilary Duff performed during her "Lucky Me Tour" in 2026 or 2027. Provide:
- The tour name confirmation
- The venue name
- The city and country where this venue is located
- The specific date (month, day, and year) when Hilary Duff performed at this venue
- Confirmation that this venue is in one of the international countries on her tour (UK, Ireland, Australia, New Zealand, Mexico, or Canada)

Item 3 - Golden Globes Drama Actress Winner:
Identify the winner of the Best Female Actor in a Motion Picture - Drama award at the 83rd Golden Globes ceremony in 2026. Provide:
- The actress's full name
- The title of the film for which she won
- The date when the 83rd Golden Globes ceremony took place
- The ceremony location (venue and city)
- Confirmation that the film was in the Drama category (not Musical/Comedy)

Item 4 - Shark Tank Investment Success:
Identify Barbara Corcoran's most successful investment deal from her time as a Shark Tank investor. Provide:
- Confirmation of Barbara Corcoran's status as a Shark Tank investor and her net worth as of 2026
- The name of the company that represents her most successful investment
- The initial dollar amount she invested in this company
- The total return amount she earned from this investment
- The equity stake percentage she received (if available)
- The timeframe over which these returns were generated

For each item, provide reference URLs that support your answers.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HoustonRodeoExtraction(BaseModel):
    performer_name: Optional[str] = None
    performer_urls: List[str] = Field(default_factory=list)

    performance_date: Optional[str] = None  # e.g., "March 9, 2026" or "March 9"
    performance_date_urls: List[str] = Field(default_factory=list)

    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_state: Optional[str] = None
    venue_urls: List[str] = Field(default_factory=list)

    genre_label: Optional[str] = None  # e.g., "country", "country pop"
    genre_urls: List[str] = Field(default_factory=list)


class HilaryDuffTourExtraction(BaseModel):
    tour_name: Optional[str] = None
    tour_urls: List[str] = Field(default_factory=list)

    venue_name: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    venue_urls: List[str] = Field(default_factory=list)

    performance_date: Optional[str] = None  # month day, year (2026 or 2027)
    performance_date_urls: List[str] = Field(default_factory=list)


class GoldenGlobesExtraction(BaseModel):
    actress_name: Optional[str] = None
    winner_urls: List[str] = Field(default_factory=list)

    film_title: Optional[str] = None
    film_urls: List[str] = Field(default_factory=list)

    ceremony_date: Optional[str] = None
    ceremony_venue: Optional[str] = None
    ceremony_city: Optional[str] = None
    ceremony_urls: List[str] = Field(default_factory=list)

    category_name: Optional[str] = None
    category_urls: List[str] = Field(default_factory=list)


class SharkTankExtraction(BaseModel):
    investor_name: Optional[str] = None
    investor_urls: List[str] = Field(default_factory=list)

    net_worth: Optional[str] = None
    net_worth_urls: List[str] = Field(default_factory=list)

    company_name: Optional[str] = None
    company_urls: List[str] = Field(default_factory=list)

    initial_investment_amount: Optional[str] = None
    investment_amount_urls: List[str] = Field(default_factory=list)

    equity_stake: Optional[str] = None
    equity_urls: List[str] = Field(default_factory=list)

    total_return_amount: Optional[str] = None
    return_amount_urls: List[str] = Field(default_factory=list)

    returns_timeframe: Optional[str] = None
    timeframe_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_houston_rodeo() -> str:
    return """
    From the answer, extract one country music performer who performed at the Houston Livestock Show and Rodeo (RodeoHouston) during March 2–22, 2026, and all required details.

    Required fields:
    - performer_name: Full name exactly as written in the answer.
    - performer_urls: An array of all URLs cited that confirm this performer's participation in RodeoHouston 2026 (official schedule, lineup page, news, etc.).
    - performance_date: The specific performance date text as written (e.g., "March 9, 2026" or "March 9").
    - performance_date_urls: An array of URLs confirming the specific date for this performer at RodeoHouston 2026.
    - venue_name: The venue name (expected: NRG Stadium) as written in the answer.
    - venue_city: City name (expected: Houston).
    - venue_state: State name (expected: Texas).
    - venue_urls: An array of URLs confirming that RodeoHouston 2026 shows were at NRG Stadium and/or that this performer’s show took place there.
    - genre_label: The genre label used to classify the performer (e.g., "country", "country pop").
    - genre_urls: An array of URLs confirming the performer's country genre classification.

    Rules:
    - Only extract URLs explicitly present in the answer.
    - If any field is missing, set it to null; for URL arrays, return an empty array if none are present.
    """.strip()


def prompt_extract_hilary_duff() -> str:
    return """
    From the answer, extract one international (non-U.S.) venue from Hilary Duff’s "Lucky Me Tour" (2026–2027) and the required details.

    Required fields:
    - tour_name: The tour name text as written (e.g., "Lucky Me Tour", "the lucky me tour").
    - tour_urls: URLs confirming the tour name and timeframe.
    - venue_name: The selected international venue name.
    - city: The city where this venue is located.
    - country: The country where this venue is located.
    - venue_urls: URLs confirming Hilary Duff performed at this venue on this tour and/or the venue’s location.
    - performance_date: The specific performance date (month day, year) for this venue as written (must be 2026 or 2027).
    - performance_date_urls: URLs confirming the specific date for this venue on the Lucky Me Tour.

    Rules:
    - Only extract URLs explicitly present in the answer.
    - If any field is missing, set it to null; for URL arrays, return an empty array if none are present.
    """.strip()


def prompt_extract_golden_globes() -> str:
    return """
    From the answer, extract the winner of Best Female Actor in a Motion Picture – Drama at the 83rd Golden Globes (2026) and all related details.

    Required fields:
    - actress_name: The winner's full name.
    - winner_urls: URLs confirming the winner in this category at the 83rd Golden Globes (2026).
    - film_title: The film for which she won.
    - film_urls: URLs confirming the film title and category association for this win.
    - ceremony_date: The ceremony date text as written (expected: "January 11, 2026").
    - ceremony_venue: Venue name text as written (expected: "The Beverly Hilton").
    - ceremony_city: City/region text as written (expected: "Beverly Hills, California").
    - ceremony_urls: URLs confirming the ceremony date and location.
    - category_name: The award category text as written (expected wording variants like "Best Female Actor in a Motion Picture – Drama" or "Best Actress in a Motion Picture – Drama").
    - category_urls: URLs confirming the category name/context for the win.

    Rules:
    - Only extract URLs explicitly present in the answer.
    - If any field is missing, set it to null; for URL arrays, return an empty array if none are present.
    """.strip()


def prompt_extract_sharktank() -> str:
    return """
    From the answer, extract details about Barbara Corcoran's most successful Shark Tank investment.

    Required fields:
    - investor_name: The investor’s full name (expected: "Barbara Corcoran").
    - investor_urls: URLs confirming she is a Shark Tank investor.
    - net_worth: The net worth amount text as written (expected: around "$100 million") with "as of 2026" context if provided.
    - net_worth_urls: URLs confirming the net worth figure as of 2026 (or clearly indicating 2026 context).
    - company_name: The company cited as her most successful Shark Tank investment.
    - company_urls: URLs supporting that this is her most successful deal and/or that she invested in it.
    - initial_investment_amount: The initial investment amount she put into this company (text as written, e.g., "$50,000").
    - investment_amount_urls: URLs confirming this initial investment amount.
    - equity_stake: The equity stake percentage she received (text as written, if provided; else null).
    - equity_urls: URLs confirming the equity stake (if provided; else empty array).
    - total_return_amount: The total return amount she earned from this investment (text as written).
    - return_amount_urls: URLs confirming the total return amount (not just revenue).
    - returns_timeframe: The timeframe over which returns were generated (text as written, if provided; else null).
    - timeframe_urls: URLs confirming the timeframe (if provided; else empty array).

    Rules:
    - Only extract URLs explicitly present in the answer.
    - If any field is missing, set it to null; for URL arrays, return an empty array if none are present.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(*args: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in args:
        for u in lst or []:
            if isinstance(u, str):
                url = u.strip()
                if url and url not in seen:
                    seen.add(url)
                    out.append(url)
    return out


def _coalesce(*vals: Optional[str]) -> str:
    for v in vals:
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""


def _strip_ordinals(s: str) -> str:
    # remove "st", "nd", "rd", "th" after day numbers
    import re
    return re.sub(r'(\d+)(st|nd|rd|th)', r'\1', s, flags=re.IGNORECASE)


def _parse_date_guess_2026(date_text: Optional[str]) -> Optional[date]:
    if not date_text:
        return None
    s = _strip_ordinals(date_text.strip())
    candidates = [
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%B %d",
        "%b %d",
        "%m/%d",
    ]
    # Try direct parsing with year
    for fmt in candidates[:6]:
        try:
            dt = datetime.strptime(s, fmt)
            # normalize two-digit years to 2026 if ambiguous
            year = dt.year
            if year < 100:
                year += 2000
            return date(year, dt.month, dt.day)
        except Exception:
            pass
    # Try without year and assume 2026
    for fmt in candidates[6:]:
        try:
            dt = datetime.strptime(s, fmt)
            return date(2026, dt.month, dt.day)
        except Exception:
            pass
    return None


def _in_march_range_2026(d: Optional[date]) -> bool:
    if d is None:
        return False
    lo = date(2026, 3, 2)
    hi = date(2026, 3, 22)
    return lo <= d <= hi


def _equals_ci(a: Optional[str], b: Optional[str]) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()


def _contains_ci(hay: Optional[str], needle: str) -> bool:
    return needle.lower() in (hay or "").lower()


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_houston_rodeo(evaluator: Evaluator, parent) -> None:
    item_node = evaluator.add_parallel(
        id="Houston_Rodeo_2026_Country_Performer",
        desc="Identify a country music performer who performed at the Houston Livestock Show and Rodeo in March 2026, with date, venue, location, and genre confirmation.",
        parent=parent,
        critical=False
    )

    # Retrieve extraction result
    ex: HoustonRodeoExtraction = evaluator._extraction_results[-4]["houston_rodeo"] if False else None  # placeholder to appease linters
    # Instead of accessing internal logs, find the actual extraction from recorded extractions
    houston_ex: Optional[HoustonRodeoExtraction] = None
    for rec in evaluator._extraction_results:
        if "houston_rodeo" in rec:
            houston_ex = HoustonRodeoExtraction(**rec["houston_rodeo"])
            break

    # Guard if not extracted
    if houston_ex is None:
        houston_ex = HoustonRodeoExtraction()

    # Performer Identity (critical group)
    perf_group = evaluator.add_parallel(
        id="Performer_Identity",
        desc="Provide the full name of a country music artist who performed at RodeoHouston during March 2–22, 2026.",
        parent=item_node,
        critical=True
    )

    perf_exists = evaluator.add_custom_node(
        result=bool(_coalesce(houston_ex.performer_name)),
        id="Performer_Name_Present",
        desc="Performer name is provided.",
        parent=perf_group,
        critical=True
    )

    # Verify performer appears in official lineup/schedule
    perf_lineup_leaf = evaluator.add_leaf(
        id="Performer_Name_Verification",
        desc="The performer name matches one listed in the official 2026 RodeoHouston lineup/schedule.",
        parent=perf_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official RodeoHouston 2026 lineup or schedule includes the performer named '{_coalesce(houston_ex.performer_name)}'.",
        node=perf_lineup_leaf,
        sources=_normalize_urls(houston_ex.performer_urls, houston_ex.performance_date_urls, houston_ex.venue_urls),
        additional_instruction="Accept lineup/schedule pages, official RodeoHouston site, or credible news recaps explicitly listing the performer for 2026."
    )

    perf_url_support = evaluator.add_leaf(
        id="Performer_Reference_URL",
        desc="Provide a reference URL confirming the performer's participation in RodeoHouston 2026.",
        parent=perf_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cited source(s) explicitly confirm that '{_coalesce(houston_ex.performer_name)}' performed at RodeoHouston in 2026.",
        node=perf_url_support,
        sources=_normalize_urls(houston_ex.performer_urls, houston_ex.performance_date_urls),
        additional_instruction="The page should clearly state the artist performed at RodeoHouston 2026 (not a different year)."
    )

    # Performance Date (critical group)
    date_group = evaluator.add_parallel(
        id="Performance_Date",
        desc="Provide the exact date when this performer took the stage at RodeoHouston 2026.",
        parent=item_node,
        critical=True
    )

    date_present = evaluator.add_custom_node(
        result=bool(_coalesce(houston_ex.performance_date)),
        id="Date_Present",
        desc="A specific performance date is provided.",
        parent=date_group,
        critical=True
    )

    # Check date within March 2–22, 2026
    parsed = _parse_date_guess_2026(houston_ex.performance_date)
    within_range = _in_march_range_2026(parsed)
    evaluator.add_custom_node(
        result=within_range,
        id="Date_Within_Event_Period",
        desc="The performance date falls within March 2–22, 2026.",
        parent=date_group,
        critical=True
    )

    # Verify date matches official schedule
    date_match_leaf = evaluator.add_leaf(
        id="Date_Match_Official_Schedule",
        desc="The date matches the official RodeoHouston 2026 entertainment schedule for this performer.",
        parent=date_group,
        critical=True
    )
    pretty_date = (
        parsed.strftime("%B %-d, %Y") if (parsed and hasattr(parsed, "strftime")) else _coalesce(houston_ex.performance_date)
    )
    # Windows compatibility: %-d not supported; fallback
    try:
        pretty_date = parsed.strftime("%B %-d, %Y") if parsed else _coalesce(houston_ex.performance_date)
    except Exception:
        pretty_date = parsed.strftime("%B %d, %Y") if parsed else _coalesce(houston_ex.performance_date)

    await evaluator.verify(
        claim=f"{_coalesce(houston_ex.performer_name)} performed at RodeoHouston on {pretty_date if pretty_date else 'the specified date'} (2026).",
        node=date_match_leaf,
        sources=_normalize_urls(houston_ex.performance_date_urls, houston_ex.performer_urls),
        additional_instruction="Verify the schedule page for the artist/date or credible coverage explicitly stating the artist's RodeoHouston 2026 date."
    )

    # Venue Name and Location (critical group)
    venue_group = evaluator.add_parallel(
        id="Venue_Name_And_Location",
        desc="Provide the name of the venue (NRG Stadium) and the city/state (Houston, Texas).",
        parent=item_node,
        critical=True
    )

    venue_present = evaluator.add_custom_node(
        result=bool(_coalesce(houston_ex.venue_name)),
        id="Venue_Name_Present",
        desc="Venue name is provided.",
        parent=venue_group,
        critical=True
    )

    venue_name_leaf = evaluator.add_leaf(
        id="Venue_Name_Correct",
        desc="The venue name must be NRG Stadium.",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"RodeoHouston 2026 concerts took place at NRG Stadium.",
        node=venue_name_leaf,
        sources=_normalize_urls(houston_ex.venue_urls, houston_ex.performance_date_urls),
        additional_instruction="The source should indicate RodeoHouston's concert venue is NRG Stadium (not NRG Arena)."
    )

    venue_loc_leaf = evaluator.add_leaf(
        id="Location_Correct",
        desc="The location must be Houston, Texas.",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="NRG Stadium is located in Houston, Texas.",
        node=venue_loc_leaf,
        sources=_normalize_urls(houston_ex.venue_urls),
        additional_instruction="Accept official stadium page or Wikipedia/credible pages clearly indicating Houston, Texas."
    )

    # Genre Classification (critical single leaf)
    genre_group = evaluator.add_parallel(
        id="Genre_Classification_Group",
        desc="Confirm country genre classification.",
        parent=item_node,
        critical=True
    )

    genre_present = evaluator.add_custom_node(
        result=bool(_coalesce(houston_ex.genre_label)),
        id="Genre_Label_Present",
        desc="A genre label is provided.",
        parent=genre_group,
        critical=True
    )

    genre_leaf = evaluator.add_leaf(
        id="Genre_Classification",
        desc="Confirm that the identified performer is classified as a country music artist.",
        parent=genre_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_coalesce(houston_ex.performer_name)} is a country music artist.",
        node=genre_leaf,
        sources=_normalize_urls(houston_ex.genre_urls, houston_ex.performer_urls),
        additional_instruction="If the artist is a crossover (e.g., country-pop), it's acceptable as 'country' classification."
    )


async def verify_hilary_duff(evaluator: Evaluator, parent) -> None:
    item_node = evaluator.add_parallel(
        id="International_Tour_Venue_2026",
        desc="Identify an international venue where Hilary Duff performed during her Lucky Me Tour (2026 or 2027), with date and location.",
        parent=parent,
        critical=False
    )

    # Load extraction
    hil_ex: Optional[HilaryDuffTourExtraction] = None
    for rec in evaluator._extraction_results:
        if "hilary_duff_tour" in rec:
            hil_ex = HilaryDuffTourExtraction(**rec["hilary_duff_tour"])
            break
    if hil_ex is None:
        hil_ex = HilaryDuffTourExtraction()

    # Artist/Tour confirmation (critical)
    tour_group = evaluator.add_parallel(
        id="Artist_Confirmation",
        desc="Confirm Hilary Duff's Lucky Me Tour (2026–2027).",
        parent=item_node,
        critical=True
    )

    tour_present = evaluator.add_custom_node(
        result=bool(_coalesce(hil_ex.tour_name)),
        id="Tour_Name_Present",
        desc="Tour name provided.",
        parent=tour_group,
        critical=True
    )

    tour_leaf = evaluator.add_leaf(
        id="Tour_Name_Verification",
        desc="The tour name must be 'the lucky me tour' or 'Lucky Me Tour'.",
        parent=tour_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The tour is called 'Lucky Me Tour' (case-insensitive).",
        node=tour_leaf,
        sources=_normalize_urls(hil_ex.tour_urls),
        additional_instruction="Treat 'the lucky me tour' and 'Lucky Me Tour' as equivalent; verify it's Hilary Duff's tour."
    )

    tour_ref_leaf = evaluator.add_leaf(
        id="Tour_Reference_URL",
        desc="Provide a reference URL confirming Hilary Duff's Lucky Me Tour.",
        parent=tour_group,
        critical=True
    )
    await evaluator.verify(
        claim="The cited source(s) confirm Hilary Duff's Lucky Me Tour and its timeframe (2026–2027).",
        node=tour_ref_leaf,
        sources=_normalize_urls(hil_ex.tour_urls),
        additional_instruction="Accept official announcements, artist site, reputable news, or Wikipedia with correct tour info."
    )

    # Venue identification (critical)
    venue_group = evaluator.add_parallel(
        id="Venue_Identification",
        desc="Provide an international venue where Hilary Duff performed on this tour.",
        parent=item_node,
        critical=True
    )

    venue_present = evaluator.add_custom_node(
        result=bool(_coalesce(hil_ex.venue_name)),
        id="Venue_Name_Present_Intl",
        desc="Venue name provided.",
        parent=venue_group,
        critical=True
    )

    venue_verify_leaf = evaluator.add_leaf(
        id="Venue_Name_Verification",
        desc="The venue name must match one listed in the official Lucky Me Tour schedule for international dates.",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Hilary Duff performed at '{_coalesce(hil_ex.venue_name)}' during the Lucky Me Tour.",
        node=venue_verify_leaf,
        sources=_normalize_urls(hil_ex.venue_urls, hil_ex.performance_date_urls, hil_ex.tour_urls),
        additional_instruction="Source should explicitly list this venue on the tour schedule or credible coverage of the specific tour stop."
    )

    # Country verification (critical in JSON). Also check it's one of allowed.
    allowed_countries = {"united kingdom", "uk", "ireland", "australia", "new zealand", "mexico", "canada"}
    in_allowed = _coalesce(hil_ex.country).lower() in allowed_countries
    evaluator.add_custom_node(
        result=in_allowed,
        id="Venue_Country_Verification",
        desc="The venue is in one of: UK, Ireland, Australia, New Zealand, Mexico, or Canada.",
        parent=venue_group,
        critical=True
    )

    # Location details (critical)
    loc_group = evaluator.add_parallel(
        id="Location_Details",
        desc="Provide the city and country where this venue is located.",
        parent=item_node,
        critical=True
    )

    city_leaf = evaluator.add_leaf(
        id="City_Name_Correct",
        desc="The city name matches the actual location of the identified venue.",
        parent=loc_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{_coalesce(hil_ex.venue_name)}' is located in the city of '{_coalesce(hil_ex.city)}'.",
        node=city_leaf,
        sources=_normalize_urls(hil_ex.venue_urls),
        additional_instruction="Verify the venue's official page, Wikipedia, or reputable listing indicating the city."
    )

    country_leaf = evaluator.add_leaf(
        id="Country_Name_Correct",
        desc="The country matches the actual location of the identified venue.",
        parent=loc_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{_coalesce(hil_ex.venue_name)}' is located in the country '{_coalesce(hil_ex.country)}'.",
        node=country_leaf,
        sources=_normalize_urls(hil_ex.venue_urls),
        additional_instruction="Verify the venue's official page, Wikipedia, or reputable listing indicating the country."
    )

    # Performance date (critical)
    date_group = evaluator.add_parallel(
        id="Performance_Date_International",
        desc="Provide the specific date (month day, year) when Hilary Duff performed at this venue.",
        parent=item_node,
        critical=True
    )

    int_date_present = evaluator.add_custom_node(
        result=bool(_coalesce(hil_ex.performance_date)),
        id="Intl_Date_Present",
        desc="International performance date is provided.",
        parent=date_group,
        critical=True
    )

    int_date_match_leaf = evaluator.add_leaf(
        id="Date_Match_Tour_Schedule",
        desc="The performance date matches the official Lucky Me Tour schedule for this venue.",
        parent=date_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Hilary Duff performed at '{_coalesce(hil_ex.venue_name)}' on {_coalesce(hil_ex.performance_date)} as part of the Lucky Me Tour.",
        node=int_date_match_leaf,
        sources=_normalize_urls(hil_ex.performance_date_urls, hil_ex.venue_urls, hil_ex.tour_urls),
        additional_instruction="Verify this exact date appears on the tour schedule or credible coverage for this specific venue."
    )


async def verify_golden_globes(evaluator: Evaluator, parent) -> None:
    item_node = evaluator.add_parallel(
        id="Golden_Globes_2026_Drama_Winner",
        desc="Identify the 83rd Golden Globes (2026) Drama actress winner, film, ceremony date, and location.",
        parent=parent,
        critical=False
    )

    # Load extraction
    gg_ex: Optional[GoldenGlobesExtraction] = None
    for rec in evaluator._extraction_results:
        if "golden_globes_2026" in rec:
            gg_ex = GoldenGlobesExtraction(**rec["golden_globes_2026"])
            break
    if gg_ex is None:
        gg_ex = GoldenGlobesExtraction()

    # Winner identity (critical)
    win_group = evaluator.add_parallel(
        id="Winner_Identity",
        desc="Provide the full name of the Drama actress winner at the 83rd Golden Globes (2026).",
        parent=item_node,
        critical=True
    )

    win_present = evaluator.add_custom_node(
        result=bool(_coalesce(gg_ex.actress_name)),
        id="Actress_Name_Present",
        desc="Actress name is provided.",
        parent=win_group,
        critical=True
    )

    win_leaf = evaluator.add_leaf(
        id="Actress_Name_Verification",
        desc="The actress name must match the official 2026 Golden Globes winner for this category.",
        parent=win_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_coalesce(gg_ex.actress_name)} won Best Female Actor in a Motion Picture – Drama at the 83rd Golden Globes (2026).",
        node=win_leaf,
        sources=_normalize_urls(gg_ex.winner_urls, gg_ex.category_urls),
        additional_instruction="Accept equivalent naming variants like 'Best Actress in a Motion Picture – Drama' or 'Best Performance by Female Actor in a Motion Picture – Drama'."
    )

    win_ref_leaf = evaluator.add_leaf(
        id="Winner_Reference_URL",
        desc="Provide a reference URL confirming this actress as the 2026 Golden Globes winner in this category.",
        parent=win_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cited source(s) explicitly list {_coalesce(gg_ex.actress_name)} as the 2026 winner of the Drama Motion Picture actress category.",
        node=win_ref_leaf,
        sources=_normalize_urls(gg_ex.winner_urls),
        additional_instruction="Prefer official Golden Globes, HFPA, or reputable coverage summarizing winners."
    )

    # Winning film (critical)
    film_group = evaluator.add_parallel(
        id="Winning_Film",
        desc="Provide the title of the film for which the actress won.",
        parent=item_node,
        critical=True
    )

    film_present = evaluator.add_custom_node(
        result=bool(_coalesce(gg_ex.film_title)),
        id="Film_Title_Present",
        desc="Film title is provided.",
        parent=film_group,
        critical=True
    )

    film_title_leaf = evaluator.add_leaf(
        id="Film_Title_Verification",
        desc="The film title must match the work for which the actress won the 2026 Golden Globe.",
        parent=film_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_coalesce(gg_ex.actress_name)} won for the film '{_coalesce(gg_ex.film_title)}' at the 83rd Golden Globes.",
        node=film_title_leaf,
        sources=_normalize_urls(gg_ex.film_urls, gg_ex.winner_urls),
        additional_instruction="The page should clearly connect the actress with the specific winning film for the Drama category."
    )

    film_genre_leaf = evaluator.add_leaf(
        id="Film_Genre_Confirmation",
        desc="The film must have been nominated in the Drama category (not Musical/Comedy).",
        parent=film_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The film '{_coalesce(gg_ex.film_title)}' competed in the Drama category at the 83rd Golden Globes (not Musical/Comedy).",
        node=film_genre_leaf,
        sources=_normalize_urls(gg_ex.film_urls, gg_ex.category_urls, gg_ex.winner_urls),
        additional_instruction="Look for category listings clearly indicating 'Drama'."
    )

    # Ceremony date and location (critical)
    cer_group = evaluator.add_parallel(
        id="Ceremony_Date",
        desc="Provide the date (expected: January 11, 2026) and confirm the location at The Beverly Hilton, Beverly Hills, California.",
        parent=item_node,
        critical=True
    )

    cer_date_present = evaluator.add_custom_node(
        result=bool(_coalesce(gg_ex.ceremony_date)),
        id="Ceremony_Date_Present",
        desc="Ceremony date is provided.",
        parent=cer_group,
        critical=True
    )

    cer_date_leaf = evaluator.add_leaf(
        id="Date_Accuracy",
        desc="The ceremony date must be January 11, 2026.",
        parent=cer_group,
        critical=True
    )
    await evaluator.verify(
        claim="The 83rd Golden Globes ceremony took place on January 11, 2026.",
        node=cer_date_leaf,
        sources=_normalize_urls(gg_ex.ceremony_urls),
        additional_instruction="Confirm the actual ceremony date (not nominations or announcements)."
    )

    cer_loc_leaf = evaluator.add_leaf(
        id="Ceremony_Location",
        desc="The ceremony took place at The Beverly Hilton in Beverly Hills, California.",
        parent=cer_group,
        critical=True
    )
    await evaluator.verify(
        claim="The 83rd Golden Globes ceremony was held at The Beverly Hilton in Beverly Hills, California.",
        node=cer_loc_leaf,
        sources=_normalize_urls(gg_ex.ceremony_urls),
        additional_instruction="Accept official Golden Globes site or reputable coverage that states venue and city."
    )

    cer_ref_leaf = evaluator.add_leaf(
        id="Ceremony_Reference_URL",
        desc="Provide a reference URL confirming the ceremony date and location.",
        parent=cer_group,
        critical=True
    )
    await evaluator.verify(
        claim="The cited source(s) explicitly state both the January 11, 2026 date and The Beverly Hilton, Beverly Hills location for the ceremony.",
        node=cer_ref_leaf,
        sources=_normalize_urls(gg_ex.ceremony_urls),
        additional_instruction="Source should include both date and location together or clearly on the same page."
    )

    # Category confirmation (critical)
    cat_leaf = evaluator.add_leaf(
        id="Award_Category_Confirmation",
        desc="Confirm that the category is Best Female Actor in a Motion Picture – Drama (not Musical/Comedy, not Supporting).",
        parent=item_node,
        critical=True
    )
    await evaluator.verify(
        claim="The award category is Best Female Actor in a Motion Picture – Drama (equivalently referred to as Best Actress in a Motion Picture – Drama), not Musical/Comedy and not Supporting.",
        node=cat_leaf,
        sources=_normalize_urls(gg_ex.category_urls, gg_ex.winner_urls),
        additional_instruction="Allow common naming variations but ensure it's the lead Drama film actress category."
    )


async def verify_shark_tank(evaluator: Evaluator, parent) -> None:
    item_node = evaluator.add_parallel(
        id="Shark_Tank_Investment_Success",
        desc="Identify Barbara Corcoran's most successful Shark Tank deal with investor status, net worth, company, investment, returns, and optional equity/timeframe.",
        parent=parent,
        critical=False
    )

    # Load extraction
    st_ex: Optional[SharkTankExtraction] = None
    for rec in evaluator._extraction_results:
        if "barbara_corcoran_investment" in rec:
            st_ex = SharkTankExtraction(**rec["barbara_corcoran_investment"])
            break
    if st_ex is None:
        st_ex = SharkTankExtraction()

    # Investor Identity and Net Worth (critical)
    inv_group = evaluator.add_parallel(
        id="Investor_Identity",
        desc="Confirm Barbara Corcoran is a Shark Tank investor and provide net worth as of 2026.",
        parent=item_node,
        critical=True
    )

    inv_name_present = evaluator.add_custom_node(
        result=bool(_coalesce(st_ex.investor_name)),
        id="Investor_Name_Present",
        desc="Investor name provided.",
        parent=inv_group,
        critical=True
    )

    inv_name_leaf = evaluator.add_leaf(
        id="Investor_Name_Verification",
        desc="The investor name must be Barbara Corcoran.",
        parent=inv_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The investor's name is Barbara Corcoran.",
        node=inv_name_leaf,
        additional_instruction="Case-insensitive exact match is fine; ensure it refers to the well-known Shark Tank investor."
    )

    inv_is_shark_leaf = evaluator.add_leaf(
        id="Investor_Reference_URL",
        desc="Provide a reference URL confirming Barbara Corcoran as a Shark Tank investor.",
        parent=inv_group,
        critical=True
    )
    await evaluator.verify(
        claim="Barbara Corcoran is an investor (a 'Shark') on the television show Shark Tank.",
        node=inv_is_shark_leaf,
        sources=_normalize_urls(st_ex.investor_urls),
        additional_instruction="Accept ABC's official page, credible bios, or Wikipedia clearly stating her role on Shark Tank."
    )

    net_worth_present = evaluator.add_custom_node(
        result=bool(_coalesce(st_ex.net_worth)),
        id="Net_Worth_Present",
        desc="Net worth value provided.",
        parent=inv_group,
        critical=True
    )

    net_worth_leaf = evaluator.add_leaf(
        id="Net_Worth_Verification",
        desc="Barbara Corcoran's net worth confirmed as $100 million as of 2026.",
        parent=inv_group,
        critical=True
    )
    await evaluator.verify(
        claim="Barbara Corcoran's net worth is approximately $100 million as of 2026.",
        node=net_worth_leaf,
        sources=_normalize_urls(st_ex.net_worth_urls),
        additional_instruction="Allow phrasing like 'about $100 million' or '$100M'; ensure the page supports 2026 context or is commonly cited."
    )

    # Company Identification (critical)
    comp_group = evaluator.add_parallel(
        id="Company_Identification",
        desc="Provide the company name for Barbara Corcoran's most successful Shark Tank investment.",
        parent=item_node,
        critical=True
    )

    comp_present = evaluator.add_custom_node(
        result=bool(_coalesce(st_ex.company_name)),
        id="Company_Name_Present",
        desc="Company name provided.",
        parent=comp_group,
        critical=True
    )

    comp_best_leaf = evaluator.add_leaf(
        id="Company_Name_Verification",
        desc="The company name must match the documented most successful investment in Barbara Corcoran's Shark Tank portfolio.",
        parent=comp_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The company '{_coalesce(st_ex.company_name)}' is widely reported as Barbara Corcoran's most successful Shark Tank investment.",
        node=comp_best_leaf,
        sources=_normalize_urls(st_ex.company_urls),
        additional_instruction="The source should explicitly state that this is her best/most successful/top-performing Shark Tank deal."
    )

    comp_invested_leaf = evaluator.add_leaf(
        id="Investment_Reference_URL",
        desc="Provide a reference URL confirming that Barbara invested in this company.",
        parent=comp_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Barbara Corcoran invested in '{_coalesce(st_ex.company_name)}' via Shark Tank.",
        node=comp_invested_leaf,
        sources=_normalize_urls(st_ex.company_urls, st_ex.investment_amount_urls),
        additional_instruction="A source stating the deal terms or investment at the time of the pitch is preferred."
    )

    # Initial Investment Amount (critical)
    invamt_group = evaluator.add_parallel(
        id="Initial_Investment_Amount",
        desc="Provide the dollar amount Barbara initially invested in this company.",
        parent=item_node,
        critical=True
    )

    invamt_present = evaluator.add_custom_node(
        result=bool(_coalesce(st_ex.initial_investment_amount)),
        id="Investment_Amount_Present",
        desc="Initial investment amount provided.",
        parent=invamt_group,
        critical=True
    )

    invamt_leaf = evaluator.add_leaf(
        id="Investment_Amount_Accuracy",
        desc="The investment amount must match the documented initial investment for this deal.",
        parent=invamt_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Barbara Corcoran invested {_coalesce(st_ex.initial_investment_amount)} in '{_coalesce(st_ex.company_name)}' on Shark Tank.",
        node=invamt_leaf,
        sources=_normalize_urls(st_ex.investment_amount_urls, st_ex.company_urls),
        additional_instruction="Verify that the specified dollar amount matches credible sources (deal recap, news, or show info)."
    )

    # Equity stake (non-critical) – kept separate to satisfy critical-parent constraint
    if _coalesce(st_ex.equity_stake):
        equity_group = evaluator.add_parallel(
            id="Equity_Stake_Info",
            desc="Equity stake information (optional).",
            parent=item_node,
            critical=False
        )
        equity_leaf = evaluator.add_leaf(
            id="Equity_Stake_Information",
            desc="Provide information about the equity stake Barbara Corcoran received for her investment.",
            parent=equity_group,
            critical=False
        )
        await evaluator.verify(
            claim=f"Barbara Corcoran received an equity stake of {_coalesce(st_ex.equity_stake)} in '{_coalesce(st_ex.company_name)}'.",
            node=equity_leaf,
            sources=_normalize_urls(st_ex.equity_urls, st_ex.company_urls),
            additional_instruction="Verify the equity percentage (or structure) from credible deal sources."
        )

    # Returns (critical)
    ret_group = evaluator.add_parallel(
        id="Return_On_Investment",
        desc="Provide the total return amount Barbara earned from this investment.",
        parent=item_node,
        critical=True
    )

    ret_present = evaluator.add_custom_node(
        result=bool(_coalesce(st_ex.total_return_amount)),
        id="Return_Amount_Present",
        desc="Return amount provided.",
        parent=ret_group,
        critical=True
    )

    ret_leaf = evaluator.add_leaf(
        id="Return_Amount_Verification",
        desc="The return amount must match the documented returns from this investment.",
        parent=ret_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Barbara Corcoran earned total returns of {_coalesce(st_ex.total_return_amount)} from her investment in '{_coalesce(st_ex.company_name)}'.",
        node=ret_leaf,
        sources=_normalize_urls(st_ex.return_amount_urls, st_ex.company_urls),
        additional_instruction="Ensure this is the investor's return (or profit/distributions), not just company revenue."
    )

    # Return timeframe (non-critical) – separate group
    if _coalesce(st_ex.returns_timeframe):
        timeframe_group = evaluator.add_parallel(
            id="Return_Timeframe_Group",
            desc="Timeframe over which the returns were generated (optional).",
            parent=item_node,
            critical=False
        )
        timeframe_leaf = evaluator.add_leaf(
            id="Return_Timeframe",
            desc="Provide information about the timeframe over which these returns were generated.",
            parent=timeframe_group,
            critical=False
        )
        await evaluator.verify(
            claim=f"The returns for '{_coalesce(st_ex.company_name)}' were generated over the timeframe {_coalesce(st_ex.returns_timeframe)}.",
            node=timeframe_leaf,
            sources=_normalize_urls(st_ex.timeframe_urls, st_ex.return_amount_urls, st_ex.company_urls),
            additional_instruction="Verify the time span or year range reported for realizing the returns."
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
    # Initialize evaluator (root must be non-critical due to framework constraint)
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

    # Create a top-level parallel node describing the overall task (non-critical to allow partial credit)
    overall_node = evaluator.add_parallel(
        id="Entertainment_Industry_Multi_Domain_Research_Task",
        desc="Comprehensive entertainment industry research task across four items (RodeoHouston performer, Hilary Duff international tour stop, Golden Globes winner, Shark Tank investment).",
        parent=root,
        critical=False
    )

    # Parallel extractions
    houston_task = evaluator.extract(
        prompt=prompt_extract_houston_rodeo(),
        template_class=HoustonRodeoExtraction,
        extraction_name="houston_rodeo"
    )
    hilary_task = evaluator.extract(
        prompt=prompt_extract_hilary_duff(),
        template_class=HilaryDuffTourExtraction,
        extraction_name="hilary_duff_tour"
    )
    globes_task = evaluator.extract(
        prompt=prompt_extract_golden_globes(),
        template_class=GoldenGlobesExtraction,
        extraction_name="golden_globes_2026"
    )
    shark_task = evaluator.extract(
        prompt=prompt_extract_sharktank(),
        template_class=SharkTankExtraction,
        extraction_name="barbara_corcoran_investment"
    )

    await asyncio.gather(houston_task, hilary_task, globes_task, shark_task)

    # Build verification subtrees
    await asyncio.gather(
        verify_houston_rodeo(evaluator, overall_node),
        verify_hilary_duff(evaluator, overall_node),
        verify_golden_globes(evaluator, overall_node),
        verify_shark_tank(evaluator, overall_node)
    )

    return evaluator.get_summary()