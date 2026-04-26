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
TASK_ID = "broadway_march_2026_plan"
TASK_DESCRIPTION = (
    "You are helping plan a Broadway theater outing in New York City for a family visiting during March 2026. "
    "The family includes: two adults, one child aged 6, and one family member who uses a wheelchair. "
    "Your task is to create a comprehensive plan that addresses all the following requirements:\n\n"
    "1. Show Selection: Identify a Broadway show that: (a) is age-appropriate for a 6-year-old child "
    "(meets theater age requirements and is family-friendly), (b) has performances scheduled during March 2026, "
    "(c) offers at least one matinee performance, and (d) is performed at a theater with full accessibility features "
    "(wheelchair seating, accessible restrooms, assistive listening devices).\n\n"
    "2. Ticketing Strategy: Provide information about: (a) when the box office is open for in-person ticket purchases, "
    "(b) the theater's refund and exchange policies, (c) how to request accessible seating (wheelchair space and companion seating), "
    "and (d) whether booster seats are available for the 6-year-old.\n\n"
    "3. Transportation: Identify: (a) at least one accessible subway station near the Theater District, and "
    "(b) alternative accessible transportation options (parking or rideshare).\n\n"
    "4. Day-of-Visit Logistics: Specify: (a) recommended arrival time before the performance, "
    "(b) theater bag policy (size restrictions and security procedures), and (c) what to expect regarding "
    "intermission and show duration.\n\n"
    "5. Contingency Awareness: Note: (a) policies regarding weather-related cancellations, and "
    "(b) possibility of understudy performances.\n\n"
    "For each component of your plan, provide: specific, verifiable information based on official theater sources, "
    "supporting URL references for key policies and information, and practical guidance that ensures the family has an accessible, "
    "age-appropriate, and well-organized theater experience."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ShowSelection(BaseModel):
    show_name: Optional[str] = None
    theater_name: Optional[str] = None

    # Age guidance
    age_guidance_text: Optional[str] = None
    age_urls: List[str] = Field(default_factory=list)

    # Schedule and matinee
    march_2026_has_performances: Optional[str] = None
    schedule_urls: List[str] = Field(default_factory=list)
    matinee_example_text: Optional[str] = None
    matinee_urls: List[str] = Field(default_factory=list)

    # Accessibility
    accessibility_features_text: Optional[str] = None
    accessibility_urls: List[str] = Field(default_factory=list)

    # General selection sources (catch-all)
    selection_sources: List[str] = Field(default_factory=list)


class Ticketing(BaseModel):
    box_office_hours_text: Optional[str] = None
    box_office_hours_urls: List[str] = Field(default_factory=list)

    refund_policy_text: Optional[str] = None
    refund_policy_urls: List[str] = Field(default_factory=list)

    accessible_seating_request_text: Optional[str] = None
    accessible_seating_urls: List[str] = Field(default_factory=list)

    booster_seat_text: Optional[str] = None
    booster_seat_urls: List[str] = Field(default_factory=list)

    ticketing_sources: List[str] = Field(default_factory=list)


class Transportation(BaseModel):
    accessible_station_name: Optional[str] = None
    accessible_station_urls: List[str] = Field(default_factory=list)

    alt_transport_type: Optional[str] = None  # e.g., "parking", "taxi", "rideshare"
    alt_transport_text: Optional[str] = None
    alt_transport_urls: List[str] = Field(default_factory=list)

    transportation_sources: List[str] = Field(default_factory=list)


class DayOf(BaseModel):
    arrival_time_text: Optional[str] = None
    arrival_urls: List[str] = Field(default_factory=list)

    bag_policy_text: Optional[str] = None
    bag_policy_urls: List[str] = Field(default_factory=list)

    running_time_text: Optional[str] = None
    intermission_text: Optional[str] = None
    duration_urls: List[str] = Field(default_factory=list)

    day_of_sources: List[str] = Field(default_factory=list)


class Contingency(BaseModel):
    weather_policy_text: Optional[str] = None
    weather_policy_urls: List[str] = Field(default_factory=list)

    understudy_policy_text: Optional[str] = None
    understudy_policy_urls: List[str] = Field(default_factory=list)

    contingency_sources: List[str] = Field(default_factory=list)


class PlanExtraction(BaseModel):
    show: Optional[ShowSelection] = None
    ticketing: Optional[Ticketing] = None
    transportation: Optional[Transportation] = None
    day_of: Optional[DayOf] = None
    contingency: Optional[Contingency] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract structured information about the Broadway outing plan from the answer. Return exactly the fields requested below.
    Only extract information explicitly present in the answer text. For URLs, extract the full actual URLs as they appear.

    Show Selection (object 'show'):
      - show_name: the specific Broadway show named
      - theater_name: the specific theater name where it plays
      - age_guidance_text: the show's or theater's stated minimum age/recommendation (e.g., "Ages 4+," "Ages 6+," "Family-friendly")
      - age_urls: URL(s) cited for age guidance or family-appropriateness (official show/theater/authorized vendor pages)
      - march_2026_has_performances: text stating performances exist in March 2026 if the answer claims this
      - schedule_urls: URL(s) that show the schedule/calendar/tickets indicating March 2026 performances
      - matinee_example_text: a specific example date/time for a matinee, if provided (e.g., "Sat, Mar 7 at 2:00 PM")
      - matinee_urls: URL(s) indicating matinee availability (can overlap with schedule URLs)
      - accessibility_features_text: text mentioning accessibility (wheelchair seating, accessible restrooms, assistive listening)
      - accessibility_urls: URL(s) that describe the theater's accessibility features
      - selection_sources: any other URL(s) the answer cites for show selection items

    Ticketing (object 'ticketing'):
      - box_office_hours_text: the hours for in-person box office purchases if stated
      - box_office_hours_urls: URL(s) that list box office hours
      - refund_policy_text: text describing refund/exchange policy
      - refund_policy_urls: URL(s) for refund/exchange policy (official show/theater/authorized vendor)
      - accessible_seating_request_text: instructions for requesting wheelchair/companion seating
      - accessible_seating_urls: URL(s) for accessible seating request/how-to
      - booster_seat_text: whether booster seats are available/how to obtain them
      - booster_seat_urls: URL(s) for booster seat info (if any)
      - ticketing_sources: any other URL(s) the answer cites for ticketing items

    Transportation (object 'transportation'):
      - accessible_station_name: the name of an accessible subway station near the Theater District (e.g., "Times Sq-42 St")
      - accessible_station_urls: URL(s) (preferably MTA) confirming station accessibility
      - alt_transport_type: the type of alternative accessible transport (e.g., "parking", "taxi", "rideshare")
      - alt_transport_text: details for the alternative accessible transport
      - alt_transport_urls: URL(s) supporting the alternative transport option
      - transportation_sources: any other URL(s) for transportation items

    Day Of (object 'day_of'):
      - arrival_time_text: the recommended arrival time before performance (e.g., "Arrive 30–45 minutes early")
      - arrival_urls: URL(s) that include arrival guidance/doors open timing
      - bag_policy_text: bag policy including size limits and security/bag check/inspection note
      - bag_policy_urls: URL(s) that describe the bag/security policy
      - running_time_text: the show's running time
      - intermission_text: info about whether there is an intermission
      - duration_urls: URL(s) supporting running time and/or intermission info
      - day_of_sources: any other URL(s) for day-of logistics

    Contingency (object 'contingency'):
      - weather_policy_text: policy for weather-related cancellations and how tickets are handled (refunds/exchanges)
      - weather_policy_urls: URL(s) for weather/cancellation policy
      - understudy_policy_text: notes that understudies may perform / cast subject to change and tickets remain valid
      - understudy_policy_urls: URL(s) for understudy/cast-change policy
      - contingency_sources: any other URL(s) for contingency items

    URL extraction rules:
      - Extract only URLs explicitly presented in the answer (including markdown links).
      - Include full URLs with protocol.
      - If a field has no URL(s) mentioned, return an empty list for that field.

    If any field is not mentioned in the answer, set it to null (for strings) or [] (for lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(*url_lists: List[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not isinstance(u, str):
                continue
            u_str = u.strip()
            if not u_str:
                continue
            if u_str not in seen:
                seen.add(u_str)
                out.append(u_str)
    return out


def _pick_sources(preferred: List[str], fallback: List[str]) -> List[str]:
    return preferred if preferred else fallback


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_show_selection(evaluator: Evaluator, parent_node, show: Optional[ShowSelection]) -> None:
    node = evaluator.add_parallel(
        id="show_selection",
        desc="Select a Broadway show and theater that satisfy age-appropriateness, March 2026 scheduling, matinee availability, and accessibility requirements.",
        parent=parent_node,
        critical=True
    )

    show = show or ShowSelection()
    all_show_urls = _dedup_urls(
        [show.selection_sources],
        [show.age_urls],
        [show.schedule_urls],
        [show.matinee_urls],
        [show.accessibility_urls],
    )

    # Sources presence (create first to gate others)
    evaluator.add_custom_node(
        result=len(all_show_urls) > 0,
        id="show_selection_sources",
        desc="Provides supporting URL reference(s) from official show/theater sources covering the key show-selection claims (age guidance, schedule/matinee, and accessibility).",
        parent=node,
        critical=True
    )

    # Identifies show and theater
    identify_leaf = evaluator.add_leaf(
        id="identifies_show_and_theater",
        desc="Names a specific Broadway show and the specific theater where it is performed.",
        parent=node,
        critical=True
    )
    claim_identify = (
        f"The official source confirms that the Broadway show '{show.show_name or '[missing show]'}' "
        f"is performed at the '{show.theater_name or '[missing theater]'}' theatre."
    )
    await evaluator.verify(
        claim=claim_identify,
        node=identify_leaf,
        sources=all_show_urls,
        additional_instruction="Verify the page(s) explicitly connect the production with the named theatre. "
                               "Allow minor naming variants (e.g., 'Theatre' vs 'Theater')."
    )

    # Age appropriateness
    age_leaf = evaluator.add_leaf(
        id="age_appropriateness",
        desc="Show meets the minimum age requirement of 4 and is appropriate for a 6-year-old (family-friendly or age recommendation suitable for ages 6–8).",
        parent=node,
        critical=True
    )
    claim_age = (
        f"The official source(s) indicate that children aged 6 may attend '{show.show_name or 'the show'}'—"
        f"either the minimum age is 6 or lower (e.g., 4+ or 5+), or the recommendation includes ages 6+ / family-friendly."
    )
    await evaluator.verify(
        claim=claim_age,
        node=age_leaf,
        sources=_pick_sources(show.age_urls, all_show_urls),
        additional_instruction="Pass if the source shows a minimum age of 4+, 5+, or 6+, or explicitly recommends the show for families/children around 6. "
                               "Minor wording differences are acceptable."
    )

    # March 2026 schedule
    march_leaf = evaluator.add_leaf(
        id="march_2026_schedule",
        desc="Show has performances scheduled during March 2026.",
        parent=node,
        critical=True
    )
    claim_march = (
        f"The official schedule/ticketing shows at least one performance of '{show.show_name or 'the show'}' in March 2026."
    )
    await evaluator.verify(
        claim=claim_march,
        node=march_leaf,
        sources=_pick_sources(show.schedule_urls, all_show_urls),
        additional_instruction="Look for any performance date in March 2026 (any day). Ticketing calendars or official schedules count."
    )

    # Matinee availability
    matinee_leaf = evaluator.add_leaf(
        id="matinee_available",
        desc="At least one matinee performance is available during March 2026 (daytime performance; typically around a 1:00–2:00 PM start time per constraints).",
        parent=node,
        critical=True
    )
    claim_matinee = (
        f"In March 2026, there is at least one matinee (a daytime performance starting before about 3:00 PM) "
        f"for '{show.show_name or 'the show'}'."
    )
    await evaluator.verify(
        claim=claim_matinee,
        node=matinee_leaf,
        sources=_pick_sources(_dedup_urls([show.matinee_urls], [show.schedule_urls]), all_show_urls),
        additional_instruction="Accept typical matinee times (e.g., 1:00 PM, 2:00 PM). "
                               "Evidence can be a calendar or ticketing page showing such a start time in March 2026."
    )

    # Theater accessibility features
    access_leaf = evaluator.add_leaf(
        id="theater_accessibility_features",
        desc="The theater provides wheelchair seating, accessible restrooms, and assistive listening devices, and complies with ADA accessibility standards.",
        parent=node,
        critical=True
    )
    claim_access = (
        f"The theatre '{show.theater_name or 'the theater'}' provides wheelchair seating, accessible restrooms, "
        f"and assistive listening devices (hearing assistance)."
    )
    await evaluator.verify(
        claim=claim_access,
        node=access_leaf,
        sources=_pick_sources(show.accessibility_urls, all_show_urls),
        additional_instruction="Verify the page lists wheelchair seating and accessible restrooms and some form of assistive listening/hearing devices. "
                               "Theatre or official accessibility pages are acceptable."
    )


async def verify_ticketing(evaluator: Evaluator, parent_node, ticketing: Optional[Ticketing], show: Optional[ShowSelection]) -> None:
    node = evaluator.add_parallel(
        id="ticketing_strategy",
        desc="Provide ticketing information needed for in-person purchase and accessible attendance.",
        parent=parent_node,
        critical=True
    )

    ticketing = ticketing or Ticketing()
    show = show or ShowSelection()

    all_ticket_urls = _dedup_urls(
        [ticketing.ticketing_sources],
        [ticketing.box_office_hours_urls],
        [ticketing.refund_policy_urls],
        [ticketing.accessible_seating_urls],
        [ticketing.booster_seat_urls],
    )

    evaluator.add_custom_node(
        result=len(all_ticket_urls) > 0,
        id="ticketing_sources",
        desc="Provides supporting URL reference(s) from official show/theater/authorized ticketing sources covering the key ticketing claims (hours, refund/exchange, accessible seating process, booster-seat policy if available).",
        parent=node,
        critical=True
    )

    # Box office hours
    bo_leaf = evaluator.add_leaf(
        id="box_office_hours",
        desc="States when the box office is open for in-person ticket purchases.",
        parent=node,
        critical=True
    )
    claim_bo = (
        f"This official page provides box office hours for in-person ticket purchases at "
        f"'{show.theater_name or 'the theater'}' or for '{show.show_name or 'the show'}'."
    )
    await evaluator.verify(
        claim=claim_bo,
        node=bo_leaf,
        sources=_pick_sources(ticketing.box_office_hours_urls, all_ticket_urls),
        additional_instruction="Look for a section explicitly titled 'Box Office Hours' or similar with days/times."
    )

    # Refund/exchange policy
    refund_leaf = evaluator.add_leaf(
        id="refund_exchange_policy",
        desc="States refund and exchange policy, including that tickets are generally non-refundable/non-exchangeable except for official show cancellations (per constraints).",
        parent=node,
        critical=True
    )
    claim_refund = (
        "This official page states that tickets are generally non-refundable and non-exchangeable, "
        "with exceptions if a performance is officially canceled or as otherwise specified by the venue/vendor."
    )
    await evaluator.verify(
        claim=claim_refund,
        node=refund_leaf,
        sources=_pick_sources(ticketing.refund_policy_urls, all_ticket_urls),
        additional_instruction="Accept standard Broadway policy language indicating all sales final, except in cases of official cancellation or explicit exceptions."
    )

    # Accessible seating request
    access_seat_leaf = evaluator.add_leaf(
        id="request_accessible_seating",
        desc="Explains how to request accessible seating, including wheelchair space with adjacent companion seating.",
        parent=node,
        critical=True
    )
    claim_access_seat = (
        "This official page explains how to request wheelchair-accessible seating and adjacent companion seating "
        "(e.g., via specific ticketing links, calling the box office, or accessibility request channels)."
    )
    await evaluator.verify(
        claim=claim_access_seat,
        node=access_seat_leaf,
        sources=_pick_sources(ticketing.accessible_seating_urls, all_ticket_urls),
        additional_instruction="Look for details on wheelchair spaces and companion seats and the process to secure them."
    )

    # Booster seats
    booster_leaf = evaluator.add_leaf(
        id="booster_seat_availability",
        desc="States whether booster seats are available and how to obtain them.",
        parent=node,
        critical=True
    )
    claim_booster = (
        "This official page indicates whether booster seats are available for children and how to obtain them "
        "(e.g., at the theatre, from ushers, limited quantities, first-come basis)."
    )
    await evaluator.verify(
        claim=claim_booster,
        node=booster_leaf,
        sources=_pick_sources(ticketing.booster_seat_urls, all_ticket_urls),
        additional_instruction="Pass if the page clearly mentions booster seats policy or availability; fail if not mentioned."
    )


async def verify_transportation(evaluator: Evaluator, parent_node, transportation: Optional[Transportation]) -> None:
    node = evaluator.add_parallel(
        id="transportation",
        desc="Identify accessible transit and alternatives to reach the Theater District.",
        parent=parent_node,
        critical=True
    )

    transportation = transportation or Transportation()

    all_trans_urls = _dedup_urls(
        [transportation.transportation_sources],
        [transportation.accessible_station_urls],
        [transportation.alt_transport_urls],
    )

    evaluator.add_custom_node(
        result=len(all_trans_urls) > 0,
        id="transportation_sources",
        desc="Provides supporting URL reference(s) from official transportation/NYC agency sources (e.g., MTA/NYC DOT) or official provider sources relevant to the proposed options, confirming accessibility where applicable.",
        parent=node,
        critical=True
    )

    # Accessible subway station
    subway_leaf = evaluator.add_leaf(
        id="accessible_subway_station",
        desc="Identifies at least one accessible subway station near the Theater District.",
        parent=node,
        critical=True
    )
    claim_subway = (
        f"The subway station '{transportation.accessible_station_name or '[missing station]'}' is an accessible station "
        f"(elevator/ramp, wheelchair accessible) and is near the Broadway Theater District in Midtown Manhattan."
    )
    await evaluator.verify(
        claim=claim_subway,
        node=subway_leaf,
        sources=_pick_sources(transportation.accessible_station_urls, all_trans_urls),
        additional_instruction="Use MTA station pages or maps that explicitly mark the station as 'Accessible' or show the wheelchair symbol. "
                               "Stations like Times Sq–42 St, 42 St–Port Authority Bus Terminal, 49 St, 50 St, etc., are acceptable if marked accessible."
    )

    # Alternative accessible transport
    alt_leaf = evaluator.add_leaf(
        id="alternative_accessible_transport",
        desc="Provides alternative accessible transportation options (parking or accessible taxi/rideshare).",
        parent=node,
        critical=True
    )
    claim_alt = (
        f"This page provides an accessible {transportation.alt_transport_type or 'parking/taxi/rideshare'} option "
        f"with accessibility details (e.g., ADA parking, wheelchair-accessible vehicles, designated drop-off)."
    )
    await evaluator.verify(
        claim=claim_alt,
        node=alt_leaf,
        sources=_pick_sources(transportation.alt_transport_urls, all_trans_urls),
        additional_instruction="Accept official provider pages (e.g., MTA/NYC DOT for parking/curb regulations, TLC/official taxi programs, or provider accessibility pages) that explicitly mention accessibility."
    )


async def verify_day_of(evaluator: Evaluator, parent_node, day_of: Optional[DayOf]) -> None:
    node = evaluator.add_parallel(
        id="day_of_visit_logistics",
        desc="Provide arrival and in-theater logistics for the performance day.",
        parent=parent_node,
        critical=True
    )

    day_of = day_of or DayOf()

    all_day_urls = _dedup_urls(
        [day_of.day_of_sources],
        [day_of.arrival_urls],
        [day_of.bag_policy_urls],
        [day_of.duration_urls],
    )

    evaluator.add_custom_node(
        result=len(all_day_urls) > 0,
        id="day_of_sources",
        desc="Provides supporting URL reference(s) from official show/theater sources for key day-of logistics (bag/security policy and running time/intermission information where available).",
        parent=node,
        critical=True
    )

    # Arrival time
    arrival_leaf = evaluator.add_leaf(
        id="arrival_time",
        desc="Specifies a recommended arrival time before the performance (at least 30 minutes per constraints).",
        parent=node,
        critical=True
    )
    claim_arrival = (
        "This page provides guidance to arrive around 30 minutes before showtime (or similar early-arrival guidance such as 30–45 minutes)."
    )
    await evaluator.verify(
        claim=claim_arrival,
        node=arrival_leaf,
        sources=_pick_sources(day_of.arrival_urls, all_day_urls),
        additional_instruction="Accept language like 'arrive 30 minutes early', 'doors open 30–45 minutes before', or equivalent practical guidance."
    )

    # Bag policy
    bag_leaf = evaluator.add_leaf(
        id="bag_policy",
        desc="States the theater bag policy, including size restrictions and security procedures (including that bags are subject to inspection per constraints).",
        parent=node,
        critical=True
    )
    claim_bag = (
        "This page states the theater's bag/security policy, including size limits and that bags are subject to inspection or screening."
    )
    await evaluator.verify(
        claim=claim_bag,
        node=bag_leaf,
        sources=_pick_sources(day_of.bag_policy_urls, all_day_urls),
        additional_instruction="Look for explicit mentions of bag size restrictions and security/bag checks."
    )

    # Intermission and duration
    inter_leaf = evaluator.add_leaf(
        id="intermission_and_duration",
        desc="Explains what to expect regarding intermission and show duration.",
        parent=node,
        critical=True
    )
    claim_inter = (
        "This page provides the show's running time and indicates whether there is an intermission."
    )
    await evaluator.verify(
        claim=claim_inter,
        node=inter_leaf,
        sources=_pick_sources(day_of.duration_urls, all_day_urls),
        additional_instruction="Accept pages showing runtime and a note like 'with one intermission' or 'no intermission'."
    )


async def verify_contingency(evaluator: Evaluator, parent_node, contingency: Optional[Contingency]) -> None:
    node = evaluator.add_parallel(
        id="contingency_awareness",
        desc="Address weather cancellations and understudy possibilities.",
        parent=parent_node,
        critical=True
    )

    contingency = contingency or Contingency()

    all_cont_urls = _dedup_urls(
        [contingency.contingency_sources],
        [contingency.weather_policy_urls],
        [contingency.understudy_policy_urls],
    )

    evaluator.add_custom_node(
        result=len(all_cont_urls) > 0,
        id="contingency_sources",
        desc="Provides supporting URL reference(s) from official show/theater/authorized ticketing sources for cancellation/refund and cast-change/understudy expectations.",
        parent=node,
        critical=True
    )

    # Weather cancellation policy
    weather_leaf = evaluator.add_leaf(
        id="weather_cancellation_policy",
        desc="Describes policies/expectations regarding weather-related cancellations and what happens to tickets for official cancellations (per constraints).",
        parent=node,
        critical=True
    )
    claim_weather = (
        "This page describes the policy for weather-related or other official cancellations and states that if a performance is canceled, "
        "tickets are refunded, exchanged, or otherwise honored per official instructions."
    )
    await evaluator.verify(
        claim=claim_weather,
        node=weather_leaf,
        sources=_pick_sources(contingency.weather_policy_urls, all_cont_urls),
        additional_instruction="Look for language about cancellations due to weather or emergencies and what happens to purchased tickets."
    )

    # Understudy possibility
    understudy_leaf = evaluator.add_leaf(
        id="understudy_possibility",
        desc="Notes that understudy performances may occur and tickets generally remain valid (including the 'above-the-title' absence caveat if applicable per constraints).",
        parent=node,
        critical=True
    )
    claim_understudy = (
        "This page notes that casting is subject to change and understudies may perform; tickets remain valid even if a particular performer is absent, "
        "unless a specific above-the-title policy states otherwise."
    )
    await evaluator.verify(
        claim=claim_understudy,
        node=understudy_leaf,
        sources=_pick_sources(contingency.understudy_policy_urls, all_cont_urls),
        additional_instruction="Accept standard 'cast subject to change' or 'understudies may appear' language."
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
    Evaluate an answer for the Broadway March 2026 outing plan task.
    """
    # Initialize evaluator (root defaults to non-critical; sections enforce critical gating)
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

    # Extraction
    plan_info = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction"
    )

    # Build verification tree per rubric
    await verify_show_selection(evaluator, root, plan_info.show)
    await verify_ticketing(evaluator, root, plan_info.ticketing, plan_info.show)
    await verify_transportation(evaluator, root, plan_info.transportation)
    await verify_day_of(evaluator, root, plan_info.day_of)
    await verify_contingency(evaluator, root, plan_info.contingency)

    # Return structured result
    return evaluator.get_summary()