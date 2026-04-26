import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.evaluator import Evaluator

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "big10_feb26_fairs"
TASK_DESCRIPTION = """
You are a recruitment coordinator for a technology company planning to attend multiple Big Ten university career fairs in February 2026 to recruit engineering and business students. Your company requires career fairs that meet specific logistical and capacity requirements for effective recruiting.

Identify four Big Ten universities that have career fairs scheduled in February 2026 meeting all of the following requirements:

1. Date: The career fair must take place in February 2026
2. Format: The career fair must include at least one in-person day (hybrid formats with both in-person and virtual components are acceptable)
3. Duration: The in-person portion must be at least 3 hours long
4. Registration: Employer registration must be currently open or scheduled to open before the event
5. Booth Capacity: At least one booth option must accommodate 4 or more company representatives
6. Pricing: Booth pricing information must be publicly available or obtainable through official university channels
7. Audience: The career fair must be open to engineering students, business students, or all majors (not restricted to a single narrow discipline such as only industrial engineering or only nursing)

For each of the four universities, provide:
- University name
- Official career fair name
- Exact date(s) in February 2026
- Specific venue/location on campus
- Event time schedule (start and end times for the in-person portion)
- Format (in-person or hybrid)
- At least one booth option that accommodates 4+ representatives, with pricing
- Direct link to the official university career fair page or employer information page
- Direct link to employer registration information or instructions

All information must be verifiable through official university career services websites or official university pages.
"""

# Helper reference lists for context (added to summary for transparency)
BIG_TEN_SCHOOLS = [
    "University of Illinois Urbana-Champaign",
    "Indiana University Bloomington",
    "University of Iowa",
    "University of Maryland",
    "University of Michigan",
    "Michigan State University",
    "University of Minnesota",
    "University of Nebraska–Lincoln",
    "University of Nebraska-Lincoln",
    "Northwestern University",
    "The Ohio State University",
    "Ohio State University",
    "Penn State University",
    "The Pennsylvania State University",
    "Purdue University",
    "Rutgers University–New Brunswick",
    "Rutgers University-New Brunswick",
    "University of Wisconsin–Madison",
    "University of Wisconsin-Madison",
]
MIDWEST_STATES = ["IL", "IN", "IA", "MI", "MN", "NE", "OH", "WI"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TimeRange(BaseModel):
    date: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None


class ModalityByDate(BaseModel):
    date: Optional[str] = None
    modality: Optional[str] = None  # e.g., "in-person", "virtual", "hybrid"


class BoothOption(BaseModel):
    name: Optional[str] = None  # e.g., "Standard Booth", "Premium Sponsor"
    capacity_text: Optional[str] = None  # free text, e.g., "up to 4 reps"
    capacity_number: Optional[int] = None  # if answer specifies a number
    price: Optional[str] = None  # free text for price to be flexible
    url: Optional[str] = None  # specific page for package/pricing if given


class CareerFairItem(BaseModel):
    university: Optional[str] = None
    state: Optional[str] = None  # optional helper (e.g., "IL", "IN")
    fair_name: Optional[str] = None
    dates: List[str] = Field(default_factory=list)  # exact dates as provided in answer
    modality: Optional[str] = None  # overall format (e.g., "in-person", "hybrid")
    modality_by_date: List[ModalityByDate] = Field(default_factory=list)
    in_person_dates: List[str] = Field(default_factory=list)  # subset of dates that are in-person
    venue: Optional[str] = None
    in_person_time_ranges: List[TimeRange] = Field(default_factory=list)  # per in-person date
    employer_info_url: Optional[str] = None  # official fair page/employer info page
    registration_url: Optional[str] = None  # direct link to registration info or instructions
    pricing_info_url: Optional[str] = None  # specific pricing page if provided
    booth_options: List[BoothOption] = Field(default_factory=list)
    audience_text: Optional[str] = None  # e.g., "All majors", "Engineering + Business"
    host_unit_text: Optional[str] = None  # e.g., "Career Services", "College of Engineering"
    extra_urls: List[str] = Field(default_factory=list)  # any other official URLs mentioned


class CareerFairsExtraction(BaseModel):
    fairs: List[CareerFairItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_career_fairs() -> str:
    return """
    Extract up to the first 8 distinct career fairs described in the answer (we will evaluate the first 4).
    For each item, extract ONLY what the answer explicitly provides. Do not invent any URLs or details.

    For each career fair, extract:
    - university: University name as stated in the answer (string)
    - state: If explicitly mentioned in the answer (e.g., "IL", "Ohio"), extract a 2-letter state code if present; else null
    - fair_name: Official fair name as stated
    - dates: Array of exact date strings (e.g., ["February 10, 2026", "February 11, 2026"])
    - modality: Overall format if single-day or general statement (e.g., "in-person", "hybrid"); else null
    - modality_by_date: Array of objects for multi-day fairs, each with:
        - date: date string
        - modality: "in-person", "virtual", or "hybrid" as stated
    - in_person_dates: Array of date strings that are explicitly in-person (if stated)
    - venue: Specific on-campus venue/location for the in-person portion, as stated (string or null)
    - in_person_time_ranges: Array of objects, each with:
        - date: date string (should correspond to an in-person date if possible)
        - start: start time as string (e.g., "10:00 AM")
        - end: end time as string (e.g., "3:00 PM")
    - employer_info_url: Direct URL to the official university career fair page or employer information page (must be explicitly present in the answer; if none, null)
    - registration_url: Direct URL to employer registration information or instructions (must be explicitly present in the answer; if none, null)
    - pricing_info_url: Specific URL where pricing is shown or instructions to obtain pricing are provided (if provided in the answer; else null)
    - booth_options: Array of objects; for any option mentioned in the answer that relates to employer packages/booths, include:
        - name
        - capacity_text (e.g., "up to 4 representatives")
        - capacity_number (if a numeric capacity is explicitly given; else null)
        - price (as presented, e.g., "$1200", "$900 early bird")
        - url (specific URL for that option if given; else null)
    - audience_text: Who the fair is open to (e.g., "All majors", "Engineering students and Business students"); exactly as stated
    - host_unit_text: The hosting unit (e.g., "Career Services", "College of Engineering"), if explicitly stated
    - extra_urls: Any other official URLs related to this specific fair that are present in the answer (e.g., schedule PDFs, venue pages, employer guides). Only include URLs that appear in the answer text.

    RULES:
    - Only extract URLs that are explicitly present in the answer (plain or in markdown link form). Do not infer.
    - Preserve date/time strings exactly as shown.
    - If a field is not present in the answer, set it to null (or an empty array for list fields).
    - Do not merge information from different fairs; keep each fair separate.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_key(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _gather_sources(cf: CareerFairItem) -> List[str]:
    urls: List[str] = []
    for u in [cf.employer_info_url, cf.registration_url, cf.pricing_info_url]:
        if u and isinstance(u, str) and u.strip():
            urls.append(u.strip())
    for b in cf.booth_options:
        if b and b.url and b.url.strip():
            urls.append(b.url.strip())
    for u in cf.extra_urls:
        if u and isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate preserving order
    seen = set()
    unique_urls: List[str] = []
    for u in urls:
        if u not in seen:
            unique_urls.append(u)
            seen.add(u)
    return unique_urls


def _has_time_schedule(cf: CareerFairItem) -> bool:
    if not cf.in_person_time_ranges:
        return False
    for tr in cf.in_person_time_ranges:
        if tr and tr.start and tr.end and tr.start.strip() and tr.end.strip():
            return True
    return False


def _format_dates_for_claim(cf: CareerFairItem) -> str:
    return ", ".join([d for d in cf.dates if d and d.strip()]) if cf.dates else "date(s) listed on the official page"


def _first_inperson_timerange(cf: CareerFairItem) -> Optional[TimeRange]:
    for tr in cf.in_person_time_ranges:
        if tr and tr.start and tr.end:
            return tr
    return None


# --------------------------------------------------------------------------- #
# Verification logic per career fair                                          #
# --------------------------------------------------------------------------- #
async def verify_career_fair(evaluator: Evaluator, parent_node, cf: CareerFairItem, index_zero_based: int) -> None:
    """
    Build verification nodes for one career fair and run verifications.
    """
    idx = index_zero_based + 1
    cf_node = evaluator.add_parallel(
        id=f"career_fair_{idx}",
        desc=f"{idx}st qualifying Big Ten university career fair (with required details and official sources)" if idx == 1 else
             (f"{idx}nd qualifying Big Ten university career fair (with required details and official sources)" if idx == 2 else
              (f"{idx}rd qualifying Big Ten university career fair (with required details and official sources)" if idx == 3 else
               f"{idx}th qualifying Big Ten university career fair (with required details and official sources)")),
        parent=parent_node,
        critical=False
    )
    prefix = f"cf{idx}"

    # Pre-calculate sources once
    sources_all = _gather_sources(cf)

    # 1) University name provided (existence)
    evaluator.add_custom_node(
        result=bool(cf.university and cf.university.strip()),
        id=f"{prefix}_university_name_provided",
        desc="University name is provided",
        parent=cf_node,
        critical=True
    )

    # 2) University is Big Ten (simple verify)
    leaf_big_ten = evaluator.add_leaf(
        id=f"{prefix}_university_is_big_ten",
        desc="University is a Big Ten institution",
        parent=cf_node,
        critical=True
    )
    uni_name = cf.university or "the stated university"
    await evaluator.verify(
        claim=f"The university '{uni_name}' is a member of the Big Ten Conference.",
        node=leaf_big_ten,
        sources=None,
        additional_instruction="Use your general knowledge; allow common naming variants (e.g., 'The Ohio State University' vs 'Ohio State University')."
    )

    # 3) University in Midwest (simple verify)
    leaf_midwest = evaluator.add_leaf(
        id=f"{prefix}_university_in_midwest",
        desc="University is located in the Midwest (per stated constraint)",
        parent=cf_node,
        critical=True
    )
    state_hint = f" (state: {cf.state})" if cf.state else ""
    await evaluator.verify(
        claim=f"The university '{uni_name}' is located in the U.S. Midwest{state_hint}.",
        node=leaf_midwest,
        sources=None,
        additional_instruction="Consider the Midwest to include the following states: Illinois (IL), Indiana (IN), Iowa (IA), Michigan (MI), Minnesota (MN), Nebraska (NE), Ohio (OH), Wisconsin (WI)."
    )

    # 4) Career fair name provided (existence)
    evaluator.add_custom_node(
        result=bool(cf.fair_name and cf.fair_name.strip()),
        id=f"{prefix}_career_fair_name_provided",
        desc="Official career fair name is provided",
        parent=cf_node,
        critical=True
    )

    # 5) Hosted by official unit (verify via URLs)
    leaf_host = evaluator.add_leaf(
        id=f"{prefix}_hosted_by_official_unit",
        desc="Career fair is officially hosted by the university's career services or an academic college",
        parent=cf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event '{cf.fair_name or 'the career fair'}' is officially hosted by {cf.university or 'the university'}'s career services office or an academic college.",
        node=leaf_host,
        sources=sources_all,
        additional_instruction="Check the page header/branding or event description for references to official units such as Career Services, Career Center, College of Engineering Career Office, Business Career Services, etc."
    )

    # 6) Date in February 2026 and exact dates provided (verify via URLs)
    leaf_dates = evaluator.add_leaf(
        id=f"{prefix}_date_in_feb_2026_and_exact_dates_provided",
        desc="Exact date(s) are provided and occur in February 2026",
        parent=cf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The career fair takes place in February 2026, and the official page(s) provide exact date(s): {_format_dates_for_claim(cf)}.",
        node=leaf_dates,
        sources=sources_all,
        additional_instruction="Confirm that the event dates clearly fall in February 2026 and that precise dates are listed on the official page(s)."
    )

    # 7) Venue provided (existence)
    evaluator.add_custom_node(
        result=bool(cf.venue and cf.venue.strip()),
        id=f"{prefix}_venue_provided",
        desc="Specific venue/location on campus is provided",
        parent=cf_node,
        critical=True
    )

    # 8) Format includes at least one in-person day (verify via URLs)
    leaf_inperson = evaluator.add_leaf(
        id=f"{prefix}_format_includes_in_person_day",
        desc="Career fair includes at least one in-person day (hybrid acceptable)",
        parent=cf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The career fair includes at least one in-person day (hybrid formats are acceptable if they contain any in-person component).",
        node=leaf_inperson,
        sources=sources_all,
        additional_instruction="Look for wording such as 'in-person', 'on campus', 'in person', or explicit venue listings for at least one day."
    )

    # 9) Format and per-day modality specified if multi-day (existence logic)
    if cf.dates and len(cf.dates) > 1:
        has_mapping = bool(cf.modality_by_date and any(m.modality for m in cf.modality_by_date))
        has_inperson_marked = bool(cf.modality_by_date and any((m.modality or "").lower().find("person") >= 0 for m in cf.modality_by_date))
        result_modality_specified = has_mapping and has_inperson_marked
    else:
        result_modality_specified = bool(cf.modality and cf.modality.strip())
    evaluator.add_custom_node(
        result=result_modality_specified,
        id=f"{prefix}_format_and_per_day_modality_specified_if_multiday",
        desc="Format is stated (in-person or hybrid); if multi-day, identifies which date(s) are in-person vs virtual",
        parent=cf_node,
        critical=True
    )

    # 10) In-person time schedule provided (existence)
    evaluator.add_custom_node(
        result=_has_time_schedule(cf),
        id=f"{prefix}_in_person_time_schedule_provided",
        desc="Start and end time(s) for the in-person portion are provided (per in-person date if multi-day)",
        parent=cf_node,
        critical=True
    )

    # 11) In-person duration at least 3 hours (verify via URLs)
    leaf_duration = evaluator.add_leaf(
        id=f"{prefix}_in_person_duration_at_least_3_hours",
        desc="In-person portion lasts at least 3 hours (verifiable from provided start/end times)",
        parent=cf_node,
        critical=True
    )
    tr = _first_inperson_timerange(cf)
    tr_text = f" from {tr.start} to {tr.end}" if tr and tr.start and tr.end else ""
    await evaluator.verify(
        claim=f"The in-person portion lasts at least 3 hours{tr_text if tr_text else ''}.",
        node=leaf_duration,
        sources=sources_all,
        additional_instruction="Use the schedule times shown on the official page(s). If multiple in-person dates exist, it suffices that one in-person date meets or exceeds 3 hours."
    )

    # 12) Employer registration open or opens before event (verify via URLs)
    leaf_reg_window = evaluator.add_leaf(
        id=f"{prefix}_employer_registration_open_or_opens_before_event",
        desc="Employer registration is currently open or scheduled to open before the event",
        parent=cf_node,
        critical=True
    )
    await evaluator.verify(
        claim="Employer registration is open now or opens before the earliest in-person event date.",
        node=leaf_reg_window,
        sources=sources_all,
        additional_instruction="Check employer registration sections or platform event pages for 'Registration open', 'Opens on', or similar. If an 'opens on' date is listed, it should precede the event date."
    )

    # 13) Registration link or official instructions provided
    if cf.registration_url and cf.registration_url.strip():
        node_reg_link = evaluator.add_leaf(
            id=f"{prefix}_registration_link_or_official_instructions_provided",
            desc="Direct link to employer registration information OR official registration instructions are provided",
            parent=cf_node,
            critical=True
        )
        await evaluator.verify(
            claim="This page provides employer registration information or official instructions for how to register.",
            node=node_reg_link,
            sources=cf.registration_url.strip(),
            additional_instruction="Accept event pages on official university platforms (e.g., Handshake, Symplicity, Brazen, 12twenty) or official university career services pages that clearly explain how to register."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{prefix}_registration_link_or_official_instructions_provided",
            desc="Direct link to employer registration information OR official registration instructions are provided",
            parent=cf_node,
            critical=True
        )

    # 14) Booth capacity 4+ (verify via URLs)
    leaf_capacity = evaluator.add_leaf(
        id=f"{prefix}_booth_capacity_4plus",
        desc="At least one booth option accommodates 4+ company representatives",
        parent=cf_node,
        critical=True
    )
    await evaluator.verify(
        claim="There is at least one employer booth or registration package that accommodates 4 or more company representatives.",
        node=leaf_capacity,
        sources=sources_all,
        additional_instruction="Look for 'representatives', 'company reps', or similar within package descriptions or employer guides."
    )

    # 15) Booth pricing available or official obtainment (verify via URLs)
    leaf_pricing = evaluator.add_leaf(
        id=f"{prefix}_booth_pricing_available_or_official_obtainment",
        desc="Booth pricing information is provided OR official instructions/channel to obtain pricing is provided",
        parent=cf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official page(s) provide either explicit booth pricing for employers or clear official instructions on how to obtain pricing.",
        node=leaf_pricing,
        sources=sources_all,
        additional_instruction="Accept explicit fee tables, pricing PDFs, or official directions such as 'contact us' or 'log in to view fees' if this is a standard official platform instruction."
    )

    # 16) Audience scope OK (verify via URLs)
    leaf_audience = evaluator.add_leaf(
        id=f"{prefix}_audience_scope_ok",
        desc="Career fair is open to engineering students, business students, or all majors (not restricted to a single narrow discipline)",
        parent=cf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The fair is open to engineering students, business students, or all majors (i.e., not restricted to a singular narrow discipline).",
        node=leaf_audience,
        sources=sources_all,
        additional_instruction="Look for 'all majors', 'engineering', 'business', or equivalent language; reject if the fair is strictly limited to a narrow field like only 'nursing' or only 'industrial engineering'."
    )

    # 17) Official career fair page link provided (existence)
    evaluator.add_custom_node(
        result=bool(cf.employer_info_url and cf.employer_info_url.strip()),
        id=f"{prefix}_official_career_fair_page_link",
        desc="Direct link to the official university career fair page or employer information page is provided",
        parent=cf_node,
        critical=True
    )

    # 18) Sources official and verifiable (verify via URLs)
    leaf_sources = evaluator.add_leaf(
        id=f"{prefix}_sources_official_and_verifiable",
        desc="Provided information is verifiable via official university pages/career services pages (or officially used systems linked from them)",
        parent=cf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided page(s) are official university pages or officially used recruiting platforms (e.g., Handshake, Symplicity, Brazen, 12twenty) representing this university’s career fair.",
        node=leaf_sources,
        sources=sources_all,
        additional_instruction="Accept *.edu pages or official platform pages that explicitly display the university event. Pages must be either on the university domain or a recognized official platform used by that university for employer events."
    )


# --------------------------------------------------------------------------- #
# Root-level helper: distinct four fairs                                      #
# --------------------------------------------------------------------------- #
def evaluate_four_distinct(fairs: List[CareerFairItem]) -> Tuple[bool, List[Tuple[str, str]]]:
    """
    Determine whether we have 4 distinct (non-duplicate) career fairs by (university, fair_name).
    Returns (result, keys) where keys is the normalized pairs considered.
    """
    pairs: List[Tuple[str, str]] = []
    for cf in fairs[:4]:
        uni = _normalize_key(cf.university)
        fair = _normalize_key(cf.fair_name)
        pairs.append((uni, fair))

    # Valid only if we have 4 items and none of the pairs is empty
    if len(pairs) < 4:
        return False, pairs

    # Check distinct and non-empty
    non_empty = all(u and f for u, f in pairs)
    distinct = len(set(pairs)) == 4
    return (non_empty and distinct), pairs


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Big Ten February 2026 career fairs task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as non-critical aggregator to allow partial credit
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_career_fairs(),
        template_class=CareerFairsExtraction,
        extraction_name="career_fairs_extraction"
    )

    # Record contextual reference info
    evaluator.add_custom_info(
        {"big_ten_reference": BIG_TEN_SCHOOLS, "midwest_states_reference": MIDWEST_STATES},
        info_type="reference_lists",
        info_name="reference_lists"
    )

    # Determine the four items to evaluate (pad with empty items if fewer)
    fairs_to_use: List[CareerFairItem] = list(extraction.fairs[:4])
    while len(fairs_to_use) < 4:
        fairs_to_use.append(CareerFairItem())

    # Root-level: Four distinct qualifying career fairs provided (distinctness check only)
    ok_distinct, considered_pairs = evaluate_four_distinct(fairs_to_use)
    evaluator.add_custom_node(
        result=ok_distinct,
        id="four_distinct_qualifying_career_fairs_provided",
        desc="Response provides 4 distinct (non-duplicate) qualifying Big Ten university career fairs",
        parent=root,
        critical=True
    )
    evaluator.add_custom_info(
        {"considered_pairs_university_fair_normalized": considered_pairs},
        info_type="distinctness_pairs"
    )

    # Build verification nodes for each career fair
    for i in range(4):
        await verify_career_fair(evaluator, root, fairs_to_use[i], i)

    # Return the evaluation summary
    return evaluator.get_summary()