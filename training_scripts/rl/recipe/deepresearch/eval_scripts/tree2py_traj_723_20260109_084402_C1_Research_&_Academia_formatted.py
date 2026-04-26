import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ai_ml_conference_2025_us_registration"
TASK_DESCRIPTION = (
    "Identify a major artificial intelligence or machine learning conference that is scheduled to take place in the "
    "United States between May 1, 2025 and December 31, 2025, and that offers in-person (physical) attendance "
    "registration. Provide the conference name, the specific dates it will be held, the city and state of the venue, "
    "and a link to the official conference registration page."
)

DATE_WINDOW_START_TEXT = "May 1, 2025"
DATE_WINDOW_END_TEXT = "December 31, 2025"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConferenceItem(BaseModel):
    conference_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    date_text: Optional[str] = None  # e.g., "June 10–14, 2025"
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    venue: Optional[str] = None
    registration_url: Optional[str] = None  # should point to official registration page
    official_site_url: Optional[str] = None  # general/home/overview page if provided
    other_urls: List[str] = Field(default_factory=list)  # any other URLs cited for this conference


class ConferenceListExtraction(BaseModel):
    conferences: List[ConferenceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
    Extract all AI/ML conferences mentioned in the answer. For each conference you find, return an object with:
    - conference_name: the conference name as written
    - start_date: the explicit start date string if provided (do NOT invent)
    - end_date: the explicit end date string if provided (do NOT invent)
    - date_text: the exact date text as written in the answer (e.g., "June 10–14, 2025")
    - city: the city of the physical venue (do NOT invent)
    - state: the state of the physical venue (e.g., 'CA', 'California', etc.; do NOT invent)
    - country: country name if present (e.g., 'USA', 'United States'); otherwise null
    - venue: venue name (e.g., 'Moscone Center'), if mentioned; otherwise null
    - registration_url: URL that points specifically to the official registration page (if the answer provides it). 
                        If multiple URLs are given, choose the one most clearly about registration (contains terms like 'register', 'registration', 'tickets', 'attend'). If none is present, set null.
    - official_site_url: the general official site/home/overview page if the answer provides one (avoid social media or news articles unless explicitly cited as the official site). If none is present, set null.
    - other_urls: list all other URLs mentioned that are relevant to this conference (supporting pages, schedule pages, venue page, CFP, etc.)
    
    IMPORTANT:
    - Only extract what is explicitly present in the answer. Do not infer or add missing information.
    - If the answer lists more than one conference, include all of them in the 'conferences' array in the order they appear.
    - If the answer provides fewer details for an item (e.g., missing state or dates), include the item with missing fields set to null.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def select_first_conference(extracted: ConferenceListExtraction) -> ConferenceItem:
    """
    Select the first conference entry from the extraction result.
    If none found, return an empty ConferenceItem placeholder.
    """
    if extracted and extracted.conferences:
        return extracted.conferences[0]
    return ConferenceItem()


def build_sources_list(conf: ConferenceItem) -> List[str]:
    """
    Build a deduplicated list of URLs to use for verification.
    Prioritize registration_url and official_site_url, then include other_urls.
    """
    seen = set()
    ordered: List[str] = []

    def add(url: Optional[str]):
        if url and isinstance(url, str):
            u = url.strip()
            if u and u not in seen:
                seen.add(u)
                ordered.append(u)

    add(conf.registration_url)
    add(conf.official_site_url)
    if conf.other_urls:
        for u in conf.other_urls:
            add(u)
    return ordered


def display_dates(conf: ConferenceItem) -> str:
    """
    Preferred human-readable date string for claims:
    - Use date_text if available; otherwise combine start_date and end_date if present; otherwise empty string.
    """
    if conf.date_text:
        return conf.date_text
    if conf.start_date and conf.end_date:
        return f"{conf.start_date} to {conf.end_date}"
    return conf.start_date or conf.end_date or ""


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def add_required_fields_checks(evaluator: Evaluator, parent_node, conf: ConferenceItem):
    """
    Add and run checks under 'Required_Response_Fields' (parallel, critical).
    - Conference_Name_Provided (custom existence)
    - Conference_Dates_Provided (custom existence)
    - City_State_Provided (custom existence)
    - Official_Registration_Link_Provided (verify via URL content)
    """
    # Conference name provided
    evaluator.add_custom_node(
        result=bool(conf.conference_name and conf.conference_name.strip()),
        id="Conference_Name_Provided",
        desc="Provides the conference name.",
        parent=parent_node,
        critical=True
    )

    # Conference dates provided (either date_text or both start+end)
    has_dates = bool(
        (conf.date_text and conf.date_text.strip())
        or (conf.start_date and conf.start_date.strip() and conf.end_date and conf.end_date.strip())
    )
    evaluator.add_custom_node(
        result=has_dates,
        id="Conference_Dates_Provided",
        desc="Provides the specific dates the conference will be held.",
        parent=parent_node,
        critical=True
    )

    # City and State provided (consistency with US checked in separate eligibility node)
    has_city_state = bool(conf.city and conf.city.strip() and conf.state and conf.state.strip())
    evaluator.add_custom_node(
        result=has_city_state,
        id="City_State_Provided",
        desc="Provides the venue city and state (and they are consistent with a U.S. location).",
        parent=parent_node,
        critical=True
    )

    # Official registration link provided AND is actually a registration page (verify by URL)
    reg_link_node = evaluator.add_leaf(
        id="Official_Registration_Link_Provided",
        desc="Provides a URL that points to the official conference registration page (not merely a general homepage).",
        parent=parent_node,
        critical=True
    )

    reg_claim = (
        f"This URL is an official registration page for the conference"
        f"{f' {conf.conference_name}' if conf.conference_name else ''}."
    )
    add_ins = (
        "The page should clearly indicate registration actions (e.g., 'Register', 'Registration', 'Tickets', 'Attend', "
        "'Purchase Pass', or a registration form). It should be an official page for the conference (not a general "
        "homepage, news article, or third-party listing) and must be specifically about registration. "
        "If no valid URL is provided, mark as not supported."
    )

    await evaluator.verify(
        claim=reg_claim,
        node=reg_link_node,
        sources=conf.registration_url if conf.registration_url else None,
        additional_instruction=add_ins
    )


async def add_conference_eligibility_checks(evaluator: Evaluator, parent_node, conf: ConferenceItem):
    """
    Add and run checks under 'Conference_Eligibility' (parallel, critical):
    - AI_ML_Focus (verify via sources)
    - US_Location_Constraint (verify via sources)
    - Date_Range_Constraint (verify via sources)
    - In_Person_Registration_Constraint (verify via sources)
    """
    sources = build_sources_list(conf)
    sources_or_none: Optional[List[str] | str | None] = sources if sources else None

    # Helper instruction to ensure evidence is required
    base_no_source_rule = (
        "Rely strictly on the provided webpage(s). If no valid URL is provided or the page(s) do not support the claim, "
        "mark as not supported (Incorrect)."
    )

    # AI/ML Focus
    ai_ml_node = evaluator.add_leaf(
        id="AI_ML_Focus",
        desc="The conference is primarily an artificial intelligence and/or machine learning conference (as indicated by official scope/description).",
        parent=parent_node,
        critical=True
    )
    ai_ml_claim = (
        "This conference is primarily focused on artificial intelligence and/or machine learning (AI/ML), as indicated by the official site."
    )
    ai_ml_ins = (
        "Look for explicit mentions of 'artificial intelligence', 'AI', 'machine learning', 'ML', 'deep learning', "
        "'neural networks', or similar. General tech, software engineering, or analytics events without a primary AI/ML "
        "focus should not pass. " + base_no_source_rule
    )
    await evaluator.verify(
        claim=ai_ml_claim,
        node=ai_ml_node,
        sources=sources_or_none,
        additional_instruction=ai_ml_ins
    )

    # US Location Constraint
    us_loc_node = evaluator.add_leaf(
        id="US_Location_Constraint",
        desc="The conference is physically held within the United States.",
        parent=parent_node,
        critical=True
    )
    loc_phrase = ""
    if conf.city or conf.state:
        c = conf.city or ""
        s = conf.state or ""
        loc_phrase = f" in {c}{', ' if c and s else ''}{s}, United States"
    us_loc_claim = f"The conference is physically held{loc_phrase}."
    us_loc_ins = (
        "Confirm the location is within the United States. Accept indicators like 'USA', 'U.S.', 'United States', or a "
        "U.S. city+state combination (e.g., 'New Orleans, LA', 'San Diego, California'). If multiple venues are listed, "
        "the primary physical event must be in the U.S. " + base_no_source_rule
    )
    await evaluator.verify(
        claim=us_loc_claim,
        node=us_loc_node,
        sources=sources_or_none,
        additional_instruction=us_loc_ins
    )

    # Date Range Constraint (between May 1, 2025 and Dec 31, 2025 inclusive)
    date_node = evaluator.add_leaf(
        id="Date_Range_Constraint",
        desc="The conference occurs between May 1, 2025 and December 31, 2025 (inclusive).",
        parent=parent_node,
        critical=True
    )
    date_disp = display_dates(conf)
    date_claim = (
        f"According to the official page(s), the conference dates {f'({date_disp}) ' if date_disp else ''}"
        f"fall between {DATE_WINDOW_START_TEXT} and {DATE_WINDOW_END_TEXT}, inclusive."
    )
    date_ins = (
        "Read the date(s) shown on the webpage(s). If any part of the event (e.g., main conference days) clearly lies "
        "outside May 1, 2025 to December 31, 2025, mark as not supported. Ignore separate workshops/tutorials if the main "
        "conference itself is within range. " + base_no_source_rule
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=sources_or_none,
        additional_instruction=date_ins
    )

    # In-Person Registration Constraint
    inperson_node = evaluator.add_leaf(
        id="In_Person_Registration_Constraint",
        desc="The conference offers in-person (physical) attendance registration options.",
        parent=parent_node,
        critical=True
    )
    inperson_claim = (
        "The conference offers in-person (onsite/physical) registration for attendees."
    )
    inperson_ins = (
        "Look for registration categories explicitly indicating in-person/onsite attendance (e.g., 'In-Person', 'Onsite', "
        "'Physical', or venue-based passes). If only virtual/remote options are offered, or registration is not yet opened "
        "nor described, mark as not supported. " + base_no_source_rule
    )
    await evaluator.verify(
        claim=inperson_claim,
        node=inperson_node,
        sources=sources_or_none,
        additional_instruction=inperson_ins
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
    """
    Evaluate an answer for the task:
    Identify a major AI/ML conference in the U.S. between May 1, 2025 and Dec 31, 2025 with in-person registration,
    returning conference name, dates, city/state, and registration link.
    """
    # Initialize evaluator with a neutral root; we will build our own critical tree beneath.
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

    # Extract potential conferences from the answer
    extracted_list = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferenceListExtraction,
        extraction_name="conference_candidates",
    )

    # Select the first conference (as required: pick the first if multiple provided)
    selected = select_first_conference(extracted_list)

    # Record selected conference details as additional info for transparency
    evaluator.add_custom_info(
        info={
            "selected_conference": selected.dict(),
            "date_window": {
                "start": DATE_WINDOW_START_TEXT,
                "end": DATE_WINDOW_END_TEXT
            }
        },
        info_type="selection_info",
        info_name="selected_conference_info"
    )

    # Build the "Answer_Compliance" critical sequential node
    compliance_node = evaluator.add_sequential(
        id="Answer_Compliance",
        desc="Evaluate whether the response identifies an eligible AI/ML conference in the specified window and provides all required details.",
        parent=root,
        critical=True
    )

    # 1) Conference_Eligibility (critical, parallel children)
    eligibility_node = evaluator.add_parallel(
        id="Conference_Eligibility",
        desc="The identified conference satisfies the eligibility constraints from the prompt.",
        parent=compliance_node,
        critical=True
    )
    await add_conference_eligibility_checks(evaluator, eligibility_node, selected)

    # 2) Required_Response_Fields (critical, parallel children)
    fields_node = evaluator.add_parallel(
        id="Required_Response_Fields",
        desc="The response includes all required fields requested in the prompt.",
        parent=compliance_node,
        critical=True
    )
    await add_required_fields_checks(evaluator, fields_node, selected)

    # Return the final structured evaluation summary
    return evaluator.get_summary()