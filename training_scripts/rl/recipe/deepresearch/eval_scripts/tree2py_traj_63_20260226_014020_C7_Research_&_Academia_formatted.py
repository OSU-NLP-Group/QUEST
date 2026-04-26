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
TASK_ID = "conf_2026_h1_planning"
TASK_DESCRIPTION = (
    "A graduate student in computer science is planning international conference attendance for professional "
    "development in 2026. They need to identify four specific conferences taking place in the first half of 2026 "
    "(January through July) that meet the following criteria:\n\n"
    "1. One conference focused on virtual reality and 3D user interfaces, held in Asia\n"
    "2. One conference focused on educational research, held in California, USA\n"
    "3. One conference focused on emerging researchers in STEM fields, held in the southeastern United States\n"
    "4. One international conference on virtual reality, held in Europe\n\n"
    "For each of the four conferences identified, provide the following information:\n"
    "- Full official conference name\n"
    "- Exact dates (including both start and end dates)\n"
    "- Host city\n"
    "- Host country (or state, for US conferences)\n"
    "- Specific venue name (if publicly available)"
)

# Regional helper definitions for additional instructions in verification
SOUTHEASTERN_US_STATES = [
    "Alabama", "Arkansas", "Florida", "Georgia", "Kentucky", "Louisiana",
    "Mississippi", "North Carolina", "South Carolina", "Tennessee",
    "Virginia", "West Virginia"
]
SOUTHEASTERN_US_ABBR = [
    "AL", "AR", "FL", "GA", "KY", "LA", "MS", "NC", "SC", "TN", "VA", "WV"
]

# Compact reference lists for region hints (non-exhaustive but helpful)
ASIA_COUNTRIES_HINT = [
    "China", "Japan", "South Korea", "Korea", "North Korea", "India", "Singapore", "Malaysia", "Thailand", "Vietnam",
    "Indonesia", "Philippines", "Taiwan", "Hong Kong", "Macao", "Macau", "Mongolia", "Bhutan", "Nepal", "Bangladesh",
    "Pakistan", "Sri Lanka", "Maldives", "Brunei", "Cambodia", "Laos", "Myanmar", "United Arab Emirates", "UAE",
    "Saudi Arabia", "Qatar", "Bahrain", "Kuwait", "Oman", "Jordan", "Israel", "Lebanon", "Turkey", "Iran", "Iraq",
    "Kazakhstan", "Kyrgyzstan", "Tajikistan", "Turkmenistan", "Uzbekistan", "Georgia", "Armenia", "Azerbaijan"
]

EUROPE_COUNTRIES_HINT = [
    "United Kingdom", "UK", "England", "Scotland", "Wales", "Northern Ireland", "Ireland", "France", "Germany",
    "Spain", "Portugal", "Italy", "Switzerland", "Austria", "Netherlands", "Belgium", "Luxembourg", "Norway",
    "Sweden", "Finland", "Denmark", "Iceland", "Poland", "Czech Republic", "Czechia", "Slovakia", "Hungary",
    "Slovenia", "Croatia", "Bosnia", "Serbia", "Montenegro", "North Macedonia", "Albania", "Greece", "Bulgaria",
    "Romania", "Moldova", "Ukraine", "Belarus", "Lithuania", "Latvia", "Estonia"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConferenceEntry(BaseModel):
    """One conference entry."""
    name: Optional[str] = None
    start_date: Optional[str] = None  # keep as string to handle various formats (e.g., "June 3, 2026")
    end_date: Optional[str] = None
    city: Optional[str] = None
    country_or_state: Optional[str] = None  # For US conferences, put the state (e.g., "California"); otherwise country.
    venue: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ConferenceExtraction(BaseModel):
    """Extraction of the four required conferences by category."""
    asia_vr_conference: Optional[ConferenceEntry] = None
    california_education_conference: Optional[ConferenceEntry] = None
    se_us_stem_conference: Optional[ConferenceEntry] = None
    europe_vr_conference: Optional[ConferenceEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
    Extract exactly four conference entries from the answer text, each corresponding to one required category.
    For each entry, extract the fields exactly as presented in the answer. Do not fabricate missing fields.

    Categories and keys to fill:
    - asia_vr_conference: a conference focused on virtual reality and/or 3D user interfaces held in Asia between January and July 2026.
    - california_education_conference: a conference focused on educational research held in California, USA between January and July 2026.
    - se_us_stem_conference: a conference focused on emerging researchers in STEM fields held in the southeastern United States between January and July 2026.
    - europe_vr_conference: an international conference on virtual reality held in Europe between January and July 2026.

    For each conference, extract:
    - name: full official conference name (string)
    - start_date: exact start date as stated (string; do not reformat)
    - end_date: exact end date as stated (string; do not reformat)
    - city: host city (string)
    - country_or_state: 
        • For any US-based conference, put the state name (e.g., "California", "Florida").
        • For non-US, put the country name (e.g., "Japan", "Germany").
    - venue: the specific venue name if provided (string), otherwise null
    - urls: an array of all reference URLs (as actual URLs); include any official site, CFP page, or host listing mentioned for that specific conference

    Important:
    - Only extract what is explicitly present in the answer.
    - For urls, extract actual URLs (plain or from markdown).
    - If a field is missing in the answer, return null (for strings) or [] (for urls).
    - Do not include any additional or alternative conferences beyond these four categories.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_valid_url(urls: List[str]) -> bool:
    return any(isinstance(u, str) and (u.startswith("http://") or u.startswith("https://")) for u in urls or [])


def _safe(s: Optional[str]) -> str:
    return s or ""


def _build_time_window_note() -> str:
    return "The event must occur within January 1 to July 31, 2026 (inclusive)."


def _asia_region_instruction() -> str:
    return (
        "Location must be in Asia. Common Asia countries include: "
        + ", ".join(ASIA_COUNTRIES_HINT)
        + ". If the page clearly indicates an Asian country/city, that qualifies. "
        + _build_time_window_note()
    )


def _europe_region_instruction() -> str:
    return (
        "Location must be in Europe. Accept UK, Switzerland, Norway, etc., alongside EU countries. Examples include: "
        + ", ".join(EUROPE_COUNTRIES_HINT)
        + ". "
        + _build_time_window_note()
    )


def _california_instruction() -> str:
    return (
        "Location must be in California, USA (the state). Confirm that the venue/city is in California. "
        + _build_time_window_note()
    )


def _se_us_instruction() -> str:
    return (
        "Location must be in the southeastern United States. Accept one of these states (full name or standard 2-letter "
        "abbreviation): "
        + ", ".join(SOUTHEASTERN_US_STATES)
        + " ("
        + ", ".join(SOUTHEASTERN_US_ABBR)
        + "). "
        + _build_time_window_note()
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_category_asia_vr(
    evaluator: Evaluator,
    parent,
    conf: ConferenceEntry
) -> None:
    """
    Asia VR/3D UI conference checks.
    """
    node = evaluator.add_parallel(
        id="Asia_VR_Conference",
        desc="Virtual reality and 3D user interfaces conference held in Asia during January-July 2026",
        parent=parent,
        critical=False
    )

    # Reference URL presence (critical)
    evaluator.add_custom_node(
        result=_has_valid_url(conf.urls if conf else []),
        id="Asia_VR_Conference_Reference",
        desc="Reference URL for the Asia VR conference information",
        parent=node,
        critical=True
    )

    # Field presence checks (critical)
    evaluator.add_custom_node(
        result=bool(conf and conf.name and conf.name.strip()),
        id="Asia_VR_Conference_Name",
        desc="The Asia VR conference has its full official name provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.start_date and conf.end_date),
        id="Asia_VR_Conference_Dates",
        desc="The Asia VR conference has its exact dates (both start and end dates) provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.city),
        id="Asia_VR_Conference_City",
        desc="The Asia VR conference has its host city provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.country_or_state),
        id="Asia_VR_Conference_Country",
        desc="The Asia VR conference has its host country provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.venue),
        id="Asia_VR_Conference_Venue",
        desc="The Asia VR conference has its specific venue name provided (if publicly available)",
        parent=node,
        critical=False
    )

    # Identified: verify constraints against sources (critical)
    identified_leaf = evaluator.add_leaf(
        id="Asia_VR_Conference_Identified",
        desc="A conference focused on virtual reality/3D user interfaces in Asia during January-July 2026 is identified",
        parent=node,
        critical=True
    )
    claim = (
        f"The conference titled '{_safe(conf.name)}' is focused on virtual reality and/or 3D user interfaces, "
        f"held in {_safe(conf.city)}, {_safe(conf.country_or_state)} in Asia, from {_safe(conf.start_date)} to {_safe(conf.end_date)}, "
        f"and those dates fall within January–July 2026."
    )
    await evaluator.verify(
        claim=claim,
        node=identified_leaf,
        sources=conf.urls if conf else [],
        additional_instruction=_asia_region_instruction()
        + " Confirm the focus (VR/AR/MR, 3D UI) from the page content. Allow reasonable synonyms (e.g., immersive technologies)."
    )


async def _verify_category_california_education(
    evaluator: Evaluator,
    parent,
    conf: ConferenceEntry
) -> None:
    """
    California educational research conference checks.
    """
    node = evaluator.add_parallel(
        id="California_Education_Conference",
        desc="Educational research conference held in California, USA during January-July 2026",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=_has_valid_url(conf.urls if conf else []),
        id="California_Education_Conference_Reference",
        desc="Reference URL for the California education conference information",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.name and conf.name.strip()),
        id="California_Education_Conference_Name",
        desc="The California education conference has its full official name provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.start_date and conf.end_date),
        id="California_Education_Conference_Dates",
        desc="The California education conference has its exact dates (both start and end dates) provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.city),
        id="California_Education_Conference_City",
        desc="The California education conference has its host city provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.country_or_state),
        id="California_Education_Conference_State",
        desc="The California education conference has its state (California) provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.venue),
        id="California_Education_Conference_Venue",
        desc="The California education conference has its specific venue name provided (if publicly available)",
        parent=node,
        critical=False
    )

    identified_leaf = evaluator.add_leaf(
        id="California_Education_Conference_Identified",
        desc="A conference focused on educational research in California during January-July 2026 is identified",
        parent=node,
        critical=True
    )
    claim = (
        f"The conference titled '{_safe(conf.name)}' is focused on educational research, "
        f"held in {_safe(conf.city)}, California, USA, from {_safe(conf.start_date)} to {_safe(conf.end_date)}, "
        f"with the dates falling within January–July 2026."
    )
    await evaluator.verify(
        claim=claim,
        node=identified_leaf,
        sources=conf.urls if conf else [],
        additional_instruction=_california_instruction()
        + " Confirm that the theme is educational research (e.g., education research, learning sciences, pedagogy). "
          "Allow reasonable synonyms."
    )


async def _verify_category_se_us_stem(
    evaluator: Evaluator,
    parent,
    conf: ConferenceEntry
) -> None:
    """
    Southeastern US emerging STEM researchers conference checks.
    """
    node = evaluator.add_parallel(
        id="SE_US_STEM_Conference",
        desc="Emerging researchers in STEM fields conference held in the southeastern United States during January-July 2026",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=_has_valid_url(conf.urls if conf else []),
        id="SE_US_STEM_Conference_Reference",
        desc="Reference URL for the southeastern US STEM conference information",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.name and conf.name.strip()),
        id="SE_US_STEM_Conference_Name",
        desc="The southeastern US STEM conference has its full official name provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.start_date and conf.end_date),
        id="SE_US_STEM_Conference_Dates",
        desc="The southeastern US STEM conference has its exact dates (both start and end dates) provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.city),
        id="SE_US_STEM_Conference_City",
        desc="The southeastern US STEM conference has its host city provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.country_or_state),
        id="SE_US_STEM_Conference_State",
        desc="The southeastern US STEM conference has its host state provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.venue),
        id="SE_US_STEM_Conference_Venue",
        desc="The southeastern US STEM conference has its specific venue name provided (if publicly available)",
        parent=node,
        critical=False
    )

    identified_leaf = evaluator.add_leaf(
        id="SE_US_STEM_Conference_Identified",
        desc="A conference focused on emerging STEM researchers in the southeastern United States during January-July 2026 is identified",
        parent=node,
        critical=True
    )
    claim = (
        f"The conference titled '{_safe(conf.name)}' is focused on emerging researchers in STEM fields, "
        f"held in {_safe(conf.city)}, {_safe(conf.country_or_state)} (a southeastern US state), "
        f"from {_safe(conf.start_date)} to {_safe(conf.end_date)}, with dates within January–July 2026."
    )
    await evaluator.verify(
        claim=claim,
        node=identified_leaf,
        sources=conf.urls if conf else [],
        additional_instruction=_se_us_instruction()
        + " Confirm the event emphasizes emerging researchers in STEM (e.g., early-career researchers, graduate students, "
          "postdocs in science/technology/engineering/mathematics)."
    )


async def _verify_category_europe_vr(
    evaluator: Evaluator,
    parent,
    conf: ConferenceEntry
) -> None:
    """
    Europe international VR conference checks.
    """
    node = evaluator.add_parallel(
        id="Europe_VR_Conference",
        desc="International virtual reality conference held in Europe during January-July 2026",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=_has_valid_url(conf.urls if conf else []),
        id="Europe_VR_Conference_Reference",
        desc="Reference URL for the Europe VR conference information",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.name and conf.name.strip()),
        id="Europe_VR_Conference_Name",
        desc="The Europe VR conference has its full official name provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.start_date and conf.end_date),
        id="Europe_VR_Conference_Dates",
        desc="The Europe VR conference has its exact dates (both start and end dates) provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.city),
        id="Europe_VR_Conference_City",
        desc="The Europe VR conference has its host city provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.country_or_state),
        id="Europe_VR_Conference_Country",
        desc="The Europe VR conference has its host country provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.venue),
        id="Europe_VR_Conference_Venue",
        desc="The Europe VR conference has its specific venue name provided (if publicly available)",
        parent=node,
        critical=False
    )

    identified_leaf = evaluator.add_leaf(
        id="Europe_VR_Conference_Identified",
        desc="An international conference focused on virtual reality in Europe during January-July 2026 is identified",
        parent=node,
        critical=True
    )
    claim = (
        f"The conference titled '{_safe(conf.name)}' is an international conference focused on virtual reality, "
        f"held in {_safe(conf.city)}, {_safe(conf.country_or_state)} in Europe, "
        f"from {_safe(conf.start_date)} to {_safe(conf.end_date)}, with dates within January–July 2026."
    )
    await evaluator.verify(
        claim=claim,
        node=identified_leaf,
        sources=conf.urls if conf else [],
        additional_instruction=_europe_region_instruction()
        + " Confirm that it is an international VR-focused conference (allow variations like immersive technologies/AR/MR when clearly VR-centric)."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the H1 2026 conference planning task.
    """
    # Initialize evaluator with parallel aggregation at root (matches rubric structure)
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

    # Optionally mirror the rubric root node as a child container (but we can reuse root directly).
    # We'll use root directly to hold the four category subtrees.

    # Extract the four conferences as structured data
    extracted = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferenceExtraction,
        extraction_name="extracted_conferences"
    )

    # Build and verify each category subtree
    await _verify_category_asia_vr(
        evaluator, root, extracted.asia_vr_conference or ConferenceEntry()
    )
    await _verify_category_california_education(
        evaluator, root, extracted.california_education_conference or ConferenceEntry()
    )
    await _verify_category_se_us_stem(
        evaluator, root, extracted.se_us_stem_conference or ConferenceEntry()
    )
    await _verify_category_europe_vr(
        evaluator, root, extracted.europe_vr_conference or ConferenceEntry()
    )

    # Return the evaluation summary
    return evaluator.get_summary()