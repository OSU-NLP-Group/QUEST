import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs_top_confs_2026"
TASK_DESCRIPTION = (
    "I am a graduate student planning to attend major computer science conferences in 2026 to present my research in "
    "machine learning and computer vision. I need to identify conferences that are: (1) Recognized as top-tier venues "
    "in the field (CORE A* or A-ranked, or equivalent standing), (2) Scheduled to take place in 2026, "
    "(3) Focused on machine learning, artificial intelligence, or computer vision research areas, and (4) Offering "
    "graduate student travel grants or financial support programs for conference attendance. Please identify three such "
    "conferences and provide the following information for each: the full conference name and its commonly used acronym, "
    "the exact dates (month and days) when the conference will be held in 2026, the specific venue location (city and country), "
    "evidence of the conference's top-tier status in the field, a link to the official conference website, confirmation that the "
    "conference offers graduate student travel grants or support, and a reference link to information about the travel grant program."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConferenceItem(BaseModel):
    # Identification
    name: Optional[str] = None
    acronym: Optional[str] = None

    # Official site
    official_url: Optional[str] = None

    # Schedule and location
    dates_2026: Optional[str] = None  # e.g., "June 10–15, 2026"
    location_city: Optional[str] = None
    location_country: Optional[str] = None

    # Top-tier recognition
    top_tier_evidence: Optional[str] = None  # e.g., "CORE A*" or short text
    top_tier_urls: List[str] = Field(default_factory=list)  # Prefer CORE ranking page(s), or reputable equivalents

    # Research area focus
    research_area: Optional[str] = None  # e.g., "Machine Learning", "Artificial Intelligence", "Computer Vision"
    research_area_urls: List[str] = Field(default_factory=list)  # Official "About" page or Wikipedia/ACM/IEEE page

    # Student travel support
    travel_grant_urls: List[str] = Field(default_factory=list)  # Official travel grant/support pages

    # Any other references explicitly cited in the answer that could help verification
    other_urls: List[str] = Field(default_factory=list)


class ConferencesExtraction(BaseModel):
    conferences: List[ConferenceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
    Extract up to three conferences mentioned in the answer that satisfy the user's request. Preserve the original order.
    For each conference, extract the following fields:

    1) name: The full official conference name (string).
    2) acronym: The commonly used acronym (string), e.g., "NeurIPS", "ICML", "CVPR".
    3) official_url: The URL to the official 2026 conference website, or the official conference website if the 2026 site is not yet available.
    4) dates_2026: The exact dates for the 2026 edition (month and day range), as a single text string. Include the year "2026" if present in the answer; otherwise provide the dates text as given (e.g., "June 10–15, 2026", "Oct 5-10, 2026").
    5) location_city: The host city for the 2026 edition.
    6) location_country: The host country for the 2026 edition.
    7) top_tier_evidence: Short text summarizing the ranking evidence (e.g., "CORE A*", "CORE A", "recognized as top-tier").
    8) top_tier_urls: An array of one or more URLs that explicitly support the top-tier status. Prefer CORE ranking pages; acceptable equivalents include CCF A, or reputable scholarly/association listings that clearly denote top-tier standing.
    9) research_area: A brief text indicating the conference's research domain (e.g., "Machine Learning", "Artificial Intelligence", "Computer Vision").
    10) research_area_urls: An array of one or more URLs confirming the conference's research area focus (official "About" page preferred; reputable pages like ACM/IEEE/Wikipedia acceptable).
    11) travel_grant_urls: An array of one or more URLs that provide information about graduate student travel grants or financial support for attending this conference (may be on the conference site or the organizer's official site).
    12) other_urls: Any additional URLs explicitly cited in the answer for this conference.

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer. Do not fabricate URLs.
    - Ensure all URLs are complete and valid. If a URL is missing a protocol, prepend "http://".
    - If any field is not provided in the answer for a conference, set it to null (for strings) or an empty array (for lists).
    - Return a JSON object with a "conferences" array of length at most 3, each element conforming to the specified schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def valid_urls(urls: List[str]) -> List[str]:
    out = []
    for u in urls or []:
        if isinstance(u, str) and u.strip():
            out.append(u.strip())
    return out


def pick_display_name(conf: ConferenceItem) -> str:
    if nonempty(conf.acronym) and nonempty(conf.name):
        return f"{conf.name} ({conf.acronym})"
    if nonempty(conf.name):
        return conf.name.strip()
    if nonempty(conf.acronym):
        return conf.acronym.strip()
    return "the conference"


# --------------------------------------------------------------------------- #
# Verification for a single conference                                        #
# --------------------------------------------------------------------------- #
async def verify_one_conference(
    evaluator: Evaluator,
    parent_node,
    conf: ConferenceItem,
    index: int,
) -> None:
    # Create a parallel node for each conference (non-critical to allow partial across conferences)
    conf_node = evaluator.add_parallel(
        id=f"conference_{index+1}",
        desc=f"Conference #{index+1} verification (meets all specified criteria)",
        parent=parent_node,
        critical=False,
    )

    display_name = pick_display_name(conf)

    # 1) Conference name and acronym (existence check as critical)
    name_acr_ok = nonempty(conf.name) and nonempty(conf.acronym)
    evaluator.add_custom_node(
        result=name_acr_ok,
        id=f"conf_{index}_name_and_acronym",
        desc="Provide the full name and commonly used acronym of the conference",
        parent=conf_node,
        critical=True,
    )

    # 2) Official website URL (existence check as critical)
    official_ok = nonempty(conf.official_url)
    evaluator.add_custom_node(
        result=official_ok,
        id=f"conf_{index}_official_website_url",
        desc="Provide the URL to the official conference website",
        parent=conf_node,
        critical=True,
    )

    # 3) Conference dates 2026 (verify against official site)
    dates_node = evaluator.add_leaf(
        id=f"conf_{index}_dates_2026",
        desc="Specify the exact dates (month and days) when the conference will be held in 2026",
        parent=conf_node,
        critical=True,
    )
    # Use only official site for schedule if available; if missing, verification will be skipped due to failed prerequisite
    await evaluator.verify(
        claim=f"According to the official website, {display_name} 2026 will be held on '{conf.dates_2026}' in 2026.",
        node=dates_node,
        sources=conf.official_url if official_ok else None,
        additional_instruction=(
            "Verify the dates for the 2026 edition on the official site. "
            "Minor formatting variants (e.g., hyphen vs en-dash) are acceptable. "
            "If multiple event components exist (tutorials/workshops), use the main conference dates."
        ),
    )

    # 4) Venue location (verify against official site)
    venue_node = evaluator.add_leaf(
        id=f"conf_{index}_venue_location",
        desc="Identify the specific city and country where the conference will take place",
        parent=conf_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"According to the official website, {display_name} 2026 will take place in {conf.location_city}, {conf.location_country}.",
        node=venue_node,
        sources=conf.official_url if official_ok else None,
        additional_instruction="Confirm the 2026 host city and country on the official site (or the 2026 landing/venue page).",
    )

    # 5) Top-tier recognition (verify by ranking/evidence URLs)
    top_tier_node = evaluator.add_leaf(
        id=f"conf_{index}_top_tier_recognition",
        desc="Provide evidence that this is a top-tier conference (e.g., CORE A* or A ranking, or equivalent recognition in the field)",
        parent=conf_node,
        critical=True,
    )
    top_tier_sources = valid_urls(conf.top_tier_urls)
    await evaluator.verify(
        claim=(
            f"The cited page(s) indicate that {display_name} is recognized as a top-tier conference "
            f"(e.g., CORE A* or A-ranked, or an equivalent top-tier designation)."
        ),
        node=top_tier_node,
        sources=top_tier_sources if top_tier_sources else None,
        additional_instruction=(
            "Accept explicit CORE A* or A ranking pages. Also acceptable: CCF A or other reputable classifications "
            "that clearly indicate top-tier status. Allow acronym/full-name fuzzy matching."
        ),
    )

    # 6) Research area focus (verify the domain is ML/AI/CV)
    area_node = evaluator.add_leaf(
        id=f"conf_{index}_research_area_focus",
        desc="Confirm the conference focuses on machine learning, artificial intelligence, or computer vision",
        parent=conf_node,
        critical=True,
    )
    area_sources = valid_urls(conf.research_area_urls) or ([conf.official_url] if official_ok else [])
    await evaluator.verify(
        claim=(
            f"{display_name} primarily focuses on one or more of these areas: machine learning, artificial intelligence, or computer vision."
        ),
        node=area_node,
        sources=area_sources if area_sources else None,
        additional_instruction=(
            "Use the official 'About' page or reputable sources to confirm domain focus. "
            "Fuzzy match acceptable (e.g., 'neural information processing' → ML/AI)."
        ),
    )

    # 7) Student travel support exists (verify with travel grant/support page)
    travel_support_node = evaluator.add_leaf(
        id=f"conf_{index}_student_travel_support",
        desc="Verify that the conference or its organizing body offers graduate student travel grants or financial support for conference attendance",
        parent=conf_node,
        critical=True,
    )
    travel_sources = valid_urls(conf.travel_grant_urls)
    await evaluator.verify(
        claim=(
            f"The cited page(s) indicate that there is a graduate student travel grant or financial support program "
            f"available for attendance at {display_name} (for the conference or via its organizing body)."
        ),
        node=travel_support_node,
        sources=travel_sources if travel_sources else None,
        additional_instruction=(
            "Look for terms such as 'travel grant', 'travel support', 'student grants', 'financial assistance'. "
            "Support offered by the official organizer/society for this conference is acceptable."
        ),
    )

    # 8) Travel grant information URL is valid/relevant (verify the page is indeed about the program)
    travel_url_node = evaluator.add_leaf(
        id=f"conf_{index}_travel_grant_information_url",
        desc="Provide a URL reference for the travel grant program details",
        parent=conf_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The cited page provides information about student travel grants or travel support for {display_name}.",
        node=travel_url_node,
        sources=travel_sources if travel_sources else None,
        additional_instruction=(
            "The page should clearly describe travel grants/support for students for this conference (or via its organizers). "
            "If multiple URLs are provided, any single page that explicitly describes such support suffices."
        ),
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
    # Initialize evaluator with a parallel root
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

    # Add a top-level grouping node. The original JSON marks this as critical,
    # but critical parents cannot have non-critical children in this framework,
    # so we set it to non-critical to allow partial credit across conferences.
    top_node = evaluator.add_parallel(
        id="Identify_Conferences",
        desc="Identify three top-tier computer science conferences in 2026 that focus on ML/AI/CV, with complete details",
        parent=root,
        critical=False,
    )

    # Extract structured conferences info
    extracted = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferencesExtraction,
        extraction_name="conferences_extraction",
    )

    # Prepare exactly three conference entries (pad with empty if fewer)
    confs = list(extracted.conferences[:3])
    while len(confs) < 3:
        confs.append(ConferenceItem())

    # Build and verify each conference subtree
    for i, conf in enumerate(confs):
        await verify_one_conference(evaluator, top_node, conf, i)

    # Return structured evaluation summary
    return evaluator.get_summary()