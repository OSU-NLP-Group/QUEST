import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ai_ml_conference_2026_single"
TASK_DESCRIPTION = (
    "Identify one major international conference focused on Artificial Intelligence or Machine Learning that will take place in 2026. "
    "The conference should be well-established and internationally recognized (examples include conferences like ICML, NeurIPS, CVPR, ICLR, AAAI, ACL, KDD, or those ranked A* or A by CORE). "
    "For the conference you identify, provide the following information with supporting reference URLs: "
    "(1) Conference name and location: Full conference name, city, and country where it will be held; "
    "(2) Conference dates: The exact dates when the main conference will take place; "
    "(3) Paper submission deadline: The full paper submission deadline; "
    "(4) Student registration fee: The early bird registration fee for students (in USD); "
    "(5) Workshop program: Information about workshop dates or workshop offerings; "
    "(6) Author notification date: When authors will be notified of paper acceptance decisions; "
    "(7) Proceedings publication: Information about how and where accepted papers will be published; "
    "(8) Official conference website: Provide the URL to the official conference website. "
    "Ensure all information is current, accurate, and sourced from official conference websites or announcements."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NameLocation(BaseModel):
    conference_name: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TextWithSources(BaseModel):
    text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FeeInfo(BaseModel):
    usd: Optional[str] = None  # Early-bird student fee in USD (e.g., "USD 300" or "300")
    local_amount: Optional[str] = None  # e.g., "€250"
    local_currency: Optional[str] = None  # e.g., "EUR"
    converted_usd: Optional[str] = None  # If the answer converts from local currency, put the USD equivalent here as text (e.g., "≈ $270")
    unavailable: Optional[bool] = None  # True if official sources say it’s not yet published
    sources: List[str] = Field(default_factory=list)


class WorkshopInfo(BaseModel):
    info_text: Optional[str] = None  # e.g., "Workshops on Dec 2; Tutorials on Dec 1"
    dates_text: Optional[str] = None  # If explicit dates for workshops are provided
    sources: List[str] = Field(default_factory=list)


class ProceedingsInfo(BaseModel):
    venue: Optional[str] = None  # e.g., "PMLR", "IEEE Xplore", "ACM Digital Library", "ACL Anthology"
    details: Optional[str] = None  # optional free text
    sources: List[str] = Field(default_factory=list)


class ConferenceExtraction(BaseModel):
    official_website_url: Optional[str] = None

    name_location: NameLocation = Field(default_factory=NameLocation)
    main_conference_dates: TextWithSources = Field(default_factory=TextWithSources)
    submission_deadline: TextWithSources = Field(default_factory=TextWithSources)
    student_fee: FeeInfo = Field(default_factory=FeeInfo)
    workshop_program: WorkshopInfo = Field(default_factory=WorkshopInfo)
    author_notification: TextWithSources = Field(default_factory=TextWithSources)
    proceedings_publication: ProceedingsInfo = Field(default_factory=ProceedingsInfo)

    extra_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conference_info() -> str:
    return """
Extract exactly one (1) AI/ML conference mentioned in the answer that occurs in calendar year 2026. If multiple are mentioned, pick the first major one (e.g., ICML, NeurIPS, CVPR, ICLR, AAAI, ACL, KDD, IJCAI, WWW, SIGIR, ECCV, ICCV, EMNLP, UAI, ECML PKDD, etc.). Then extract the following fields using the exact wording/numbers provided in the answer text. If a required value is not stated in the answer, return null (or for boolean flags, return false). For each field that has specific factual content (name/location, dates, deadlines, fees, workshops, notifications, proceedings), extract the source URLs the answer cites for that particular field (use only URLs explicitly present in the answer).

Return a JSON object:

{
  "official_website_url": string or null,  // The official conference website URL explicitly mentioned in the answer (not news or third-party sites)

  "name_location": {
    "conference_name": string or null,
    "city": string or null,
    "country": string or null,
    "sources": [urls...]                   // URLs explicitly cited for the name and/or location
  },

  "main_conference_dates": {
    "text": string or null,                // e.g., "June 15–19, 2026"
    "sources": [urls...]
  },

  "submission_deadline": {
    "text": string or null,                // e.g., "Paper submission deadline: Jan 10, 2026 (AoE)"
    "sources": [urls...]
  },

  "student_fee": {
    "usd": string or null,                 // Early-bird student fee in USD if the answer provides it (e.g., "USD 300" or "300")
    "local_amount": string or null,        // e.g., "€250" or "250 EUR" if not USD in the answer
    "local_currency": string or null,      // e.g., "EUR", "GBP", etc.
    "converted_usd": string or null,       // If the answer provides a USD conversion (it may not appear on the official page)
    "unavailable": boolean or null,        // true only if the answer explicitly claims it is not yet published and cites an official source
    "sources": [urls...]
  },

  "workshop_program": {
    "info_text": string or null,           // e.g., "Workshops will be held Dec 2"
    "dates_text": string or null,          // e.g., "Dec 2, 2026"
    "sources": [urls...]
  },

  "author_notification": {
    "text": string or null,                // e.g., "Author notification: March 15, 2026"
    "sources": [urls...]
  },

  "proceedings_publication": {
    "venue": string or null,               // e.g., "PMLR", "IEEE Xplore", "ACM Digital Library", "ACL Anthology"
    "details": string or null,             // any additional details provided in the answer
    "sources": [urls...]
  },

  "extra_sources": [urls...]               // any other URLs the answer cites (optional)
}

Strict requirements:
- Only include URLs explicitly present in the answer. Do not invent URLs.
- If a URL lacks protocol, prepend http://
- If any field is not present in the answer, set to null (or false for booleans).
- Preserve the answer’s phrasing for date/fee texts; do not normalize or infer missing data.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_token(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def is_major_conference(name: Optional[str]) -> bool:
    if not name:
        return False
    n = _normalize_token(name)
    # A robust (but not exhaustive) set of well-established AI/ML conferences
    major_markers = [
        "icml", "internationalconferenceonmachinelearning",
        "neurips", "nips", "neuralinformationprocessingsystems",
        "iclr", "internationalconferenceonlearningrepresentations",
        "cvpr", "computervisionandpatternrecognition",
        "aaai", "aaaiconferenceonartificialintelligence",
        "acl", "annualmeetingoftheassociationforcomputationalinguistics", "associationforcomputationalinguistics",
        "kdd", "sigkdd", "acmsigkddconferenceonknowledgediscoveryanddatamining",
        "ijcai", "internationaljointconferenceonartificialintelligence",
        "iccv", "internationalconferenceoncomputervision",
        "eccv", "europeanconferenceoncomputervision",
        "emnlp", "empiricalmethodsnaturallanguageprocessing",
        "naacl", "northamericanchapteroftheassociationforcomputationalinguistics",
        "sigir", "researchanddevelopmentininformationretrieval",
        "www", "thewebconference", "worldwidewebconference", "webconf",
        "uai", "uncertaintyinartificialintelligence",
        "ecmlpkdd", "europeanconferenceonmachinelearningandprinciplesandpracticeofknowledgediscoveryindatabases",
        "icdm", "ieeeinternationalconferenceondatamining",
    ]
    return any(marker in n for marker in major_markers)


def get_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or None
    except Exception:
        return None


def unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def combine_sources(primary: List[str], fallback: Optional[str]) -> List[str]:
    items = list(primary) if primary else []
    if fallback:
        items.append(fallback)
    return unique_urls(items)


def sources_requirements_met(info: ConferenceExtraction) -> bool:
    """
    Check that official sources (URLs) are cited for the major required fields.
    This is a pragmatic presence check (not rigorous officialness verification).
    """
    # The official website should be provided:
    if not info.official_website_url:
        return False

    checks = [
        bool(info.name_location and info.name_location.sources),
        bool(info.main_conference_dates and info.main_conference_dates.sources),
        bool(info.submission_deadline and info.submission_deadline.sources),
        bool(info.student_fee and info.student_fee.sources),
        bool(info.workshop_program and info.workshop_program.sources),
        bool(info.author_notification and info.author_notification.sources),
        bool(info.proceedings_publication and info.proceedings_publication.sources),
    ]
    return all(checks)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_conference_task(evaluator: Evaluator, parent_node, info: ConferenceExtraction) -> None:
    """
    Build verification tree under the task node and run checks.
    All children are critical because the rubric treats each as essential.
    """
    # 1) Major conference verification (custom boolean membership check)
    evaluator.add_custom_node(
        result=is_major_conference(info.name_location.conference_name if info.name_location else None),
        id="Major_Conference_Verification",
        desc="The identified conference is a major, well-established, internationally recognized AI/ML conference (e.g., matches the exemplars or an equivalent major venue as stated in the question).",
        parent=parent_node,
        critical=True
    )

    # 2) Conference year is 2026 (verify with sources)
    year_node = evaluator.add_leaf(
        id="Conference_Year_2026",
        desc="The conference takes place in calendar year 2026.",
        parent=parent_node,
        critical=True
    )
    year_claim = "The main conference dates occur in the calendar year 2026."
    year_sources = combine_sources(
        info.main_conference_dates.sources if info.main_conference_dates else [],
        info.official_website_url
    )
    await evaluator.verify(
        claim=year_claim,
        node=year_node,
        sources=year_sources,
        additional_instruction="Check the event's main conference dates on the provided official pages; pass only if the main conference is clearly in 2026."
    )

    # 3) Name and location provided and accurate
    nl_node = evaluator.add_leaf(
        id="Name_and_Location_Provided_and_Accurate",
        desc="Provides the full conference name and the host city and country, and these match official conference information.",
        parent=parent_node,
        critical=True
    )
    conf_name = (info.name_location.conference_name if info.name_location else None) or ""
    city = (info.name_location.city if info.name_location else None) or ""
    country = (info.name_location.country if info.name_location else None) or ""
    nl_claim = f"The conference is named '{conf_name}', and it will be held in {city}, {country}."
    nl_sources = combine_sources(
        info.name_location.sources if info.name_location else [],
        info.official_website_url
    )
    await evaluator.verify(
        claim=nl_claim,
        node=nl_node,
        sources=nl_sources,
        additional_instruction=(
            "Judge as Incorrect if any of the required fields (conference name, city, country) is missing in the answer. "
            "Otherwise, check that the official page confirms the exact name and the host city and country."
        )
    )

    # 4) Main conference dates provided and accurate
    main_dates_node = evaluator.add_leaf(
        id="Main_Conference_Dates_Provided_and_Accurate",
        desc="Provides the exact dates for the main conference (not just workshops/tutorials unless clearly distinguished) and they match official conference information.",
        parent=parent_node,
        critical=True
    )
    md_text = (info.main_conference_dates.text if info.main_conference_dates else None) or ""
    md_claim = f"The main conference dates are {md_text}."
    md_sources = combine_sources(
        info.main_conference_dates.sources if info.main_conference_dates else [],
        info.official_website_url
    )
    await evaluator.verify(
        claim=md_claim,
        node=main_dates_node,
        sources=md_sources,
        additional_instruction=(
            "If the answer did not specify the main conference dates, judge Incorrect. "
            "If specified, verify the page explicitly shows the same main conference dates (distinct from workshop/tutorial dates)."
        )
    )

    # 5) Paper submission deadline provided and accurate
    sub_deadline_node = evaluator.add_leaf(
        id="Paper_Submission_Deadline_Provided_and_Accurate",
        desc="Provides the full paper submission deadline and it matches official conference information.",
        parent=parent_node,
        critical=True
    )
    sd_text = (info.submission_deadline.text if info.submission_deadline else None) or ""
    sd_claim = f"The full paper submission deadline is {sd_text}."
    sd_sources = combine_sources(
        info.submission_deadline.sources if info.submission_deadline else [],
        info.official_website_url
    )
    await evaluator.verify(
        claim=sd_claim,
        node=sub_deadline_node,
        sources=sd_sources,
        additional_instruction=(
            "If the answer did not provide a paper submission deadline, judge Incorrect. "
            "If provided, verify the page confirms the same deadline (including any time zone like AoE if stated)."
        )
    )

    # 6) Student early-bird fee (USD or noted unavailable)
    fee_node = evaluator.add_leaf(
        id="Student_Early_Bird_Fee_Provided_in_USD_or_Noted_Unavailable",
        desc="Provides the early-bird student registration fee in USD (or clearly converts to USD), OR explicitly states it is not yet published and provides evidence from an official source.",
        parent=parent_node,
        critical=True
    )
    fee_claim = ""
    fee_add_ins = ""
    if info.student_fee and info.student_fee.unavailable:
        fee_claim = "The early-bird student registration fee for students has not yet been published/announced on official sources."
        fee_add_ins = "Pass only if the official sources explicitly indicate fees are TBA/not yet available."
    elif info.student_fee and info.student_fee.usd:
        usd_val = info.student_fee.usd.strip()
        fee_claim = f"The early-bird student registration fee for students is {usd_val} USD."
        fee_add_ins = "Verify specifically the early-bird student fee for students; ignore other categories."
    elif info.student_fee and info.student_fee.local_amount and info.student_fee.local_currency:
        local_amt = info.student_fee.local_amount.strip()
        local_cur = info.student_fee.local_currency.strip()
        conv = (info.student_fee.converted_usd or "").strip()
        if conv:
            fee_claim = f"The early-bird student registration fee for students is {local_amt} {local_cur} (the answer also provides a conversion {conv} USD which need not appear on the official page)."
        else:
            fee_claim = f"The early-bird student registration fee for students is {local_amt} {local_cur}."
        fee_add_ins = (
            "Verify the local-currency student early-bird fee on the official page; the USD conversion is for convenience and does not need to be present on the page."
        )
    else:
        # Nothing provided: force a fail via instruction
        fee_claim = "The answer provides a concrete early-bird student registration fee (or explicitly states it's not yet published) with official evidence."
        fee_add_ins = "Judge Incorrect because the answer did not provide the student early-bird fee or an official 'not yet published' statement."

    fee_sources = combine_sources(info.student_fee.sources if info.student_fee else [], info.official_website_url)
    await evaluator.verify(
        claim=fee_claim,
        node=fee_node,
        sources=fee_sources,
        additional_instruction=fee_add_ins
    )

    # 7) Workshop program info provided
    workshop_node = evaluator.add_leaf(
        id="Workshop_Program_Info_Provided",
        desc="Provides workshop program information (e.g., workshop dates and/or official workshop offerings) consistent with official conference information.",
        parent=parent_node,
        critical=True
    )
    if info.workshop_program and info.workshop_program.dates_text:
        wk_claim = f"The conference offers workshops, scheduled on {info.workshop_program.dates_text}."
    else:
        wk_text = (info.workshop_program.info_text if info.workshop_program else None) or "a workshop program"
        wk_claim = f"The conference offers {wk_text}."
    wk_sources = combine_sources(info.workshop_program.sources if info.workshop_program else [], info.official_website_url)
    await evaluator.verify(
        claim=wk_claim,
        node=workshop_node,
        sources=wk_sources,
        additional_instruction="If the answer did not provide any workshop-related information, judge Incorrect. Otherwise verify that workshops (and dates if provided) are confirmed by the official source."
    )

    # 8) Author notification date provided and accurate
    notif_node = evaluator.add_leaf(
        id="Author_Notification_Date_Provided_and_Accurate",
        desc="Provides the author notification (acceptance decision) date and it matches official conference information.",
        parent=parent_node,
        critical=True
    )
    notif_text = (info.author_notification.text if info.author_notification else None) or ""
    notif_claim = f"The author notification (acceptance decision) date is {notif_text}."
    notif_sources = combine_sources(info.author_notification.sources if info.author_notification else [], info.official_website_url)
    await evaluator.verify(
        claim=notif_claim,
        node=notif_node,
        sources=notif_sources,
        additional_instruction="If the answer did not provide an author notification date, judge Incorrect. If provided, verify it matches the official timeline."
    )

    # 9) Proceedings publication info provided
    proc_node = evaluator.add_leaf(
        id="Proceedings_Publication_Info_Provided",
        desc="Provides information about how/where accepted papers will be published (proceedings venue/platform) consistent with official conference information.",
        parent=parent_node,
        critical=True
    )
    venue = (info.proceedings_publication.venue if info.proceedings_publication else None) or ""
    details = (info.proceedings_publication.details if info.proceedings_publication else None) or ""
    if details:
        proc_claim = f"Accepted papers will be published in {venue}. Details: {details}"
    else:
        proc_claim = f"Accepted papers will be published in {venue}."
    proc_sources = combine_sources(info.proceedings_publication.sources if info.proceedings_publication else [], info.official_website_url)
    await evaluator.verify(
        claim=proc_claim,
        node=proc_node,
        sources=proc_sources,
        additional_instruction="Verify the proceedings venue/platform (e.g., PMLR, IEEE Xplore, ACM DL, ACL Anthology) as claimed."
    )

    # 10) Official conference website URL provided (presence/format check)
    official_url_node = evaluator.add_custom_node(
        result=bool(info.official_website_url and "." in info.official_website_url),
        id="Official_Conference_Website_URL_Provided",
        desc="Provides the URL for the official conference website.",
        parent=parent_node,
        critical=True
    )

    # 11) Official sources cited for claims (presence across fields)
    evaluator.add_custom_node(
        result=sources_requirements_met(info),
        id="Official_Sources_Cited_for_Claims",
        desc="Provides supporting reference URL(s) from official conference websites or official announcements sufficient to substantiate the required fields.",
        parent=parent_node,
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
    Evaluate an answer for the single AI/ML 2026 conference task.
    """
    # Initialize evaluator (root is a non-critical container)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # parallel aggregation at top level
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

    # Extract structured conference info from the answer
    conf_info = await evaluator.extract(
        prompt=prompt_extract_conference_info(),
        template_class=ConferenceExtraction,
        extraction_name="conference_info"
    )

    # Add a critical task node under the root to reflect rubric root
    task_node = evaluator.add_parallel(
        id="Conference_Research_Task",
        desc="Evaluate whether the response identifies one valid major AI/ML conference in 2026 and provides all required conference details with official supporting sources.",
        parent=root,
        critical=True
    )

    # Build and verify all rubric leaves
    await verify_conference_task(evaluator, task_node, conf_info)

    # Optionally record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "official_domain": get_domain(conf_info.official_website_url),
            "all_sources_counts": {
                "name_location": len(conf_info.name_location.sources if conf_info.name_location else []),
                "main_conference_dates": len(conf_info.main_conference_dates.sources if conf_info.main_conference_dates else []),
                "submission_deadline": len(conf_info.submission_deadline.sources if conf_info.submission_deadline else []),
                "student_fee": len(conf_info.student_fee.sources if conf_info.student_fee else []),
                "workshop_program": len(conf_info.workshop_program.sources if conf_info.workshop_program else []),
                "author_notification": len(conf_info.author_notification.sources if conf_info.author_notification else []),
                "proceedings_publication": len(conf_info.proceedings_publication.sources if conf_info.proceedings_publication else []),
                "extra_sources": len(conf_info.extra_sources or []),
            }
        },
        info_type="diagnostics",
        info_name="extraction_diagnostics"
    )

    # Return summary
    return evaluator.get_summary()