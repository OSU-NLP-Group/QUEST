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
TASK_ID = "ml_conf_canada_2025"
TASK_DESCRIPTION = """
Identify the major international Machine Learning conference scheduled to take place in a Canadian city in 2025. For this conference, provide the following information: (1) The full name of the conference, (2) The specific city and country where it will be held, (3) The complete date range (start and end dates) of the main conference, (4) The name of the venue/convention center, (5) The paper submission deadline, (6) Confirmation that the conference spans at least 5 days, (7) The approximate date for acceptance notifications, (8) The author registration requirement for paper publication, (9) Confirmation of in-person presentation format, (10) Information about tutorial day(s) if offered, (11) The dates when workshops are scheduled, (12) The official conference website URL, (13) At least one authoritative reference URL. All information must be verifiable through official conference sources or reputable academic websites.
"""

YEAR = 2025

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConferenceExtraction(BaseModel):
    """Structured extraction of all requested conference fields."""
    name_full: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    main_start_date: Optional[str] = None
    main_end_date: Optional[str] = None
    venue: Optional[str] = None
    submission_deadline: Optional[str] = None
    acceptance_notification_date: Optional[str] = None
    author_registration_requirement: Optional[str] = None
    in_person_format: Optional[str] = None
    tutorial_info: Optional[str] = None
    workshop_dates: Optional[str] = None
    official_url: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conference_info() -> str:
    return f"""
    Extract the information about a major international Machine Learning conference in a Canadian city in {YEAR} from the provided answer. Return JSON with the following fields:

    - name_full: Full official name of the conference exactly as stated.
    - city: The host city.
    - country: The host country (must be Canada).
    - main_start_date: The start date of the main conference program (string as in the answer; e.g., "Dec 1, 2025").
    - main_end_date: The end date of the main conference program (string).
    - venue: The venue/convention center or facility name.
    - submission_deadline: Paper submission deadline (main track), including date and optional time zone if provided.
    - acceptance_notification_date: Approximate acceptance notification date or phrase (e.g., "late June 2025", "June 28, 2025").
    - author_registration_requirement: The policy statement as quoted or paraphrased from sources (e.g., "At least one author must register at the full in-person rate for publication").
    - in_person_format: A statement indicating presentation format (e.g., "In-person", "Hybrid with in-person presentations", etc.).
    - tutorial_info: Tutorial day(s) information if tutorials are offered; otherwise a phrase like "No tutorials" if explicitly stated in the answer.
    - workshop_dates: Dates when workshops are scheduled (string or phrase).
    - official_url: The official conference website URL (full URL including http/https).
    - reference_urls: An array of at least one authoritative reference URL (official website or reputable academic site). Extract only URLs mentioned in the answer.

    RULES:
    - Return exactly the values as presented in the answer text; do not invent missing values.
    - If any item is not mentioned, set its field to null. For reference_urls, return an empty array if none are mentioned.
    - For URLs, include the full URL. If a URL is given without protocol, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(info: ConferenceExtraction) -> List[str]:
    """Combine official and reference URLs, deduplicated."""
    urls: List[str] = []
    if info.official_url and info.official_url.strip():
        urls.append(info.official_url.strip())
    urls.extend([u.strip() for u in (info.reference_urls or []) if u and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def _has_text(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, info: ConferenceExtraction) -> None:
    """
    Build the verification tree from the rubric and execute all verifications.
    Root is a critical parallel node; all children must be critical.
    """
    root = evaluator.find_node("root")
    if root is None:
        root = evaluator.initialize(task_id=TASK_ID, strategy=AggregationStrategy.PARALLEL)

    # Prepare common sources
    all_sources = _combine_sources(info)

    # Existence (custom) gating nodes for key fields (all critical under critical root)
    evaluator.add_custom_node(
        result=_has_text(info.name_full),
        id="Conference_Name_Full_Official_Exists",
        desc="Full official conference name is provided",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(info.city) and _has_text(info.country),
        id="Location_City_and_Country_Canada_Provided",
        desc="City and country are provided",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=(_has_text(info.country) and info.country.strip().lower() == "canada"),
        id="Location_Country_Is_Canada",
        desc="The country is Canada (case-insensitive check on provided field)",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(info.main_start_date) and _has_text(info.main_end_date),
        id="Main_Conference_Date_Range_Provided",
        desc="Main conference start and end dates are provided",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(info.venue),
        id="Venue_Convention_Center_Name_Provided",
        desc="Venue/convention center name is provided",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(info.submission_deadline),
        id="Paper_Submission_Deadline_Provided",
        desc="Paper submission deadline is provided",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(info.acceptance_notification_date),
        id="Acceptance_Notification_Date_Provided",
        desc="Acceptance notification date (approx) is provided",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(info.author_registration_requirement),
        id="Author_Registration_Requirement_Provided",
        desc="Author registration requirement is provided",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(info.in_person_format),
        id="In_Person_Presentation_Format_Provided",
        desc="In-person presentation format statement is provided",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(info.tutorial_info),
        id="Tutorial_Days_Info_Provided",
        desc="Tutorial day(s) information is provided",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(info.workshop_dates),
        id="Workshop_Dates_Provided",
        desc="Workshop dates are provided",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(info.official_url),
        id="Official_Conference_Website_URL_Provided",
        desc="Official conference website URL is provided",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=(isinstance(info.reference_urls, list) and len(info.reference_urls) > 0),
        id="Authoritative_Reference_URL_Provided",
        desc="At least one authoritative reference URL is provided",
        parent=root,
        critical=True
    )

    # Leaf verifications per rubric (all critical under critical root)
    claims_and_nodes: List[Dict[str, Any]] = []

    # 1. Full official name verification
    node_name = evaluator.add_leaf(
        id="Conference_Name_Full_Official",
        desc="Provide the full official name of the conference",
        parent=root,
        critical=True
    )
    name_claim = f"The conference's full official name is '{info.name_full}'."
    claims_and_nodes.append({
        "claim": name_claim,
        "sources": all_sources,
        "node": node_name,
        "instruction": "Use official pages to check how the conference names itself. Allow minor variants (e.g., acronym + full title)."
    })

    # 2. Major international ML verification
    node_major = evaluator.add_leaf(
        id="Conference_Is_Major_International_ML",
        desc="Conference qualifies as a major international Machine Learning conference",
        parent=root,
        critical=True
    )
    major_claim = "This conference is a major international Machine Learning conference."
    claims_and_nodes.append({
        "claim": major_claim,
        "sources": all_sources,
        "node": node_major,
        "instruction": "Judge based on recognition and standing (e.g., NeurIPS, ICML, ICLR). Consider reputable sources indicating its prominence."
    })

    # 3. Location (city/country must be Canada)
    node_loc = evaluator.add_leaf(
        id="Location_City_and_Country_Canada",
        desc="Provide the specific host city and the country; the country must be Canada",
        parent=root,
        critical=True
    )
    location_claim = f"The conference is held in {info.city}, {info.country}, and the country is Canada."
    claims_and_nodes.append({
        "claim": location_claim,
        "sources": all_sources,
        "node": node_loc,
        "instruction": "Confirm city and country from official schedule or venue pages. Ensure the country is Canada."
    })

    # 4. Main conference date range (must be in YEAR)
    node_dates = evaluator.add_leaf(
        id="Main_Conference_Date_Range_2025",
        desc=f"Provide the start and end dates of the main conference; the dates must be in {YEAR}",
        parent=root,
        critical=True
    )
    dates_claim = f"The main conference runs from {info.main_start_date} to {info.main_end_date} in {YEAR}."
    claims_and_nodes.append({
        "claim": dates_claim,
        "sources": all_sources,
        "node": node_dates,
        "instruction": f"Verify the main program dates and ensure they are in {YEAR}."
    })

    # 5. Venue
    node_venue = evaluator.add_leaf(
        id="Venue_Convention_Center_Name",
        desc="Provide the name of the venue/convention center or facility where the conference is held",
        parent=root,
        critical=True
    )
    venue_claim = f"The conference venue/convention center is '{info.venue}'."
    claims_and_nodes.append({
        "claim": venue_claim,
        "sources": all_sources,
        "node": node_venue,
        "instruction": "Confirm the venue from official site or venue announcement pages. Allow minor naming variants."
    })

    # 6. Submission deadline
    node_deadline = evaluator.add_leaf(
        id="Paper_Submission_Deadline",
        desc="Provide the paper submission deadline",
        parent=root,
        critical=True
    )
    deadline_claim = f"The paper submission deadline (main track) is {info.submission_deadline}."
    claims_and_nodes.append({
        "claim": deadline_claim,
        "sources": all_sources,
        "node": node_deadline,
        "instruction": "Use the official call for papers or submission timeline. Ignore workshop deadlines."
    })

    # 7. Conference spans at least 5 days
    node_span = evaluator.add_leaf(
        id="Conference_Spans_At_Least_5_Days",
        desc="Confirm the conference spans at least 5 days",
        parent=root,
        critical=True
    )
    span_claim = f"The main conference duration from {info.main_start_date} to {info.main_end_date} spans at least five calendar days."
    claims_and_nodes.append({
        "claim": span_claim,
        "sources": all_sources,
        "node": node_span,
        "instruction": "Judge from the official schedule: inclusive duration between start and end is ≥ 5 days."
    })

    # 8. Acceptance notification date (approx)
    node_accept = evaluator.add_leaf(
        id="Acceptance_Notification_Date",
        desc="Provide the approximate acceptance notification date",
        parent=root,
        critical=True
    )
    accept_claim = f"The acceptance notifications are around {info.acceptance_notification_date}."
    claims_and_nodes.append({
        "claim": accept_claim,
        "sources": all_sources,
        "node": node_accept,
        "instruction": "Accept approximate phrasing if used by official sources (e.g., 'late June 2025')."
    })

    # 9. Author registration requirement
    node_reg = evaluator.add_leaf(
        id="Author_Registration_Requirement",
        desc="State the author registration requirement for paper publication, including that at least one author must register at the full in-person rate",
        parent=root,
        critical=True
    )
    reg_claim = "At least one author must register at the full in-person rate for the paper to be included/eligible for publication/presentation."
    claims_and_nodes.append({
        "claim": reg_claim,
        "sources": all_sources,
        "node": node_reg,
        "instruction": "Check official registration policies; equivalent statements count (require one author to register to present/publish)."
    })

    # 10. In-person presentation format
    node_format = evaluator.add_leaf(
        id="In_Person_Presentation_Format",
        desc="Confirm the conference is in-person (not virtual-only)",
        parent=root,
        critical=True
    )
    format_claim = "The 2025 conference uses an in-person presentation format (not virtual-only)."
    claims_and_nodes.append({
        "claim": format_claim,
        "sources": all_sources,
        "node": node_format,
        "instruction": "Confirm from official site; hybrid still counts as in-person if presentations require in-person attendance."
    })

    # 11. Tutorial day(s)
    node_tutorial = evaluator.add_leaf(
        id="Tutorial_Days_If_Offered",
        desc="Provide tutorial day(s) information if tutorials are offered (e.g., tutorial date(s) when available from sources)",
        parent=root,
        critical=True
    )
    tutorial_claim = f"Tutorials are offered and scheduled on {info.tutorial_info}."
    claims_and_nodes.append({
        "claim": tutorial_claim,
        "sources": all_sources,
        "node": node_tutorial,
        "instruction": "Verify tutorial schedule from official pages; if tutorials are not offered, the claim should reflect that."
    })

    # 12. Workshop dates
    node_workshop = evaluator.add_leaf(
        id="Workshop_Dates",
        desc="Provide the dates when workshops are scheduled",
        parent=root,
        critical=True
    )
    workshop_claim = f"Workshops are scheduled on {info.workshop_dates}."
    claims_and_nodes.append({
        "claim": workshop_claim,
        "sources": all_sources,
        "node": node_workshop,
        "instruction": "Verify workshop schedule from official pages."
    })

    # 13. Official website URL validity
    node_official = evaluator.add_leaf(
        id="Official_Conference_Website_URL",
        desc="Provide the official conference website URL",
        parent=root,
        critical=True
    )
    official_claim = f"This URL is the official website for the conference: {info.official_url}."
    claims_and_nodes.append({
        "claim": official_claim,
        "sources": info.official_url if _has_text(info.official_url) else None,
        "node": node_official,
        "instruction": "Assess whether the URL belongs to the conference's official domain/site."
    })

    # 14. Authoritative reference URL(s)
    node_refs = evaluator.add_leaf(
        id="Authoritative_Reference_URL",
        desc="Provide at least one authoritative reference URL",
        parent=root,
        critical=True
    )
    refs_claim = "The provided reference URL(s) are authoritative sources for the conference (official sites or reputable academic sources)."
    claims_and_nodes.append({
        "claim": refs_claim,
        "sources": info.reference_urls if info.reference_urls else None,
        "node": node_refs,
        "instruction": "Evaluate domain reputation and relevance (e.g., official conference site, reputable academic organizations)."
    })

    # 15. Verifiability from allowed sources
    node_verifiable = evaluator.add_leaf(
        id="Information_Verifiable_From_Allowed_Sources",
        desc="Ensure the provided information is verifiable via official conference sources or reputable academic websites (as evidenced by the provided URL(s))",
        parent=root,
        critical=True
    )
    verifiable_claim = "All provided information about the conference is verifiable via the supplied official or reputable academic sources."
    claims_and_nodes.append({
        "claim": verifiable_claim,
        "sources": all_sources if all_sources else None,
        "node": node_verifiable,
        "instruction": "Judge at a high level if the combined sources cover and support the provided information."
    })

    # Execute batch verification in parallel
    await evaluator.batch_verify([
        (entry["claim"], entry["sources"], entry["node"], entry["instruction"])
        for entry in claims_and_nodes
    ])


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
    Evaluate an answer for the ML conference in Canada 2025 task.
    """
    # Initialize evaluator with critical root (parallel aggregation)
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

    # Extract conference information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_conference_info(),
        template_class=ConferenceExtraction,
        extraction_name="conference_info",
    )

    # Add custom info to summary (optional)
    evaluator.add_custom_info(
        info={"combined_sources_count": len(_combine_sources(extracted_info))},
        info_type="stats",
        info_name="source_stats"
    )

    # Build tree and run verifications
    await build_and_verify_tree(evaluator, extracted_info)

    # Return structured summary
    return evaluator.get_summary()