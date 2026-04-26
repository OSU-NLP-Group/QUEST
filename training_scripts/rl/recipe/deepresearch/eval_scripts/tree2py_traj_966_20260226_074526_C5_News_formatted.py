import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "early_2026_gov_events"
TASK_DESCRIPTION = (
    "For each of the following three U.S. government officials, identify and provide detailed information about the "
    "specific legal proceeding, government action, or major policy announcement involving them that occurred between "
    "January 1 and February 26, 2026:\n\n"
    "1. Senator Mark Kelly (D-Arizona)\n"
    "2. Former Venezuelan President Nicolás Maduro\n"
    "3. Minnesota Governor Tim Walz\n\n"
    "For each official, you must provide:\n"
    "- The exact date of the primary event\n"
    "- The type of legal proceeding, government action, or policy announcement\n"
    "- The key outcome, result, or announcement details\n"
    "- A reference URL from a reliable news source that documents the event\n\n"
    "The events should represent significant government actions that received substantial news coverage during this time period."
)

START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 2, 26)

# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class EventEntry(BaseModel):
    exact_date: Optional[str] = None
    event_type: Optional[str] = None
    outcome: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class EventsExtraction(BaseModel):
    senator_mark_kelly: Optional[EventEntry] = None
    nicolas_maduro: Optional[EventEntry] = None
    governor_tim_walz: Optional[EventEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return (
        "Extract, for each of the three officials listed below, the single primary event described in the answer.\n"
        "For each official, extract the following fields exactly as presented in the answer:\n"
        "1) exact_date: The exact date string of the primary event (e.g., 'February 3, 2026'). If multiple dates are mentioned, pick the primary event date the answer focuses on.\n"
        "2) event_type: A short phrase describing the type of event (e.g., 'Senate hearing', 'sanctions announcement', 'executive order', 'arrest', 'indictment', 'policy announcement').\n"
        "3) outcome: A concise sentence (or two) summarizing the key outcome/result/announcement details.\n"
        "4) reference_urls: An array of all URLs cited in the answer that directly report on or document this event. Only include URLs explicitly present in the answer text. If none, return an empty array.\n\n"
        "Return a JSON object with these top-level keys mapped to each official's event: \n"
        "- senator_mark_kelly\n"
        "- nicolas_maduro\n"
        "- governor_tim_walz\n"
        "Each value should be an object with the four fields described above. Use null if the official is missing entirely from the answer."
    )


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
_MONTH_FIX = {
    "Jan.": "Jan",
    "Feb.": "Feb",
    "Mar.": "Mar",
    "Apr.": "Apr",
    "Jun.": "Jun",
    "Jul.": "Jul",
    "Aug.": "Aug",
    "Sep.": "Sep",
    "Sept.": "Sep",
    "Oct.": "Oct",
    "Nov.": "Nov",
    "Dec.": "Dec",
}


def _sanitize_date_string(s: str) -> str:
    if not s:
        return s
    t = s.strip()
    for k, v in _MONTH_FIX.items():
        t = t.replace(k, v)
    # remove ordinal suffixes (1st, 2nd, 3rd, 4th...)
    import re
    t = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', t)
    # Ensure comma space normalization
    t = t.replace(" ,", ",")
    return t


def parse_date_maybe(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    t = _sanitize_date_string(s)
    fmts = [
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%d %B %Y",
        "%d %b %Y",
    ]
    for f in fmts:
        try:
            return datetime.strptime(t, f).date()
        except Exception:
            continue
    return None


def date_in_required_window(d: Optional[date]) -> bool:
    if not d:
        return False
    return START_DATE <= d <= END_DATE


def nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u2 = u.strip()
        if not u2:
            continue
        if not (u2.startswith("http://") or u2.startswith("https://")):
            u2 = "http://" + u2
        cleaned.append(u2)
    return cleaned


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_kelly_checks(evaluator: Evaluator, parent, ev: Optional[EventEntry]) -> None:
    event_node = evaluator.add_parallel(
        id="Senator_Mark_Kelly_Event",
        desc="Event information for Senator Mark Kelly meeting all stated constraints.",
        parent=parent,
        critical=False
    )

    # Date group
    k_date_group = evaluator.add_parallel(
        id="Kelly_Event_Date_Exact_And_In_Range",
        desc="Provides an exact primary-event date, and that date is between Jan 1 and Feb 26, 2026 (inclusive).",
        parent=event_node,
        critical=True
    )
    date_provided = evaluator.add_custom_node(
        result=bool(ev and nonempty(ev.exact_date)),
        id="Kelly_Date_Provided",
        desc="Kelly: Exact primary-event date is provided.",
        parent=k_date_group,
        critical=True
    )
    parsed = parse_date_maybe(ev.exact_date if ev else None)
    date_in_range = evaluator.add_custom_node(
        result=date_in_required_window(parsed),
        id="Kelly_Date_In_Range",
        desc="Kelly: Provided date is within 2026-01-01 to 2026-02-26 inclusive.",
        parent=k_date_group,
        critical=True
    )
    k_date_supported = evaluator.add_leaf(
        id="Kelly_Date_Supported_By_URL",
        desc="Kelly: The primary event involving Sen. Mark Kelly occurred on the provided date (supported by the cited source).",
        parent=k_date_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The primary event involving Senator Mark Kelly occurred on {ev.exact_date}." if ev and ev.exact_date else "The primary event involving Senator Mark Kelly occurred on the provided date.",
        node=k_date_supported,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Verify the event date as stated in the article. Do not confuse the publication date with the occurrence date; look for language like 'on Feb. X, 2026' describing when the event happened."
    )

    # Event type meets constraint: legal proceeding or government action
    k_type_group = evaluator.add_parallel(
        id="Kelly_Event_Type_Meets_Constraint",
        desc="Specifies the event type and it is a legal proceeding or government action.",
        parent=event_node,
        critical=True
    )
    k_type_provided = evaluator.add_custom_node(
        result=bool(ev and nonempty(ev.event_type)),
        id="Kelly_Event_Type_Provided",
        desc="Kelly: Event type is provided.",
        parent=k_type_group,
        critical=True
    )
    k_type_meets = evaluator.add_leaf(
        id="Kelly_Event_Type_Is_Legal_Or_Gov_Action",
        desc="Kelly: The event qualifies as a legal proceeding or government action.",
        parent=k_type_group,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The event involving Senator Mark Kelly on {ev.exact_date} was a '{ev.event_type}', which qualifies as a legal proceeding or a government action."
               if ev and ev.exact_date and ev.event_type else
               "The event involving Senator Mark Kelly qualifies as a legal proceeding or a government action."),
        node=k_type_meets,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Accept official hearings, subpoenas, court filings, indictments, sanctions, executive orders, official agency announcements, etc., as legal proceedings or government actions. Reject mere commentary or speculation."
    )

    # Topic constraint: Pentagon or military oversight
    k_topic = evaluator.add_leaf(
        id="Kelly_Topic_Constraint",
        desc="The event relates to Pentagon actions or military oversight matters.",
        parent=event_node,
        critical=True
    )
    await evaluator.verify(
        claim="The event primarily concerns the Pentagon (Department of Defense) or congressional military oversight matters involving Senator Mark Kelly.",
        node=k_topic,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Look for mention of DoD, Pentagon, defense officials, defense policy, or a Senate/committee action overseeing military affairs."
    )

    # Outcome described
    k_outcome_group = evaluator.add_parallel(
        id="Kelly_Outcome_Described",
        desc="Describes the key outcome/result/announcement details of the event.",
        parent=event_node,
        critical=True
    )
    k_outcome_provided = evaluator.add_custom_node(
        result=bool(ev and nonempty(ev.outcome)),
        id="Kelly_Outcome_Provided",
        desc="Kelly: Outcome/result/announcement details are provided.",
        parent=k_outcome_group,
        critical=True
    )
    k_outcome_supported = evaluator.add_leaf(
        id="Kelly_Outcome_Supported_By_URL",
        desc="Kelly: The key outcome/result/announcement details are accurate per the cited source.",
        parent=k_outcome_group,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The article reports the following as the key outcome/result/announcement: '{ev.outcome}'."
               if ev and ev.outcome else
               "The article reports the stated outcome/result/announcement as described in the answer."),
        node=k_outcome_supported,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Confirm the outcome text matches the article's description of the main result or announcement for the event."
    )

    # Reference URL reliability and documentation
    k_ref_group = evaluator.add_parallel(
        id="Kelly_Reference_URL_Reliable_And_Documents_Event",
        desc="Provides at least one reliable URL that directly reports on/documents the event.",
        parent=event_node,
        critical=True
    )
    k_ref_exists = evaluator.add_custom_node(
        result=bool(ev and len(valid_urls(ev.reference_urls)) > 0),
        id="Kelly_Reference_URL_Provided",
        desc="Kelly: At least one reference URL is provided.",
        parent=k_ref_group,
        critical=True
    )
    k_ref_reliable = evaluator.add_leaf(
        id="Kelly_URL_Is_Reliable_And_Reports_Event",
        desc="Kelly: At least one provided URL is a reliable news report that directly documents the event.",
        parent=k_ref_group,
        critical=True
    )
    await evaluator.verify(
        claim=("At least one of these URLs is a reliable news report that directly documents the event involving Senator Mark Kelly on "
               f"{ev.exact_date}." if ev and ev.exact_date else
               "At least one of these URLs is a reliable news report that directly documents the event involving Senator Mark Kelly."),
        node=k_ref_reliable,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Treat established national outlets (e.g., AP, Reuters, major newspapers/networks) or reputable local newsrooms as reliable. The page should be a news report directly covering the event, not just an opinion post or unrelated content."
    )


async def build_maduro_checks(evaluator: Evaluator, parent, ev: Optional[EventEntry]) -> None:
    event_node = evaluator.add_parallel(
        id="Nicolas_Maduro_Event",
        desc="Event information for Nicolás Maduro meeting all stated constraints.",
        parent=parent,
        critical=False
    )

    # Date group
    m_date_group = evaluator.add_parallel(
        id="Maduro_Event_Date_Exact_And_In_Range",
        desc="Provides an exact primary-event date, and that date is between Jan 1 and Feb 26, 2026 (inclusive).",
        parent=event_node,
        critical=True
    )
    date_provided = evaluator.add_custom_node(
        result=bool(ev and nonempty(ev.exact_date)),
        id="Maduro_Date_Provided",
        desc="Maduro: Exact primary-event date is provided.",
        parent=m_date_group,
        critical=True
    )
    parsed = parse_date_maybe(ev.exact_date if ev else None)
    date_in_range = evaluator.add_custom_node(
        result=date_in_required_window(parsed),
        id="Maduro_Date_In_Range",
        desc="Maduro: Provided date is within 2026-01-01 to 2026-02-26 inclusive.",
        parent=m_date_group,
        critical=True
    )
    m_date_supported = evaluator.add_leaf(
        id="Maduro_Date_Supported_By_URL",
        desc="Maduro: The primary event occurred on the provided date (supported by the cited source).",
        parent=m_date_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The primary event involving Nicolás Maduro occurred on {ev.exact_date}." if ev and ev.exact_date else "The primary event involving Nicolás Maduro occurred on the provided date.",
        node=m_date_supported,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Verify the event occurrence date from the article text; do not rely on publication date."
    )

    # Event is a U.S. government action
    m_type_group = evaluator.add_parallel(
        id="Maduro_Event_Is_US_Gov_Action",
        desc="Specifies the event type and it is a U.S. government action.",
        parent=event_node,
        critical=True
    )
    m_type_provided = evaluator.add_custom_node(
        result=bool(ev and nonempty(ev.event_type)),
        id="Maduro_Event_Type_Provided",
        desc="Maduro: Event type is provided.",
        parent=m_type_group,
        critical=True
    )
    m_us_gov_action = evaluator.add_leaf(
        id="Maduro_Event_Is_US_Government_Action",
        desc="Maduro: The event was a U.S. government action.",
        parent=m_type_group,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The event involving Nicolás Maduro on {ev.exact_date} was a U.S. government action (by a U.S. federal department/agency or U.S. court), described as '{ev.event_type}'."
               if ev and ev.exact_date and ev.event_type else
               "The event involving Nicolás Maduro was a U.S. government action by a U.S. federal department/agency or court."),
        node=m_us_gov_action,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Look for actions by U.S. DOJ, DHS, DoD, Treasury, State, FBI, DEA, federal courts, etc."
    )

    # Operation constraint: involves U.S. military or law enforcement operations
    m_operation = evaluator.add_leaf(
        id="Maduro_Operation_Constraint",
        desc="The event involves U.S. military or law enforcement operations.",
        parent=event_node,
        critical=True
    )
    await evaluator.verify(
        claim="The event involved U.S. military or U.S. law enforcement operations (e.g., arrests, raids, seizures, interdictions, deployments).",
        node=m_operation,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Confirm that U.S. operational activity (military or law enforcement) is a core element of the event."
    )

    # Outcome described
    m_outcome_group = evaluator.add_parallel(
        id="Maduro_Outcome_Described",
        desc="Describes the key outcome/result/announcement details of the event.",
        parent=event_node,
        critical=True
    )
    m_outcome_provided = evaluator.add_custom_node(
        result=bool(ev and nonempty(ev.outcome)),
        id="Maduro_Outcome_Provided",
        desc="Maduro: Outcome/result/announcement details are provided.",
        parent=m_outcome_group,
        critical=True
    )
    m_outcome_supported = evaluator.add_leaf(
        id="Maduro_Outcome_Supported_By_URL",
        desc="Maduro: The key outcome/result/announcement details are accurate per the cited source.",
        parent=m_outcome_group,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The article reports the following as the key outcome/result/announcement: '{ev.outcome}'."
               if ev and ev.outcome else
               "The article reports the stated outcome/result/announcement as described in the answer."),
        node=m_outcome_supported,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Match the stated outcome to the article's description of the main result or announcement."
    )

    # Reference URLs
    m_ref_group = evaluator.add_parallel(
        id="Maduro_Reference_URL_Reliable_And_Documents_Event",
        desc="Provides at least one URL from a reliable, verifiable news source that directly reports on/documents the event.",
        parent=event_node,
        critical=True
    )
    m_ref_exists = evaluator.add_custom_node(
        result=bool(ev and len(valid_urls(ev.reference_urls)) > 0),
        id="Maduro_Reference_URL_Provided",
        desc="Maduro: At least one reference URL is provided.",
        parent=m_ref_group,
        critical=True
    )
    m_ref_reliable = evaluator.add_leaf(
        id="Maduro_URL_Is_Reliable_And_Reports_Event",
        desc="Maduro: At least one provided URL is a reliable news report that directly documents the event.",
        parent=m_ref_group,
        critical=True
    )
    await evaluator.verify(
        claim=("At least one of these URLs is a reliable news report that directly documents the event involving Nicolás Maduro on "
               f"{ev.exact_date}." if ev and ev.exact_date else
               "At least one of these URLs is a reliable news report that directly documents the event involving Nicolás Maduro."),
        node=m_ref_reliable,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Treat established national outlets or reputable local/international newsrooms as reliable; ensure the page is a report directly covering the event."
    )


async def build_walz_checks(evaluator: Evaluator, parent, ev: Optional[EventEntry]) -> None:
    event_node = evaluator.add_parallel(
        id="Governor_Tim_Walz_Event",
        desc="Event information for Minnesota Governor Tim Walz meeting all stated constraints.",
        parent=parent,
        critical=False
    )

    # Date group
    w_date_group = evaluator.add_parallel(
        id="Walz_Event_Date_Exact_And_In_Range",
        desc="Provides an exact primary-event date, and that date is between Jan 1 and Feb 26, 2026 (inclusive).",
        parent=event_node,
        critical=True
    )
    date_provided = evaluator.add_custom_node(
        result=bool(ev and nonempty(ev.exact_date)),
        id="Walz_Date_Provided",
        desc="Walz: Exact primary-event date is provided.",
        parent=w_date_group,
        critical=True
    )
    parsed = parse_date_maybe(ev.exact_date if ev else None)
    date_in_range = evaluator.add_custom_node(
        result=date_in_required_window(parsed),
        id="Walz_Date_In_Range",
        desc="Walz: Provided date is within 2026-01-01 to 2026-02-26 inclusive.",
        parent=w_date_group,
        critical=True
    )
    w_date_supported = evaluator.add_leaf(
        id="Walz_Date_Supported_By_URL",
        desc="Walz: The primary event occurred on the provided date (supported by the cited source).",
        parent=w_date_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The primary event involving Minnesota Governor Tim Walz occurred on {ev.exact_date}." if ev and ev.exact_date else "The primary event involving Minnesota Governor Tim Walz occurred on the provided date.",
        node=w_date_supported,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Verify the occurrence date from the article text (not just the publication date)."
    )

    # Event type meets constraint: major policy announcement or legislative action by Governor Walz
    w_type_group = evaluator.add_parallel(
        id="Walz_Event_Type_Meets_Constraint",
        desc="Specifies the event type and it is a major policy announcement or legislative action by Governor Tim Walz.",
        parent=event_node,
        critical=True
    )
    w_type_provided = evaluator.add_custom_node(
        result=bool(ev and nonempty(ev.event_type)),
        id="Walz_Event_Type_Provided",
        desc="Walz: Event type is provided.",
        parent=w_type_group,
        critical=True
    )
    w_type_meets = evaluator.add_leaf(
        id="Walz_Type_Is_Major_Policy_Or_Legislative",
        desc="Walz: The event is a major policy announcement or legislative action by the governor.",
        parent=w_type_group,
        critical=True
    )
    await evaluator.verify(
        claim=(f"On {ev.exact_date}, Governor Tim Walz's event was a '{ev.event_type}', which constitutes a major policy announcement or legislative action by the governor."
               if ev and ev.exact_date and ev.event_type else
               "The event constitutes a major policy announcement or legislative action by Governor Tim Walz."),
        node=w_type_meets,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Confirm that Walz announced major policy or took/advanced legislative action (e.g., bills, executive actions, formal proposals)."
    )

    # Topic constraint: relates to fraud prevention or investigation response
    w_topic = evaluator.add_leaf(
        id="Walz_Topic_Constraint",
        desc="The event relates to fraud prevention or investigation response.",
        parent=event_node,
        critical=True
    )
    await evaluator.verify(
        claim="The event primarily relates to fraud prevention measures or a response to an ongoing/completed investigation (e.g., policy reforms, enforcement steps, audits).",
        node=w_topic,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Look for explicit linkage to fraud-prevention initiatives or policy responses to investigation findings."
    )

    # Outcome described
    w_outcome_group = evaluator.add_parallel(
        id="Walz_Outcome_Described",
        desc="Describes the key outcome/result/announcement details of the event.",
        parent=event_node,
        critical=True
    )
    w_outcome_provided = evaluator.add_custom_node(
        result=bool(ev and nonempty(ev.outcome)),
        id="Walz_Outcome_Provided",
        desc="Walz: Outcome/result/announcement details are provided.",
        parent=w_outcome_group,
        critical=True
    )
    w_outcome_supported = evaluator.add_leaf(
        id="Walz_Outcome_Supported_By_URL",
        desc="Walz: The key outcome/result/announcement details are accurate per the cited source.",
        parent=w_outcome_group,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The article reports the following as the key outcome/result/announcement: '{ev.outcome}'."
               if ev and ev.outcome else
               "The article reports the stated outcome/result/announcement as described in the answer."),
        node=w_outcome_supported,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Match the stated outcome to the article's description of the main result or announcement."
    )

    # Reference URLs
    w_ref_group = evaluator.add_parallel(
        id="Walz_Reference_URL_Reliable_And_Documents_Event",
        desc="Provides at least one URL from a reliable, verifiable news source that directly reports on/documents the event.",
        parent=event_node,
        critical=True
    )
    w_ref_exists = evaluator.add_custom_node(
        result=bool(ev and len(valid_urls(ev.reference_urls)) > 0),
        id="Walz_Reference_URL_Provided",
        desc="Walz: At least one reference URL is provided.",
        parent=w_ref_group,
        critical=True
    )
    w_ref_reliable = evaluator.add_leaf(
        id="Walz_URL_Is_Reliable_And_Reports_Event",
        desc="Walz: At least one provided URL is a reliable news report that directly documents the event.",
        parent=w_ref_group,
        critical=True
    )
    await evaluator.verify(
        claim=("At least one of these URLs is a reliable news report that directly documents the event involving Governor Tim Walz on "
               f"{ev.exact_date}." if ev and ev.exact_date else
               "At least one of these URLs is a reliable news report that directly documents the event involving Governor Tim Walz."),
        node=w_ref_reliable,
        sources=valid_urls(ev.reference_urls if ev else []),
        additional_instruction="Treat established national outlets or reputable Minnesota/local newsrooms as reliable; ensure the page is a report directly covering the event."
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Add custom info about the required time window
    evaluator.add_custom_info(
        info={
            "required_window_start": START_DATE.isoformat(),
            "required_window_end": END_DATE.isoformat()
        },
        info_type="constraints",
        info_name="date_constraints"
    )

    # Build the top-level rubric node (parallel aggregator)
    top = evaluator.add_parallel(
        id="Early_2026_Government_Events",
        desc="For each of the three specified officials, provide one qualifying event between Jan 1 and Feb 26, 2026 with required attributes and a reliable news URL.",
        parent=root,
        critical=False
    )

    # Build per-official checks
    await build_kelly_checks(evaluator, top, extracted.senator_mark_kelly)
    await build_maduro_checks(evaluator, top, extracted.nicolas_maduro)
    await build_walz_checks(evaluator, top, extracted.governor_tim_walz)

    # Return evaluation summary
    return evaluator.get_summary()