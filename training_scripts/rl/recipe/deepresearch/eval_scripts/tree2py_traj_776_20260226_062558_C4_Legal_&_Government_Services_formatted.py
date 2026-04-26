import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "foia_e_services_eligibility"
TASK_DESCRIPTION = (
    "I need to submit a Freedom of Information Act (FOIA) request to a federal agency and want to ensure they have "
    "comprehensive electronic services to facilitate the process. Please identify one federal agency that meets all of "
    "the following requirements:\n\n"
    "1. The agency must provide a dedicated online portal for submitting FOIA requests (not just postal mail or email)\n"
    "2. The agency must clearly state the 20-business-day statutory response timeline on their FOIA information webpage\n"
    "3. The agency must offer an online system that allows requesters to track the status of their submitted FOIA requests\n"
    "4. The agency must list specific FOIA office contact information, including at least an email address or phone number, on their FOIA webpage\n"
    "5. The agency must explicitly accept electronic FOIA request submissions through their online system\n\n"
    "Additionally, please note whether the agency also provides:\n"
    "- An online FOIA reading room or library containing previously released documents\n"
    "- Public access to its annual FOIA report on its website\n\n"
    "For your answer, provide the agency name, the URL to its FOIA portal or main FOIA information page, and confirmation of how it meets each requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FOIAAgencyExtraction(BaseModel):
    agency_name: Optional[str] = None
    foia_main_url: Optional[str] = None                 # The main FOIA info or portal URL the answer anchors on
    portal_url: Optional[str] = None                    # Explicit portal or submission form URL (if given)
    tracking_url: Optional[str] = None                  # Explicit tracking/status page URL (if given)
    contact_page_url: Optional[str] = None              # FOIA office contacts page URL (if given)
    reading_room_url: Optional[str] = None              # FOIA reading room / library URL (if given)
    annual_report_url: Optional[str] = None             # Annual FOIA report URL (if given)
    support_urls: List[str] = Field(default_factory=list)  # All other URLs mentioned in the answer (dedup later)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_foia_agency() -> str:
    return """
    Extract exactly one federal agency described in the answer and the URLs the answer relies on.
    If multiple agencies are mentioned, select the first one that has a URL provided.

    Return the following fields:
    - agency_name: The agency's name as written in the answer.
    - foia_main_url: The primary URL the answer uses as the agency’s FOIA portal or main FOIA information page. This should be the single most central page that the answer cites for the agency's FOIA process.
    - portal_url: If the answer explicitly provides a dedicated online submission portal URL (e.g., FOIA.gov entry or an agency eFOIA form URL), put it here; otherwise null.
    - tracking_url: If the answer provides a URL for tracking FOIA request status, put it here; otherwise null.
    - contact_page_url: If the answer provides a page URL listing FOIA office contact information (email or phone), put it here; otherwise null.
    - reading_room_url: If the answer provides a URL to a FOIA Reading Room, Electronic Reading Room, or similar library of previously released records, put it here; otherwise null.
    - annual_report_url: If the answer provides a URL where the agency’s Annual FOIA Report is accessible (can be on the agency domain or FOIA.gov), put it here; otherwise null.
    - support_urls: A list of all other URLs present in the answer (deduplicate, exclude any nulls). Include any of the above URLs as well if they appear in the answer text.

    Rules:
    - Extract only URLs explicitly present in the answer text (including markdown links).
    - Use full URLs including protocol. If the protocol is missing, prepend "http://".
    - Do not invent or infer URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_preserve_order(items: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _sources_for_portal(data: FOIAAgencyExtraction) -> List[str]:
    candidates = [data.portal_url, data.foia_main_url] + (data.support_urls or [])
    return _dedupe_preserve_order(candidates)


def _sources_for_timeline(data: FOIAAgencyExtraction) -> List[str]:
    candidates = [data.foia_main_url] + [data.contact_page_url] + [
        data.portal_url, data.tracking_url, data.reading_room_url, data.annual_report_url
    ] + (data.support_urls or [])
    return _dedupe_preserve_order(candidates)


def _sources_for_tracking(data: FOIAAgencyExtraction) -> List[str]:
    candidates = [data.tracking_url, data.portal_url, data.foia_main_url] + (data.support_urls or [])
    return _dedupe_preserve_order(candidates)


def _sources_for_contact(data: FOIAAgencyExtraction) -> List[str]:
    candidates = [data.contact_page_url, data.foia_main_url] + (data.support_urls or [])
    return _dedupe_preserve_order(candidates)


def _sources_for_electronic_submission(data: FOIAAgencyExtraction) -> List[str]:
    candidates = [data.portal_url, data.foia_main_url] + (data.support_urls or [])
    return _dedupe_preserve_order(candidates)


def _sources_for_reading_room(data: FOIAAgencyExtraction) -> List[str]:
    candidates = [data.reading_room_url, data.foia_main_url] + (data.support_urls or [])
    return _dedupe_preserve_order(candidates)


def _sources_for_annual_report(data: FOIAAgencyExtraction) -> List[str]:
    candidates = [data.annual_report_url, data.foia_main_url] + (data.support_urls or [])
    return _dedupe_preserve_order(candidates)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root, extracted: FOIAAgencyExtraction) -> None:
    agency = extracted.agency_name or "the selected agency"

    # Required criteria (critical leaves)
    node_portal = evaluator.add_leaf(
        id="online_portal",
        desc="Agency provides a dedicated online FOIA request submission portal accessible to the public",
        parent=root,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{agency} provides a dedicated online FOIA request submission portal accessible to the public (e.g., an online form such as FOIA.gov or an agency eFOIA portal).",
        node=node_portal,
        sources=_sources_for_portal(extracted),
        additional_instruction=(
            "Look for evidence of a web-based submission portal or online form for FOIA requests. "
            "Do not count mere email addresses or postal mail instructions. "
            "FOIA.gov or an agency-hosted eFOIA submission page qualifies."
        ),
    )

    node_timeline = evaluator.add_leaf(
        id="response_timeline",
        desc="Agency clearly states the 20-business-day statutory response timeline on their FOIA webpage or parent agency's FOIA webpage",
        parent=root,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The FOIA webpage for {agency} (or its parent agency) clearly states the statutory FOIA response time is 20 business days (a.k.a. 20 working days).",
        node=node_timeline,
        sources=_sources_for_timeline(extracted),
        additional_instruction=(
            "Look for an explicit statement like '20 business days' or '20 working days' to make a determination. "
            "The statement may appear on the agency’s own FOIA page or its parent department’s FOIA page."
        ),
    )

    node_tracking = evaluator.add_leaf(
        id="request_tracking",
        desc="Agency provides an online system for requesters to track the status of their submitted FOIA requests",
        parent=root,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{agency} provides an online system that allows requesters to track the status of their FOIA requests.",
        node=node_tracking,
        sources=_sources_for_tracking(extracted),
        additional_instruction=(
            "Accept pages that provide a 'track status' tool or link (including FOIA.gov's status checker if it covers this agency). "
            "Evidence should clearly indicate that requesters can check the status online."
        ),
    )

    node_contact = evaluator.add_leaf(
        id="contact_information",
        desc="Agency lists specific FOIA office contact information including email or phone number on their FOIA webpage",
        parent=root,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{agency}'s FOIA webpage lists specific FOIA office contact information that includes at least an email address or a phone number.",
        node=node_contact,
        sources=_sources_for_contact(extracted),
        additional_instruction=(
            "Look for explicit FOIA contact details such as a FOIA email (e.g., foia@...) or a phone number for the FOIA office or FOIA Requester Service Center."
        ),
    )

    node_e_submit = evaluator.add_leaf(
        id="electronic_submission",
        desc="Agency explicitly accepts electronic FOIA request submissions through an online system (not limited to postal mail)",
        parent=root,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{agency} explicitly accepts electronic FOIA request submissions via an online system (beyond email or mail).",
        node=node_e_submit,
        sources=_sources_for_electronic_submission(extracted),
        additional_instruction=(
            "Confirm that an online system or form is officially supported for submitting FOIA requests electronically. "
            "This can be the agency’s own system or FOIA.gov. Mere email or postal mail is insufficient."
        ),
    )

    # Optional criteria (non-critical leaves)
    node_reading_room = evaluator.add_leaf(
        id="reading_room",
        desc="Agency maintains an online FOIA reading room or library with previously released documents",
        parent=root,
        critical=False,
    )
    await evaluator.verify(
        claim=f"{agency} maintains an online FOIA reading room (or electronic reading room) with previously released records or frequently requested records.",
        node=node_reading_room,
        sources=_sources_for_reading_room(extracted),
        additional_instruction=(
            "Look for phrasing like 'FOIA Reading Room', 'Electronic Reading Room', 'FOIA Library', or pages that provide frequently requested records."
        ),
    )

    node_annual = evaluator.add_leaf(
        id="annual_report",
        desc="Agency makes its annual FOIA report publicly accessible on its website",
        parent=root,
        critical=False,
    )
    await evaluator.verify(
        claim=f"{agency} makes its Annual FOIA Report publicly accessible on its website (can be hosted on the agency domain or on FOIA.gov).",
        node=node_annual,
        sources=_sources_for_annual_report(extracted),
        additional_instruction=(
            "Accept annual report pages or PDFs on the agency website or FOIA.gov that clearly identify Annual FOIA Reports."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent checks
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_foia_agency(),
        template_class=FOIAAgencyExtraction,
        extraction_name="foia_agency_extraction",
    )

    # Build the tree and run verifications
    await build_and_verify_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()