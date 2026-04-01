import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from dateutil.relativedelta import relativedelta

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ra_vip_ticket"
TASK_DESCRIPTION = """
Identify an event listed on Resident Advisor (ra.co) that will take place in the United Kingdom within the next six months, that has VIP tickets for sale and details benefits about VIP tickets in the event description. Provide its link on Resident Advisor and list the benefits of VIP tickets.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RAEvent(BaseModel):
    """Information about a Resident Advisor event."""
    event_url: Optional[str]
    vip_benefits: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_event_info() -> str:
    return """
    Extract the following information about the Resident Advisor event mentioned in the answer:
    1. The event's URL on Resident Advisor (starting with "https://ra.co/" or containing "ra.co")
    2. A list of benefits associated with VIP tickets mentioned in the answer (each benefit as a separate item in the list. but each of them should be as detailed as possible as they are mentioned in the answer).

    If the information is not explicitly mentioned in the answer, return null for the corresponding field.
    For the list of VIP benefits, extract each distinct benefit as a separate item. If no benefits are mentioned, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                           #
# --------------------------------------------------------------------------- #
def is_valid_ra_url(url: Optional[str]) -> bool:
    """Check if a URL is a valid Resident Advisor event URL."""
    if not url:
        return False
    return "ra.co" in url.lower()


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_ra_url_exists(
        evaluator: Evaluator,
        parent_node,
        event_info: RAEvent
):
    """
    Verify that a valid Resident Advisor URL is provided and accessible.
    This is critical as we need a valid URL to verify other aspects.
    """
    # Create parallel container for URL verification
    url_verification = evaluator.add_parallel(
        id="ra_url_verification",
        desc="Verify Resident Advisor URL exists and is accessible",
        parent=parent_node,
        critical=True
    )
    
    # Add existence check
    url_exists_check = evaluator.add_custom_node(
        result=bool(event_info.event_url) and is_valid_ra_url(event_info.event_url),
        id="url_exists",
        desc="Check if a valid RA URL is provided",
        parent=url_verification,
        critical=True
    )
    
    # Add accessibility verification
    url_accessibility_node = evaluator.add_leaf(
        id="url_accessibility",
        desc="The URL is a valid, accessible Resident Advisor event page with event details",
        parent=url_verification,
        critical=True
    )
    
    # Always verify - let the framework handle missing URLs
    claim = f"The URL {event_info.event_url} is a valid, accessible Resident Advisor event page with event details"
    await evaluator.verify(
        claim=claim,
        node=url_accessibility_node,
        sources=[event_info.event_url] if event_info.event_url else []
    )


async def verify_event_in_uk(
        evaluator: Evaluator,
        parent_node,
        event_info: RAEvent
):
    """
    Verify that the event is located in the United Kingdom.
    """
    uk_node = evaluator.add_leaf(
        id="uk_location_check",
        desc="The event is located in the United Kingdom (England, Scotland, Wales, or Northern Ireland)",
        parent=parent_node,
        critical=True
    )
    
    claim = "The event referenced by this URL takes place in the United Kingdom (England, Scotland, Wales, or Northern Ireland)"
    await evaluator.verify(
        claim=claim,
        node=uk_node,
        sources=[event_info.event_url] if event_info.event_url else []
    )


async def verify_event_within_six_months(
        evaluator: Evaluator,
        parent_node,
        event_info: RAEvent
):
    """
    Verify that the event takes place within the next six months from today.
    """
    time_node = evaluator.add_leaf(
        id="time_frame_check",
        desc="The event takes place within the next six months from today",
        parent=parent_node,
        critical=True
    )
    
    current_date = datetime.utcnow().date()
    six_months_later = current_date + relativedelta(months=6)
    current_date_in_natural = current_date.strftime("%B %d, %Y")
    six_months_in_natural = six_months_later.strftime("%B %d, %Y")
    claim = (
        f"Check whether this event on this page takes place within the next six months "
        f"(between {current_date_in_natural} and {six_months_in_natural})."
    )
    
    await evaluator.verify(
        claim=claim,
        node=time_node,
        sources=[event_info.event_url] if event_info.event_url else []
    )


async def verify_vip_tickets_available(
        evaluator: Evaluator,
        parent_node,
        event_info: RAEvent
):
    """
    Verify that VIP tickets are available for sale for the event.
    """
    vip_node = evaluator.add_leaf(
        id="vip_tickets_check",
        desc="VIP tickets are available for sale for the event",
        parent=parent_node,
        critical=True
    )
    
    claim = "VIP tickets are available for sale for this event (VIP tickets are mentioned as being on sale or purchasable)"
    await evaluator.verify(
        claim=claim,
        node=vip_node,
        sources=[event_info.event_url] if event_info.event_url else []
    )


async def verify_vip_benefits_listed(
        evaluator: Evaluator,
        parent_node,
        event_info: RAEvent
):
    """
    Verify that VIP benefits are listed in the answer.
    """
    # Create parallel container for benefits verification
    benefits_verification = evaluator.add_parallel(
        id="benefits_verification",
        desc="Verify VIP benefits are listed and accurate",
        parent=parent_node,
        critical=True
    )
    
    # Add existence check for benefits
    benefits_exist_check = evaluator.add_custom_node(
        result=bool(event_info.vip_benefits) and len(event_info.vip_benefits) > 0,
        id="benefits_exist",
        desc="Check if VIP benefits are listed in the answer",
        parent=benefits_verification,
        critical=True
    )
    
    # Verify that the VIP benefits listed in the answer are actually mentioned in the event description.
    benefits_accuracy_node = evaluator.add_leaf(
        id="benefits_accuracy",
        desc="The VIP benefits listed in the answer are actually mentioned in the event description or details",
        parent=parent_node,
        critical=True
    )
    
    # Always verify - let the framework handle missing data
    benefits_list = ", ".join(event_info.vip_benefits) if event_info.vip_benefits else "no benefits listed"
    claim = f"The following VIP ticket benefits are mentioned or described on this event page: {benefits_list}"
    
    await evaluator.verify(
        claim=claim,
        node=benefits_accuracy_node,
        sources=[event_info.event_url] if event_info.event_url else []
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer to the Resident Advisor VIP ticket task.

    The evaluation checks sequentially:
    1. If a valid RA event URL is provided and accessible
    2. If the event is in the UK (based on URL)
    3. If the event is within the next six months (based on URL)
    4. If VIP tickets are available for sale (based on URL)
    5. If VIP benefits are listed in the answer and are reasonable
    6. If the listed benefits are actually mentioned in the event description
    """
    # Initialize evaluator with sequential strategy for dependency handling
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract event information from the answer
    event_info = await evaluator.extract(
        prompt=prompt_extract_event_info(),
        template_class=RAEvent,
        extraction_name="event_info"
    )

    # Create all verification nodes - the framework will handle sequential short-circuiting
    await verify_ra_url_exists(evaluator, root, event_info)
    await verify_event_in_uk(evaluator, root, event_info)
    await verify_event_within_six_months(evaluator, root, event_info)
    await verify_vip_tickets_available(evaluator, root, event_info)
    await verify_vip_benefits_listed(evaluator, root, event_info)

    # Add extracted info as custom info for the summary
    evaluator.add_custom_info({
        "extracted_url": event_info.event_url,
        "extracted_benefits_count": len(event_info.vip_benefits) if event_info.vip_benefits else 0,
        "extracted_benefits": event_info.vip_benefits,
        "is_valid_ra_url": is_valid_ra_url(event_info.event_url)
    }, "extraction_summary")

    # Return structured result using the new summary format
    return evaluator.get_summary()
