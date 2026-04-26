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
TASK_ID = "shu_career_center_info"
TASK_DESCRIPTION = (
    "I need to visit or contact Seton Hall University's Career Center for career counseling services. "
    "What is the location of the Career Center on campus, what are the contact details (phone number and/or email for students), "
    "and what are their operating hours?"
)

EXPECTED_LOCATION = "Bayley Hall, Room 209"
EXPECTED_PHONE_DIGITS = "9737619355"  # normalized digits for (973) 761-9355
EXPECTED_PHONE_DISPLAY = "(973) 761-9355"
EXPECTED_EMAIL = "tccpirates@shu.edu"
EXPECTED_HOURS_CANONICAL = "Monday through Friday, 8:45 a.m. to 4:45 p.m."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CareerCenterExtraction(BaseModel):
    location: Optional[str] = None
    operating_hours: Optional[str] = None
    phone_numbers: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_career_center_info() -> str:
    return """
    Extract from the answer the following fields about Seton Hall University's Career Center:

    - location: The location of the Career Center as stated in the answer (e.g., "Bayley Hall, Room 209"). If not stated, null.
    - operating_hours: The operating/office hours as stated in the answer (e.g., "Monday through Friday, 8:45 a.m. to 4:45 p.m."). If not stated, null.
    - phone_numbers: An array of any phone numbers mentioned in the answer that are presented as contact for the Career Center (e.g., "(973) 761-9355"). If none, return an empty array.
    - emails: An array of any email addresses mentioned in the answer that are presented as contact for the Career Center (e.g., "tccpirates@shu.edu"). If none, return an empty array.
    - source_urls: An array of all URLs explicitly present in the answer that the answer uses as sources or references for the Career Center information (location, contact details, or hours). Extract the actual URLs only (from plain URLs or markdown links). If none, return an empty array.

    Do not infer values; only extract what is explicitly present in the answer text.
    Make sure 'source_urls' contains only valid URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_phone_digits(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())


def includes_expected_phone(phones: List[str]) -> bool:
    for p in phones:
        if normalize_phone_digits(p) == EXPECTED_PHONE_DIGITS:
            return True
    return False


def includes_expected_email(emails: List[str]) -> bool:
    return any((e or "").strip().lower() == EXPECTED_EMAIL for e in emails)


def prefer_shu_urls(urls: List[str]) -> List[str]:
    # Prefer SHU domain URLs; if none, return original list
    shu_like = [u for u in urls if isinstance(u, str) and ("shu.edu" in u.lower() or "setonhall.edu" in u.lower())]
    return shu_like if shu_like else urls


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_location_checks(evaluator: Evaluator, parent_node, info: CareerCenterExtraction):
    """
    Build and run verification for location.
    """
    loc_node = evaluator.add_parallel(
        id="Location",
        desc="The answer must correctly identify that the Career Center is located in Bayley Hall, Room 209",
        parent=parent_node,
        critical=True
    )

    # Existence check: answer includes a location string and at least one source URL
    location_exists = bool(info.location and info.location.strip())
    has_sources = bool(info.source_urls)
    evaluator.add_custom_node(
        result=location_exists and has_sources,
        id="location_exists",
        desc="Answer includes a location and provides at least one source URL",
        parent=loc_node,
        critical=True
    )

    # Match check (answer text vs expected)
    location_match_leaf = evaluator.add_leaf(
        id="location_match",
        desc="Answer's stated location matches 'Bayley Hall, Room 209'",
        parent=loc_node,
        critical=True
    )
    stated_loc = info.location or ""
    await evaluator.verify(
        claim=f"The answer's location ('{stated_loc}') refers to the same location as 'Bayley Hall, Room 209' for Seton Hall University's Career Center.",
        node=location_match_leaf,
        additional_instruction=(
            "Consider formatting variants equivalent (e.g., 'Bayley Hall Room 209', 'Bayley Hall, Rm. 209', 'Bayley Hall 209'). "
            "Treat these as the same location if they clearly refer to Bayley Hall, Room 209."
        )
    )

    # Source support check
    location_source_leaf = evaluator.add_leaf(
        id="location_source_supported",
        desc="Sources support that the Career Center is in Bayley Hall, Room 209",
        parent=loc_node,
        critical=True
    )
    srcs = prefer_shu_urls(info.source_urls)
    await evaluator.verify(
        claim="Seton Hall University's Career Center is located in Bayley Hall, Room 209.",
        node=location_source_leaf,
        sources=srcs,
        additional_instruction=(
            "Verify that at least one provided source explicitly supports or clearly indicates the Career Center location as Bayley Hall, Room 209. "
            "Allow reasonable formatting variants such as 'Bayley Hall Room 209', 'Rm. 209', or 'Suite 209'."
        )
    )


async def build_contact_checks(evaluator: Evaluator, parent_node, info: CareerCenterExtraction):
    """
    Build and run verification for contact information (phone/email).
    Requirement: The answer must provide the correct phone (973) 761-9355 and/or the student/alumni email tccpirates@shu.edu.
    """
    contact_node = evaluator.add_parallel(
        id="Contact_Information",
        desc="The answer must provide the correct phone number (973) 761-9355 and/or the student/alumni email address tccpirates@shu.edu",
        parent=parent_node,
        critical=True
    )

    # Existence + basic correctness (from the answer content) and sources present
    phone_ok = includes_expected_phone(info.phone_numbers)
    email_ok = includes_expected_email(info.emails)
    has_sources = bool(info.source_urls)
    evaluator.add_custom_node(
        result=(has_sources and (phone_ok or email_ok)),
        id="contact_in_answer_and_sources_present",
        desc="Answer includes at least one correct contact detail (phone or email) and provides at least one source URL",
        parent=contact_node,
        critical=True
    )

    # Match check (answer contains at least one correct item)
    contact_match_leaf = evaluator.add_leaf(
        id="contact_match_in_answer",
        desc="Answer includes at least one correct contact detail (phone or email)",
        parent=contact_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The answer includes at least one of the following correct contact details for Seton Hall University's Career Center: "
            "phone number (973) 761-9355 or email address tccpirates@shu.edu."
        ),
        node=contact_match_leaf,
        additional_instruction=(
            "Check the answer text itself for presence of these details. "
            "For the phone, accept standard formatting variations that normalize to 9737619355."
        )
    )

    # Source support check (accept if at least one of phone or email is supported by any provided source)
    contact_source_leaf = evaluator.add_leaf(
        id="contact_source_supported",
        desc="Sources support at least one correct contact detail (phone or email)",
        parent=contact_node,
        critical=True
    )
    srcs = prefer_shu_urls(info.source_urls)
    await evaluator.verify(
        claim=(
            "At least one of the following official contact details for Seton Hall University's Career Center is present: "
            "phone (973) 761-9355 or email tccpirates@shu.edu."
        ),
        node=contact_source_leaf,
        sources=srcs,
        additional_instruction=(
            "Pass if the source confirms either the phone number (accept formatting variants of 973-761-9355) or the email address tccpirates@shu.edu for the Career Center."
        )
    )


async def build_hours_checks(evaluator: Evaluator, parent_node, info: CareerCenterExtraction):
    """
    Build and run verification for operating hours.
    Must be Monday through Friday, 8:45 a.m. to 4:45 p.m.
    """
    hours_node = evaluator.add_parallel(
        id="Operating_Hours",
        desc="The answer must correctly state that the office hours are Monday through Friday, 8:45 a.m. to 4:45 p.m.",
        parent=parent_node,
        critical=True
    )

    # Existence: hours present in answer and at least one source
    hours_exists = bool(info.operating_hours and info.operating_hours.strip())
    has_sources = bool(info.source_urls)
    evaluator.add_custom_node(
        result=hours_exists and has_sources,
        id="hours_exist",
        desc="Answer includes operating hours and provides at least one source URL",
        parent=hours_node,
        critical=True
    )

    # Match check (answer text vs expected)
    hours_match_leaf = evaluator.add_leaf(
        id="hours_match",
        desc="Answer's stated hours match 'Monday through Friday, 8:45 a.m. to 4:45 p.m.'",
        parent=hours_node,
        critical=True
    )
    stated_hours = info.operating_hours or ""
    await evaluator.verify(
        claim=(
            f"The answer's stated hours ('{stated_hours}') correspond to 'Monday through Friday, 8:45 a.m. to 4:45 p.m.' "
            "for Seton Hall University's Career Center."
        ),
        node=hours_match_leaf,
        additional_instruction=(
            "Allow stylistic variants such as 'Mon–Fri' or 'Monday-Friday', "
            "'8:45 AM - 4:45 PM' (case-insensitive, with or without periods in a.m./p.m.), "
            "and minor punctuation differences."
        )
    )

    # Source support check
    hours_source_leaf = evaluator.add_leaf(
        id="hours_source_supported",
        desc="Sources support the hours Monday through Friday, 8:45 a.m. to 4:45 p.m.",
        parent=hours_node,
        critical=True
    )
    srcs = prefer_shu_urls(info.source_urls)
    await evaluator.verify(
        claim="Seton Hall University's Career Center office hours are Monday through Friday, 8:45 a.m. to 4:45 p.m.",
        node=hours_source_leaf,
        sources=srcs,
        additional_instruction=(
            "Verify that at least one provided source supports these hours. "
            "Accept equivalent phrasing and standard time formatting variants."
        )
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
    Evaluate an answer for Seton Hall University's Career Center information.
    """
    # Initialize evaluator
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
    extracted: CareerCenterExtraction = await evaluator.extract(
        prompt=prompt_extract_career_center_info(),
        template_class=CareerCenterExtraction,
        extraction_name="career_center_extraction"
    )

    # Add ground truth for context
    evaluator.add_ground_truth({
        "expected_location": EXPECTED_LOCATION,
        "expected_phone": EXPECTED_PHONE_DISPLAY,
        "expected_email": EXPECTED_EMAIL,
        "expected_hours": EXPECTED_HOURS_CANONICAL
    })

    # Build a critical parent node to reflect the rubric's top-level requirement
    main_node = evaluator.add_parallel(
        id="Seton_Hall_Career_Center_Information",
        desc="Verify that the answer provides accurate information about Seton Hall University's Career Center, including location, contact details, and operating hours",
        parent=root,
        critical=True
    )

    # Build checks for each aspect
    await build_location_checks(evaluator, main_node, extracted)
    await build_contact_checks(evaluator, main_node, extracted)
    await build_hours_checks(evaluator, main_node, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()