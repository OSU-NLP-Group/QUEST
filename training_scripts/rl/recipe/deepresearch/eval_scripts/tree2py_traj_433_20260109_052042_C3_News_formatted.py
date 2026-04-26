import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_intl_orgs_withdrawal_investigation"
TASK_DESCRIPTION = (
    "In early 2025, the Trump administration issued an Executive Order directing a comprehensive review of all "
    "international organizations to which the United States is a member. This review was completed and resulted in a "
    "Presidential Memorandum in January 2026 announcing the withdrawal from dozens of organizations.\n\n"
    "Please provide a detailed report addressing the following:\n\n"
    "1. Identify the specific Executive Order (including number and date) that mandated this review of international organizations.\n\n"
    "2. What deadline did this Executive Order specify for completing the review?\n\n"
    "3. Which U.S. government official was designated to conduct this review, and on what date was this person confirmed by the Senate to their position?\n\n"
    "4. The UN Framework Convention on Climate Change (UNFCCC) was among the organizations identified for withdrawal. "
    "Confirm that UNFCCC is listed in the withdrawal announcement and identify the city and country where the UNFCCC secretariat is headquartered.\n\n"
    "5. As additional context, provide the year UNFCCC was established and the date the United States ratified this convention.\n\n"
    "For each factual claim, you must provide supporting reference URLs from reliable sources."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ExecutiveOrderExtraction(BaseModel):
    eo_number: Optional[str] = None
    eo_date: Optional[str] = None
    eo_urls: List[str] = Field(default_factory=list)
    review_deadline_text: Optional[str] = None  # e.g., "180-day deadline", "within 180 days"
    deadline_urls: List[str] = Field(default_factory=list)


class OfficialExtraction(BaseModel):
    role: Optional[str] = None  # e.g., "Secretary of State"
    name: Optional[str] = None  # e.g., "Marco Rubio"
    confirmation_date: Optional[str] = None  # e.g., "January 20, 2025"
    role_urls: List[str] = Field(default_factory=list)
    name_urls: List[str] = Field(default_factory=list)
    confirmation_urls: List[str] = Field(default_factory=list)


class UNFCCCExtraction(BaseModel):
    # Withdrawal memo listing
    withdrawal_listing_urls: List[str] = Field(default_factory=list)
    # Secretariat HQ info
    headquarters_location: Optional[str] = None  # e.g., "Bonn, Germany"
    headquarters_urls: List[str] = Field(default_factory=list)
    # Establishment and ratification context
    establishment_year: Optional[str] = None  # e.g., "1992"
    establishment_urls: List[str] = Field(default_factory=list)
    us_ratification_date: Optional[str] = None  # e.g., "October 7, 1992"
    us_ratification_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_eo() -> str:
    return """
    Extract the Executive Order details from the answer. You must return:
    - eo_number: the Executive Order number mentioned (string; do not invent)
    - eo_date: the Executive Order date mentioned (string; keep full month-day-year if provided)
    - eo_urls: a list of URLs explicitly cited in the answer that support the EO identification (official sources preferred such as whitehouse.gov, federalregister.gov). If none, return an empty list.
    - review_deadline_text: the text describing the deadline specified by the EO for completing the review (e.g., "180 days", "180-day deadline"). If not mentioned, return null.
    - deadline_urls: a list of URLs explicitly cited in the answer that support the deadline statement. If none, return an empty list.

    Special rules for URL extraction:
    - Extract only URLs explicitly present in the answer text.
    - Normalize markdown links to the raw URL.
    - Ignore malformed URLs. If a URL lacks protocol, prepend http://.
    """


def prompt_extract_official() -> str:
    return """
    Extract the reviewing official details from the answer. Return:
    - role: the role designated to conduct the review (e.g., "Secretary of State"). If not present, return null.
    - name: the specific official's name (e.g., "Marco Rubio"). If not present, return null.
    - confirmation_date: the Senate confirmation date for this official (string; month day, year). If not present, return null.
    - role_urls: URLs cited supporting the designated role from the EO or official documentation. If none, return an empty list.
    - name_urls: URLs cited supporting that the named person held that role during the review period. If none, return an empty list.
    - confirmation_urls: URLs cited supporting the Senate confirmation date. If none, return an empty list.

    Special rules for URL extraction apply as previously described.
    """


def prompt_extract_unfccc() -> str:
    return """
    Extract UNFCCC-related information and supporting URLs from the answer. Return:
    - withdrawal_listing_urls: URLs cited showing UNFCCC is listed in the January 2026 Presidential Memorandum withdrawal announcement. If none, return an empty list.
    - headquarters_location: the stated city and country for the UNFCCC secretariat headquarters (e.g., "Bonn, Germany"). If not present, return null.
    - headquarters_urls: URLs cited supporting the headquarters location. If none, return an empty list.
    - establishment_year: the year UNFCCC was established (e.g., "1992"). If not present, return null.
    - establishment_urls: URLs cited supporting the establishment year. If none, return an empty list.
    - us_ratification_date: the date the United States ratified the UNFCCC (e.g., "October 7, 1992"). If not present, return null.
    - us_ratification_urls: URLs cited supporting the U.S. ratification date. If none, return an empty list.

    Special rules for URL extraction apply as previously described.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def unify_sources(*lists: List[str]) -> List[str]:
    """Flatten and deduplicate URLs, filter out empty strings."""
    seen = set()
    result: List[str] = []
    for lst in lists:
        for url in lst:
            if not url:
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_eo_identification(
    evaluator: Evaluator,
    parent_node,
    eo: ExecutiveOrderExtraction
) -> None:
    group = evaluator.add_parallel(
        id="Executive_Order_Identification",
        desc="Identify the specific Executive Order that mandated the review (number, date) and cite a reliable source.",
        parent=parent_node,
        critical=True
    )

    # EO Number (expected)
    eo_number_leaf = evaluator.add_leaf(
        id="EO_Number",
        desc="Executive Order number is 14199.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="The Executive Order number is 14199.",
        node=eo_number_leaf,
        additional_instruction="Judge using the answer text. This is a simple factual check on the EO number mentioned in the answer."
    )

    # EO Date (expected)
    eo_date_leaf = evaluator.add_leaf(
        id="EO_Date",
        desc="Executive Order date is February 4, 2025.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="The Executive Order date is February 4, 2025.",
        node=eo_date_leaf,
        additional_instruction="Judge using the answer text. Allow minor formatting variations (e.g., Feb. 4, 2025)."
    )

    # EO Reference URL – support number and date
    eo_ref_leaf = evaluator.add_leaf(
        id="EO_Reference_URL",
        desc="Provide a valid reference URL supporting the Executive Order number and date.",
        parent=group,
        critical=True
    )
    eo_urls = unify_sources(eo.eo_urls)
    await evaluator.verify(
        claim="This page shows Executive Order 14199 and that it was issued on February 4, 2025.",
        node=eo_ref_leaf,
        sources=eo_urls if len(eo_urls) >= 1 else None,
        additional_instruction=(
            "Verify both the EO number (14199) and the date (February 4, 2025) on the provided URL(s). "
            "Prefer official sources like federalregister.gov or whitehouse.gov. "
            "If no URL is provided or the page is irrelevant/invalid, conclude not supported."
        )
    )


async def build_eo_review_deadline(
    evaluator: Evaluator,
    parent_node,
    eo: ExecutiveOrderExtraction
) -> None:
    group = evaluator.add_parallel(
        id="Executive_Order_Review_Deadline",
        desc="State the deadline specified by the Executive Order for completing the review and cite a reliable source.",
        parent=parent_node,
        critical=True
    )

    # Deadline length (expected)
    deadline_leaf = evaluator.add_leaf(
        id="Review_Deadline_Length",
        desc="The Executive Order specifies a 180-day deadline for completing the review.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="The Executive Order specifies a 180-day deadline for completing the review.",
        node=deadline_leaf,
        additional_instruction="Judge using the answer text. Accept reasonable phrasing variations such as 'within 180 days' or '180-day deadline'."
    )

    # Deadline reference URL – support 180 days
    deadline_ref_leaf = evaluator.add_leaf(
        id="Deadline_Reference_URL",
        desc="Provide a valid reference URL supporting the 180-day deadline.",
        parent=group,
        critical=True
    )
    deadline_urls = unify_sources(eo.deadline_urls, eo.eo_urls)
    await evaluator.verify(
        claim="This page states that the Executive Order mandates completion of the review within 180 days.",
        node=deadline_ref_leaf,
        sources=deadline_urls if len(deadline_urls) >= 1 else None,
        additional_instruction=(
            "Look for explicit language like 'within 180 days' or 'no later than 180 days' on the page. "
            "Prefer official sources (federalregister.gov/whitehouse.gov). "
            "If no URL is provided or the content does not state 180 days, conclude not supported."
        )
    )


async def build_official_and_confirmation(
    evaluator: Evaluator,
    parent_node,
    official: OfficialExtraction
) -> None:
    group = evaluator.add_parallel(
        id="Reviewing_Official_and_Confirmation",
        desc="Identify the designated reviewing official and their Senate confirmation date, with supporting sources.",
        parent=parent_node,
        critical=True
    )

    # Designated official role (expected)
    role_leaf = evaluator.add_leaf(
        id="Designated_Official_Role",
        desc="The designated official to conduct the review is the Secretary of State.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="The designated official to conduct the review is the Secretary of State.",
        node=role_leaf,
        additional_instruction="Judge using the answer text. Focus on the EO assignment of responsibility."
    )

    # Designated official name (expected)
    name_leaf = evaluator.add_leaf(
        id="Designated_Official_Name",
        desc="Marco Rubio is identified as the Secretary of State who conducted the review.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="Marco Rubio is identified as the Secretary of State who conducted the review.",
        node=name_leaf,
        additional_instruction="Judge using the answer text. Allow reasonable naming variants; ensure role alignment."
    )

    # Senate confirmation date (expected)
    conf_date_leaf = evaluator.add_leaf(
        id="Senate_Confirmation_Date",
        desc="Marco Rubio's Senate confirmation date is January 20, 2025.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="Marco Rubio's Senate confirmation date is January 20, 2025.",
        node=conf_date_leaf,
        additional_instruction="Judge using the answer text. Allow minor date formatting variants."
    )

    # Role reference URL – support Secretary of State designation
    role_ref_leaf = evaluator.add_leaf(
        id="Official_Role_Reference_URL",
        desc="Provide a valid reference URL supporting that the Secretary of State was designated to conduct the review.",
        parent=group,
        critical=True
    )
    role_urls = unify_sources(official.role_urls)
    await evaluator.verify(
        claim="This page shows that the Executive Order designates the Secretary of State to conduct the review.",
        node=role_ref_leaf,
        sources=role_urls if len(role_urls) >= 1 else None,
        additional_instruction=(
            "Prefer official EO text. Verify that the EO assigns the review to the Secretary of State. "
            "If no relevant URL is provided, conclude not supported."
        )
    )

    # Name reference URL – support that Marco Rubio was Secretary of State during the review period
    name_ref_leaf = evaluator.add_leaf(
        id="Official_Name_Reference_URL",
        desc="Provide a valid reference URL supporting that Marco Rubio was Secretary of State during the review period.",
        parent=group,
        critical=True
    )
    name_urls = unify_sources(official.name_urls)
    await evaluator.verify(
        claim="This page shows that Marco Rubio served as the U.S. Secretary of State during 2025, covering the review timeframe.",
        node=name_ref_leaf,
        sources=name_urls if len(name_urls) >= 1 else None,
        additional_instruction=(
            "Prefer official State Department or Senate sources. "
            "If no relevant URL is provided, conclude not supported."
        )
    )

    # Confirmation reference URL – support the January 20, 2025 confirmation date
    conf_ref_leaf = evaluator.add_leaf(
        id="Confirmation_Reference_URL",
        desc="Provide a valid reference URL supporting Marco Rubio's Senate confirmation date.",
        parent=group,
        critical=True
    )
    conf_urls = unify_sources(official.confirmation_urls)
    await evaluator.verify(
        claim="This page shows that Marco Rubio was confirmed by the Senate on January 20, 2025 as Secretary of State.",
        node=conf_ref_leaf,
        sources=conf_urls if len(conf_urls) >= 1 else None,
        additional_instruction=(
            "Prefer official Senate records, congressional sources, or reputable press releases. "
            "If no relevant URL is provided, conclude not supported."
        )
    )


async def build_unfccc_withdrawal_and_hq(
    evaluator: Evaluator,
    parent_node,
    unfccc: UNFCCCExtraction
) -> None:
    group = evaluator.add_parallel(
        id="UNFCCC_Withdrawal_and_Headquarters",
        desc="Confirm UNFCCC is listed in the withdrawal announcement and provide UNFCCC secretariat headquarters city/country with sources.",
        parent=parent_node,
        critical=True
    )

    # UNFCCC listed in Jan 7, 2026 memo (verify via URLs)
    listed_leaf = evaluator.add_leaf(
        id="UNFCCC_Listed_In_Withdrawal_Memo",
        desc="UNFCCC is listed in the January 7, 2026 Presidential Memorandum withdrawal announcement.",
        parent=group,
        critical=True
    )
    listing_urls = unify_sources(unfccc.withdrawal_listing_urls)
    await evaluator.verify(
        claim="This page shows that UNFCCC is listed among the organizations in the January 7, 2026 Presidential Memorandum withdrawal announcement.",
        node=listed_leaf,
        sources=listing_urls if len(listing_urls) >= 1 else None,
        additional_instruction=(
            "Look for the organization list in the January 2026 memo and confirm UNFCCC appears. "
            "If no relevant URL is provided, conclude not supported."
        )
    )

    # Explicit reference URL check (redundant support, but required by rubric)
    listing_ref_leaf = evaluator.add_leaf(
        id="UNFCCC_Listing_Reference_URL",
        desc="Provide a valid reference URL showing UNFCCC in the withdrawal announcement/list.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="This page lists UNFCCC in the withdrawal announcement.",
        node=listing_ref_leaf,
        sources=listing_urls if len(listing_urls) >= 1 else None,
        additional_instruction=(
            "Confirm UNFCCC appears in the list on the provided page(s). "
            "If no URL is provided, conclude not supported."
        )
    )

    # Headquarters location (expected)
    hq_leaf = evaluator.add_leaf(
        id="UNFCCC_Secretariat_HQ_Location",
        desc="UNFCCC secretariat headquarters is in Bonn, Germany.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="The UNFCCC secretariat headquarters is in Bonn, Germany.",
        node=hq_leaf,
        additional_instruction="Judge using the answer text. Allow minor variants like 'Bonn (Germany)'."
    )

    # HQ reference URL – support Bonn, Germany
    hq_ref_leaf = evaluator.add_leaf(
        id="UNFCCC_HQ_Reference_URL",
        desc="Provide a valid reference URL supporting that the UNFCCC secretariat is headquartered in Bonn, Germany.",
        parent=group,
        critical=True
    )
    hq_urls = unify_sources(unfccc.headquarters_urls)
    await evaluator.verify(
        claim="This page states that the UNFCCC secretariat is headquartered in Bonn, Germany.",
        node=hq_ref_leaf,
        sources=hq_urls if len(hq_urls) >= 1 else None,
        additional_instruction=(
            "Prefer official UNFCCC or UN sources. "
            "If no relevant URL is provided, conclude not supported."
        )
    )


async def build_unfccc_establishment_and_ratification(
    evaluator: Evaluator,
    parent_node,
    unfccc: UNFCCCExtraction
) -> None:
    group = evaluator.add_parallel(
        id="UNFCCC_Establishment_and_US_Ratification",
        desc="Provide additional context: UNFCCC establishment year and U.S. ratification date, with sources.",
        parent=parent_node,
        critical=True
    )

    # Establishment year (expected)
    est_leaf = evaluator.add_leaf(
        id="UNFCCC_Establishment_Year",
        desc="UNFCCC establishment year is 1992.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="UNFCCC was established in 1992.",
        node=est_leaf,
        additional_instruction="Judge using the answer text. Accept variants like 'established in 1992' or 'founded in 1992'."
    )

    # Establishment reference URL – support 1992
    est_ref_leaf = evaluator.add_leaf(
        id="UNFCCC_Establishment_Reference_URL",
        desc="Provide a valid reference URL supporting the UNFCCC establishment year (1992).",
        parent=group,
        critical=True
    )
    est_urls = unify_sources(unfccc.establishment_urls)
    await evaluator.verify(
        claim="This page states that the UNFCCC was established in 1992.",
        node=est_ref_leaf,
        sources=est_urls if len(est_urls) >= 1 else None,
        additional_instruction=(
            "Prefer official UNFCCC, UN, or treaty documentation sources. "
            "If no relevant URL is provided, conclude not supported."
        )
    )

    # U.S. ratification date (expected)
    rat_leaf = evaluator.add_leaf(
        id="US_Ratification_Date",
        desc="The United States ratified UNFCCC on October 7, 1992.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="The United States ratified the UNFCCC on October 7, 1992.",
        node=rat_leaf,
        additional_instruction="Judge using the answer text. Allow minimal date formatting variations."
    )

    # Ratification reference URL – support October 7, 1992
    rat_ref_leaf = evaluator.add_leaf(
        id="US_Ratification_Reference_URL",
        desc="Provide a valid reference URL supporting the U.S. ratification date (October 7, 1992).",
        parent=group,
        critical=True
    )
    rat_urls = unify_sources(unfccc.us_ratification_urls)
    await evaluator.verify(
        claim="This page states that the United States ratified the UNFCCC on October 7, 1992.",
        node=rat_ref_leaf,
        sources=rat_urls if len(rat_urls) >= 1 else None,
        additional_instruction=(
            "Prefer official treaty records (e.g., UN Treaty Collection, U.S. Senate/State Department). "
            "If no relevant URL is provided, conclude not supported."
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
    Evaluate an answer for the U.S. International Organizations Withdrawal Investigation task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
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

    # Top-level critical aggregator mirroring rubric root
    top = evaluator.add_parallel(
        id="US_International_Organizations_Withdrawal_Investigation",
        desc="Answer all requested sub-questions about the Executive Order review, the reviewing official, and UNFCCC withdrawal details, with supporting URLs.",
        parent=root,
        critical=True
    )

    # Extract all groups concurrently
    eo_task = evaluator.extract(
        prompt=prompt_extract_eo(),
        template_class=ExecutiveOrderExtraction,
        extraction_name="executive_order_details"
    )
    official_task = evaluator.extract(
        prompt=prompt_extract_official(),
        template_class=OfficialExtraction,
        extraction_name="reviewing_official_info"
    )
    unfccc_task = evaluator.extract(
        prompt=prompt_extract_unfccc(),
        template_class=UNFCCCExtraction,
        extraction_name="unfccc_info"
    )

    eo_info, official_info, unfccc_info = await asyncio.gather(eo_task, official_task, unfccc_task)

    # Ground truth expectations (from rubric)
    evaluator.add_ground_truth({
        "eo_number_expected": "14199",
        "eo_date_expected": "February 4, 2025",
        "review_deadline_expected": "180 days",
        "official_role_expected": "Secretary of State",
        "official_name_expected": "Marco Rubio",
        "confirmation_date_expected": "January 20, 2025",
        "withdrawal_memo_date_expected": "January 7, 2026",
        "unfccc_hq_location_expected": "Bonn, Germany",
        "unfccc_establishment_year_expected": "1992",
        "us_ratification_date_expected": "October 7, 1992"
    }, gt_type="expected_values")

    # Build verification subtrees
    await build_eo_identification(evaluator, top, eo_info)
    await build_eo_review_deadline(evaluator, top, eo_info)
    await build_official_and_confirmation(evaluator, top, official_info)
    await build_unfccc_withdrawal_and_hq(evaluator, top, unfccc_info)
    await build_unfccc_establishment_and_ratification(evaluator, top, unfccc_info)

    # Summary
    return evaluator.get_summary()