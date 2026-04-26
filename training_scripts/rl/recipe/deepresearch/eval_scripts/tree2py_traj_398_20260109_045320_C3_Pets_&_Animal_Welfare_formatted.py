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
TASK_ID = "utah_animal_nonprofit_990_cn"
TASK_DESCRIPTION = (
    "Identify a 501(c)(3) animal welfare nonprofit organization headquartered in Utah that operates the nation's largest "
    "sanctuary for homeless animals and has a four-star rating on Charity Navigator. For this organization, provide the following "
    "information extracted from publicly available documents: (1) The organization's official legal name as it appears on IRS documentation, "
    "(2) The IRS Employer Identification Number (EIN/Tax ID), (3) The total number of voting members serving on the board of directors, "
    "(4) The organization's total revenue for the most recent fiscal year available, (5) The name and position title of the organization's "
    "highest compensated employee, and (6) Direct URL links to: (a) the organization's Charity Navigator profile page, and (b) the organization's "
    "publicly available Form 990 for the most recent fiscal year."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OrgExtraction(BaseModel):
    # Core identification
    organization_name_official: Optional[str] = None
    ein: Optional[str] = None
    headquarters_city: Optional[str] = None
    headquarters_state: Optional[str] = None

    # Qualifying claims and key URLs
    org_website_url: Optional[str] = None
    charity_navigator_profile_url: Optional[str] = None
    charity_navigator_rating_text: Optional[str] = None  # e.g., "Four-Star", "4-star", "4/4"
    form_990_url: Optional[str] = None

    # Required details from (most recent) Form 990
    board_voting_members_count: Optional[str] = None  # Keep as string; formatting may vary
    total_revenue_most_recent_fy: Optional[str] = None  # Keep as string to allow commas, $ symbols
    highest_comp_employee_name: Optional[str] = None
    highest_comp_employee_title: Optional[str] = None

    # Support for additional claims
    largest_sanctuary_claim_text: Optional[str] = None  # Any text the answer uses for the "largest sanctuary" claim

    # Any other supporting URLs explicitly cited in the answer
    additional_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_org() -> str:
    return """
    Extract the primary organization described in the answer that is claimed to be:
    - A 501(c)(3) animal welfare nonprofit organization,
    - Headquartered in Utah,
    - Operating the nation's largest sanctuary for homeless animals,
    - Having a four-star rating on Charity Navigator.

    Extract exactly as presented in the answer text (do not invent). Return a single organization object with the fields below:

    Required fields to extract (use null if missing in the answer):
    - organization_name_official: The organization's official legal name as stated (prefer the IRS/Form 990 legal name if explicitly mentioned).
    - ein: The IRS Employer Identification Number (EIN / Tax ID) as presented (allow formats with or without hyphen).
    - headquarters_city: City of headquarters (if mentioned).
    - headquarters_state: State of headquarters (if mentioned).
    - org_website_url: The organization's official website URL (if cited).
    - charity_navigator_profile_url: Direct URL to the organization's Charity Navigator profile page (if cited).
    - charity_navigator_rating_text: The rating text mentioned in the answer for Charity Navigator (e.g., "Four Star", "4-star", "four-star", "4/4").
    - form_990_url: A direct publicly accessible URL to the organization's Form 990 for the most recent fiscal year (as claimed in the answer).
    - board_voting_members_count: The total number of voting members on the board of directors (as stated or quoted from Form 990 Part VI).
    - total_revenue_most_recent_fy: The total revenue figure for the most recent fiscal year (as stated or quoted from Form 990 Part I).
    - highest_comp_employee_name: The name of the highest compensated employee (as stated or quoted from Form 990 Part VII).
    - highest_comp_employee_title: The position title of the highest compensated employee (as stated or quoted from Form 990 Part VII).
    - largest_sanctuary_claim_text: The exact phrasing used in the answer to describe the organization's claim of operating the nation's largest sanctuary for homeless animals (if any).
    - additional_source_urls: An array of any other URLs in the answer that support the claims above (e.g., IRS TEOS, ProPublica/Candid 990 pages, organization pages, reputable articles). Include only valid URLs explicitly present in the answer.

    Notes:
    - Do not fabricate URLs or facts. Only extract what's explicitly present in the answer.
    - If multiple organizations are mentioned, extract only the one that the answer presents as meeting all the criteria.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str], fallback: str = ""):  # safe string
    return s if s is not None else fallback


def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def build_all_sources(info: OrgExtraction) -> List[str]:
    base = [
        info.form_990_url,
        info.charity_navigator_profile_url,
        info.org_website_url,
    ]
    if info.additional_source_urls:
        base.extend(info.additional_source_urls)
    return _dedup_urls(base)


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_organization_identification(evaluator: Evaluator, parent_node, info: OrgExtraction) -> None:
    """
    Build and verify the 'organization_identification' parallel critical node.
    This verifies the four qualifying constraints:
    - 501(c)(3) IRS status
    - Utah headquarters
    - nation's largest sanctuary claim
    - Charity Navigator four-star rating (as shown on the CN page)
    """
    org_name = _safe(info.organization_name_official, "the organization")
    cn_url = info.charity_navigator_profile_url
    form990_url = info.form_990_url
    website = info.org_website_url
    all_sources = build_all_sources(info)

    org_node = evaluator.add_parallel(
        id="organization_identification",
        desc="Correctly identify an organization meeting all qualifying criteria.",
        parent=parent_node,
        critical=True,
    )

    # 1) 501(c)(3) status
    n_501 = evaluator.add_leaf(
        id="501c3_status",
        desc="Organization is a registered 501(c)(3) tax-exempt nonprofit with the IRS.",
        parent=org_node,
        critical=True,
    )
    claim_501 = f"The organization '{org_name}' is a registered 501(c)(3) tax‑exempt nonprofit organization with the IRS."
    await evaluator.verify(
        claim=claim_501,
        node=n_501,
        sources=all_sources,
        additional_instruction=(
            "Support can come from an official Form 990 (header indicates 501(c)(3)) or an official profile page stating "
            "501(c)(3) status. If the sources indicate any other tax status or there is no explicit evidence of 501(c)(3), mark incorrect."
        ),
    )

    # 2) Utah headquarters
    n_utah = evaluator.add_leaf(
        id="utah_headquarters",
        desc="Organization is headquartered in Utah.",
        parent=org_node,
        critical=True,
    )
    claim_utah = f"The organization '{org_name}' is headquartered in Utah."
    await evaluator.verify(
        claim=claim_utah,
        node=n_utah,
        sources=_dedup_urls([cn_url, website] + (info.additional_source_urls or [])),
        additional_instruction=(
            "Accept evidence that clearly states the headquarters location in Utah (city + Utah is acceptable). "
            "Reasonable variants allowed (e.g., 'Kanab, UT')."
        ),
    )

    # 3) Nation's largest sanctuary for homeless animals
    n_largest = evaluator.add_leaf(
        id="largest_sanctuary_status",
        desc="Organization operates the nation's largest sanctuary for homeless animals.",
        parent=org_node,
        critical=True,
    )
    # Use any claim text the answer provided, but verify the generic superlative claim
    claim_largest = (
        f"The organization '{org_name}' operates the nation's largest sanctuary for homeless animals."
    )
    await evaluator.verify(
        claim=claim_largest,
        node=n_largest,
        sources=_dedup_urls([website] + (info.additional_source_urls or [])),
        additional_instruction=(
            "Look for language such as 'nation's largest', 'largest no‑kill animal sanctuary in the U.S.', or equivalent phrasing "
            "on the organization's official website or highly reputable sources. If the sources do not clearly support the superlative, mark incorrect."
        ),
    )

    # 4) Charity Navigator four-star rating shown on its CN profile page
    n_cn4 = evaluator.add_leaf(
        id="charity_navigator_four_star",
        desc="Organization has a four-star rating on Charity Navigator (as shown on its Charity Navigator profile page).",
        parent=org_node,
        critical=True,
    )
    claim_cn4 = "This Charity Navigator profile page shows a four-star (4-star) rating for the organization."
    await evaluator.verify(
        claim=claim_cn4,
        node=n_cn4,
        sources=cn_url,
        additional_instruction=(
            "Verify specifically on the Charity Navigator profile page that the rating is 'Four Star' or equivalent (4/4). "
            "If the page shows fewer than four stars or cannot confirm four stars, mark incorrect."
        ),
    )


async def verify_required_information(evaluator: Evaluator, parent_node, info: OrgExtraction) -> None:
    """
    Build and verify the 'required_information_extraction' parallel critical node.
    This verifies that the required details are provided and supported by URLs:
      - official legal name (IRS/990)
      - EIN
      - board voting members count (Form 990 Part VI)
      - total revenue for most recent FY (Form 990 Part I)
      - highest compensated employee name & title (Form 990 Part VII)
      - CN profile URL correctness
      - most recent FY Form 990 URL correctness
    """
    org_name = _safe(info.organization_name_official, "the organization")
    cn_url = info.charity_navigator_profile_url
    form990_url = info.form_990_url
    all_sources = build_all_sources(info)

    req_node = evaluator.add_parallel(
        id="required_information_extraction",
        desc="Provide all required organization details extracted from publicly available documents, including direct URLs.",
        parent=parent_node,
        critical=True,
    )

    # a) Official legal name (as on IRS documentation)
    n_legal = evaluator.add_leaf(
        id="official_legal_name",
        desc="Provide the organization's official legal name exactly as it appears on IRS documentation.",
        parent=req_node,
        critical=True,
    )
    claim_legal = f"The organization's official legal name is '{_safe(info.organization_name_official)}'."
    await evaluator.verify(
        claim=claim_legal,
        node=n_legal,
        sources=_dedup_urls([form990_url, cn_url] + (info.additional_source_urls or [])),
        additional_instruction=(
            "Confirm the exact legal name as it appears on the organization's IRS/Form 990 header or IRS TEOS listing. "
            "Minor punctuation or capitalization differences can be acceptable if it is clearly the same legal entity."
        ),
    )

    # b) EIN (Tax ID)
    n_ein = evaluator.add_leaf(
        id="ein_tax_id",
        desc="Provide the organization's EIN/Tax ID, verifiable via public records.",
        parent=req_node,
        critical=True,
    )
    claim_ein = f"The organization's IRS EIN (Tax ID) is '{_safe(info.ein)}'."
    await evaluator.verify(
        claim=claim_ein,
        node=n_ein,
        sources=_dedup_urls([form990_url, cn_url] + (info.additional_source_urls or [])),
        additional_instruction=(
            "Verify the EIN on the Form 990 header or on Charity Navigator/IRS TEOS. Allow hyphenation variants (e.g., 12-3456789 vs 123456789)."
        ),
    )

    # c) Board voting members count (Form 990 Part VI)
    n_board = evaluator.add_leaf(
        id="board_voting_member_count",
        desc="Provide the total number of voting board members, extracted from the most recent Form 990 Part VI.",
        parent=req_node,
        critical=True,
    )
    claim_board = (
        f"The total number of voting members on the board of directors is '{_safe(info.board_voting_members_count)}'."
    )
    await evaluator.verify(
        claim=claim_board,
        node=n_board,
        sources=form990_url,
        additional_instruction=(
            "Locate Part VI, Section A (typically line 1a) on the Form 990 to confirm the number of voting members of the governing body."
        ),
    )

    # d) Total revenue (most recent FY) (Form 990 Part I)
    n_revenue = evaluator.add_leaf(
        id="total_revenue_most_recent_fy",
        desc="Provide total revenue for the most recent fiscal year available, extracted from the most recent Form 990 Part I.",
        parent=req_node,
        critical=True,
    )
    claim_revenue = (
        f"The organization's total revenue for the most recent fiscal year is '{_safe(info.total_revenue_most_recent_fy)}'."
    )
    await evaluator.verify(
        claim=claim_revenue,
        node=n_revenue,
        sources=form990_url,
        additional_instruction=(
            "Check Form 990 Part I (Summary) for 'Total revenue'. Allow reasonable formatting variations (e.g., with commas or '$')."
        ),
    )

    # e) Highest compensated employee (name and position title) (Form 990 Part VII)
    n_top_emp = evaluator.add_leaf(
        id="highest_compensated_employee",
        desc="Provide the name and position title of the highest compensated employee, extracted from Form 990 Part VII.",
        parent=req_node,
        critical=True,
    )
    claim_top_emp = (
        f"The highest compensated employee is '{_safe(info.highest_comp_employee_name)}' with the position title "
        f"'{_safe(info.highest_comp_employee_title)}'."
    )
    await evaluator.verify(
        claim=claim_top_emp,
        node=n_top_emp,
        sources=form990_url,
        additional_instruction=(
            "Confirm on Form 990 Part VII (and/or Schedule J if referenced) which individual has the highest reported compensation and verify the title."
        ),
    )

    # f) CN profile URL is direct and correct for the organization
    n_cn_url = evaluator.add_leaf(
        id="charity_navigator_profile_url",
        desc="Provide a direct, publicly accessible URL to the organization's Charity Navigator profile page.",
        parent=req_node,
        critical=True,
    )
    claim_cn_url = f"This URL is the Charity Navigator profile page for '{org_name}'."
    await evaluator.verify(
        claim=claim_cn_url,
        node=n_cn_url,
        sources=cn_url,
        additional_instruction=(
            "Ensure the page is on charitynavigator.org and clearly corresponds to the specified organization (matching name/EIN)."
        ),
    )

    # g) Form 990 URL is direct and for the most recent FY
    n_990_url = evaluator.add_leaf(
        id="form_990_url_most_recent_fy",
        desc="Provide a direct, publicly accessible URL to the organization's most recent Form 990 (hosted on the organization's website or an official database).",
        parent=req_node,
        critical=True,
    )
    claim_990_url = (
        f"This URL is a publicly accessible Form 990 for '{org_name}', and it is for the most recent fiscal year available."
    )
    await evaluator.verify(
        claim=claim_990_url,
        node=n_990_url,
        sources=form990_url,
        additional_instruction=(
            "The link should directly open a Form 990 document viewer or PDF for the organization. "
            "If the page indicates multiple years, confirm that the linked/selected filing is the most recent available. "
            "If recency cannot be confidently established from the page, mark not supported."
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
    """
    Evaluate an answer for the Utah animal-welfare nonprofit task using the Mind2Web2 framework.
    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator and root node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Enforce order: identify qualifying org -> verify required details
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
    extracted: OrgExtraction = await evaluator.extract(
        prompt=prompt_extract_org(),
        template_class=OrgExtraction,
        extraction_name="organization_info",
    )

    # Build and verify the tree based on rubric
    # Root is sequential and critical; its children are both critical parallel nodes
    # 1) Organization Identification
    await verify_organization_identification(evaluator, root, extracted)

    # 2) Required Information Extraction
    await verify_required_information(evaluator, root, extracted)

    # Optionally record custom info for debugging and transparency
    evaluator.add_custom_info(
        info={
            "extracted_name": extracted.organization_name_official,
            "extracted_ein": extracted.ein,
            "hq_city": extracted.headquarters_city,
            "hq_state": extracted.headquarters_state,
            "cn_url": extracted.charity_navigator_profile_url,
            "form_990_url": extracted.form_990_url,
            "board_voting_members_count": extracted.board_voting_members_count,
            "total_revenue_most_recent_fy": extracted.total_revenue_most_recent_fy,
            "highest_comp_employee_name": extracted.highest_comp_employee_name,
            "highest_comp_employee_title": extracted.highest_comp_employee_title,
            "largest_sanctuary_claim_text": extracted.largest_sanctuary_claim_text,
            "additional_source_urls": extracted.additional_source_urls,
        },
        info_type="extracted_fields_snapshot",
    )

    # Return final summary with verification tree and score
    return evaluator.get_summary()