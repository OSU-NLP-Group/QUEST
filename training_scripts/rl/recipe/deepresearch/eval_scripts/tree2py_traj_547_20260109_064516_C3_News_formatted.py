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
TASK_ID = "stablecoin_reg_119th"
TASK_DESCRIPTION = (
    "In the 119th Congress, a significant piece of federal legislation establishing a comprehensive "
    "regulatory framework for payment stablecoins was signed into law in July 2025. This bill was sponsored "
    "by a U.S. Senator who serves in leadership positions across multiple Senate committees.\n\n"
    "Identify this stablecoin regulation bill and its primary Senate sponsor. Then, trace this sponsor's "
    "committee leadership roles by identifying:\n\n"
    "1. The specific subcommittee under the Senate Banking, Housing, and Urban Affairs Committee that this sponsor chairs\n\n"
    "2. The specific subcommittee under the Senate Foreign Relations Committee that this sponsor chairs, which has jurisdiction "
    "over State Department and USAID management, international operations, and bilateral international development\n\n"
    "3. The specific subcommittee under the Senate Appropriations Committee that this sponsor chairs, which has jurisdiction "
    "over financial services and general government\n\n"
    "Provide the bill number, the sponsor's name, and the full names of all three subcommittees the sponsor chairs, along with reference URLs for verification."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StablecoinTaskExtraction(BaseModel):
    # Bill info
    bill_number: Optional[str] = None  # e.g., S.____ or H.R.___
    bill_title: Optional[str] = None
    bill_congress: Optional[str] = None  # e.g., "119th Congress"
    bill_enactment_date: Optional[str] = None  # e.g., "July 2025"
    bill_subject_summary: Optional[str] = None  # short phrase or summary as mentioned in the answer
    bill_urls: List[str] = Field(default_factory=list)

    # Sponsor info
    sponsor_name: Optional[str] = None
    sponsor_urls: List[str] = Field(default_factory=list)

    # Committee subcommittee leadership roles (full official names as stated in the answer)
    banking_subcommittee_full_name: Optional[str] = None
    banking_urls: List[str] = Field(default_factory=list)

    foreign_relations_subcommittee_full_name: Optional[str] = None
    foreign_relations_urls: List[str] = Field(default_factory=list)

    appropriations_subcommittee_full_name: Optional[str] = None
    appropriations_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the specific structured information requested below from the provided answer text. Do not infer or fabricate any information—only extract what is explicitly present in the answer. If an item is missing, set it to null (for strings) or an empty list (for URLs). When extracting URLs, include only valid, explicit URLs present in the answer (plain or within markdown).

    Required fields:
    - bill_number: The bill identifier (e.g., "S.___" or "H.R.___") as stated.
    - bill_title: The bill's title (if provided).
    - bill_congress: The congress number/label (e.g., "119th Congress") as stated.
    - bill_enactment_date: The signing/enactment month and year as stated (e.g., "July 2025").
    - bill_subject_summary: A short phrase summarizing the subject matter as stated (e.g., "comprehensive regulatory framework for payment stablecoins").
    - bill_urls: An array of all URLs cited for the bill identification and enactment details.

    - sponsor_name: The primary Senate sponsor’s full name as stated in the answer.
    - sponsor_urls: An array of URLs that support the sponsor identification/sponsorship claim (e.g., congress.gov, senate.gov, press releases).

    For the three committee leadership roles (full official subcommittee names and URLs cited for those roles):
    - banking_subcommittee_full_name: Full official name of the Banking Committee subcommittee the sponsor chairs.
    - banking_urls: URLs supporting Banking membership and chairmanship/subcommittee identification.

    - foreign_relations_subcommittee_full_name: Full official name of the Foreign Relations Committee subcommittee the sponsor chairs (with jurisdiction over State Department & USAID management, international operations, and bilateral international development).
    - foreign_relations_urls: URLs supporting Foreign Relations membership, the chair position, and the jurisdiction/subcommittee identification.

    - appropriations_subcommittee_full_name: Full official name of the Appropriations Committee subcommittee the sponsor chairs (jurisdiction over financial services and general government).
    - appropriations_urls: URLs supporting Appropriations membership, the chair position, and the jurisdiction/subcommittee identification.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _unique_nonempty(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    out = []
    for u in urls:
        if isinstance(u, str):
            v = u.strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
    return out


def _merge_sources(*url_lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        merged.extend(_unique_nonempty(lst))
    # Deduplicate while preserving order
    return _unique_nonempty(merged)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_bill_identification_nodes(
    evaluator: Evaluator,
    parent_node,
    data: StablecoinTaskExtraction,
) -> None:
    bill_node = evaluator.add_parallel(
        id="Bill_Identification",
        desc="Correctly identify the bill that matches all bill-related constraints and provide its bill number and citations.",
        parent=parent_node,
        critical=True,
    )

    bill_urls = _unique_nonempty(data.bill_urls)
    bill_number = data.bill_number or ""
    bill_congress = data.bill_congress or ""
    bill_enactment_date = data.bill_enactment_date or ""
    subject_summary = data.bill_subject_summary or ""

    # Bill_Number_Provided (existence check)
    evaluator.add_custom_node(
        result=_non_empty_str(bill_number),
        id="Bill_Number_Provided",
        desc="The answer provides the bill number/identifier (e.g., S.___ / H.R.___).",
        parent=bill_node,
        critical=True,
    )

    # Bill_Reference_URLs_Provided (existence check)
    evaluator.add_custom_node(
        result=len(bill_urls) > 0,
        id="Bill_Reference_URLs_Provided",
        desc="At least one reference URL is provided that supports the bill identification and enactment details.",
        parent=bill_node,
        critical=True,
    )

    # Bill_Congress
    bill_congress_node = evaluator.add_leaf(
        id="Bill_Congress",
        desc="The identified bill is from the 119th Congress.",
        parent=bill_node,
        critical=True,
    )
    claim_congress = f"The bill {bill_number} is from the 119th Congress."
    await evaluator.verify(
        claim=claim_congress,
        node=bill_congress_node,
        sources=bill_urls,
        additional_instruction="Confirm that the official page(s) explicitly indicate that the bill is in the 119th Congress.",
    )

    # Bill_Subject_Matter
    bill_subject_node = evaluator.add_leaf(
        id="Bill_Subject_Matter",
        desc="The identified bill establishes a comprehensive regulatory framework for payment stablecoins.",
        parent=bill_node,
        critical=True,
    )
    claim_subject = f"The bill {bill_number} establishes a comprehensive regulatory framework for payment stablecoins."
    await evaluator.verify(
        claim=claim_subject,
        node=bill_subject_node,
        sources=bill_urls,
        additional_instruction="Look for language that the bill creates/establishes a comprehensive regulatory framework for 'payment stablecoins', including regulatory structure, licensing, supervision, and related provisions.",
    )

    # Bill_Enactment_Date
    bill_enact_node = evaluator.add_leaf(
        id="Bill_Enactment_Date",
        desc="The identified bill was signed into law in July 2025.",
        parent=bill_node,
        critical=True,
    )
    claim_enact = f"The bill {bill_number} was signed into law in July 2025."
    await evaluator.verify(
        claim=claim_enact,
        node=bill_enact_node,
        sources=bill_urls,
        additional_instruction="Verify the bill's enactment or signing date on authoritative sources (e.g., Congress.gov, White House, official releases). It must indicate July 2025.",
    )


async def build_sponsor_identification_nodes(
    evaluator: Evaluator,
    parent_node,
    data: StablecoinTaskExtraction,
) -> None:
    sponsor_node = evaluator.add_parallel(
        id="Sponsor_Identification",
        desc="Correctly identify the bill's primary Senate sponsor and provide citations.",
        parent=parent_node,
        critical=True,
    )

    sponsor_name = data.sponsor_name or ""
    sponsor_urls = _unique_nonempty(data.sponsor_urls)
    bill_urls = _unique_nonempty(data.bill_urls)
    bill_number = data.bill_number or ""

    # Sponsor_Name_Provided
    evaluator.add_custom_node(
        result=_non_empty_str(sponsor_name),
        id="Sponsor_Name_Provided",
        desc="The answer provides the sponsor's name.",
        parent=sponsor_node,
        critical=True,
    )

    # Sponsor_Reference_URLs_Provided
    evaluator.add_custom_node(
        result=len(sponsor_urls) > 0,
        id="Sponsor_Reference_URLs_Provided",
        desc="At least one reference URL is provided that supports the sponsor identification and sponsorship claim.",
        parent=sponsor_node,
        critical=True,
    )

    # Sponsor_Is_Primary_Sponsor
    sponsor_primary_node = evaluator.add_leaf(
        id="Sponsor_Is_Primary_Sponsor",
        desc="The identified person is the bill's primary sponsor.",
        parent=sponsor_node,
        critical=True,
    )
    sponsor_primary_claim = f"{sponsor_name} is the primary (lead) sponsor of the bill {bill_number}."
    await evaluator.verify(
        claim=sponsor_primary_claim,
        node=sponsor_primary_node,
        sources=_merge_sources(sponsor_urls, bill_urls),
        additional_instruction="Check the bill page (e.g., Congress.gov) or official sources to confirm that the named Senator is the bill's primary sponsor.",
    )

    # Sponsor_Is_US_Senator
    sponsor_senator_node = evaluator.add_leaf(
        id="Sponsor_Is_US_Senator",
        desc="The primary sponsor is a U.S. Senator (not a House member).",
        parent=sponsor_node,
        critical=True,
    )
    sponsor_senator_claim = f"{sponsor_name} is a United States Senator."
    await evaluator.verify(
        claim=sponsor_senator_claim,
        node=sponsor_senator_node,
        sources=sponsor_urls,
        additional_instruction="Verify the person's current role as a U.S. Senator using official sources (e.g., senate.gov profile or official site).",
    )


async def build_banking_role_nodes(
    evaluator: Evaluator,
    parent_node,
    sponsor_name: str,
    full_name: Optional[str],
    role_urls: List[str],
    sponsor_urls: List[str],
) -> None:
    node = evaluator.add_parallel(
        id="Banking_Committee_Role",
        desc="Provide the Banking Committee subcommittee chaired by the sponsor (full name) and verify eligibility.",
        parent=parent_node,
        critical=True,
    )

    # Banking_Subcommittee_Full_Name_Provided
    evaluator.add_custom_node(
        result=_non_empty_str(full_name),
        id="Banking_Subcommittee_Full_Name_Provided",
        desc="The answer provides the full official name of the Banking subcommittee chaired by the sponsor.",
        parent=node,
        critical=True,
    )

    # Banking_Role_Reference_URLs_Provided
    evaluator.add_custom_node(
        result=len(role_urls) > 0,
        id="Banking_Role_Reference_URLs_Provided",
        desc="At least one reference URL is provided supporting the Banking membership and chairmanship/subcommittee identification.",
        parent=node,
        critical=True,
    )

    # Banking_Committee_Membership
    membership_node = evaluator.add_leaf(
        id="Banking_Committee_Membership",
        desc="The sponsor is a member of the Senate Banking, Housing, and Urban Affairs Committee.",
        parent=node,
        critical=True,
    )
    claim_membership = f"{sponsor_name} is a member of the Senate Committee on Banking, Housing, and Urban Affairs."
    await evaluator.verify(
        claim=claim_membership,
        node=membership_node,
        sources=_merge_sources(role_urls, sponsor_urls),
        additional_instruction="Confirm committee membership on official committee pages or the senator's official page.",
    )

    # Banking_Subcommittee_Chairmanship
    chair_node = evaluator.add_leaf(
        id="Banking_Subcommittee_Chairmanship",
        desc="The sponsor chairs a subcommittee under the Senate Banking, Housing, and Urban Affairs Committee.",
        parent=node,
        critical=True,
    )
    claim_chair = f"{sponsor_name} chairs the '{full_name}' subcommittee under the Senate Committee on Banking, Housing, and Urban Affairs."
    await evaluator.verify(
        claim=claim_chair,
        node=chair_node,
        sources=_merge_sources(role_urls, sponsor_urls),
        additional_instruction="Verify chairmanship of the specified Banking subcommittee on official senate/committee sources.",
    )

    # Banking_Subcommittee_Is_Standing_119th
    standing_node = evaluator.add_leaf(
        id="Banking_Subcommittee_Is_Standing_119th",
        desc="The identified Banking subcommittee is a formal standing subcommittee of the Senate Banking Committee in the 119th Congress.",
        parent=node,
        critical=True,
    )
    claim_standing = f"The '{full_name}' is a formal standing subcommittee of the Senate Committee on Banking, Housing, and Urban Affairs in the 119th Congress."
    await evaluator.verify(
        claim=claim_standing,
        node=standing_node,
        sources=role_urls,
        additional_instruction="Check the committee's official site/list of subcommittees for the 119th Congress to confirm this subcommittee is a standing subcommittee.",
    )


async def build_foreign_relations_role_nodes(
    evaluator: Evaluator,
    parent_node,
    sponsor_name: str,
    full_name: Optional[str],
    role_urls: List[str],
    sponsor_urls: List[str],
) -> None:
    node = evaluator.add_parallel(
        id="Foreign_Relations_Subcommittee_Role",
        desc="Provide the Foreign Relations subcommittee chaired by the sponsor (full name) that matches the specified jurisdiction and standing status.",
        parent=parent_node,
        critical=True,
    )

    # Foreign_Relations_Subcommittee_Full_Name_Provided
    evaluator.add_custom_node(
        result=_non_empty_str(full_name),
        id="Foreign_Relations_Subcommittee_Full_Name_Provided",
        desc="The answer provides the full official name of the Foreign Relations subcommittee chaired by the sponsor.",
        parent=node,
        critical=True,
    )

    # Foreign_Relations_Role_Reference_URLs_Provided
    evaluator.add_custom_node(
        result=len(role_urls) > 0,
        id="Foreign_Relations_Role_Reference_URLs_Provided",
        desc="At least one reference URL is provided supporting the Foreign Relations membership, chairmanship, and jurisdiction/subcommittee identification.",
        parent=node,
        critical=True,
    )

    # Foreign_Relations_Committee_Membership
    membership_node = evaluator.add_leaf(
        id="Foreign_Relations_Committee_Membership",
        desc="The sponsor is a member of the Senate Foreign Relations Committee.",
        parent=node,
        critical=True,
    )
    claim_membership = f"{sponsor_name} is a member of the Senate Committee on Foreign Relations."
    await evaluator.verify(
        claim=claim_membership,
        node=membership_node,
        sources=_merge_sources(role_urls, sponsor_urls),
        additional_instruction="Confirm committee membership on official Foreign Relations committee or senator pages.",
    )

    # Foreign_Relations_Subcommittee_Chairmanship
    chair_node = evaluator.add_leaf(
        id="Foreign_Relations_Subcommittee_Chairmanship",
        desc="The sponsor chairs a subcommittee under the Senate Foreign Relations Committee.",
        parent=node,
        critical=True,
    )
    claim_chair = f"{sponsor_name} chairs the '{full_name}' subcommittee under the Senate Committee on Foreign Relations."
    await evaluator.verify(
        claim=claim_chair,
        node=chair_node,
        sources=_merge_sources(role_urls, sponsor_urls),
        additional_instruction="Verify chairmanship of the specified Foreign Relations subcommittee on official sources.",
    )

    # Foreign_Relations_Jurisdiction
    juris_node = evaluator.add_leaf(
        id="Foreign_Relations_Jurisdiction",
        desc="The identified Foreign Relations subcommittee has jurisdiction over State Department and USAID management, international operations, and bilateral international development.",
        parent=node,
        critical=True,
    )
    claim_jurisdiction = (
        f"The '{full_name}' subcommittee has jurisdiction over State Department and USAID management, "
        "international operations, and bilateral international development."
    )
    await evaluator.verify(
        claim=claim_jurisdiction,
        node=juris_node,
        sources=role_urls,
        additional_instruction="Confirm the listed jurisdiction on the official subcommittee page or authoritative Senate documentation.",
    )

    # Foreign_Relations_Subcommittee_Is_Standing_119th
    standing_node = evaluator.add_leaf(
        id="Foreign_Relations_Subcommittee_Is_Standing_119th",
        desc="The identified Foreign Relations subcommittee is a formal standing subcommittee of the Senate Foreign Relations Committee in the 119th Congress.",
        parent=node,
        critical=True,
    )
    claim_standing = f"The '{full_name}' is a formal standing subcommittee of the Senate Committee on Foreign Relations in the 119th Congress."
    await evaluator.verify(
        claim=claim_standing,
        node=standing_node,
        sources=role_urls,
        additional_instruction="Check the committee's official site/list of subcommittees for the 119th Congress to confirm this subcommittee is a standing subcommittee.",
    )


async def build_appropriations_role_nodes(
    evaluator: Evaluator,
    parent_node,
    sponsor_name: str,
    full_name: Optional[str],
    role_urls: List[str],
    sponsor_urls: List[str],
) -> None:
    node = evaluator.add_parallel(
        id="Appropriations_Subcommittee_Role",
        desc="Provide the Appropriations subcommittee chaired by the sponsor (full name) that matches the specified jurisdiction and standing status.",
        parent=parent_node,
        critical=True,
    )

    # Appropriations_Subcommittee_Full_Name_Provided
    evaluator.add_custom_node(
        result=_non_empty_str(full_name),
        id="Appropriations_Subcommittee_Full_Name_Provided",
        desc="The answer provides the full official name of the Appropriations subcommittee chaired by the sponsor.",
        parent=node,
        critical=True,
    )

    # Appropriations_Role_Reference_URLs_Provided
    evaluator.add_custom_node(
        result=len(role_urls) > 0,
        id="Appropriations_Role_Reference_URLs_Provided",
        desc="At least one reference URL is provided supporting the Appropriations membership, chairmanship, and jurisdiction/subcommittee identification.",
        parent=node,
        critical=True,
    )

    # Appropriations_Committee_Membership
    membership_node = evaluator.add_leaf(
        id="Appropriations_Committee_Membership",
        desc="The sponsor is a member of the Senate Appropriations Committee.",
        parent=node,
        critical=True,
    )
    claim_membership = f"{sponsor_name} is a member of the Senate Committee on Appropriations."
    await evaluator.verify(
        claim=claim_membership,
        node=membership_node,
        sources=_merge_sources(role_urls, sponsor_urls),
        additional_instruction="Confirm committee membership on official Appropriations committee or senator pages.",
    )

    # Appropriations_Subcommittee_Chairmanship
    chair_node = evaluator.add_leaf(
        id="Appropriations_Subcommittee_Chairmanship",
        desc="The sponsor chairs a subcommittee under the Senate Appropriations Committee.",
        parent=node,
        critical=True,
    )
    claim_chair = f"{sponsor_name} chairs the '{full_name}' subcommittee under the Senate Committee on Appropriations."
    await evaluator.verify(
        claim=claim_chair,
        node=chair_node,
        sources=_merge_sources(role_urls, sponsor_urls),
        additional_instruction="Verify chairmanship of the specified Appropriations subcommittee on official sources.",
    )

    # Appropriations_Jurisdiction
    juris_node = evaluator.add_leaf(
        id="Appropriations_Jurisdiction",
        desc="The identified Appropriations subcommittee has jurisdiction over financial services and general government.",
        parent=node,
        critical=True,
    )
    claim_jurisdiction = f"The '{full_name}' subcommittee has jurisdiction over financial services and general government."
    await evaluator.verify(
        claim=claim_jurisdiction,
        node=juris_node,
        sources=role_urls,
        additional_instruction="Confirm the listed jurisdiction on the official subcommittee page or authoritative Senate documentation.",
    )

    # Appropriations_Subcommittee_Is_Standing_119th
    standing_node = evaluator.add_leaf(
        id="Appropriations_Subcommittee_Is_Standing_119th",
        desc="The identified Appropriations subcommittee is a formal standing subcommittee of the Senate Appropriations Committee in the 119th Congress.",
        parent=node,
        critical=True,
    )
    claim_standing = f"The '{full_name}' is a formal standing subcommittee of the Senate Committee on Appropriations in the 119th Congress."
    await evaluator.verify(
        claim=claim_standing,
        node=standing_node,
        sources=role_urls,
        additional_instruction="Check the committee's official site/list of subcommittees for the 119th Congress to confirm this subcommittee is a standing subcommittee.",
    )


async def build_committee_leadership_nodes(
    evaluator: Evaluator,
    parent_node,
    data: StablecoinTaskExtraction,
) -> None:
    roles_node = evaluator.add_parallel(
        id="Committee_Leadership_Roles",
        desc="Identify the three requested subcommittees chaired by the sponsor (Banking, Foreign Relations, Appropriations) with correct jurisdictions and standing status in the 119th Congress.",
        parent=parent_node,
        critical=True,
    )

    sponsor_name = data.sponsor_name or ""
    sponsor_urls = _unique_nonempty(data.sponsor_urls)

    # Banking
    await build_banking_role_nodes(
        evaluator=evaluator,
        parent_node=roles_node,
        sponsor_name=sponsor_name,
        full_name=data.banking_subcommittee_full_name,
        role_urls=_unique_nonempty(data.banking_urls),
        sponsor_urls=sponsor_urls,
    )

    # Foreign Relations
    await build_foreign_relations_role_nodes(
        evaluator=evaluator,
        parent_node=roles_node,
        sponsor_name=sponsor_name,
        full_name=data.foreign_relations_subcommittee_full_name,
        role_urls=_unique_nonempty(data.foreign_relations_urls),
        sponsor_urls=sponsor_urls,
    )

    # Appropriations
    await build_appropriations_role_nodes(
        evaluator=evaluator,
        parent_node=roles_node,
        sponsor_name=sponsor_name,
        full_name=data.appropriations_subcommittee_full_name,
        role_urls=_unique_nonempty(data.appropriations_urls),
        sponsor_urls=sponsor_urls,
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
    Evaluate an answer for the 119th Congress stablecoin regulation and sponsor committee leadership roles task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # As per rubric: sequential stages (bill -> sponsor -> committees)
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

    # Extract all required structured fields from the answer
    extracted: StablecoinTaskExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=StablecoinTaskExtraction,
        extraction_name="stablecoin_task_extraction",
    )

    # Build Task_Completion stages (as children of root)
    # 1) Bill Identification
    await build_bill_identification_nodes(evaluator, root, extracted)

    # 2) Sponsor Identification
    await build_sponsor_identification_nodes(evaluator, root, extracted)

    # 3) Committee Leadership Roles
    await build_committee_leadership_nodes(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()