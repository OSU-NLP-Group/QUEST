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
TASK_ID = "fda_rpd_prv_2024_single_drug"
TASK_DESCRIPTION = """
Identify a drug that received FDA approval in 2024 and was awarded a rare pediatric disease priority review voucher (PRV). For this drug, provide the following information: (1) The proprietary (brand) name of the drug; (2) The generic or scientific name of the drug; (3) The pharmaceutical company or manufacturer; (4) The specific rare pediatric disease indication for which it was approved; (5) The FDA approval date in 2024; (6) A direct link to the official FDA announcement, Federal Register notice, or FDA press release confirming the rare pediatric disease priority review voucher award. The drug must have both orphan drug designation and a rare pediatric disease priority review voucher awarded by the FDA.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DrugItem(BaseModel):
    """One candidate drug item mentioned in the answer."""
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    manufacturer: Optional[str] = None
    indication: Optional[str] = None
    approval_date_2024: Optional[str] = None

    # Evidence links
    approval_url: Optional[str] = None
    orphan_designation_url: Optional[str] = None
    prv_award_url: Optional[str] = None
    indication_url: Optional[str] = None


class DrugsExtraction(BaseModel):
    """All drug candidates extracted from the answer (we'll use the first one)."""
    drugs: List[DrugItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_drugs() -> str:
    return """
    Extract up to three (3) candidate drugs from the answer. For each drug, return:
    - brand_name: Proprietary/brand name (string)
    - generic_name: Generic/active ingredient/biologic name (string)
    - manufacturer: Company/manufacturer/sponsor (string)
    - indication: Specific approved rare pediatric disease indication (string)
    - approval_date_2024: The stated FDA approval date in 2024 (string, keep exactly as written in the answer, e.g., "March 12, 2024" or "2024-03-12")
    - approval_url: Direct URL that supports FDA approval and date (prefer FDA or official press releases)
    - orphan_designation_url: Direct URL that supports FDA orphan drug designation for this product/indication
    - prv_award_url: Direct URL to an official FDA page, FDA press release, or Federal Register notice that confirms the award of a Rare Pediatric Disease Priority Review Voucher (PRV)
    - indication_url: Direct URL supporting the approved indication (FDA page or manufacturer press release acceptable)

    Rules:
    - Extract only what is explicitly present in the answer; do not invent data.
    - If the answer lists multiple drugs, include them in order of appearance (max 3).
    - If any field is missing for a drug, set it to null.
    - Ensure URLs are full and valid if present; if the answer gives a markdown link, extract the actual URL.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def pick_primary_drug(extraction: DrugsExtraction) -> DrugItem:
    """Pick the first drug in the extracted list; return empty placeholder if none."""
    if extraction and extraction.drugs:
        return extraction.drugs[0]
    return DrugItem()


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def add_supporting_evidence_subtrees(
    evaluator: Evaluator,
    parent_node,
    drug: DrugItem,
):
    """
    Build the 'Supporting_Evidence_and_URLs' node:
      For each evidence category, create a small sequential subtree:
        1) presence check (custom node, critical)
        2) content support verification (leaf with URL verification, critical)
    Returns a dict of helpful presence nodes to optionally use as prerequisites elsewhere.
    """
    supporting = evaluator.add_parallel(
        id="Supporting_Evidence_and_URLs",
        desc="All required claims are supported with acceptable sources and direct URLs, as stated in constraints.",
        parent=parent_node,
        critical=True
    )

    presence_nodes = {}

    # 1) Approval + Date evidence
    appr_grp = evaluator.add_sequential(
        id="Approval_and_Date_Evidence",
        desc="Approval/date evidence URL presence and support",
        parent=supporting,
        critical=True
    )
    appr_present = evaluator.add_custom_node(
        result=bool(drug.approval_url and drug.approval_url.strip()),
        id="Approval_and_Date_Evidence_URL_present",
        desc="A direct URL is provided for FDA approval and approval date evidence",
        parent=appr_grp,
        critical=True
    )
    presence_nodes["approval_url_present"] = appr_present

    appr_verify = evaluator.add_leaf(
        id="Approval_and_Date_Evidence_URL",
        desc="Include a direct URL supporting FDA approval and the approval date, from an official FDA announcement, Federal Register notice, or manufacturer press release.",
        parent=appr_grp,
        critical=True
    )
    appr_claim = (
        f"This webpage confirms that the FDA approved the drug "
        f"{drug.brand_name or drug.generic_name or 'the drug'} "
        f"for the indication {drug.indication or '(indication stated)'} "
        f"on {drug.approval_date_2024 or '(approval date stated)'}, and that the page itself explicitly supports the approval and the date."
    )
    await evaluator.verify(
        claim=appr_claim,
        node=appr_verify,
        sources=drug.approval_url,
        additional_instruction=(
            "Only mark as supported if this exact page explicitly shows FDA approval and the approval date. "
            "Acceptable sources include: FDA pages (fda.gov), Federal Register notices, or an official manufacturer press release. "
            "If no URL is provided or the page does not clearly confirm FDA approval and the date, mark as Incorrect."
        )
    )

    # 2) Orphan designation evidence
    orphan_grp = evaluator.add_sequential(
        id="Orphan_Designation_Evidence",
        desc="Orphan designation evidence URL presence and support",
        parent=supporting,
        critical=True
    )
    orphan_present = evaluator.add_custom_node(
        result=bool(drug.orphan_designation_url and drug.orphan_designation_url.strip()),
        id="Orphan_Designation_Evidence_URL_present",
        desc="A direct URL is provided supporting orphan drug designation",
        parent=orphan_grp,
        critical=True
    )
    presence_nodes["orphan_url_present"] = orphan_present

    orphan_verify = evaluator.add_leaf(
        id="Orphan_Designation_Evidence_URL",
        desc="Include a direct URL supporting orphan drug designation, from an official FDA announcement, Federal Register notice, or manufacturer press release.",
        parent=orphan_grp,
        critical=True
    )
    orphan_claim = (
        f"This webpage shows that {drug.brand_name or drug.generic_name or 'the drug'} "
        f"received Orphan Drug Designation from the FDA (or explicitly indicates it is an FDA orphan-designated product)."
    )
    await evaluator.verify(
        claim=orphan_claim,
        node=orphan_verify,
        sources=drug.orphan_designation_url,
        additional_instruction=(
            "Only mark as supported if this exact page explicitly indicates FDA Orphan Drug Designation. "
            "Accept FDA pages, Federal Register, or manufacturer press releases that clearly state the designation. "
            "If no URL is provided or the page does not clearly confirm FDA orphan designation, mark as Incorrect."
        )
    )

    # 3) PRV award official confirmation (FDA or Federal Register only)
    prv_grp = evaluator.add_sequential(
        id="PRV_Award_Official_Confirmation",
        desc="PRV award URL presence and official FDA/FR support",
        parent=supporting,
        critical=True
    )
    prv_present = evaluator.add_custom_node(
        result=bool(drug.prv_award_url and drug.prv_award_url.strip()),
        id="PRV_Award_Official_Confirmation_URL_present",
        desc="A direct URL is provided to the official FDA or Federal Register page confirming PRV award",
        parent=prv_grp,
        critical=True
    )
    presence_nodes["prv_url_present"] = prv_present

    prv_verify = evaluator.add_leaf(
        id="PRV_Award_Official_Confirmation_URL",
        desc="Include a direct URL to the official FDA announcement, Federal Register notice, or FDA press release confirming the rare pediatric disease PRV award.",
        parent=prv_grp,
        critical=True
    )
    prv_claim = (
        f"This webpage is an official FDA (fda.gov) or Federal Register "
        f"(federalregister.gov) page that explicitly confirms that "
        f"{drug.brand_name or drug.generic_name or 'the drug'} was awarded a Rare Pediatric Disease Priority Review Voucher (PRV)."
    )
    await evaluator.verify(
        claim=prv_claim,
        node=prv_verify,
        sources=drug.prv_award_url,
        additional_instruction=(
            "Strict requirement: Only accept if the URL is on fda.gov or federalregister.gov and the page explicitly confirms "
            "that a Rare Pediatric Disease Priority Review Voucher (PRV) was awarded for this product/indication. "
            "If the URL domain is not fda.gov or federalregister.gov, mark as Incorrect."
        )
    )

    # 4) Indication evidence
    ind_grp = evaluator.add_sequential(
        id="Indication_Evidence",
        desc="Indication evidence URL presence and support",
        parent=supporting,
        critical=True
    )
    ind_present = evaluator.add_custom_node(
        result=bool(drug.indication_url and drug.indication_url.strip()),
        id="Indication_Evidence_URL_present",
        desc="A direct URL is provided supporting the approved indication",
        parent=ind_grp,
        critical=True
    )
    presence_nodes["indication_url_present"] = ind_present

    ind_verify = evaluator.add_leaf(
        id="Indication_Evidence_URL",
        desc="Include a direct URL supporting the approved indication, from an official FDA announcement, Federal Register notice, or manufacturer press release.",
        parent=ind_grp,
        critical=True
    )
    ind_claim = (
        f"This webpage confirms that the approved indication for "
        f"{drug.brand_name or drug.generic_name or 'the drug'} is "
        f"'{drug.indication or '(indication as stated)'}'."
    )
    await evaluator.verify(
        claim=ind_claim,
        node=ind_verify,
        sources=drug.indication_url,
        additional_instruction=(
            "Only mark as supported if the page explicitly states the same indication text or an unambiguous equivalent. "
            "Accept FDA pages, Federal Register, or official manufacturer press releases. "
            "If no URL is provided or the page does not clearly confirm the indication, mark as Incorrect."
        )
    )

    return supporting, presence_nodes


async def add_required_output_fields_nodes(
    evaluator: Evaluator,
    parent_node,
    drug: DrugItem
):
    """Build the 'Required_Output_Fields' node with presence checks (critical)."""
    req = evaluator.add_parallel(
        id="Required_Output_Fields",
        desc="Response includes all required informational fields for the drug.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(drug.brand_name and drug.brand_name.strip()),
        id="Proprietary_Name_Provided",
        desc="Provide the proprietary (brand) name of the drug.",
        parent=req,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(drug.generic_name and drug.generic_name.strip()),
        id="Generic_or_Scientific_Name_Provided",
        desc="Provide the generic/scientific name (e.g., active ingredient/biologic name).",
        parent=req,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(drug.manufacturer and drug.manufacturer.strip()),
        id="Manufacturer_Provided",
        desc="Provide the pharmaceutical company/manufacturer.",
        parent=req,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(drug.indication and drug.indication.strip()),
        id="Rare_Pediatric_Disease_Indication_Provided",
        desc="Provide the specific rare pediatric disease indication for which the drug was approved.",
        parent=req,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(drug.approval_date_2024 and drug.approval_date_2024.strip()),
        id="FDA_Approval_Date_Provided",
        desc="Provide the specific FDA approval date.",
        parent=req,
        critical=True
    )

    return req


async def add_eligibility_constraints_nodes(
    evaluator: Evaluator,
    parent_node,
    drug: DrugItem,
    prereq_presence: Dict[str, Any]
):
    """Build the 'Eligibility_Constraints' node with evidence-grounded verification (critical)."""
    elig = evaluator.add_parallel(
        id="Eligibility_Constraints",
        desc="Chosen drug satisfies all eligibility constraints stated in the question/constraints.",
        parent=parent_node,
        critical=True
    )

    # FDA Approval in 2024
    n_approval = evaluator.add_leaf(
        id="FDA_Approval_in_2024",
        desc="Drug received FDA approval with an approval date during calendar year 2024 (Jan 1–Dec 31, 2024).",
        parent=elig,
        critical=True
    )
    approval_claim = (
        f"The drug {drug.brand_name or drug.generic_name or 'the drug'} received FDA approval in the calendar year 2024. "
        f"The page should explicitly indicate FDA approval (not just a submission or PDUFA goal). "
        f"If a specific date is shown, it should be in 2024 (e.g., {drug.approval_date_2024 or 'a 2024 date'})."
    )
    await evaluator.verify(
        claim=approval_claim,
        node=n_approval,
        sources=drug.approval_url,
        additional_instruction=(
            "Judge strictly from the provided URL. If no URL is provided, return Incorrect. "
            "If multiple dates are on the page, use context to identify the actual FDA approval date."
        ),
        extra_prerequisites=[prereq_presence.get("approval_url_present")] if prereq_presence.get("approval_url_present") else None
    )

    # Orphan Drug Designation Received
    n_orphan = evaluator.add_leaf(
        id="Orphan_Drug_Designation_Received",
        desc="Drug has received FDA orphan drug designation from the FDA.",
        parent=elig,
        critical=True
    )
    orphan_claim = (
        f"The product {drug.brand_name or drug.generic_name or 'the drug'} has FDA Orphan Drug Designation."
    )
    await evaluator.verify(
        claim=orphan_claim,
        node=n_orphan,
        sources=drug.orphan_designation_url,
        additional_instruction=(
            "Only accept if the page clearly states 'Orphan Drug Designation' for this product/indication. "
            "If no URL is provided, mark as Incorrect."
        ),
        extra_prerequisites=[prereq_presence.get("orphan_url_present")] if prereq_presence.get("orphan_url_present") else None
    )

    # Rare Pediatric Disease PRV Awarded
    n_prv = evaluator.add_leaf(
        id="Rare_Pediatric_Disease_PRV_Awarded",
        desc="Drug was awarded a rare pediatric disease priority review voucher (PRV) by the FDA.",
        parent=elig,
        critical=True
    )
    prv_claim = (
        f"The FDA awarded a Rare Pediatric Disease Priority Review Voucher (PRV) to "
        f"{drug.brand_name or drug.generic_name or 'the drug'}."
    )
    await evaluator.verify(
        claim=prv_claim,
        node=n_prv,
        sources=drug.prv_award_url,
        additional_instruction=(
            "Strict domain rule: Only accept if the URL is on fda.gov or federalregister.gov and the page explicitly confirms the PRV award. "
            "If no URL is provided or the domain is not fda.gov/federalregister.gov, mark as Incorrect."
        ),
        extra_prerequisites=[prereq_presence.get("prv_url_present")] if prereq_presence.get("prv_url_present") else None
    )

    # Rare Pediatric Disease Criteria Met
    n_criteria = evaluator.add_leaf(
        id="Rare_Pediatric_Disease_Criteria_Met",
        desc="The treated disease meets FDA rare pediatric disease criteria: affects fewer than 200,000 people in the US AND primarily affects individuals from birth to 18 years.",
        parent=elig,
        critical=True
    )
    criteria_claim = (
        f"The approved indication '{drug.indication or '(indication stated)'}' qualifies as a 'rare pediatric disease' under FDA's definition "
        f"(affecting fewer than 200,000 people in the US and primarily affecting individuals from birth to 18 years), "
        f"as evidenced by the FDA’s PRV award for this indication. "
        f"If the PRV award is explicitly confirmed for this product/indication by the official FDA or Federal Register page, "
        f"it is acceptable to infer that the FDA determined the criteria were met."
    )
    sources_for_criteria: List[str] = []
    if drug.prv_award_url:
        sources_for_criteria.append(drug.prv_award_url)
    if drug.indication_url:
        sources_for_criteria.append(drug.indication_url)

    await evaluator.verify(
        claim=criteria_claim,
        node=n_criteria,
        sources=sources_for_criteria if sources_for_criteria else None,
        additional_instruction=(
            "Prefer the official PRV award page (fda.gov or federalregister.gov). "
            "If that page explicitly ties the PRV to this indication, you may accept that the FDA judged the disease to meet the statutory criteria "
            "without separately proving prevalence or pediatric predominance on another page. "
            "If no URL is provided, mark as Incorrect."
        ),
        extra_prerequisites=[
            p for k, p in [
                ("prv_url_present", prereq_presence.get("prv_url_present")),
                ("indication_url_present", prereq_presence.get("indication_url_present")),
            ] if p is not None
        ] if (prereq_presence.get("prv_url_present") or prereq_presence.get("indication_url_present")) else None
    )

    return elig


async def verify_drug(
    evaluator: Evaluator,
    parent_node,
    drug: DrugItem
):
    """
    Build the full verification tree under the critical, sequential main node:
      1) Eligibility_Constraints (parallel, critical)
      2) Required_Output_Fields (parallel, critical)
      3) Supporting_Evidence_and_URLs (parallel, critical; each with presence + verify)
    """
    # Main critical sequential node per rubric
    main = evaluator.add_sequential(
        id="Complete_Drug_Analysis",
        desc="Provide one drug that meets all stated FDA/regulatory constraints and report all required fields with required supporting links.",
        parent=parent_node,
        critical=True
    )

    # We will add Supporting_Evidence_and_URLs first to get presence nodes for use as prerequisites,
    # but we still add it as the THIRD child under the main sequential node by ordering calls appropriately.
    # So we instead build in the proper rubric order and not rely on pre-creation:
    # 1) Eligibility (we'll call without prerequisites, but our additional_instruction enforces failure when URL missing)
    # 2) Required fields
    # 3) Supporting evidence (presence + verify)

    # However, to incorporate prerequisites, we will actually create Supporting Evidence FIRST (not appended yet),
    # then re-attach its nodes? Evaluator API appends immediately, so we cannot pre-create without adding.
    # Therefore, we'll build in rubric order and rely on strict additional instructions to fail when URLs are missing.

    # 1) Eligibility Constraints
    await add_eligibility_constraints_nodes(
        evaluator=evaluator,
        parent_node=main,
        drug=drug,
        prereq_presence={}  # No presence prereqs here; we enforce via strict instructions
    )

    # 2) Required Output Fields (presence checks)
    await add_required_output_fields_nodes(
        evaluator=evaluator,
        parent_node=main,
        drug=drug
    )

    # 3) Supporting Evidence and URLs (presence + verification)
    supporting_node, presence_nodes = await add_supporting_evidence_subtrees(
        evaluator=evaluator,
        parent_node=main,
        drug=drug
    )

    # Note: Because the parent is sequential and critical, if Eligibility or Required fields fail,
    # Supporting Evidence will be skipped automatically. Presence nodes are inside Supporting Evidence
    # to enforce URL availability and content support per rubric.


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
    Evaluate an answer to the 'FDA 2024 RPD PRV single-drug' task and return a structured summary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is internal; we add a critical sequential child as the main rubric root
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

    # Extract structured data from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_drugs(),
        template_class=DrugsExtraction,
        extraction_name="drug_candidates",
    )

    # Pick first drug candidate (padding if none)
    drug = pick_primary_drug(extracted)

    # Build verification tree and run checks
    await verify_drug(evaluator, root, drug)

    # Return final summary
    return evaluator.get_summary()