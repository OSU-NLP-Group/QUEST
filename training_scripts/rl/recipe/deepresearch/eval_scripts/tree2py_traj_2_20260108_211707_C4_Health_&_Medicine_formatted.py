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
TASK_ID = "fda_mito_peds_2025"
TASK_DESCRIPTION = (
    "I am researching recently approved treatments for rare pediatric diseases affecting cellular energy production. "
    "Can you identify the FDA-approved novel drug from 2025 that meets ALL of the following criteria: "
    "(1) designated for a rare disease affecting mitochondrial function, "
    "(2) approved for use in pediatric patients, "
    "(3) represents the first-ever FDA-approved therapy for its specific condition, "
    "(4) received accelerated approval from the FDA, and "
    "(5) was approved between July and December 2025? "
    "Please provide the drug name, its specific indication, the approval date, and supporting reference URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DrugInfoExtraction(BaseModel):
    drug_name: Optional[str] = None
    specific_indication: Optional[str] = None
    approval_date: Optional[str] = None
    route_of_administration: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_drug_info() -> str:
    return """
    Extract the single drug the answer claims satisfies the criteria. Provide:
    - drug_name: The drug name as written in the answer (generic, brand, or both as provided).
    - specific_indication: The specific FDA-approved indication as stated in the answer (do not generalize).
    - approval_date: The FDA approval date as provided in the answer (string, any format).
    - route_of_administration: The route or dosage form (e.g., injection, intravenous, oral), if explicitly stated in the answer.
    - reference_urls: All supporting URLs cited in the answer that substantiate the approval and the criteria. Extract only explicit URLs.
    If a field is missing in the answer, set it to null (or [] for arrays). Do not invent information.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_and_clean_urls(urls: List[str]) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_provided_fields_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: DrugInfoExtraction
) -> None:
    """
    Creates existence checks for required output fields.
    """
    provides_node = evaluator.add_parallel(
        id="Provides_Requested_Output_Fields",
        desc="The response includes all fields explicitly requested by the question.",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(extracted.drug_name and extracted.drug_name.strip()),
        id="Provides_Drug_Name",
        desc="Provides the drug name.",
        parent=provides_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.specific_indication and extracted.specific_indication.strip()),
        id="Provides_Specific_Indication",
        desc="Provides the drug’s specific FDA-approved indication relevant to the criteria.",
        parent=provides_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.approval_date and extracted.approval_date.strip()),
        id="Provides_Approval_Date",
        desc="Provides the FDA approval date.",
        parent=provides_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.reference_urls and len(extracted.reference_urls) > 0),
        id="Provides_Supporting_Reference_URLs",
        desc="Provides supporting reference URL(s) that substantiate the approval and key criteria claims.",
        parent=provides_node,
        critical=True
    )


async def build_and_verify_eligibility_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: DrugInfoExtraction
) -> None:
    """
    Creates and verifies the constraint checks using the cited sources.
    This node is critical overall. Individual leaves align with rubric.
    Note: The 'Injectable' check is treated as non-critical because it is not part of the original question's constraints.
    """
    meets_node = evaluator.add_parallel(
        id="Meets_All_Eligibility_Criteria",
        desc="The identified drug satisfies every constraint specified in the question/constraints list.",
        parent=parent_node,
        critical=True
    )

    # Prepare inputs
    name = (extracted.drug_name or "").strip()
    indication = (extracted.specific_indication or "").strip()
    urls = _dedupe_and_clean_urls(extracted.reference_urls)

    # 1) Novel Drug Approval 2025
    node_novel = evaluator.add_leaf(
        id="Novel_Drug_Approval_2025",
        desc="The drug appears on the FDA's official list of novel drug approvals for calendar year 2025.",
        parent=meets_node,
        critical=True
    )
    claim_novel = (
        f"The drug {name if name else 'in question'} appears on FDA/CDER's 'Novel Drug Approvals for 2025' list."
    )
    await evaluator.verify(
        claim=claim_novel,
        node=node_novel,
        sources=urls,
        additional_instruction=(
            "Check the FDA/CDER 'Novel Drug Approvals 2025' page to confirm the drug is listed. "
            "Accept if the provided sources include that list or an FDA page explicitly stating it is a 2025 novel drug approval."
        )
    )

    # 2) Rare Disease / Orphan designation
    node_orphan = evaluator.add_leaf(
        id="Rare_Disease_Orphan_Designation",
        desc="The drug is designated for a rare disease (orphan drug designation).",
        parent=meets_node,
        critical=True
    )
    claim_orphan = (
        f"{name if name else 'This drug'} has FDA orphan drug designation (i.e., is for a rare disease)."
    )
    await evaluator.verify(
        claim=claim_orphan,
        node=node_orphan,
        sources=urls,
        additional_instruction=(
            "Look for 'Orphan Drug' designation on FDA pages (e.g., press releases, Drugs@FDA, "
            "or the Orphan Drug Designations and Approvals database) confirming orphan/rare disease designation."
        )
    )

    # 3) Mitochondrial function disease
    node_mito = evaluator.add_leaf(
        id="Mitochondrial_Function_Disease",
        desc="The drug is indicated for a disease affecting mitochondrial function / cellular energy production.",
        parent=meets_node,
        critical=True
    )
    claim_mito = (
        f"The indicated disease for {name if name else 'the drug'} affects mitochondrial function or cellular energy production."
    )
    await evaluator.verify(
        claim=claim_mito,
        node=node_mito,
        sources=urls,
        additional_instruction=(
            "Verify from the sources that the condition involves mitochondrial dysfunction or energy production defects "
            "(e.g., mitochondrial DNA depletion, oxidative phosphorylation defects, mitochondrial disease). "
            "Explicit mention of 'mitochondrial' or equivalent pathophysiology should be present."
        )
    )

    # 4) Pediatric use approval
    node_peds = evaluator.add_leaf(
        id="Pediatric_Use",
        desc="The drug is approved for use in pediatric patients (including those under 18 years or adolescents aged 12 and older).",
        parent=meets_node,
        critical=True
    )
    claim_peds = (
        f"{name if name else 'This drug'} is approved for pediatric patients (under 18), including adolescents."
    )
    await evaluator.verify(
        claim=claim_peds,
        node=node_peds,
        sources=urls,
        additional_instruction=(
            "Confirm that the approved indication or labeling includes pediatric patients (e.g., 'patients aged 12 years and older', "
            "'pediatric', 'children'). Approval in adolescents counts as pediatric."
        )
    )

    # 5) First-ever therapy for condition
    node_first = evaluator.add_leaf(
        id="First_Ever_Therapy_For_Condition",
        desc="The drug represents the first-ever FDA-approved therapy for its specific condition/indication.",
        parent=meets_node,
        critical=True
    )
    claim_first = (
        f"{name if name else 'This drug'} is the first FDA-approved therapy for {indication if indication else 'the specified condition'}."
    )
    await evaluator.verify(
        claim=claim_first,
        node=node_first,
        sources=urls,
        additional_instruction=(
            "Look for language like 'first therapy', 'first approved treatment', or 'first FDA-approved for this condition' on FDA sources."
        )
    )

    # 6) Accelerated approval
    node_acc = evaluator.add_leaf(
        id="Accelerated_Approval",
        desc="The drug received accelerated approval from the FDA.",
        parent=meets_node,
        critical=True
    )
    claim_acc = (
        f"{name if name else 'This drug'} received accelerated approval from FDA."
    )
    await evaluator.verify(
        claim=claim_acc,
        node=node_acc,
        sources=urls,
        additional_instruction=(
            "Check for explicit mention of 'Accelerated Approval' (Subpart H/E) on FDA pages such as press releases, "
            "Drugs@FDA, or approval summaries."
        )
    )

    # 7) Approval window July–December 2025
    node_h22025 = evaluator.add_leaf(
        id="Approval_Window_H2_2025",
        desc="The FDA approval date is between July 1, 2025 and December 31, 2025.",
        parent=meets_node,
        critical=True
    )
    claim_h22025 = (
        f"FDA approval for {name if name else 'this drug'} occurred between July 1, 2025 and December 31, 2025 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_h22025,
        node=node_h22025,
        sources=urls,
        additional_instruction=(
            "Use the FDA approval date on Drugs@FDA, approval letters, or FDA press releases to confirm the date lies within the specified window."
        )
    )

    # 8) Injectable (treated as non-critical; not part of original question constraints)
    node_inj = evaluator.add_leaf(
        id="Injectable",
        desc="The drug is administered via injection.",
        parent=meets_node,
        critical=False  # Adjusted to non-critical since not requested in the original question
    )
    claim_inj = (
        f"{name if name else 'This drug'} is administered by injection (including intravenous, subcutaneous, or intramuscular routes)."
    )
    await evaluator.verify(
        claim=claim_inj,
        node=node_inj,
        sources=urls,
        additional_instruction=(
            "Check labeling or FDA description for dosage form/route such as 'injection', 'intravenous', 'intramuscular', or 'subcutaneous'."
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
    Evaluate an answer for the FDA 2025 rare pediatric mitochondrial-related novel drug task.
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
        default_model=model,
    )

    # Create top-level critical evaluation node
    task_node = evaluator.add_parallel(
        id="Drug_Response_Evaluation",
        desc="Evaluates whether the response identifies an FDA-approved 2025 novel drug meeting all stated criteria and provides the requested output fields with supporting references.",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_drug_info(),
        template_class=DrugInfoExtraction,
        extraction_name="drug_info_extraction"
    )

    # Build "Provides requested fields" checks first (so they can gate eligibility verifications)
    await build_provided_fields_checks(evaluator, task_node, extracted)

    # Build and verify eligibility criteria
    await build_and_verify_eligibility_checks(evaluator, task_node, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()