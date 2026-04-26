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
TASK_ID = "fda_2024_rare_endocrine_drug"
TASK_DESCRIPTION = (
    "Identify the FDA-approved drug from 2024 that meets all of the following criteria:\n\n"
    "1. It was approved by the FDA's Center for Drug Evaluation and Research (CDER) in 2024 as a novel drug (new molecular entity)\n"
    "2. It is indicated for treating a rare genetic endocrine disorder that affects adrenal hormone production\n"
    "3. It received FDA orphan drug designation\n"
    "4. It received FDA breakthrough therapy designation\n"
    "5. It is approved for use in both adult patients and pediatric patients aged 4 years and older\n"
    "6. It is available in at least two different pharmaceutical dosage forms\n"
    "7. It is indicated for use as adjunctive treatment to glucocorticoid replacement therapy\n"
    "8. It was approved in December 2024\n\n"
    "Provide the drug's proprietary (brand) name and include a reference URL from an official FDA source (such as the FDA's 2024 New Drug Therapy Approvals Annual Report or an FDA press release) that confirms the approval details."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DrugIdentification(BaseModel):
    """
    Extract the key identification info about the single drug the answer claims
    satisfies the prompt (brand/generic) and the official FDA reference URLs.
    """
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    fda_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt builders                                                  #
# --------------------------------------------------------------------------- #
def prompt_extract_drug_identification() -> str:
    return (
        "Extract the single drug that the answer claims meets ALL of the listed criteria. "
        "Return:\n"
        "1) brand_name: the proprietary (brand) name explicitly shown in the answer\n"
        "2) generic_name: the nonproprietary name if present in the answer\n"
        "3) fda_reference_urls: a list (up to 5) of official FDA URLs explicitly provided in the answer that are used as references. "
        "Include ONLY fda.gov domain links (including subdomains like www.fda.gov, labels.fda.gov, cdER.fda.gov, or FDA-hosted PDFs). "
        "If the answer lists multiple drugs, pick the one the answer ultimately identifies as the final result.\n\n"
        "Rules:\n"
        "- Only extract URLs that are explicitly present in the answer text.\n"
        "- For URLs missing protocol, prepend http://.\n"
        "- Do not infer or create any URLs.\n"
        "- Preserve the URLs exactly as they appear, normalized with protocol.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def format_drug_identity(brand: Optional[str], generic: Optional[str]) -> str:
    if brand and generic:
        return f"{brand} ({generic})"
    if brand:
        return brand
    if generic:
        return generic
    return "the identified drug"


def _set_failed(node, reason: str | None = None):
    node.score = 0.0
    node.status = "failed"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_meets_all_eligibility_criteria(
    evaluator: Evaluator,
    parent_node,
    drug: DrugIdentification,
):
    """
    Build and verify the 'Meets_All_Eligibility_Criteria' branch with 8 critical leaves.
    Each leaf is a single verification step checked against the provided FDA reference URLs.
    """
    drug_label = format_drug_identity(drug.brand_name, drug.generic_name)
    urls: List[str] = drug.fda_reference_urls or []

    criteria_node = evaluator.add_parallel(
        id="Meets_All_Eligibility_Criteria",
        desc="The identified drug satisfies every regulatory/clinical constraint in the prompt.",
        parent=parent_node,
        critical=True,
    )

    # Prepare all eight leaves
    leaf_novel = evaluator.add_leaf(
        id="Novel_Drug_Approval_2024",
        desc="Approved by FDA CDER in 2024 as a novel drug (new molecular entity).",
        parent=criteria_node,
        critical=True,
    )
    leaf_rare_endocrine = evaluator.add_leaf(
        id="Rare_Genetic_Endocrine_Indication",
        desc="Indicated to treat a rare genetic endocrine disorder affecting adrenal hormone production.",
        parent=criteria_node,
        critical=True,
    )
    leaf_orphan = evaluator.add_leaf(
        id="Orphan_Drug_Designation",
        desc="Received FDA orphan drug designation.",
        parent=criteria_node,
        critical=True,
    )
    leaf_breakthrough = evaluator.add_leaf(
        id="Breakthrough_Therapy_Designation",
        desc="Received FDA breakthrough therapy designation.",
        parent=criteria_node,
        critical=True,
    )
    leaf_age = evaluator.add_leaf(
        id="Adult_And_Pediatric_4plus",
        desc="Approved for adults and for pediatric patients aged 4 years and older.",
        parent=criteria_node,
        critical=True,
    )
    leaf_dosage_forms = evaluator.add_leaf(
        id="At_Least_Two_Dosage_Forms",
        desc="Available in at least two different pharmaceutical dosage forms.",
        parent=criteria_node,
        critical=True,
    )
    leaf_adjunct = evaluator.add_leaf(
        id="Adjunctive_To_Glucocorticoid_Replacement",
        desc="Indicated as adjunctive treatment to glucocorticoid replacement therapy.",
        parent=criteria_node,
        critical=True,
    )
    leaf_december = evaluator.add_leaf(
        id="Approved_In_December_2024",
        desc="FDA approval occurred in December 2024.",
        parent=criteria_node,
        critical=True,
    )

    # If no FDA URLs are provided, mark all leaves failed (they require evidence)
    if not urls:
        for node in [
            leaf_novel, leaf_rare_endocrine, leaf_orphan, leaf_breakthrough,
            leaf_age, leaf_dosage_forms, leaf_adjunct, leaf_december
        ]:
            _set_failed(node, "No FDA URLs provided in answer to support this criterion.")
        return

    claims_and_sources: List[tuple[str, List[str], Any, Optional[str]]] = []

    # 1) Novel drug (NME) approved by CDER in 2024
    claim_novel = (
        f"{drug_label} was approved by FDA's Center for Drug Evaluation and Research (CDER) in 2024 "
        f"as a novel drug (new molecular entity, NME)."
    )
    add_ins_novel = (
        "Confirm this page shows the drug is a 'novel drug' or 'new molecular entity (NME)' and that the approval year is 2024. "
        "Evidence from the CDER 'Novel Drug Therapy Approvals' 2024 report or FDA approvals listings is acceptable. "
        "A clear indication that CDER (not CBER) oversaw the approval is required; textual cues that the drug is in CDER's 2024 novel drugs list suffice."
    )
    claims_and_sources.append((claim_novel, urls, leaf_novel, add_ins_novel))

    # 2) Rare genetic endocrine disorder affecting adrenal hormone production
    claim_rare = (
        f"{drug_label} is indicated to treat a rare genetic endocrine disorder that affects adrenal hormone production."
    )
    add_ins_rare = (
        "Verify the indication describes a rare genetic endocrine disease impacting adrenal hormone synthesis or regulation "
        "(e.g., congenital adrenal hyperplasia due to 21-hydroxylase deficiency). The page should explicitly support this."
    )
    claims_and_sources.append((claim_rare, urls, leaf_rare_endocrine, add_ins_rare))

    # 3) Orphan Drug designation
    claim_orphan = f"{drug_label} received FDA Orphan Drug designation."
    add_ins_orphan = (
        "Look for explicit mention of 'Orphan Drug designation' on the FDA page or in an FDA-hosted document. "
        "References to the Orphan Drug Product designation or FDA press release that states orphan designation are acceptable."
    )
    claims_and_sources.append((claim_orphan, urls, leaf_orphan, add_ins_orphan))

    # 4) Breakthrough Therapy designation
    claim_breakthrough = f"{drug_label} received FDA Breakthrough Therapy designation."
    add_ins_breakthrough = (
        "Look for explicit mention of 'Breakthrough Therapy designation' on the FDA page or in an FDA-hosted document. "
        "Press releases or approval communications that state breakthrough therapy designation are acceptable."
    )
    claims_and_sources.append((claim_breakthrough, urls, leaf_breakthrough, add_ins_breakthrough))

    # 5) Adults and pediatric patients aged 4 years and older
    claim_age = (
        f"The approval for {drug_label} includes adults and pediatric patients aged 4 years and older (inclusive)."
    )
    add_ins_age = (
        "Verify in the indication/patient population text that both adults and pediatric patients age 4+ are covered. "
        "Mentions like 'adults and pediatric patients 4 years and older' or similar phrasing suffice."
    )
    claims_and_sources.append((claim_age, urls, leaf_age, add_ins_age))

    # 6) At least two dosage forms
    claim_dosage = f"{drug_label} is available in at least two different pharmaceutical dosage forms."
    add_ins_dosage = (
        "Check the Dosage Forms and Strengths or product presentation sections. "
        "Examples of distinct dosage forms include: tablet vs oral solution/suspension, capsule vs oral granules, or injection vs oral form. "
        "Confirm there are two or more different dosage forms (not just strengths)."
    )
    claims_and_sources.append((claim_dosage, urls, leaf_dosage_forms, add_ins_dosage))

    # 7) Adjunctive to glucocorticoid replacement therapy
    claim_adjunct = (
        f"{drug_label} is indicated for use as adjunctive treatment to glucocorticoid replacement therapy."
    )
    add_ins_adjunct = (
        "Verify that the indication explicitly states use as 'adjunct' to glucocorticoid replacement therapy "
        "(e.g., adjunct to hydrocortisone/glucocorticoids)."
    )
    claims_and_sources.append((claim_adjunct, urls, leaf_adjunct, add_ins_adjunct))

    # 8) Approved in December 2024
    claim_december = f"{drug_label} received FDA approval in December 2024."
    add_ins_december = (
        "Confirm the FDA approval date is in December 2024 (e.g., December X, 2024). "
        "Approval communications, press releases, or listings must show a December 2024 approval date."
    )
    claims_and_sources.append((claim_december, urls, leaf_december, add_ins_december))

    # Run verifications in parallel for efficiency
    await evaluator.batch_verify(claims_and_sources)


async def build_required_output_provided(
    evaluator: Evaluator,
    parent_node,
    drug: DrugIdentification,
):
    """
    Build and verify the 'Required_Output_Provided' branch:
    - Provides brand name (simple presence check against the answer)
    - Provides official FDA reference URL supporting approval details
    """
    output_node = evaluator.add_parallel(
        id="Required_Output_Provided",
        desc="The response includes the required fields requested by the prompt.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Provides brand name
    leaf_brand = evaluator.add_leaf(
        id="Provides_Proprietary_Brand_Name",
        desc="Provides the drug’s proprietary (brand) name.",
        parent=output_node,
        critical=True,
    )
    if drug.brand_name and drug.brand_name.strip():
        claim_brand = f"The answer explicitly provides the proprietary (brand) name: '{drug.brand_name}'."
        await evaluator.verify(
            claim=claim_brand,
            node=leaf_brand,
            sources=None,
            additional_instruction=(
                "Check the provided answer text to confirm that this brand name string appears clearly "
                "as the proprietary/brand name for the identified drug."
            ),
        )
    else:
        _set_failed(leaf_brand, "No brand name extracted from the answer.")

    # Leaf: Provides official FDA reference URL that supports approval details
    leaf_fda_ref = evaluator.add_leaf(
        id="Provides_Official_FDA_Reference_URL",
        desc="Provides at least one reference URL from an official FDA source that supports the stated approval details (e.g., FDA annual report page/PDF, press release, or another FDA-hosted approval communication).",
        parent=output_node,
        critical=True,
    )

    urls = drug.fda_reference_urls or []
    if not urls:
        _set_failed(leaf_fda_ref, "No official FDA reference URL provided in the answer.")
    else:
        claim_fda_ref = (
            f"This webpage is an official FDA source (fda.gov) and provides/supports the approval details for {format_drug_identity(drug.brand_name, drug.generic_name)} "
            f"from 2024 (e.g., brand/generic identification, indication, and approval timing)."
        )
        add_ins_ref = (
            "Confirm the page is hosted on an official FDA domain (e.g., fda.gov) and that it supports the approval details "
            "for the identified drug. Supporting details can include indication, special designations (orphan/breakthrough), "
            "approval year/month, patient population, or dosage forms. At least one of the provided URLs must satisfy this."
        )
        await evaluator.verify(
            claim=claim_fda_ref,
            node=leaf_fda_ref,
            sources=urls,
            additional_instruction=add_ins_ref,
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
    Evaluate an answer for the FDA 2024 rare endocrine drug identification task.
    Returns a structured summary with the verification tree and final score.
    """
    # Initialize evaluator (root is non-critical by framework design)
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
        default_model=model,
    )

    # Add a critical sequential child to represent the task node per rubric
    task_node = evaluator.add_sequential(
        id="Drug_Identification_Task",
        desc="Identify a single FDA-approved (CDER) 2024 novel drug meeting all listed criteria, and provide the brand name plus an official FDA reference URL supporting the approval details.",
        parent=root,
        critical=True,
    )

    # Extract core fields from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_drug_identification(),
        template_class=DrugIdentification,
        extraction_name="drug_identification",
    )

    # Build and verify rubric branches
    await build_meets_all_eligibility_criteria(evaluator, task_node, extracted)
    await build_required_output_provided(evaluator, task_node, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()