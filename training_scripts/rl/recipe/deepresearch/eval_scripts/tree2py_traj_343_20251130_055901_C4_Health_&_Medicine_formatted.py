import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fda_il15_nmibc_medication_2024"
TASK_DESCRIPTION = (
    "Identify the brand name of the FDA-approved immunotherapy medication that meets all of the following criteria: "
    "(1) approved by the FDA on April 22, 2024; "
    "(2) indicated for the treatment of BCG-unresponsive non-muscle invasive bladder cancer (NMIBC) with carcinoma in situ (CIS) with or without papillary tumors; "
    "(3) is a first-in-class IL-15 receptor agonist; "
    "(4) administered intravesically (directly into the bladder); "
    "(5) used in combination with Bacillus Calmette-Guérin (BCG); "
    "(6) has a recommended induction therapy dosage of 400 mcg administered intravesically once weekly for 6 weeks; and "
    "(7) is manufactured by ImmunityBio. Please provide the brand name and confirm each of these seven specifications."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MedicationExtraction(BaseModel):
    brand_name: Optional[str] = None
    fda_approval_date: Optional[str] = None
    disease_indication: Optional[str] = None
    mechanism: Optional[str] = None
    administration_route: Optional[str] = None
    combination_therapy: Optional[str] = None
    dosage_induction: Optional[str] = None
    manufacturer: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_medication() -> str:
    return (
        "From the provided answer, extract the following fields about the single FDA-approved immunotherapy medication "
        "that meets the task criteria. Return null for any missing field.\n"
        "Fields to extract:\n"
        "1. brand_name: The brand/trade name of the medication.\n"
        "2. fda_approval_date: The FDA approval date as stated in the answer (keep exact formatting).\n"
        "3. disease_indication: The indication text provided in the answer for BCG-unresponsive NMIBC with CIS with or without papillary tumors.\n"
        "4. mechanism: The mechanism of action text, including whether it is first-in-class and IL-15 receptor agonist wording.\n"
        "5. administration_route: The administration route wording, e.g., intravesical.\n"
        "6. combination_therapy: The text stating use in combination with BCG.\n"
        "7. dosage_induction: The dosage schedule wording, including 400 mcg weekly for 6 weeks intravesical, as described.\n"
        "8. manufacturer: The company/manufacturer name (e.g., ImmunityBio).\n"
        "9. sources: An array of all URLs explicitly cited in the answer that are relevant to and support any of the above claims. "
        "   Include only valid URLs; if none are cited, return an empty array."
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_medication(evaluator: Evaluator, parent_node, extracted: MedicationExtraction) -> None:
    """
    Construct the verification tree and run all checks according to the rubric.
    """
    med_node = evaluator.add_parallel(
        id="Medication_Identification",
        desc="Correctly identify and provide details about the FDA-approved immunotherapy medication meeting all specified criteria",
        parent=parent_node,
        critical=True  # All child checks are mandatory; failing any should fail the parent
    )

    brand = (extracted.brand_name or "").strip()
    med_ref = brand if brand else "the medication"
    sources_list = extracted.sources if extracted.sources else []

    # Create leaf nodes
    brand_node = evaluator.add_leaf(
        id="Brand_Name",
        desc="The brand name of the medication is correctly provided",
        parent=med_node,
        critical=True
    )
    approval_date_node = evaluator.add_leaf(
        id="FDA_Approval_Date",
        desc="The medication was approved by the FDA on April 22, 2024",
        parent=med_node,
        critical=True
    )
    indication_node = evaluator.add_leaf(
        id="Disease_Indication",
        desc="The medication is indicated for BCG-unresponsive non-muscle invasive bladder cancer (NMIBC) with carcinoma in situ (CIS) with or without papillary tumors",
        parent=med_node,
        critical=True
    )
    moa_node = evaluator.add_leaf(
        id="Mechanism_of_Action",
        desc="The medication is an IL-15 receptor agonist and is identified as first-in-class with this mechanism",
        parent=med_node,
        critical=True
    )
    route_node = evaluator.add_leaf(
        id="Administration_Route",
        desc="The medication is administered intravesically (directly into the bladder)",
        parent=med_node,
        critical=True
    )
    combo_node = evaluator.add_leaf(
        id="Combination_Therapy",
        desc="The medication is used in combination with Bacillus Calmette-Guérin (BCG)",
        parent=med_node,
        critical=True
    )
    dosage_node = evaluator.add_leaf(
        id="Dosage_Induction",
        desc="The recommended dosage for induction therapy is 400 mcg administered intravesically once weekly for 6 weeks",
        parent=med_node,
        critical=True
    )
    manufacturer_node = evaluator.add_leaf(
        id="Manufacturer",
        desc="The manufacturer is identified as ImmunityBio",
        parent=med_node,
        critical=True
    )

    claims_and_sources = [
        (
            f"The brand (trade) name of the medication meeting the specified criteria is '{brand}'.",
            sources_list,
            brand_node,
            "Confirm from the cited sources that the medication described in the answer is marketed under the brand name given. "
            "Allow minor stylization or capitalization differences. If the brand is absent or mismatched, mark as not supported."
        ),
        (
            f"{med_ref} was approved by the U.S. FDA on April 22, 2024.",
            sources_list,
            approval_date_node,
            "Check FDA press releases, approval notices, prescribing information, or official communications showing the approval date. "
            "Accept reasonable date formatting variants (e.g., '22 April 2024'). Distinguish approval from other regulatory milestones."
        ),
        (
            f"{med_ref} is indicated for BCG-unresponsive non-muscle invasive bladder cancer (NMIBC) with carcinoma in situ (CIS), "
            "with or without papillary tumors.",
            sources_list,
            indication_node,
            "Verify the indication wording in the official label or credible sources; allow minor synonyms or rephrasings. "
            "Ensure that the 'BCG-unresponsive', 'NMIBC', 'CIS', and 'with or without papillary tumors' components are present."
        ),
        (
            f"{med_ref} is a first-in-class IL-15 receptor agonist.",
            sources_list,
            moa_node,
            "Check mechanism-of-action descriptions. Allow synonyms like 'IL-15 superagonist', 'IL-15RA superagonist', or 'IL-15 receptor alpha agonist'. "
            "Also confirm the 'first-in-class' designation when available from authoritative sources."
        ),
        (
            f"{med_ref} is administered intravesically (directly into the bladder).",
            sources_list,
            route_node,
            "Verify administration route details from dosing/administration sections. Ensure it is intravesical; allow close phrasing variants."
        ),
        (
            f"{med_ref} is used in combination with Bacillus Calmette-Guérin (BCG).",
            sources_list,
            combo_node,
            "Confirm combination use with BCG from label or authoritative clinical sources. Allow 'with BCG' phrasing variations."
        ),
        (
            f"The recommended induction therapy dosage for {med_ref} is 400 mcg administered intravesically once weekly for 6 weeks.",
            sources_list,
            dosage_node,
            "Check dosing recommendations in the label or authoritative sources. "
            "Allow 'μg' as equivalent to 'mcg' and accept 'six weeks' phrasing variations. "
            "Ensure weekly schedule and intravesical route are present."
        ),
        (
            f"The manufacturer of {med_ref} is ImmunityBio.",
            sources_list,
            manufacturer_node,
            "Confirm the manufacturer/company as ImmunityBio (e.g., 'ImmunityBio, Inc.'). Minor naming variations acceptable."
        ),
    ]

    # Run verifications (in parallel for efficiency)
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate the answer for the FDA-approved immunotherapy medication identification and specifications.
    """
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_medication(),
        template_class=MedicationExtraction,
        extraction_name="medication_extraction",
    )

    # Record helpful custom info to aid debugging
    evaluator.add_custom_info(
        info={
            "brand_name": extraction.brand_name,
            "sources_count": len(extraction.sources),
            "sources_preview": extraction.sources[:5],
        },
        info_type="extraction_summary",
        info_name="medication_extraction_summary"
    )

    # Build verification tree and run checks
    await verify_medication(evaluator, root, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()