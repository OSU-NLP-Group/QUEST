import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_ccr_16_2032_5_emergency_ads"
TASK_DESCRIPTION = (
    "According to California Code of Regulations Title 16, Section 2032.5, what three specific pieces of "
    "information must be clearly stated in advertisements for any veterinary premises that advertises itself "
    "as an emergency veterinary clinic or hospital?"
)

GROUND_TRUTH_REQUIREMENTS = [
    "Advertisements must clearly state the hours the facility will provide emergency services.",
    "Advertisements must clearly state that a licensed veterinarian is on the premises during the posted emergency hours.",
    "Advertisements must clearly state the address and telephone number of the premises."
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EmergencyAdsExtraction(BaseModel):
    """
    Extract the three required items as they are presented in the answer.
    Each field should contain the exact phrase or a close paraphrase from the answer.
    Return null if the answer does not mention the item.
    """
    licensed_veterinarian_statement: Optional[str] = None
    emergency_hours_statement: Optional[str] = None
    contact_information_statement: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    From the answer, extract the three specific pieces of information that it says must be clearly stated in
    advertisements for any veterinary premises that advertises itself as an emergency veterinary clinic or hospital.
    Return a JSON object with the following fields:
    - licensed_veterinarian_statement: The exact phrase (or very close paraphrase) indicating that advertisements must
      state that a licensed veterinarian is on the premises during the posted emergency hours. If missing, return null.
    - emergency_hours_statement: The exact phrase (or very close paraphrase) indicating that advertisements must
      state the hours the facility will provide emergency services. If missing, return null.
    - contact_information_statement: The exact phrase (or very close paraphrase) indicating that advertisements must
      state the address and telephone number of the premises. If missing, return null.

    Notes:
    - Prefer text spans quoted from the answer. If the answer uses synonyms (e.g., "on-site vet during emergency hours"
      for "a licensed veterinarian on the premises during posted emergency hours"), extract that paraphrase.
    - Do not invent content; set a field to null if the answer does not include it.
    """


# --------------------------------------------------------------------------- #
# Helpful source URLs (multiple mirrors of CCR 16 §2032.5)                    #
# --------------------------------------------------------------------------- #
def build_law_urls() -> List[str]:
    return [
        # Commonly accessible regulation aggregators
        "https://law.justia.com/regulations/california/code-of-regulations/title-16/division-20/article-4/section-2032-5/",
        "https://law.lawstack.com/california/regulations/title-16-professional-and-vocational-regulations/division-20-veterinary-medical-board/article-4-practice/section-2032-5-emergency-animal-hospital-or-emergency-veterinary-clinic",
        "https://law.onecle.com/california/regulations/title-16/section-2032.5.html",
        # Some sites normalize section numbers without the dot
        "https://casetext.com/regulation/california-code-of-regulations/title-16-professional-and-vocational-regulations/division-20-veterinary-medical-board/article-4-practice/section-20325-emergency-animal-hospital-or-emergency-veterinary-clinic",
    ]


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def add_requirement_checks(
    evaluator: Evaluator,
    parent,
    *,
    id_base: str,
    parent_desc: str,
    extracted_text: Optional[str],
    in_answer_claim: str,
    law_claim: str,
    additional_answer_instruction: Optional[str] = None,
    additional_law_instruction: Optional[str] = None,
) -> None:
    """
    Create a critical parallel block for one requirement with:
      - existence check (custom)
      - in-answer verification (simple verify)
      - law support verification (verify by URLs)
    """
    block = evaluator.add_parallel(
        id=id_base,
        desc=parent_desc,
        parent=parent,
        critical=True
    )

    # Existence check (critical gate)
    exists_node = evaluator.add_custom_node(
        result=bool(extracted_text and extracted_text.strip()),
        id=f"{id_base}_exists",
        desc=f"{parent_desc} — mentioned in the answer",
        parent=block,
        critical=True
    )

    # Check presence explicitly in the answer (LLM check)
    in_answer_node = evaluator.add_leaf(
        id=f"{id_base}_in_answer",
        desc=f"{parent_desc} — explicitly identified in the answer",
        parent=block,
        critical=True
    )
    await evaluator.verify(
        claim=in_answer_claim,
        node=in_answer_node,
        additional_instruction=additional_answer_instruction
        or "Look only at the provided answer text. Accept reasonable paraphrases (e.g., 'on-site veterinarian during emergency hours' "
           "for 'a licensed veterinarian is on the premises during posted emergency hours')."
    )

    # Law support (URLs)
    law_node = evaluator.add_leaf(
        id=f"{id_base}_supported_by_law",
        desc=f"{parent_desc} — supported by CCR Title 16 §2032.5",
        parent=block,
        critical=True
    )
    await evaluator.verify(
        claim=law_claim,
        node=law_node,
        sources=build_law_urls(),
        additional_instruction=additional_law_instruction
        or "Verify that California Code of Regulations Title 16, Section 2032.5 explicitly requires this item to be clearly "
           "stated in advertisements for premises advertising emergency veterinary services. Minor wording differences are fine."
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
    Evaluate whether the answer correctly identifies all three advertisement requirements under CCR Title 16 §2032.5.
    """
    # Initialize evaluator (framework root is non-critical; we'll add a critical top-level criteria node)
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=EmergencyAdsExtraction,
        extraction_name="emergency_ads_requirements_extraction"
    )

    # Add ground truth for reference
    evaluator.add_ground_truth({
        "regulation": "California Code of Regulations Title 16, Section 2032.5 (Emergency Animal Hospital or Emergency Veterinary Clinic)",
        "required_advertisement_items": GROUND_TRUTH_REQUIREMENTS
    })

    # Top-level critical criteria node
    criteria_root = evaluator.add_parallel(
        id="emergency_ad_requirements",
        desc="Correctly identifies all three pieces of information required in emergency veterinary hospital advertisements under California Code of Regulations Title 16, Section 2032.5(a)(2)",
        parent=root,
        critical=True
    )

    # 1) Licensed veterinarian on the premises during posted emergency hours
    await add_requirement_checks(
        evaluator,
        criteria_root,
        id_base="licensed_veterinarian_statement",
        parent_desc="Identifies that advertisements must clearly state a licensed veterinarian is on the premises during the posted emergency hours",
        extracted_text=extracted.licensed_veterinarian_statement,
        in_answer_claim="The answer explicitly identifies that advertisements for emergency veterinary clinics/hospitals must clearly state that a licensed veterinarian is on the premises during the posted emergency hours.",
        law_claim="California Code of Regulations Title 16, Section 2032.5 requires advertisements for any veterinary premises that advertises itself as an emergency veterinary clinic or hospital to clearly state that a licensed veterinarian is on the premises during the posted emergency hours."
    )

    # 2) Hours the facility will provide emergency services
    await add_requirement_checks(
        evaluator,
        criteria_root,
        id_base="emergency_hours_statement",
        parent_desc="Identifies that advertisements must clearly state the hours the facility will provide emergency services",
        extracted_text=extracted.emergency_hours_statement,
        in_answer_claim="The answer explicitly identifies that advertisements must clearly state the hours the facility will provide emergency services.",
        law_claim="California Code of Regulations Title 16, Section 2032.5 requires advertisements for any veterinary premises that advertises itself as an emergency veterinary clinic or hospital to clearly state the hours the facility will provide emergency services."
    )

    # 3) Address and telephone number of the premises
    await add_requirement_checks(
        evaluator,
        criteria_root,
        id_base="contact_information",
        parent_desc="Identifies that advertisements must clearly state the address and telephone number of the premises",
        extracted_text=extracted.contact_information_statement,
        in_answer_claim="The answer explicitly identifies that advertisements must clearly state the address and telephone number of the premises.",
        law_claim="California Code of Regulations Title 16, Section 2032.5 requires advertisements for any veterinary premises that advertises itself as an emergency veterinary clinic or hospital to clearly state the address and telephone number of the premises.",
        additional_answer_instruction="Look only at the answer text and confirm it explicitly mentions both address and telephone/phone number (synonyms like 'location' for address and 'phone' for telephone are acceptable)."
    )

    return evaluator.get_summary()