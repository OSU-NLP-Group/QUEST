import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "orphan_neuro_metabolic_2024"
TASK_DESCRIPTION = (
    "In 2024, the FDA approved several novel drugs with orphan drug designation for rare diseases. "
    "I am researching drugs that specifically target neurological manifestations of genetic metabolic disorders that can be administered orally to diverse patient populations.\n\n"
    "Identify an FDA-approved drug that meets all of the following criteria:\n\n"
    "1. Received FDA orphan drug designation for a rare disease (affecting fewer than 200,000 people in the United States)\n"
    "2. Was approved by the FDA in 2024 as a novel drug therapy (never before approved or marketed in the U.S.)\n"
    "3. Is specifically indicated for treating neurological manifestations of a rare genetic metabolic disorder\n"
    "4. Is approved for use in both adult and pediatric patient populations\n"
    "5. Is formulated as an oral preparation (not injectable or intravenous)\n\n"
    "For the identified drug, provide:\n"
    "- The drug's proprietary (brand) name\n"
    "- The specific rare disease it treats\n"
    "- The FDA approval date in 2024\n"
    "- A reference URL from FDA.gov confirming its orphan drug designation\n"
    "- A reference URL from FDA.gov confirming its 2024 approval\n"
    "- A reference URL describing the disease's neurological and genetic metabolic characteristics\n"
    "- Confirmation of its patient population coverage (adults and pediatric patients)\n"
    "- Confirmation of its oral formulation type"
)


# ------------------------------ Data Models ------------------------------ #
class DrugInfo(BaseModel):
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    disease_name: Optional[str] = None
    approval_date: Optional[str] = None
    fda_orphan_url: Optional[str] = None
    fda_approval_url: Optional[str] = None
    disease_url: Optional[str] = None
    novel_list_url: Optional[str] = None
    oral_formulation_type: Optional[str] = None
    adult_pediatric_confirmation_text: Optional[str] = None


# -------------------------- Extraction Prompt ---------------------------- #
def prompt_extract_drug() -> str:
    return """
    Extract exactly one drug candidate from the answer that the author intends to present as satisfying all constraints.
    If multiple drugs are mentioned, choose the FIRST one that appears to meet the constraints and is presented as the main answer.

    For that single drug, extract the following fields:
    - brand_name: The proprietary/brand name of the drug, exactly as written in the answer. If not provided, return null.
    - generic_name: The generic or nonproprietary name, if mentioned. If not provided, return null.
    - disease_name: The specific rare disease the drug treats. If not provided, return null.
    - approval_date: The FDA approval calendar date as written in the answer (e.g., 'March 12, 2024', '2024-03-12'). If not provided, return null.
    - fda_orphan_url: A URL from FDA.gov that confirms orphan drug designation for this drug (must be explicitly present in the answer text). If not provided, return null.
    - fda_approval_url: A URL from FDA.gov that confirms the drug’s 2024 approval (must be explicitly present in the answer text). If not provided, return null.
    - disease_url: A URL describing the disease’s neurological characteristics and its genetic metabolic nature (must be explicitly present in the answer text). If not provided, return null.
    - novel_list_url: If the answer provides a URL to the 2024 CDER novel drug approvals list or a page listing orphan-designated novel drugs, extract it; otherwise return null.
    - oral_formulation_type: The stated oral formulation type (e.g., tablet, capsule, oral suspension, oral solution), exactly as in the answer. If not provided, return null.
    - adult_pediatric_confirmation_text: The exact phrase or sentence in the answer that explicitly confirms coverage for both adult and pediatric patients. If not provided, return null.

    SPECIAL RULES FOR URLS:
    - Extract only URLs that are explicitly present in the answer. Do not invent or infer any URLs.
    - URLs can be plain text or inside markdown. Always return the actual URL string.
    - If a URL is missing a protocol, prepend 'http://'.
    """


# ------------------------------ Helpers ---------------------------------- #
def is_nonempty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def is_fda_url(url: Optional[str]) -> bool:
    return is_nonempty_str(url) and ("fda.gov" in url.lower())


def get_display_drug_name(info: DrugInfo) -> str:
    """
    Prefer brand name; if missing, fall back to generic name; otherwise empty string.
    """
    if is_nonempty_str(info.brand_name):
        return info.brand_name.strip()
    if is_nonempty_str(info.generic_name):
        return info.generic_name.strip()
    return ""


# --------------------------- Verification Logic -------------------------- #
async def build_and_verify_drug_tree(evaluator: Evaluator, drug: DrugInfo, parent_node) -> None:
    """
    Build the verification tree according to the rubric and execute verifications.
    All nodes under Drug_Answer are critical (as per rubric).
    """

    # Drug_Answer root under the evaluator root (critical, parallel aggregation)
    drug_answer_node = evaluator.add_parallel(
        id="Drug_Answer",
        desc="Identify exactly one FDA-approved drug that satisfies all stated constraints and provide all required fields and URLs.",
        parent=parent_node,
        critical=True
    )

    # 1) Drug Identification Fields
    id_fields_node = evaluator.add_parallel(
        id="Drug_Identification_Fields",
        desc="Provide the drug identification fields requested in the prompt.",
        parent=drug_answer_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_nonempty_str(drug.brand_name),
        id="Brand_Name_Provided",
        desc="Answer includes the drug's proprietary (brand) name.",
        parent=id_fields_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_nonempty_str(drug.disease_name),
        id="Rare_Disease_Name_Provided",
        desc="Answer includes the specific rare disease the drug treats.",
        parent=id_fields_node,
        critical=True
    )

    # 2) Orphan designation + FDA.gov URL
    orphan_node = evaluator.add_parallel(
        id="Orphan_Designation_Rare_Disease_And_FDA_URL",
        desc="Drug has FDA orphan drug designation for a rare disease (<200,000 in the US) and the answer provides an FDA.gov URL confirming the orphan designation.",
        parent=drug_answer_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_fda_url(drug.fda_orphan_url),
        id="FDAgov_Orphan_Designation_URL_Provided",
        desc="Provide an FDA.gov reference URL confirming orphan drug designation.",
        parent=orphan_node,
        critical=True
    )

    orphan_leaf = evaluator.add_leaf(
        id="Orphan_Designation_For_Rare_Disease_Under_200k",
        desc="Drug received FDA orphan drug designation for the relevant rare disease meeting the <200,000 US prevalence criterion (as stated in the prompt/constraints).",
        parent=orphan_node,
        critical=True
    )
    orphan_claim_drug = get_display_drug_name(drug)
    orphan_claim_disease = drug.disease_name or ""
    orphan_claim = (
        f"FDA orphan drug designation was granted to {orphan_claim_drug} for the treatment of {orphan_claim_disease}. "
        f"Orphan designation corresponds to rare diseases under 200,000 U.S. prevalence."
    )
    await evaluator.verify(
        claim=orphan_claim,
        node=orphan_leaf,
        sources=drug.fda_orphan_url,
        additional_instruction=(
            "Verify the page confirms orphan drug designation for the specific drug and disease. "
            "You may treat the '<200,000' criterion as inherent to FDA orphan designation even if not explicitly stated on the page."
        ),
    )

    # 3) FDA approval (2024) as novel + FDA.gov approval URL
    approval_node = evaluator.add_parallel(
        id="FDA_Approval_2024_Novel_And_FDA_URL",
        desc="Drug was approved by FDA in 2024 as a novel drug therapy (first-time US approval/marketing), and the answer provides the approval date and an FDA.gov approval URL.",
        parent=drug_answer_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_fda_url(drug.fda_approval_url),
        id="FDAgov_2024_Approval_URL_Provided",
        desc="Provide an FDA.gov reference URL confirming the drug’s 2024 approval.",
        parent=approval_node,
        critical=True
    )

    approval_date_leaf = evaluator.add_leaf(
        id="FDA_Approval_Date_In_2024_Provided",
        desc="Answer provides a specific FDA approval calendar date and that date is in 2024.",
        parent=approval_node,
        critical=True
    )
    approval_date_str = drug.approval_date or ""
    approval_claim_drug = get_display_drug_name(drug)
    approval_date_claim = (
        f"The FDA approved {approval_claim_drug} on '{approval_date_str}', and that date is in the year 2024."
    )
    await evaluator.verify(
        claim=approval_date_claim,
        node=approval_date_leaf,
        sources=drug.fda_approval_url,
        additional_instruction=(
            "Confirm that the FDA.gov page shows an approval date in 2024. "
            "Minor formatting differences in the date are acceptable if clearly the same calendar date in 2024."
        ),
    )

    novel_leaf = evaluator.add_leaf(
        id="Novel_Drug_Therapy_First_Time_US",
        desc="Drug is a novel drug therapy (never before approved or marketed in the U.S.).",
        parent=approval_node,
        critical=True
    )
    novel_sources: List[str] = []
    if is_nonempty_str(drug.novel_list_url):
        novel_sources.append(drug.novel_list_url)  # CDER 2024 novel drug approvals list if provided
    if is_nonempty_str(drug.fda_approval_url):
        novel_sources.append(drug.fda_approval_url)
    novel_claim = (
        f"{approval_claim_drug} is a novel drug therapy approved in 2024 (first-time U.S. approval/marketing). "
        f"Recognition can be indicated by inclusion in CDER's 2024 novel drug approvals or by statements such as 'new molecular entity (NME)' or 'novel drug'."
    )
    await evaluator.verify(
        claim=novel_claim,
        node=novel_leaf,
        sources=novel_sources if novel_sources else None,
        additional_instruction=(
            "Verify either on the CDER 2024 novel approvals list or the FDA page that the drug is considered a 'novel drug' "
            "or 'new molecular entity' or otherwise first-time U.S. approval/marketing in 2024."
        ),
    )

    # 4) Neurologic indication + disease genetic metabolic characteristics + disease URL
    neuro_node = evaluator.add_parallel(
        id="Neurologic_Indication_And_Genetic_Metabolic_Disease_With_URL",
        desc="Drug is specifically indicated for neurological manifestations of a rare genetic metabolic disorder characterized by progressive neurological symptoms, and the answer provides a supporting disease-characteristics URL.",
        parent=drug_answer_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_nonempty_str(drug.disease_url),
        id="Disease_Characteristics_URL_Provided",
        desc="Provide a reference URL describing the disease’s neurological characteristics and its genetic metabolic nature (including progressive neurological symptoms).",
        parent=neuro_node,
        critical=True
    )

    neuro_indication_leaf = evaluator.add_leaf(
        id="Indicated_For_Neurological_Manifestations",
        desc="Drug is specifically indicated for treating neurological manifestations (not merely general symptoms).",
        parent=neuro_node,
        critical=True
    )
    neuro_claim_drug = get_display_drug_name(drug)
    neuro_claim_disease = drug.disease_name or ""
    neuro_claim = (
        f"{neuro_claim_drug} is indicated to treat neurological manifestations of {neuro_claim_disease}."
    )
    await evaluator.verify(
        claim=neuro_claim,
        node=neuro_indication_leaf,
        sources=drug.fda_approval_url,
        additional_instruction=(
            "Confirm that the FDA approval or label page explicitly associates the indication with neurological manifestations. "
            "Look for terms like 'neurologic', 'neurological', 'central nervous system', 'seizures', 'neurodevelopmental', "
            "'neuropathy', 'movement disorder', or similar explicit neurological involvement."
        ),
    )

    disease_character_leaf = evaluator.add_leaf(
        id="Disease_Is_Genetic_Metabolic_With_Progressive_Neurologic_Symptoms",
        desc="Targeted rare disease is a genetic metabolic disorder characterized by progressive neurological symptoms (per constraints).",
        parent=neuro_node,
        critical=True
    )
    disease_character_claim_disease = drug.disease_name or ""
    disease_character_claim = (
        f"{disease_character_claim_disease} is a genetic metabolic disorder and is characterized by progressive neurological symptoms."
    )
    await evaluator.verify(
        claim=disease_character_claim,
        node=disease_character_leaf,
        sources=drug.disease_url,
        additional_instruction=(
            "Verify the disease has both genetic metabolic etiology (e.g., inborn error of metabolism, enzyme deficiency, metabolic pathway defect) "
            "and progressive neurological manifestations (e.g., neurodegeneration, worsening CNS symptoms)."
        ),
    )

    # 5) Adult and pediatric coverage confirmed
    coverage_leaf = evaluator.add_leaf(
        id="Adult_And_Pediatric_Coverage_Confirmed",
        desc="Drug is approved for use in both adult and pediatric populations and the answer explicitly confirms this coverage.",
        parent=drug_answer_node,
        critical=True
    )
    coverage_claim_drug = get_display_drug_name(drug)
    coverage_claim = (
        f"{coverage_claim_drug} is FDA-approved for use in both adult and pediatric patient populations."
    )
    await evaluator.verify(
        claim=coverage_claim,
        node=coverage_leaf,
        sources=drug.fda_approval_url,
        additional_instruction=(
            "Confirm both adult and pediatric populations are included. Pediatric may be expressed via specific age ranges (e.g., ≥12 years, ≥2 years)."
        ),
    )

    # 6) Oral formulation and stated type
    oral_node = evaluator.add_parallel(
        id="Oral_Formulation_And_Type_Stated",
        desc="Drug is formulated as an oral preparation (not injectable/IV) and the answer states the oral formulation type.",
        parent=drug_answer_node,
        critical=True
    )

    oral_route_leaf = evaluator.add_leaf(
        id="Oral_Not_Injectable_IV",
        desc="Drug is an oral preparation (not injectable or intravenous).",
        parent=oral_node,
        critical=True
    )
    oral_claim_drug = get_display_drug_name(drug)
    oral_claim = f"{oral_claim_drug} is administered orally and is not injectable or intravenous."
    await evaluator.verify(
        claim=oral_claim,
        node=oral_route_leaf,
        sources=drug.fda_approval_url,
        additional_instruction=(
            "Verify the route of administration is oral (e.g., tablet, capsule, oral solution/suspension) and confirm that it's not injection or IV."
        ),
    )

    evaluator.add_custom_node(
        result=is_nonempty_str(drug.oral_formulation_type),
        id="Oral_Formulation_Type_Stated",
        desc="Answer states the oral formulation type (e.g., tablet, capsule, oral suspension).",
        parent=oral_node,
        critical=True
    )

    # 7) CDER 2024 orphan novel list membership
    cder_leaf = evaluator.add_leaf(
        id="CDER_2024_26_Orphan_Novel_List_Membership",
        desc="Drug is one of the 26 orphan-designated novel drugs approved by CDER in 2024 (per constraints).",
        parent=drug_answer_node,
        critical=True
    )
    cder_sources: List[str] = []
    if is_nonempty_str(drug.novel_list_url):
        cder_sources.append(drug.novel_list_url)
    if is_nonempty_str(drug.fda_approval_url):
        cder_sources.append(drug.fda_approval_url)
    cder_claim_drug = get_display_drug_name(drug)
    cder_claim = (
        f"{cder_claim_drug} is included among CDER's 26 orphan-designated novel drug approvals for 2024."
    )
    await evaluator.verify(
        claim=cder_claim,
        node=cder_leaf,
        sources=cder_sources if cder_sources else None,
        additional_instruction=(
            "Check the CDER 2024 novel drug approvals listing and confirm the drug is specifically counted among the orphan-designated novel approvals. "
            "If the page labels entries as 'orphan', use that to verify membership."
        ),
    )


# ----------------------------- Main Function ----------------------------- #
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
    Entry point to evaluate an agent's answer for the orphan/neurologic/metabolic 2024 FDA drug task.
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

    # Extract the single drug record from the answer
    drug_info = await evaluator.extract(
        prompt=prompt_extract_drug(),
        template_class=DrugInfo,
        extraction_name="drug_extraction",
    )

    # Add custom info block (optional)
    evaluator.add_custom_info(
        info={
            "task_focus": "2024 FDA novel drug with orphan designation; neurologic manifestations of genetic metabolic disorder; oral route; adult+pediatric coverage.",
            "expected_sources": {
                "orphan_url": "FDA.gov",
                "approval_url": "FDA.gov",
                "disease_url": "author-provided disease characteristics page",
                "novel_list_url": "CDER novel approvals 2024 page (optional but helpful)"
            }
        },
        info_type="context"
    )

    # Build and verify the rubric tree for the single drug
    await build_and_verify_drug_tree(evaluator, drug_info, root)

    return evaluator.get_summary()