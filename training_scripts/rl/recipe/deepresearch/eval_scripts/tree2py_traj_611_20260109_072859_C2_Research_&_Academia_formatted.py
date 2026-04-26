import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "uc_postdoc_eligibility_and_nih_ineligibility"
TASK_DESCRIPTION = (
    "A computational neuroscience researcher who is a citizen of Germany completed their PhD in neuroscience at ETH Zurich in May 2022 "
    "and has been working as a postdoctoral researcher at the Max Planck Institute since January 2023. They are now seeking a postdoctoral "
    "position at a University of California campus starting in July 2025 and will require J-1 visa sponsorship. Based on UC system policies "
    "and federal regulations, identify: (1) the three mandatory eligibility requirements this researcher must satisfy to be appointed as a "
    "postdoctoral scholar at any UC campus, including the specific minimum health insurance coverage amounts required for J-1 visa holders, "
    "and (2) which type of NIH institutional training grant (T32 or F32) they would be ineligible for based solely on citizenship requirements."
)

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class StatementWithSources(BaseModel):
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

class J1InsuranceMinimums(BaseModel):
    medical_benefits_min: Optional[str] = None
    repatriation_min: Optional[str] = None
    medical_evacuation_min: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

class UCEligibilityExtraction(BaseModel):
    doctoral_degree_requirement: Optional[StatementWithSources] = None
    five_year_maximum_requirement: Optional[StatementWithSources] = None
    j1_health_insurance_minimums: Optional[J1InsuranceMinimums] = None

class NIHIneligibilityExtraction(BaseModel):
    t32_citizenship: Optional[StatementWithSources] = None
    f32_citizenship: Optional[StatementWithSources] = None

class RequirementsExtraction(BaseModel):
    uc_eligibility_requirements: Optional[UCEligibilityExtraction] = None
    nih_citizenship_ineligibility: Optional[NIHIneligibilityExtraction] = None

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    From the answer, extract the UC postdoctoral eligibility requirements and the NIH citizenship-based ineligibility determinations, along with any URLs cited as sources.

    Required fields:
    1) uc_eligibility_requirements:
       - doctoral_degree_requirement:
         • statement: A concise statement indicating that UC Postdoctoral Scholar appointments require possession of a doctoral degree (PhD, MD, or equivalent).
         • sources: All URLs in the answer that are cited for this requirement.
       - five_year_maximum_requirement:
         • statement: A concise statement indicating that UC imposes a five-year maximum duration limit for total postdoctoral service, and that prior postdoctoral service at other institutions counts toward this limit.
         • sources: All URLs in the answer that are cited for this requirement.
       - j1_health_insurance_minimums:
         • medical_benefits_min: The minimum medical benefits amount required per accident/illness for J-1 visa holders as explicitly stated in the answer (e.g., "$100,000").
         • repatriation_min: The minimum repatriation of remains coverage amount (e.g., "$25,000").
         • medical_evacuation_min: The minimum medical evacuation coverage amount (e.g., "$50,000").
         • sources: All URLs in the answer that are cited for J-1 health insurance minimums.

    2) nih_citizenship_ineligibility:
       - t32_citizenship:
         • statement: A concise statement indicating that NIH NRSA T32 trainees must be U.S. citizens or permanent residents (and therefore a non-U.S. citizen, such as a German citizen, is ineligible based solely on citizenship).
         • sources: All URLs in the answer specifically cited for T32 citizenship requirements.
       - f32_citizenship:
         • statement: A concise statement indicating that NIH NRSA F32 fellows must be U.S. citizens or permanent residents (and therefore a non-U.S. citizen, such as a German citizen, is ineligible based solely on citizenship).
         • sources: All URLs in the answer specifically cited for F32 citizenship requirements.

    Rules:
    - Extract only what is explicitly stated in the answer. Do not invent or infer URLs.
    - URLs may appear as plain text or as markdown links; extract the actual URL strings.
    - If a field is not present in the answer, set it to null (or an empty list for sources).
    - Preserve currency symbols and formatting for amounts exactly as they appear (e.g., "$100,000").
    """

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(sources: Optional[List[str]]) -> Optional[List[str]]:
    if not sources:
        return None
    if isinstance(sources, list) and len(sources) == 0:
        return None
    return sources

# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_uc_requirements(
    evaluator: Evaluator,
    parent: VerificationNode,
    extraction: RequirementsExtraction
) -> VerificationNode:
    uc_node = evaluator.add_parallel(
        id="uc_eligibility_requirements",
        desc="Identify the three mandatory eligibility requirements for UC postdoctoral scholar appointment",
        parent=parent,
        critical=True
    )

    uc = extraction.uc_eligibility_requirements or UCEligibilityExtraction()

    # Leaf 1: Doctoral degree requirement
    dd_leaf = evaluator.add_leaf(
        id="doctoral_degree_requirement",
        desc="States that the appointee must possess a doctoral degree (PhD, MD, or equivalent) for UC postdoctoral appointment eligibility",
        parent=uc_node,
        critical=True
    )
    dd_claim = "University of California Postdoctoral Scholar appointments require that appointees possess a doctoral degree (PhD, MD, or equivalent)."
    dd_sources = _safe_sources((uc.doctoral_degree_requirement or StatementWithSources()).sources)
    await evaluator.verify(
        claim=dd_claim,
        node=dd_leaf,
        sources=dd_sources,
        additional_instruction="Verify against UC policy pages (e.g., APM-390 or campus academic personnel/postdoc pages). Minor wording differences are acceptable if the requirement is clearly present."
    )

    # Leaf 2: Five-year maximum requirement
    five_leaf = evaluator.add_leaf(
        id="five_year_maximum_requirement",
        desc="States that the appointee must comply with the UC five-year maximum duration limit for total postdoctoral service (including prior postdoc experience at other institutions)",
        parent=uc_node,
        critical=True
    )
    five_claim = (
        "University of California imposes a five-year maximum limit on total postdoctoral service, and prior postdoctoral service at other institutions "
        "counts toward this five-year maximum."
    )
    five_sources = _safe_sources((uc.five_year_maximum_requirement or StatementWithSources()).sources)
    await evaluator.verify(
        claim=five_claim,
        node=five_leaf,
        sources=five_sources,
        additional_instruction="Verify on UC policy sources (e.g., APM-390, campus postdoc appointment rules) that there is a five-year cap and that prior service elsewhere counts."
    )

    # Leaf 3: J-1 health insurance minimums requirement
    j1_leaf = evaluator.add_leaf(
        id="j1_health_insurance_minimums",
        desc="States the required J-1 minimum health insurance coverages: medical benefits ≥ $100,000 per accident/illness, repatriation of remains ≥ $25,000, and medical evacuation ≥ $50,000",
        parent=uc_node,
        critical=True
    )
    j1 = uc.j1_health_insurance_minimums or J1InsuranceMinimums()
    # If amounts are missing in the answer, default to the canonical thresholds specified in the rubric
    med_min = j1.medical_benefits_min or "$100,000"
    rep_min = j1.repatriation_min or "$25,000"
    evac_min = j1.medical_evacuation_min or "$50,000"
    j1_claim = (
        f"For J-1 Exchange Visitors, minimum required health insurance coverages are: medical benefits of at least {med_min} per accident or illness, "
        f"repatriation of remains of at least {rep_min}, and medical evacuation coverage of at least {evac_min}."
    )
    j1_sources = _safe_sources(j1.sources)
    await evaluator.verify(
        claim=j1_claim,
        node=j1_leaf,
        sources=j1_sources,
        additional_instruction="Verify against U.S. Department of State or university international office pages that clearly state the J-1 minimums (medical benefits ≥ $100,000; repatriation ≥ $25,000; medical evacuation ≥ $50,000)."
    )

    return uc_node

async def build_and_verify_nih_ineligibility(
    evaluator: Evaluator,
    parent: VerificationNode,
    extraction: RequirementsExtraction
) -> VerificationNode:
    nih_node = evaluator.add_parallel(
        id="nih_citizenship_ineligibility",
        desc="Identify which NIH mechanism(s) (T32 and/or F32) the researcher is ineligible for based solely on citizenship/permanent residency requirements",
        parent=parent,
        critical=True
    )

    nih = extraction.nih_citizenship_ineligibility or NIHIneligibilityExtraction()

    # Leaf: T32 citizenship requirement and ineligibility for non-US citizens
    t32_leaf = evaluator.add_leaf(
        id="t32_citizenship",
        desc="States that T32 appointees must be U.S. citizens or permanent residents, hence the German citizen is ineligible for T32 on citizenship grounds",
        parent=nih_node,
        critical=True
    )
    t32_claim = (
        "NIH NRSA T32 trainees must be U.S. citizens or permanent residents; therefore a non-U.S. citizen (such as a German citizen) is ineligible for T32 "
        "based solely on citizenship."
    )
    t32_sources = _safe_sources((nih.t32_citizenship or StatementWithSources()).sources)
    await evaluator.verify(
        claim=t32_claim,
        node=t32_leaf,
        sources=t32_sources,
        additional_instruction="Verify the citizenship/permanent residency requirement for NRSA T32 on NIH policy/FOA pages. The logical implication about a non-U.S. citizen being ineligible is acceptable."
    )

    # Leaf: F32 citizenship requirement and ineligibility for non-US citizens
    # NOTE: Parent node is critical; to satisfy framework constraint, this child must be critical as well.
    f32_leaf = evaluator.add_leaf(
        id="f32_citizenship",
        desc="States that F32 awardees/applicants must be U.S. citizens or permanent residents, hence the German citizen is ineligible for F32 on citizenship grounds",
        parent=nih_node,
        critical=True
    )
    f32_claim = (
        "NIH NRSA F32 applicants/awardees must be U.S. citizens or permanent residents; therefore a non-U.S. citizen (such as a German citizen) is ineligible "
        "for F32 based solely on citizenship."
    )
    f32_sources = _safe_sources((nih.f32_citizenship or StatementWithSources()).sources)
    await evaluator.verify(
        claim=f32_claim,
        node=f32_leaf,
        sources=f32_sources,
        additional_instruction="Verify the citizenship/permanent residency requirement for NRSA F32 on NIH policy/FOA pages. The logical implication about a non-U.S. citizen being ineligible is acceptable."
    )

    return nih_node

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
    Evaluate the answer for UC postdoctoral eligibility requirements and NIH citizenship-based ineligibility (T32/F32).
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify UC postdoctoral appointment mandatory eligibility requirements and NIH citizenship-based ineligibility (T32/F32)",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )
    # Adjust root to be critical per rubric
    root.critical = True

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementsExtraction,
        extraction_name="requirements_extraction"
    )

    # Build verification branches
    await build_and_verify_uc_requirements(evaluator, root, extracted)
    await build_and_verify_nih_ineligibility(evaluator, root, extracted)

    # Optionally, record simple custom info for debugging
    evaluator.add_custom_info(
        {
            "doctoral_degree_sources_count": len((extracted.uc_eligibility_requirements or UCEligibilityExtraction()).doctoral_degree_requirement.sources if extracted.uc_eligibility_requirements and extracted.uc_eligibility_requirements.doctoral_degree_requirement and extracted.uc_eligibility_requirements.doctoral_degree_requirement.sources else []),
            "five_year_sources_count": len((extracted.uc_eligibility_requirements or UCEligibilityExtraction()).five_year_maximum_requirement.sources if extracted.uc_eligibility_requirements and extracted.uc_eligibility_requirements.five_year_maximum_requirement and extracted.uc_eligibility_requirements.five_year_maximum_requirement.sources else []),
            "j1_sources_count": len((extracted.uc_eligibility_requirements or UCEligibilityExtraction()).j1_health_insurance_minimums.sources if extracted.uc_eligibility_requirements and extracted.uc_eligibility_requirements.j1_health_insurance_minimums and extracted.uc_eligibility_requirements.j1_health_insurance_minimums.sources else []),
            "t32_sources_count": len((extracted.nih_citizenship_ineligibility or NIHIneligibilityExtraction()).t32_citizenship.sources if extracted.nih_citizenship_ineligibility and extracted.nih_citizenship_ineligibility.t32_citizenship and extracted.nih_citizenship_ineligibility.t32_citizenship.sources else []),
            "f32_sources_count": len((extracted.nih_citizenship_ineligibility or NIHIneligibilityExtraction()).f32_citizenship.sources if extracted.nih_citizenship_ineligibility and extracted.nih_citizenship_ineligibility.f32_citizenship and extracted.nih_citizenship_ineligibility.f32_citizenship.sources else []),
        },
        info_type="extraction_stats"
    )

    # Return structured summary
    return evaluator.get_summary()