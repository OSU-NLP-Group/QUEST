import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ncaa_division1_coaching_pathway"
TASK_DESCRIPTION = (
    "What are the complete educational, experiential, and certification requirements for the career pathway from "
    "entry-level graduate assistant to assistant football coach at an NCAA Division I institution? Your answer should "
    "include: (1) the minimum educational requirement to enter college football coaching, (2) the specific requirements "
    "for graduate assistant positions, (3) the minimum qualifications for assistant coach positions including typical "
    "experience requirements, (4) the mandatory certifications required upon hire, and (5) the essential knowledge and "
    "skill competencies required for these positions."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ClaimInfo(BaseModel):
    """
    A generic structure for extracting a claim the answer asserts
    and any URLs (sources) the answer cites for that specific claim.
    """
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CareerPathwayExtraction(BaseModel):
    """
    Flat mapping of all rubric claims to extracted statements and sources from the answer.
    For each field, `statement` should be the verbatim or concise paraphrase of the answer's assertion,
    and `sources` should include only URLs explicitly present in the answer (if any).
    """
    # Minimum education
    min_education_bachelors: Optional[ClaimInfo] = None

    # Graduate Assistant (GA) requirements
    ga_active_grad_enrollment: Optional[ClaimInfo] = None
    ga_prior_college_experience: Optional[ClaimInfo] = None

    # Assistant Coach (AC) qualifications
    ac_sports_background_experience: Optional[ClaimInfo] = None
    ac_typical_coaching_experience_1_3_years: Optional[ClaimInfo] = None

    # Mandatory certifications upon hire
    cert_ncaa_coaches_certification: Optional[ClaimInfo] = None
    cert_15_passenger_van_driving: Optional[ClaimInfo] = None

    # Essential knowledge and skill competencies
    comp_ncaa_rules_compliance_knowledge: Optional[ClaimInfo] = None
    comp_sport_specific_knowledge_teaching: Optional[ClaimInfo] = None
    comp_recruiting_capability: Optional[ClaimInfo] = None
    comp_academic_monitoring_ability: Optional[ClaimInfo] = None
    comp_computer_literacy: Optional[ClaimInfo] = None
    comp_communication_skills: Optional[ClaimInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_career_pathway() -> str:
    return """
    Extract from the provided answer the explicit statements (verbatim or concise paraphrase) and any cited URL sources 
    corresponding to each of the following items. For each item, return:
      - statement: The explicit assertion the answer makes about that item (use concise paraphrase if needed). If the 
        answer does not contain the assertion, return null.
      - sources: A list of the URLs explicitly mentioned in the answer that support this specific item. If none are 
        cited, return an empty list.

    The items to extract (ensure the JSON fields match these names exactly):
      1. min_education_bachelors: "Bachelor’s (baccalaureate) degree is the minimum educational requirement to enter 
         college football coaching (including eligibility for GA and assistant coach roles)."
      2. ga_active_grad_enrollment: "Graduate assistant (GA) positions require active enrollment in a graduate degree program."
      3. ga_prior_college_experience: "GA positions require prior playing or coaching experience at the college football level."
      4. ac_sports_background_experience: "Assistant coach positions require a sports background and experience."
      5. ac_typical_coaching_experience_1_3_years: "Assistant coach positions typically require 1–3 years of prior coaching 
         experience at the collegiate or professional level."
      6. cert_ncaa_coaches_certification: "Upon hire, coaches must obtain NCAA Coaches Certification."
      7. cert_15_passenger_van_driving: "Upon hire, coaches must obtain 15-passenger van driving certification."
      8. comp_ncaa_rules_compliance_knowledge: "Coaches must have strong working knowledge of NCAA rules and regulations regarding compliance."
      9. comp_sport_specific_knowledge_teaching: "Coaches must have extensive sport-specific knowledge and the ability to teach technical skills."
     10. comp_recruiting_capability: "Coaches must have recruiting capability to evaluate and compare athletic talent."
     11. comp_academic_monitoring_ability: "Coaches must have academic monitoring ability."
     12. comp_computer_literacy: "Coaches must have computer literacy."
     13. comp_communication_skills: "Coaches must have high-level communication skills."

    URL source extraction rules:
      - Only include URLs explicitly present in the answer (plain URLs or markdown links).
      - If a URL is missing a protocol, prepend http://.
      - If the answer references a site without a concrete URL, do not invent the URL; return an empty list for sources.

    Return a single JSON object matching the CareerPathwayExtraction schema.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _presence_instruction_general() -> str:
    """
    General additional instruction for presence checks.
    """
    return (
        "Judge whether the answer explicitly asserts the given requirement. Allow reasonable synonymy and phrasing "
        "(e.g., 'baccalaureate' vs. 'bachelor’s', 'BA' or 'BS' as bachelor's degree; 'enrolled' can be 'actively enrolled' or "
        "'matriculated'; '1–3 years' may be written as 'one to three years' or '1-3 years'). Focus on whether the "
        "answer states the requirement."
    )


def _presence_instruction_competencies() -> str:
    return (
        "Judge whether the answer explicitly lists or asserts the competency. Accept close paraphrases (e.g., "
        "strong working knowledge of NCAA rules/compliance; sport-specific knowledge with teaching ability; "
        "recruiting capability for talent evaluation; academic monitoring; computer literacy; high-level communication skills)."
    )


async def _add_and_verify_leaf(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    add_ins: str,
) -> None:
    """
    Create a critical leaf node under parent_node and verify the claim presence in the answer.
    node_desc is reused as the claim text for verification.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=True,
    )
    await evaluator.verify(
        claim=node_desc,
        node=leaf,
        additional_instruction=add_ins,
    )


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_career_pathway(
    evaluator: Evaluator,
    root_node,
    extracted: CareerPathwayExtraction,
) -> None:
    """
    Constructs the verification tree based on the rubric and verifies each leaf.
    All nodes are critical per rubric; parent critical nodes enforce pass/fail without partial credit.
    """

    # Top-level critical node aggregating all requirements in parallel
    cp_node = evaluator.add_parallel(
        id="Career_Pathway_Information",
        desc="Covers all educational, experiential, certification, and competency requirements for the pathway from entry-level graduate assistant to assistant football coach at an NCAA Division I institution, per the provided constraints.",
        parent=root_node,
        critical=True,
    )

    # 1) Minimum educational requirement (leaf)
    await _add_and_verify_leaf(
        evaluator,
        cp_node,
        "Minimum_Educational_Requirement",
        "States that completion of a bachelor’s (baccalaureate) degree is the minimum educational requirement to enter the college football coaching pathway (including eligibility for GA roles and assistant coach roles).",
        "The answer should explicitly assert that a bachelor’s degree (BA/BS/baccalaureate) is the minimum requirement to enter college football coaching and to be eligible for GA/assistant coach roles. Allow minor paraphrase variations.",
    )

    # 2) Graduate Assistant Position Requirements (parallel)
    ga_node = evaluator.add_parallel(
        id="Graduate_Assistant_Position_Requirements",
        desc="States the specific requirements for graduate assistant (GA) positions beyond the minimum education requirement.",
        parent=cp_node,
        critical=True,
    )
    await _add_and_verify_leaf(
        evaluator,
        ga_node,
        "GA_Active_Grad_Enrollment",
        "GA positions require active enrollment in a graduate degree program.",
        _presence_instruction_general(),
    )
    await _add_and_verify_leaf(
        evaluator,
        ga_node,
        "GA_Prior_College_Football_Playing_or_Coaching_Experience",
        "GA positions require prior playing or coaching experience at the college football level.",
        _presence_instruction_general(),
    )

    # 3) Assistant Coach Position Qualifications (parallel)
    ac_node = evaluator.add_parallel(
        id="Assistant_Coach_Position_Qualifications",
        desc="States the minimum qualifications for assistant coach positions, including typical experience requirements (beyond the minimum education requirement).",
        parent=cp_node,
        critical=True,
    )
    await _add_and_verify_leaf(
        evaluator,
        ac_node,
        "AC_Sports_Background_and_Experience",
        "Assistant coach positions require a sports background and experience.",
        _presence_instruction_general(),
    )
    await _add_and_verify_leaf(
        evaluator,
        ac_node,
        "AC_Typical_Coaching_Experience_1_to_3_Years",
        "Assistant coach positions typically require 1–3 years of prior coaching experience at the collegiate or professional level.",
        "Accept '1–3 years', 'one to three years', or '1-3 years' phrasing. The answer should explicitly assert this typical experience range for assistant coach roles.",
    )

    # 4) Mandatory Certifications Upon Hire (parallel)
    certs_node = evaluator.add_parallel(
        id="Mandatory_Certifications_Upon_Hire",
        desc="Identifies mandatory certifications required upon hire.",
        parent=cp_node,
        critical=True,
    )
    await _add_and_verify_leaf(
        evaluator,
        certs_node,
        "NCAA_Coaches_Certification",
        "Upon hire, coaches must obtain NCAA Coaches Certification.",
        _presence_instruction_general(),
    )
    await _add_and_verify_leaf(
        evaluator,
        certs_node,
        "Van_Driving_Certification",
        "Upon hire, coaches must obtain 15-passenger van driving certification.",
        _presence_instruction_general(),
    )

    # 5) Essential Knowledge and Skill Competencies (parallel)
    comp_node = evaluator.add_parallel(
        id="Essential_Knowledge_and_Skill_Competencies",
        desc="Identifies essential knowledge and skill competencies required for these positions.",
        parent=cp_node,
        critical=True,
    )
    await _add_and_verify_leaf(
        evaluator,
        comp_node,
        "NCAA_Rules_Compliance_Knowledge",
        "Coaches must have strong working knowledge of NCAA rules and regulations regarding compliance.",
        _presence_instruction_competencies(),
    )
    await _add_and_verify_leaf(
        evaluator,
        comp_node,
        "Sport_Specific_Knowledge_and_Teaching",
        "Coaches must have extensive sport-specific knowledge with ability to teach technical skills.",
        _presence_instruction_competencies(),
    )
    await _add_and_verify_leaf(
        evaluator,
        comp_node,
        "Recruiting_Capability",
        "Coaches must have recruiting capability to evaluate and compare athletic talent.",
        _presence_instruction_competencies(),
    )
    await _add_and_verify_leaf(
        evaluator,
        comp_node,
        "Academic_Monitoring_Ability",
        "Coaches must have academic monitoring ability.",
        _presence_instruction_competencies(),
    )
    await _add_and_verify_leaf(
        evaluator,
        comp_node,
        "Computer_Literacy",
        "Coaches must have computer literacy.",
        _presence_instruction_competencies(),
    )
    await _add_and_verify_leaf(
        evaluator,
        comp_node,
        "Communication_Skills",
        "Coaches must have high-level communication skills.",
        _presence_instruction_competencies(),
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
    Evaluate an answer for the NCAA Division I coaching career pathway requirements task.
    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator (root is non-critical by framework; we'll add a critical top-level node under it)
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

    # Extraction (record structured info; mostly used for transparency in eval_breakdown)
    extraction = await evaluator.extract(
        prompt=prompt_extract_career_pathway(),
        template_class=CareerPathwayExtraction,
        extraction_name="career_pathway_extraction",
    )

    # Optional: record expected rubric items as ground truth descriptors for clarity
    evaluator.add_ground_truth({
        "expected_items": [
            "Bachelor’s degree is minimum education to enter college coaching (incl. GA and assistant roles).",
            "GA requires active graduate enrollment.",
            "GA requires prior college playing/coaching experience.",
            "Assistant coach requires sports background and experience.",
            "Assistant coach typically requires 1–3 years prior collegiate/professional coaching.",
            "NCAA Coaches Certification upon hire.",
            "15-passenger van driving certification upon hire.",
            "Competencies: NCAA rules/compliance knowledge.",
            "Competencies: sport-specific knowledge and technical teaching.",
            "Competencies: recruiting capability (evaluate talent).",
            "Competencies: academic monitoring ability.",
            "Competencies: computer literacy.",
            "Competencies: high-level communication skills.",
        ]
    }, gt_type="rubric_expectations")

    # Build tree and run verifications
    await build_and_verify_career_pathway(evaluator, root, extraction)

    # Return structured result
    return evaluator.get_summary()