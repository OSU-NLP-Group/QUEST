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
TASK_ID = "la_adaptive_reuse_2026_assessment"
TASK_DESCRIPTION = (
    "A property owner in Los Angeles has a 20-year-old office building and is interested in converting it to "
    "residential apartments under the city's 2026 Citywide Adaptive Reuse Ordinance. Provide a comprehensive assessment "
    "that addresses: (1) whether this building qualifies for conversion under the ordinance, (2) what the key eligibility "
    "requirements are, (3) what the approval process entails, and (4) any important specifications or contextual information "
    "about the program."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class AspectEvidence(BaseModel):
    """
    Generic structure capturing what the answer claims (verbatim or close paraphrase)
    for a particular aspect, along with exactly the URLs cited in the answer for that claim.
    """
    claim_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AssessmentExtraction(BaseModel):
    """
    Extract key elements from the answer for verification.
    All fields should be populated ONLY with content explicitly present in the answer text.
    If the answer does not mention an aspect, set the claim_text to null and sources to [].
    """
    # 1) Qualification determination (explicit yes/no)
    qualification_determination: Optional[str] = None  # one of: "qualifies", "does_not_qualify"
    qualification_statement: Optional[str] = None      # exact sentence/phrase indicating the determination

    # 2) Key eligibility requirements
    req_15yr: Optional[AspectEvidence] = None                   # 2a
    citywide_scope: Optional[AspectEvidence] = None             # 2b
    eligible_building_types: Optional[AspectEvidence] = None    # 2c
    rolling_age_threshold: Optional[AspectEvidence] = None      # 2d

    # 3) Approval process
    staff_level_approval: Optional[AspectEvidence] = None       # 3a
    no_environmental_review: Optional[AspectEvidence] = None    # 3b

    # 4) Program specifications and context
    parking_minimum: Optional[AspectEvidence] = None            # 4a
    no_min_unit_size: Optional[AspectEvidence] = None           # 4b
    effective_date: Optional[AspectEvidence] = None             # 4c
    empty_office_space: Optional[AspectEvidence] = None         # 4d
    vacancy_rate_q4_2025: Optional[AspectEvidence] = None       # 4e
    historical_comparison_1999: Optional[AspectEvidence] = None # 4f


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_assessment() -> str:
    return """
    You will extract structured information from the answer text about Los Angeles’ 2026 Citywide Adaptive Reuse Ordinance.

    IMPORTANT GENERAL RULES:
    - Do not invent or infer any content not explicitly present in the answer text.
    - For every 'sources' array, include ONLY URLs that appear in the answer text. If the answer gives no URLs for that aspect, return an empty array.
    - For each 'claim_text', copy the exact statement from the answer when possible. If the answer paraphrases, capture the closest faithful paraphrase that cleanly expresses the proposition.
    - If an aspect is not addressed in the answer, set its claim_text to null and sources to [].
    - Do not rely on your own knowledge or the web; only extract from the answer.

    FIELDS TO EXTRACT:

    1) qualification_determination (string|null):
       - One of: "qualifies", "does_not_qualify".
       - Return null if the answer does not clearly provide a yes/no determination.

    2) qualification_statement (string|null):
       - The exact sentence/phrase that explicitly states the determination.
       - Return null if not present.

    3) req_15yr (object|null):
       - claim_text: A sentence in the answer that states the age requirement (e.g., "At least 15 years old ... at time of permit ...").
       - sources: URLs cited in the answer that support this claim.

    4) citywide_scope (object|null):
       - claim_text: A sentence that the ordinance applies citywide across Los Angeles (or a contrary claim if that is what the answer states).
       - sources: URLs from the answer for that claim.

    5) eligible_building_types (object|null):
       - claim_text: A sentence listing which building types are eligible (the answer may list office, industrial, retail, parking garages, etc.).
       - sources: URLs from the answer for that claim.

    6) rolling_age_threshold (object|null):
       - claim_text: A sentence that explains the 15-year threshold is rolling and new buildings become eligible each year (or contrary claim if stated).
       - sources: URLs from the answer for that claim.

    7) staff_level_approval (object|null):
       - claim_text: A sentence that qualifying projects receive staff/ministerial approval without lengthy discretionary review.
       - sources: URLs from the answer for that claim.

    8) no_environmental_review (object|null):
       - claim_text: A sentence that qualifying projects do not require environmental impact review/EIR.
       - sources: URLs from the answer for that claim.

    9) parking_minimum (object|null):
       - claim_text: A sentence that specifies the minimum parking requirement (e.g., "one space per 200 sq ft") if the answer claims such a requirement.
       - sources: URLs from the answer for that claim.

    10) no_min_unit_size (object|null):
        - claim_text: A sentence that the 2026 ordinance removes minimum unit size requirements.
        - sources: URLs from the answer for that claim.

    11) effective_date (object|null):
        - claim_text: A sentence that the ordinance took effect in February 2026 (if stated).
        - sources: URLs from the answer for that claim.

    12) empty_office_space (object|null):
        - claim_text: A sentence providing the contextual fact that Los Angeles has more than 50 million sq ft of empty office space as of 2026 (if stated).
        - sources: URLs from the answer for that claim.

    13) vacancy_rate_q4_2025 (object|null):
        - claim_text: A sentence that the Los Angeles office market vacancy rate was 23.4% in Q4 2025 (if stated).
        - sources: URLs from the answer for that claim.

    14) historical_comparison_1999 (object|null):
        - claim_text: A sentence that the 1999 adaptive reuse ordinance primarily applied to buildings erected before 1975 and focused mainly on Downtown Los Angeles, and that the 2026 ordinance is broader (if the answer mentions this).
        - sources: URLs from the answer for that claim.

    Return a single JSON object that conforms to the provided schema exactly.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_presence_of_clear_determination(evaluator: Evaluator, parent_node) -> None:
    """
    Leaf for: 1_Qualification_Determination_For_This_Building
    We judge only whether the answer explicitly provides a clear yes/no (qualifies/does not qualify).
    This is an intra-answer check (no external sources).
    """
    node = evaluator.add_leaf(
        id="1_Qualification_Determination_For_This_Building",
        desc="Explicitly states whether the described 20-year-old office building in Los Angeles qualifies for conversion under the ordinance (clear yes/no or qualifies/does not qualify).",
        parent=parent_node,
        critical=True
    )
    claim = (
        "The answer explicitly provides a clear yes/no determination about whether the described 20-year-old "
        "Los Angeles office building qualifies for conversion under the 2026 Citywide Adaptive Reuse Ordinance."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=None,
        additional_instruction=(
            "Judge solely by the answer text: It must clearly say 'qualifies' or 'does not qualify', or an "
            "equivalent unambiguous yes/no determination."
        )
    )


async def verify_aspect_leaf(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    aspect: Optional[AspectEvidence],
    additional_instruction: str
) -> None:
    """
    Generic verifier for rubric leaves backed by web sources.
    Enforces source-grounding: if no claim_text or no sources are provided in the answer, mark failed.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=True
    )

    # Enforce presence of a claim in the answer
    if aspect is None or not aspect.claim_text or not aspect.claim_text.strip():
        leaf.score = 0.0
        leaf.status = "failed"
        return

    # Enforce presence of URLs cited in the answer (source-grounding)
    sources = aspect.sources if aspect.sources else []
    if len(sources) == 0:
        leaf.score = 0.0
        leaf.status = "failed"
        return

    # Verify the claim against the cited URLs
    await evaluator.verify(
        claim=aspect.claim_text.strip(),
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_eligibility_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: AssessmentExtraction
) -> None:
    """
    Node: 2_Key_Eligibility_Requirements (parallel, critical)
    Children:
      2a_Building_Age_Requirement_15_Years
      2b_Geographic_Scope_Citywide
      2c_Eligible_Building_Types_List
      2d_Rolling_Age_Threshold
    """
    node = evaluator.add_parallel(
        id="2_Key_Eligibility_Requirements",
        desc="States the ordinance’s key eligibility requirements as listed in the constraints.",
        parent=parent_node,
        critical=True
    )

    await verify_aspect_leaf(
        evaluator,
        node,
        "2a_Building_Age_Requirement_15_Years",
        "States that commercial buildings must be at least 15 years old at the time of permit application to qualify for conversion to housing.",
        extracted.req_15yr,
        additional_instruction="Verify the numeric threshold (15 years) and the timing ('at time of permit application') as stated. Minor wording differences are fine; numbers and meaning must match."
    )

    await verify_aspect_leaf(
        evaluator,
        node,
        "2b_Geographic_Scope_Citywide",
        "Confirms that the ordinance applies citywide throughout Los Angeles (not limited to specific districts).",
        extracted.citywide_scope,
        additional_instruction="Confirm that the ordinance applies citywide across the City of Los Angeles, not limited to Downtown or special districts."
    )

    await verify_aspect_leaf(
        evaluator,
        node,
        "2c_Eligible_Building_Types_List",
        "States that office buildings, industrial buildings, retail stores, and parking garages are eligible for conversion to residential housing under the ordinance.",
        extracted.eligible_building_types,
        additional_instruction="Check that the page supports the specific eligible building types listed in the claim (office, industrial, retail, parking garages) for conversion to residential housing."
    )

    await verify_aspect_leaf(
        evaluator,
        node,
        "2d_Rolling_Age_Threshold",
        "Explains that the 15-year age threshold is rolling (i.e., new buildings become eligible each year).",
        extracted.rolling_age_threshold,
        additional_instruction="Verify that the requirement is described as a rolling 15-year threshold such that additional buildings qualify as they age."
    )


async def build_approval_process(
    evaluator: Evaluator,
    parent_node,
    extracted: AssessmentExtraction
) -> None:
    """
    Node: 3_Approval_Process (parallel, critical)
    Children:
      3a_Approval_Process_Staff_Level_No_Lengthy_Discretionary_Review
      3b_No_Environmental_Impact_Review
    """
    node = evaluator.add_parallel(
        id="3_Approval_Process",
        desc="Explains what the approval process entails under the ordinance, per constraints.",
        parent=parent_node,
        critical=True
    )

    await verify_aspect_leaf(
        evaluator,
        node,
        "3a_Approval_Process_Staff_Level_No_Lengthy_Discretionary_Review",
        "Explains that qualifying projects receive approval from city staff rather than requiring lengthy discretionary review processes that may reach City Council.",
        extracted.staff_level_approval,
        additional_instruction="Verify that approvals are ministerial/staff-level and do not require lengthy discretionary processes or City Council hearings when qualifying."
    )

    await verify_aspect_leaf(
        evaluator,
        node,
        "3b_No_Environmental_Impact_Review",
        "States that qualifying conversion projects under the ordinance do not require environmental impact reviews.",
        extracted.no_environmental_review,
        additional_instruction="Verify that qualifying conversion projects are exempt from environmental impact reports/reviews (e.g., EIR) as claimed."
    )


async def build_program_specs_context(
    evaluator: Evaluator,
    parent_node,
    extracted: AssessmentExtraction
) -> None:
    """
    Node: 4_Program_Specifications_And_Context (parallel, critical)
    Children:
      4a_Parking_Minimum_One_Per_200_SqFt
      4b_No_Minimum_Unit_Size_Requirements
      4c_Effective_Date_February_2026
      4d_Empty_Office_Space_More_Than_50_Million_SqFt
      4e_Vacancy_Rate_23_4_Percent_Q4_2025
      4f_Historical_Comparison_1999_Ordinance_Scope
    """
    node = evaluator.add_parallel(
        id="4_Program_Specifications_And_Context",
        desc="Provides important specifications and contextual information about the program as listed in the constraints.",
        parent=parent_node,
        critical=True
    )

    await verify_aspect_leaf(
        evaluator,
        node,
        "4a_Parking_Minimum_One_Per_200_SqFt",
        "Specifies the minimum parking requirement of one parking space for each 200 square feet of floor area for commercial/industrial conversion projects in Los Angeles.",
        extracted.parking_minimum,
        additional_instruction="Verify the exact ratio and applicability. If sources indicate reductions/waivers or a different metric, mark as not supported."
    )

    await verify_aspect_leaf(
        evaluator,
        node,
        "4b_No_Minimum_Unit_Size_Requirements",
        "Notes that the 2026 ordinance removes minimum unit size requirements, allowing flexibility in creating residential units.",
        extracted.no_min_unit_size,
        additional_instruction="Verify that the 2026 ordinance removes minimum unit size requirements or otherwise allows flexibility in minimum unit sizes."
    )

    await verify_aspect_leaf(
        evaluator,
        node,
        "4c_Effective_Date_February_2026",
        "States that the Citywide Adaptive Reuse Ordinance went into effect in February 2026.",
        extracted.effective_date,
        additional_instruction="Verify the effective date as February 2026."
    )

    await verify_aspect_leaf(
        evaluator,
        node,
        "4d_Empty_Office_Space_More_Than_50_Million_SqFt",
        "Provides the contextual fact that Los Angeles has more than 50 million square feet of empty office space as of 2026.",
        extracted.empty_office_space,
        additional_instruction="Verify the magnitude ('more than 50 million square feet') and timing (as of 2026) in the cited source."
    )

    await verify_aspect_leaf(
        evaluator,
        node,
        "4e_Vacancy_Rate_23_4_Percent_Q4_2025",
        "Provides the contextual fact that the Los Angeles office market vacancy rate was 23.4% in Q4 2025.",
        extracted.vacancy_rate_q4_2025,
        additional_instruction="Verify the vacancy rate value (23.4%) and the time period (Q4 2025). Allow minor rounding if clearly equivalent."
    )

    await verify_aspect_leaf(
        evaluator,
        node,
        "4f_Historical_Comparison_1999_Ordinance_Scope",
        "Includes a historical comparison noting the 1999 adaptive reuse ordinance was primarily for buildings erected before 1975 and focused mainly on downtown Los Angeles, and characterizes the 2026 ordinance as a broader expansion (without re-stating already-checked 2026 age/citywide specifics).",
        extracted.historical_comparison_1999,
        additional_instruction="Verify that the 1999 ARO primarily covered pre-1975 buildings and focused on Downtown LA, and that the 2026 ordinance is broader in scope."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Entry point for evaluating an answer against the rubric for LA's 2026 Citywide Adaptive Reuse Ordinance assessment.
    """
    # Initialize evaluator/root
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

    # Create the rubric's top-level node (critical, parallel)
    comp_node = evaluator.add_parallel(
        id="Comprehensive_Office_Building_Conversion_Assessment",
        desc=("Response assesses whether the described 20-year-old Los Angeles office building qualifies for conversion "
              "under the 2026 Citywide Adaptive Reuse Ordinance and covers the eligibility requirements, approval "
              "process, and key program specifications/context listed in the constraints."),
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extracted: AssessmentExtraction = await evaluator.extract(
        prompt=prompt_extract_assessment(),
        template_class=AssessmentExtraction,
        extraction_name="assessment_extraction"
    )

    # Add optional GT/context info for transparency (not used for scoring)
    evaluator.add_custom_info(
        info={
            "evaluation_focus": "LA 2026 Citywide Adaptive Reuse Ordinance",
            "building_context": "20-year-old office building in Los Angeles",
        },
        info_type="context",
        info_name="task_context"
    )

    # 1) Qualification determination (leaf)
    await verify_presence_of_clear_determination(evaluator, comp_node)

    # 2) Eligibility requirements (subtree)
    await build_eligibility_requirements(evaluator, comp_node, extracted)

    # 3) Approval process (subtree)
    await build_approval_process(evaluator, comp_node, extracted)

    # 4) Program specs and context (subtree)
    await build_program_specs_context(evaluator, comp_node, extracted)

    # Return the final summary
    return evaluator.get_summary()