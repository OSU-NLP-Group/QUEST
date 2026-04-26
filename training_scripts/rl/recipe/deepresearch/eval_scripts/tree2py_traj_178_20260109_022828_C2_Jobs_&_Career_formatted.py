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
TASK_ID = "state_cpa_alt_path_2026"
TASK_DESCRIPTION = (
    "Which U.S. state implemented an alternative pathway to CPA licensure, effective January 1, 2026, that allows "
    "candidates to obtain certification with a bachelor's degree and 2 years of work experience (instead of the "
    "traditional 150 semester hours plus 1 year of experience), while also requiring 24 semester hours of upper-level "
    "accounting courses, 24 semester hours of upper-level business courses, and a state-specific ethics course and exam?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CPAPathwayExtraction(BaseModel):
    """
    Extract the key elements from the answer text:
    - The identified state name
    - The cited source URLs
    - Optional details echoed from the answer (for record)
    """
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

    # Optional textual fields (recorded for analysis; not strictly required for verification)
    effective_date: Optional[str] = None
    alternative_pathway_description: Optional[str] = None
    alternative_degree_requirement: Optional[str] = None  # e.g., "bachelor's degree"
    alternative_experience_requirement: Optional[str] = None  # e.g., "2 years"
    mentions_no_150_hours_in_alternative: Optional[str] = None  # e.g., "does not require 150 hours"
    traditional_pathway_description: Optional[str] = None
    accounting_upper_hours: Optional[str] = None  # often "24"
    business_upper_hours: Optional[str] = None    # often "24"
    ethics_requirement_description: Optional[str] = None  # e.g., "state-specific ethics course and exam"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cpa_pathway() -> str:
    return """
    You must extract the key information from the answer regarding the CPA licensure pathway question.

    Extract the following fields:
    1) state: The single U.S. state explicitly identified by the answer as having the described CPA licensure alternative pathway.
    2) sources: All explicit URLs cited in the answer that support the state's CPA licensure requirements. Extract only valid URLs that appear in the answer text. If none are present, return an empty array.

    Additionally (optional textual echoes from the answer; if absent, return null):
    3) effective_date: The effective date for the alternative pathway as stated in the answer (e.g., 'January 1, 2026').
    4) alternative_pathway_description: The answer's description of the alternative pathway.
    5) alternative_degree_requirement: The degree requirement for the alternative pathway (e.g., 'bachelor's degree').
    6) alternative_experience_requirement: The experience requirement for the alternative pathway (e.g., '2 years').
    7) mentions_no_150_hours_in_alternative: Text (if any) indicating that the alternative pathway does not require 150 semester hours.
    8) traditional_pathway_description: The answer's description of the traditional 150-hours + 1-year pathway.
    9) accounting_upper_hours: The number of required upper-level accounting semester hours (if the answer mentions it).
    10) business_upper_hours: The number of required upper-level business semester hours (if the answer mentions it).
    11) ethics_requirement_description: The answer's description of the state-specific ethics course and exam requirement.

    Rules:
    - Do not invent URLs. Extract only URLs that appear in the answer. Accept plain URLs or markdown links.
    - For any missing field, return null (or empty array for sources).
    """


# --------------------------------------------------------------------------- #
# Helper to build claims                                                      #
# --------------------------------------------------------------------------- #
def _fmt_state(state: Optional[str]) -> str:
    return state if (state and state.strip()) else "the state in question"


def build_claim_effective_date(state: Optional[str]) -> str:
    s = _fmt_state(state)
    return f"In {s}, the alternative pathway to CPA licensure became effective on January 1, 2026."

def build_claim_traditional_pathway(state: Optional[str]) -> str:
    s = _fmt_state(state)
    return f"{s} continues to offer the traditional CPA licensure pathway requiring 150 semester hours of education plus 1 year of work experience."

def build_claim_alternative_pathway_structure(state: Optional[str]) -> str:
    s = _fmt_state(state)
    return (
        f"In {s}, the alternative pathway allows CPA licensure with a bachelor's degree and exactly 2 years of relevant "
        f"work experience, and it does not require completing 150 semester hours."
    )

def build_claim_accounting_hours(state: Optional[str]) -> str:
    s = _fmt_state(state)
    return f"{s}'s CPA licensure requirements include 24 semester hours of upper-level (upper-division) accounting courses."

def build_claim_business_hours(state: Optional[str]) -> str:
    s = _fmt_state(state)
    return f"{s}'s CPA licensure requirements include 24 semester hours of upper-level (upper-division) business courses."

def build_claim_state_specific_ethics(state: Optional[str]) -> str:
    s = _fmt_state(state)
    return (
        f"{s} requires a state-specific ethics course and exam (distinct from the generic AICPA ethics exam)."
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
    Evaluate an answer for the CPA alternative pathway identification task.

    The rubric tree is implemented under a main critical node "State_CPA_Requirements_Identification"
    with the following critical sub-criteria:
      - Alternative_Pathway_Implementation (parallel): Effective_Date, Traditional_Pathway_Structure, Alternative_Pathway_Structure
      - Education_Credit_Requirements (parallel): Accounting_Credit_Hours, Business_Credit_Hours
      - State_Specific_Ethics (leaf)

    Additionally, we add two critical gating checks under the main node:
      - State_Identified (answer names a state)
      - Sources_Provided (answer provides at least one URL)
    These gating checks ensure meaningful downstream verification.
    """
    # 1) Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall aggregation at root (non-critical root by framework)
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

    # 2) Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_cpa_pathway(),
        template_class=CPAPathwayExtraction,
        extraction_name="cpa_pathway_extraction"
    )

    # 3) Build rubric tree under a critical main node (matching JSON structure)
    main_node = evaluator.add_parallel(
        id="State_CPA_Requirements_Identification",
        desc="Correctly identify the U.S. state that meets all specified CPA licensure requirements",
        parent=root,
        critical=True
    )

    # 3.a) Critical gating: state identified and sources provided
    state_present = bool(extracted.state and extracted.state.strip())
    evaluator.add_custom_node(
        result=state_present,
        id="State_Identified",
        desc="State name is identified in the answer",
        parent=main_node,
        critical=True
    )

    sources_present = bool(extracted.sources and len(extracted.sources) > 0)
    evaluator.add_custom_node(
        result=sources_present,
        id="Sources_Provided",
        desc="At least one supporting source URL is provided in the answer",
        parent=main_node,
        critical=True
    )

    # 3.b) Alternative Pathway Implementation (critical, parallel)
    alt_impl_node = evaluator.add_parallel(
        id="Alternative_Pathway_Implementation",
        desc="Verify the state implemented an alternative CPA licensure pathway with the specified effective date and pathway structures",
        parent=main_node,
        critical=True
    )

    # Effective Date leaf
    effective_date_leaf = evaluator.add_leaf(
        id="Effective_Date",
        desc="The alternative pathway became effective on January 1, 2026",
        parent=alt_impl_node,
        critical=True
    )
    await evaluator.verify(
        claim=build_claim_effective_date(extracted.state),
        node=effective_date_leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Confirm that the page(s) clearly indicate an alternative CPA licensure pathway effective date of "
            "January 1, 2026 for the specified state. If the date differs or is missing, mark as not supported."
        )
    )

    # Traditional Pathway Structure leaf
    traditional_leaf = evaluator.add_leaf(
        id="Traditional_Pathway_Structure",
        desc="The state still offers the traditional pathway requiring 150 semester hours of education plus 1 year of work experience",
        parent=alt_impl_node,
        critical=True
    )
    await evaluator.verify(
        claim=build_claim_traditional_pathway(extracted.state),
        node=traditional_leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Verify that the state's CPA licensure framework continues to include the traditional pathway requiring "
            "150 semester hours plus 1 year of experience. Equivalent phrasing is acceptable."
        )
    )

    # Alternative Pathway Structure leaf
    alternative_leaf = evaluator.add_leaf(
        id="Alternative_Pathway_Structure",
        desc="The alternative pathway allows CPA licensure with a bachelor's degree (not requiring 150 semester hours) plus exactly 2 years of work experience",
        parent=alt_impl_node,
        critical=True
    )
    await evaluator.verify(
        claim=build_claim_alternative_pathway_structure(extracted.state),
        node=alternative_leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Check that the alternative pathway explicitly allows licensure with a bachelor's degree and exactly 2 years "
            "of relevant work experience, and that it does not require 150 semester hours. Synonyms for 'bachelor’s degree' "
            "are acceptable. If the experience period is not 2 years, mark as not supported."
        )
    )

    # 3.c) Education Credit Requirements (critical, parallel)
    edu_node = evaluator.add_parallel(
        id="Education_Credit_Requirements",
        desc="Verify specific upper-level credit hour requirements for accounting and business courses",
        parent=main_node,
        critical=True
    )

    accounting_leaf = evaluator.add_leaf(
        id="Accounting_Credit_Hours",
        desc="Requires 24 semester hours of upper-level accounting courses",
        parent=edu_node,
        critical=True
    )
    await evaluator.verify(
        claim=build_claim_accounting_hours(extracted.state),
        node=accounting_leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Confirm that the requirements include 24 semester hours in upper-level (upper-division) accounting courses. "
            "Accept 'upper-division' as equivalent to 'upper-level'."
        )
    )

    business_leaf = evaluator.add_leaf(
        id="Business_Credit_Hours",
        desc="Requires 24 semester hours of upper-level business courses",
        parent=edu_node,
        critical=True
    )
    await evaluator.verify(
        claim=build_claim_business_hours(extracted.state),
        node=business_leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Confirm that the requirements include 24 semester hours in upper-level (upper-division) business courses. "
            "Synonyms such as 'non-accounting business courses' are acceptable if clearly equivalent."
        )
    )

    # 3.d) State-specific Ethics (critical leaf)
    ethics_leaf = evaluator.add_leaf(
        id="State_Specific_Ethics",
        desc="The state requires a state-specific ethics course and exam (not just the AICPA ethics exam)",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=build_claim_state_specific_ethics(extracted.state),
        node=ethics_leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Verify that the requirement is a state-specific ethics course and exam, distinct from only the AICPA ethics exam. "
            "If the page indicates only the generic AICPA ethics exam with no state-specific component, mark as not supported."
        )
    )

    # 4) Return the structured evaluation summary
    return evaluator.get_summary()