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
TASK_ID = "state_grad_requirements_benchmark_exit_exam"
TASK_DESCRIPTION = """
Among the states that still require mandatory high school exit exams for graduation (as of late 2024/early 2025), identify the one state that also requires students to meet specific "remediation-free" benchmark scores on standardized college entrance tests (SAT or ACT) as part of their graduation requirements. Provide the name of this state and its total minimum credit requirement for high school graduation according to NCES data.
"""

# The six states referenced in the rubric that still require mandatory high school exit exams (as of late 2024/early 2025)
EXIT_EXAM_STATES = ["Florida", "Ohio", "Louisiana", "New Jersey", "Texas", "Virginia"]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StatePolicyExtraction(BaseModel):
    """Structured extraction from the agent's answer."""
    state_name: Optional[str] = None
    total_min_credits: Optional[str] = None

    # Source categorization
    exit_exam_sources: List[str] = Field(default_factory=list)
    benchmark_sources: List[str] = Field(default_factory=list)
    nces_sources: List[str] = Field(default_factory=list)
    all_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_state_policy_info() -> str:
    return """
    Extract the following structured information from the answer:

    1) state_name:
       - The single U.S. state the answer identifies as meeting BOTH of the following:
         (a) still requires mandatory high school exit exams for graduation (as of late 2024/early 2025), and
         (b) requires students to meet specific "remediation-free" benchmark scores on standardized college entrance tests (SAT or ACT) as part of its graduation requirements (e.g., as an explicit graduation pathway or competency benchmark).
       - If multiple states are mentioned, choose the one the answer ultimately selects to satisfy BOTH conditions.
       - If no clear single state is identified, return null.

    2) total_min_credits:
       - The total minimum credit requirement for high school graduation for the identified state.
       - Prefer a numeric or string form (e.g., "20" or "20 credits"). If presented as "units" or "Carnegie units", extract as-is (e.g., "22 units").
       - If the answer does not provide this number, return null.

    3) exit_exam_sources:
       - All URLs the answer cites that specifically support the claim that the identified state still requires mandatory high school exit exams for graduation.
       - Include plain URLs or markdown links. Deduplicate exact duplicates.

    4) benchmark_sources:
       - All URLs the answer cites that specifically support the claim that the identified state has "remediation-free" benchmark score requirements on SAT or ACT as part of its graduation requirements.
       - Include plain URLs or markdown links. Deduplicate exact duplicates.

    5) nces_sources:
       - All URLs from NCES (e.g., domains containing "nces.ed.gov" or other official ED/NCES pages) that support or report the state's total minimum credit requirement.
       - If the answer references NCES without a direct URL, return an empty list.

    6) all_sources:
       - A deduplicated list of every URL mentioned anywhere in the answer, regardless of category.

    IMPORTANT:
    - Extract only URLs explicitly present in the answer. Do not invent or infer URLs.
    - If any field is missing, return null (for scalar fields) or an empty list (for URL lists).
    - Normalize markdown links by extracting the actual URL target.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def filter_nces_urls(urls: List[str]) -> List[str]:
    """Filter a list of URLs to those that appear to be from NCES/ED domains."""
    filtered = []
    for u in urls:
        if not isinstance(u, str):
            continue
        lu = u.lower()
        # Basic heuristics to catch NCES / ED
        if ("nces.ed.gov" in lu) or ("ed.gov" in lu) or ("/nces/" in lu):
            filtered.append(u)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in filtered:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def normalize_credit_value(credit_text: Optional[str]) -> str:
    """
    Return a friendly normalized representation of the credit requirement.
    Keeps original text, but trims whitespace for cleaner claim statements.
    """
    if not credit_text:
        return ""
    return credit_text.strip()


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: StatePolicyExtraction,
    parent_node=None
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """

    # Root-level task node (critical, sequential), mirroring rubric's top-level
    task_node = evaluator.add_sequential(
        id="State_Identification_Task",
        desc="Evaluate whether the answer correctly identifies a state and its graduation requirements based on specified education policy criteria",
        parent=parent_node,
        critical=True
    )

    # Add ground truth list for context
    evaluator.add_ground_truth({
        "exit_exam_states": EXIT_EXAM_STATES,
        "policy_window": "late 2024 / early 2025"
    }, gt_type="policy_reference")

    # State policy verification (critical, sequential)
    policy_node = evaluator.add_sequential(
        id="State_Policy_Verification",
        desc="Verify that the identified state meets all specified policy requirements",
        parent=task_node,
        critical=True
    )

    # Existence check for state name (critical)
    state_name_present = evaluator.add_custom_node(
        result=bool(extraction.state_name and extraction.state_name.strip()),
        id="State_Name_Provided",
        desc="The answer provides a specific U.S. state name",
        parent=policy_node,
        critical=True
    )

    # Selection criteria (critical, parallel): exit exam state + benchmark requirement
    selection_node = evaluator.add_parallel(
        id="State_Selection_Criteria",
        desc="Verify the state satisfies both the exit exam and benchmark requirements",
        parent=policy_node,
        critical=True
    )

    # Leaf: Exit exam state membership (critical)
    exit_exam_leaf = evaluator.add_leaf(
        id="Exit_Exam_State",
        desc="The identified state is among the six states (Florida, Ohio, Louisiana, New Jersey, Texas, Virginia) that still require mandatory high school exit exams for graduation as of late 2024/early 2025",
        parent=selection_node,
        critical=True
    )
    state_name = extraction.state_name or ""

    exit_claim = (
        f"The state '{state_name}' is one of the following: "
        f"{', '.join(EXIT_EXAM_STATES)}."
    )
    # Simple membership check; no URLs needed
    await evaluator.verify(
        claim=exit_claim,
        node=exit_exam_leaf,
        sources=None,
        additional_instruction=(
            "This is a simple membership check. Return Correct if the provided state string "
            "exactly matches any of the listed states (case-insensitive and minor spelling variations acceptable); "
            "otherwise return Incorrect. Do not rely on external knowledge; just check membership."
        )
    )

    # Leaf: Benchmark requirement (critical)
    benchmark_leaf = evaluator.add_leaf(
        id="Benchmark_Requirement",
        desc="The identified state requires students to meet specific 'remediation-free' benchmark scores on standardized college entrance tests (SAT or ACT) as part of graduation requirements",
        parent=selection_node,
        critical=True
    )

    # Choose benchmark sources: prefer categorized list, else fall back to all sources
    benchmark_urls = extraction.benchmark_sources if extraction.benchmark_sources else extraction.all_sources
    benchmark_claim = (
        f"The state '{state_name}' includes explicit 'remediation-free' benchmark scores on SAT or ACT "
        f"as an official component of high school graduation requirements (for example, as a defined graduation pathway "
        f"or competency benchmark that satisfies graduation eligibility)."
    )
    await evaluator.verify(
        claim=benchmark_claim,
        node=benchmark_leaf,
        sources=benchmark_urls if benchmark_urls else None,
        additional_instruction=(
            "Verify using the provided webpages whether the state's graduation policy explicitly includes "
            "'remediation-free' ACT/SAT benchmark scores as part of graduation requirements. "
            "It can be a mandatory component or one of recognized graduation pathways/demonstrations that qualifies a student to graduate. "
            "Do not accept pages that only discuss college admissions readiness unrelated to graduation eligibility."
        )
    )

    # Existence check for credits value (critical), placed after selection node due to sequential gating
    credits_provided = evaluator.add_custom_node(
        result=bool(extraction.total_min_credits and extraction.total_min_credits.strip()),
        id="Credit_Value_Provided",
        desc="The answer provides a total minimum credit/units requirement value for the identified state",
        parent=policy_node,
        critical=True
    )

    # Leaf: Credit requirement accuracy per NCES (critical)
    credit_leaf = evaluator.add_leaf(
        id="Credit_Requirement_Accuracy",
        desc="The provided total minimum credit requirement matches the state's actual requirement according to NCES data",
        parent=policy_node,
        critical=True
    )

    # Prefer NCES sources; if absent, try to filter NCES-like URLs from all_sources
    credit_urls = extraction.nces_sources if extraction.nces_sources else filter_nces_urls(extraction.all_sources)
    # If still none, pass all_sources to show lack of NCES (the instruction will require NCES to support)
    if not credit_urls:
        credit_urls = extraction.all_sources

    credit_value = normalize_credit_value(extraction.total_min_credits)
    credit_claim = (
        f"According to NCES (National Center for Education Statistics), the total minimum requirement "
        f"for high school graduation in {state_name} is '{credit_value}'."
    )
    await evaluator.verify(
        claim=credit_claim,
        node=credit_leaf,
        sources=credit_urls if credit_urls else None,
        additional_instruction=(
            "Only accept this claim if the NCES/ED webpage explicitly reports the state's total minimum credits/units. "
            "Treat 'units' as equivalent to 'credits' when clearly used in graduation requirements contexts. "
            "If none of the provided sources are from NCES/ED and the claim cannot be confirmed on an NCES/ED page, return Incorrect."
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
    Evaluate an answer for the state identification and graduation policy task.
    """
    # Initialize evaluator (root as sequential to mirror rubric intent)
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
        default_model=model
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_state_policy_info(),
        template_class=StatePolicyExtraction,
        extraction_name="state_policy_extraction"
    )

    # Record custom info about extraction (for debugging/traceability)
    evaluator.add_custom_info(
        info={
            "extracted_state": extraction.state_name,
            "extracted_total_min_credits": extraction.total_min_credits,
            "exit_exam_sources_count": len(extraction.exit_exam_sources),
            "benchmark_sources_count": len(extraction.benchmark_sources),
            "nces_sources_count": len(extraction.nces_sources),
            "all_sources_count": len(extraction.all_sources),
        },
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extraction, parent_node=root)

    # Return standardized summary
    return evaluator.get_summary()