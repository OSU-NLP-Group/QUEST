import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "multi_state_llc_requirements_2025_2026"
TASK_DESCRIPTION = (
    "You are considering forming a Limited Liability Company (LLC) and want to compare the formation requirements "
    "and ongoing compliance obligations in California, Pennsylvania, New York, and Nevada. For each of these four "
    "states, provide the following information, sourced from official state government websites: "
    "(1) The current initial LLC filing fee, "
    "(2) Whether annual reports are required, when they are due, and the associated filing fee, "
    "(3) Whether a registered agent is required and whether the registered agent must have a physical address in the state, "
    "(4) The standard processing time for LLC formation applications, and "
    "(5) A direct URL to the official state government source (Secretary of State or equivalent agency) where this information can be verified. "
    "Present findings in a structured format that allows for easy comparison across all four states."
)


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class StateData(BaseModel):
    initial_filing_fee: Optional[str] = None
    annual_report_required: Optional[str] = None
    annual_report_due: Optional[str] = None
    annual_report_fee: Optional[str] = None
    registered_agent_required: Optional[str] = None
    registered_agent_physical_instate_required: Optional[str] = None
    processing_time: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)


class LLCComparisonExtraction(BaseModel):
    california: Optional[StateData] = None
    pennsylvania: Optional[StateData] = None
    new_york: Optional[StateData] = None
    nevada: Optional[StateData] = None
    states_covered: List[str] = Field(default_factory=list)
    structured_format_description: Optional[str] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_llc_comparison() -> str:
    return """
    Extract a structured comparison for the LLC requirements across exactly four states: California, Pennsylvania, New York, and Nevada, from the given answer.

    For each of these states, extract the following fields exactly as written in the answer (use strings, do not normalize or infer):
    - initial_filing_fee: The current initial LLC filing fee amount or description.
    - annual_report_required: Whether periodic/annual/biennial reports/statements are required (e.g., "Yes (biennial)", "No", or a description).
    - annual_report_due: When such report(s) are due (e.g., "within 90 days and every two years", "each year by April 1", etc.).
    - annual_report_fee: The filing fee for such report(s) (e.g., "$20", "$0", "varies", etc.).
    - registered_agent_required: Whether a registered agent (or equivalent) is required (e.g., "Yes", "No", or description).
    - registered_agent_physical_instate_required: Whether the RA must have a physical in-state street address (not a P.O. Box), as stated in the answer.
    - processing_time: The standard/typical processing time for LLC formation applications as given in the answer (e.g., "5-10 business days", "varies", etc.).
    - official_urls: A list of direct official state government URL(s) (Secretary of State or equivalent) that the answer cites for that state's information. Only include URLs explicitly mentioned in the answer and that appear to be official government sites. If none are cited for that state in the answer, return an empty list.

    Also extract:
    - states_covered: The list of state names that the answer explicitly covers (as strings in the answer, e.g., "California", "New York", etc.).
    - structured_format_description: Briefly describe how the answer structures the comparison (e.g., "Markdown table with consistent columns", "Bulleted lists per state with the same fields", "Unstructured prose", etc.).

    Return a single JSON object with top-level fields:
    - california: StateData or null
    - pennsylvania: StateData or null
    - new_york: StateData or null
    - nevada: StateData or null
    - states_covered: string[]
    - structured_format_description: string or null

    Rules:
    - Do not invent values; if a specific field is not present in the answer, return null for that field (or empty list for official_urls).
    - Keep amounts and descriptions exactly as written (e.g., "$70", "$75 online / $70 by mail", "varies").
    - Only extract URLs that are explicitly present in the answer text.
    """


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _nz(v: Optional[str], fallback: str = "unspecified") -> str:
    return v.strip() if isinstance(v, str) and v.strip() else fallback


def _state_attr_key(id_base: str) -> str:
    # Map id base to attribute in extraction model
    mapping = {
        "California": "california",
        "Pennsylvania": "pennsylvania",
        "New_York": "new_york",
        "Nevada": "nevada",
    }
    return mapping[id_base]


def _human_name_from_id_base(id_base: str) -> str:
    return {
        "California": "California",
        "Pennsylvania": "Pennsylvania",
        "New_York": "New York",
        "Nevada": "Nevada",
    }[id_base]


def _official_domain_guidance_for_state(state_name: str) -> str:
    # Guidance examples for domains considered official
    examples = {
        "California": "*.ca.gov (e.g., sos.ca.gov)",
        "Pennsylvania": "*.pa.gov (e.g., dos.pa.gov) or *.state.pa.us",
        "New York": "*.ny.gov (e.g., dos.ny.gov)",
        "Nevada": "*.nv.gov (e.g., nvsos.gov or silverflume.nv.gov)",
    }
    hint = examples.get(state_name, "*.gov domain of the state's official agencies")
    return (
        f"Treat a URL as 'official state government' if it is hosted on an official government domain, "
        f"such as {hint}. Avoid private/commercial sites."
    )


# -----------------------------------------------------------------------------
# Verification builders per state
# -----------------------------------------------------------------------------
async def verify_state_requirements(
    evaluator: Evaluator,
    parent_node,
    id_base: str,
    data: Optional[StateData],
) -> None:
    """
    Build verification subtree for one state under a critical PARALLEL node.
    All children are critical (as required by the rubric).
    """
    state_node = evaluator.add_parallel(
        id=f"{id_base}_Requirements",
        desc=f"Required LLC formation and ongoing compliance details for {_human_name_from_id_base(id_base)}.",
        parent=parent_node,
        critical=True,
    )

    # Ensure we have a StateData instance to avoid attribute errors
    data = data or StateData()
    state_name = _human_name_from_id_base(id_base)
    urls = list(data.official_urls or [])

    # 1) Official State Gov URL(s)
    official_url_node = evaluator.add_leaf(
        id=f"{id_base}_Official_State_Gov_URL",
        desc=f"Provides direct URL(s) to official {state_name} state government source(s) (Secretary of State or equivalent) where the listed information can be verified.",
        parent=state_node,
        critical=True,
    )

    claim_official = (
        f"For {state_name}, at least one of these URL(s) is an official state government website and is a direct page "
        f"related to LLC formation/compliance information: {urls}."
    )
    add_ins_official = (
        "Judge officialness primarily by the domain (e.g., a .gov domain or the official Secretary of State/Department of State site). "
        + _official_domain_guidance_for_state(state_name)
        + " If the provided list is empty or contains only non-governmental sources, mark as Incorrect."
    )
    # If URL list is empty, this will still run; the model should mark it incorrect per instruction.
    await evaluator.verify(
        claim=claim_official,
        node=official_url_node,
        sources=urls if urls else None,
        additional_instruction=add_ins_official,
    )

    # Helper to add leaves that depend on the official URL node
    async def add_and_verify(
        leaf_id_suffix: str,
        leaf_desc: str,
        claim_text: str,
        additional_instruction: str = "None",
    ):
        node = evaluator.add_leaf(
            id=f"{id_base}_{leaf_id_suffix}",
            desc=leaf_desc,
            parent=state_node,
            critical=True,
        )
        await evaluator.verify(
            claim=claim_text,
            node=node,
            sources=urls if urls else None,
            additional_instruction=additional_instruction,
            extra_prerequisites=[official_url_node],  # Gate by official URL success
        )

    # 2) Initial filing fee
    claim_fee = (
        f"In {state_name}, the current initial LLC filing fee (i.e., the base filing fee to form an LLC) is "
        f"'{_nz(data.initial_filing_fee)}'."
    )
    await add_and_verify(
        leaf_id_suffix="Initial_LLC_Filing_Fee",
        leaf_desc=f"States the current initial LLC filing fee for {state_name}.",
        claim_text=claim_fee,
        additional_instruction=(
            "Verify the base/standard initial formation filing fee for an LLC as shown on the official page. "
            "Allow minor formatting differences (currency symbol, punctuation) as long as the amount matches. "
            "If multiple standard options (e.g., online vs mail) are shown, consider the answer correct if any one "
            "stated fee clearly matches the official page."
        ),
    )

    # 3) Annual report requirements (required?, due, fee)
    claim_annual = (
        f"For {state_name}, the periodic report requirement for LLCs is described as follows: "
        f"required status '{_nz(data.annual_report_required)}'; due '{_nz(data.annual_report_due)}'; "
        f"filing fee '{_nz(data.annual_report_fee)}'."
    )
    await add_and_verify(
        leaf_id_suffix="Annual_Report_Requirements",
        leaf_desc=(
            f"States whether annual reports are required in {state_name}; if required, includes when due and the filing fee."
        ),
        claim_text=claim_annual,
        additional_instruction=(
            "Confirm on the official page whether LLCs must file a periodic report/statement (annual or biennial), "
            "when it is due, and the fee amount. For New York, the biennial statement counts as the periodic requirement. "
            "Minor wording differences are acceptable if the meaning matches."
        ),
    )

    # 4) Registered agent requirements
    claim_ra = (
        f"In {state_name}, a registered agent requirement is '{_nz(data.registered_agent_required)}', "
        f"and the registered agent must have a physical in-state address: '{_nz(data.registered_agent_physical_instate_required)}'."
    )
    await add_and_verify(
        leaf_id_suffix="Registered_Agent_Requirements",
        leaf_desc=(
            f"States whether a registered agent is required in {state_name} and whether the registered agent must have a physical in-state address."
        ),
        claim_text=claim_ra,
        additional_instruction=(
            "Confirm that an LLC must designate a registered agent (or agent for service of process) and whether that agent "
            "must maintain a physical street address in the state (no P.O. Boxes). Accept synonymous terms such as "
            "'registered office' or 'agent for service of process'."
        ),
    )

    # 5) Standard processing time
    claim_proc = (
        f"For {state_name}, the standard (non-expedited) processing time for LLC formation applications is stated as "
        f"'{_nz(data.processing_time)}'."
    )
    await add_and_verify(
        leaf_id_suffix="Standard_Processing_Time",
        leaf_desc=f"States the standard processing time for {state_name} LLC formation applications.",
        claim_text=claim_proc,
        additional_instruction=(
            "Check the official page for typical/standard processing times (not expedited). Ranges or qualitative "
            "descriptions like 'varies' are acceptable if clearly indicated on the official source."
        ),
    )

    # 6) Timeframe current (2025–2026)
    claim_timeframe = (
        f"The official source page(s) used for {state_name} are current/applicable for the 2025–2026 timeframe "
        f"(e.g., page last updated in 2024–2026 or otherwise indicates that the listed fees/requirements are current)."
    )
    await add_and_verify(
        leaf_id_suffix="Timeframe_Current_2025_2026",
        leaf_desc=f"Information provided for {state_name} is current/applicable to the 2025–2026 timeframe.",
        claim_text=claim_timeframe,
        additional_instruction=(
            "Consider this satisfied if the official page indicates a last‑updated/effective date in 2024–2026 or otherwise "
            "clearly presents current fees/requirements with no signs of being outdated. If the page is obviously old "
            "and states an outdated year (e.g., 2018), mark as Incorrect."
        ),
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Entry point for evaluating the multi-state LLC requirements task.
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

    # Extract structured data from the answer
    extraction: LLCComparisonExtraction = await evaluator.extract(
        prompt=prompt_extract_llc_comparison(),
        template_class=LLCComparisonExtraction,
        extraction_name="llc_multi_state_extraction",
    )

    # Ground truth meta (expected coverage only)
    evaluator.add_ground_truth(
        {
            "expected_states": ["California", "Pennsylvania", "New York", "Nevada"],
            "required_fields": [
                "initial_filing_fee",
                "annual_report_required/due/fee",
                "registered_agent_required/physical_instate_required",
                "processing_time",
                "official_urls",
            ],
            "timeframe": "2025–2026",
        },
        gt_type="task_requirements",
    )

    # Main critical parallel node (as per rubric)
    main = evaluator.add_parallel(
        id="Multi_State_LLC_Formation_Requirements_Analysis",
        desc=(
            "Compare LLC formation requirements and ongoing compliance obligations across California, Pennsylvania, "
            "New York, and Nevada using official state government sources, and present results in a structured format for easy comparison."
        ),
        parent=root,
        critical=True,
    )

    # Coverage check: exactly the 4 required states, no extras
    # Normalize states from extraction: rely on states_covered + presence of blocks
    covered = set([s.strip().lower() for s in (extraction.states_covered or []) if isinstance(s, str)])
    # Also infer coverage from presence of state blocks if missing in list
    if extraction.california:
        covered.add("california")
    if extraction.pennsylvania:
        covered.add("pennsylvania")
    if extraction.new_york:
        covered.add("new york")
    if extraction.nevada:
        covered.add("nevada")

    required_set = {"california", "pennsylvania", "new york", "nevada"}
    coverage_ok = (covered == required_set)

    evaluator.add_custom_node(
        result=coverage_ok,
        id="Coverage_Exactly_Four_States",
        desc="Output covers exactly California, Pennsylvania, New York, and Nevada (no missing states; no extra states).",
        parent=main,
        critical=True,
    )

    # Structured comparison format check (simple verification)
    structured_node = evaluator.add_leaf(
        id="Structured_Comparison_Format",
        desc=(
            "Findings are presented in a structured format that enables easy comparison across all four states "
            "(consistent fields/columns for each required category)."
        ),
        parent=main,
        critical=True,
    )
    structured_claim = (
        "The answer presents information for California, Pennsylvania, New York, and Nevada in a structured comparison "
        "format with consistent fields for: initial filing fee; periodic/annual report requirement (with due schedule and fee); "
        "registered agent requirement (including in-state physical address requirement); standard processing time; and official source URL(s)."
    )
    await evaluator.verify(
        claim=structured_claim,
        node=structured_node,
        additional_instruction=(
            "Consider tables or clearly aligned bullet lists as structured. Minor wording/order differences are acceptable as long as "
            "each of the five categories appears for each state."
        ),
    )

    # Per-state verification subtrees (all critical under main)
    await verify_state_requirements(
        evaluator,
        main,
        id_base="California",
        data=getattr(extraction, _state_attr_key("California")),
    )
    await verify_state_requirements(
        evaluator,
        main,
        id_base="Pennsylvania",
        data=getattr(extraction, _state_attr_key("Pennsylvania")),
    )
    await verify_state_requirements(
        evaluator,
        main,
        id_base="New_York",
        data=getattr(extraction, _state_attr_key("New_York")),
    )
    await verify_state_requirements(
        evaluator,
        main,
        id_base="Nevada",
        data=getattr(extraction, _state_attr_key("Nevada")),
    )

    # Return final summary
    return evaluator.get_summary()