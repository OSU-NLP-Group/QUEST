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
TASK_ID = "mcps_top4_cabinet_fy2026"
TASK_DESCRIPTION = (
    "Identify the top 4 highest-paid cabinet-level administrative positions (excluding the superintendent) at "
    "Montgomery County Public Schools (MCPS) in Maryland for fiscal year 2026. For each position, provide: "
    "(1) the exact job title, (2) the annual salary, and (3) verification that the salary falls within the "
    "documented national salary range for that specific type of educational leadership role."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    title: Optional[str] = None
    salary: Optional[str] = None
    mcps_urls: List[str] = Field(default_factory=list)
    national_role_type: Optional[str] = None
    national_range_min: Optional[str] = None
    national_range_max: Optional[str] = None
    national_range_urls: List[str] = Field(default_factory=list)


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract up to the top 4 cabinet-level administrative positions (excluding the superintendent) as they are presented in the answer text.
    For each listed position, extract the following fields exactly as stated in the answer:

    - title: The exact job title for the MCPS position (string; return null if not provided).
    - salary: The annual salary for FY2026 as stated in the answer (string; include currency symbol or formatting if present; return null if missing).
    - mcps_urls: All URLs in the answer that document the position and/or its FY2026 salary from MCPS or official MCPS-linked documents (array of strings; empty array if none).
    - national_role_type: The role type/category used by the answer to justify the national salary range (e.g., 'Deputy Superintendent', 'Chief Operating Officer', 'Assistant Superintendent'; return null if missing).
    - national_range_min: The lower bound of the national salary range for this role type as presented in the answer (string; return null if not provided).
    - national_range_max: The upper bound of the national salary range for this role type as presented in the answer (string; return null if not provided).
    - national_range_urls: All URLs the answer cites for the national salary range evidence for this role type (array of strings; empty if none).

    Rules:
    - Do not invent any information. Only extract content explicitly present in the answer.
    - Extract URLs exactly as shown (including those in markdown links).
    - If the answer lists more than 4 positions, extract only the first 4 in the order presented.
    - If fewer than 4 positions are present, extract only those and leave the rest absent (the evaluator will pad).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal_name(idx: int) -> str:
    mapping = {0: "First", 1: "Second", 2: "Third", 3: "Fourth"}
    return mapping.get(idx, f"Position {idx+1}")


def safe_str(val: Optional[str]) -> str:
    return val if (val is not None and str(val).strip() != "") else ""


# --------------------------------------------------------------------------- #
# Verification for a single position                                          #
# --------------------------------------------------------------------------- #
async def verify_cabinet_position(
    evaluator: Evaluator,
    parent_node,
    position: PositionItem,
    index: int,
) -> None:
    """
    Build the verification subtree for a single cabinet position, following the rubric.
    """
    ord_label = ordinal_name(index)

    # Parent node for this position (parallel, non-critical)
    pos_node = evaluator.add_parallel(
        id=f"Cabinet_Position_{index+1}",
        desc=f"{ord_label} highest-paid cabinet-level position correctly identified and verified",
        parent=parent_node,
        critical=False
    )

    # 1) Source URL existence (critical presence gate)
    has_mcps_source = bool(position.mcps_urls) and any(u.strip() for u in position.mcps_urls)
    evaluator.add_custom_node(
        result=has_mcps_source,
        id=f"Position_{index+1}_Source_URL",
        desc="Valid source URL provided documenting this position and salary at MCPS",
        parent=pos_node,
        critical=True
    )

    # 2) Title correctness against MCPS source(s)
    title_node = evaluator.add_leaf(
        id=f"Position_{index+1}_Title_Correct",
        desc="Exact job title correctly stated for this position",
        parent=pos_node,
        critical=True
    )
    title_claim = (
        f"The MCPS source(s) mention this job title (or a clear equivalent at MCPS): '{safe_str(position.title)}'. "
        f"Allow minor variations in capitalization or phrasing that clearly refer to the same role."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_node,
        sources=position.mcps_urls,
        additional_instruction="Verify that the cited MCPS page(s) clearly identify the same job title for this MCPS role."
    )

    # 3) Cabinet-level status (reports directly to superintendent)
    cabinet_node = evaluator.add_leaf(
        id=f"Position_{index+1}_Cabinet_Level",
        desc="Position is a cabinet-level role reporting directly to superintendent",
        parent=pos_node,
        critical=True
    )
    cabinet_claim = (
        "This position is a cabinet-level role at MCPS that reports directly to the Superintendent "
        "and/or is described as part of the Superintendent's executive leadership/cabinet team."
    )
    await evaluator.verify(
        claim=cabinet_claim,
        node=cabinet_node,
        sources=position.mcps_urls,
        additional_instruction="Look for phrasing like 'Superintendent's Cabinet', 'executive leadership team', "
                               "'reports directly to the Superintendent', or similar indications."
    )

    # 4) Top-4 highest-paid (excluding superintendent)
    top4_node = evaluator.add_leaf(
        id=f"Position_{index+1}_Top_4_Salary",
        desc="Position is confirmed as one of top 4 highest-paid positions excluding superintendent",
        parent=pos_node,
        critical=True
    )
    top4_claim = (
        "For FY2026 at MCPS, this position is among the top four highest-paid cabinet-level administrative roles "
        "excluding the Superintendent."
    )
    await evaluator.verify(
        claim=top4_claim,
        node=top4_node,
        sources=position.mcps_urls,
        additional_instruction="Use the cited MCPS documents (salary schedules, budgets, board docs) to determine whether "
                               "this role's salary is within the top four among cabinet-level roles (excluding the Superintendent). "
                               "If ties occur, consider the position as top-four if appropriate."
    )

    # 5) Salary correctness (exact/normalized figure on MCPS source)
    salary_node = evaluator.add_leaf(
        id=f"Position_{index+1}_Salary_Correct",
        desc="Annual salary amount correctly stated from official MCPS sources",
        parent=pos_node,
        critical=True
    )
    salary_claim = (
        f"For FY2026, the annual salary for the position '{safe_str(position.title)}' at MCPS is stated as "
        f"'{safe_str(position.salary)}' (normalization of formatting or rounding is acceptable)."
    )
    await evaluator.verify(
        claim=salary_claim,
        node=salary_node,
        sources=position.mcps_urls,
        additional_instruction="Check that the FY2026 salary on the MCPS source(s) matches the stated figure. "
                               "Allow minor formatting differences (commas, currency symbol) or reasonable rounding."
    )

    # 6) Salary within documented national range for this role type
    nat_range_node = evaluator.add_leaf(
        id=f"Position_{index+1}_National_Range",
        desc="Salary verified to fall within documented national range for this specific role type",
        parent=pos_node,
        critical=True
    )
    nat_role = safe_str(position.national_role_type)
    nat_min = safe_str(position.national_range_min)
    nat_max = safe_str(position.national_range_max)
    nat_claim = (
        f"The stated salary '{safe_str(position.salary)}' falls within the documented national salary range for "
        f"the role type '{nat_role}', which is between '{nat_min}' and '{nat_max}', according to the cited national sources."
    )
    await evaluator.verify(
        claim=nat_claim,
        node=nat_range_node,
        sources=position.national_range_urls,
        additional_instruction="Verify that the cited national source(s) provide a range (or equivalent bounds) for the specified role type "
                               "that includes the given salary. Allow close synonyms in role naming (e.g., 'Deputy Superintendent' vs "
                               "'Associate Superintendent'). If the source provides multiple ranges (e.g., 25th–75th percentile), it's acceptable "
                               "if the salary falls within any range explicitly presented."
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
    Evaluate an answer for the MCPS top-4 cabinet-level positions in FY2026 task.
    """
    # Initialize evaluator
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

    # Group node to mirror rubric root
    group_node = evaluator.add_parallel(
        id="MCPS_Top_4_Cabinet_Positions",
        desc="Correctly identify the top 4 highest-paid cabinet-level administrative positions (excluding superintendent) at MCPS for FY2026",
        parent=root,
        critical=False
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Normalize to exactly 4 entries (pad or truncate)
    positions: List[PositionItem] = list(extracted.positions or [])
    if len(positions) > 4:
        positions = positions[:4]
    while len(positions) < 4:
        positions.append(PositionItem())

    # Build verification subtrees for each of the 4 positions
    for idx in range(4):
        await verify_cabinet_position(
            evaluator=evaluator,
            parent_node=group_node,
            position=positions[idx],
            index=idx
        )

    # Return structured result
    return evaluator.get_summary()