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
TASK_ID = "btc_atm_operator_investment_eval"
TASK_DESCRIPTION = """
You are evaluating investment opportunities in the Bitcoin ATM operator market in the United States.
Identify two distinct Bitcoin ATM operators that meet ALL of the following criteria:
(1) The operator must rank among the top 10 Bitcoin ATM operators in the United States by number of installations,
(2) The operator must have a minimum of 5,000 installed Bitcoin ATM locations as of 2025-2026,
(3) The operator must support daily transaction limits of at least $20,000 per customer,
(4) The operator must have multi-state geographic coverage across the United States, and
(5) The operator's transaction fee structure must be publicly documented and available.
For each qualifying operator, provide the operator's name, a brief description of their operational scale and market position,
and reference URL(s) that verify they meet the specified criteria.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OperatorCriteriaSources(BaseModel):
    """Categorized source URLs per operator for each required criterion."""
    top10: List[str] = Field(default_factory=list)                # Ranks among US top 10 by installations
    min5000_locations: List[str] = Field(default_factory=list)    # >= 5,000 installed locations (as of 2025–2026)
    daily_limit_20000: List[str] = Field(default_factory=list)    # Daily limit >= $20,000
    multi_state_coverage: List[str] = Field(default_factory=list) # Multi-state US coverage
    public_fees: List[str] = Field(default_factory=list)          # Public fee documentation
    general: List[str] = Field(default_factory=list)              # Any other supporting/summary URLs


class OperatorExtraction(BaseModel):
    """Extracted content for a single operator."""
    operator_name: Optional[str] = None
    brief_scale_and_position_description: Optional[str] = None
    sources: OperatorCriteriaSources = Field(default_factory=OperatorCriteriaSources)


class OperatorsExtraction(BaseModel):
    """Two-operator extraction result."""
    operator_1: Optional[OperatorExtraction] = None
    operator_2: Optional[OperatorExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_operators() -> str:
    return """
    Extract exactly two Bitcoin ATM operators as presented in the answer, along with a brief description and categorized source URLs.
    For each operator, extract the following fields:

    - operator_name: The operator's name (string).
    - brief_scale_and_position_description: One or two sentences summarizing the operator’s operational scale and market position
      (e.g., coverage, ranking, approximate footprint, leadership claims). This should be text directly from the answer (or condensed
      if overly long, but do not invent).
    - sources: An object with URL arrays for each criterion. Categorize any URLs cited in the answer into the following lists:
        • top10: URLs that support that the operator ranks among the top 10 Bitcoin ATM operators in the United States by number of installations.
        • min5000_locations: URLs that support the claim that the operator has at least 5,000 installed Bitcoin ATM locations as of 2025–2026.
        • daily_limit_20000: URLs that support the claim that the operator supports daily transaction limits of at least $20,000 per customer.
        • multi_state_coverage: URLs that support the operator's multi-state geographic coverage across the United States.
        • public_fees: URLs that publicly document the operator’s transaction fee structure (fee schedule or fees page).
        • general: Any additional related/supporting URLs for the operator that do not clearly fit the above categories.

    IMPORTANT RULES:
    - Only extract URLs explicitly present in the answer text. If the same URL is relevant to multiple criteria, include it in each relevant list.
    - If a field or a category of URLs is not provided in the answer, return null for text fields or an empty list for URLs.
    - Normalize URLs; include full protocol (prepend http:// if missing).
    - Preserve up to 15 URLs per list if many are provided; keep the most directly relevant first.

    Return a JSON object with fields: operator_1 and operator_2, each matching the specified structure.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _all_urls(src: OperatorCriteriaSources) -> List[str]:
    if not src:
        return []
    merged = (
        list(src.top10)
        + list(src.min5000_locations)
        + list(src.daily_limit_20000)
        + list(src.multi_state_coverage)
        + list(src.public_fees)
        + list(src.general)
    )
    return _dedup_preserve_order(merged)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_operator(
    evaluator: Evaluator,
    parent_node,
    index: int,
    op: OperatorExtraction,
    other_operator_name: Optional[str] = None,
    other_name_leaf: Optional[Any] = None,  # VerificationNode, kept as Any for typing lightness
) -> Dict[str, Any]:
    """
    Build the verification subtree for a single operator and execute verifications.
    Returns key nodes for potential cross-operator checks (e.g., name node).
    """
    idx = index  # 1 or 2
    disp = f"operator_{idx}"
    op_name = (op.operator_name or "").strip()

    # Create operator node as CRITICAL under root (both operators must pass)
    op_node = evaluator.add_parallel(
        id=f"{disp}",
        desc=f"{'First' if idx == 1 else 'Second'} qualifying Bitcoin ATM operator (must satisfy all criteria)",
        parent=parent_node,
        critical=True
    )

    # 1) Operator name provided (critical existence)
    name_leaf = evaluator.add_custom_node(
        result=bool(op_name),
        id=f"{disp}_operator_name",
        desc="Provide the operator's name",
        parent=op_node,
        critical=True
    )

    # 2) Brief description provided (critical existence as per rubric)
    desc_text = (op.brief_scale_and_position_description or "").strip()
    desc_leaf = evaluator.add_custom_node(
        result=bool(desc_text),
        id=f"{disp}_brief_scale_and_position_description",
        desc="Provide a brief description of the operator’s operational scale and market position",
        parent=op_node,
        critical=True
    )

    # 3) Supporting reference URLs overall (critical presence)
    all_urls = _all_urls(op.sources) if op and op.sources else []
    refs_leaf = evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id=f"{disp}_supporting_reference_urls",
        desc="Provide reference URL(s) that support the claims used to satisfy the required criteria for this operator",
        parent=op_node,
        critical=True
    )

    # 4) Criterion: Top 10 by US installations
    top10_sources = list(op.sources.top10 if op and op.sources else [])
    top10_src_exist = evaluator.add_custom_node(
        result=len(top10_sources) > 0,
        id=f"{disp}_top10_sources_present",
        desc="At least one URL is provided to support the Top-10-by-installations claim",
        parent=op_node,
        critical=True
    )
    top10_leaf = evaluator.add_leaf(
        id=f"{disp}_top_10_us_by_installations",
        desc="Operator ranks among the top 10 Bitcoin ATM operators in the United States by number of installations",
        parent=op_node,
        critical=True
    )

    # 5) Criterion: >= 5,000 installed locations as of 2025–2026
    loc_sources = list(op.sources.min5000_locations if op and op.sources else [])
    loc_src_exist = evaluator.add_custom_node(
        result=len(loc_sources) > 0,
        id=f"{disp}_min5000_sources_present",
        desc="At least one URL is provided to support the >=5,000 installed locations (as of 2025–2026) claim",
        parent=op_node,
        critical=True
    )
    min5000_leaf = evaluator.add_leaf(
        id=f"{disp}_min_5000_locations_2025_2026",
        desc="Operator has at least 5,000 installed Bitcoin ATM locations as of 2025–2026",
        parent=op_node,
        critical=True
    )

    # 6) Criterion: Daily transaction limit >= $20,000
    limit_sources = list(op.sources.daily_limit_20000 if op and op.sources else [])
    limit_src_exist = evaluator.add_custom_node(
        result=len(limit_sources) > 0,
        id=f"{disp}_daily_limit_sources_present",
        desc="At least one URL is provided to support the >=$20,000 daily limit claim",
        parent=op_node,
        critical=True
    )
    daily_limit_leaf = evaluator.add_leaf(
        id=f"{disp}_daily_limit_at_least_20000",
        desc="Operator supports daily transaction limits of at least $20,000 per customer",
        parent=op_node,
        critical=True
    )

    # 7) Criterion: Multi-state US coverage
    cov_sources = list(op.sources.multi_state_coverage if op and op.sources else [])
    cov_src_exist = evaluator.add_custom_node(
        result=len(cov_sources) > 0,
        id=f"{disp}_coverage_sources_present",
        desc="At least one URL is provided to support the multi-state US coverage claim",
        parent=op_node,
        critical=True
    )
    multistate_leaf = evaluator.add_leaf(
        id=f"{disp}_multi_state_us_coverage",
        desc="Operator has multi-state geographic coverage across the United States",
        parent=op_node,
        critical=True
    )

    # 8) Criterion: Public fee documentation available
    fee_sources = list(op.sources.public_fees if op and op.sources else [])
    fee_src_exist = evaluator.add_custom_node(
        result=len(fee_sources) > 0,
        id=f"{disp}_fees_sources_present",
        desc="At least one URL is provided that publicly documents the operator's fees",
        parent=op_node,
        critical=True
    )
    fees_leaf = evaluator.add_leaf(
        id=f"{disp}_public_fee_documentation",
        desc="Operator's transaction fee structure is publicly documented and available",
        parent=op_node,
        critical=True
    )

    # Prepare batch verifications for the criteria that must be URL-grounded.
    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # Top 10 ranking
    claims_and_sources.append((
        f"The operator '{op_name}' ranks among the top 10 Bitcoin ATM operators in the United States by number of installations.",
        top10_sources,
        top10_leaf,
        "Verify that the provided page(s) explicitly place this operator in a 'Top 10' for US Bitcoin ATM installations (by number of machines). "
        "Accept clear lists or rankings (e.g., industry trackers). If unclear, not US-specific, or not by installations, mark as not supported."
    ))

    # >= 5,000 installed locations as of 2025–2026
    claims_and_sources.append((
        f"As of 2025 or 2026, the operator '{op_name}' has at least 5,000 installed Bitcoin ATM locations.",
        loc_sources,
        min5000_leaf,
        "Look for explicit statements of the operator's installed count being >= 5,000, with timing that is clearly 2025 or 2026 (or a current/updated page explicitly indicating that timeframe). "
        "Phrases like 'over 5,000 locations nationwide' are acceptable if timing aligns. If date is clearly much earlier or number < 5,000, mark as not supported."
    ))

    # Daily transaction limit >= $20,000
    claims_and_sources.append((
        f"The operator '{op_name}' supports daily transaction limits of at least $20,000 per customer.",
        limit_sources,
        daily_limit_leaf,
        "Confirm that per-customer daily limits (possibly at higher KYC/verification tiers) are $20,000 or higher for buy/sell. "
        "If limits are per day and >= $20,000 in any tier, accept. If only lower limits are shown, mark as not supported."
    ))

    # Multi-state coverage
    claims_and_sources.append((
        f"The operator '{op_name}' operates across multiple U.S. states (more than one state).",
        cov_sources,
        multistate_leaf,
        "Validate that the operator has active machines or services across multiple US states (e.g., a coverage map, list of states, 'nationwide coverage' with states shown). "
        "Operating in just a single state does not qualify."
    ))

    # Public fee documentation available
    claims_and_sources.append((
        f"The operator '{op_name}' publicly documents its transaction fee structure (e.g., lists fees or a fee schedule) on the provided page(s).",
        fee_sources,
        fees_leaf,
        "Verify that the page(s) explicitly describe the operator's transaction fees (e.g., % fees, fixed fees, or ranges). "
        "General marketing pages without fee information do not qualify."
    ))

    # Run URL-grounded verifications in parallel (each will auto-skip if critical siblings failed)
    await evaluator.batch_verify(claims_and_sources)

    # Distinctness check for operator #2
    if idx == 2:
        distinct_leaf = evaluator.add_leaf(
            id=f"{disp}_distinct_from_operator_1",
            desc="Second operator is a distinct entity from the first operator (not the same operator under different naming)",
            parent=op_node,
            critical=True
        )
        other_name = (other_operator_name or "").strip()
        await evaluator.verify(
            claim=f"'{op_name}' and '{other_name}' are different operator entities (i.e., not the same company under a variant name).",
            node=distinct_leaf,
            sources=None,  # Logical check only
            additional_instruction="Consider common aliases and brand variations. If the two names clearly refer to the same legal entity or brand/operator, mark as incorrect.",
            extra_prerequisites=[name_leaf] + ([other_name_leaf] if other_name_leaf is not None else [])
        )

    return {
        "operator_node": op_node,
        "name_leaf": name_leaf
    }


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
    Evaluate an answer for the Bitcoin ATM operator investment screening task.
    """
    # Initialize evaluator (root set to PARALLEL; keep root non-critical to allow critical children)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify two distinct Bitcoin ATM operators meeting all specified investment criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract operators and categorized sources from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_operators(),
        template_class=OperatorsExtraction,
        extraction_name="operators_extraction"
    )

    # Normalize missing operators
    op1 = extracted.operator_1 or OperatorExtraction()
    op2 = extracted.operator_2 or OperatorExtraction()

    # Build and verify operator #1 subtree
    op1_result = await verify_operator(
        evaluator=evaluator,
        parent_node=root,
        index=1,
        op=op1
    )

    # Build and verify operator #2 subtree (including distinctness from operator #1)
    await verify_operator(
        evaluator=evaluator,
        parent_node=root,
        index=2,
        op=op2,
        other_operator_name=op1.operator_name,
        other_name_leaf=op1_result["name_leaf"]
    )

    # Return structured summary
    return evaluator.get_summary()