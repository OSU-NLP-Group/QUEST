import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# ---------------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------------
TASK_ID = "tx_grocery_holidays_2025"
TASK_DESCRIPTION = (
    "I'm planning my grocery shopping around the Thanksgiving and Christmas holidays in 2025 and live in Texas. "
    "I need to know which major grocery store chains will be open during these holidays so I can plan accordingly.\n\n"
    "Please identify at least 4 major grocery store chains that operate in Texas and provide the following information for each chain:\n\n"
    "For Thanksgiving Day 2025 (Thursday, November 27, 2025):\n"
    "- Whether the chain's stores will be open or closed on Thanksgiving Day\n"
    "- If open, what are the specific operating hours or closing time for Thanksgiving Day\n"
    "- Whether the chain's pharmacies will be open or closed on Thanksgiving Day\n"
    "- A reference URL that confirms this Thanksgiving Day operating information\n\n"
    "For Christmas Day 2025 (Thursday, December 25, 2025):\n"
    "- Whether the chain's stores will be open or closed on Christmas Day\n"
    "- If open, what are the specific operating hours; if closed, confirm the closure\n"
    "- Whether the chain's pharmacies will be open or closed on Christmas Day\n"
    "- A reference URL that confirms this Christmas Day operating information\n\n"
    "Please ensure all information is specific to the 2025 holiday season and includes verifiable sources."
)

THANKSGIVING_LABEL = "Thanksgiving Day 2025"
THANKSGIVING_DATE = "Thursday, November 27, 2025"
CHRISTMAS_LABEL = "Christmas Day 2025"
CHRISTMAS_DATE = "Thursday, December 25, 2025"


# ---------------------------------------------------------------------------
# Extraction models
# ---------------------------------------------------------------------------
class HolidayInfo(BaseModel):
    operating_status: Optional[str] = None  # e.g., "open", "closed", "open with limited hours", etc.
    hours: Optional[str] = None             # e.g., "6am–3pm", "open until 2pm", etc. If closed, may be null.
    pharmacy_status: Optional[str] = None   # e.g., "open", "closed", "varies by location"
    urls: List[str] = Field(default_factory=list)  # URLs explicitly cited for this holiday's info


class ChainInfo(BaseModel):
    chain_name: Optional[str] = None
    thanksgiving: Optional[HolidayInfo] = None
    christmas: Optional[HolidayInfo] = None


class ChainsExtraction(BaseModel):
    chains: List[ChainInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------
def prompt_extract_chains() -> str:
    return """
    Extract up to 6 major grocery store chains (in order of appearance) that the answer mentions as operating in Texas and for which the answer provides holiday hours information for 2025. For each chain, extract:
    - chain_name: The grocery store chain name as written.
    - thanksgiving: Object containing fields for Thanksgiving 2025 (Thursday, November 27, 2025)
        * operating_status: Whether the chain's stores are "open" or "closed" (or equivalent wording) on Thanksgiving 2025, exactly as stated in the answer. If unknown or not provided, set to null.
        * hours: The specific operating hours or closing time if the stores are open on Thanksgiving 2025; if the stores are closed or hours are not provided, set to null.
        * pharmacy_status: Whether the chain's pharmacies are "open" or "closed" (or equivalent wording) on Thanksgiving 2025, exactly as stated in the answer. If unknown or not provided, set to null.
        * urls: Array of all URLs cited in the answer that specifically support or confirm Thanksgiving 2025 operating information for this chain. Use only actual URLs explicitly present in the answer.
    - christmas: Object containing fields for Christmas 2025 (Thursday, December 25, 2025)
        * operating_status: Whether the chain's stores are "open" or "closed" (or equivalent wording) on Christmas 2025, exactly as stated in the answer. If unknown or not provided, set to null.
        * hours: The specific operating hours if open on Christmas 2025; if closed or hours not provided, set to null.
        * pharmacy_status: Whether the chain's pharmacies are "open" or "closed" on Christmas 2025, exactly as stated in the answer. If unknown or not provided, set to null.
        * urls: Array of all URLs cited in the answer that specifically support or confirm Christmas 2025 operating information for this chain. Use only actual URLs explicitly present in the answer.

    Requirements and rules:
    - Do not invent or infer any information; only extract what is explicitly in the answer.
    - Each holiday's urls must be URLs directly present in the answer text (plain URL or inside a markdown link).
    - If a field is missing in the answer, set it to null (or [] for urls).
    - Prefer chain-level holiday pages for 2025. If a single page covers multiple holidays (e.g., both Thanksgiving and Christmas 2025), include it in both holidays' urls arrays if the answer associates it so.
    - The same URL may appear in both holidays' url arrays if the answer uses it for both.
    - Return the result under a top-level field "chains" as an array of objects with the fields exactly as defined by the schema.
    """


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def safe_text(v: Optional[str]) -> str:
    return v.strip() if isinstance(v, str) and v.strip() else "NOT PROVIDED"

def has_text(v: Optional[str]) -> bool:
    return bool(isinstance(v, str) and v.strip())

def holiday_urls(info: Optional[HolidayInfo]) -> List[str]:
    return list(info.urls) if (info and info.urls) else []

def chain_display_name(name: Optional[str], fallback: str) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else fallback


# ---------------------------------------------------------------------------
# Verification builders
# ---------------------------------------------------------------------------
async def verify_holiday_block(
    evaluator: Evaluator,
    parent_node,
    chain_index: int,
    chain_name: Optional[str],
    holiday_key: str,
    holiday_label: str,
    holiday_date_str: str,
    info: Optional[HolidayInfo],
) -> None:
    """
    Build and verify the holiday sub-tree for one chain.
    Structure (all children critical under this holiday block per rubric):
      - operating_status (leaf, verify by URLs)
      - hours (leaf, verify by URLs; if closed, confirming closure is acceptable)
      - pharmacy (leaf, verify by URLs)
      - url (leaf-equivalent using custom check: at least one URL present)
    """
    # Parent holiday node (parallel, non-critical)
    holiday_node = evaluator.add_parallel(
        id=f"chain_{chain_index+1}_{holiday_key}_2025",
        desc=f"{holiday_label} ({holiday_date_str}) operational information",
        parent=parent_node,
        critical=False
    )

    # Critical URL existence gate (custom node)
    urls = holiday_urls(info)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"chain_{chain_index+1}_{holiday_key}_url",
        desc=f"Provide a reference URL confirming the {holiday_label} operating information",
        parent=holiday_node,
        critical=True
    )

    # Build leaves
    # 1) Operating status
    op_leaf = evaluator.add_leaf(
        id=f"chain_{chain_index+1}_{holiday_key}_operating_status",
        desc=f"Confirm whether the chain is open or closed on {holiday_label}",
        parent=holiday_node,
        critical=True
    )
    op_status = safe_text(info.operating_status if info else None)
    op_claim = (
        f"On {holiday_label} ({holiday_date_str}), {chain_display_name(chain_name, 'this grocery chain')} stores are {op_status}."
    )
    op_addins = (
        f"- Verify using only the provided page(s) whether the chain is open or closed on {holiday_label} in 2025.\n"
        f"- The statement must be specific to 2025. If the page is for a different year or lacks a clear 2025 indication, mark Incorrect.\n"
        f"- Accept synonymous phrasings (e.g., 'closed' ~ 'not open').\n"
        f"- If the claim uses the placeholder 'NOT PROVIDED', you must mark it Incorrect.\n"
        f"- Chain-level US holiday policy counts for Texas stores unless the page explicitly excludes Texas.\n"
    )
    # 2) Hours (conditional)
    hours_leaf = evaluator.add_leaf(
        id=f"chain_{chain_index+1}_{holiday_key}_hours",
        desc=f"Provide specific operating hours or closing time for {holiday_label} if open",
        parent=holiday_node,
        critical=True
    )
    # If status indicates closed, acceptable to confirm closure; otherwise require hours
    is_closed = "closed" in op_status.lower()
    hours_str = safe_text(info.hours if info else None)
    if is_closed:
        hours_claim = (
            f"On {holiday_label} ({holiday_date_str}), {chain_display_name(chain_name, 'this grocery chain')} stores are closed."
        )
        hours_addins = (
            f"- For the hours requirement, confirming closure satisfies this check when stores are closed.\n"
            f"- Ensure the page explicitly indicates closure on {holiday_label} in 2025.\n"
            f"- If the page is for another year or does not specify 2025, mark Incorrect."
        )
    else:
        hours_claim = (
            f"On {holiday_label} ({holiday_date_str}), {chain_display_name(chain_name, 'this grocery chain')} stores operate with hours: {hours_str}."
        )
        hours_addins = (
            f"- This check requires specific 2025 {holiday_label} hours or a concrete closing time if open. "
            f"Generic language like 'hours vary by location' without times is not sufficient to support a specific hours claim.\n"
            f"- Accept reasonable format variations (e.g., 6am–3pm vs 6:00 AM to 3:00 PM) but they must semantically match.\n"
            f"- If the claim uses 'NOT PROVIDED', mark Incorrect.\n"
            f"- The page must clearly pertain to 2025."
        )

    # 3) Pharmacy status
    pharm_leaf = evaluator.add_leaf(
        id=f"chain_{chain_index+1}_{holiday_key}_pharmacy",
        desc=f"Indicate whether pharmacies at this chain are open or closed on {holiday_label}",
        parent=holiday_node,
        critical=True
    )
    pharm_status = safe_text(info.pharmacy_status if info else None)
    pharm_claim = (
        f"On {holiday_label} ({holiday_date_str}), pharmacies at {chain_display_name(chain_name, 'this grocery chain')} are {pharm_status}."
    )
    pharm_addins = (
        f"- Verify pharmacy-specific status for {holiday_label} in 2025 based on the provided page(s).\n"
        f"- Accept synonymous phrasings and notes like 'pharmacy hours may differ' only if the page still clearly indicates open vs closed for the holiday.\n"
        f"- If the claim uses 'NOT PROVIDED', mark Incorrect.\n"
        f"- If the page only provides general store info with no pharmacy mention, do not assume; mark Incorrect."
    )

    # Batch verify (three leaves) using the holiday URLs, gated by the URL existence node
    await evaluator.batch_verify(
        [
            (op_claim, urls, op_leaf, op_addins),
            (hours_claim, urls, hours_leaf, hours_addins),
            (pharm_claim, urls, pharm_leaf, pharm_addins),
        ]
    )


async def verify_chain(
    evaluator: Evaluator,
    root_node,
    chain: ChainInfo,
    chain_index: int,
) -> None:
    """
    Build the per-chain node and its holiday subtrees.
    """
    chain_node = evaluator.add_parallel(
        id=f"store_chain_{chain_index+1}",
        desc=f"{['First','Second','Third','Fourth','Fifth','Sixth'][chain_index] if chain_index < 6 else f'Chain #{chain_index+1}'} grocery store chain with complete holiday information",
        parent=root_node,
        critical=False
    )

    # Thanksgiving 2025
    await verify_holiday_block(
        evaluator=evaluator,
        parent_node=chain_node,
        chain_index=chain_index,
        chain_name=chain.chain_name,
        holiday_key="thanksgiving",
        holiday_label=THANKSGIVING_LABEL,
        holiday_date_str=THANKSGIVING_DATE,
        info=chain.thanksgiving,
    )

    # Christmas 2025
    await verify_holiday_block(
        evaluator=evaluator,
        parent_node=chain_node,
        chain_index=chain_index,
        chain_name=chain.chain_name,
        holiday_key="christmas",
        holiday_label=CHRISTMAS_LABEL,
        holiday_date_str=CHRISTMAS_DATE,
        info=chain.christmas,
    )


# ---------------------------------------------------------------------------
# Main evaluation entry
# ---------------------------------------------------------------------------
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify at least 4 major grocery store chains operating in Texas and provide comprehensive holiday schedule information for both Thanksgiving 2025 and Christmas 2025",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured chains info
    chains_extraction = await evaluator.extract(
        prompt=prompt_extract_chains(),
        template_class=ChainsExtraction,
        extraction_name="chains_extraction"
    )

    # Keep first 4 chains; pad with placeholders if fewer
    extracted_chains: List[ChainInfo] = list(chains_extraction.chains[:4])
    while len(extracted_chains) < 4:
        extracted_chains.append(ChainInfo())

    # Add useful GT/context info
    evaluator.add_ground_truth({
        "required_holidays": {
            "thanksgiving_2025": THANKSGIVING_DATE,
            "christmas_2025": CHRISTMAS_DATE
        },
        "region_focus": "Texas (chain-level US policy acceptable if applicable to Texas)"
    }, gt_type="task_requirements")

    # Build and verify each chain block
    for idx in range(4):
        await verify_chain(evaluator, root, extracted_chains[idx], idx)

    return evaluator.get_summary()