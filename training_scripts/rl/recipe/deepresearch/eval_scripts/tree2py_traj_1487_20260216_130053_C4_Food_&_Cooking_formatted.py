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
TASK_ID = "thanksgiving_retail_ops_2025"
TASK_DESCRIPTION = (
    "Based on official corporate holiday policies announced for 2025, identify which of the following four major "
    "national retail chains will have stores open for regular business operations on Thanksgiving Day, Thursday, "
    "November 27, 2025: Walmart, Target, Kroger, and Dollar General."
)

THANKSGIVING_DATE_LONG = "Thursday, November 27, 2025"
THANKSGIVING_SHORT = "Thanksgiving Day 2025"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RetailerInfo(BaseModel):
    status: Optional[str] = None  # e.g., "closed nationwide", "open with reduced hours", "open", "closed"
    hours: Optional[str] = None   # free-form hours text if provided (esp. for Kroger, Dollar General)
    sources: List[str] = Field(default_factory=list)  # URLs explicitly cited for this retailer


class HolidayOpsExtraction(BaseModel):
    walmart: Optional[RetailerInfo] = None
    target: Optional[RetailerInfo] = None
    kroger: Optional[RetailerInfo] = None
    dollar_general: Optional[RetailerInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_holiday_ops() -> str:
    return """
    Extract, for each of the four retailers Walmart, Target, Kroger, and Dollar General, the Thanksgiving Day 2025 (Thursday, November 27, 2025) operating status as presented in the answer.

    For each retailer, extract:
    1) status: A concise text description of whether stores are open or closed nationwide on Thanksgiving Day 2025. Use the exact phrasing from the answer if possible (e.g., "closed nationwide", "open with reduced hours", "open", "closed").
    2) hours: If the answer provides specific or typical operating hours for Thanksgiving Day 2025 (e.g., "open 8 AM–10 PM", "closing early around 3–4 PM"), extract that text as-is; otherwise return null.
    3) sources: All URLs (if any) that the answer cites for this retailer's Thanksgiving 2025 status/hours. Return a list of full URLs. Extract only URLs explicitly present in the answer (including markdown links). Do not invent URLs.

    Return a JSON object with the following top-level fields:
    - walmart: { status, hours, sources[] }
    - target: { status, hours, sources[] }
    - kroger: { status, hours, sources[] }
    - dollar_general: { status, hours, sources[] }

    If a retailer is not mentioned or some field is missing in the answer, set that field to null or an empty list accordingly.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def infer_open_closed(status_text: Optional[str]) -> str:
    """
    Infer a normalized open/closed label from a free-form status text.
    Returns: "open", "closed", or "unknown".
    """
    if not status_text:
        return "unknown"
    s = status_text.strip().lower()
    # If both words appear, prefer 'closed' if explicitly stated as "closed".
    if "closed" in s:
        return "closed"
    if "open" in s:
        return "open"
    return "unknown"


def get_sources(info: Optional[RetailerInfo]) -> List[str]:
    if not info:
        return []
    return info.sources or []


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_walmart(evaluator: Evaluator, parent_node, info: Optional[RetailerInfo]) -> None:
    walmart_node = evaluator.add_parallel(
        id="Walmart",
        desc="Walmart Thanksgiving Day 2025 status.",
        parent=parent_node,
        critical=False
    )

    # Leaf: Walmart_Status (critical)
    status_leaf = evaluator.add_leaf(
        id="Walmart_Status",
        desc="Correctly states whether Walmart stores are open or closed nationwide on Thanksgiving Day 2025 (Nov 27, 2025), consistent with the constraints.",
        parent=walmart_node,
        critical=True
    )

    norm = infer_open_closed(info.status if info else None)
    if norm == "open":
        status_claim = f"Walmart stores in the United States are open on {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG})."
    elif norm == "closed":
        status_claim = f"Walmart stores in the United States are closed on {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG})."
    else:
        # Ambiguous/unspecified in the answer; craft a claim that will be judged unsupported without sources.
        status_claim = f"The nationwide Walmart store status for {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG}) is clearly stated (open or closed)."

    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=get_sources(info),
        additional_instruction=(
            "Only mark as supported if at least one provided URL is an official Walmart corporate source (e.g., walmart.com or corporate.walmart.com) "
            "that explicitly mentions the Thanksgiving Day 2025 status (open or closed). "
            "If no valid official 2025 source is provided, or the claim is ambiguous, mark as not supported."
        )
    )

    # Leaf: Walmart_Optional_Official_Support (non-critical)
    support_leaf = evaluator.add_leaf(
        id="Walmart_Optional_Official_Support",
        desc="Optionally provides an official-policy-based support statement or source for Walmart's Thanksgiving Day 2025 status.",
        parent=walmart_node,
        critical=False
    )

    # We check that at least one official corporate source explicitly states Walmart's Thanksgiving 2025 status.
    support_claim = (
        f"This webpage is an official Walmart corporate page that explicitly states whether Walmart stores are open or closed on "
        f"{THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG})."
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_leaf,
        sources=get_sources(info),
        additional_instruction=(
            "Consider this supported only if the domain is clearly official (e.g., walmart.com, corporate.walmart.com, newsroom.walmart.com) "
            "and the content explicitly mentions Thanksgiving Day for year 2025 and the store status (open/closed). "
            "Third-party news or older-year policies do NOT count."
        )
    )


async def verify_target(evaluator: Evaluator, parent_node, info: Optional[RetailerInfo]) -> None:
    target_node = evaluator.add_parallel(
        id="Target",
        desc="Target Thanksgiving Day 2025 status.",
        parent=parent_node,
        critical=False
    )

    # Leaf: Target_Status (critical)
    status_leaf = evaluator.add_leaf(
        id="Target_Status",
        desc="Correctly states whether Target stores are open or closed nationwide on Thanksgiving Day 2025 (Nov 27, 2025), consistent with the constraints.",
        parent=target_node,
        critical=True
    )

    norm = infer_open_closed(info.status if info else None)
    if norm == "open":
        status_claim = f"Target stores in the United States are open on {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG})."
    elif norm == "closed":
        status_claim = f"Target stores in the United States are closed on {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG})."
    else:
        status_claim = f"The nationwide Target store status for {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG}) is clearly stated (open or closed)."

    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=get_sources(info),
        additional_instruction=(
            "Only mark as supported if at least one provided URL is an official Target corporate source (e.g., target.com, corporate.target.com, or Target Newsroom) "
            "that explicitly mentions the Thanksgiving Day 2025 status (open or closed). "
            "If no valid official 2025 source is provided, or the claim is ambiguous, mark as not supported."
        )
    )

    # Leaf: Target_Optional_Official_Support (non-critical)
    support_leaf = evaluator.add_leaf(
        id="Target_Optional_Official_Support",
        desc="Optionally provides an official-policy-based support statement or source for Target's Thanksgiving Day 2025 status.",
        parent=target_node,
        critical=False
    )

    support_claim = (
        f"This webpage is an official Target corporate page that explicitly states whether Target stores are open or closed on "
        f"{THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG})."
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_leaf,
        sources=get_sources(info),
        additional_instruction=(
            "Consider this supported only if the domain is clearly official (e.g., target.com, corporate.target.com) and the content explicitly mentions "
            "Thanksgiving Day 2025 and the store status (open/closed). Third-party news or non-2025 pages do NOT count."
        )
    )


async def verify_kroger(evaluator: Evaluator, parent_node, info: Optional[RetailerInfo]) -> None:
    kroger_node = evaluator.add_parallel(
        id="Kroger",
        desc="Kroger Thanksgiving Day 2025 status (and hours if required by constraints).",
        parent=parent_node,
        critical=False
    )

    # Leaf: Kroger_Open_Status (critical)
    open_leaf = evaluator.add_leaf(
        id="Kroger_Open_Status",
        desc="Correctly states that Kroger family stores are open on Thanksgiving Day 2025 (Nov 27, 2025), consistent with the constraints.",
        parent=kroger_node,
        critical=True
    )

    norm = infer_open_closed(info.status if info else None)
    if norm == "open":
        open_claim = f"Kroger (including Kroger family supermarkets) stores are open on {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG})."
    elif norm == "closed":
        # If the answer claims closed, we still verify that claim against the provided sources.
        open_claim = f"Kroger (including Kroger family supermarkets) stores are closed on {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG})."
    else:
        open_claim = f"The Kroger family stores' nationwide status for {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG}) is clearly stated (open or closed)."

    await evaluator.verify(
        claim=open_claim,
        node=open_leaf,
        sources=get_sources(info),
        additional_instruction=(
            "Only mark as supported if at least one provided URL is an official Kroger corporate source (e.g., kroger.com or corporate.kroger.com) "
            "that explicitly mentions 2025 Thanksgiving Day store status. Treat ambiguous or non-2025 pages as not supported."
        )
    )

    # Leaf: Kroger_Hours_Detail (critical)
    hours_leaf = evaluator.add_leaf(
        id="Kroger_Hours_Detail",
        desc="Correctly states that Kroger operates with reduced hours on Thanksgiving Day 2025, typically closing around 3–4 PM local time, consistent with the constraints.",
        parent=kroger_node,
        critical=True
    )

    hours_text = (info.hours or "").strip() if info else ""
    if hours_text:
        hours_claim = (
            f"On {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG}), Kroger stores operate with reduced hours as described: '{hours_text}'. "
            f"Typically, they close mid‑afternoon (around 3–4 PM local time)."
        )
    else:
        hours_claim = (
            f"On {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG}), Kroger stores operate with reduced hours and typically close mid‑afternoon (around 3–4 PM local time)."
        )

    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=get_sources(info),
        additional_instruction=(
            "Consider this supported only if the official Kroger (or Kroger family) corporate source explicitly indicates reduced/shortened hours for Thanksgiving Day 2025 "
            "and an early closing time approximately in the mid‑afternoon range (about 3–4 PM local time). Minor phrasing variations are acceptable. "
            "If the provided sources do not mention reduced hours for 2025 or suggest normal evening hours, mark as not supported."
        )
    )

    # Leaf: Kroger_Optional_Official_Support (non-critical)
    support_leaf = evaluator.add_leaf(
        id="Kroger_Optional_Official_Support",
        desc="Optionally provides an official-policy-based support statement or source for Kroger's Thanksgiving Day 2025 status/hours.",
        parent=kroger_node,
        critical=False
    )

    support_claim = (
        f"This webpage is an official Kroger corporate page that explicitly states Thanksgiving Day 2025 store status and/or special hours (e.g., early closing)."
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_leaf,
        sources=get_sources(info),
        additional_instruction=(
            "Only count as supported if the domain is clearly official (e.g., kroger.com or corporate.kroger.com) and the content explicitly references Thanksgiving Day 2025 "
            "with status/hours details. Third‑party sources do not count."
        )
    )


async def verify_dollar_general(evaluator: Evaluator, parent_node, info: Optional[RetailerInfo]) -> None:
    dg_node = evaluator.add_parallel(
        id="Dollar_General",
        desc="Dollar General Thanksgiving Day 2025 status (and hours if required by constraints).",
        parent=parent_node,
        critical=False
    )

    # Leaf: Dollar_General_Open_Status (critical)
    open_leaf = evaluator.add_leaf(
        id="Dollar_General_Open_Status",
        desc="Correctly states that Dollar General stores are open on Thanksgiving Day 2025 (Nov 27, 2025), consistent with the constraints.",
        parent=dg_node,
        critical=True
    )

    norm = infer_open_closed(info.status if info else None)
    if norm == "open":
        open_claim = f"Dollar General stores are open on {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG})."
    elif norm == "closed":
        open_claim = f"Dollar General stores are closed on {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG})."
    else:
        open_claim = f"The Dollar General nationwide store status for {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG}) is clearly stated (open or closed)."

    await evaluator.verify(
        claim=open_claim,
        node=open_leaf,
        sources=get_sources(info),
        additional_instruction=(
            "Only mark as supported if at least one provided URL is an official Dollar General corporate source (e.g., dollargeneral.com or corporate/newsroom subdomains) "
            "that explicitly mentions 2025 Thanksgiving Day status. Ambiguous or non‑2025 pages should fail."
        )
    )

    # Leaf: Dollar_General_Hours_Detail (critical)
    hours_leaf = evaluator.add_leaf(
        id="Dollar_General_Hours_Detail",
        desc="Correctly provides the typical Thanksgiving Day operating hours (about 8 AM–10 PM) and notes that hours vary by location, consistent with the constraints.",
        parent=dg_node,
        critical=True
    )

    hours_text = (info.hours or "").strip() if info else ""
    if hours_text:
        hours_claim = (
            f"On {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG}), Dollar General's typical operating hours are described as '{hours_text}', "
            f"and hours vary by location."
        )
    else:
        hours_claim = (
            f"On {THANKSGIVING_SHORT} ({THANKSGIVING_DATE_LONG}), Dollar General stores are typically open about 8 AM–10 PM, and hours vary by location."
        )

    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=get_sources(info),
        additional_instruction=(
            "Treat this as supported if an official Dollar General page indicates typical Thanksgiving 2025 hours roughly around 8 AM–10 PM "
            "(allow modest variations like ±1–2 hours) and notes that hours vary by location. "
            "If sources do not provide 2025 Thanksgiving hours information, mark as not supported."
        )
    )

    # Leaf: Dollar_General_Optional_Official_Support (non-critical)
    support_leaf = evaluator.add_leaf(
        id="Dollar_General_Optional_Official_Support",
        desc="Optionally provides an official-policy-based support statement or source for Dollar General's Thanksgiving Day 2025 status/hours.",
        parent=dg_node,
        critical=False
    )

    support_claim = (
        f"This webpage is an official Dollar General corporate page that explicitly states Thanksgiving Day 2025 store status and/or typical hours (with location variability)."
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_leaf,
    sources=get_sources(info),
        additional_instruction=(
            "Only count as supported if the domain is clearly official (e.g., dollargeneral.com, news/press subdomains) and the content explicitly references Thanksgiving Day 2025 "
            "with store status and/or hours. Third‑party sources do not count."
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
    Evaluate an answer for the 2025 Thanksgiving retail operations task.
    """
    evaluator = Evaluator()

    # Root node: Use parallel aggregation. Set non-critical to allow partial scoring across retailers.
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Determine, for each of Walmart, Target, Kroger, and Dollar General, whether stores are open on Thanksgiving Day (Thu, Nov 27, 2025), using the provided constraints as the correctness conditions.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_holiday_ops(),
        template_class=HolidayOpsExtraction,
        extraction_name="holiday_ops_extraction"
    )

    # Build retailer verification subtrees
    await verify_walmart(evaluator, root, extracted.walmart)
    await verify_target(evaluator, root, extracted.target)
    await verify_kroger(evaluator, root, extracted.kroger)
    await verify_dollar_general(evaluator, root, extracted.dollar_general)

    # Return evaluation summary
    return evaluator.get_summary()