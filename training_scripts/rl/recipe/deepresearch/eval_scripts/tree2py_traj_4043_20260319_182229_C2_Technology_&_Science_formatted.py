import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "verizon_esim_international_setup"
TASK_DESCRIPTION = """
You have an unlocked smartphone that you want to activate on Verizon using eSIM technology. You are planning a 5-day trip to Canada and need to maintain cellular service during your visit. What are the three essential device requirements that must be met for Verizon eSIM activation, and which Verizon international roaming option would be the most cost-effective for your 5-day Canada trip? Calculate the total cost for all 5 days of service in Canada.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementEvidence(BaseModel):
    mentioned: Optional[bool] = None  # Whether this requirement is explicitly stated in the answer
    snippet: Optional[str] = None     # Exact quote or the closest phrasing from the answer, if available
    sources: List[str] = Field(default_factory=list)  # URLs the answer cites for this requirement


class DeviceRequirementsExtraction(BaseModel):
    unlocked_requirement: Optional[RequirementEvidence] = None
    esim_support_requirement: Optional[RequirementEvidence] = None
    network_compat_requirement: Optional[RequirementEvidence] = None


class PlanSelectionExtraction(BaseModel):
    plan_name: Optional[str] = None                 # e.g., "TravelPass"
    daily_cost: Optional[str] = None                # e.g., "$5/day" or "$10 per day"
    total_cost_5_days: Optional[str] = None         # e.g., "$25"
    plan_sources: List[str] = Field(default_factory=list)  # URLs the answer cites for the plan and pricing


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_device_requirements() -> str:
    return """
    From the answer, extract evidence for the three essential device requirements for Verizon eSIM activation.
    Map the answer content to the following three canonical requirements:
    1) unlocked_requirement: The device must be unlocked (i.e., not carrier-locked to another provider).
    2) esim_support_requirement: The device must support eSIM technology.
    3) network_compat_requirement: The device must be compatible with Verizon's network, typically verified via an IMEI/MEID check.

    For each requirement, provide:
    - mentioned (boolean): whether the answer explicitly states this requirement.
    - snippet (string): the exact sentence or closest phrase from the answer that corresponds to this requirement (if present).
    - sources (array of strings): the URLs the answer cites that specifically support this requirement (if any). Extract only actual URLs present in the answer.

    Return a JSON object with fields:
    {
      "unlocked_requirement": { "mentioned": bool | null, "snippet": string | null, "sources": string[] },
      "esim_support_requirement": { "mentioned": bool | null, "snippet": string | null, "sources": string[] },
      "network_compat_requirement": { "mentioned": bool | null, "snippet": string | null, "sources": string[] }
    }

    Important:
    - Do not invent URLs. If there are no URLs for a requirement, return an empty array for sources.
    - If the answer does not mention a requirement, set mentioned to false and snippet to null.
    """


def prompt_extract_plan_selection() -> str:
    return """
    From the answer, extract the Verizon international roaming option selected for a 5-day trip to Canada and the related costs.

    Provide:
    - plan_name: The name of the plan (e.g., "TravelPass", "International Monthly Plan", etc.).
    - daily_cost: The per-day price in Canada as stated in the answer (e.g., "$5/day" or "$10 per day"). If not provided, set to null.
    - total_cost_5_days: The total price for 5 full days of use in Canada as calculated in the answer (e.g., "$25"). If not provided, set to null.
    - plan_sources: All URLs the answer cites that support this plan and pricing. Extract only actual URLs present in the answer.

    Return a JSON object:
    {
      "plan_name": string | null,
      "daily_cost": string | null,
      "total_cost_5_days": string | null,
      "plan_sources": string[]
    }

    Notes:
    - Do not invent URLs. If the answer does not include any, return an empty array.
    - If multiple plans are discussed, extract the one the answer ultimately recommends as most cost-effective for a 5-day Canada trip.
    """


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0 and any(isinstance(u, str) and u.strip() for u in urls)


async def _verify_or_fail_due_to_missing_sources(
    evaluator: Evaluator,
    *,
    node,
    claim: str,
    sources: Optional[List[str]] | Optional[str],
    additional_instruction: str
) -> bool:
    """
    Verify a claim by URLs if sources are available; otherwise mark node as failed to
    enforce source-grounded verification for factual checks.
    """
    # Normalize missing sources handling
    if isinstance(sources, list):
        src_ok = _has_sources(sources)
    else:
        src_ok = isinstance(sources, str) and bool(sources.strip())

    if not src_ok:
        node.score = 0.0
        node.status = "failed"
        return False

    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=additional_instruction
    )


async def _verify_device_requirement(
    evaluator: Evaluator,
    parent_node,
    leaf_id: str,
    leaf_desc: str,
    evidence: Optional[RequirementEvidence],
    requirement_natural_text: str,
    extra_guidance: str
) -> None:
    """
    Create a leaf node for a specific device requirement and verify that:
    1) The answer explicitly includes this requirement; and
    2) The cited source(s) support it.

    This leverages the Verifier URL prompt, which includes the full answer as context.
    """
    leaf = evaluator.add_leaf(
        id=leaf_id,
        desc=leaf_desc,
        parent=parent_node,
        critical=True
    )

    # Build a claim that checks both "mentioned in answer" and "supported by sources"
    mentioned_text = "explicitly lists" if (evidence and evidence.mentioned) else "explicitly lists"
    # We always phrase positively; if the answer didn't actually include it,
    # the verifier (which sees the full answer as context) should mark it unsupported/incorrect.

    claim = (
        f"In the answer, the device requirements for Verizon eSIM activation {mentioned_text} that: {requirement_natural_text}. "
        f"This requirement is a legitimate prerequisite according to the cited source(s)."
    )

    add_ins = (
        "Two aspects to verify:\n"
        "1) Check the provided answer text (included in the context) to confirm it explicitly states this requirement "
        f"(allow synonymous wording; e.g., {extra_guidance}).\n"
        "2) Using the cited URL(s), confirm that Verizon or an authoritative page actually states this requirement for "
        "Bring Your Own Device (BYOD) or eSIM activation. If sources are missing or irrelevant, mark as not supported."
    )

    sources = evidence.sources if evidence else []
    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        node=leaf,
        claim=claim,
        sources=sources,
        additional_instruction=add_ins
    )


async def _verify_plan_type(
    evaluator: Evaluator,
    parent_node,
    plan: PlanSelectionExtraction
) -> None:
    """
    Verify that the selected plan is appropriate for a 5-day trip to Canada:
    - It works in Canada, and
    - It is structured in a way that fits short trips (e.g., per-day pricing).
    """
    leaf = evaluator.add_leaf(
        id="Plan_Type_Identification",
        desc="Correct international plan option identified based on Canada travel and duration",
        parent=parent_node,
        critical=True
    )

    if not plan or not plan.plan_name or not plan.plan_name.strip():
        leaf.score = 0.0
        leaf.status = "failed"
        return

    claim = (
        f"In the answer, the selected Verizon international option is '{plan.plan_name}' for a 5-day trip to Canada. "
        "Based on the cited Verizon (or authoritative) page(s), this option is valid for use in Canada and is appropriate "
        "for short trips because it charges per day (or otherwise clearly fits a 5-day stay)."
    )
    add_ins = (
        "Confirm from the cited page(s) that the named plan explicitly includes Canada (or works when roaming in Canada). "
        "Also confirm that it is structured for short stays (e.g., per-day pricing such as TravelPass). "
        "You do not need to prove it is cheaper than every alternative; just ensure it is a correct and appropriate option "
        "for a 5-day Canada trip per Verizon documentation."
    )

    sources = plan.plan_sources if plan else []
    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        node=leaf,
        claim=claim,
        sources=sources,
        additional_instruction=add_ins
    )


async def _verify_cost_calculation(
    evaluator: Evaluator,
    parent_node,
    plan: PlanSelectionExtraction
) -> None:
    """
    Verify that the total cost calculation for 5 days matches the plan's per-day rate per the cited sources.
    """
    leaf = evaluator.add_leaf(
        id="Cost_Calculation",
        desc="Accurate total cost calculated for 5-day Canada trip",
        parent=parent_node,
        critical=True
    )

    # Require plan name, daily cost, total 5-day cost, and sources
    if not plan or not plan.plan_name or not plan.daily_cost or not plan.total_cost_5_days:
        leaf.score = 0.0
        leaf.status = "failed"
        return

    claim = (
        f"According to the cited documentation, '{plan.plan_name}' costs {plan.daily_cost} per day in Canada. "
        f"Therefore, the total for 5 days is {plan.total_cost_5_days}. "
        "Verify that the per-day price on the source page matches the stated daily cost and that 5 days of service equals the stated total."
    )
    add_ins = (
        "First, confirm the daily price on the cited page(s) for Canada. Then verify the arithmetic: "
        "5 × (daily price) == total. Allow reasonable formatting differences (e.g., '$5/day' vs '$5 per day'). "
        "If taxes/fees are mentioned, focus on the core plan charge unless the answer explicitly includes extra fees."
    )

    sources = plan.plan_sources if plan else []
    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        node=leaf,
        claim=claim,
        sources=sources,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Verizon eSIM device requirements and Canada plan selection task.
    """
    # Initialize the evaluator and root
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

    # Create a critical top-level node to mirror rubric root since Evaluator root is non-critical by design
    top_node = evaluator.add_parallel(
        id="Verizon_eSIM_International_Setup",
        desc="Successfully determine device eligibility for Verizon eSIM activation and identify appropriate international plan for Canada travel",
        parent=root,
        critical=True
    )

    # Run extractions in parallel
    device_req_task = evaluator.extract(
        prompt=prompt_extract_device_requirements(),
        template_class=DeviceRequirementsExtraction,
        extraction_name="device_requirements"
    )
    plan_task = evaluator.extract(
        prompt=prompt_extract_plan_selection(),
        template_class=PlanSelectionExtraction,
        extraction_name="plan_selection"
    )

    device_requirements, plan_selection = await asyncio.gather(device_req_task, plan_task)

    # ---------------- Device Compatibility Assessment (parallel, all critical) ---------------- #
    device_node = evaluator.add_parallel(
        id="Device_Compatibility_Assessment",
        desc="Verify device meets all requirements for Verizon eSIM activation",
        parent=top_node,
        critical=True
    )

    # Device must be unlocked
    await _verify_device_requirement(
        evaluator=evaluator,
        parent_node=device_node,
        leaf_id="Device_Unlock_Requirement",
        leaf_desc="Device must be unlocked from any carrier other than Verizon",
        evidence=device_requirements.unlocked_requirement if device_requirements else None,
        requirement_natural_text="the phone must be unlocked (i.e., not carrier‑locked to another provider) to activate on Verizon using eSIM",
        extra_guidance="accept synonyms like 'unlocked phone', 'carrier-unlocked', 'SIM lock removed', 'not locked to another carrier'"
    )

    # Device must support eSIM
    await _verify_device_requirement(
        evaluator=evaluator,
        parent_node=device_node,
        leaf_id="eSIM_Technology_Support",
        leaf_desc="Device must support eSIM technology",
        evidence=device_requirements.esim_support_requirement if device_requirements else None,
        requirement_natural_text="the phone must support eSIM technology to activate on Verizon using eSIM",
        extra_guidance="phrases like 'supports eSIM', 'embedded SIM compatible', 'eSIM-capable' are acceptable"
    )

    # Device must be compatible with Verizon network (IMEI verification)
    await _verify_device_requirement(
        evaluator=evaluator,
        parent_node=device_node,
        leaf_id="Network_Compatibility_Check",
        leaf_desc="Device must be compatible with Verizon network (IMEI verification required)",
        evidence=device_requirements.network_compat_requirement if device_requirements else None,
        requirement_natural_text="the device must be compatible with Verizon’s network, typically verified via an IMEI/MEID check",
        extra_guidance="look for mentions of Verizon's compatibility/IMEI checker or instructions to verify compatibility via IMEI/MEID"
    )

    # ---------------- International Plan Selection for Canada (parallel, all critical) -------- #
    plan_node = evaluator.add_parallel(
        id="International_Plan_Selection_Canada",
        desc="Identify the most cost-effective international roaming option for a 5-day trip to Canada",
        parent=top_node,
        critical=True
    )

    await _verify_plan_type(
        evaluator=evaluator,
        parent_node=plan_node,
        plan=plan_selection
    )

    await _verify_cost_calculation(
        evaluator=evaluator,
        parent_node=plan_node,
        plan=plan_selection
    )

    # Return evaluation summary
    return evaluator.get_summary()