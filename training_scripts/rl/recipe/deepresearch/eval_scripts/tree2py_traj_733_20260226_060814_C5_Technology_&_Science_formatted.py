import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "premium_wireless_top_tier_4lines"
TASK_DESCRIPTION = """You are planning to select a premium wireless plan for your family of 4 lines and want to compare the top-tier offerings from the three major US carriers. Identify one premium unlimited plan from each carrier (AT&T, Verizon, and T-Mobile) that meets ALL of the following requirements:

1. Must be the carrier's premium/top-tier unlimited plan offering
2. Must include access to the carrier's fastest 5G network technology (5G+, 5G Ultra Wideband, or Ultra Capacity 5G)
3. Must include at least 50GB of mobile hotspot data per month per line
4. Must include unlimited talk, text, and data in Mexico and Canada without additional roaming charges
5. Must support satellite connectivity features (satellite messaging or emergency SOS capability)
6. Must include at least one streaming service perk or subscription benefit
7. Must include international roaming capabilities beyond North America
8. Must provide the total monthly cost for 4 lines (with autopay discounts applied, before taxes/fees)

For each of the three plans, provide:
- The specific plan name
- A brief description of how it meets each requirement
- The total monthly cost for 4 lines
- A direct link to the carrier's official webpage for that specific plan
"""

# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class CarrierPlan(BaseModel):
    carrier: Optional[str] = None
    plan_name: Optional[str] = None
    plan_url: Optional[str] = None  # Official plan page URL if provided in answer
    total_monthly_cost_4_lines: Optional[str] = None  # As stated in answer (autopay applied, before taxes/fees)

    # Evidence snippets (verbatim text from the answer if available)
    fast_5g_term: Optional[str] = None            # e.g., "5G+", "5G Ultra Wideband", "Ultra Capacity 5G"
    hotspot_desc: Optional[str] = None            # text mentioning hotspot quantity (e.g., "75GB hotspot")
    mexico_canada_desc: Optional[str] = None      # text stating unlimited in Mexico/Canada
    satellite_desc: Optional[str] = None          # text mentioning satellite messaging/emergency SOS
    streaming_desc: Optional[str] = None          # text mentioning included streaming perk
    intl_desc: Optional[str] = None               # text mentioning roaming beyond North America

    # All additional URLs explicitly present in the answer that support the plan/features
    supporting_urls: List[str] = Field(default_factory=list)


class ThreePlansExtraction(BaseModel):
    att: Optional[CarrierPlan] = None
    verizon: Optional[CarrierPlan] = None
    tmobile: Optional[CarrierPlan] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_three_plans() -> str:
    return """
    Extract exactly one premium/top-tier UNLIMITED plan from each of the three major US carriers mentioned in the answer: AT&T, Verizon, and T-Mobile.
    If multiple plans per carrier are mentioned, pick the highest-tier unlimited plan the answer actually uses for evaluation.

    For each carrier, return the following fields (use null if missing):
    - carrier: One of "AT&T", "Verizon", or "T-Mobile".
    - plan_name: The specific plan name stated in the answer.
    - plan_url: The direct official carrier URL for that plan if present in the answer (e.g., att.com, verizon.com, t-mobile.com).
    - total_monthly_cost_4_lines: The total monthly cost for 4 lines WITH autopay discounts applied, BEFORE taxes/fees, exactly as shown in the answer (keep the currency symbol/text string).
    - fast_5g_term: The fastest 5G branding term used for that carrier as presented in the answer (e.g., "5G+", "5G Ultra Wideband", or "Ultra Capacity 5G"), if present in the answer.
    - hotspot_desc: The text snippet from the answer that mentions the hotspot allowance.
    - mexico_canada_desc: The text snippet that mentions unlimited talk/text/data in Mexico & Canada without extra charges.
    - satellite_desc: The text snippet that mentions satellite messaging or emergency SOS support.
    - streaming_desc: The text snippet that mentions an included streaming service or subscription benefit.
    - intl_desc: The text snippet that mentions roaming beyond North America.
    - supporting_urls: An array of ALL additional URLs explicitly present in the answer that support features/pricing/roaming/coverage for that plan. Include only actual URLs found in the answer text (including markdown links). If the same URL appears multiple times, include it once. Prefer official carrier URLs when available, but include whatever URLs the answer provided.

    Return a JSON object with three top-level objects: "att", "verizon", and "tmobile", each following the CarrierPlan schema.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(plan: Optional[CarrierPlan]) -> List[str]:
    """Combine plan_url and supporting_urls into a unique list while preserving order."""
    if plan is None:
        return []
    seen = set()
    result: List[str] = []
    if plan.plan_url and plan.plan_url not in seen:
        seen.add(plan.plan_url)
        result.append(plan.plan_url)
    for url in plan.supporting_urls or []:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


async def _verify_with_sources_or_fail(
    evaluator: Evaluator,
    node_id: str,
    desc: str,
    parent,
    claim: str,
    sources: List[str],
    add_ins: str,
    critical: bool = True,
) -> bool:
    """Create a leaf node and verify if sources exist; otherwise mark as failed."""
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    if not sources:
        leaf.score = 0.0
        leaf.status = "failed"
        return False
    return await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=add_ins,
    )


# --------------------------------------------------------------------------- #
# Verification logic per carrier                                              #
# --------------------------------------------------------------------------- #
async def verify_carrier_plan(
    evaluator: Evaluator,
    parent,
    plan: Optional[CarrierPlan],
    id_prefix: str,
    brand_name: str,
    fastest_5g_term_hint: str,
    official_domain_hint: str,
):
    """
    Build the verification subtree for a single carrier's plan.

    Leaves (all critical under the carrier node):
    - identity: premium/top-tier unlimited plan from the correct carrier
    - 5g_access: includes access to fastest 5G branding (5G+/UWB/UC)
    - hotspot: >= 50GB hotspot per line per month
    - north_america: unlimited talk/text/data in Mexico & Canada at no extra charge
    - satellite: supports satellite messaging/emergency SOS capability (plan-level support)
    - streaming: includes at least one streaming service perk
    - international: includes roaming beyond North America
    - pricing: total monthly cost for 4 lines (autopay applied, before taxes/fees)
    - url: direct official plan page URL
    """
    # Parent node for this carrier (parallel aggregation)
    plan_node = evaluator.add_parallel(
        id=f"{id_prefix}_plan",
        desc=f"Identify a premium unlimited plan from {brand_name} that meets all requirements",
        parent=parent,
        critical=False,
    )

    sources_all = _combine_sources(plan)
    plan_name = (plan.plan_name or "").strip()
    total_4_lines = (plan.total_monthly_cost_4_lines or "").strip()

    # 1) Identity: premium/top-tier unlimited from correct carrier
    identity_claim = (
        f"The plan named '{plan_name}' is an unlimited plan from {brand_name} and is the carrier's "
        f"premium or top-tier unlimited offering."
        if plan_name
        else f"This is {brand_name}'s premium or top-tier unlimited plan."
    )
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id=f"{id_prefix}_identity",
        desc=f"The plan must be from {brand_name} and must be their premium/top-tier unlimited offering",
        parent=plan_node,
        claim=identity_claim,
        sources=sources_all,
        add_ins=(
            "Verify on the official plan or carrier pages that this is an UNLIMITED plan and the most premium/top-tier offering. "
            "Accept phrasing such as 'top-tier', 'most premium', 'most advanced', or 'our best plan'."
        ),
        critical=True,
    )

    # 2) Fastest 5G access
    access_claim = (
        f"This plan includes access to {brand_name}'s fastest 5G network technology, such as {fastest_5g_term_hint}."
    )
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id=f"{id_prefix}_5g_access",
        desc=f"The plan must include access to {brand_name}'s fastest 5G network ({fastest_5g_term_hint} or equivalent)",
        parent=plan_node,
        claim=access_claim,
        sources=sources_all,
        add_ins=(
            f"Look for branding such as '{fastest_5g_term_hint}', 'mmWave', 'C-Band high-capacity', or 'Ultra/Plus/UC' wording on official pages. "
            "The page should clearly state this plan can use the carrier's fastest 5G tier, not just generic 5G."
        ),
        critical=True,
    )

    # 3) Hotspot >= 50GB per line
    hotspot_claim = (
        "This plan includes at least 50 GB of high-speed mobile hotspot data per line per month."
    )
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id=f"{id_prefix}_hotspot",
        desc="The plan must include at least 50GB of mobile hotspot data per month per line",
        parent=plan_node,
        claim=hotspot_claim,
        sources=sources_all,
        add_ins=(
            "Confirm a hotspot allowance of 50GB or more per line, per month (high-speed). "
            "If multiple tiers are listed, use the high-speed hotspot amount before throttling."
        ),
        critical=True,
    )

    # 4) Unlimited in Mexico & Canada (no extra fees)
    na_claim = (
        "This plan includes unlimited talk, text, and data in Mexico and Canada without additional roaming charges."
    )
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id=f"{id_prefix}_north_america",
        desc="The plan must include unlimited talk, text, and data coverage in Mexico and Canada without additional roaming fees",
        parent=plan_node,
        claim=na_claim,
        sources=sources_all,
        add_ins=(
            "Look for 'included', 'at no extra cost', or similar language indicating unlimited usage in Mexico and Canada. "
            "Minor speed reductions may apply but should still be included without extra roaming fees."
        ),
        critical=True,
    )

    # 5) Satellite connectivity support
    satellite_claim = (
        "This plan supports satellite connectivity features such as satellite messaging or emergency SOS via satellite."
    )
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id=f"{id_prefix}_satellite",
        desc="The plan must support satellite connectivity features for emergency communication or messaging",
        parent=plan_node,
        claim=satellite_claim,
        sources=sources_all,
        add_ins=(
            "Verify on official carrier pages that satellite messaging or emergency SOS via satellite is supported with this plan. "
            "It may be described as 'satellite messaging', 'emergency SOS via satellite', or similar. "
            "The capability should be supported as part of the plan/carrier offering (not only as a separate paid third-party service)."
        ),
        critical=True,
    )

    # 6) Streaming perk included
    streaming_claim = (
        "This plan includes at least one streaming service perk or subscription benefit as part of the plan."
    )
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id=f"{id_prefix}_streaming",
        desc="The plan must include or offer at least one streaming service benefit (such as Netflix, Hulu, Disney+, Max, or similar)",
        parent=plan_node,
        claim=streaming_claim,
        sources=sources_all,
        add_ins=(
            "Accept 'included' or 'on us' streaming subscriptions (e.g., Netflix, Hulu, Disney+, Max, Apple TV+, etc.). "
            "Do NOT count mere discounts or optional paid add-ons unless explicitly included at no extra cost on this plan."
        ),
        critical=True,
    )

    # 7) International roaming beyond North America
    intl_claim = (
        "This plan includes international roaming capabilities in destinations beyond Mexico and Canada."
    )
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id=f"{id_prefix}_international",
        desc="The plan must include international roaming capabilities beyond North America (Mexico/Canada)",
        parent=plan_node,
        claim=intl_claim,
        sources=sources_all,
        add_ins=(
            "Look for included day passes, included data/talk/text in many international destinations, or similar "
            "benefits beyond Mexico/Canada. The inclusion should be part of the plan, not purely pay-per-use."
        ),
        critical=True,
    )

    # 8) Pricing for 4 lines total (autopay applied, before taxes/fees)
    price_node = evaluator.add_leaf(
        id=f"{id_prefix}_pricing",
        desc="Provide the total monthly cost for 4 lines on this plan (including any autopay discounts but before taxes and fees)",
        parent=plan_node,
        critical=True,
    )
    if not sources_all or not total_4_lines:
        price_node.score = 0.0
        price_node.status = "failed"
    else:
        price_claim = (
            f"The total monthly cost for 4 lines with autopay discounts applied, before taxes and fees, is {total_4_lines}."
        )
        await evaluator.verify(
            claim=price_claim,
            node=price_node,
            sources=sources_all,
            additional_instruction=(
                "Use the official plan page and related official pricing pages to verify. "
                "If only per-line pricing is shown, compute the 4-line total (e.g., 4 × per-line rate after autopay). "
                "Allow small rounding inconsistencies (e.g., within $1)."
            ),
        )

    # 9) Official plan URL validity
    url_node = evaluator.add_leaf(
        id=f"{id_prefix}_url",
        desc=f"Provide a direct URL to {brand_name}'s official page documenting this specific plan and its features",
        parent=plan_node,
        critical=True,
    )
    if not plan or not plan.plan_url:
        url_node.score = 0.0
        url_node.status = "failed"
    else:
        url_claim = (
            f"This URL is an official {brand_name} webpage for the specific plan '{plan_name}', documenting the plan's features and pricing."
            if plan_name
            else f"This URL is an official {brand_name} webpage for the specific plan, documenting the plan's features and pricing."
        )
        await evaluator.verify(
            claim=url_claim,
            node=url_node,
            sources=plan.plan_url,
            additional_instruction=(
                f"Confirm the URL is on the official domain (e.g., contains '{official_domain_hint}'), and that the page clearly "
                "describes this specific plan (name, features, pricing). Non-official or third-party sites should not pass."
            ),
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
    Evaluate an answer for the premium top-tier wireless plans task.
    """
    # Initialize evaluator (root is non-critical to comply with framework constraints on critical parents)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates carrier subtrees in parallel
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

    # Extract structured info for the three carriers
    extracted = await evaluator.extract(
        prompt=prompt_extract_three_plans(),
        template_class=ThreePlansExtraction,
        extraction_name="three_premium_plans",
    )

    # Build three parallel subtrees (AT&T, Verizon, T-Mobile)
    await verify_carrier_plan(
        evaluator=evaluator,
        parent=root,
        plan=extracted.att if extracted else None,
        id_prefix="carrier_1",
        brand_name="AT&T",
        fastest_5g_term_hint="5G+",
        official_domain_hint="att.com",
    )

    await verify_carrier_plan(
        evaluator=evaluator,
        parent=root,
        plan=extracted.verizon if extracted else None,
        id_prefix="carrier_2",
        brand_name="Verizon",
        fastest_5g_term_hint="5G Ultra Wideband",
        official_domain_hint="verizon.com",
    )

    await verify_carrier_plan(
        evaluator=evaluator,
        parent=root,
        plan=extracted.tmobile if extracted else None,
        id_prefix="carrier_3",
        brand_name="T-Mobile",
        fastest_5g_term_hint="Ultra Capacity 5G",
        official_domain_hint="t-mobile.com",
    )

    # Return structured summary
    return evaluator.get_summary()