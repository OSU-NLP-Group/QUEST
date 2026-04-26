import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "carrier_performance_2025q4_2026early"
TASK_DESCRIPTION = """
For the three major US wireless carriers—Verizon, AT&T, and T-Mobile—identify which carrier leads in each of the following performance categories based on data from Q4 2025 and early 2026. Additionally, provide specific numerical values where requested:

1. Highest Total Subscriber Count: Which carrier has the most subscribers as of Q4 2025?
2. Second Place in Subscribers: Which carrier ranks second in total subscriber count?
3. Largest Market Share: Which carrier holds the highest percentage of the US wireless market?
4. Highest 5G Availability Score: Which carrier achieved the highest 5G availability score in the US?
5. Greatest 5G Coverage Percentage: Which carrier provides 5G coverage to the highest percentage of Americans?
6. Dominant 4G LTE Coverage: Which carrier's 4G LTE network covers more than 99% of the US population?
7. Customer Satisfaction Leader: Which carrier ranked highest in the 2026 J.D. Power customer satisfaction study (Volume 1)?
8. Top Customer Satisfaction Score: What was the exact satisfaction score for the highest-ranked carrier?
9. Second Customer Satisfaction Score: What was the satisfaction score for the second-ranked carrier?
10. Third Customer Satisfaction Score: What was the satisfaction score for the third-ranked carrier?
11. Network Performance Leader: Which carrier received top overall honors for national performance and reliability from RootMetrics?
12. Largest 5G Network: Which carrier is described as having the largest and fastest 5G network in the United States?
13. Verizon Subscriber Count: What is Verizon's exact subscriber count (in millions) as of Q4 2025?
14. T-Mobile Subscriber Count: What is T-Mobile's exact subscriber count (in millions) as of Q4 2025?
15. AT&T Subscriber Count: What is AT&T's exact subscriber count (in millions) as of Q4 2025?

Provide reference URLs from reliable sources to support each answer.
"""

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #

class CarrierMetricsExtraction(BaseModel):
    # Leaders and rankings
    subscriber_leader_carrier: Optional[str] = None
    subscriber_leader_sources: List[str] = Field(default_factory=list)

    subscriber_second_carrier: Optional[str] = None
    subscriber_second_sources: List[str] = Field(default_factory=list)

    market_share_leader_carrier: Optional[str] = None
    market_share_leader_sources: List[str] = Field(default_factory=list)

    fiveg_availability_leader_carrier: Optional[str] = None
    fiveg_availability_sources: List[str] = Field(default_factory=list)

    fiveg_coverage_leader_carrier: Optional[str] = None
    fiveg_coverage_sources: List[str] = Field(default_factory=list)

    lte_99_carrier: Optional[str] = None
    lte_99_sources: List[str] = Field(default_factory=list)

    # J.D. Power 2026 Volume 1 satisfaction
    jd_power_leader_carrier: Optional[str] = None
    jd_power_leader_sources: List[str] = Field(default_factory=list)

    jd_power_top_carrier: Optional[str] = None
    jd_power_top_score: Optional[str] = None
    jd_power_top_sources: List[str] = Field(default_factory=list)

    jd_power_second_carrier: Optional[str] = None
    jd_power_second_score: Optional[str] = None
    jd_power_second_sources: List[str] = Field(default_factory=list)

    jd_power_third_carrier: Optional[str] = None
    jd_power_third_score: Optional[str] = None
    jd_power_third_sources: List[str] = Field(default_factory=list)

    # RootMetrics performance
    rootmetrics_leader_carrier: Optional[str] = None
    rootmetrics_sources: List[str] = Field(default_factory=list)

    # Largest & fastest 5G network descriptor
    largest_fastest_5g_carrier: Optional[str] = None
    largest_fastest_5g_sources: List[str] = Field(default_factory=list)

    # Exact subscriber counts (Q4 2025)
    verizon_subs_millions: Optional[str] = None
    verizon_subs_sources: List[str] = Field(default_factory=list)

    tmobile_subs_millions: Optional[str] = None
    tmobile_subs_sources: List[str] = Field(default_factory=list)

    att_subs_millions: Optional[str] = None
    att_subs_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #

def prompt_extract_carrier_metrics() -> str:
    return """
Extract the following information exactly as stated in the answer, including the carrier names, any specific numbers, and the cited URLs. Only extract URLs explicitly mentioned in the answer (plain or markdown). Do not invent URLs.

Return a single JSON object with these fields (use null for any missing field; empty array when no sources are provided):

1) Leaders and rankings (carriers should be among Verizon, AT&T, or T-Mobile; minor naming variants acceptable):
- subscriber_leader_carrier
- subscriber_leader_sources (array of URLs)
- subscriber_second_carrier
- subscriber_second_sources (array of URLs)
- market_share_leader_carrier
- market_share_leader_sources (array of URLs)
- fiveg_availability_leader_carrier
- fiveg_availability_sources (array of URLs)
- fiveg_coverage_leader_carrier
- fiveg_coverage_sources (array of URLs)
- lte_99_carrier
- lte_99_sources (array of URLs)

2) J.D. Power 2026 Volume 1 (customer satisfaction) leader and scores:
- jd_power_leader_carrier
- jd_power_leader_sources (array of URLs)
- jd_power_top_carrier
- jd_power_top_score (as it appears, e.g., "827" or "827/1000")
- jd_power_top_sources (array of URLs)
- jd_power_second_carrier
- jd_power_second_score
- jd_power_second_sources (array of URLs)
- jd_power_third_carrier
- jd_power_third_score
- jd_power_third_sources (array of URLs)

3) RootMetrics performance:
- rootmetrics_leader_carrier
- rootmetrics_sources (array of URLs)

4) Largest & fastest 5G descriptor:
- largest_fastest_5g_carrier
- largest_fastest_5g_sources (array of URLs)

5) Exact subscriber counts as of Q4 2025 (extract numbers as written, keep units if present like 'million' or 'm'):
- verizon_subs_millions
- verizon_subs_sources (array of URLs)
- tmobile_subs_millions
- tmobile_subs_sources (array of URLs)
- att_subs_millions
- att_subs_sources (array of URLs)
"""


# --------------------------------------------------------------------------- #
# Helper verification builders                                                #
# --------------------------------------------------------------------------- #

async def add_leader_check(
    evaluator: Evaluator,
    parent,
    metric_id: str,
    metric_desc: str,
    carrier_value: Optional[str],
    sources: Optional[List[str]],
    claim_template: str,
    add_ins: str,
    verify_critical: bool,
):
    """
    Build a small sequential sub-tree for a 'leader' type check:
      - Existence (critical): has carrier and at least one source URL
      - Verification (leaf): claim supported by the provided source URLs
    """
    seq_node = evaluator.add_sequential(
        id=metric_id,
        desc=metric_desc,
        parent=parent,
        critical=False  # parent node kept non-critical to avoid upward constraint on children
    )

    # Normalize sources
    srcs = (sources or [])
    has_carrier = carrier_value is not None and str(carrier_value).strip() != ""
    has_sources = len(srcs) > 0

    evaluator.add_custom_node(
        result=(has_carrier and has_sources),
        id=f"{metric_id}_exists",
        desc=f"{metric_desc} — answer provides a carrier and at least one URL source",
        parent=seq_node,
        critical=True
    )

    # Build the verification leaf
    verify_leaf = evaluator.add_leaf(
        id=f"{metric_id}_supported",
        desc=f"{metric_desc} — claim is supported by cited sources",
        parent=seq_node,
        critical=verify_critical
    )

    # Prepare claim
    claim = claim_template.format(carrier=carrier_value or "[missing]")

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=srcs,
        additional_instruction=add_ins
    )


async def add_score_check(
    evaluator: Evaluator,
    parent,
    metric_id: str,
    metric_desc: str,
    carrier_value: Optional[str],
    score_value: Optional[str],
    sources: Optional[List[str]],
    add_ins: str,
    verify_critical: bool,
):
    """
    Build a sequential sub-tree to verify a specific score for a given carrier and study.
      - Existence: carrier, score, and sources must exist
      - Verification: claim includes carrier and exact score
    """
    seq_node = evaluator.add_sequential(
        id=metric_id,
        desc=metric_desc,
        parent=parent,
        critical=False
    )

    srcs = (sources or [])
    has_carrier = carrier_value is not None and str(carrier_value).strip() != ""
    has_score = score_value is not None and str(score_value).strip() != ""
    has_sources = len(srcs) > 0

    evaluator.add_custom_node(
        result=(has_carrier and has_score and has_sources),
        id=f"{metric_id}_exists",
        desc=f"{metric_desc} — answer provides carrier, score, and at least one URL source",
        parent=seq_node,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{metric_id}_supported",
        desc=f"{metric_desc} — score is supported by cited sources",
        parent=seq_node,
        critical=verify_critical
    )

    # Claim with carrier and score
    claim = (
        f"In the 2026 J.D. Power U.S. wireless customer satisfaction (Volume 1) study, "
        f"{carrier_value} received a score of {score_value}."
    )

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=srcs,
        additional_instruction=add_ins
    )


async def add_numeric_value_check(
    evaluator: Evaluator,
    parent,
    metric_id: str,
    metric_desc: str,
    claim_template: str,
    value: Optional[str],
    sources: Optional[List[str]],
    add_ins: str,
    verify_critical: bool,
):
    """
    For numeric values like subscriber counts (per carrier):
      - Existence: value and sources present
      - Verification: claim states the value for the specific carrier and timeframe
    """
    seq_node = evaluator.add_sequential(
        id=metric_id,
        desc=metric_desc,
        parent=parent,
        critical=False
    )

    srcs = (sources or [])
    has_value = value is not None and str(value).strip() != ""
    has_sources = len(srcs) > 0

    evaluator.add_custom_node(
        result=(has_value and has_sources),
        id=f"{metric_id}_exists",
        desc=f"{metric_desc} — answer provides a value and at least one URL source",
        parent=seq_node,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{metric_id}_supported",
        desc=f"{metric_desc} — value is supported by cited sources",
        parent=seq_node,
        critical=verify_critical
    )

    claim = claim_template.format(value=value or "[missing]")

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=srcs,
        additional_instruction=add_ins
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
    Evaluate an answer for the carrier performance metrics task.
    """

    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent checks
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

    # Extract structured metrics from the answer
    extracted: CarrierMetricsExtraction = await evaluator.extract(
        prompt=prompt_extract_carrier_metrics(),
        template_class=CarrierMetricsExtraction,
        extraction_name="carrier_metrics"
    )

    # Add contextual info for the evaluation record
    evaluator.add_ground_truth({
        "carriers_considered": ["Verizon", "AT&T", "T-Mobile"],
        "timeframe": "Q4 2025 and early 2026",
        "requested_items": [
            "Subscriber leader & second place",
            "Market share leader",
            "Highest 5G availability leader",
            "Greatest 5G coverage leader",
            "4G LTE > 99% coverage",
            "J.D. Power 2026 Vol.1 leader + top 3 scores",
            "RootMetrics national performance & reliability leader",
            "Exact subscriber counts as of Q4 2025 for Verizon/T-Mobile/AT&T"
        ]
    }, gt_type="task_context")

    # Build a parent grouping node (parallel) for clarity
    carrier_analysis = evaluator.add_parallel(
        id="CarrierAnalysis",
        desc="Identify which of the three major US wireless carriers (Verizon, AT&T, T-Mobile) leads in each specified performance metric",
        parent=root,
        critical=False
    )

    # -------- 1. Highest Total Subscriber Count (Critical) --------------- #
    await add_leader_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="SubscriberCount",
        metric_desc="Highest total subscriber count as of Q4 2025",
        carrier_value=extracted.subscriber_leader_carrier,
        sources=extracted.subscriber_leader_sources,
        claim_template="As of Q4 2025 (or early 2026 reporting), {carrier} has the highest total number of wireless subscribers among Verizon, AT&T, and T-Mobile in the United States.",
        add_ins="Verify the timeframe (Q4 2025 or early 2026). Ensure the metric refers to total wireless subscribers (not lines or connections). Prefer reputable industry reports/earnings releases.",
        verify_critical=True
    )

    # -------- 2. Second Place in Subscribers (Non-Critical) -------------- #
    await add_leader_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="SecondPlaceSubscribers",
        metric_desc="Second place in total subscriber count as of Q4 2025",
        carrier_value=extracted.subscriber_second_carrier,
        sources=extracted.subscriber_second_sources,
        claim_template="As of Q4 2025 (or early 2026 reporting), {carrier} ranks second in total wireless subscribers among Verizon, AT&T, and T-Mobile in the United States.",
        add_ins="Verify that the page explicitly or implicitly establishes a second-place ranking in that timeframe.",
        verify_critical=False
    )

    # -------- 3. Largest Market Share (Critical) ------------------------- #
    await add_leader_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="MarketShareLeader",
        metric_desc="Largest market share in the U.S. wireless market",
        carrier_value=extracted.market_share_leader_carrier,
        sources=extracted.market_share_leader_sources,
        claim_template="As of late 2025 or early 2026, {carrier} holds the largest share of the U.S. wireless market.",
        add_ins="Look for explicit market share comparisons or a ranked list by percentage for the timeframe around Q4 2025/early 2026.",
        verify_critical=True
    )

    # -------- 4. Highest 5G Availability Score (Critical) ---------------- #
    await add_leader_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="FiveGAvailability",
        metric_desc="Highest 5G availability score in the U.S.",
        carrier_value=extracted.fiveg_availability_leader_carrier,
        sources=extracted.fiveg_availability_sources,
        claim_template="{carrier} achieved the highest 5G availability score in the United States around Q4 2025/early 2026.",
        add_ins="Often measured by Opensignal; confirm the source and timeframe. Allow minor naming variations (e.g., 'Availability').",
        verify_critical=True
    )

    # -------- 5. Greatest 5G Coverage Percentage (Critical) -------------- #
    await add_leader_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="FiveGCoveragePercentage",
        metric_desc="Highest percentage of Americans covered by 5G",
        carrier_value=extracted.fiveg_coverage_leader_carrier,
        sources=extracted.fiveg_coverage_sources,
        claim_template="In the U.S., {carrier} provides 5G coverage to the highest percentage of Americans (population coverage) in the timeframe around Q4 2025/early 2026.",
        add_ins="Confirm that the statement or chart supports 'highest population coverage' among national carriers.",
        verify_critical=True
    )

    # -------- 6. Dominant 4G LTE Coverage >99% (Critical) ---------------- #
    await add_leader_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="FourGLTECoverage",
        metric_desc="4G LTE network covers more than 99% of the U.S. population",
        carrier_value=extracted.lte_99_carrier,
        sources=extracted.lte_99_sources,
        claim_template="{carrier}'s 4G LTE network covers more than 99% of the U.S. population.",
        add_ins="The page should explicitly mention '>99%' or a percentage ≥ 99%. Official carrier pages are acceptable if explicit.",
        verify_critical=True
    )

    # -------- 7. Customer Satisfaction Leader (Critical) ----------------- #
    await add_leader_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="CustomerSatisfactionLeader",
        metric_desc="Highest rank in 2026 J.D. Power customer satisfaction (Volume 1)",
        carrier_value=extracted.jd_power_leader_carrier,
        sources=extracted.jd_power_leader_sources,
        claim_template="In the 2026 J.D. Power U.S. wireless customer satisfaction (Volume 1) study, {carrier} ranked highest overall.",
        add_ins="Allow minor naming variants such as 'U.S. Wireless Customer Care Study' if clearly the 2026 Volume 1 satisfaction study for national carriers.",
        verify_critical=True
    )

    # -------- 8. Top Satisfaction Score (Non-Critical) -------------------- #
    await add_score_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="CustomerSatisfactionScore_First",
        metric_desc="Exact score for the top-ranked carrier in 2026 J.D. Power (Vol.1)",
        carrier_value=extracted.jd_power_top_carrier,
        score_value=extracted.jd_power_top_score,
        sources=extracted.jd_power_top_sources,
        add_ins="Confirm the exact score for the top-ranked carrier in the 2026 Volume 1 study; allow minor formatting like '/1000'.",
        verify_critical=False
    )

    # -------- 9. Second Satisfaction Score (Non-Critical) ----------------- #
    await add_score_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="CustomerSatisfactionScore_Second",
        metric_desc="Exact score for the second-ranked carrier in 2026 J.D. Power (Vol.1)",
        carrier_value=extracted.jd_power_second_carrier,
        score_value=extracted.jd_power_second_score,
        sources=extracted.jd_power_second_sources,
        add_ins="Confirm the second-place score in the 2026 Volume 1 study; verify that it's for the correct carrier.",
        verify_critical=False
    )

    # -------- 10. Third Satisfaction Score (Non-Critical) ----------------- #
    await add_score_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="CustomerSatisfactionScore_Third",
        metric_desc="Exact score for the third-ranked carrier in 2026 J.D. Power (Vol.1)",
        carrier_value=extracted.jd_power_third_carrier,
        score_value=extracted.jd_power_third_score,
        sources=extracted.jd_power_third_sources,
        add_ins="Confirm the third-place score in the 2026 Volume 1 study; verify correct carrier association.",
        verify_critical=False
    )

    # -------- 11. RootMetrics Performance Leader (Critical) -------------- #
    await add_leader_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="NetworkPerformanceLeader",
        metric_desc="Top national performance & reliability (RootMetrics)",
        carrier_value=extracted.rootmetrics_leader_carrier,
        sources=extracted.rootmetrics_sources,
        claim_template="{carrier} received top overall honors for national performance and reliability from RootMetrics (latest awards around 2H 2025 or early 2026).",
        add_ins="Look for 'RootMetrics RootScore Awards' national overall/performance/reliability leader. The timeframe should be near 2H 2025 or early 2026.",
        verify_critical=True
    )

    # -------- 12. Largest & Fastest 5G Network (Critical) ---------------- #
    await add_leader_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="FiveGNetworkSize",
        metric_desc="Carrier described as having the largest and fastest 5G network in the U.S.",
        carrier_value=extracted.largest_fastest_5g_carrier,
        sources=extracted.largest_fastest_5g_sources,
        claim_template="{carrier} is described as having the largest and fastest 5G network in the United States.",
        add_ins="The page should explicitly state 'largest and fastest 5G network' (or close paraphrase) in the U.S.",
        verify_critical=True
    )

    # -------- 13. Verizon Subscriber Count Q4 2025 (Non-Critical) -------- #
    await add_numeric_value_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="SubscriberCount_Verizon",
        metric_desc="Verizon exact subscriber count as of Q4 2025",
        claim_template="As of Q4 2025, Verizon had {value} subscribers (in millions if unit specified).",
        value=extracted.verizon_subs_millions,
        sources=extracted.verizon_subs_sources,
        add_ins="Confirm the figure refers to total wireless subscribers in Q4 2025 (or the earnings release for that quarter). Allow minor rounding.",
        verify_critical=False
    )

    # -------- 14. T-Mobile Subscriber Count Q4 2025 (Non-Critical) ------- #
    await add_numeric_value_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="SubscriberCount_TMobile",
        metric_desc="T-Mobile exact subscriber count as of Q4 2025",
        claim_template="As of Q4 2025, T-Mobile had {value} subscribers (in millions if unit specified).",
        value=extracted.tmobile_subs_millions,
        sources=extracted.tmobile_subs_sources,
        add_ins="Confirm the figure refers to total wireless subscribers in Q4 2025. Allow minor rounding.",
        verify_critical=False
    )

    # -------- 15. AT&T Subscriber Count Q4 2025 (Non-Critical) ----------- #
    await add_numeric_value_check(
        evaluator=evaluator,
        parent=carrier_analysis,
        metric_id="SubscriberCount_ATT",
        metric_desc="AT&T exact subscriber count as of Q4 2025",
        claim_template="As of Q4 2025, AT&T had {value} subscribers (in millions if unit specified).",
        value=extracted.att_subs_millions,
        sources=extracted.att_subs_sources,
        add_ins="Confirm the figure refers to total wireless subscribers in Q4 2025. Allow minor rounding.",
        verify_critical=False
    )

    # Return the evaluation summary
    return evaluator.get_summary()