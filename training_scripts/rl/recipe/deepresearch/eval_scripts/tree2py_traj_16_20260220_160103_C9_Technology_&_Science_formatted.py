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
TASK_ID = "US_Mobile_Carrier_Analysis_2025_2026"
TASK_DESCRIPTION = (
    "Based on the network performance data from Opensignal's January 2026 Mobile Network Experience Report "
    "(covering September 1 - November 29, 2025) and RootMetrics' second half 2025 (2H 2025) State of the Mobile Union Report, "
    "as well as news reports about network incidents in January 2026, identify three major US mobile carriers that meet the specified criteria. "
    "For each identified carrier, provide the carrier's name, a brief description explaining how it meets the criteria, "
    "and URLs to the Opensignal January 2026 report, the RootMetrics 2H 2025 report, and (for Carrier 2 only) a news article documenting the January 14, 2026 network outage."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CarrierInfo(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    opensignal_url: Optional[str] = None
    rootmetrics_url: Optional[str] = None
    outage_news_url: Optional[str] = None  # Only required for Carrier 2; others can be null


class CarrierExtraction(BaseModel):
    carrier1: Optional[CarrierInfo] = None
    carrier2: Optional[CarrierInfo] = None
    carrier3: Optional[CarrierInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_carriers() -> str:
    return """
    You must extract structured information about three specific US mobile carriers referenced in the answer.
    Extract exactly these fields for each carrier:
    - name: The carrier's name (e.g., T-Mobile, Verizon, AT&T)
    - description: A brief explanation from the answer of how the carrier meets the specified criteria
    - opensignal_url: A URL to Opensignal's January 2026 Mobile Network Experience report that the answer cites for this carrier
    - rootmetrics_url: A URL to RootMetrics' 2H 2025 State of the Mobile Union report that the answer cites for this carrier
    - outage_news_url: For Carrier 2 only, a URL to a news article documenting the January 14, 2026 outage; for Carrier 1 and Carrier 3, set to null

    Return a JSON object with the following top-level keys:
    - carrier1
    - carrier2
    - carrier3

    Each of carrier1/carrier2/carrier3 must be an object with:
    {
      "name": string | null,
      "description": string | null,
      "opensignal_url": string | null,
      "rootmetrics_url": string | null,
      "outage_news_url": string | null
    }

    IMPORTANT:
    - Extract only URLs explicitly present in the answer. If an expected URL is not provided, set it to null.
    - Accept URLs given as plain links or markdown links. Include the actual URL.
    - Do not fabricate data. If the answer does not include the fields, return null for those fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str]) -> str:
    return name if (name and name.strip()) else "the carrier"

def _url_list(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if (isinstance(u, str) and u.strip())]

async def _add_verify_leaf(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str] | str],
    critical: bool = True,
    additional_instruction: str = "None",
    extra_prereq_nodes: Optional[List[Any]] = None,
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction,
        extra_prerequisites=extra_prereq_nodes
    )

def _add_source_presence_node(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    url: Optional[str],
    critical: bool = True
):
    result = bool(url and url.strip())
    return evaluator.add_custom_node(
        result=result,
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )

def _add_required_info_node(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    info: CarrierInfo,
    require_outage_url: bool = False
):
    result = bool(info and info.name and info.name.strip() and info.opensignal_url and info.rootmetrics_url)
    if require_outage_url:
        result = result and bool(info.outage_news_url and info.outage_news_url.strip())
    return evaluator.add_custom_node(
        result=result,
        id=node_id,
        desc=desc,
        parent=parent,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Verification functions for each carrier                                     #
# --------------------------------------------------------------------------- #
async def verify_carrier_1(evaluator: Evaluator, parent_node, c1: Optional[CarrierInfo]) -> None:
    info = c1 or CarrierInfo()

    carrier_node = evaluator.add_parallel(
        id="Carrier_1_Highest_Download_Speeds",
        desc="Identify the carrier with the highest overall and 5G download speeds in Opensignal January 2026",
        parent=parent_node,
        critical=False
    )

    required_info_node = _add_required_info_node(
        evaluator, carrier_node,
        node_id="C1_required_info",
        desc="Carrier 1 has name and Opensignal/RootMetrics source URLs",
        info=info,
        require_outage_url=False
    )

    # Opensignal Metrics
    os_node = evaluator.add_parallel(
        id="C1_Opensignal_Metrics",
        desc="Opensignal January 2026 performance metrics",
        parent=carrier_node,
        critical=True
    )

    os_source_presence = _add_source_presence_node(
        evaluator, os_node,
        node_id="C1_Opensignal_Source_URL",
        desc="Provide the Opensignal January 2026 report URL as reference",
        url=info.opensignal_url,
        critical=True
    )

    name = _safe_name(info.name)
    os_url = info.opensignal_url

    await _add_verify_leaf(
        evaluator, os_node, "C1_Overall_Download_Speed",
        "Overall Download Speed Experience equals 184.7 Mbps",
        f"In the Opensignal January 2026 report, {name} has an Overall Download Speed Experience of 184.7 Mbps.",
        os_url,
        additional_instruction="Look for the Overall Download Speed Experience metric for the specified carrier and confirm it equals 184.7 Mbps.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C1_5G_Download_Speed",
        "5G Download Speed equals 249.0 Mbps",
        f"In the Opensignal January 2026 report, {name} has a 5G Download Speed of 249.0 Mbps.",
        os_url,
        additional_instruction="Confirm the 5G Download Speed value equals 249.0 Mbps.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C1_Upload_Speed",
        "Upload Speed Experience equals 14.5 Mbps",
        f"In the Opensignal January 2026 report, {name} has an Upload Speed Experience of 14.5 Mbps.",
        os_url,
        additional_instruction="Confirm the Upload Speed Experience equals 14.5 Mbps.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C1_5G_Upload_Speed",
        "5G Upload Speed equals 17.6 Mbps",
        f"In the Opensignal January 2026 report, {name} has a 5G Upload Speed of 17.6 Mbps.",
        os_url,
        additional_instruction="Confirm the 5G Upload Speed equals 17.6 Mbps.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C1_5G_Availability",
        "5G Availability equals 91.2%",
        f"In the Opensignal January 2026 report, {name} has a 5G Availability of 91.2%.",
        os_url,
        additional_instruction="Confirm the 5G Availability percentage equals 91.2%.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C1_5G_Coverage_Experience",
        "5G Coverage Experience equals 8.3 out of 10 points",
        f"In the Opensignal January 2026 report, {name} has a 5G Coverage Experience score of 8.3 out of 10.",
        os_url,
        additional_instruction="Confirm the 5G Coverage Experience score equals 8.3/10.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C1_Games_Experience",
        "Games Experience score equals 77.1 points",
        f"In the Opensignal January 2026 report, {name} has a Games Experience score of 77.1 points.",
        os_url,
        additional_instruction="Confirm the Games Experience score equals 77.1 points.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C1_5G_Games_Experience",
        "5G Games Experience score equals 84.1 points",
        f"In the Opensignal January 2026 report, {name} has a 5G Games Experience score of 84.1 points.",
        os_url,
        additional_instruction="Confirm the 5G Games Experience score equals 84.1 points.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C1_Video_Experience",
        "Overall Video Experience score equals 66.5 points",
        f"In the Opensignal January 2026 report, {name} has an Overall Video Experience score of 66.5 points.",
        os_url,
        additional_instruction="Confirm the Overall Video Experience score equals 66.5 points.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C1_Live_Video_Experience",
        "Live Video Experience score equals 67.8 points",
        f"In the Opensignal January 2026 report, {name} has a Live Video Experience score of 67.8 points.",
        os_url,
        additional_instruction="Confirm the Live Video Experience score equals 67.8 points.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )

    # RootMetrics Metrics
    rm_node = evaluator.add_parallel(
        id="C1_RootMetrics_Metrics",
        desc="RootMetrics 2H 2025 performance metrics",
        parent=carrier_node,
        critical=True
    )
    rm_source_presence = _add_source_presence_node(
        evaluator, rm_node,
        node_id="C1_RootMetrics_Source_URL",
        desc="Provide the RootMetrics 2H 2025 report URL as reference",
        url=info.rootmetrics_url,
        critical=True
    )
    rm_url = info.rootmetrics_url

    await _add_verify_leaf(
        evaluator, rm_node, "C1_RM_5G_Availability",
        "RootMetrics 5G Availability equals 95.2%",
        f"In the RootMetrics 2H 2025 report, {name} has a 5G Availability of 95.2%.",
        rm_url,
        additional_instruction="Confirm the 5G Availability percentage equals 95.2% for the carrier in the 2H 2025 report.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, rm_node, "C1_RM_National_Median_Download",
        "National median download speed equals 374.5 Mbps",
        f"In the RootMetrics 2H 2025 report, {name} has a national median download speed of 374.5 Mbps.",
        rm_url,
        additional_instruction="Confirm the national median download speed equals 374.5 Mbps.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, rm_node, "C1_RM_Metro_Performance",
        "Achieved median download speeds of at least 100 Mbps in all 125 tested metro markets",
        f"In the RootMetrics 2H 2025 report, {name} achieved median download speeds of at least 100 Mbps in all 125 tested metro markets.",
        rm_url,
        additional_instruction="Confirm that the carrier delivered >=100 Mbps median download speeds in all 125 metro markets.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, rm_node, "C1_RM_State_Awards",
        "Won 119 state awards in 2H 2025",
        f"In the RootMetrics 2H 2025 report, {name} won exactly 119 state awards.",
        rm_url,
        additional_instruction="Verify the exact state award count equals 119.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, rm_node, "C1_RM_Metro_Awards",
        "Won 588 metro awards in 2H 2025",
        f"In the RootMetrics 2H 2025 report, {name} won exactly 588 metro awards.",
        rm_url,
        additional_instruction="Verify the exact metro award count equals 588.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )

    # 5G Technology & Deployment
    tech_node = evaluator.add_parallel(
        id="C1_5G_Technology_Deployment",
        desc="5G technology and deployment characteristics",
        parent=carrier_node,
        critical=True
    )

    # Use RootMetrics URL for technology sampling stats; other claims may be present in RM narrative; if not, they will fail.
    await _add_verify_leaf(
        evaluator, tech_node, "C1_5G_SA_Usage",
        "Used 5G Standalone (SA) technology in 93.2% of metro testing samples",
        f"In the RootMetrics 2H 2025 report, {name} used 5G Standalone (SA) technology in 93.2% of metro testing samples.",
        rm_url,
        additional_instruction="Confirm SA technology usage equals 93.2% of metro samples for the carrier.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, tech_node, "C1_Carrier_Aggregation",
        "Over two-thirds of 5G samples used 4-carrier aggregation",
        f"In the RootMetrics 2H 2025 report, over two-thirds of {name}'s 5G samples used 4-carrier aggregation.",
        rm_url,
        additional_instruction="Confirm the narrative indicating >66% of 5G samples using 4-carrier aggregation.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, tech_node, "C1_Rural_Coverage_Goal",
        "Aims to cover 90% of rural households by 2026",
        f"{name} aims to cover 90% of rural households by 2026 using low-band 5G spectrum.",
        rm_url,
        additional_instruction="Confirm the stated rural coverage goal in the provided sources. If not in RootMetrics, the claim may not be supported.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, tech_node, "C1_Ultra_Capacity_Coverage",
        "Covers 306 million Americans with Ultra Capacity 5G network",
        f"{name} covers 306 million Americans with its Ultra Capacity 5G network.",
        _url_list(os_url, rm_url),
        additional_instruction="Confirm coverage figure 306 million for the carrier's Ultra Capacity 5G (if available in provided sources).",
        extra_prereq_nodes=[required_info_node]
    )


async def verify_carrier_2(evaluator: Evaluator, parent_node, c2: Optional[CarrierInfo]) -> None:
    info = c2 or CarrierInfo()

    carrier_node = evaluator.add_parallel(
        id="Carrier_2_Highest_Overall_Performance",
        desc="Identify the carrier with the best overall network performance and reliability in RootMetrics 2H 2025",
        parent=parent_node,
        critical=False
    )

    required_info_node = _add_required_info_node(
        evaluator, carrier_node,
        node_id="C2_required_info",
        desc="Carrier 2 has name, Opensignal/RootMetrics source URLs, and outage news URL",
        info=info,
        require_outage_url=True
    )

    # RootMetrics Leadership
    rm_node = evaluator.add_parallel(
        id="C2_RootMetrics_Leadership",
        desc="RootMetrics 2H 2025 leadership and awards",
        parent=carrier_node,
        critical=True
    )
    rm_source_presence = _add_source_presence_node(
        evaluator, rm_node,
        node_id="C2_RootMetrics_Source_URL",
        desc="Provide the RootMetrics 2H 2025 report URL as reference",
        url=info.rootmetrics_url,
        critical=True
    )
    name = _safe_name(info.name)
    rm_url = info.rootmetrics_url

    await _add_verify_leaf(
        evaluator, rm_node, "C2_RM_Total_US_Awards",
        "Won exactly 7 US RootScore Awards",
        f"In the RootMetrics 2H 2025 report, {name} won exactly 7 US RootScore Awards.",
        rm_url,
        additional_instruction="Verify total count of US RootScore Awards equals 7.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, rm_node, "C2_RM_Overall_Award",
        "Won the Overall RootScore Award for best overall network performance",
        f"In the RootMetrics 2H 2025 report, {name} won the Overall RootScore Award for best overall performance.",
        rm_url,
        additional_instruction="Confirm the Overall RootScore Award.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, rm_node, "C2_RM_Reliability_Award",
        "Won the Reliability RootScore Award",
        f"In the RootMetrics 2H 2025 report, {name} won the Reliability RootScore Award.",
        rm_url,
        additional_instruction="Confirm the Reliability RootScore Award.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, rm_node, "C2_RM_5G_Leadership",
        "Won Best 5G Experience, Fastest 5G, and Most Reliable 5G awards",
        f"In the RootMetrics 2H 2025 report, {name} won Best 5G Experience, Fastest 5G, and Most Reliable 5G awards.",
        rm_url,
        additional_instruction="Confirm the trio of 5G leadership awards.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, rm_node, "C2_RM_State_Awards",
        "Won 329 state awards, more than any other carrier",
        f"In the RootMetrics 2H 2025 report, {name} won exactly 329 state awards, more than any other carrier.",
        rm_url,
        additional_instruction="Confirm state awards count equals 329 and that it is the highest among carriers.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, rm_node, "C2_RM_Metro_Awards",
        "Won 801 metro awards, more than any other carrier",
        f"In the RootMetrics 2H 2025 report, {name} won exactly 801 metro awards, more than any other carrier.",
        rm_url,
        additional_instruction="Confirm metro awards count equals 801 and highest among carriers.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )

    # Opensignal Video Performance
    os_node = evaluator.add_parallel(
        id="C2_Opensignal_Video_Performance",
        desc="Opensignal January 2026 video experience metrics",
        parent=carrier_node,
        critical=True
    )
    os_source_presence = _add_source_presence_node(
        evaluator, os_node,
        node_id="C2_Opensignal_Source_URL",
        desc="Provide the Opensignal January 2026 report URL as reference",
        url=info.opensignal_url,
        critical=True
    )
    os_url = info.opensignal_url

    await _add_verify_leaf(
        evaluator, os_node, "C2_5G_Video_Experience",
        "5G Video Experience equals 71.4 points",
        f"In the Opensignal January 2026 report, {name} has a 5G Video Experience score of 71.4 points.",
        os_url,
        additional_instruction="Confirm 5G Video Experience equals 71.4.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C2_5G_Live_Video_Experience",
        "5G Live Video Experience equals 73.4 points",
        f"In the Opensignal January 2026 report, {name} has a 5G Live Video Experience score of 73.4 points.",
        os_url,
        additional_instruction="Confirm 5G Live Video Experience equals 73.4.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C2_Coverage_Experience",
        "Coverage Experience equals 9.7 out of 10 points",
        f"In the Opensignal January 2026 report, {name} has a Coverage Experience score of 9.7 out of 10.",
        os_url,
        additional_instruction="Confirm Coverage Experience equals 9.7/10.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )

    # Network Outage (January 2026)
    outage_node = evaluator.add_parallel(
        id="C2_Network_Outage_January_2026",
        desc="Major network outage incident details",
        parent=carrier_node,
        critical=True
    )
    outage_source_presence = _add_source_presence_node(
        evaluator, outage_node,
        node_id="C2_Outage_Source_URL",
        desc="Provide a news article URL documenting the January 14, 2026 outage",
        url=info.outage_news_url,
        critical=True
    )
    news_url = info.outage_news_url

    await _add_verify_leaf(
        evaluator, outage_node, "C2_Outage_Date",
        "Experienced a nationwide outage on January 14, 2026",
        f"News coverage confirms that {name} experienced a nationwide outage on January 14, 2026.",
        news_url,
        additional_instruction="Verify date: January 14, 2026; scope: nationwide.",
        extra_prereq_nodes=[required_info_node, outage_source_presence]
    )
    await _add_verify_leaf(
        evaluator, outage_node, "C2_Outage_Duration",
        "Outage lasted over 10 hours or described as 'all-day'",
        f"News coverage reports the outage duration was over 10 hours (or described as all-day) for {name}.",
        news_url,
        additional_instruction="Confirm duration phrasing indicating >10 hours or all-day.",
        extra_prereq_nodes=[required_info_node, outage_source_presence]
    )
    await _add_verify_leaf(
        evaluator, outage_node, "C2_Outage_Scale",
        "Affected approximately 2 million customers or generated approximately 2.3 million Downdetector reports",
        f"News coverage indicates approximately 2 million customers affected or approximately 2.3 million Downdetector reports for {name}.",
        news_url,
        additional_instruction="Confirm either ~2 million customers affected or ~2.3 million Downdetector reports.",
        extra_prereq_nodes=[required_info_node, outage_source_presence]
    )
    await _add_verify_leaf(
        evaluator, outage_node, "C2_Outage_Cause",
        "Root cause identified as a software issue related to 5G Standalone (5G SA) core",
        f"News coverage identifies the outage cause as a software issue related to the 5G Standalone (5G SA) core for {name}.",
        news_url,
        additional_instruction="Confirm the cause explicitly mentions 5G SA core software issue.",
        extra_prereq_nodes=[required_info_node, outage_source_presence]
    )
    await _add_verify_leaf(
        evaluator, outage_node, "C2_Outage_Resolution",
        "Outage was resolved by January 15, 2026",
        f"News coverage confirms the outage was resolved by January 15, 2026 for {name}.",
        news_url,
        additional_instruction="Confirm resolution timeline by Jan 15, 2026.",
        extra_prereq_nodes=[required_info_node, outage_source_presence]
    )
    await _add_verify_leaf(
        evaluator, outage_node, "C2_Customer_Credit",
        "Offered $20 credit to affected customers",
        f"News coverage states that {name} offered a $20 credit to customers affected by the January 14, 2026 outage.",
        news_url,
        additional_instruction="Confirm the $20 credit offer.",
        extra_prereq_nodes=[required_info_node, outage_source_presence]
    )

    # 5G Technology Strategy
    tech_node = evaluator.add_parallel(
        id="C2_5G_Technology_Strategy",
        desc="5G technology deployment strategy and characteristics",
        parent=carrier_node,
        critical=True
    )

    await _add_verify_leaf(
        evaluator, tech_node, "C2_5G_SA_Expansion",
        "Expanded 5G SA usage from 24.5% to 59.7% of metro samples between 1H and 2H 2025",
        f"In the RootMetrics 2H 2025 report, {name} expanded 5G SA usage from 24.5% to 59.7% of metro samples between H1 and H2 2025.",
        rm_url,
        additional_instruction="Confirm the change from 24.5% to 59.7% in SA usage across metro samples.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, tech_node, "C2_C_Band_Usage",
        "Increased C-band spectrum usage to 81.3% of metro samples by Q4 2025",
        f"In the RootMetrics 2H 2025 report, {name} increased C-band spectrum usage to 81.3% of metro samples by Q4 2025.",
        rm_url,
        additional_instruction="Confirm the C-band usage percentage equals 81.3% by Q4 2025.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, tech_node, "C2_Rural_LTE_Strategy",
        "Nearly half of state-area testing samples connected to 4G LTE network, leveraging mature infrastructure",
        f"In the RootMetrics 2H 2025 report, nearly half of state-area testing samples for {name} connected to the 4G LTE network.",
        rm_url,
        additional_instruction="Confirm phrasing indicating ~50% of state-area samples using LTE.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )


async def verify_carrier_3(evaluator: Evaluator, parent_node, c3: Optional[CarrierInfo]) -> None:
    info = c3 or CarrierInfo()

    carrier_node = evaluator.add_parallel(
        id="Carrier_3_Highest_Reliability",
        desc="Identify the carrier with the highest Time on Network percentage in Opensignal January 2026",
        parent=parent_node,
        critical=False
    )

    required_info_node = _add_required_info_node(
        evaluator, carrier_node,
        node_id="C3_required_info",
        desc="Carrier 3 has name and Opensignal/RootMetrics source URLs",
        info=info,
        require_outage_url=False
    )

    # Opensignal Reliability Metrics
    os_node = evaluator.add_parallel(
        id="C3_Opensignal_Reliability",
        desc="Opensignal January 2026 reliability and coverage metrics",
        parent=carrier_node,
        critical=True
    )
    os_source_presence = _add_source_presence_node(
        evaluator, os_node,
        node_id="C3_Opensignal_Source_URL",
        desc="Provide the Opensignal January 2026 report URL as reference",
        url=info.opensignal_url,
        critical=True
    )
    name = _safe_name(info.name)
    os_url = info.opensignal_url

    await _add_verify_leaf(
        evaluator, os_node, "C3_Time_on_Network",
        "Time on Network equals 99.6% (highest among all carriers)",
        f"In the Opensignal January 2026 report, {name} has a Time on Network of 99.6%, the highest among carriers.",
        os_url,
        additional_instruction="Confirm Time on Network equals 99.6% and note it is the highest.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C3_Coverage_Experience",
        "Coverage Experience equals 9.3 out of 10 points",
        f"In the Opensignal January 2026 report, {name} has a Coverage Experience score of 9.3 out of 10.",
        os_url,
        additional_instruction="Confirm Coverage Experience equals 9.3/10.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C3_5G_Coverage_Experience",
        "5G Coverage Experience equals 7.4 out of 10 points",
        f"In the Opensignal January 2026 report, {name} has a 5G Coverage Experience score of 7.4 out of 10.",
        os_url,
        additional_instruction="Confirm 5G Coverage Experience equals 7.4/10.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )
    await _add_verify_leaf(
        evaluator, os_node, "C3_5G_Availability",
        "5G Availability equals 88.7%",
        f"In the Opensignal January 2026 report, {name} has a 5G Availability of 88.7%.",
        os_url,
        additional_instruction="Confirm the 5G Availability equals 88.7%.",
        extra_prereq_nodes=[required_info_node, os_source_presence]
    )

    # RootMetrics Performance
    rm_node = evaluator.add_parallel(
        id="C3_RootMetrics_Performance",
        desc="RootMetrics 2H 2025 performance metrics",
        parent=carrier_node,
        critical=True
    )
    rm_source_presence = _add_source_presence_node(
        evaluator, rm_node,
        node_id="C3_RootMetrics_Source_URL",
        desc="Provide the RootMetrics 2H 2025 report URL as reference",
        url=info.rootmetrics_url,
        critical=True
    )
    rm_url = info.rootmetrics_url

    await _add_verify_leaf(
        evaluator, rm_node, "C3_RM_US_Awards",
        "Shared three US RootScore Awards: Network Speed, Call Performance, and Text Performance",
        f"In the RootMetrics 2H 2025 report, {name} shared three US RootScore Awards: Network Speed, Call Performance, and Text Performance.",
        rm_url,
        additional_instruction="Confirm sharing of these three award categories.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, rm_node, "C3_RM_State_Awards",
        "Won 253 state awards in 2H 2025",
        f"In the RootMetrics 2H 2025 report, {name} won exactly 253 state awards.",
        rm_url,
        additional_instruction="Confirm the state award count equals 253.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, rm_node, "C3_RM_Metro_Awards",
        "Won 648 metro awards in 2H 2025",
        f"In the RootMetrics 2H 2025 report, {name} won exactly 648 metro awards.",
        rm_url,
        additional_instruction="Confirm the metro award count equals 648.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, rm_node, "C3_RM_Metro_Speed_Coverage",
        "Delivered median download speeds of at least 100 Mbps in 122 of 125 tested markets",
        f"In the RootMetrics 2H 2025 report, {name} delivered median download speeds of at least 100 Mbps in 122 of 125 markets.",
        rm_url,
        additional_instruction="Confirm >=100 Mbps in 122/125 markets.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )

    # Network Infrastructure and Technology
    infra_node = evaluator.add_parallel(
        id="C3_Network_Infrastructure",
        desc="Network infrastructure and technology deployment",
        parent=carrier_node,
        critical=True
    )

    await _add_verify_leaf(
        evaluator, infra_node, "C3_Spectrum_Expansion",
        "Expanding use of 3.45 GHz midband spectrum, generally deploying 60 MHz blocks with some 100 MHz blocks",
        f"In the RootMetrics 2H 2025 report, {name} is expanding use of 3.45 GHz midband spectrum, generally deploying 60 MHz blocks with some 100 MHz blocks.",
        rm_url,
        additional_instruction="Confirm the 3.45 GHz expansion details.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, infra_node, "C3_RAN_Vendor_Shift",
        "Over 80% of radio access network (RAN) samples now use Ericsson equipment, shifted from Nokia",
        f"In the RootMetrics 2H 2025 report, over 80% of {name}'s RAN samples now use Ericsson equipment, indicating a shift from Nokia.",
        rm_url,
        additional_instruction="Confirm vendor share >80% Ericsson.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
    )
    await _add_verify_leaf(
        evaluator, infra_node, "C3_Voice_Technology",
        "Vast majority of calls use VoLTE (Voice over LTE) technology",
        f"In the RootMetrics 2H 2025 report, the vast majority of {name}'s calls use VoLTE.",
        rm_url,
        additional_instruction="Confirm VoLTE majority usage.",
        extra_prereq_nodes=[required_info_node, rm_source_presence]
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
    Evaluate an answer for the US Mobile Carrier Analysis 2025-2026 task.
    Builds a verification tree and runs URL-grounded checks for each required metric/claim.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root follows sequential aggregation to reflect staged evaluation
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

    # IMPORTANT: Root cannot be critical because the framework enforces that critical parents have only critical children.
    # The provided rubric marks many non-critical children; hence we set root as non-critical here to comply with framework constraints.

    # 1) Extract carriers and their source URLs from the answer
    carriers = await evaluator.extract(
        prompt=prompt_extract_carriers(),
        template_class=CarrierExtraction,
        extraction_name="carrier_extraction"
    )

    # 2) Build the "Carrier Identification" parallel node
    carrier_ident_node = evaluator.add_parallel(
        id="Carrier_Identification",
        desc="Identify three distinct major US mobile carriers based on the specified performance criteria",
        parent=root,
        critical=False
    )

    # 3) Verify Carrier 1
    await verify_carrier_1(evaluator, carrier_ident_node, carriers.carrier1)

    # 4) Verify Carrier 2
    await verify_carrier_2(evaluator, carrier_ident_node, carriers.carrier2)

    # 5) Verify Carrier 3
    await verify_carrier_3(evaluator, carrier_ident_node, carriers.carrier3)

    # Return final structured summary
    return evaluator.get_summary()