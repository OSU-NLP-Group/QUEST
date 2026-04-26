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
TASK_ID = "ftth_feasibility_2026"
TASK_DESCRIPTION = """
A regional telecommunications provider is planning a fiber-to-the-home (FTTH) deployment project in the United States for 2026 and needs a comprehensive feasibility analysis. Prepare an analysis report that addresses the following requirements:

1. Cost Projections: Provide estimated costs per foot for both underground and aerial fiber deployment in 2026, accounting for the industry trend that deployment costs rose in 2025 and are expected to increase again in 2026. Include the approximate cost ratio between underground and aerial methods, and note any regional cost variations across U.S. regions (West, South, Midwest).

2. Tax Incentives: Analyze the federal tax policy change for 2026 regarding bonus depreciation for fiber infrastructure. Identify the specific depreciation percentage that was restored, the expected industry-wide impact on FTTH capital expenditures, and explain how this incentive applies to fiber infrastructure investments.

3. Deployment Method Comparison: Compare underground versus aerial deployment methods, including the labor cost percentages for each method, the potential impact of make-ready costs (including their variability), and factors that could affect deployment timelines such as permitting and engineering requirements.

4. Federal Funding Context: Provide context about the BEAD (Broadband Equity, Access, and Deployment) program, including its total allocation and the expected timeline for peak construction activity.

5. Infrastructure Reliability: Discuss network redundancy best practices for fiber infrastructure, reference at least one major telecommunications outage from 2026 that highlights the importance of reliability planning, and address recovery time considerations for network infrastructure.

All information must be supported with reference URLs from credible sources. The analysis should be grounded in actual 2026 industry data, federal policies, and recent telecommunications events.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RegionalNotes(BaseModel):
    west: Optional[str] = None
    south: Optional[str] = None
    midwest: Optional[str] = None


class CostProjectionExtraction(BaseModel):
    underground_cost_per_foot: Optional[str] = None
    underground_sources: List[str] = Field(default_factory=list)
    aerial_cost_per_foot: Optional[str] = None
    aerial_sources: List[str] = Field(default_factory=list)
    cost_ratio_statement: Optional[str] = None  # e.g., "Underground is ~2x aerial"
    ratio_sources: List[str] = Field(default_factory=list)
    regional_variations: Optional[RegionalNotes] = None
    regional_sources: List[str] = Field(default_factory=list)


class TaxIncentiveExtraction(BaseModel):
    bonus_depreciation_percentage: Optional[str] = None  # e.g., "100%"
    bonus_dep_sources: List[str] = Field(default_factory=list)
    industry_impact_range: Optional[str] = None  # e.g., "5-15%"
    impact_sources: List[str] = Field(default_factory=list)
    application_explanation: Optional[str] = None
    application_sources: List[str] = Field(default_factory=list)


class DeploymentComparisonExtraction(BaseModel):
    underground_labor_percent: Optional[str] = None  # e.g., "72%"
    aerial_labor_percent: Optional[str] = None       # e.g., "64%"
    labor_sources: List[str] = Field(default_factory=list)
    make_ready_notes: Optional[str] = None
    make_ready_sources: List[str] = Field(default_factory=list)
    timeline_factors: List[str] = Field(default_factory=list)  # e.g., ["permitting", "engineering", ...]
    timeline_sources: List[str] = Field(default_factory=list)


class FederalFundingExtraction(BaseModel):
    bead_total_allocation: Optional[str] = None  # e.g., "$42.45 billion"
    bead_sources: List[str] = Field(default_factory=list)
    peak_construction_timeline: Optional[str] = None  # e.g., "Obligations by late 2025; peak construction 2026–2027"
    funding_timeline_sources: List[str] = Field(default_factory=list)


class ReliabilityExtraction(BaseModel):
    redundancy_best_practices: List[str] = Field(default_factory=list)  # e.g., ["diverse routing", "multiple providers"]
    redundancy_sources: List[str] = Field(default_factory=list)
    outage_2026_event: Optional[str] = None  # e.g., "Verizon outage January 2026" or "Azure outage February 2026"
    outage_sources: List[str] = Field(default_factory=list)
    recovery_time_considerations: Optional[str] = None  # e.g., "RTO of 4 hours"
    recovery_sources: List[str] = Field(default_factory=list)


class FTTHAnalysisExtraction(BaseModel):
    cost: Optional[CostProjectionExtraction] = None
    tax: Optional[TaxIncentiveExtraction] = None
    deploy: Optional[DeploymentComparisonExtraction] = None
    funding: Optional[FederalFundingExtraction] = None
    reliability: Optional[ReliabilityExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ftth_analysis() -> str:
    return """
    From the provided answer, extract a structured summary of the FTTH (fiber-to-the-home) feasibility analysis for 2026, including URLs as evidence.

    IMPORTANT:
    - Extract only information explicitly present in the answer.
    - For every 'sources' field, include only actual URLs explicitly provided in the answer (plain URLs or URLs embedded in markdown links). Do not infer or invent.
    - Use strings for numeric values (percentages, dollars per foot) exactly as stated (e.g., "72%", "$10/ft", "5–15%").

    Return a JSON object with the following structure:

    {
      "cost": {
        "underground_cost_per_foot": string | null,
        "underground_sources": [url, ...],
        "aerial_cost_per_foot": string | null,
        "aerial_sources": [url, ...],
        "cost_ratio_statement": string | null,  // e.g., "Underground is ~2x aerial"
        "ratio_sources": [url, ...],
        "regional_variations": {
          "west": string | null,
          "south": string | null,
          "midwest": string | null
        } | null,
        "regional_sources": [url, ...]
      },
      "tax": {
        "bonus_depreciation_percentage": string | null, // e.g., "100%"
        "bonus_dep_sources": [url, ...],
        "industry_impact_range": string | null,         // e.g., "5–15%"
        "impact_sources": [url, ...],
        "application_explanation": string | null,
        "application_sources": [url, ...]
      },
      "deploy": {
        "underground_labor_percent": string | null, // e.g., "72%"
        "aerial_labor_percent": string | null,      // e.g., "64%"
        "labor_sources": [url, ...],
        "make_ready_notes": string | null,          // e.g., "Can exceed 150% of construction budget"
        "make_ready_sources": [url, ...],
        "timeline_factors": [string, ...],          // e.g., ["permitting", "engineering", "delays"]
        "timeline_sources": [url, ...]
      },
      "funding": {
        "bead_total_allocation": string | null,     // e.g., "$42.45 billion"
        "bead_sources": [url, ...],
        "peak_construction_timeline": string | null,// e.g., "Obligations by late 2025; peak construction 2026–2027"
        "funding_timeline_sources": [url, ...]
      },
      "reliability": {
        "redundancy_best_practices": [string, ...],
        "redundancy_sources": [url, ...],
        "outage_2026_event": string | null,         // e.g., "Verizon outage January 2026" or "Azure outage February 2026"
        "outage_sources": [url, ...],
        "recovery_time_considerations": string | null,
        "recovery_sources": [url, ...]
      }
    }

    If any field is missing from the answer, set it to null (or an empty list for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_any_sources(url_lists: List[List[str]]) -> bool:
    return any(bool(lst) for lst in url_lists)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_cost_projection_nodes(evaluator: Evaluator, parent_node, cost: Optional[CostProjectionExtraction]) -> None:
    cost_node = evaluator.add_parallel(
        id="Cost_Projection_Analysis",
        desc="Analysis of 2026 fiber deployment costs with proper consideration of industry trends and regional factors",
        parent=parent_node,
        critical=True
    )

    # Underground cost
    underground_seq = evaluator.add_sequential(
        id="Underground_Deployment_Cost_Main",
        desc="Provides underground fiber deployment cost estimate based on 2026 industry data, considering rising costs",
        parent=cost_node,
        critical=True
    )
    underground_provided = evaluator.add_custom_node(
        result=bool(cost and cost.underground_cost_per_foot and cost.underground_sources),
        id="Underground_Cost_Provided",
        desc="Underground cost estimate and sources are provided",
        parent=underground_seq,
        critical=True
    )
    underground_leaf = evaluator.add_leaf(
        id="Underground_Deployment_Cost",
        desc="Provides underground fiber deployment cost estimate based on 2026 industry data, with consideration that costs rose in 2025 and are expected to increase again in 2026",
        parent=underground_seq,
        critical=True
    )
    underground_claim = f"The estimated cost per foot for underground fiber deployment in 2026 is {cost.underground_cost_per_foot}."
    await evaluator.verify(
        claim=underground_claim,
        node=underground_leaf,
        sources=cost.underground_sources if cost else [],
        additional_instruction="Verify the per-foot underground cost and ensure the source context acknowledges increased deployment costs in 2025 and expected increases in 2026."
    )

    # Aerial cost
    aerial_seq = evaluator.add_sequential(
        id="Aerial_Deployment_Cost_Main",
        desc="Provides aerial fiber deployment cost estimate based on 2026 industry data, considering rising costs",
        parent=cost_node,
        critical=True
    )
    aerial_provided = evaluator.add_custom_node(
        result=bool(cost and cost.aerial_cost_per_foot and cost.aerial_sources),
        id="Aerial_Cost_Provided",
        desc="Aerial cost estimate and sources are provided",
        parent=aerial_seq,
        critical=True
    )
    aerial_leaf = evaluator.add_leaf(
        id="Aerial_Deployment_Cost",
        desc="Provides aerial fiber deployment cost estimate based on 2026 industry data, with consideration that costs rose in 2025 and are expected to increase again in 2026",
        parent=aerial_seq,
        critical=True
    )
    aerial_claim = f"The estimated cost per foot for aerial fiber deployment in 2026 is {cost.aerial_cost_per_foot}."
    await evaluator.verify(
        claim=aerial_claim,
        node=aerial_leaf,
        sources=cost.aerial_sources if cost else [],
        additional_instruction="Verify the per-foot aerial cost and ensure the source context acknowledges increased deployment costs in 2025 and expected increases in 2026."
    )

    # Ratio underground vs aerial
    ratio_seq = evaluator.add_sequential(
        id="Cost_Comparison_Ratio_Main",
        desc="Acknowledges approximate 2x underground vs aerial cost ratio",
        parent=cost_node,
        critical=True
    )
    ratio_provided = evaluator.add_custom_node(
        result=bool(cost and cost.cost_ratio_statement and cost.ratio_sources),
        id="Cost_Ratio_Provided",
        desc="Cost ratio statement and sources are provided",
        parent=ratio_seq,
        critical=True
    )
    ratio_leaf = evaluator.add_leaf(
        id="Cost_Comparison_Ratio",
        desc="Acknowledges that underground deployment is approximately twice as costly as aerial deployment based on industry benchmarks",
        parent=ratio_seq,
        critical=True
    )
    ratio_claim = "Underground fiber deployment costs are approximately twice as high as aerial deployment costs."
    await evaluator.verify(
        claim=ratio_claim,
        node=ratio_leaf,
        sources=cost.ratio_sources if cost else [],
        additional_instruction="Accept approximate ratios (e.g., 1.8x–2.5x) as 'approximately twice' if supported by the source."
    )

    # Regional variations (West, South, Midwest)
    reg_seq = evaluator.add_sequential(
        id="Regional_Cost_Variation_Main",
        desc="Notes regional cost differences across West, South, Midwest",
        parent=cost_node,
        critical=True
    )
    regional_provided = evaluator.add_custom_node(
        result=bool(cost and cost.regional_variations and (
            cost.regional_variations.west or cost.regional_variations.south or cost.regional_variations.midwest) and cost.regional_sources),
        id="Regional_Cost_Variation_Provided",
        desc="Regional cost variation notes and sources are provided",
        parent=reg_seq,
        critical=True
    )
    regional_leaf = evaluator.add_leaf(
        id="Regional_Cost_Variation",
        desc="Notes regional cost differences across U.S. regions (West, South, Midwest) as explicitly required by the question",
        parent=reg_seq,
        critical=True
    )
    regional_claim = "There are regional cost differences across the U.S. regions West, South, and Midwest for fiber deployment costs."
    await evaluator.verify(
        claim=regional_claim,
        node=regional_leaf,
        sources=cost.regional_sources if cost else [],
        additional_instruction="Confirm that the cited source(s) explicitly discuss cost differences or drivers across the West, South, and Midwest regions."
    )

    # Overall cost projection sources presence
    cost_sources_leaf = evaluator.add_custom_node(
        result=bool(cost) and _has_any_sources([
            cost.underground_sources if cost else [],
            cost.aerial_sources if cost else [],
            cost.ratio_sources if cost else [],
            cost.regional_sources if cost else [],
        ]),
        id="Cost_Projection_Source",
        desc="Provides reference URL(s) supporting the cost estimates",
        parent=cost_node,
        critical=True
    )


async def build_tax_incentive_nodes(evaluator: Evaluator, parent_node, tax: Optional[TaxIncentiveExtraction]) -> None:
    tax_node = evaluator.add_parallel(
        id="Tax_Incentive_Calculation",
        desc="Analysis of the 100% bonus depreciation tax benefit restored for 2026",
        parent=parent_node,
        critical=True
    )

    # Bonus depreciation policy
    bonus_seq = evaluator.add_sequential(
        id="Bonus_Depreciation_Policy_Main",
        desc="Identifies 100% bonus depreciation restored for 2026",
        parent=tax_node,
        critical=True
    )
    bonus_provided = evaluator.add_custom_node(
        result=bool(tax and tax.bonus_depreciation_percentage and tax.bonus_dep_sources),
        id="Bonus_Depreciation_Policy_Provided",
        desc="Bonus depreciation percentage and sources are provided",
        parent=bonus_seq,
        critical=True
    )
    bonus_leaf = evaluator.add_leaf(
        id="Bonus_Depreciation_Policy",
        desc="Correctly identifies that 100% bonus depreciation was restored in federal tax law for 2026",
        parent=bonus_seq,
        critical=True
    )
    bonus_claim = "100% bonus depreciation was restored for qualifying property in 2026 under federal tax law."
    await evaluator.verify(
        claim=bonus_claim,
        node=bonus_leaf,
        sources=tax.bonus_dep_sources if tax else [],
        additional_instruction="Verify that the cited source(s) explicitly state 100% bonus depreciation applies for 2026."
    )

    # Expected industry impact (5–15% increase in FTTH capex)
    impact_seq = evaluator.add_sequential(
        id="Expected_Industry_Impact_Main",
        desc="Analyst expectation of 5–15% increase in FTTH capex",
        parent=tax_node,
        critical=True
    )
    impact_provided = evaluator.add_custom_node(
        result=bool(tax and tax.industry_impact_range and tax.impact_sources),
        id="Expected_Industry_Impact_Provided",
        desc="Industry impact range and sources are provided",
        parent=impact_seq,
        critical=True
    )
    impact_leaf = evaluator.add_leaf(
        id="Expected_Industry_Impact",
        desc="References the analyst expectation that the tax change will fuel a 5-15% increase in FTTH capital expenditures",
        parent=impact_seq,
        critical=True
    )
    impact_claim = "Analysts expect the restored bonus depreciation to fuel approximately a 5–15% increase in FTTH capital expenditures."
    await evaluator.verify(
        claim=impact_claim,
        node=impact_leaf,
        sources=tax.impact_sources if tax else [],
        additional_instruction="Confirm that the cited source(s) provide an analyst estimate or expectation for FTTH capex increase in the 5–15% range due to the 2026 bonus depreciation."
    )

    # Tax benefit application to fiber infrastructure investments
    app_seq = evaluator.add_sequential(
        id="Tax_Benefit_Application_Main",
        desc="Explains how 100% bonus depreciation applies to fiber investments",
        parent=tax_node,
        critical=True
    )
    app_provided = evaluator.add_custom_node(
        result=bool(tax and tax.application_explanation and tax.application_sources),
        id="Tax_Benefit_Application_Provided",
        desc="Application explanation and sources are provided",
        parent=app_seq,
        critical=True
    )
    app_leaf = evaluator.add_leaf(
        id="Tax_Benefit_Application",
        desc="Explains how the 100% bonus depreciation applies to fiber infrastructure investments",
        parent=app_seq,
        critical=True
    )
    app_claim = "100% bonus depreciation applies to qualifying fiber infrastructure investments, allowing full expensing in the first year (2026)."
    await evaluator.verify(
        claim=app_claim,
        node=app_leaf,
        sources=tax.application_sources if tax else [],
        additional_instruction="Verify that fiber infrastructure (or its applicable asset class) qualifies under the 2026 bonus depreciation rules to allow 100% first-year expensing."
    )

    # Overall tax policy sources presence
    tax_sources_leaf = evaluator.add_custom_node(
        result=bool(tax) and _has_any_sources([
            tax.bonus_dep_sources if tax else [],
            tax.impact_sources if tax else [],
            tax.application_sources if tax else [],
        ]),
        id="Tax_Policy_Source",
        desc="Provides reference URL(s) supporting the tax policy information",
        parent=tax_node,
        critical=True
    )


async def build_deployment_method_nodes(evaluator: Evaluator, parent_node, deploy: Optional[DeploymentComparisonExtraction]) -> None:
    deploy_node = evaluator.add_parallel(
        id="Deployment_Method_Comparison",
        desc="Comparison of underground versus aerial deployment methods with consideration of make-ready costs and project-specific factors",
        parent=parent_node,
        critical=True
    )

    # Labor cost breakdown (72% underground, 64% aerial)
    labor_seq = evaluator.add_sequential(
        id="Labor_Cost_Breakdown_Main",
        desc="Labor cost shares for underground vs aerial",
        parent=deploy_node,
        critical=True
    )
    labor_provided = evaluator.add_custom_node(
        result=bool(deploy and deploy.underground_labor_percent and deploy.aerial_labor_percent and deploy.labor_sources),
        id="Labor_Cost_Breakdown_Provided",
        desc="Labor cost percentages and sources are provided",
        parent=labor_seq,
        critical=True
    )
    labor_leaf = evaluator.add_leaf(
        id="Labor_Cost_Breakdown",
        desc="Acknowledges that labor accounts for approximately 72% of underground deployment costs versus 64% for aerial",
        parent=labor_seq,
        critical=True
    )
    labor_claim = f"Labor accounts for approximately {deploy.underground_labor_percent} of underground deployment costs and approximately {deploy.aerial_labor_percent} for aerial deployment."
    await evaluator.verify(
        claim=labor_claim,
        node=labor_leaf,
        sources=deploy.labor_sources if deploy else [],
        additional_instruction="Accept percentages close to 72% (underground) and 64% (aerial) as 'approximately', allowing a small tolerance (±5%) if supported."
    )

    # Make-ready considerations (variable; can exceed 150% of construction budget)
    mr_seq = evaluator.add_sequential(
        id="Make_Ready_Considerations_Main",
        desc="Make-ready cost variability and potential exceedance",
        parent=deploy_node,
        critical=True
    )
    mr_provided = evaluator.add_custom_node(
        result=bool(deploy and deploy.make_ready_notes and deploy.make_ready_sources),
        id="Make_Ready_Considerations_Provided",
        desc="Make-ready notes and sources are provided",
        parent=mr_seq,
        critical=True
    )
    mr_leaf = evaluator.add_leaf(
        id="Make_Ready_Considerations",
        desc="Discusses make-ready costs and notes that they can be highly variable and potentially exceed 150% of construction budget",
        parent=mr_seq,
        critical=True
    )
    mr_claim = "Make-ready costs for aerial fiber can be highly variable and can exceed 150% of the construction budget in some cases."
    await evaluator.verify(
        claim=mr_claim,
        node=mr_leaf,
        sources=deploy.make_ready_sources if deploy else [],
        additional_instruction="Verify that the source(s) explicitly indicate high variability and cite the possibility of make-ready costs exceeding 150% of construction budget."
    )

    # Deployment timeline factors (permitting, engineering, potential delays)
    tl_seq = evaluator.add_sequential(
        id="Deployment_Timeline_Factors_Main",
        desc="Timeline factors such as permitting and engineering",
        parent=deploy_node,
        critical=True
    )
    tl_provided = evaluator.add_custom_node(
        result=bool(deploy and deploy.timeline_factors and len(deploy.timeline_factors) > 0 and deploy.timeline_sources),
        id="Deployment_Timeline_Factors_Provided",
        desc="Timeline factors and sources are provided",
        parent=tl_seq,
        critical=True
    )
    tl_leaf = evaluator.add_leaf(
        id="Deployment_Timeline_Factors",
        desc="Identifies and discusses factors that could affect deployment timelines such as permitting, engineering requirements, and potential delays",
        parent=tl_seq,
        critical=True
    )
    tl_claim = "Factors affecting fiber deployment timelines include permitting processes and engineering requirements, which can introduce delays."
    await evaluator.verify(
        claim=tl_claim,
        node=tl_leaf,
        sources=deploy.timeline_sources if deploy else [],
        additional_instruction="Confirm that the source(s) explicitly mention permitting and engineering (or closely related processes) as timeline drivers and acknowledge potential delays."
    )

    # Overall deployment method sources presence
    deploy_sources_leaf = evaluator.add_custom_node(
        result=bool(deploy) and _has_any_sources([
            deploy.labor_sources if deploy else [],
            deploy.make_ready_sources if deploy else [],
            deploy.timeline_sources if deploy else [],
        ]),
        id="Deployment_Method_Source",
        desc="Provides reference URL(s) supporting the deployment method comparison",
        parent=deploy_node,
        critical=True
    )


async def build_federal_funding_nodes(evaluator: Evaluator, parent_node, funding: Optional[FederalFundingExtraction]) -> None:
    funding_node = evaluator.add_parallel(
        id="Federal_Funding_Context",
        desc="Analysis of relevant federal broadband funding programs and their impact on the project timeline",
        parent=parent_node,
        critical=True
    )

    # BEAD program overview ($42.45B)
    bead_seq = evaluator.add_sequential(
        id="BEAD_Program_Overview_Main",
        desc="BEAD program total allocation",
        parent=funding_node,
        critical=True
    )
    bead_provided = evaluator.add_custom_node(
        result=bool(funding and funding.bead_total_allocation and funding.bead_sources),
        id="BEAD_Program_Overview_Provided",
        desc="BEAD allocation and sources are provided",
        parent=bead_seq,
        critical=True
    )
    bead_leaf = evaluator.add_leaf(
        id="BEAD_Program_Overview",
        desc="Identifies the BEAD program's total allocation of $42.45 billion as context for the broader fiber deployment landscape",
        parent=bead_seq,
        critical=True
    )
    bead_claim = f"The BEAD program's total allocation is {funding.bead_total_allocation}."
    await evaluator.verify(
        claim=bead_claim,
        node=bead_leaf,
        sources=funding.bead_sources if funding else [],
        additional_instruction="Verify that the source explicitly states the BEAD program total allocation (target: $42.45 billion)."
    )

    # Peak construction timeline (obligations by late 2025; peak 2026–2027)
    peak_seq = evaluator.add_sequential(
        id="Peak_Construction_Timeline_Main",
        desc="Peak construction timeline 2026–2027 following late-2025 obligations",
        parent=funding_node,
        critical=True
    )
    peak_provided = evaluator.add_custom_node(
        result=bool(funding and funding.peak_construction_timeline and funding.funding_timeline_sources),
        id="Peak_Construction_Timeline_Provided",
        desc="Peak timeline and sources are provided",
        parent=peak_seq,
        critical=True
    )
    peak_leaf = evaluator.add_leaf(
        id="Peak_Construction_Timeline",
        desc="Provides the expected timeline for peak construction activity, noting that states are expected to obligate most BEAD funds by late 2025 with peak construction occurring in 2026-2027",
        parent=peak_seq,
        critical=True
    )
    peak_claim = "States are expected to obligate most BEAD funds by late 2025, with peak construction occurring in 2026–2027."
    await evaluator.verify(
        claim=peak_claim,
        node=peak_leaf,
        sources=funding.funding_timeline_sources if funding else [],
        additional_instruction="Verify that the source(s) state obligations or awards by late 2025 and indicate peak construction in 2026–2027."
    )

    # Overall funding sources presence
    funding_sources_leaf = evaluator.add_custom_node(
        result=bool(funding) and _has_any_sources([
            funding.bead_sources if funding else [],
            funding.funding_timeline_sources if funding else [],
        ]),
        id="Funding_Context_Source",
        desc="Provides reference URL(s) supporting the federal funding information",
        parent=funding_node,
        critical=True
    )


async def build_reliability_nodes(evaluator: Evaluator, parent_node, reliability: Optional[ReliabilityExtraction]) -> None:
    # Parent set to non-critical to allow a non-critical child inside (RTO/downtime)
    rel_node = evaluator.add_parallel(
        id="Infrastructure_Reliability_Requirements",
        desc="Analysis of network reliability and redundancy considerations based on recent telecommunications outages",
        parent=parent_node,
        critical=False
    )

    # Redundancy best practices
    red_seq = evaluator.add_sequential(
        id="Redundancy_Best_Practices_Main",
        desc="Best practices for network redundancy",
        parent=rel_node,
        critical=True
    )
    red_provided = evaluator.add_custom_node(
        result=bool(reliability and reliability.redundancy_best_practices and len(reliability.redundancy_best_practices) > 0 and reliability.redundancy_sources),
        id="Redundancy_Best_Practices_Provided",
        desc="Redundancy practices and sources are provided",
        parent=red_seq,
        critical=True
    )
    red_leaf = evaluator.add_leaf(
        id="Redundancy_Best_Practices",
        desc="References industry best practices for network redundancy, such as using different service providers or diverse routing paths",
        parent=red_seq,
        critical=True
    )
    red_claim = "Industry best practices for network redundancy include using diverse routing paths and multiple service providers."
    await evaluator.verify(
        claim=red_claim,
        node=red_leaf,
        sources=reliability.redundancy_sources if reliability else [],
        additional_instruction="Confirm that the source(s) explicitly recommend diverse routing (path diversity) and/or multiple providers for redundancy."
    )

    # Outage risk awareness (major outage in 2026)
    out_seq = evaluator.add_sequential(
        id="Outage_Risk_Awareness_Main",
        desc="Awareness of a major telecom outage in 2026",
        parent=rel_node,
        critical=True
    )
    out_provided = evaluator.add_custom_node(
        result=bool(reliability and reliability.outage_2026_event and reliability.outage_sources),
        id="Outage_Risk_Awareness_Provided",
        desc="2026 outage event and sources are provided",
        parent=out_seq,
        critical=True
    )
    out_leaf = evaluator.add_leaf(
        id="Outage_Risk_Awareness",
        desc="References at least one major telecommunications outage from 2026 (Verizon January 2026 or Azure February 2026) as context for reliability planning",
        parent=out_seq,
        critical=True
    )
    out_claim = f"In 2026, there was a major telecommunications outage: {reliability.outage_2026_event}."
    await evaluator.verify(
        claim=out_claim,
        node=out_leaf,
        sources=reliability.outage_sources if reliability else [],
        additional_instruction="Verify that the cited source(s) report a major outage event in 2026 (e.g., Verizon January 2026 or Azure February 2026)."
    )

    # Recovery time considerations (non-critical)
    rto_seq = evaluator.add_sequential(
        id="Recovery_Time_Considerations_Main",
        desc="Recovery time objective (RTO) or acceptable downtime parameters",
        parent=rel_node,
        critical=False
    )
    rto_provided = evaluator.add_custom_node(
        result=bool(reliability and reliability.recovery_time_considerations and reliability.recovery_sources),
        id="Recovery_Time_Considerations_Provided",
        desc="Recovery time considerations and sources are provided",
        parent=rto_seq,
        critical=False
    )
    rto_leaf = evaluator.add_leaf(
        id="Recovery_Time_Considerations",
        desc="Discusses Recovery Time Objective (RTO) or acceptable downtime parameters for network infrastructure",
        parent=rto_seq,
        critical=False
    )
    rto_claim = "Recovery Time Objective (RTO) and acceptable downtime parameters should be defined for network infrastructure planning."
    await evaluator.verify(
        claim=rto_claim,
        node=rto_leaf,
        sources=reliability.recovery_sources if reliability else [],
        additional_instruction="Verify that the source(s) discuss RTO or acceptable downtime for network or IT infrastructure. Accept synonymous terms like MTTR or downtime targets."
    )

    # Overall reliability sources presence
    rel_sources_leaf = evaluator.add_custom_node(
        result=bool(reliability) and _has_any_sources([
            reliability.redundancy_sources if reliability else [],
            reliability.outage_sources if reliability else [],
            reliability.recovery_sources if reliability else [],
        ]),
        id="Reliability_Source",
        desc="Provides reference URL(s) supporting the reliability and redundancy information",
        parent=rel_node,
        critical=True
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
    Evaluate an answer for the 2026 FTTH feasibility analysis task.
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

    # Extract structured analysis information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_ftth_analysis(),
        template_class=FTTHAnalysisExtraction,
        extraction_name="ftth_analysis_extraction",
    )

    # Root node for the full analysis (non-critical to allow partial credit across major sections)
    analysis_root = evaluator.add_parallel(
        id="Fiber_Deployment_Project_Analysis",
        desc="Complete analysis of a 2026 fiber optic deployment project including cost projections, tax incentive calculations, deployment method comparison, and infrastructure requirements",
        parent=root,
        critical=False
    )

    # Build subtrees per rubric sections
    await build_cost_projection_nodes(evaluator, analysis_root, extracted.cost)
    await build_tax_incentive_nodes(evaluator, analysis_root, extracted.tax)
    await build_deployment_method_nodes(evaluator, analysis_root, extracted.deploy)
    await build_federal_funding_nodes(evaluator, analysis_root, extracted.funding)
    await build_reliability_nodes(evaluator, analysis_root, extracted.reliability)

    # Optional: Add expected benchmarks for transparency (not used in scoring)
    evaluator.add_custom_info(
        info={
            "expected_benchmarks": {
                "cost_ratio_expected": "Underground ≈ 2x aerial",
                "labor_shares_expected": {"underground": "~72%", "aerial": "~64%"},
                "bead_total_expected": "$42.45 billion",
                "peak_construction_expected": "2026–2027 (after late-2025 obligations)",
                "outage_examples_2026": ["Verizon January 2026", "Azure February 2026"]
            }
        },
        info_type="benchmarks",
        info_name="expected_benchmarks"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()