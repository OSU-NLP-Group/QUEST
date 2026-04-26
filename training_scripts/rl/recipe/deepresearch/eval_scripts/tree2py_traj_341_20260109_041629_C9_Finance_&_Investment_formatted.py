import asyncio
import logging
from typing import Optional, List, Dict, Any, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "asset_manager_esg_due_diligence"
TASK_DESCRIPTION = """Identify a global asset management firm with Assets Under Management (AUM) exceeding $100 billion USD that offers ESG-integrated equity investment strategies and satisfies the following comprehensive due diligence requirements:

Firm Requirements:
1. Operates in multiple international markets
2. Offers dedicated ESG-integrated equity strategies

ESG Framework & Standards:
3. Has adopted at least one major ESG reporting framework: TCFD (Task Force on Climate-related Financial Disclosures), GRI (Global Reporting Initiative), ISSB IFRS S1/S2, or CSRD/ESRS
4. Is a signatory to the UN Principles for Responsible Investment (PRI)

Climate Strategy:
5. Has committed to or validated Science-Based Targets with the Science Based Targets initiative (SBTi)
6. Climate targets cover at least 95% of Scope 1 and Scope 2 emissions, and include Scope 3 emissions if they represent 40% or more of total emissions
7. Has made a publicly stated net-zero commitment with a specific target year

Portfolio Climate Metrics:
8. Reports Weighted Average Carbon Intensity (WACI) for portfolios in tons CO2e per $million revenue
9. Discloses temperature alignment scoring or Implied Temperature Rise (ITR) for investment portfolios
10. Calculates financed emissions using the Partnership for Carbon Accounting Financials (PCAF) methodology

ESG Ratings:
11. Is rated by at least two major ESG rating agencies from: MSCI, Sustainalytics, S&P Global ESG, ISS ESG, or CDP
12. Discloses to the CDP Climate Change questionnaire

ESG Integration Process:
13. Documents how ESG factors are integrated into the investment analysis and decision-making process
14. Identifies and assesses material ESG factors per sector or industry
15. Defines exclusionary screening criteria with specific thresholds for controversial activities

Stewardship & Engagement:
16. Makes proxy voting guidelines on ESG issues publicly available
17. Documents a climate engagement strategy with defined escalation procedures
18. Participates in collaborative engagement initiatives such as Climate Action 100+ or the Institutional Investors Group on Climate Change (IIGCC)

Impact & Nature:
19. Aligns impact metrics with recognized frameworks such as IRIS+, IMP Five Dimensions, or SDG alignment methodology
20. For nature-focused strategies, demonstrates alignment with the TNFD (Taskforce on Nature-related Financial Disclosures) framework or states commitment to adopt TNFD

Reporting & Transparency:
21. Reports comprehensive ESG metrics at least annually in a dedicated sustainability report or integrated report
22. Makes ESG reports and key policies publicly available on the firm's website
23. Follows recognized ESG reporting standards such as GRI, SASB, ISSB, or equivalent

Verification:
24. Obtains independent third-party verification or assurance for ESG data following standards such as ISAE 3000 or equivalent

Provide the name of the asset management firm that meets all critical requirements (items 1-2, 3, 5-6, 8-9, 11, 13-14, 16-17, 21-22) and as many additional requirements as possible, along with supporting documentation URLs for verification."""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #

class FirmSelection(BaseModel):
    firm_name: Optional[str] = None
    firm_homepage_url: Optional[str] = None
    general_support_urls: List[str] = Field(default_factory=list)


class AUMInfo(BaseModel):
    exceeds_100b: Optional[bool] = None
    stated_aum_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class GlobalPresenceInfo(BaseModel):
    operates_in_multiple_international_markets: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class ESGEquityInfo(BaseModel):
    offers_esg_integrated_equity_strategies: Optional[bool] = None
    examples: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class ESGFrameworksInfo(BaseModel):
    adopted_frameworks: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    tcfd_four_pillars_covered: Optional[bool] = None
    tcfd_sources: List[str] = Field(default_factory=list)


class SBTiInfo(BaseModel):
    committed_or_validated: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class EmissionsTargetInfo(BaseModel):
    covers_95_percent_scope1_2: Optional[bool] = None
    scope3_included_if_over_40_percent: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class WACIInfo(BaseModel):
    reports_waci: Optional[bool] = None
    unit_is_tCO2e_per_million_usd_revenue: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class TempAlignInfo(BaseModel):
    discloses_temperature_alignment_or_itr: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class ESGRatingsInfo(BaseModel):
    agencies: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class PRIInfo(BaseModel):
    signatory: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class NetZeroInfo(BaseModel):
    commitment_made: Optional[bool] = None
    target_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PCAFInfo(BaseModel):
    uses_pcaf: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class CDPInfo(BaseModel):
    discloses_to_cdp: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class IntegrationInfo(BaseModel):
    documented_investment_process_integration: Optional[bool] = None
    docs_urls: List[str] = Field(default_factory=list)


class MaterialSectorInfo(BaseModel):
    material_esg_by_sector: Optional[bool] = None
    docs_urls: List[str] = Field(default_factory=list)


class ProxyVotingInfo(BaseModel):
    guidelines_public: Optional[bool] = None
    docs_urls: List[str] = Field(default_factory=list)


class EngagementInfo(BaseModel):
    climate_engagement_with_escalation: Optional[bool] = None
    docs_urls: List[str] = Field(default_factory=list)


class ReportingInfo(BaseModel):
    annual_reporting: Optional[bool] = None
    report_urls: List[str] = Field(default_factory=list)


class PublicOnWebsiteInfo(BaseModel):
    reports_publicly_available: Optional[bool] = None
    urls: List[str] = Field(default_factory=list)


class RecognizedStandardsInfo(BaseModel):
    standards: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class ExclusionaryScreeningInfo(BaseModel):
    defined_thresholds: Optional[bool] = None
    docs_urls: List[str] = Field(default_factory=list)


class CollaborativeEngagementInfo(BaseModel):
    initiatives: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class ImpactMetricsInfo(BaseModel):
    frameworks: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class TNFDInfo(BaseModel):
    nature_strategies_tnfd_aligned_or_commit: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class AssuranceInfo(BaseModel):
    independent_assurance: Optional[bool] = None
    standard: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FirmDueDiligenceExtraction(BaseModel):
    firm: FirmSelection = Field(default_factory=FirmSelection)
    aum: AUMInfo = Field(default_factory=AUMInfo)
    global_presence: GlobalPresenceInfo = Field(default_factory=GlobalPresenceInfo)
    esg_equity: ESGEquityInfo = Field(default_factory=ESGEquityInfo)
    esg_frameworks: ESGFrameworksInfo = Field(default_factory=ESGFrameworksInfo)
    sbti: SBTiInfo = Field(default_factory=SBTiInfo)
    emissions_targets: EmissionsTargetInfo = Field(default_factory=EmissionsTargetInfo)
    waci: WACIInfo = Field(default_factory=WACIInfo)
    temperature_alignment: TempAlignInfo = Field(default_factory=TempAlignInfo)
    esg_ratings: ESGRatingsInfo = Field(default_factory=ESGRatingsInfo)
    pri: PRIInfo = Field(default_factory=PRIInfo)
    net_zero: NetZeroInfo = Field(default_factory=NetZeroInfo)
    pcaf: PCAFInfo = Field(default_factory=PCAFInfo)
    cdp: CDPInfo = Field(default_factory=CDPInfo)
    integration: IntegrationInfo = Field(default_factory=IntegrationInfo)
    materiality: MaterialSectorInfo = Field(default_factory=MaterialSectorInfo)
    proxy_voting: ProxyVotingInfo = Field(default_factory=ProxyVotingInfo)
    engagement: EngagementInfo = Field(default_factory=EngagementInfo)
    annual_reporting: ReportingInfo = Field(default_factory=ReportingInfo)
    public_on_website: PublicOnWebsiteInfo = Field(default_factory=PublicOnWebsiteInfo)
    recognized_standards: RecognizedStandardsInfo = Field(default_factory=RecognizedStandardsInfo)
    exclusionary_screening: ExclusionaryScreeningInfo = Field(default_factory=ExclusionaryScreeningInfo)
    collaborative_engagement: CollaborativeEngagementInfo = Field(default_factory=CollaborativeEngagementInfo)
    impact_metrics: ImpactMetricsInfo = Field(default_factory=ImpactMetricsInfo)
    tnfd: TNFDInfo = Field(default_factory=TNFDInfo)
    assurance: AssuranceInfo = Field(default_factory=AssuranceInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #

def prompt_extract_due_diligence() -> str:
    return """
Extract the firm's identity and due diligence evidence from the answer. Return a single JSON object following exactly the provided schema. Use null for missing scalar fields and empty arrays for missing lists. Critically: include explicit URLs (not just site names) for all 'sources' or 'docs_urls' fields whenever they appear in the answer. If multiple URLs are provided for a requirement, include them all.

Schema to extract:

{
  "firm": {
    "firm_name": string or null,
    "firm_homepage_url": string or null,
    "general_support_urls": string[]    // any general firm-level URLs cited in the answer
  },
  "aum": {
    "exceeds_100b": boolean or null,
    "stated_aum_text": string or null,  // e.g., "$1.2 trillion as of Dec 31, 2024"
    "sources": string[]                 // URLs that substantiate AUM
  },
  "global_presence": {
    "operates_in_multiple_international_markets": boolean or null,
    "sources": string[]
  },
  "esg_equity": {
    "offers_esg_integrated_equity_strategies": boolean or null,
    "examples": string[],               // names of strategies if provided
    "sources": string[]
  },
  "esg_frameworks": {
    "adopted_frameworks": string[],     // e.g., ["TCFD", "GRI", "ISSB IFRS S1/S2", "CSRD/ESRS"]
    "sources": string[],                // URLs showing framework adoption
    "tcfd_four_pillars_covered": boolean or null,
    "tcfd_sources": string[]            // URLs showing TCFD four pillars, if applicable
  },
  "sbti": {
    "committed_or_validated": boolean or null,
    "sources": string[]
  },
  "emissions_targets": {
    "covers_95_percent_scope1_2": boolean or null,
    "scope3_included_if_over_40_percent": boolean or null,
    "sources": string[]
  },
  "waci": {
    "reports_waci": boolean or null,
    "unit_is_tCO2e_per_million_usd_revenue": boolean or null,
    "sources": string[]
  },
  "temperature_alignment": {
    "discloses_temperature_alignment_or_itr": boolean or null,
    "sources": string[]
  },
  "esg_ratings": {
    "agencies": string[],               // e.g., ["MSCI", "Sustainalytics", "S&P Global ESG", "ISS ESG", "CDP"]
    "sources": string[]                 // rating profile pages or firm docs referencing ratings
  },
  "pri": {
    "signatory": boolean or null,
    "sources": string[]
  },
  "net_zero": {
    "commitment_made": boolean or null,
    "target_year": string or null,      // e.g., "2040"
    "sources": string[]
  },
  "pcaf": {
    "uses_pcaf": boolean or null,
    "sources": string[]
  },
  "cdp": {
    "discloses_to_cdp": boolean or null,
    "sources": string[]
  },
  "integration": {
    "documented_investment_process_integration": boolean or null,
    "docs_urls": string[]
  },
  "materiality": {
    "material_esg_by_sector": boolean or null,
    "docs_urls": string[]
  },
  "proxy_voting": {
    "guidelines_public": boolean or null,
    "docs_urls": string[]
  },
  "engagement": {
    "climate_engagement_with_escalation": boolean or null,
    "docs_urls": string[]
  },
  "annual_reporting": {
    "annual_reporting": boolean or null,
    "report_urls": string[]
  },
  "public_on_website": {
    "reports_publicly_available": boolean or null,
    "urls": string[]
  },
  "recognized_standards": {
    "standards": string[],              // e.g., ["GRI", "SASB", "ISSB"]
    "sources": string[]
  },
  "exclusionary_screening": {
    "defined_thresholds": boolean or null,
    "docs_urls": string[]
  },
  "collaborative_engagement": {
    "initiatives": string[],            // e.g., ["Climate Action 100+", "IIGCC"]
    "sources": string[]
  },
  "impact_metrics": {
    "frameworks": string[],             // e.g., ["IRIS+", "IMP", "SDG"]
    "sources": string[]
  },
  "tnfd": {
    "nature_strategies_tnfd_aligned_or_commit": boolean or null,
    "sources": string[]
  },
  "assurance": {
    "independent_assurance": boolean or null,
    "standard": string or null,         // e.g., "ISAE 3000"
    "sources": string[]
  }
}

Rules:
- Do NOT invent URLs. Only include URLs explicitly present in the answer.
- Normalize markdown links to raw URLs.
- When frameworks, standards, or agencies are mentioned, extract their names exactly as presented.
- If the answer provides multiple candidate firms, select the primary one explicitly advocated in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #

def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    s: Set[str] = set()
    for lst in lists:
        if lst:
            for u in lst:
                if isinstance(u, str) and u.strip():
                    s.add(u.strip())
    return list(s)


def _collect_all_urls(data: FirmDueDiligenceExtraction) -> List[str]:
    # Gather every URL list field into one big flattened list
    urls = []
    urls += data.firm.general_support_urls or []
    if data.firm.firm_homepage_url:
        urls.append(data.firm.firm_homepage_url)

    urls += data.aum.sources or []
    urls += data.global_presence.sources or []
    urls += data.esg_equity.sources or []
    urls += data.esg_frameworks.sources or []
    urls += data.esg_frameworks.tcfd_sources or []
    urls += data.sbti.sources or []
    urls += data.emissions_targets.sources or []
    urls += data.waci.sources or []
    urls += data.temperature_alignment.sources or []
    urls += data.esg_ratings.sources or []
    urls += data.pri.sources or []
    urls += data.net_zero.sources or []
    urls += data.pcaf.sources or []
    urls += data.cdp.sources or []
    urls += data.integration.docs_urls or []
    urls += data.materiality.docs_urls or []
    urls += data.proxy_voting.docs_urls or []
    urls += data.engagement.docs_urls or []
    urls += data.annual_reporting.report_urls or []
    urls += data.public_on_website.urls or []
    urls += data.recognized_standards.sources or []
    urls += data.exclusionary_screening.docs_urls or []
    urls += data.collaborative_engagement.sources or []
    urls += data.impact_metrics.sources or []
    urls += data.tnfd.sources or []
    urls += data.assurance.sources or []
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


async def _add_and_verify(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str] | str],
    critical: bool,
    additional_instruction: str = "None",
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Build verification tree                                                     #
# --------------------------------------------------------------------------- #

async def build_critical_checks(
    evaluator: Evaluator,
    parent,
    data: FirmDueDiligenceExtraction,
) -> None:
    """
    Build all critical requirements: items 1-2, 3, 5-6, 8-9, 11, 13-14, 16-17, 21-22 plus mandatory AUM threshold.
    We design this as a critical parallel node; all its children are critical leaves.
    """
    firm_name = data.firm.firm_name or "the firm"
    fallback = _merge_sources(data.firm.general_support_urls, [data.firm.firm_homepage_url] if data.firm.firm_homepage_url else [])

    crit_node = evaluator.add_parallel(
        id="critical_requirements",
        desc="Firm satisfies all critical due diligence requirements (plus AUM threshold).",
        parent=parent,
        critical=True,  # IMPORTANT: children must be critical
    )

    # AUM > $100B
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="aum_threshold",
        desc="Firm has Assets Under Management (AUM) exceeding $100 billion USD.",
        claim=f"{firm_name} has AUM exceeding $100 billion USD. The answer cites: '{data.aum.stated_aum_text or 'N/A'}'.",
        sources=_merge_sources(data.aum.sources, fallback),
        critical=True,
        additional_instruction="Verify that the firm's AUM exceeds $100B USD (allowing approximate wording like 'over $100 billion', 'hundreds of billions', or explicit values above 100B).",
    )

    # Global presence (multiple international markets)
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="global_presence",
        desc="Firm operates in multiple international markets.",
        claim=f"{firm_name} operates in multiple international markets (has operations, offices, or regulatory approvals across multiple countries/regions).",
        sources=_merge_sources(data.global_presence.sources, fallback),
        critical=True,
        additional_instruction="Look for explicit evidence of a multi-country footprint (e.g., offices across regions, global regulatory registrations, or services offered internationally).",
    )

    # ESG-integrated equity strategies
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="esg_equity_strategies",
        desc="Firm offers dedicated ESG-integrated equity investment strategies.",
        claim=f"{firm_name} offers dedicated ESG-integrated equity investment strategies (examples: {', '.join(data.esg_equity.examples) if data.esg_equity.examples else 'N/A'}).",
        sources=_merge_sources(data.esg_equity.sources, fallback),
        critical=True,
        additional_instruction="Confirm existence of named equity strategies that explicitly integrate ESG factors in the investment process.",
    )

    # ESG framework adoption (at least one of TCFD, GRI, ISSB IFRS S1/S2, CSRD/ESRS)
    frameworks_list = data.esg_frameworks.adopted_frameworks or []
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="esg_framework_adoption",
        desc="Firm has adopted at least one major ESG reporting framework (TCFD, GRI, ISSB IFRS S1/S2, or CSRD/ESRS).",
        claim=f"{firm_name} has adopted at least one major ESG reporting framework from [TCFD, GRI, ISSB IFRS S1/S2, CSRD/ESRS]. Disclosed frameworks: {frameworks_list}.",
        sources=_merge_sources(data.esg_frameworks.sources, fallback),
        critical=True,
        additional_instruction="At least one among TCFD, GRI, ISSB IFRS S1/S2, CSRD/ESRS must be clearly adopted.",
    )

    # If TCFD is cited, coverage of four pillars
    tcfd_claimed = any(str(framework).strip().upper().startswith("TCFD") for framework in frameworks_list)
    if tcfd_claimed:
        tcfd_claim_text = f"{firm_name}'s TCFD disclosure covers all four pillars: Governance, Strategy, Risk Management, and Metrics & Targets."
        tcfd_sources = _merge_sources(data.esg_frameworks.tcfd_sources, data.esg_frameworks.sources, fallback)
        add_ins = "Confirm that the TCFD disclosure explicitly covers all four pillars: Governance, Strategy, Risk Management, and Metrics & Targets."
    else:
        tcfd_claim_text = f"This TCFD four-pillar coverage requirement is not applicable because {firm_name} does not claim TCFD adoption; consider this requirement satisfied as N/A."
        tcfd_sources = None
        add_ins = "The answer does not claim TCFD adoption; treat this check as not applicable and satisfied."

    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="tcfd_four_pillars_if_applicable",
        desc="If TCFD is cited, disclosure covers all four pillars.",
        claim=tcfd_claim_text,
        sources=tcfd_sources,
        critical=True,
        additional_instruction=add_ins,
    )

    # SBTi commitment/validation
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="sbti_commitment",
        desc="Firm has committed to or validated Science-Based Targets with SBTi.",
        claim=f"{firm_name} has committed to (or has validated) Science-Based Targets with SBTi.",
        sources=_merge_sources(data.sbti.sources, fallback),
        critical=True,
        additional_instruction="Confirm SBTi commitment or validation on SBTi site or firm's disclosures.",
    )

    # Emissions target coverage
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="emissions_target_coverage",
        desc="Climate targets cover >=95% of Scope 1 & 2; include Scope 3 if it is >=40% of total emissions.",
        claim=f"{firm_name}'s climate targets cover at least 95% of Scope 1 and Scope 2 emissions; and include Scope 3 if it represents 40% or more of total emissions.",
        sources=_merge_sources(data.emissions_targets.sources, fallback),
        critical=True,
        additional_instruction="Look for explicit coverage percentages or statements satisfying the coverage rule for Scope 1/2 and the conditional inclusion of Scope 3.",
    )

    # WACI
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="waci_reporting",
        desc="Portfolios report WACI in tons CO2e per $million revenue.",
        claim=f"{firm_name} reports portfolio WACI and the unit is tons CO2e per $million revenue.",
        sources=_merge_sources(data.waci.sources, fallback),
        critical=True,
        additional_instruction="Verify that WACI is disclosed and explicitly stated in 'tCO2e per $M revenue' (or equivalent phrasing).",
    )

    # Temperature alignment / ITR
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="temperature_alignment",
        desc="Temperature alignment or Implied Temperature Rise (ITR) is disclosed for investment portfolios.",
        claim=f"{firm_name} discloses temperature alignment scoring or Implied Temperature Rise (ITR) for investment portfolios.",
        sources=_merge_sources(data.temperature_alignment.sources, fallback),
        critical=True,
        additional_instruction="Look for 'temperature alignment', 'implied temperature rise', or similar terminology in portfolio metrics.",
    )

    # ESG ratings: at least two among MSCI, Sustainalytics, S&P Global ESG, ISS ESG, CDP
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="esg_ratings",
        desc="Rated by at least two major ESG rating agencies (MSCI, Sustainalytics, S&P Global ESG, ISS ESG, or CDP).",
        claim=f"{firm_name} is rated by at least two among [MSCI, Sustainalytics, S&P Global ESG, ISS ESG, CDP]. Agencies listed in the answer: {data.esg_ratings.agencies or []}.",
        sources=_merge_sources(data.esg_ratings.sources, fallback),
        critical=True,
        additional_instruction="Confirm at least two ratings from the specified set. Ratings pages or credible disclosures referencing ratings are acceptable.",
    )

    # ESG Integration documented
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="esg_integration_documented",
        desc="Investment process documents how ESG factors are integrated into analysis and decision-making.",
        claim=f"{firm_name}'s investment process documentation describes how ESG factors are integrated into investment analysis and decision-making.",
        sources=_merge_sources(data.integration.docs_urls, fallback),
        critical=True,
        additional_instruction="Look for investment process or ESG integration documents that clearly describe the integration approach.",
    )

    # Material ESG by sector/industry
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="material_esg_by_sector",
        desc="Material ESG factors are identified and assessed per sector or industry.",
        claim=f"{firm_name} identifies and assesses material ESG factors by sector or industry.",
        sources=_merge_sources(data.materiality.docs_urls, fallback),
        critical=True,
        additional_instruction="Check for sector- or industry-specific ESG materiality assessments or frameworks.",
    )

    # Proxy voting guidelines
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="proxy_voting_guidelines",
        desc="Proxy voting guidelines on ESG issues are publicly available.",
        claim=f"{firm_name} makes proxy voting guidelines on ESG issues publicly available.",
        sources=_merge_sources(data.proxy_voting.docs_urls, fallback),
        critical=True,
        additional_instruction="Look for a public proxy voting policy/guidelines document covering ESG issues.",
    )

    # Climate engagement with escalation
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="climate_engagement_escalation",
        desc="Climate engagement strategy with defined escalation procedures is documented.",
        claim=f"{firm_name} documents a climate engagement strategy with defined escalation procedures.",
        sources=_merge_sources(data.engagement.docs_urls, fallback),
        critical=True,
        additional_instruction="Confirm a written escalation framework in engagements (e.g., voting against, filing resolutions, divestment as last resort).",
    )

    # Annual ESG reporting
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="annual_esg_reporting",
        desc="Comprehensive ESG metrics are reported at least annually.",
        claim=f"{firm_name} reports comprehensive ESG metrics at least annually in a sustainability or integrated report.",
        sources=_merge_sources(data.annual_reporting.report_urls, fallback),
        critical=True,
        additional_instruction="Confirm at least annual frequency and comprehensive ESG metrics in the report.",
    )

    # Reports and policies publicly available on website
    await _add_and_verify(
        evaluator,
        parent=crit_node,
        node_id="reports_public_on_website",
        desc="ESG reports and key policies are publicly available on the firm's website.",
        claim=f"{firm_name}'s ESG reports and key policies are publicly available on its website.",
        sources=_merge_sources(data.public_on_website.urls, fallback),
        critical=True,
        additional_instruction="Check that ESG reports and core policies can be accessed publicly without login/paywall.",
    )


async def build_additional_checks(
    evaluator: Evaluator,
    parent,
    data: FirmDueDiligenceExtraction,
) -> None:
    """
    Build non-critical (additional) requirements to allow partial credit.
    """
    firm_name = data.firm.firm_name or "the firm"
    fallback = _merge_sources(data.firm.general_support_urls, [data.firm.firm_homepage_url] if data.firm.firm_homepage_url else [])

    add_node = evaluator.add_parallel(
        id="additional_requirements",
        desc="Additional (non-critical) requirements satisfied where possible.",
        parent=parent,
        critical=False,
    )

    # PRI signatory
    await _add_and_verify(
        evaluator,
        parent=add_node,
        node_id="pri_signatory",
        desc="Firm is a signatory to the UN PRI.",
        claim=f"{firm_name} is a signatory to the UN Principles for Responsible Investment (PRI).",
        sources=_merge_sources(data.pri.sources, fallback),
        critical=False,
        additional_instruction="Verify PRI signatory status (PRI website signatory directory or firm's disclosure).",
    )

    # Net-zero target year
    await _add_and_verify(
        evaluator,
        parent=add_node,
        node_id="net_zero_target_year",
        desc="Firm has a publicly stated net-zero commitment with a specific target year.",
        claim=f"{firm_name} has a publicly stated net-zero commitment with target year: {data.net_zero.target_year or 'unspecified'}.",
        sources=_merge_sources(data.net_zero.sources, fallback),
        critical=False,
        additional_instruction="Look for a public net-zero pledge and a specific target year.",
    )

    # PCAF financed emissions
    await _add_and_verify(
        evaluator,
        parent=add_node,
        node_id="pcaf_financed_emissions",
        desc="Financed emissions are calculated using PCAF methodology.",
        claim=f"{firm_name} calculates financed emissions using the PCAF methodology.",
        sources=_merge_sources(data.pcaf.sources, fallback),
        critical=False,
        additional_instruction="Confirm explicit reference to PCAF for financed emissions accounting.",
    )

    # CDP disclosure
    await _add_and_verify(
        evaluator,
        parent=add_node,
        node_id="cdp_climate_disclosure",
        desc="Firm discloses to the CDP Climate Change questionnaire.",
        claim=f"{firm_name} discloses to the CDP Climate Change questionnaire.",
        sources=_merge_sources(data.cdp.sources, fallback),
        critical=False,
        additional_instruction="Look for CDP disclosure evidence (CDP site or firm's statement).",
    )

    # Exclusionary screening thresholds
    await _add_and_verify(
        evaluator,
        parent=add_node,
        node_id="exclusionary_screening_thresholds",
        desc="Exclusionary screening criteria are defined with specific thresholds for controversial activities.",
        claim=f"{firm_name} defines exclusionary screening criteria with specific thresholds for controversial activities.",
        sources=_merge_sources(data.exclusionary_screening.docs_urls, fallback),
        critical=False,
        additional_instruction="Check for explicit thresholds (e.g., revenue %, production %, or absolute limits) for activities like thermal coal, tobacco, controversial weapons.",
    )

    # Collaborative engagement participation
    await _add_and_verify(
        evaluator,
        parent=add_node,
        node_id="collaborative_engagement",
        desc="Firm participates in collaborative engagement initiatives such as CA100+ or IIGCC.",
        claim=f"{firm_name} participates in collaborative engagement initiatives (e.g., Climate Action 100+, IIGCC, or similar). Initiatives cited: {data.collaborative_engagement.initiatives or []}.",
        sources=_merge_sources(data.collaborative_engagement.sources, fallback),
        critical=False,
        additional_instruction="Confirm active participation or membership in recognized collaborative engagement initiatives.",
    )

    # Impact metrics framework
    await _add_and_verify(
        evaluator,
        parent=add_node,
        node_id="impact_metrics_framework",
        desc="Impact metrics align with recognized frameworks such as IRIS+, IMP, or SDG alignment methodology.",
        claim=f"{firm_name} aligns impact metrics with recognized frameworks (IRIS+, IMP Five Dimensions, or SDG alignment methodology). Frameworks cited: {data.impact_metrics.frameworks or []}.",
        sources=_merge_sources(data.impact_metrics.sources, fallback),
        critical=False,
        additional_instruction="Look for explicit mapping of impact KPIs to IRIS+ metrics, IMP dimensions, or SDG targets.",
    )

    # TNFD (if applicable)
    await _add_and_verify(
        evaluator,
        parent=add_node,
        node_id="tnfd_nature_if_applicable",
        desc="If nature-focused strategies are claimed, firm aligns with TNFD or commits to adopt TNFD.",
        claim=f"If {firm_name} has nature-focused strategies, it demonstrates alignment with TNFD or commits to adopt TNFD.",
        sources=_merge_sources(data.tnfd.sources, fallback),
        critical=False,
        additional_instruction="Verify TNFD alignment or commitment ONLY if nature/biodiversity strategies are claimed; otherwise this may legitimately be absent.",
    )

    # Recognized reporting standards (GRI, SASB, ISSB, etc.)
    await _add_and_verify(
        evaluator,
        parent=add_node,
        node_id="recognized_reporting_standards",
        desc="ESG reporting follows recognized standards such as GRI, SASB, ISSB, or equivalent.",
        claim=f"{firm_name}'s ESG reporting follows recognized standards such as {data.recognized_standards.standards or []}.",
        sources=_merge_sources(data.recognized_standards.sources, fallback),
        critical=False,
        additional_instruction="Look for explicit statements of reporting under GRI, SASB, ISSB, or equivalent frameworks in reports or policies.",
    )

    # Independent assurance (ISAE 3000 or equivalent)
    await _add_and_verify(
        evaluator,
        parent=add_node,
        node_id="independent_assurance_should",
        desc="ESG data undergoes independent third-party verification or assurance (ISAE 3000 or equivalent).",
        claim=f"{firm_name}'s ESG data undergoes independent third-party assurance (e.g., {data.assurance.standard or 'ISAE 3000 or equivalent'}).",
        sources=_merge_sources(data.assurance.sources, fallback),
        critical=False,
        additional_instruction="Look for assurance statements by audit/assurance firms and mention of ISAE 3000 or equivalent standards.",
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
    Evaluate an answer for the ESG due diligence asset manager identification task.
    Note: Root node is set to non-critical to allow mixing critical and non-critical children
    (framework requires all children of a critical node to be critical).
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

    # 1) Extraction
    extracted: FirmDueDiligenceExtraction = await evaluator.extract(
        prompt=prompt_extract_due_diligence(),
        template_class=FirmDueDiligenceExtraction,
        extraction_name="firm_due_diligence",
    )

    # 2) Root-level essential existence checks (critical)
    firm_named_node = evaluator.add_custom_node(
        result=bool(extracted.firm.firm_name and extracted.firm.firm_name.strip()),
        id="firm_named",
        desc="Response clearly names a specific global asset management firm being evaluated.",
        parent=root,
        critical=True,
    )

    all_urls = _collect_all_urls(extracted)
    urls_present_node = evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id="supporting_documentation_urls",
        desc="Response provides supporting documentation URL(s) for verification (at least one URL).",
        parent=root,
        critical=True,
    )

    # 3) Critical requirements block
    await build_critical_checks(evaluator, root, extracted)

    # 4) Additional requirements (non-critical)
    await build_additional_checks(evaluator, root, extracted)

    # 5) Return structured summary
    return evaluator.get_summary()