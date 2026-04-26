import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_hftd_emergency_preparedness"
TASK_DESCRIPTION = """
For telecommunications infrastructure resilience planning in California's High Fire Threat Districts (HFTD) Tier 2 and Tier 3 areas, compile a comprehensive emergency preparedness compliance documentation that includes:

1. Identification of at least two major wireline telecommunications providers that operate in HFTD Tier 2 and Tier 3 areas and are subject to California's enhanced backup power requirements, including their CPUC certificate numbers and evidence of their HFTD presence

2. California Public Utilities Commission backup power requirements, specifically:
   - The mandated backup power duration for facilities in HFTD Tier 2 and 3 areas with the relevant CPUC Decision number
   - Backup power durations required for central offices
   - Backup power durations required for remote terminals
   - Backup power requirements for wireless facilities

3. Federal Communications Commission backup power standards under 47 CFR § 9.20, including:
   - The minimum standby backup power duration for the 8-hour option
   - The minimum standby backup power duration for the 24-hour option requirement
   - The types of equipment that must be covered by backup power for 911 access
   - The rule's sunset date

4. FCC 911 reliability annual certification requirements for covered 911 service providers, including:
   - The three specific measures that must be certified
   - The URL of the certification portal where providers must file
   
5. Telecommunications outage notification and reporting requirements:
   - The maximum time allowed for notifying PSAPs after discovering an outage affecting 911 service
   - The time window for filing an initial communications outage report to the FCC after discovering a reportable outage

For all regulatory requirements listed above, provide the specific reference URLs to official FCC, CPUC, or Cornell Law School Legal Information Institute (LII) sources that document these requirements.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ProviderEntry(BaseModel):
    name: Optional[str] = None
    major_provider_evidence: Optional[str] = None
    wireline_evidence: Optional[str] = None
    facilities_based_evidence: Optional[str] = None
    operates_in_ca_evidence: Optional[str] = None
    cpuc_certificate_number: Optional[str] = None
    cpuc_certificate_url: Optional[str] = None
    hftd_presence_evidence: Optional[str] = None
    hftd_evidence_urls: List[str] = Field(default_factory=list)
    subject_to_d_21_02_029_evidence: Optional[str] = None
    subject_to_d_21_02_029_urls: List[str] = Field(default_factory=list)
    evidence_urls: List[str] = Field(default_factory=list)


class ProvidersExtraction(BaseModel):
    providers: List[ProviderEntry] = Field(default_factory=list)


class CPUCRequirements(BaseModel):
    hftd_tier2_3_duration_text: Optional[str] = None
    hftd_decision_number: Optional[str] = None
    hftd_official_urls: List[str] = Field(default_factory=list)

    central_office_duration_text: Optional[str] = None
    central_office_official_urls: List[str] = Field(default_factory=list)

    remote_terminal_min_text: Optional[str] = None
    remote_terminal_objective_text: Optional[str] = None
    remote_terminal_official_urls: List[str] = Field(default_factory=list)

    wireless_hftd_duration_text: Optional[str] = None
    wireless_official_urls: List[str] = Field(default_factory=list)


class CFR920Requirements(BaseModel):
    eight_hour_min_text: Optional[str] = None
    eight_hour_official_urls: List[str] = Field(default_factory=list)

    twentyfour_hour_min_text: Optional[str] = None
    twentyfour_hour_effective_date_text: Optional[str] = None
    twentyfour_hour_official_urls: List[str] = Field(default_factory=list)

    equipment_coverage_text: Optional[str] = None
    equipment_coverage_official_urls: List[str] = Field(default_factory=list)

    sunset_date_text: Optional[str] = None
    sunset_official_urls: List[str] = Field(default_factory=list)


class FCC911CertificationRequirements(BaseModel):
    measures: List[str] = Field(default_factory=list)
    measures_official_urls: List[str] = Field(default_factory=list)
    portal_url: Optional[str] = None
    portal_official_urls: List[str] = Field(default_factory=list)


class OutageRequirements(BaseModel):
    psap_notify_timeline_text: Optional[str] = None
    psap_official_urls: List[str] = Field(default_factory=list)

    initial_report_timeline_text: Optional[str] = None
    governing_authority_citation: Optional[str] = None
    outage_reporting_official_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_providers() -> str:
    return """
    From the answer, extract at least two distinct major, facilities-based WIRELINE telecommunications providers operating in California that serve HFTD Tier 2 and/or Tier 3 areas and are subject to CPUC D.21-02-029. For each provider, return:
    - name
    - major_provider_evidence: textual evidence mentioned (e.g., ILEC status, statewide footprint, subscriber counts)
    - wireline_evidence: textual evidence the provider is a wireline carrier
    - facilities_based_evidence: textual evidence the provider is facilities-based
    - operates_in_ca_evidence: textual evidence the provider operates in California
    - cpuc_certificate_number: CPUC certificate number (e.g., U-####-C), if provided in the answer
    - cpuc_certificate_url: URL to CPUC certificate or CPUC utility page if present in the answer
    - hftd_presence_evidence: textual evidence that the provider serves HFTD Tier 2/3, as described in the answer
    - hftd_evidence_urls: all URLs cited that support the HFTD presence (if any)
    - subject_to_d_21_02_029_evidence: textual claim that provider is subject to CPUC D.21-02-029
    - subject_to_d_21_02_029_urls: all URLs cited that support D.21-02-029 applicability
    - evidence_urls: any other URLs cited about the provider (company pages, CPUC pages, filings, maps, etc.)
    Return a JSON object with 'providers' as an array of provider entries. If some fields are not present in the answer, set them to null or empty list.
    """


def prompt_extract_cpuc_requirements() -> str:
    return """
    Extract California CPUC backup power requirements referenced in the answer. Return:
    - hftd_tier2_3_duration_text (e.g., '72 hours' if present)
    - hftd_decision_number (e.g., 'D.21-02-029')
    - hftd_official_urls: official CPUC URL(s) documenting the HFTD Tier 2/3 duration requirement
    - central_office_duration_text (e.g., '24 hours')
    - central_office_official_urls: official CPUC URL(s) for central office duration
    - remote_terminal_min_text (e.g., '4 hours')
    - remote_terminal_objective_text (e.g., '8 hours')
    - remote_terminal_official_urls: official CPUC URL(s) for remote terminal requirements
    - wireless_hftd_duration_text (e.g., '72 hours')
    - wireless_official_urls: official CPUC URL(s) for wireless HFTD duration
    Only include URLs explicitly cited in the answer. Prefer official CPUC domains.
    """


def prompt_extract_cfr920() -> str:
    return """
    Extract FCC backup power standards under 47 CFR § 9.20, as cited. Return:
    - eight_hour_min_text (e.g., '8 hours')
    - eight_hour_official_urls: official FCC/eCFR/Cornell LII URL(s) for the 8-hour option
    - twentyfour_hour_min_text (e.g., '24 hours')
    - twentyfour_hour_effective_date_text (e.g., 'February 13, 2019')
    - twentyfour_hour_official_urls: official URL(s) for the 24-hour option requirement/effective date
    - equipment_coverage_text (e.g., 'provider-furnished equipment necessary for 911 access')
    - equipment_coverage_official_urls: official URL(s) for equipment coverage
    - sunset_date_text (e.g., 'September 1, 2025')
    - sunset_official_urls: official URL(s) for the rule's sunset date
    Only include URLs explicitly cited in the answer. Prefer fcc.gov, ecfr.gov, or law.cornell.edu.
    """


def prompt_extract_911_cert() -> str:
    return """
    Extract FCC 911 reliability annual certification requirements. Return:
    - measures: list of the three measures that must be certified (e.g., 'circuit diversity', 'central office backup power', 'network monitoring')
    - measures_official_urls: official FCC URL(s) documenting these measures
    - portal_url: the filing portal URL (e.g., 'https://apps2.fcc.gov/rcs911/')
    - portal_official_urls: official FCC URL(s) referencing or documenting the portal
    Only include URLs explicitly cited in the answer. Prefer fcc.gov.
    """


def prompt_extract_outage() -> str:
    return """
    Extract outage notification and reporting timelines. Return:
    - psap_notify_timeline_text (e.g., 'within 30 minutes')
    - psap_official_urls: official FCC/eCFR/Cornell LII URL(s) documenting PSAP notification timeline
    - initial_report_timeline_text (e.g., 'within 72 hours')
    - governing_authority_citation (e.g., '47 CFR Part 4')
    - outage_reporting_official_urls: official URL(s) for 47 CFR Part 4 outage reporting requirements
    Only include URLs explicitly cited in the answer. Prefer fcc.gov, ecfr.gov, law.cornell.edu.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _host(url: str) -> str:
    try:
        parsed = urlparse(url if (url.startswith("http://") or url.startswith("https://")) else "http://" + url)
        return parsed.hostname or ""
    except Exception:
        return ""


def _any_official(urls: List[str], allowed_domains: List[str]) -> bool:
    for u in urls:
        h = _host(u).lower()
        if any(h.endswith(dom) for dom in allowed_domains):
            return True
    return False


def _combine_urls(*url_lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        if lst:
            combined.extend(lst)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in combined:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_provider(evaluator: Evaluator, parent_node, provider: ProviderEntry, idx: int) -> None:
    prov_label = f"Provider_{idx+1}"
    name = provider.name or f"Provider #{idx+1}"

    node = evaluator.add_parallel(
        id=prov_label,
        desc=f"Provider entry #{idx+1} completeness and qualification.",
        parent=parent_node,
        critical=True,
    )

    # P*_Provider_Name
    evaluator.add_custom_node(
        result=_non_empty_str(provider.name),
        id=f"P{idx+1}_Provider_Name",
        desc=f"Provider #{idx+1} is explicitly named.",
        parent=node,
        critical=True,
    )

    provider_urls = _combine_urls(
        provider.evidence_urls,
        [provider.cpuc_certificate_url] if _non_empty_str(provider.cpuc_certificate_url) else [],
        provider.subject_to_d_21_02_029_urls,
        provider.hftd_evidence_urls,
    )

    # P*_Major_Provider_Evidence
    leaf_major = evaluator.add_leaf(
        id=f"P{idx+1}_Major_Provider_Evidence",
        desc=f"Provider #{idx+1} is supported as a 'major' provider via stated evidence.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Public evidence shows that {name} is a major, statewide or incumbent/local exchange carrier or otherwise a major facilities-based wireline provider in California.",
        node=leaf_major,
        sources=provider_urls,
        additional_instruction="Accept ILEC/CLEC statewide footprint, significant subscriber/line counts, or comparable indicators as evidence of 'major'. Focus on the provided URLs."
    )

    # P*_Wireline_Carrier
    leaf_wireline = evaluator.add_leaf(
        id=f"P{idx+1}_Wireline_Carrier",
        desc=f"Provider #{idx+1} is identified as a wireline telecommunications carrier.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} is a wireline telecommunications carrier.",
        node=leaf_wireline,
        sources=provider_urls,
        additional_instruction="Confirm the provider offers wireline (fixed) telecom services (e.g., ILEC/CLEC, fiber/copper-based voice/telecom)."
    )

    # P*_Facilities_Based
    leaf_fac = evaluator.add_leaf(
        id=f"P{idx+1}_Facilities_Based",
        desc=f"Provider #{idx+1} is identified as facilities-based.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} is a facilities-based provider (owns/controls network facilities for service delivery).",
        node=leaf_fac,
        sources=provider_urls,
        additional_instruction="Look for language such as 'facilities-based', ILEC, or indications the provider owns/manages its own network infrastructure."
    )

    # P*_Operates_In_California
    leaf_ca = evaluator.add_leaf(
        id=f"P{idx+1}_Operates_In_California",
        desc=f"Provider #{idx+1} is identified as operating in California.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} operates in California.",
        node=leaf_ca,
        sources=provider_urls,
        additional_instruction="Use CPUC pages or official/provider materials that explicitly show California operations."
    )

    # P*_CPUC_Certificate_Number
    evaluator.add_custom_node(
        result=_non_empty_str(provider.cpuc_certificate_number),
        id=f"P{idx+1}_CPUC_Certificate_Number",
        desc=f"Provider #{idx+1} CPUC certificate number is provided.",
        parent=node,
        critical=True,
    )

    # P*_HFTD_Tier_2_3_Presence_Evidence
    leaf_hftd = evaluator.add_leaf(
        id=f"P{idx+1}_HFTD_Tier_2_3_Presence_Evidence",
        desc=f"Evidence is provided that Provider #{idx+1} operates/serves in HFTD Tier 2 and/or Tier 3 areas.",
        parent=node,
        critical=True,
    )
    hftd_urls = provider.hftd_evidence_urls if provider.hftd_evidence_urls else provider_urls
    await evaluator.verify(
        claim=f"{name} operates or serves in California HFTD Tier 2 and/or Tier 3 areas.",
        node=leaf_hftd,
        sources=hftd_urls,
        additional_instruction="Accept evidence that the provider's service area covers HFTD Tier 2/3 regions (maps, CPUC filings, wildfire mitigation documents, or other credible references)."
    )

    # P*_Subject_To_D_21_02_029_Evidence
    leaf_d2102029 = evaluator.add_leaf(
        id=f"P{idx+1}_Subject_To_D_21_02_029_Evidence",
        desc=f"Evidence is provided that Provider #{idx+1} is subject to CPUC Decision 21-02-029 backup power requirements.",
        parent=node,
        critical=True,
    )
    d_urls = provider.subject_to_d_21_02_029_urls if provider.subject_to_d_21_02_029_urls else provider_urls
    await evaluator.verify(
        claim=f"{name} is subject to CPUC Decision 21-02-029 backup power requirements for HFTD Tier 2/3 areas.",
        node=leaf_d2102029,
        sources=d_urls,
        additional_instruction="Look for explicit statements tying the provider or the category it falls into to D.21-02-029 applicability."
    )

    # P*_Evidence_Citations_Or_URLs
    evaluator.add_custom_node(
        result=len(provider_urls) > 0,
        id=f"P{idx+1}_Evidence_Citations_Or_URLs",
        desc=f"Citations and/or URLs are provided to support the provider evidence.",
        parent=node,
        critical=True,
    )


async def verify_cpuc_requirements(evaluator: Evaluator, parent_node, cpuc: CPUCRequirements) -> None:
    node = evaluator.add_parallel(
        id="California_State_Requirements",
        desc="Document CPUC backup power requirements (HFTD Tier 2/3 facilities, central offices, remote terminals, wireless facilities) with official CPUC reference URLs.",
        parent=parent_node,
        critical=True,
    )

    # HFTD Tier2/3 72-hour backup
    hftd_node = evaluator.add_parallel(
        id="HFTD_Tier2_3_72_Hour_Backup",
        desc="Mandated backup power duration for facilities in HFTD Tier 2/3 and the relevant CPUC Decision number, with official CPUC URL(s).",
        parent=node,
        critical=True,
    )
    # Duration 72 hours
    leaf_hftd_72 = evaluator.add_leaf(
        id="HFTD_Duration_72_Hours",
        desc="Backup power duration for HFTD Tier 2/3 facilities is stated as 72 hours (per constraints).",
        parent=hftd_node,
        critical=True,
    )
    await evaluator.verify(
        claim="California CPUC requires 72 hours of backup power for applicable facilities in High Fire Threat District (HFTD) Tier 2 and Tier 3 areas.",
        node=leaf_hftd_72,
        sources=cpuc.hftd_official_urls,
        additional_instruction="Verify on official CPUC sources (e.g., D.21-02-029 summaries or decisions)."
    )
    # Decision number D.21-02-029
    leaf_hftd_dec = evaluator.add_leaf(
        id="HFTD_Source_Decision_Number",
        desc="CPUC Decision 21-02-029 is provided as the source decision (per constraints).",
        parent=hftd_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The controlling CPUC decision for the HFTD 72-hour backup power requirement is Decision D.21-02-029.",
        node=leaf_hftd_dec,
        sources=cpuc.hftd_official_urls,
        additional_instruction="Confirm the decision number D.21-02-029 is correctly cited in the provided CPUC source(s)."
    )
    # Official CPUC URL present
    evaluator.add_custom_node(
        result=_any_official(cpuc.hftd_official_urls, ["cpuc.ca.gov", "docs.cpuc.ca.gov", "apps.cpuc.ca.gov"]),
        id="HFTD_Official_CPUC_URL",
        desc="At least one official CPUC reference URL is provided documenting the HFTD Tier 2/3 requirement.",
        parent=hftd_node,
        critical=True,
    )

    # Central offices backup duration 24 hours
    co_node = evaluator.add_parallel(
        id="Central_Offices_Backup_Duration",
        desc="Backup power duration required for central offices, with official CPUC URL(s).",
        parent=node,
        critical=True,
    )
    leaf_co_24 = evaluator.add_leaf(
        id="Central_Office_Duration_24_Hours",
        desc="Central office backup power duration is stated as 24 hours (per constraints).",
        parent=co_node,
        critical=True,
    )
    await evaluator.verify(
        claim="California CPUC requires at least 24 hours of backup power for central offices.",
        node=leaf_co_24,
        sources=cpuc.central_office_official_urls,
        additional_instruction="Confirm on official CPUC materials specifying central office backup power duration."
    )
    evaluator.add_custom_node(
        result=_any_official(cpuc.central_office_official_urls, ["cpuc.ca.gov", "docs.cpuc.ca.gov", "apps.cpuc.ca.gov"]),
        id="Central_Office_Official_CPUC_URL",
        desc="At least one official CPUC reference URL is provided documenting the central office duration requirement/standard.",
        parent=co_node,
        critical=True,
    )

    # Remote terminals backup duration 4 hours min, 8 hours objective
    rt_node = evaluator.add_parallel(
        id="Remote_Terminals_Backup_Duration",
        desc="Backup power duration required for remote terminals, with official CPUC URL(s).",
        parent=node,
        critical=True,
    )
    leaf_rt_4 = evaluator.add_leaf(
        id="Remote_Terminal_Min_4_Hours",
        desc="Remote terminal minimum backup power is stated as 4 hours (per constraints).",
        parent=rt_node,
        critical=True,
    )
    await evaluator.verify(
        claim="California CPUC requires a minimum of 4 hours of backup power for remote terminals.",
        node=leaf_rt_4,
        sources=cpuc.remote_terminal_official_urls,
        additional_instruction="Confirm the 'minimum 4 hours' requirement for remote terminals on an official CPUC page."
    )
    leaf_rt_8 = evaluator.add_leaf(
        id="Remote_Terminal_Objective_8_Hours",
        desc="Remote terminal design objective is stated as 8 hours (per constraints).",
        parent=rt_node,
        critical=True,
    )
    await evaluator.verify(
        claim="California CPUC specifies a design objective of 8 hours of backup power for remote terminals.",
        node=leaf_rt_8,
        sources=cpuc.remote_terminal_official_urls,
        additional_instruction="Confirm that CPUC materials reference an 8-hour design objective for remote terminals."
    )
    evaluator.add_custom_node(
        result=_any_official(cpuc.remote_terminal_official_urls, ["cpuc.ca.gov", "docs.cpuc.ca.gov", "apps.cpuc.ca.gov"]),
        id="Remote_Terminal_Official_CPUC_URL",
        desc="At least one official CPUC reference URL is provided documenting the remote terminal requirement/standard.",
        parent=rt_node,
        critical=True,
    )

    # Wireless facilities in HFTD Tier 2/3 72 hours
    wl_node = evaluator.add_parallel(
        id="Wireless_Facilities_Backup_Requirement",
        desc="Backup power requirements for wireless facilities in HFTD Tier 2/3, with official CPUC URL(s).",
        parent=node,
        critical=True,
    )
    leaf_wl_72 = evaluator.add_leaf(
        id="Wireless_HFTD_Duration_72_Hours",
        desc="Wireless facilities in HFTD Tier 2/3 are stated as requiring 72 hours backup power (per constraints).",
        parent=wl_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Wireless facilities in California's HFTD Tier 2 and Tier 3 areas must have 72 hours of backup power per CPUC requirements.",
        node=leaf_wl_72,
        sources=cpuc.wireless_official_urls,
        additional_instruction="Confirm on official CPUC materials that wireless sites in HFTD Tier 2/3 require 72 hours."
    )
    evaluator.add_custom_node(
        result=_any_official(cpuc.wireless_official_urls, ["cpuc.ca.gov", "docs.cpuc.ca.gov", "apps.cpuc.ca.gov"]),
        id="Wireless_Official_CPUC_URL",
        desc="At least one official CPUC reference URL is provided documenting the wireless backup power requirement.",
        parent=wl_node,
        critical=True,
    )


async def verify_cfr920(evaluator: Evaluator, parent_node, cfr: CFR920Requirements) -> None:
    node = evaluator.add_parallel(
        id="Federal_Requirements_47_CFR_9_20",
        desc="Document FCC backup power standards under 47 CFR § 9.20 (8-hour option, 24-hour option, covered equipment, sunset date) with official FCC/eCFR/Cornell LII URL(s).",
        parent=parent_node,
        critical=True,
    )

    # 8-hour option
    eight_node = evaluator.add_parallel(
        id="CFR_9_20_8_Hour_Option",
        desc="Minimum standby backup power duration for the 8-hour option, with official URL(s).",
        parent=node,
        critical=True,
    )
    leaf_8 = evaluator.add_leaf(
        id="Eight_Hour_Minimum",
        desc="8-hour minimum standby backup power option requirement is stated (per constraints).",
        parent=eight_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Under 47 CFR § 9.20, providers must offer a minimum 8-hour standby backup power option for ensuring 911 access.",
        node=leaf_8,
        sources=cfr.eight_hour_official_urls,
        additional_instruction="Verify on official FCC/eCFR/LII sources that the rule includes an 8-hour backup power option."
    )
    evaluator.add_custom_node(
        result=_any_official(cfr.eight_hour_official_urls, ["fcc.gov", "ecfr.gov", "law.cornell.edu"]),
        id="Eight_Hour_Official_URL",
        desc="At least one official FCC/eCFR/Cornell LII reference URL is provided for the 8-hour option requirement.",
        parent=eight_node,
        critical=True,
    )

    # 24-hour option
    tf_node = evaluator.add_parallel(
        id="CFR_9_20_24_Hour_Option",
        desc="Minimum standby backup power duration for the 24-hour option, with effective date and official URL(s).",
        parent=node,
        critical=True,
    )
    leaf_24 = evaluator.add_leaf(
        id="Twenty_Four_Hour_Minimum",
        desc="24-hour minimum standby backup power option requirement is stated (per constraints).",
        parent=tf_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Under 47 CFR § 9.20, providers must offer a 24-hour minimum standby backup power option for ensuring 911 access.",
        node=leaf_24,
        sources=cfr.twentyfour_hour_official_urls,
        additional_instruction="Verify on official FCC/eCFR/LII sources that the rule includes a 24-hour option."
    )
    leaf_24_eff = evaluator.add_leaf(
        id="Twenty_Four_Hour_Effective_Date",
        desc="Effective date is stated as Feb 13, 2019 (per constraints).",
        parent=tf_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The 24-hour option under 47 CFR § 9.20 has an effective date of February 13, 2019.",
        node=leaf_24_eff,
        sources=cfr.twentyfour_hour_official_urls,
        additional_instruction="Confirm the effective date on the cited official source."
    )
    evaluator.add_custom_node(
        result=_any_official(cfr.twentyfour_hour_official_urls, ["fcc.gov", "ecfr.gov", "law.cornell.edu"]),
        id="Twenty_Four_Hour_Official_URL",
        desc="At least one official FCC/eCFR/Cornell LII reference URL is provided for the 24-hour option requirement.",
        parent=tf_node,
        critical=True,
    )

    # Equipment coverage for 911
    eq_node = evaluator.add_parallel(
        id="CFR_9_20_Equipment_Coverage_For_911",
        desc="Scope of equipment that must be covered by backup power for 911 access, with official URL(s).",
        parent=node,
        critical=True,
    )
    leaf_eq = evaluator.add_leaf(
        id="Provider_Furnished_Equipment_Necessary_For_911",
        desc="Scope is stated as all provider-furnished equipment necessary for 911 access (per constraints).",
        parent=eq_node,
        critical=True,
    )
    await evaluator.verify(
        claim="47 CFR § 9.20 requires backup power coverage for provider-furnished equipment necessary to ensure access to 911.",
        node=leaf_eq,
        sources=cfr.equipment_coverage_official_urls,
        additional_instruction="Verify scope on official FCC/eCFR/LII sources."
    )
    evaluator.add_custom_node(
        result=_any_official(cfr.equipment_coverage_official_urls, ["fcc.gov", "ecfr.gov", "law.cornell.edu"]),
        id="Equipment_Coverage_Official_URL",
        desc="At least one official FCC/eCFR/Cornell LII reference URL is provided for the equipment coverage requirement.",
        parent=eq_node,
        critical=True,
    )

    # Sunset date
    sun_node = evaluator.add_parallel(
        id="CFR_9_20_Sunset_Date",
        desc="Rule sunset date, with official URL(s).",
        parent=node,
        critical=True,
    )
    leaf_sun = evaluator.add_leaf(
        id="Sunset_September_1_2025",
        desc="Sunset date is stated as September 1, 2025 (per constraints).",
        parent=sun_node,
        critical=True,
    )
    await evaluator.verify(
        claim="47 CFR § 9.20 sunsets on September 1, 2025.",
        node=leaf_sun,
        sources=cfr.sunset_official_urls,
        additional_instruction="Confirm on the official eCFR/LII/FCC page noting the rule's sunset date."
    )
    evaluator.add_custom_node(
        result=_any_official(cfr.sunset_official_urls, ["fcc.gov", "ecfr.gov", "law.cornell.edu"]),
        id="Sunset_Official_URL",
        desc="At least one official FCC/eCFR/Cornell LII reference URL is provided for the sunset date.",
        parent=sun_node,
        critical=True,
    )


async def verify_911_cert(evaluator: Evaluator, parent_node, cert: FCC911CertificationRequirements) -> None:
    node = evaluator.add_parallel(
        id="FCC_911_Reliability_Annual_Certification",
        desc="Document FCC 911 reliability annual certification requirements (three measures + filing portal URL) with official FCC URL(s).",
        parent=parent_node,
        critical=True,
    )

    # Measures
    meas_node = evaluator.add_parallel(
        id="Three_Certification_Measures",
        desc="The three specific measures that must be certified are listed, with official FCC URL(s).",
        parent=node,
        critical=True,
    )

    leaf_circuit = evaluator.add_leaf(
        id="Measure_Circuit_Diversity",
        desc="Circuit diversity is listed as a measure to be certified (per constraints).",
        parent=meas_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Circuit diversity is one of the three measures covered 911 service providers must certify annually.",
        node=leaf_circuit,
        sources=cert.measures_official_urls,
        additional_instruction="Confirm on an official FCC page describing the 911 reliability certification measures."
    )

    leaf_co_backup = evaluator.add_leaf(
        id="Measure_Central_Office_Backup_Power",
        desc="Central office backup power is listed as a measure to be certified (per constraints).",
        parent=meas_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Central office backup power is one of the three measures covered 911 service providers must certify annually.",
        node=leaf_co_backup,
        sources=cert.measures_official_urls,
        additional_instruction="Confirm on an official FCC page describing the 911 reliability certification measures."
    )

    leaf_monitor = evaluator.add_leaf(
        id="Measure_Network_Monitoring",
        desc="Network monitoring is listed as a measure to be certified (per constraints).",
        parent=meas_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Network monitoring is one of the three measures covered 911 service providers must certify annually.",
        node=leaf_monitor,
        sources=cert.measures_official_urls,
        additional_instruction="Confirm on an official FCC page describing the 911 reliability certification measures."
    )

    evaluator.add_custom_node(
        result=_any_official(cert.measures_official_urls, ["fcc.gov"]),
        id="Measures_Official_FCC_URL",
        desc="At least one official FCC reference URL is provided documenting these measures.",
        parent=meas_node,
        critical=True,
    )

    # Portal URL
    portal_node = evaluator.add_parallel(
        id="Certification_Portal_URL",
        desc="The URL of the certification portal where providers must file is provided, with official FCC URL(s).",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_str(cert.portal_url) and ("apps2.fcc.gov/rcs911" in (cert.portal_url or "")),
        id="Portal_URL_Provided",
        desc="The filing portal URL is provided as https://apps2.fcc.gov/rcs911/ (per constraints).",
        parent=portal_node,
        critical=True,
    )
    leaf_portal_official = evaluator.add_leaf(
        id="Portal_Official_FCC_URL",
        desc="At least one official FCC reference URL is provided documenting the portal.",
        parent=portal_node,
        critical=True,
    )
    # Verify using either a separate official page referencing the portal, or the portal URL itself if cited
    portal_sources = cert.portal_official_urls if cert.portal_official_urls else ([cert.portal_url] if cert.portal_url else [])
    await evaluator.verify(
        claim="This is the official FCC 911 reliability certification filing portal, or an official FCC page referencing it.",
        node=leaf_portal_official,
        sources=portal_sources,
        additional_instruction="Confirm on FCC domain pages that the URL is the official filing portal for 911 reliability certifications."
    )


async def verify_outage_requirements(evaluator: Evaluator, parent_node, outage: OutageRequirements) -> None:
    node = evaluator.add_parallel(
        id="Outage_Notification_And_Reporting",
        desc="Document PSAP notification and FCC outage reporting timelines with official FCC/eCFR/Cornell LII URL(s).",
        parent=parent_node,
        critical=True,
    )

    # PSAP notification within 30 minutes
    psap_node = evaluator.add_parallel(
        id="PSAP_Notification",
        desc="Maximum time allowed for notifying PSAPs after discovering an outage affecting 911 service, with official URL(s).",
        parent=node,
        critical=True,
    )
    leaf_psap_30 = evaluator.add_leaf(
        id="PSAP_Notify_Within_30_Minutes",
        desc="PSAP notification timeline is stated as within 30 minutes (per constraints).",
        parent=psap_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Covered providers must notify affected PSAPs as soon as possible, but no later than 30 minutes after discovering an outage affecting 911 service.",
        node=leaf_psap_30,
        sources=outage.psap_official_urls,
        additional_instruction="Confirm on Part 4 or FCC PSAP notification guidance pages (official sources only)."
    )
    evaluator.add_custom_node(
        result=_any_official(outage.psap_official_urls, ["fcc.gov", "ecfr.gov", "law.cornell.edu"]),
        id="PSAP_Official_URL",
        desc="At least one official FCC/eCFR/Cornell LII reference URL is provided documenting the PSAP notification requirement.",
        parent=psap_node,
        critical=True,
    )

    # Initial outage report to FCC within 72 hours; cite 47 CFR Part 4
    init_node = evaluator.add_parallel(
        id="FCC_Initial_Outage_Report",
        desc="Time window for filing an initial communications outage report to the FCC after discovering a reportable outage, with official URL(s).",
        parent=node,
        critical=True,
    )
    leaf_init_72 = evaluator.add_leaf(
        id="Initial_Report_Within_72_Hours",
        desc="Initial outage report timeline is stated as within 72 hours (per constraints).",
        parent=init_node,
        critical=True,
    )
    await evaluator.verify(
        claim="An initial communications outage report must be filed with the FCC within 72 hours after discovering a reportable outage.",
        node=leaf_init_72,
        sources=outage.outage_reporting_official_urls,
        additional_instruction="Confirm this timeline on official FCC/eCFR/LII Part 4 outage reporting rules."
    )
    leaf_auth = evaluator.add_leaf(
        id="Governing_Authority_47_CFR_Part_4",
        desc="47 CFR Part 4 is cited as the governing authority (per constraints).",
        parent=init_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The governing authority for federal communications outage reporting is 47 CFR Part 4.",
        node=leaf_auth,
        sources=outage.outage_reporting_official_urls,
        additional_instruction="Ensure the cited pages explicitly refer to 47 CFR Part 4 as the controlling rules."
    )
    evaluator.add_custom_node(
        result=_any_official(outage.outage_reporting_official_urls, ["fcc.gov", "ecfr.gov", "law.cornell.edu"]),
        id="Outage_Reporting_Official_URL",
        desc="At least one official FCC/eCFR/Cornell LII reference URL is provided for the 47 CFR Part 4 outage reporting requirements.",
        parent=init_node,
        critical=True,
    )


async def verify_providers_section(evaluator: Evaluator, parent_node, providers: ProvidersExtraction) -> None:
    node = evaluator.add_parallel(
        id="Provider_Identification",
        desc="Provide at least two distinct qualifying major facilities-based wireline telecommunications carriers operating in California that serve HFTD Tier 2/3 and are subject to CPUC Decision 21-02-029; include CPUC certificate numbers and evidence of HFTD presence and applicability.",
        parent=parent_node,
        critical=True,
    )

    # At least two providers named
    count_named = sum(1 for p in providers.providers if _non_empty_str(p.name))
    evaluator.add_custom_node(
        result=count_named >= 2,
        id="At_Least_Two_Providers_Provided",
        desc="Documentation includes at least two distinct provider entries.",
        parent=node,
        critical=True,
    )

    # Verify up to first 2 providers
    p_list = providers.providers[:2] if providers.providers else []
    while len(p_list) < 2:
        p_list.append(ProviderEntry())  # pad to ensure nodes exist

    await verify_provider(evaluator, node, p_list[0], 0)
    await verify_provider(evaluator, node, p_list[1], 1)


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
    Evaluate an answer for the CA HFTD emergency preparedness compliance documentation task.
    """

    # Initialize evaluator (root is a neutral container)
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

    # Create the top-level assessment node as critical (to mirror rubric)
    assessment_root = evaluator.add_parallel(
        id="Emergency_Preparedness_Compliance_Assessment",
        desc="Complete emergency preparedness compliance documentation for CA HFTD Tier 2/3 covering: (1) ≥2 qualifying major facilities-based wireline providers with CPUC certificate numbers and evidence of HFTD presence and D.21-02-029 applicability; (2) CPUC backup power requirements; (3) FCC 47 CFR § 9.20 backup power requirements; (4) FCC 911 reliability annual certification requirements; (5) outage notification/reporting requirements. For regulatory requirements, include official FCC/CPUC/eCFR/Cornell LII reference URLs.",
        parent=root,
        critical=True,
    )

    # Extract structured information (can be parallelized)
    providers_extraction_task = evaluator.extract(
        prompt=prompt_extract_providers(),
        template_class=ProvidersExtraction,
        extraction_name="providers_extraction",
    )
    cpuc_req_task = evaluator.extract(
        prompt=prompt_extract_cpuc_requirements(),
        template_class=CPUCRequirements,
        extraction_name="cpuc_requirements",
    )
    cfr_req_task = evaluator.extract(
        prompt=prompt_extract_cfr920(),
        template_class=CFR920Requirements,
        extraction_name="cfr_9_20_requirements",
    )
    cert_req_task = evaluator.extract(
        prompt=prompt_extract_911_cert(),
        template_class=FCC911CertificationRequirements,
        extraction_name="fcc_911_cert_requirements",
    )
    outage_req_task = evaluator.extract(
        prompt=prompt_extract_outage(),
        template_class=OutageRequirements,
        extraction_name="outage_requirements",
    )

    providers_extraction, cpuc_req, cfr_req, cert_req, outage_req = await asyncio.gather(
        providers_extraction_task, cpuc_req_task, cfr_req_task, cert_req_task, outage_req_task
    )

    # Build and verify tree sections
    await verify_providers_section(evaluator, assessment_root, providers_extraction)
    await verify_cpuc_requirements(evaluator, assessment_root, cpuc_req)
    await verify_cfr920(evaluator, assessment_root, cfr_req)
    await verify_911_cert(evaluator, assessment_root, cert_req)
    await verify_outage_requirements(evaluator, assessment_root, outage_req)

    # Return structured evaluation summary
    return evaluator.get_summary()