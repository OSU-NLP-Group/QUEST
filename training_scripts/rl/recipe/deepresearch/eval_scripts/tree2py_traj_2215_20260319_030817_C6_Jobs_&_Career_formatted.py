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
TASK_ID = "np_career_planning_2026"
TASK_DESCRIPTION = """
You are conducting comprehensive career planning research for someone considering becoming a nurse practitioner in the United States. Your research must address the following requirements:

1. Occupation Analysis (BLS 2024–2034):
   - NP projected employment growth rate
   - NP ranking among fastest-growing occupations
   - NP median annual wage (as of 2024)
   - Healthcare & social assistance sector growth rate and ranking
   - Projected annual number of job openings in healthcare occupations

2. State Selection (as of 2026):
   - Identify ALL states that grant full practice authority (FPA) to NPs AND have enacted the APRN Compact
   - Verify total number of states/territories with FPA for NPs (as of 2026)

3. APRN Compact Status (as of 2026):
   - How many states have enacted the APRN Compact
   - Minimum number of states required to implement the Compact
   - Whether the Compact is implemented based on those numbers

4. Education Pathway:
   - Minimum degree required for NP licensure
   - Requirement for an active RN license
   - National certification requirement and certifying organizations (AANP/ANCC)
   - Typical GPA expectations for graduate nursing admissions

5. Interstate Licensure Options:
   - IMLC participation as of March 2026
   - Compare scope and participation between APRN Compact and IMLC

Provide authoritative URLs (e.g., BLS, AANP, NCSBN, IMLC, state boards) supporting each fact.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class BLSFacts(BaseModel):
    # Nurse Practitioner specific
    np_projected_growth_rate_2024_2034: Optional[str] = None
    np_growth_support_urls: List[str] = Field(default_factory=list)

    np_fastest_growing_rank_2024_2034: Optional[str] = None
    np_rank_support_urls: List[str] = Field(default_factory=list)

    np_median_annual_wage_2024: Optional[str] = None
    np_wage_support_urls: List[str] = Field(default_factory=list)

    # Sector (Healthcare & Social Assistance) projections
    hsa_projected_growth_rate_2024_2034: Optional[str] = None
    hsa_growth_support_urls: List[str] = Field(default_factory=list)

    hsa_fastest_growing_sector_statement: Optional[str] = None
    hsa_fastest_support_urls: List[str] = Field(default_factory=list)

    # Healthcare occupations openings
    healthcare_annual_openings_2024_2034: Optional[str] = None
    healthcare_openings_support_urls: List[str] = Field(default_factory=list)


class BothStateItem(BaseModel):
    state: Optional[str] = None
    fpa_urls: List[str] = Field(default_factory=list)
    compact_urls: List[str] = Field(default_factory=list)
    extra_urls: List[str] = Field(default_factory=list)


class StateSelectionFacts(BaseModel):
    fpa_total_count_states_and_territories_2026: Optional[str] = None
    fpa_total_support_urls: List[str] = Field(default_factory=list)

    states_with_both_fpa_and_enacted_compact_2026: List[BothStateItem] = Field(default_factory=list)


class APRNCompactFacts(BaseModel):
    aprn_compact_enacted_count_2026: Optional[str] = None
    aprn_enacted_support_urls: List[str] = Field(default_factory=list)

    aprn_compact_min_states_to_implement: Optional[str] = None
    aprn_min_support_urls: List[str] = Field(default_factory=list)

    aprn_compact_implemented_status_statement: Optional[str] = None  # e.g., "not implemented"
    aprn_status_support_urls: List[str] = Field(default_factory=list)


class EducationLicensureFacts(BaseModel):
    min_degree_requirement_for_np_licensure: Optional[str] = None
    min_degree_support_urls: List[str] = Field(default_factory=list)

    active_rn_license_required_statement: Optional[str] = None
    active_rn_license_support_urls: List[str] = Field(default_factory=list)

    national_cert_required_statement: Optional[str] = None
    certification_orgs_mentioned: List[str] = Field(default_factory=list)
    national_cert_support_urls: List[str] = Field(default_factory=list)

    typical_grad_gpa_expectation_statement: Optional[str] = None
    typical_grad_gpa_support_urls: List[str] = Field(default_factory=list)


class InterstateLicensureFacts(BaseModel):
    imlc_participation_count_march_2026: Optional[str] = None
    imlc_count_support_urls: List[str] = Field(default_factory=list)

    scope_comparison_statement_aprn_vs_imlc: Optional[str] = None
    scope_comparison_support_urls: List[str] = Field(default_factory=list)

    participation_comparison_statement_aprn_vs_imlc: Optional[str] = None
    participation_comparison_support_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_bls() -> str:
    return """
    Extract all BLS-related facts and their cited URLs exactly as stated in the answer.

    Required fields:
    - np_projected_growth_rate_2024_2034: The stated NP projected employment growth rate for 2024–2034 (e.g., "40%")
    - np_growth_support_urls: All URLs cited for that NP growth rate (prefer official BLS bls.gov pages)

    - np_fastest_growing_rank_2024_2034: The stated NP rank among fastest-growing occupations (e.g., "3rd")
    - np_rank_support_urls: All URLs cited for that rank (prefer bls.gov)

    - np_median_annual_wage_2024: The stated NP median annual wage for 2024 (e.g., "$129,210")
    - np_wage_support_urls: All URLs cited for that wage (prefer bls.gov OEWS/OOH pages)

    - hsa_projected_growth_rate_2024_2034: The stated projected growth rate for healthcare & social assistance sector (e.g., "8.4%")
    - hsa_growth_support_urls: All URLs cited for that rate (prefer bls.gov)

    - hsa_fastest_growing_sector_statement: The statement that healthcare & social assistance is the fastest-growing sector (verbatim or paraphrased from the answer)
    - hsa_fastest_support_urls: All URLs cited for that claim (prefer bls.gov)

    - healthcare_annual_openings_2024_2034: The stated annual job openings for healthcare occupations (e.g., "about 1.9 million")
    - healthcare_openings_support_urls: All URLs cited for that figure (prefer bls.gov)

    Return null for any missing value and [] for missing URL arrays.
    """


def prompt_extract_state_selection() -> str:
    return """
    Extract state-selection facts for 2026 exactly as stated in the answer.

    Required fields:
    - fpa_total_count_states_and_territories_2026: The stated total number of states and territories with full practice authority (FPA) for nurse practitioners as of 2026 (e.g., "30")
    - fpa_total_support_urls: All URLs cited for that FPA total (prefer aanp.org or state/national authorities)

    - states_with_both_fpa_and_enacted_compact_2026: The full set of states that the answer lists as BOTH full practice authority and having enacted the APRN Compact (as of 2026).
      For each state, extract:
        - state: State name
        - fpa_urls: URLs the answer cites supporting that the state is FPA
        - compact_urls: URLs the answer cites supporting that the state enacted the APRN Compact
        - extra_urls: Any other URLs the answer grouped for this state's verification

    Return null for any missing scalar and [] for missing arrays.
    """


def prompt_extract_aprn_compact() -> str:
    return """
    Extract APRN Compact status as stated in the answer (as of 2026).

    Required fields:
    - aprn_compact_enacted_count_2026: The stated number of states that have enacted the APRN Compact (e.g., "4")
    - aprn_enacted_support_urls: All URLs cited for that count (prefer ncsbn.org)

    - aprn_compact_min_states_to_implement: The stated minimum number of states required for the APRN Compact to be implemented (e.g., "7")
    - aprn_min_support_urls: All URLs cited for that requirement (prefer ncsbn.org)

    - aprn_compact_implemented_status_statement: The answer's explicit conclusion on whether the APRN Compact is implemented (e.g., "not implemented")
    - aprn_status_support_urls: URLs cited for the implementation status, if any (may be empty if the status is a logical deduction)

    Return null for any missing scalar and [] for missing arrays.
    """


def prompt_extract_education_licensure() -> str:
    return """
    Extract education and licensure pathway facts for NP licensure exactly as stated in the answer.

    Required fields:
    - min_degree_requirement_for_np_licensure: The minimum degree requirement (e.g., "MSN or higher (DNP)")
    - min_degree_support_urls: All URLs cited for that (prefer aanp.org, ncsbn.org, or official state boards)

    - active_rn_license_required_statement: Whether an active RN license is required (state the answer's phrasing)
    - active_rn_license_support_urls: URLs cited for that (prefer authorities)

    - national_cert_required_statement: The statement that national certification is required (answer's phrasing)
    - certification_orgs_mentioned: Organizations named (e.g., "AANP", "ANCC")
    - national_cert_support_urls: URLs cited for that (prefer aanp.org, ancc.org, ncsbn.org, or state boards)

    - typical_grad_gpa_expectation_statement: Typical graduate nursing GPA expectation (e.g., "around 3.0 minimum; varies by program")
    - typical_grad_gpa_support_urls: At least one authoritative URL (e.g., university graduate nursing admissions page, AANP, AACN)

    Return null for any missing scalar and [] for missing arrays.
    """


def prompt_extract_interstate_licensure() -> str:
    return """
    Extract Interstate Medical Licensure Compact (IMLC) and comparison facts as stated in the answer.

    Required fields:
    - imlc_participation_count_march_2026: The stated IMLC participation count as of March 2026 (e.g., "42 states plus DC and Guam")
    - imlc_count_support_urls: URLs cited for that (prefer imlcc.org)

    - scope_comparison_statement_aprn_vs_imlc: The answer's scope comparison between APRN Compact vs IMLC (e.g., "APRN Compact covers APRNs; IMLC covers physicians")
    - scope_comparison_support_urls: URLs cited for scope comparison (use ncsbn.org for APRN Compact and imlcc.org for IMLC if available)

    - participation_comparison_statement_aprn_vs_imlc: The answer's participation comparison using counts (e.g., "IMLC far wider (42+DC+Guam) vs APRN 4 enacted")
    - participation_comparison_support_urls: URLs cited (ncsbn.org + imlcc.org if available)

    Return null for any missing scalar and [] for missing arrays.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for urls in url_lists:
        for u in urls or []:
            if not u:
                continue
            if u not in seen:
                out.append(u)
                seen.add(u)
    return out


def _has_domain(urls: List[str], domain_sub: str) -> bool:
    ds = domain_sub.lower()
    return any((isinstance(u, str) and ds in u.lower()) for u in urls or [])


def _has_any_domain(urls: List[str], domains: List[str]) -> bool:
    return any(_has_domain(urls, d) for d in (domains or []))


def _bool_str(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    sv = s.strip().lower()
    if sv in {"true", "yes", "y", "implemented"}:
        return True
    if sv in {"false", "no", "n", "not implemented", "not-implemented"}:
        return False
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_bls_section(evaluator: Evaluator, parent_node, bls: BLSFacts) -> None:
    node = evaluator.add_parallel(
        id="Occupation_Analysis_BLS_2024_2034",
        desc="Verify the required BLS projection and wage facts for NPs and related healthcare projections for 2024–2034 (and 2024 wage), with authoritative URLs.",
        parent=parent_node,
        critical=True
    )

    # Template for grouped fact verification with a source presence gate
    async def _group_fact(id_prefix: str, title: str, claim: str, src_urls: List[str], required_domain: Optional[str], add_ins: str):
        group = evaluator.add_sequential(
            id=f"{id_prefix}_group",
            desc=f"{title} - gated by source presence",
            parent=node,
            critical=True
        )
        # Gate: sources provided (+ optional domain requirement)
        has_sources = bool(src_urls)
        domain_ok = True if (required_domain is None) else _has_domain(src_urls, required_domain)
        evaluator.add_custom_node(
            result=has_sources and domain_ok,
            id=f"{id_prefix}_sources_present",
            desc=f"{title}: supporting sources provided" + (f" and include {required_domain}" if required_domain else ""),
            parent=group,
            critical=True
        )
        # Main verification leaf
        leaf = evaluator.add_leaf(
            id=id_prefix,
            desc=title,
            parent=group,
            critical=True
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=src_urls,
            additional_instruction=add_ins
        )

    # 1) NP projected growth rate 40%
    await _group_fact(
        id_prefix="NP_Projected_Growth_Rate",
        title="NP projected growth rate is 40% for 2024–2034 (BLS)",
        claim="According to U.S. Bureau of Labor Statistics (BLS) 2024–2034 employment projections, nurse practitioners have a projected employment growth rate of 40%.",
        src_urls=bls.np_growth_support_urls,
        required_domain="bls.gov",
        add_ins="Use only BLS authoritative content on bls.gov. Accept minor textual variants of '2024–2034'."
    )

    # 2) NP fastest-growing rank 3rd
    await _group_fact(
        id_prefix="NP_Fastest_Growing_Rank",
        title="NP rank is 3rd among fastest-growing occupations (BLS)",
        claim="According to BLS projections for 2024–2034, nurse practitioners rank 3rd among the fastest-growing occupations.",
        src_urls=bls.np_rank_support_urls,
        required_domain="bls.gov",
        add_ins="Confirm the ranking specifically refers to 'fastest-growing occupations' list for 2024–2034 on bls.gov."
    )

    # 3) NP median annual wage $129,210 (2024)
    await _group_fact(
        id_prefix="NP_Median_Annual_Wage_2024",
        title="NP median annual wage is $129,210 as of 2024 (BLS)",
        claim="According to BLS May 2024 data, the median annual wage for nurse practitioners is $129,210.",
        src_urls=bls.np_wage_support_urls,
        required_domain="bls.gov",
        add_ins="Prefer BLS OEWS/OOH pages for May 2024. Allow reasonable formatting variants (e.g., $129,210 vs 129,210 dollars)."
    )

    # 4) Healthcare & Social Assistance growth rate 8.4%
    await _group_fact(
        id_prefix="Healthcare_and_Social_Assistance_Growth_Rate",
        title="Healthcare & social assistance projected growth rate is 8.4% for 2024–2034 (BLS)",
        claim="BLS projects the healthcare and social assistance industry sector to grow by 8.4% over 2024–2034.",
        src_urls=bls.hsa_growth_support_urls,
        required_domain="bls.gov",
        add_ins="Verify the figure for the industry sector, not occupational group."
    )

    # 5) Healthcare & Social Assistance is fastest-growing sector
    await _group_fact(
        id_prefix="Healthcare_and_Social_Assistance_Sector_Ranking",
        title="Healthcare & social assistance is the fastest-growing industry sector for 2024–2034 (BLS)",
        claim="For 2024–2034, BLS identifies healthcare and social assistance as the fastest-growing industry sector.",
        src_urls=bls.hsa_fastest_support_urls,
        required_domain="bls.gov",
        add_ins="Confirm the superlative 'fastest-growing' among industry sectors."
    )

    # 6) Healthcare occupations ~1.9M annual openings
    await _group_fact(
        id_prefix="Healthcare_Occupations_Annual_Openings",
        title="Healthcare occupations have approximately 1.9 million job openings annually (2024–2034) (BLS)",
        claim="BLS projects that healthcare occupations will have approximately 1.9 million job openings annually over 2024–2034.",
        src_urls=bls.healthcare_openings_support_urls,
        required_domain="bls.gov",
        add_ins="Allow reasonable rounding (e.g., 1.9 million ≈ 1,900,000). Ensure the figure applies to healthcare occupations, not a different group."
    )


async def build_state_selection_section(evaluator: Evaluator, parent_node, ss: StateSelectionFacts) -> None:
    node = evaluator.add_parallel(
        id="State_Selection_FPA_and_APRN_Compact_2026",
        desc="Verify the FPA total and identify all states meeting BOTH criteria (FPA + enacted APRN Compact) as of 2026, with authoritative URLs.",
        parent=parent_node,
        critical=True
    )

    # 1) FPA total count = 30 (states/territories)
    fpa_group = evaluator.add_sequential(
        id="FPA_Total_Count_group",
        desc="FPA total count gate + verification",
        parent=node,
        critical=True
    )
    fpa_urls = ss.fpa_total_support_urls
    evaluator.add_custom_node(
        result=bool(fpa_urls) and _has_any_domain(fpa_urls, ["aanp.org"]),
        id="FPA_Total_Count_sources_present",
        desc="FPA total count: supporting sources provided (prefer aanp.org)",
        parent=fpa_group,
        critical=True
    )
    fpa_leaf = evaluator.add_leaf(
        id="FPA_Total_Count_(States_and_Territories)",
        desc="States there are exactly 30 states/territories with NP full practice authority as of 2026 and provides an authoritative URL supporting the count.",
        parent=fpa_group,
        critical=True
    )
    await evaluator.verify(
        claim="As of 2026, there are exactly 30 states and territories with full practice authority for nurse practitioners.",
        node=fpa_leaf,
        sources=fpa_urls,
        additional_instruction="Use authoritative sources (prefer AANP). Accept phrasing like '30 states' if context implies states/territories total."
    )

    # 2) States with BOTH (FPA + enacted APRN Compact) as of 2026 = {DE, ND, SD, UT}
    both_group = evaluator.add_sequential(
        id="States_With_Both_group",
        desc="States with BOTH (FPA + enacted APRN Compact) - gate + verification",
        parent=node,
        critical=True
    )

    # Collect and verify that for each listed state we have both FPA and Compact sources
    combined_urls: List[str] = []
    states_items = ss.states_with_both_fpa_and_enacted_compact_2026 or []

    # Build combined URL pool and check per-state supports
    def per_state_ok(item: BothStateItem) -> bool:
        has_fpa = bool(item.fpa_urls)
        has_compact = bool(item.compact_urls)
        fpa_ok = _has_any_domain(item.fpa_urls, ["aanp.org"]) or any(".gov" in (u or "").lower() for u in item.fpa_urls)
        compact_ok = _has_any_domain(item.compact_urls, ["ncsbn.org"]) or any(".gov" in (u or "").lower() for u in item.compact_urls)
        return has_fpa and has_compact and fpa_ok and compact_ok

    per_state_checks = all(per_state_ok(s) for s in states_items) if states_items else False
    for s in states_items:
        combined_urls = _dedup_urls(combined_urls, s.fpa_urls, s.compact_urls, s.extra_urls)

    evaluator.add_custom_node(
        result=per_state_checks and bool(combined_urls),
        id="States_With_Both_sources_present",
        desc="States with BOTH: per-state FPA and Compact sources provided (aanp.org and ncsbn.org preferred)",
        parent=both_group,
        critical=True
    )

    both_leaf = evaluator.add_leaf(
        id="States_With_Both_(FPA_AND_Enacted_APRN_Compact)_AsOf2026",
        desc="Provides the complete set of states meeting BOTH criteria (FPA + enacted APRN Compact) as of 2026 (DE, ND, SD, UT) with authoritative support.",
        parent=both_group,
        critical=True
    )
    await evaluator.verify(
        claim="As of 2026, the states that both grant full practice authority to nurse practitioners and have enacted the APRN Compact are exactly Delaware, North Dakota, South Dakota, and Utah.",
        node=both_leaf,
        sources=combined_urls,
        additional_instruction="Confirm BOTH conditions (FPA + enacted APRN Compact) for exactly the four states listed. Prefer aanp.org for FPA and ncsbn.org for Compact enactment."
    )


async def build_aprn_compact_section(evaluator: Evaluator, parent_node, aprn: APRNCompactFacts) -> None:
    node = evaluator.add_parallel(
        id="APRN_Compact_Status_2026",
        desc="Verify enacted-state count, implementation threshold, and implemented/not-implemented conclusion for the APRN Compact as of 2026, with authoritative URLs.",
        parent=parent_node,
        critical=True
    )

    # 1) Enacted count = 4
    enacted_group = evaluator.add_sequential(
        id="APRN_Compact_Enacted_Count_group",
        desc="APRN Compact enacted count gate + verification",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(aprn.aprn_enacted_support_urls) and _has_any_domain(aprn.aprn_enacted_support_urls, ["ncsbn.org"]),
        id="APRN_Compact_Enacted_Count_sources_present",
        desc="APRN Compact enacted count: supporting sources provided (prefer ncsbn.org)",
        parent=enacted_group,
        critical=True
    )
    enacted_leaf = evaluator.add_leaf(
        id="APRN_Compact_Enacted_Count",
        desc="States that 4 states have enacted the APRN Compact as of 2026 and cites an authoritative URL supporting the count.",
        parent=enacted_group,
        critical=True
    )
    await evaluator.verify(
        claim="As of 2026, 4 states have enacted the APRN Compact.",
        node=enacted_leaf,
        sources=aprn.aprn_enacted_support_urls,
        additional_instruction="Use NCSBN (ncsbn.org) APRN Compact information. Verify that the page explicitly lists 4 enacted states as of 2026."
    )

    # 2) Minimum states to implement = 7
    min_group = evaluator.add_sequential(
        id="APRN_Compact_Minimum_To_Implement_group",
        desc="APRN Compact minimum states gate + verification",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(aprn.aprn_min_support_urls) and _has_any_domain(aprn.aprn_min_support_urls, ["ncsbn.org"]),
        id="APRN_Compact_Minimum_To_Implement_sources_present",
        desc="APRN Compact minimum to implement: supporting sources provided (prefer ncsbn.org)",
        parent=min_group,
        critical=True
    )
    min_leaf = evaluator.add_leaf(
        id="APRN_Compact_Minimum_To_Implement",
        desc="States that at least 7 states are required for APRN Compact implementation and cites an authoritative URL supporting it.",
        parent=min_group,
        critical=True
    )
    await evaluator.verify(
        claim="The APRN Compact requires at least 7 states to enact before it can be implemented.",
        node=min_leaf,
        sources=aprn.aprn_min_support_urls,
        additional_instruction="Use NCSBN APRN Compact description pages. Confirm the threshold is 7 states."
    )

    # 3) Implemented status = not implemented (because 4 < 7)
    status_group = evaluator.add_sequential(
        id="APRN_Compact_Implemented_Status_group",
        desc="APRN Compact implemented status (logical check) gate + verification",
        parent=node,
        critical=True
    )
    # We logically depend on the two previous leaves
    status_leaf = evaluator.add_leaf(
        id="APRN_Compact_Implemented_Status",
        desc="Correctly concludes the APRN Compact is not implemented as of 2026 because enacted (4) is less than required (7).",
        parent=status_group,
        critical=True
    )
    await evaluator.verify(
        claim="Because only 4 states have enacted the APRN Compact and at least 7 are required, the APRN Compact is not implemented as of 2026.",
        node=status_leaf,
        sources=None,  # pure logical conclusion
        additional_instruction="This is a simple logical verification: 4 < 7 implies not implemented. No external URL is needed."
    )


async def build_education_licensure_section(evaluator: Evaluator, parent_node, edu: EducationLicensureFacts) -> None:
    node = evaluator.add_parallel(
        id="Education_and_Licensure_Pathway",
        desc="Verify minimum education and licensure prerequisites (degree, RN license, national certification) and provide typical graduate GPA expectations, with authoritative URLs.",
        parent=parent_node,
        critical=True
    )

    # Helper to build gated verification
    async def _edu_group(id_prefix: str, title: str, claim: str, urls: List[str], allowed_domains: List[str], add_ins: str):
        group = evaluator.add_sequential(
            id=f"{id_prefix}_group",
            desc=f"{title} - gate + verification",
            parent=node,
            critical=True
        )
        evaluator.add_custom_node(
            result=bool(urls) and _has_any_domain(urls, allowed_domains + [".gov"]),
            id=f"{id_prefix}_sources_present",
            desc=f"{title}: supporting sources provided (prefer authorities: {', '.join(allowed_domains)} or .gov)",
            parent=group,
            critical=True
        )
        leaf = evaluator.add_leaf(
            id=id_prefix,
            desc=title,
            parent=group,
            critical=True
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=add_ins
        )

    # 1) Minimum degree = MSN or higher (e.g., DNP)
    await _edu_group(
        id_prefix="Minimum_Degree_For_NP_Licensure",
        title="Minimum degree requirement is MSN or higher (e.g., DNP)",
        claim="The minimum educational degree required for nurse practitioner licensure is a graduate nursing degree—at least a Master of Science in Nursing (MSN) or higher (such as a Doctor of Nursing Practice, DNP).",
        urls=edu.min_degree_support_urls,
        allowed_domains=["aanp.org", "ncsbn.org"],
        add_ins="Prefer AANP, NCSBN, or official state board sources. Confirm that NP licensure requires at least an MSN (or DNP)."
    )

    # 2) Active RN license prerequisite
    await _edu_group(
        id_prefix="Active_RN_License_Prerequisite",
        title="Active RN license is required before NP licensure",
        claim="An active Registered Nurse (RN) license is required prior to applying for nurse practitioner licensure.",
        urls=edu.active_rn_license_support_urls,
        allowed_domains=["aanp.org", "ncsbn.org"],
        add_ins="Prefer authorities; confirm the RN license prerequisite explicitly."
    )

    # 3) National certification required; orgs include AANP/ANCC
    await _edu_group(
        id_prefix="National_Certification_Requirement",
        title="National certification required for NP licensure (AANP and/or ANCC)",
        claim="National certification is required for NP licensure, typically through recognized certifying bodies such as AANP or ANCC.",
        urls=edu.national_cert_support_urls,
        allowed_domains=["aanp.org", "nursingworld.org", "ancc.org", "ncsbn.org"],
        add_ins="Confirm that certification is required and that AANP and/or ANCC are recognized certifying organizations."
    )

    # 4) Typical GPA expectations for graduate nursing admissions
    await _edu_group(
        id_prefix="Typical_Graduate_Nursing_GPA_Expectation",
        title="Typical GPA expectations for MSN/DNP admissions",
        claim="Graduate nursing (MSN/DNP) admissions commonly expect a minimum GPA around 3.0 (many programs may be higher), with exact thresholds varying by program.",
        urls=edu.typical_grad_gpa_support_urls,
        allowed_domains=["aanp.org", "aacnnursing.org", ".edu"],
        add_ins="Accept reasonable variations across universities. At least one authoritative source (e.g., a university program page or AACN) must state a typical minimum like ~3.0."
    )


async def build_interstate_section(evaluator: Evaluator, parent_node, inter: InterstateLicensureFacts) -> None:
    node = evaluator.add_parallel(
        id="Interstate_Licensure_Options_IMLC_vs_APRN_Compact",
        desc="Verify IMLC participation as of March 2026 and compare scope/participation between IMLC and APRN Compact, with authoritative URLs.",
        parent=parent_node,
        critical=True
    )

    # 1) IMLC participation count: 42 states + DC + Guam (as of March 2026)
    imlc_group = evaluator.add_sequential(
        id="IMLC_Participation_Count_group",
        desc="IMLC participation count gate + verification",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(inter.imlc_count_support_urls) and _has_any_domain(inter.imlc_count_support_urls, ["imlcc.org"]),
        id="IMLC_Participation_Count_sources_present",
        desc="IMLC participation count: supporting sources provided (prefer imlcc.org)",
        parent=imlc_group,
        critical=True
    )
    imlc_leaf = evaluator.add_leaf(
        id="IMLC_Participation_Count_(March_2026)",
        desc="IMLC includes 42 states plus Washington, D.C. and Guam as of March 2026 with authoritative support.",
        parent=imlc_group,
        critical=True
    )
    await evaluator.verify(
        claim="As of March 2026, the Interstate Medical Licensure Compact (IMLC) includes 42 states plus Washington, D.C., and Guam.",
        node=imlc_leaf,
        sources=inter.imlc_count_support_urls,
        additional_instruction="Use IMLC official sources (imlcc.org). Confirm the participation count and inclusion of DC and Guam."
    )

    # 2) Scope comparison: APRN Compact vs IMLC
    scope_group = evaluator.add_sequential(
        id="Compare_Scope_group",
        desc="Scope comparison gate + verification",
        parent=node,
        critical=True
    )
    scope_urls = _dedup_urls(inter.scope_comparison_support_urls)
    evaluator.add_custom_node(
        result=bool(scope_urls) and (_has_any_domain(scope_urls, ["ncsbn.org"]) or _has_any_domain(scope_urls, ["imlcc.org"])),
        id="Compare_Scope_sources_present",
        desc="Scope comparison: supporting sources provided (ncsbn.org and/or imlcc.org preferred)",
        parent=scope_group,
        critical=True
    )
    scope_leaf = evaluator.add_leaf(
        id="Compare_Scope_(APRN_Compact_vs_IMLC)",
        desc="Compares APRN Compact vs IMLC scope/profession covered with authoritative URLs.",
        parent=scope_group,
        critical=True
    )
    await evaluator.verify(
        claim="The APRN Compact applies to Advanced Practice Registered Nurses (including nurse practitioners), while the IMLC applies to physicians; they are distinct compacts.",
        node=scope_leaf,
        sources=scope_urls,
        additional_instruction="Verify roles/scope from NCSBN (for APRN Compact) and IMLC (imlcc.org) pages."
    )

    # 3) Participation comparison: APRN 4 enacted vs IMLC 42+DC+Guam
    part_group = evaluator.add_sequential(
        id="Compare_Participation_group",
        desc="Participation comparison gate + verification",
        parent=node,
        critical=True
    )
    part_urls = _dedup_urls(inter.participation_comparison_support_urls, inter.imlc_count_support_urls)
    evaluator.add_custom_node(
        result=bool(part_urls) and (_has_any_domain(part_urls, ["ncsbn.org"]) or _has_any_domain(part_urls, ["imlcc.org"])),
        id="Compare_Participation_sources_present",
        desc="Participation comparison: supporting sources provided (ncsbn.org/imlcc.org preferred)",
        parent=part_group,
        critical=True
    )
    part_leaf = evaluator.add_leaf(
        id="Compare_Participation_(APRN_Compact_vs_IMLC)",
        desc="Participation comparison between APRN Compact and IMLC using the stated counts, with authoritative URLs.",
        parent=part_group,
        critical=True
    )
    await evaluator.verify(
        claim="As of 2026, participation is far higher in the IMLC (42 states + Washington, D.C. + Guam) than in the APRN Compact (4 enacted states).",
        node=part_leaf,
        sources=part_urls,
        additional_instruction="Use NCSBN APRN Compact and IMLC official sources to substantiate the participation disparity."
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
    Evaluate an answer for the comprehensive NP career-planning research task (as of 2026).
    """
    # Initialize evaluator with parallel root (root is always non-critical in framework)
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

    # Batch extract all sections concurrently
    bls_task = evaluator.extract(
        prompt=prompt_extract_bls(),
        template_class=BLSFacts,
        extraction_name="bls_facts"
    )
    state_task = evaluator.extract(
        prompt=prompt_extract_state_selection(),
        template_class=StateSelectionFacts,
        extraction_name="state_selection_facts"
    )
    aprn_task = evaluator.extract(
        prompt=prompt_extract_aprn_compact(),
        template_class=APRNCompactFacts,
        extraction_name="aprn_compact_facts"
    )
    edu_task = evaluator.extract(
        prompt=prompt_extract_education_licensure(),
        template_class=EducationLicensureFacts,
        extraction_name="education_licensure_facts"
    )
    inter_task = evaluator.extract(
        prompt=prompt_extract_interstate_licensure(),
        template_class=InterstateLicensureFacts,
        extraction_name="interstate_licensure_facts"
    )

    bls, state_sel, aprn, edu, inter = await asyncio.gather(
        bls_task, state_task, aprn_task, edu_task, inter_task
    )

    # Build top-level critical category node (to mirror rubric organization)
    # Even though root cannot be critical in the framework, we create one critical umbrella parallel node
    top_node = evaluator.add_parallel(
        id="Career_Planning_Research_for_Nurse_Practitioner",
        desc="Evaluate the required career-planning research outputs for becoming a nurse practitioner in the US, including BLS projections, state/compact status, education/licensure, and interstate compact comparison, with authoritative references.",
        parent=root,
        critical=True
    )

    # Build sections
    await build_bls_section(evaluator, top_node, bls)
    await build_state_selection_section(evaluator, top_node, state_sel)
    await build_aprn_compact_section(evaluator, top_node, aprn)
    await build_education_licensure_section(evaluator, top_node, edu)
    await build_interstate_section(evaluator, top_node, inter)

    return evaluator.get_summary()