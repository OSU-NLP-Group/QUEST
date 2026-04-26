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
TASK_ID = "phd_stipend_comparison_mit_stanford_cmu"
TASK_DESCRIPTION = """
Among the three computer science doctoral programs at MIT (EECS), Stanford (CS), and Carnegie Mellon University (CSD), determine which program provides the highest guaranteed minimum annual stipend for PhD students in the 2024-2025 or 2025-2026 academic year. For the program you identify, verify and provide detailed information about all of the following funding components: (1) base monthly RA stipend amount, (2) base monthly TA stipend amount, (3) annual 12-month stipend total, (4) tuition coverage policy, (5) health insurance annual cost, (6) funding duration guarantee, (7) summer funding policy, (8) external fellowship supplementation policy, (9) dependency allowance policy, (10) high-range stipend option (if applicable), (11) funding contingency requirements, and (12) official source URL for each component. Additionally, for the two programs that did NOT have the highest stipend, provide their base annual stipend amounts and explain why they rank lower.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FundingComponent(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class SelectedInstitutionComponents(BaseModel):
    program_name: Optional[str] = None
    year_range: Optional[str] = None
    phd_scope_statement: Optional[str] = None
    standard_ra_ta_basis_statement: Optional[str] = None
    twelve_month_minimum_metric_statement: Optional[str] = None

    base_monthly_ra: FundingComponent = Field(default_factory=FundingComponent)
    base_monthly_ta: FundingComponent = Field(default_factory=FundingComponent)
    annual_12mo_total: FundingComponent = Field(default_factory=FundingComponent)

    tuition_coverage: FundingComponent = Field(default_factory=FundingComponent)
    health_insurance_annual_cost: FundingComponent = Field(default_factory=FundingComponent)
    funding_duration: FundingComponent = Field(default_factory=FundingComponent)
    summer_funding: FundingComponent = Field(default_factory=FundingComponent)
    external_fellowship: FundingComponent = Field(default_factory=FundingComponent)
    dependency_allowance: FundingComponent = Field(default_factory=FundingComponent)
    high_range_stipend_option: FundingComponent = Field(default_factory=FundingComponent)
    contingency_requirements: FundingComponent = Field(default_factory=FundingComponent)


class NonHighestProgram(BaseModel):
    program_name: Optional[str] = None
    base_annual_12mo_total: FundingComponent = Field(default_factory=FundingComponent)
    reason_lower: FundingComponent = Field(default_factory=FundingComponent)


class StipendExtraction(BaseModel):
    selected: SelectedInstitutionComponents = Field(default_factory=SelectedInstitutionComponents)
    non_highest: List[NonHighestProgram] = Field(default_factory=list)
    ranking_explanation_text: Optional[str] = None
    ranking_explanation_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract structured information from the answer for a stipend comparison among MIT EECS (doctoral/PhD), Stanford CS (doctoral/PhD), and CMU CSD (doctoral/PhD). The comparison must be for the 2024–2025 or 2025–2026 academic year and should rank by the guaranteed minimum annual stipend on a full 12-month basis using standard RA/TA base rates, not special fellowships.

    Return a JSON with the following structure:

    {
      "selected": {
        "program_name": string or null,  // The program identified as having the highest guaranteed minimum annual stipend; use a concise program label such as "MIT EECS", "Stanford CS", or "CMU CSD" if possible.
        "year_range": string or null,    // e.g., "2024–2025" or "2025–2026"
        "phd_scope_statement": string or null, // A phrase/sentence from the answer that shows this comparison is for PhD students
        "standard_ra_ta_basis_statement": string or null, // A phrase/sentence indicating the base is standard RA/TA appointments
        "twelve_month_minimum_metric_statement": string or null, // A phrase/sentence confirming the metric is a guaranteed minimum annual 12-month stipend

        "base_monthly_ra": {"value": string or null, "urls": [urls...]},
        "base_monthly_ta": {"value": string or null, "urls": [urls...]},
        "annual_12mo_total": {"value": string or null, "urls": [urls...]},

        "tuition_coverage": {"value": string or null, "urls": [urls...]},
        "health_insurance_annual_cost": {"value": string or null, "urls": [urls...]},
        "funding_duration": {"value": string or null, "urls": [urls...]},
        "summer_funding": {"value": string or null, "urls": [urls...]},
        "external_fellowship": {"value": string or null, "urls": [urls...]},
        "dependency_allowance": {"value": string or null, "urls": [urls...]},
        "high_range_stipend_option": {"value": string or null, "urls": [urls...]},
        "contingency_requirements": {"value": string or null, "urls": [urls...]}
      },

      "non_highest": [
        {
          "program_name": string or null,
          "base_annual_12mo_total": {"value": string or null, "urls": [urls...]},
          "reason_lower": {"value": string or null, "urls": [urls...]} // explanation for why this program ranks lower (e.g., lower 12-month base or non-comparable/undisclosed), with official source URL(s)
        },
        {
          "program_name": string or null,
          "base_annual_12mo_total": {"value": string or null, "urls": [urls...]},
          "reason_lower": {"value": string or null, "urls": [urls...]}
        }
      ],

      "ranking_explanation_text": string or null,   // overall ranking narrative
      "ranking_explanation_urls": [urls...]         // official URL(s) cited for the justification
    }

    IMPORTANT GUIDELINES:
    - Only extract URLs explicitly present in the answer text (including markdown links); do not invent or infer any URLs. Include complete URLs.
    - Prefer official university (.edu) URLs when present.
    - Values can be strings (e.g., "$4,500 per month" or "$54,000 per year"); do not normalize to numeric types.
    - If some field is not present in the answer, set it to null and an empty URL list for that field.
    - The two items in "non_highest" should correspond to the two programs that are NOT selected as the highest.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _merge_urls(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for u in lst or []:
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _first_two_non_highest(non_highest: List[NonHighestProgram], selected_name: Optional[str]) -> List[NonHighestProgram]:
    if not non_highest:
        return []
    # Filter out any entry that matches the selected program name (loose contains match)
    filtered: List[NonHighestProgram] = []
    for item in non_highest:
        if not selected_name or not item.program_name:
            filtered.append(item)
        else:
            if item.program_name.strip().lower() != selected_name.strip().lower():
                filtered.append(item)
    # Take first two
    return filtered[:2]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_highest_stipend_determination(evaluator: Evaluator, root_node, data: StipendExtraction) -> None:
    node = evaluator.add_parallel(
        id="Highest_Stipend_Determination",
        desc="Select the highest guaranteed minimum annual (12-month) PhD stipend program among the three for 2024–2025 or 2025–2026, using the required comparison metric and basis.",
        parent=root_node,
        critical=True
    )

    # 1) Selected institution is among the three
    leaf_sel = evaluator.add_leaf(
        id="Selected_Institution_Is_Among_Three",
        desc="The identified highest-stipend program is one of: MIT EECS, Stanford CS, or CMU CSD.",
        parent=node,
        critical=True
    )
    sel_name = data.selected.program_name or ""
    claim_sel = f"The selected highest-stipend program named in the answer ('{sel_name}') is one of MIT EECS, Stanford CS, or CMU CSD (allowing reasonable naming variants)."
    await evaluator.verify(
        claim=claim_sel,
        node=leaf_sel,
        additional_instruction="Judge based only on the answer text. Minor naming variants like 'MIT EECS PhD' or 'Stanford Computer Science PhD' should count as matches."
    )

    # 2) Comparison is for PhD students
    leaf_phd = evaluator.add_leaf(
        id="Comparison_Is_For_PhD_Students",
        desc="The comparison is explicitly for doctoral (PhD) students, not master’s students.",
        parent=node,
        critical=True
    )
    phd_stmt = data.selected.phd_scope_statement or ""
    claim_phd = f"The comparison is explicitly for doctoral (PhD) students, as indicated by: '{phd_stmt}'."
    await evaluator.verify(
        claim=claim_phd,
        node=leaf_phd,
        additional_instruction="Verify that the answer explicitly frames the comparison for PhD students."
    )

    # 3) Year in range
    leaf_year = evaluator.add_leaf(
        id="Comparison_Year_In_Range",
        desc="The comparison uses stipend information for 2024–2025 or 2025–2026.",
        parent=node,
        critical=True
    )
    yr = data.selected.year_range or ""
    claim_year = f"The comparison uses stipend information for 2024–2025 or 2025–2026; the year/term specified in the answer is '{yr}'."
    await evaluator.verify(
        claim=claim_year,
        node=leaf_year,
        additional_instruction="Accept if the answer clearly indicates 2024–2025 or 2025–2026. Minor formatting like '2024-25' is acceptable."
    )

    # 4) Uses standard RA/TA base
    leaf_base = evaluator.add_leaf(
        id="Comparison_Uses_Standard_RA_TA_Base",
        desc="The base stipend figures used for comparison reflect standard RA/TA appointments, not special fellowships or external funding rates.",
        parent=node,
        critical=True
    )
    base_stmt = data.selected.standard_ra_ta_basis_statement or ""
    claim_base = f"The comparison uses standard RA/TA base rates (not special fellowships), as indicated by: '{base_stmt}'."
    await evaluator.verify(
        claim=claim_base,
        node=leaf_base,
        additional_instruction="Judge from the answer text whether base RA/TA rates are used as the comparison basis."
    )

    # 5) Metric is guaranteed minimum annual 12-month
    leaf_metric = evaluator.add_leaf(
        id="Comparison_Metric_Is_Guaranteed_Minimum_Annual_12_Month",
        desc="The ranking metric is a guaranteed minimum annual stipend on a full 12-month basis.",
        parent=node,
        critical=True
    )
    metric_stmt = data.selected.twelve_month_minimum_metric_statement or ""
    claim_metric = f"The ranking metric is a guaranteed minimum annual stipend on a full 12-month basis, as indicated by: '{metric_stmt}'."
    await evaluator.verify(
        claim=claim_metric,
        node=leaf_metric,
        additional_instruction="Judge based on the answer text. The metric should be clearly '12-month' and 'guaranteed minimum'."
    )

    # 6) Selected program shown highest by reported figures
    leaf_highest = evaluator.add_leaf(
        id="Selected_Program_Shown_Highest_By_Reported_Figures",
        desc="Given the stipend/guarantee figures (or documented non-disclosure/non-comparability) reported for the three programs elsewhere in the answer, the selected program is correctly justified as ranking highest under the required metric.",
        parent=node,
        critical=True
    )
    # Collect amounts as strings
    sel_annual = data.selected.annual_12mo_total.value or ""
    others = _first_two_non_highest(data.non_highest, data.selected.program_name)
    other1 = others[0] if len(others) > 0 else NonHighestProgram()
    other2 = others[1] if len(others) > 1 else NonHighestProgram()

    claim_highest = (
        f"Based on the extracted annual 12-month stipend totals or official explanations, "
        f"the selected program '{sel_name}' with annual total '{sel_annual}' ranks highest "
        f"relative to {other1.program_name or 'Program A'} (annual '{other1.base_annual_12mo_total.value or 'N/A'}') "
        f"and {other2.program_name or 'Program B'} (annual '{other2.base_annual_12mo_total.value or 'N/A'}'). "
        f"If one of the non-selected programs lacks a directly comparable guaranteed minimum annual base, "
        f"the explanation provided supports why it cannot outrank the selected program."
    )
    await evaluator.verify(
        claim=claim_highest,
        node=leaf_highest,
        additional_instruction="Use the values/explanations in the answer to assess relative ranking conceptually; do not fetch new info."
    )


async def build_selected_components(evaluator: Evaluator, root_node, sel: SelectedInstitutionComponents) -> None:
    node = evaluator.add_parallel(
        id="Highest_Institution_Funding_Components",
        desc="For the selected highest-stipend institution, provide and verify all required funding components for PhD students with standard RA/TA appointments for 2024–2025 or 2025–2026.",
        parent=root_node,
        critical=True
    )

    prog = sel.program_name or "the selected program"
    yr = sel.year_range or "the specified year"

    # Component 1: Base Monthly RA
    c1 = evaluator.add_parallel(
        id="Component_1_Base_Monthly_RA_Stipend",
        desc="Provide the base monthly RA stipend amount with an official source URL.",
        parent=node,
        critical=True
    )
    # Existence
    evaluator.add_custom_node(
        result=bool(sel.base_monthly_ra.value and sel.base_monthly_ra.value.strip()),
        id="RA_Stipend_Amount_Stated",
        desc="A base RA stipend amount is stated as a monthly amount for the relevant year range.",
        parent=c1,
        critical=True
    )
    # URL support
    ra_url_leaf = evaluator.add_leaf(
        id="RA_Stipend_Official_URL",
        desc="An accessible official university (.edu) URL is provided that supports the RA stipend amount.",
        parent=c1,
        critical=True
    )
    ra_claim = f"For {prog} in {yr}, the base monthly Research Assistant (RA) stipend for a standard PhD appointment is '{sel.base_monthly_ra.value or ''}'."
    await evaluator.verify(
        claim=ra_claim,
        node=ra_url_leaf,
        sources=sel.base_monthly_ra.urls,
        additional_instruction="Confirm the stipend amount and that the URL is an official university (.edu) source relevant to PhD RA base rates for 2024–2025 or 2025–2026."
    )

    # Component 2: Base Monthly TA
    c2 = evaluator.add_parallel(
        id="Component_2_Base_Monthly_TA_Stipend",
        desc="Provide the base monthly TA stipend amount with an official source URL.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(sel.base_monthly_ta.value and sel.base_monthly_ta.value.strip()),
        id="TA_Stipend_Amount_Stated",
        desc="A base TA stipend amount is stated as a monthly amount for the relevant year range.",
        parent=c2,
        critical=True
    )
    ta_url_leaf = evaluator.add_leaf(
        id="TA_Stipend_Official_URL",
        desc="An accessible official university (.edu) URL is provided that supports the TA stipend amount.",
        parent=c2,
        critical=True
    )
    ta_claim = f"For {prog} in {yr}, the base monthly Teaching Assistant (TA) stipend for a standard PhD appointment is '{sel.base_monthly_ta.value or ''}'."
    await evaluator.verify(
        claim=ta_claim,
        node=ta_url_leaf,
        sources=sel.base_monthly_ta.urls,
        additional_instruction="Confirm the stipend amount and that the URL is an official university (.edu) source relevant to PhD TA base rates for 2024–2025 or 2025–2026."
    )

    # Component 3: Annual 12-month Total
    c3 = evaluator.add_parallel(
        id="Component_3_Annual_12_Month_Stipend_Total",
        desc="Provide the annual stipend total on a full 12-month basis with an official source URL (and/or verifiable derivation).",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(sel.annual_12mo_total.value and sel.annual_12mo_total.value.strip()),
        id="Annual_Total_Provided_On_12_Month_Basis",
        desc="The annual stipend total is explicitly given on a full 12-month basis.",
        parent=c3,
        critical=True
    )
    # Consistency check via reasoning
    annual_consistency_leaf = evaluator.add_leaf(
        id="Annual_Total_Consistent_With_Monthly_Rates",
        desc="The annual 12-month total is consistent with the stated base monthly rate(s) (or the calculation is shown clearly).",
        parent=c3,
        critical=True
    )
    ra_txt = sel.base_monthly_ra.value or ""
    ta_txt = sel.base_monthly_ta.value or ""
    annual_txt = sel.annual_12mo_total.value or ""
    consistency_claim = (
        f"The stated annual 12-month stipend total '{annual_txt}' is consistent with the base monthly RA/TA rate(s) "
        f"(RA: '{ra_txt}', TA: '{ta_txt}') times 12 within reasonable rounding, or the answer shows a clear derivation."
    )
    await evaluator.verify(
        claim=consistency_claim,
        node=annual_consistency_leaf,
        additional_instruction="Use the provided amounts in the answer to judge arithmetic consistency. Minor rounding and whether RA or TA base applies are acceptable if justified."
    )
    annual_url_leaf = evaluator.add_leaf(
        id="Annual_Total_Official_URL",
        desc="An accessible official university (.edu) URL is provided that states the annual total or provides the base rate(s) needed to verify the 12-month total.",
        parent=c3,
        critical=True
    )
    annual_sources = _merge_urls(sel.annual_12mo_total.urls, sel.base_monthly_ra.urls, sel.base_monthly_ta.urls)
    annual_claim = (
        f"For {prog} in {yr}, the guaranteed minimum annual stipend on a 12-month basis is '{sel.annual_12mo_total.value or ''}', "
        f"or the base monthly rate(s) provided can be used to compute the 12-month total."
    )
    await evaluator.verify(
        claim=annual_claim,
        node=annual_url_leaf,
        sources=annual_sources,
        additional_instruction="Accept if the URL either states the annual total explicitly or clearly states base monthly RA/TA rates that imply the 12-month total. Confirm it is an official (.edu) page."
    )

    # Generic builder for the policy components (4–11)
    async def _policy_component(comp_id_base: str, title: str, stated_desc: str, url_desc: str,
                                comp_field: FundingComponent, policy_kind: str) -> None:
        comp_node = evaluator.add_parallel(
            id=comp_id_base,
            desc=title,
            parent=node,
            critical=True
        )
        # Stated/existence check
        evaluator.add_custom_node(
            result=bool(comp_field.value and comp_field.value.strip()),
            id=f"{comp_id_base}_Stated",
            desc=stated_desc,
            parent=comp_node,
            critical=True
        )
        # URL support
        url_leaf = evaluator.add_leaf(
            id=f"{comp_id_base}_Official_URL",
            desc=url_desc,
            parent=comp_node,
            critical=True
        )
        policy_claim = f"For {prog} in {yr}, the {policy_kind} is described as: '{comp_field.value or ''}'."
        await evaluator.verify(
            claim=policy_claim,
            node=url_leaf,
            sources=comp_field.urls,
            additional_instruction="Verify that the official (.edu) page supports this exact policy statement for PhD students in the specified year. If the policy is that no such benefit exists, confirm the page documents its absence."
        )

    # Component 4: Tuition coverage
    await _policy_component(
        "Component_4_Tuition_Coverage_Policy",
        "Provide the tuition coverage policy for standard RA/TA appointments with an official source URL.",
        "Tuition coverage is described for standard RA/TA appointments, specifying whether full or partial tuition is covered.",
        "An accessible official university (.edu) URL is provided supporting the tuition coverage policy.",
        sel.tuition_coverage,
        "tuition coverage policy for standard RA/TA appointments"
    )

    # Component 5: Health insurance annual cost
    await _policy_component(
        "Component_5_Health_Insurance_Annual_Cost",
        "Provide the annual cost of the university student health insurance plan with an official source URL.",
        "The annual cost amount for the university’s student health insurance plan is stated for the relevant year if available.",
        "An accessible official university (.edu) URL is provided supporting the health insurance annual cost.",
        sel.health_insurance_annual_cost,
        "student health insurance annual cost"
    )

    # Component 6: Funding duration guarantee
    await _policy_component(
        "Component_6_Funding_Duration_Guarantee",
        "Provide the funding duration guarantee with an official source URL.",
        "The funding duration guarantee is stated (e.g., throughout PhD or limited number of years) for PhD students.",
        "An accessible official university (.edu) URL is provided supporting the funding duration guarantee.",
        sel.funding_duration,
        "funding duration guarantee"
    )

    # Component 7: Summer funding policy
    await _policy_component(
        "Component_7_Summer_Funding_Policy",
        "Provide the summer funding policy with an official source URL.",
        "The summer funding policy is stated, clarifying whether summer support is included in 12-month appointments or requires separate arrangements.",
        "An accessible official university (.edu) URL is provided supporting the summer funding policy.",
        sel.summer_funding,
        "summer funding policy"
    )

    # Component 8: External fellowship supplementation
    await _policy_component(
        "Component_8_External_Fellowship_Supplementation",
        "Provide the external fellowship supplementation policy with an official source URL.",
        "The policy describing how external fellowships interact with institutional funding (supplementation/top-ups/offsets) is stated.",
        "An accessible official university (.edu) URL is provided supporting the external fellowship supplementation policy.",
        sel.external_fellowship,
        "external fellowship supplementation policy"
    )

    # Component 9: Dependency allowance policy
    await _policy_component(
        "Component_9_Dependency_Allowance_Policy",
        "Provide the dependency allowance policy with an official source URL.",
        "Whether additional financial support is provided for students with dependent children or spouses is stated (including explicit absence if none).",
        "An accessible official university (.edu) URL is provided supporting the dependency allowance policy (or documenting its absence).",
        sel.dependency_allowance,
        "dependency allowance (or the explicit absence of it)"
    )

    # Component 10: High-range stipend option (if applicable)
    await _policy_component(
        "Component_10_High_Range_Stipend_Option_If_Applicable",
        "Address whether a high-range stipend option above the base rate exists, with an official source URL.",
        "If a high-range option exists, it is described; otherwise, the answer explicitly notes that no such option is documented.",
        "An accessible official university (.edu) URL is provided supporting the high-range stipend option (or its absence).",
        sel.high_range_stipend_option,
        "high-range stipend option status"
    )

    # Component 11: Funding contingency requirements
    await _policy_component(
        "Component_11_Funding_Contingency_Requirements",
        "Provide the funding contingency requirements with an official source URL.",
        "Any academic progress or other conditions required to maintain funding are stated.",
        "An accessible official university (.edu) URL is provided supporting the funding contingency requirements.",
        sel.contingency_requirements,
        "funding contingency requirements (e.g., satisfactory progress)"
    )


async def build_other_programs_and_ranking(evaluator: Evaluator, root_node, data: StipendExtraction) -> None:
    node = evaluator.add_parallel(
        id="Other_Two_Programs_And_Ranking_Explanation",
        desc="For the two programs that are not highest, provide their base annual stipend amounts (or explain why they cannot be directly compared) and explain why they rank lower, using verifiable official sources.",
        parent=root_node,
        critical=True
    )

    others = _first_two_non_highest(data.non_highest, data.selected.program_name)
    while len(others) < 2:
        others.append(NonHighestProgram())

    # Helper to build for each non-highest program
    async def _build_non_highest(idx: int, program: NonHighestProgram, node_id: str, node_desc: str,
                                 leaf_base_id: str, leaf_url_id: str):
        program_node = evaluator.add_parallel(
            id=node_id,
            desc=node_desc,
            parent=node,
            critical=True
        )

        # Leaf: base annual stipend OR non-comparability explanation, verified by URL
        base_or_expl_leaf = evaluator.add_leaf(
            id=leaf_base_id,
            desc="Provides a base annual stipend amount for PhD standard RA/TA for 2024–2025 or 2025–2026, OR explains (based on official info) why it cannot be directly compared/verified.",
            parent=program_node,
            critical=True
        )
        if program.base_annual_12mo_total.value and program.base_annual_12mo_total.value.strip():
            claim = (
                f"For {program.program_name or f'Program {idx+1}'}, the base annual 12-month stipend for standard PhD RA/TA in the specified year "
                f"is '{program.base_annual_12mo_total.value}'."
            )
            srcs = program.base_annual_12mo_total.urls
        else:
            claim = (
                f"For {program.program_name or f'Program {idx+1}'}, the official source explains why a directly comparable guaranteed minimum annual base cannot be verified: "
                f"'{program.reason_lower.value or ''}'."
            )
            srcs = program.reason_lower.urls

        await evaluator.verify(
            claim=claim,
            node=base_or_expl_leaf,
            sources=srcs,
            additional_instruction="Confirm that the URL is an official (.edu) source and that it either states the base annual 12-month stipend or documents why a comparable guaranteed minimum cannot be verified."
        )

        # Leaf: official URL presence and .edu (structural check)
        url_presence = evaluator.add_custom_node(
            result=bool((_merge_urls(program.base_annual_12mo_total.urls, program.reason_lower.urls))),
            id=leaf_url_id,
            desc="An accessible official university (.edu) URL is provided supporting the stated amount or the non-comparability claim.",
            parent=program_node,
            critical=True
        )

    await _build_non_highest(
        0, others[0],
        node_id="NonHighest_Program_1",
        node_desc="Provide base annual stipend information (or non-comparability explanation) for one non-highest program, with an official URL.",
        leaf_base_id="Program1_Base_Annual_Stipend_Or_NonComparability",
        leaf_url_id="Program1_Official_URL"
    )
    await _build_non_highest(
        1, others[1],
        node_id="NonHighest_Program_2",
        node_desc="Provide base annual stipend information (or non-comparability explanation) for the other non-highest program, with an official URL.",
        leaf_base_id="Program2_Base_Annual_Stipend_Or_NonComparability",
        leaf_url_id="Program2_Official_URL"
    )

    # Ranking justification node
    rj = evaluator.add_parallel(
        id="Ranking_Justification",
        desc="Explain why the selected program ranks highest and the other two rank lower under the required metric, consistent with the officially sourced information provided.",
        parent=node,
        critical=True
    )
    # Existence: explanation provided
    evaluator.add_custom_node(
        result=bool(data.ranking_explanation_text and data.ranking_explanation_text.strip()),
        id="Ranking_Explanation_Provided",
        desc="A clear ranking justification is provided (e.g., higher guaranteed minimum annual 12-month stipend, or others cannot be compared due to non-disclosed guaranteed minimum).",
        parent=rj,
        critical=True
    )
    # Consistency: explanation aligns with cited data
    consistency_leaf = evaluator.add_leaf(
        id="Ranking_Explanation_Consistent_With_Cited_Data",
        desc="The ranking explanation is consistent with the stipend/guarantee figures or non-comparability claims supported by the official URLs provided for the three programs.",
        parent=rj,
        critical=True
    )
    sel_name = data.selected.program_name or "Selected Program"
    sel_annual = data.selected.annual_12mo_total.value or ""
    o1n = others[0].program_name or "Program A"
    o1a = others[0].base_annual_12mo_total.value or ""
    o2n = others[1].program_name or "Program B"
    o2a = others[1].base_annual_12mo_total.value or ""
    rank_claim = (
        f"The ranking explanation is consistent with the extracted amounts/explanations: "
        f"{sel_name} '{sel_annual}' vs {o1n} '{o1a}' and {o2n} '{o2a}', and any non-comparability justifications."
    )
    await evaluator.verify(
        claim=rank_claim,
        node=consistency_leaf,
        additional_instruction="Judge consistency using only the extracted amounts/explanations and the answer's citations."
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Root is critical in the rubric; all children must be critical (enforced by VerificationNode)
    # We use three sequential phases: determination -> selected components -> other programs and ranking.
    # Extract all structured data in one shot
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=StipendExtraction,
        extraction_name="stipend_comparison_extraction"
    )

    # Phase 1: Highest stipend determination
    await build_highest_stipend_determination(evaluator, root, extraction)

    # Phase 2: Selected institution components
    await build_selected_components(evaluator, root, extraction.selected)

    # Phase 3: Other two programs & ranking explanation
    await build_other_programs_and_ranking(evaluator, root, extraction)

    return evaluator.get_summary()