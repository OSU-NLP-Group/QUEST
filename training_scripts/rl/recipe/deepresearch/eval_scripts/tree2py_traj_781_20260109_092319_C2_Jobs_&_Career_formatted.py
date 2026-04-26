import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cissp_eligibility"
TASK_DESCRIPTION = """
A cybersecurity professional is evaluating their eligibility to apply for the CISSP certification. They have the following background:

- Education: Bachelor's degree in Computer Science from an accredited university
- Work Experience:
  - 3 years full-time (35+ hours/week) as a Security Analyst focusing on security operations and incident response
  - 2080 hours part-time (25 hours/week) as a Network Security Consultant working on network architecture and firewall configurations

Their work experience primarily falls under:
1. Domain 4: Communication and Network Security
2. Domain 7: Security Operations

Based on the current CISSP certification requirements from ISC2, does this professional meet the eligibility requirements to apply for CISSP certification? Provide a detailed analysis including:
1. Total qualifying years of work experience after appropriate conversions
2. Whether and how any education waiver applies
3. Whether the domain coverage requirement is satisfied
4. Final determination of eligibility with supporting reasoning
"""

# Ground-truth interpretation based on widely accepted ISC2 CISSP rules
# – Full-time experience: >= 35 hours/week
# – Part-time experience: 20–34 hours/week; 1,000 hours = 0.5 year; 2,000 hours = 1 year; max 2 years can be credited
# – Experience requirement: 5 years total in 2+ domains; a valid one-year waiver for a four-year degree or approved credential can reduce to 4 years (no stacking)
EXPECTED_FULL_TIME_YEARS = 3.0
EXPECTED_PART_TIME_HOURS = 2080
EXPECTED_CONVERTED_PT_YEARS = 1.0  # 2080h satisfies the 2,000h threshold → 1.0 year credited
EXPECTED_TOTAL_YEARS = EXPECTED_FULL_TIME_YEARS + EXPECTED_CONVERTED_PT_YEARS  # 4.0 years
EXPECTED_WAIVER_APPLIES = True  # Bachelor's degree in CS from an accredited institution qualifies for a 1-year waiver
EXPECTED_DOMAINS = [
    "Domain 4: Communication and Network Security",
    "Domain 7: Security Operations",
]
EXPECTED_ELIGIBILITY = True  # Eligible via waiver path: 4 years experience + 1-year waiver + coverage in >= 2 domains

RULES_SUMMARY = {
    "experience_rules": [
        "Full-time experience is defined as ≥35 hours/week.",
        "Part-time experience is 20–34 hours/week.",
        "Part-time conversion: 1,000 hours = 0.5 year; 2,000 hours = 1 year; a maximum of 2 years of part-time credit can be applied.",
        "Total qualifying experience = full-time years + converted part-time years."
    ],
    "waiver_rules": [
        "A one-year experience waiver is available for a four-year college degree (or equivalent) or certain approved credentials.",
        "Only one waiver can be applied; waivers cannot be stacked."
    ],
    "eligibility_logic": [
        "Eligible if total qualifying experience ≥ 5 years; OR",
        "Eligible if a valid 1-year waiver is applied AND total qualifying experience ≥ 4 years.",
        "Experience must span at least two of the eight CISSP CBK domains."
    ]
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CISSPAnalysisExtraction(BaseModel):
    """Structured extraction of the agent's CISSP eligibility analysis."""
    # Experience computation as stated in the answer
    total_experience_years: Optional[str] = None
    full_time_years: Optional[str] = None
    part_time_hours: Optional[str] = None
    part_time_weekly_hours: Optional[str] = None
    converted_part_time_years: Optional[str] = None
    conversion_basis: Optional[str] = None  # e.g., "2000 hours = 1 year" or similar text

    # Waiver analysis
    waiver_applied: Optional[bool] = None
    waiver_years: Optional[str] = None
    waiver_basis: Optional[str] = None  # e.g., "Bachelor's degree in CS from accredited university"

    # Domain coverage
    domains_list: List[str] = Field(default_factory=list)

    # Final eligibility determination
    eligible_final: Optional[bool] = None
    eligibility_reason: Optional[str] = None

    # Policy or requirement URLs cited in the answer (if any)
    policy_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cissp_analysis() -> str:
    return """
    Extract the CISSP eligibility analysis as presented in the answer. Return a single JSON object with the following fields:

    Experience:
    - total_experience_years: The total qualifying full-time-equivalent experience computed in the answer (as a string, e.g., "4 years" or "4.0")
    - full_time_years: The full-time years counted (string as presented, e.g., "3")
    - part_time_hours: The part-time hours counted (string as presented, e.g., "2080")
    - part_time_weekly_hours: The weekly hours for the part-time experience (string as presented, e.g., "25")
    - converted_part_time_years: The converted part-time years credited (string as presented, e.g., "1" or "1.0")
    - conversion_basis: Any explicit conversion rule the answer uses (free text, e.g., "2000 hours = 1 year")

    Waiver:
    - waiver_applied: true/false if the answer says a waiver applies
    - waiver_years: The number of years waived (string, e.g., "1" or "1.0")
    - waiver_basis: The stated basis for the waiver (e.g., "Bachelor's degree in Computer Science from accredited university")

    Domains:
    - domains_list: An array of domain labels or names the answer claims the experience covers (e.g., ["Domain 4: Communication and Network Security", "Domain 7: Security Operations"])

    Final determination:
    - eligible_final: true/false if the answer concludes eligible or ineligible
    - eligibility_reason: The stated reasoning for the final determination (free text)

    Sources:
    - policy_urls: An array of URLs the answer cites for CISSP requirements (extract actual URLs; if none, return an empty array)

    IMPORTANT:
    - Preserve wording and numbers as they appear in the answer for the string fields.
    - If the answer does not include a field, set it to null (for scalars) or an empty array (for lists).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: CISSPAnalysisExtraction) -> None:
    """
    Build the verification tree according to the rubric and perform verifications.
    Root in Evaluator is non-critical by design; we add a top-level critical parallel node
    to reflect rubric's "Root" critical aggregation over four critical leaf checks.
    """
    # Top-level critical node to mirror rubric Root
    root_critical = evaluator.add_parallel(
        id="cissp_main",
        desc="Answer correctly determines CISSP eligibility and provides all requested analysis components",
        parent=evaluator.root,
        critical=True
    )

    # 1) Total Qualifying Experience (critical leaf)
    total_exp_node = evaluator.add_leaf(
        id="Total_Qualifying_Experience",
        desc="Computes total qualifying full-time-equivalent experience correctly, including correct part-time conversion per the provided rules",
        parent=root_critical,
        critical=True
    )

    # Build claim using extraction and expected interpretation
    total_exp_claim = (
        f"The answer correctly computes the total qualifying FTE experience. It should be 4.0 years, "
        f"derived from 3 full-time years (≥35 h/week) plus 2080 hours of part-time at 25 h/week, "
        f"where ISC2 counts part-time as 0.5 year per 1000 hours and 1.0 year per 2000 hours (max 2 years). "
        f"The answer's stated total is '{extraction.total_experience_years}', and its converted part-time years are "
        f"'{extraction.converted_part_time_years}'. Under the stated rules, 2080 hours qualifies as 1.0 year, "
        f"so the correct total is 4.0 years."
    )
    await evaluator.verify(
        claim=total_exp_claim,
        node=total_exp_node,
        additional_instruction=(
            "Use these CISSP experience rules:\n"
            "1) Full-time experience is ≥35 hours/week.\n"
            "2) Part-time experience is 20–34 hours/week; 1000 hours = 0.5 year; 2000 hours = 1 year; maximum 2 years part-time credit.\n"
            "3) Total qualifying experience = full-time years + converted part-time years.\n"
            "Evaluate the answer's computation given 3 full-time years and 2080 part-time hours at 25 h/week; "
            "the correct credited part-time is 1.0 year and the correct total is 4.0 years."
        ),
    )

    # 2) Waiver Analysis (critical leaf)
    waiver_node = evaluator.add_leaf(
        id="Waiver_Analysis",
        desc="Correctly determines whether a waiver applies and applies it consistent with constraints (max 1 year; cannot stack multiple waivers)",
        parent=root_critical,
        critical=True
    )

    waiver_claim = (
        f"The answer correctly identifies and applies the CISSP experience waiver: a one-year waiver applies "
        f"for a four-year Bachelor's degree from an accredited institution. The answer indicates waiver_applied="
        f"{extraction.waiver_applied}, waiver_years='{extraction.waiver_years}', basis='{extraction.waiver_basis}'. "
        f"The waiver must be at most 1 year and cannot be stacked with other waivers."
    )
    await evaluator.verify(
        claim=waiver_claim,
        node=waiver_node,
        additional_instruction=(
            "Waiver rules: A one-year experience waiver is available for a four-year college degree (or equivalent) "
            "or certain approved credentials. Only one waiver can be applied; waivers cannot be stacked. "
            "Assess whether the answer's waiver analysis (applied or not, years waived, and basis) is correct for the stated Bachelor's degree in CS."
        ),
    )

    # 3) Domain Coverage (critical leaf)
    domains_node = evaluator.add_leaf(
        id="Domain_Coverage",
        desc="Correctly verifies experience spans at least two of the eight CISSP domains (and identifies the domains used)",
        parent=root_critical,
        critical=True
    )

    domains_list_display = ", ".join(extraction.domains_list) if extraction.domains_list else "[]"
    domains_claim = (
        f"The answer shows experience spanning at least two CISSP CBK domains. "
        f"In particular, it identifies: {domains_list_display}. "
        f"This satisfies the requirement of experience in two or more of the eight CISSP domains."
    )
    await evaluator.verify(
        claim=domains_claim,
        node=domains_node,
        additional_instruction=(
            "Requirement: Experience must span at least two of the eight CISSP CBK domains. "
            "Verify that the answer explicitly identifies at least two distinct domains covered by the experience. "
            "Allow minor naming or formatting variants (e.g., 'Domain 4: Communication and Network Security', 'Domain 7: Security Operations')."
        ),
    )

    # 4) Eligibility Determination (critical leaf)
    eligibility_node = evaluator.add_leaf(
        id="Eligibility_Determination",
        desc="Provides the final eligible/ineligible determination with reasoning consistent with: (experience ≥5 years) OR (valid waiver applied AND experience ≥4 years), and with the domain coverage requirement",
        parent=root_critical,
        critical=True
    )

    final_label = "eligible" if extraction.eligible_final else "ineligible"
    eligibility_claim = (
        f"The final determination in the answer is '{final_label}'. "
        f"This must be consistent with the rule: (experience ≥ 5 years) OR (valid 1-year waiver applied AND experience ≥ 4 years), "
        f"and experience must cover ≥ 2 domains. "
        f"The answer's totals are: total_experience_years='{extraction.total_experience_years}', waiver_applied={extraction.waiver_applied}, "
        f"waiver_years='{extraction.waiver_years}', domains={domains_list_display}. "
        f"Given the provided background (3 full-time years + 2080 part-time hours → 4.0 total years; 1-year education waiver applies; "
        f"coverage in Domain 4 and Domain 7), the correct outcome is eligibility via the waiver path."
    )
    await evaluator.verify(
        claim=eligibility_claim,
        node=eligibility_node,
        additional_instruction=(
            "Eligibility rule: Eligible if total qualifying experience ≥ 5 years; OR eligible if a valid 1-year waiver is applied AND total qualifying experience ≥ 4 years; "
            "in all cases experience must cover ≥ 2 CISSP domains. "
            "For this background, the correct path is: 4-year degree → 1-year waiver, total experience 4.0 years (3 full-time + 1.0 part-time credit), "
            "domains covered include Domain 4 and Domain 7 → Eligible."
        ),
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for CISSP eligibility and return a structured result dictionary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation; rubric's logical root is added as a child critical node
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

    # Extract structured analysis from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_cissp_analysis(),
        template_class=CISSPAnalysisExtraction,
        extraction_name="cissp_analysis_extraction",
    )

    # Add ground truth interpretation and rules overview
    evaluator.add_ground_truth({
        "expected_full_time_years": EXPECTED_FULL_TIME_YEARS,
        "expected_part_time_hours": EXPECTED_PART_TIME_HOURS,
        "expected_converted_part_time_years": EXPECTED_CONVERTED_PT_YEARS,
        "expected_total_years": EXPECTED_TOTAL_YEARS,
        "expected_waiver_applies": EXPECTED_WAIVER_APPLIES,
        "expected_domains": EXPECTED_DOMAINS,
        "expected_eligibility": EXPECTED_ELIGIBILITY,
    }, gt_type="expected_interpretation")

    evaluator.add_custom_info(RULES_SUMMARY, info_type="rules_summary")

    # Build and run verification tree
    await build_verification_tree(evaluator, extraction)

    # Return final evaluation summary
    return evaluator.get_summary()