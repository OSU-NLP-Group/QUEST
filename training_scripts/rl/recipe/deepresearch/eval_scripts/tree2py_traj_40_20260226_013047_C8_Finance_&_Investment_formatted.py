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
TASK_ID = "ira_planning_2026"
TASK_DESCRIPTION = """For the 2026 tax year, provide a comprehensive IRA planning guide that addresses the following components:

1. Traditional IRA Contributions: What are the standard contribution limits for individuals under age 50 and the catch-up contribution limits for those age 50 or older? What is the deadline for making 2026 contributions? What are the Modified Adjusted Gross Income (MAGI) phase-out ranges for deducting Traditional IRA contributions for: (a) single filers covered by a workplace retirement plan, (b) married filing jointly where the contributor is covered by a workplace plan, and (c) married filing jointly where the contributor is not covered but the spouse is covered?

2. Roth IRA Contributions: What are the contribution limits for 2026? What are the MAGI thresholds and phase-out ranges for eligibility to make full Roth IRA contributions for: (a) single filers and (b) married filing jointly? Are there income limits for Roth IRA conversions?

3. Required Minimum Distributions and Withdrawals: At what age must individuals begin taking Required Minimum Distributions (RMDs) from Traditional IRAs in 2026? What is the early withdrawal penalty for distributions taken before age 59½? At what age can individuals make Qualified Charitable Distributions (QCDs), and what is the annual QCD limit for 2026? Provide at least three exceptions to the early withdrawal penalty.

4. Rollover and Transfer Rules: What is the time limit for completing an indirect IRA rollover to avoid taxes and penalties? How many IRA-to-IRA rollovers are permitted per 12-month period? Are direct trustee-to-trustee transfers subject to this frequency limitation?

5. Spousal IRA Provisions: Can a non-working spouse contribute to an IRA, and if so, what are the contribution limits for 2026? What is the maximum combined IRA contribution amount for a married couple both under age 50 in 2026?

6. Excess Contribution Penalties: What is the annual penalty rate for excess IRA contributions that remain in the account? By what deadline must excess contributions be withdrawn to avoid this penalty?

For each component, provide authoritative reference URLs from the IRS or reputable financial institutions to support your answers.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TraditionalIRAInfo(BaseModel):
    standard_limit_under50: Optional[str] = None
    catch_up_amount_50plus: Optional[str] = None
    total_limit_50plus: Optional[str] = None
    contribution_deadline_date: Optional[str] = None
    deduction_phaseout_single_range: Optional[str] = None
    deduction_phaseout_mfj_covered_range: Optional[str] = None
    deduction_phaseout_mfj_spouse_range: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class RothIRAInfo(BaseModel):
    contribution_limit_under50: Optional[str] = None
    catch_up_amount_50plus: Optional[str] = None
    single_full_eligibility_threshold: Optional[str] = None
    single_phaseout_range: Optional[str] = None
    mfj_full_eligibility_threshold: Optional[str] = None
    mfj_phaseout_range: Optional[str] = None
    conversion_income_limit_statement: Optional[str] = None
    conversion_tax_statement: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class DistributionInfo(BaseModel):
    rmd_rule_summary_2026: Optional[str] = None
    early_withdrawal_penalty_statement: Optional[str] = None
    qcd_eligibility_age: Optional[str] = None
    qcd_annual_limit_2026: Optional[str] = None
    early_withdrawal_exceptions: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class RolloverTransferInfo(BaseModel):
    indirect_rollover_time_limit: Optional[str] = None
    rollover_frequency_limit: Optional[str] = None
    direct_transfer_frequency_rule: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class SpousalIRAInfo(BaseModel):
    spousal_eligibility_rule: Optional[str] = None
    spousal_contribution_limit_under50: Optional[str] = None
    spousal_catchup_50plus: Optional[str] = None
    combined_contribution_max_under50: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class ExcessContributionInfo(BaseModel):
    excess_penalty_rate: Optional[str] = None
    correction_deadline_rule: Optional[str] = None
    correction_method_rule: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class IRAPlanExtraction(BaseModel):
    traditional: Optional[TraditionalIRAInfo] = None
    roth: Optional[RothIRAInfo] = None
    distribution: Optional[DistributionInfo] = None
    rollover_transfer: Optional[RolloverTransferInfo] = None
    spousal: Optional[SpousalIRAInfo] = None
    excess: Optional[ExcessContributionInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ira_plan() -> str:
    return """
Extract, from the provided answer, the exact 2026 IRA facts and the URLs cited for each section. Follow these rules strictly:
- Do not invent or infer values. Use exactly what is written in the answer.
- Keep amounts, ranges, dates, and ages as strings exactly as stated (preserve $ signs, commas, and punctuation).
- Extract only explicit URLs shown in the answer text (including markdown links); do not infer domains.

Return a single JSON object with the following structure (set any missing field to null or []):
{
  "traditional": {
    "standard_limit_under50": string|null,
    "catch_up_amount_50plus": string|null,
    "total_limit_50plus": string|null,
    "contribution_deadline_date": string|null,
    "deduction_phaseout_single_range": string|null,
    "deduction_phaseout_mfj_covered_range": string|null,
    "deduction_phaseout_mfj_spouse_range": string|null,
    "source_urls": [string, ...]
  },
  "roth": {
    "contribution_limit_under50": string|null,
    "catch_up_amount_50plus": string|null,
    "single_full_eligibility_threshold": string|null,
    "single_phaseout_range": string|null,
    "mfj_full_eligibility_threshold": string|null,
    "mfj_phaseout_range": string|null,
    "conversion_income_limit_statement": string|null,
    "conversion_tax_statement": string|null,
    "source_urls": [string, ...]
  },
  "distribution": {
    "rmd_rule_summary_2026": string|null,
    "early_withdrawal_penalty_statement": string|null,
    "qcd_eligibility_age": string|null,
    "qcd_annual_limit_2026": string|null,
    "early_withdrawal_exceptions": [string, ...],
    "source_urls": [string, ...]
  },
  "rollover_transfer": {
    "indirect_rollover_time_limit": string|null,
    "rollover_frequency_limit": string|null,
    "direct_transfer_frequency_rule": string|null,
    "source_urls": [string, ...]
  },
  "spousal": {
    "spousal_eligibility_rule": string|null,
    "spousal_contribution_limit_under50": string|null,
    "spousal_catchup_50plus": string|null,
    "combined_contribution_max_under50": string|null,
    "source_urls": [string, ...]
  },
  "excess": {
    "excess_penalty_rate": string|null,
    "correction_deadline_rule": string|null,
    "correction_method_rule": string|null,
    "source_urls": [string, ...]
  }
}
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _src(lst: Optional[List[str]]) -> List[str]:
    return lst or []


def _first_n(items: List[str], n: int) -> List[str]:
    return items[:n] if items else []


# --------------------------------------------------------------------------- #
# Section verifications                                                       #
# --------------------------------------------------------------------------- #
async def verify_traditional(evaluator: Evaluator, parent_node, info: TraditionalIRAInfo) -> None:
    section = evaluator.add_parallel(
        id="Traditional_IRA_Analysis",
        desc="Verify Traditional IRA contribution eligibility, limits, and deduction calculations for 2026",
        parent=parent_node,
        critical=True,
    )

    sources = _src(info.source_urls)

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Traditional_IRA_Sources_Provided",
        desc="At least one authoritative URL is provided for Traditional IRA section",
        parent=section,
        critical=True
    )

    # Contribution limit (under 50)
    node = evaluator.add_leaf(
        id="Contribution_Limit_Standard",
        desc="Confirm the standard Traditional IRA contribution limit of $7,500 for individuals under age 50 in 2026",
        parent=section,
        critical=True
    )
    claim = "For tax year 2026, the standard Traditional IRA contribution limit for individuals under age 50 is $7,500."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Verify this exact 2026 limit on at least one provided authoritative URL (IRS or major financial institution)."
    )

    # Catch-up (50+)
    node = evaluator.add_leaf(
        id="Contribution_Limit_Catchup",
        desc="Confirm the catch-up contribution amount of additional $1,100 (total $8,600) for individuals age 50 or older in 2026",
        parent=section,
        critical=True
    )
    claim = "For tax year 2026, the IRA catch-up contribution for individuals age 50 or older is an additional $1,100, for a total limit of $8,600."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm both parts: the catch-up amount ($1,100) and the total limit ($8,600) for 2026."
    )

    # Deadline
    node = evaluator.add_leaf(
        id="Contribution_Deadline",
        desc="Verify that contributions for tax year 2026 can be made until April 15, 2027",
        parent=section,
        critical=True
    )
    claim = "Contributions for the 2026 tax year can be made up to April 15, 2027."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Verify the IRA contribution deadline for tax year 2026 specifically."
    )

    # Deduction phase-outs
    node = evaluator.add_leaf(
        id="Deduction_Phaseout_Single",
        desc="For single filers covered by workplace plan, verify deduction phase-out range of $81,000-$91,000 MAGI",
        parent=section,
        critical=True
    )
    claim = "In 2026, for single filers covered by a workplace retirement plan, the Traditional IRA deduction phase-out range is $81,000 to $91,000 of MAGI."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm the 2026 MAGI phase-out range for single filers covered by a workplace plan."
    )

    node = evaluator.add_leaf(
        id="Deduction_Phaseout_MFJ_Covered",
        desc="For married filing jointly where contributor is covered by workplace plan, verify deduction phase-out range of $129,000-$149,000 MAGI",
        parent=section,
        critical=True
    )
    claim = "In 2026, for married filing jointly where the contributor is covered by a workplace plan, the Traditional IRA deduction phase-out range is $129,000 to $149,000 of MAGI."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm the 2026 MFJ covered contributor deduction MAGI phase-out range."
    )

    node = evaluator.add_leaf(
        id="Deduction_Phaseout_MFJ_Spouse",
        desc="For married filing jointly where contributor is not covered but spouse is, verify deduction phase-out range of $242,000-$252,000 MAGI",
        parent=section,
        critical=True
    )
    claim = "In 2026, for married filing jointly where the contributor is not covered by a workplace plan but the spouse is, the Traditional IRA deduction phase-out range is $242,000 to $252,000 of MAGI."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm the 2026 MFJ 'spousal coverage' deduction MAGI phase-out range."
    )

    # Reference
    node = evaluator.add_leaf(
        id="Traditional_IRA_Reference",
        desc="Provide authoritative IRS or financial institution URL confirming Traditional IRA rules for 2026",
        parent=section,
        critical=True
    )
    claim = "This source is an official IRS page or a reputable financial institution and provides authoritative information about 2026 Traditional IRA rules."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Treat domains like irs.gov, fidelity.com, vanguard.com, schwab.com, troweprice.com as reputable. Verify the page discusses 2026 Traditional IRA rules."
    )


async def verify_roth(evaluator: Evaluator, parent_node, info: RothIRAInfo) -> None:
    section = evaluator.add_parallel(
        id="Roth_IRA_Analysis",
        desc="Verify Roth IRA contribution eligibility, income limits, and phase-out ranges for 2026",
        parent=parent_node,
        critical=True,
    )

    sources = _src(info.source_urls)

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Roth_IRA_Sources_Provided",
        desc="At least one authoritative URL is provided for Roth IRA section",
        parent=section,
        critical=True
    )

    node = evaluator.add_leaf(
        id="Roth_Contribution_Limit",
        desc="Confirm Roth IRA contribution limit matches Traditional IRA limit of $7,500 ($8,600 age 50+) for 2026",
        parent=section,
        critical=True
    )
    claim = "For 2026, the Roth IRA contribution limit matches the Traditional IRA limit: $7,500 (or $8,600 if age 50 or older)."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Verify explicit 2026 Roth IRA limits, including the age 50+ total."
    )

    node = evaluator.add_leaf(
        id="Roth_Income_Limit_Single",
        desc="For single filers, verify full contribution eligibility requires MAGI below $153,000 in 2026",
        parent=section,
        critical=True
    )
    claim = "In 2026, single filers have full Roth IRA contribution eligibility if MAGI is below $153,000."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm the full eligibility MAGI threshold for single filers in 2026."
    )

    node = evaluator.add_leaf(
        id="Roth_Phaseout_Single",
        desc="For single filers, verify contribution phase-out range of $153,000-$168,000 MAGI in 2026",
        parent=section,
        critical=True
    )
    claim = "In 2026, the Roth IRA contribution phase-out range for single filers is $153,000 to $168,000 of MAGI."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm the 2026 single filer Roth phase-out range."
    )

    node = evaluator.add_leaf(
        id="Roth_Income_Limit_MFJ",
        desc="For married filing jointly, verify full contribution eligibility requires MAGI below $242,000 in 2026",
        parent=section,
        critical=True
    )
    claim = "In 2026, married filing jointly have full Roth IRA contribution eligibility if MAGI is below $242,000."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm the 2026 MFJ full eligibility MAGI threshold."
    )

    node = evaluator.add_leaf(
        id="Roth_Phaseout_MFJ",
        desc="For married filing jointly, verify contribution phase-out range of $242,000-$252,000 MAGI in 2026",
        parent=section,
        critical=True
    )
    claim = "In 2026, the Roth IRA contribution phase-out range for married filing jointly is $242,000 to $252,000 of MAGI."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm the 2026 MFJ Roth phase-out range."
    )

    node = evaluator.add_leaf(
        id="Roth_Conversion_Rules",
        desc="Verify that Roth conversions have no income limits but require paying taxes on converted amounts",
        parent=section,
        critical=True
    )
    claim = "Roth IRA conversions have no income limits, but converted amounts are taxable as ordinary income."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Verify no income cap exists for conversions and that taxes are owed on pre-tax amounts converted."
    )

    node = evaluator.add_leaf(
        id="Roth_IRA_Reference",
        desc="Provide authoritative IRS or financial institution URL confirming Roth IRA rules for 2026",
        parent=section,
        critical=True
    )
    claim = "This source is an official IRS page or a reputable financial institution and provides authoritative information about 2026 Roth IRA rules."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Treat domains like irs.gov, fidelity.com, vanguard.com, schwab.com, troweprice.com as reputable. Verify the page discusses 2026 Roth IRA rules."
    )


async def verify_distribution(evaluator: Evaluator, parent_node, info: DistributionInfo) -> None:
    section = evaluator.add_parallel(
        id="Distribution_Requirements",
        desc="Verify withdrawal rules, RMD requirements, and early withdrawal penalties",
        parent=parent_node,
        critical=True,
    )

    sources = _src(info.source_urls)

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Distribution_Sources_Provided",
        desc="At least one authoritative URL is provided for Distribution/Withdrawal section",
        parent=section,
        critical=True
    )

    node = evaluator.add_leaf(
        id="RMD_Age_Requirement",
        desc="Verify that RMDs must begin at age 73 (or age 75 if born after 1959) in 2026",
        parent=section,
        critical=True
    )
    claim = "In 2026, Required Minimum Distributions (RMDs) must begin at age 73; for individuals born after 1959 (i.e., in 1960 or later), the RMD age is 75."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm the 2026 RMD start age and the 75 rule for those born after 1959."
    )

    node = evaluator.add_leaf(
        id="Early_Withdrawal_Penalty",
        desc="Confirm 10% early withdrawal penalty applies before age 59½ unless an exception applies",
        parent=section,
        critical=True
    )
    claim = "An additional 10% early withdrawal penalty applies to IRA distributions taken before age 59½ unless a qualifying exception applies."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Verify the 10% additional tax rule and note that exceptions can waive it."
    )

    node = evaluator.add_leaf(
        id="QCD_Eligibility_Age",
        desc="Verify Qualified Charitable Distributions (QCDs) are available starting at age 70½",
        parent=section,
        critical=True
    )
    claim = "Qualified Charitable Distributions (QCDs) from IRAs are permitted starting at age 70½."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm QCD start age."
    )

    node = evaluator.add_leaf(
        id="QCD_Annual_Limit",
        desc="Confirm the QCD annual limit is $111,000 for 2026",
        parent=section,
        critical=True
    )
    claim = "For 2026, the annual Qualified Charitable Distribution (QCD) limit is $111,000."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Verify the specific 2026 QCD dollar limit."
    )

    node = evaluator.add_leaf(
        id="Early_Withdrawal_Exceptions",
        desc="Identify at least three valid exceptions to the 10% early withdrawal penalty (e.g., first-time home purchase, qualified education expenses, unreimbursed medical expenses)",
        parent=section,
        critical=True
    )
    # Use up to first three items for the claim; if fewer than three, instruct failure
    exceptions_list = _first_n(info.early_withdrawal_exceptions, 3)
    listed = "; ".join(exceptions_list) if exceptions_list else ""
    claim = f"At least three valid exceptions to the IRA 10% early withdrawal penalty include: {listed}."
    add_ins = "Verify each listed item is a legitimate exception under IRS rules. If fewer than three valid exceptions are listed, judge the claim as Incorrect."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=add_ins
    )

    node = evaluator.add_leaf(
        id="Distribution_Reference",
        desc="Provide authoritative IRS URL confirming withdrawal and distribution rules",
        parent=section,
        critical=True
    )
    claim = "This source is an official IRS page and provides authoritative information about IRA withdrawals, RMDs, and early withdrawal rules."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Prefer irs.gov publications or pages that explicitly cover RMDs and early distribution additional taxes for IRAs."
    )


async def verify_rollover(evaluator: Evaluator, parent_node, info: RolloverTransferInfo) -> None:
    section = evaluator.add_parallel(
        id="Rollover_Transfer_Rules",
        desc="Verify IRA rollover and transfer regulations for 2026",
        parent=parent_node,
        critical=True,
    )

    sources = _src(info.source_urls)

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Rollover_Sources_Provided",
        desc="At least one authoritative URL is provided for Rollover/Transfer section",
        parent=section,
        critical=True
    )

    node = evaluator.add_leaf(
        id="Indirect_Rollover_Timeframe",
        desc="Verify indirect rollovers must be completed within 60 days to avoid taxes and penalties",
        parent=section,
        critical=True
    )
    claim = "Indirect IRA rollovers must be completed within 60 days to avoid taxes and penalties."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm the 60-day rule for IRA indirect rollovers."
    )

    node = evaluator.add_leaf(
        id="Rollover_Frequency_Limit",
        desc="Confirm only one IRA-to-IRA rollover is permitted per 12-month period",
        parent=section,
        critical=True
    )
    claim = "Only one IRA-to-IRA rollover is permitted in any 12-month period."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm the one-rollover-per-12-month period rule (not per calendar year)."
    )

    node = evaluator.add_leaf(
        id="Direct_Transfer_Unlimited",
        desc="Verify that direct trustee-to-trustee transfers are not subject to the one-rollover-per-year limitation",
        parent=section,
        critical=True
    )
    claim = "Direct trustee-to-trustee transfers are not subject to the one-rollover-per-year limitation."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Verify the exemption for direct trustee-to-trustee transfers."
    )

    node = evaluator.add_leaf(
        id="Rollover_Reference",
        desc="Provide authoritative IRS or financial institution URL confirming rollover rules",
        parent=section,
        critical=True
    )
    claim = "This source is an official IRS page or a reputable financial institution and confirms IRA rollover and transfer rules."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="IRS pages are preferred; major financial institutions are acceptable if they correctly cite IRS rules."
    )


async def verify_spousal(evaluator: Evaluator, parent_node, info: SpousalIRAInfo) -> None:
    section = evaluator.add_parallel(
        id="Spousal_IRA_Provisions",
        desc="Verify spousal IRA contribution rules for non-working or low-income spouses in 2026",
        parent=parent_node,
        critical=True,
    )

    sources = _src(info.source_urls)

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Spousal_Sources_Provided",
        desc="At least one authoritative URL is provided for Spousal IRA section",
        parent=section,
        critical=True
    )

    node = evaluator.add_leaf(
        id="Spousal_IRA_Eligibility",
        desc="Confirm that a non-working spouse can contribute to an IRA if the working spouse has sufficient taxable compensation",
        parent=section,
        critical=True
    )
    claim = "A non-working spouse can contribute to an IRA if the working spouse has sufficient taxable compensation."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Verify the spousal IRA eligibility rule (a.k.a. 'spousal IRA') per IRS rules."
    )

    node = evaluator.add_leaf(
        id="Spousal_Contribution_Limit",
        desc="Verify spousal IRA contribution limits are the same as individual limits: $7,500 ($8,600 age 50+) for 2026",
        parent=section,
        critical=True
    )
    claim = "For 2026, spousal IRA contribution limits equal the individual limits: $7,500, or $8,600 if age 50 or older."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm the 2026 spousal IRA limits match the general IRA limits."
    )

    node = evaluator.add_leaf(
        id="Combined_Contribution_Maximum",
        desc="Calculate the maximum combined IRA contributions for a married couple under age 50 in 2026 ($15,000 total)",
        parent=section,
        critical=True
    )
    claim = "For a married couple where both spouses are under age 50 in 2026, the maximum combined IRA contributions total $15,000."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm that two times the under-50 limit yields a $15,000 combined maximum for 2026."
    )

    node = evaluator.add_leaf(
        id="Spousal_IRA_Reference",
        desc="Provide authoritative source URL confirming spousal IRA rules",
        parent=section,
        critical=True
    )
    claim = "This source is an official IRS page or a reputable financial institution and confirms 2026 spousal IRA rules."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="IRS pages are preferred; major financial institutions acceptable if accurate."
    )


async def verify_excess(evaluator: Evaluator, parent_node, info: ExcessContributionInfo) -> None:
    section = evaluator.add_parallel(
        id="Excess_Contribution_Penalties",
        desc="Verify rules and penalties for excess IRA contributions in 2026",
        parent=parent_node,
        critical=True,
    )

    sources = _src(info.source_urls)

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Excess_Sources_Provided",
        desc="At least one authoritative URL is provided for Excess Contributions section",
        parent=section,
        critical=True
    )

    node = evaluator.add_leaf(
        id="Excess_Penalty_Rate",
        desc="Confirm excess contributions are subject to 6% penalty per year until corrected",
        parent=section,
        critical=True
    )
    claim = "Excess IRA contributions are subject to a 6% penalty per year for each year the excess remains in the account."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Verify the 6% per year excise tax rule on excess IRA contributions."
    )

    node = evaluator.add_leaf(
        id="Correction_Deadline",
        desc="Verify excess contributions must be withdrawn by the tax filing deadline (including extensions) to avoid penalties",
        parent=section,
        critical=True
    )
    claim = "Excess IRA contributions must be withdrawn by the tax filing deadline, including extensions, to avoid the 6% excess contribution penalty."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Confirm the correction deadline language per IRS rules."
    )

    node = evaluator.add_leaf(
        id="Excess_Correction_Method",
        desc="Confirm that withdrawing excess contributions plus earnings before the deadline avoids the 6% penalty",
        parent=section,
        critical=True
    )
    claim = "Withdrawing the excess IRA contributions plus earnings before the deadline avoids the 6% excess contribution penalty."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction="Verify that removal of excess plus earnings by the deadline avoids the 6% excise tax."
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

    # Wrap rubric root as a critical node under framework root
    ira_root = evaluator.add_parallel(
        id="IRA_Planning_Assessment_2026",
        desc="Comprehensive evaluation of IRA planning recommendations for a client scenario in 2026",
        parent=root,
        critical=True
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_ira_plan(),
        template_class=IRAPlanExtraction,
        extraction_name="ira_plan_extraction"
    )

    # Build and verify each section
    await verify_traditional(
        evaluator,
        ira_root,
        extracted.traditional or TraditionalIRAInfo()
    )
    await verify_roth(
        evaluator,
        ira_root,
        extracted.roth or RothIRAInfo()
    )
    await verify_distribution(
        evaluator,
        ira_root,
        extracted.distribution or DistributionInfo()
    )
    await verify_rollover(
        evaluator,
        ira_root,
        extracted.rollover_transfer or RolloverTransferInfo()
    )
    await verify_spousal(
        evaluator,
        ira_root,
        extracted.spousal or SpousalIRAInfo()
    )
    await verify_excess(
        evaluator,
        ira_root,
        extracted.excess or ExcessContributionInfo()
    )

    return evaluator.get_summary()