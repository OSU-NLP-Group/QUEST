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
TASK_ID = "ph_mm_qc_property_tax_2026"
TASK_DESCRIPTION = """
A property investor in Metro Manila, Philippines, needs comprehensive guidance for 2026 regarding their residential properties in Quezon City. They need to optimize their 2026 real property tax payments and understand the tax implications of selling one property. Provide a detailed guide that includes: (1) Quezon City's early payment discount program for 2026 property taxes (including the discount percentage, payment deadline, and any eligibility requirements), (2) complete information about the Philippines real property tax amnesty program currently available (including the application deadline, which prior-year taxes are covered, and what relief is provided), (3) all applicable taxes for selling residential property in Metro Manila (identifying each tax type and stating the applicable rate for each), and (4) strategic recommendations for optimizing payment timing and utilizing available benefits.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class QCDiscountInfo(BaseModel):
    percentage: Optional[str] = None
    window_text: Optional[str] = None  # e.g., "Jan 1–Mar 31, 2026"
    deadline: Optional[str] = None     # e.g., "March 31, 2026"
    eligibility: Optional[str] = None  # e.g., "full payment of annual RPT required"
    sources: List[str] = Field(default_factory=list)


class MetroManilaRPTInfo(BaseModel):
    rate_ceiling: Optional[str] = None  # e.g., "2%"
    sources: List[str] = Field(default_factory=list)


class TaxComponent(BaseModel):
    rate_text: Optional[str] = None        # e.g., "6%", "1.5%", "0.75%"
    base_rule: Optional[str] = None        # e.g., "whichever is higher ..."
    sources: List[str] = Field(default_factory=list)


class SaleTaxesInfo(BaseModel):
    cgt: Optional[TaxComponent] = None
    dst: Optional[TaxComponent] = None
    local_transfer_tax: Optional[TaxComponent] = None


class AmnestyInfo(BaseModel):
    application_deadline: Optional[str] = None   # e.g., "July 5, 2026"
    coverage_period: Optional[str] = None        # e.g., "prior to July 5, 2024"
    relief: Optional[str] = None                 # e.g., "waiver of penalties, surcharges, and interest"
    sources: List[str] = Field(default_factory=list)


class StrategyInfo(BaseModel):
    discount_timing_strategy: Optional[str] = None
    amnesty_strategy: Optional[str] = None
    sale_tax_planning_strategy: Optional[str] = None


class GuideExtraction(BaseModel):
    qc_discount: Optional[QCDiscountInfo] = None
    mm_rpt_ceiling: Optional[MetroManilaRPTInfo] = None
    amnesty: Optional[AmnestyInfo] = None
    sale_taxes: Optional[SaleTaxesInfo] = None
    strategies: Optional[StrategyInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_guide() -> str:
    return """
    You will extract structured information from the answer regarding Quezon City (QC) 2026 real property taxes, the Philippines real property tax amnesty, applicable taxes on selling a residential property in Metro Manila, and strategic recommendations.

    Extract the following fields exactly as stated in the answer (do not invent):

    qc_discount:
      - percentage: the QC 2026 early payment discount percentage (include the % sign if present)
      - window_text: the described payment window (e.g., "Jan 1–Mar 31, 2026") if provided in the answer
      - deadline: the exact deadline date as stated (e.g., "March 31, 2026") if provided
      - eligibility: the key eligibility requirement text, typically mentioning "full payment" or equivalent
      - sources: list of all URLs cited in the answer that support the QC discount details

    mm_rpt_ceiling:
      - rate_ceiling: the maximum RPT rate mentioned for Metro Manila cities (e.g., "2%")
      - sources: list of URLs cited for this information

    amnesty:
      - application_deadline: the deadline date for applying to the real property tax amnesty (as stated)
      - coverage_period: the coverage statement for which delinquent taxes are covered (e.g., "prior to July 5, 2024")
      - relief: the relief provided (e.g., waiver/condonation of penalties, surcharges, and interest)
      - sources: list of URLs cited for these amnesty details

    sale_taxes:
      cgt:
        - rate_text: the stated capital gains tax rate (e.g., "6%")
        - base_rule: the base rule as stated (e.g., "of the gross selling price or current fair market value, whichever is higher")
        - sources: list of URLs cited for CGT
      dst:
        - rate_text: the stated documentary stamp tax rate (e.g., "1.5%")
        - base_rule: the base rule as stated (e.g., "of the selling price or zonal value, whichever is higher")
        - sources: list of URLs cited for DST
      local_transfer_tax:
        - rate_text: the stated local transfer tax rate for Metro Manila (e.g., "0.75%")
        - base_rule: the base rule as stated (e.g., "of the selling price, zonal value, or fair market value, whichever is higher")
        - sources: list of URLs cited for local transfer tax

    strategies:
      - discount_timing_strategy: a concise 1-3 sentence summary of the recommended timing strategy for availing the QC early-payment discount (or null if absent)
      - amnesty_strategy: a concise 1-3 sentence summary of how/when to use the RPT amnesty, ideally tying to the coverage period and deadline (or null if absent)
      - sale_tax_planning_strategy: a concise 1-3 sentence summary of sale planning guidance considering CGT, DST, and local transfer tax (or null if absent)

    Rules:
    - Extract only what appears in the answer. If a field is missing, return null (or [] for sources).
    - For URL fields, extract only valid URLs actually present in the answer (plain or markdown-formatted).
    """


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_qc_discount_section(evaluator: Evaluator, parent, data: GuideExtraction):
    node = evaluator.add_parallel(
        id="Quezon_City_Discount_Program",
        desc="QC 2026 RPT early payment discount program details.",
        parent=parent,
        critical=True
    )

    qc = data.qc_discount or QCDiscountInfo()

    # Gate: sources and field existence checks (critical siblings)
    evaluator.add_custom_node(
        result=bool(qc.sources),
        id="QC_Sources_Provided",
        desc="QC discount: Sources provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(qc.percentage and qc.percentage.strip()),
        id="QC_Percentage_Provided",
        desc="QC discount: Percentage stated in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool((qc.window_text and qc.window_text.strip()) or (qc.deadline and qc.deadline.strip())),
        id="QC_WindowOrDeadline_Provided",
        desc="QC discount: Payment window or deadline stated in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(qc.eligibility and qc.eligibility.strip()),
        id="QC_Eligibility_Provided",
        desc="QC discount: Eligibility requirement stated in the answer",
        parent=node,
        critical=True
    )

    # Leaf: QC_Discount_Percentage
    leaf_pct = evaluator.add_leaf(
        id="QC_Discount_Percentage",
        desc="States the QC 2026 RPT early-payment discount is 10%.",
        parent=node,
        critical=True
    )
    pct_text = qc.percentage or ""
    await evaluator.verify(
        claim=f"Quezon City's 2026 real property tax early-payment discount percentage is {pct_text}.",
        node=leaf_pct,
        sources=qc.sources,
        additional_instruction="Verify the stated percentage against the cited sources. Minor textual variants like 'ten percent' vs '10%' are acceptable as long as the numeric value matches."
    )

    # Leaf: QC_Discount_Payment_Window_or_Deadline
    leaf_deadline = evaluator.add_leaf(
        id="QC_Discount_Payment_Window_or_Deadline",
        desc="States the discount applies for full payment made between Jan 1, 2026 and Mar 31, 2026 (i.e., deadline Mar 31, 2026).",
        parent=node,
        critical=True
    )
    if qc.window_text and qc.window_text.strip():
        window_claim = f"The QC early-payment discount applies for full payment {qc.window_text}."
    elif qc.deadline and qc.deadline.strip():
        window_claim = f"The QC early-payment discount applies for full payment made on or before {qc.deadline}."
    else:
        window_claim = "The QC early-payment discount timing window or deadline is as stated."
    await evaluator.verify(
        claim=window_claim,
        node=leaf_deadline,
        sources=qc.sources,
        additional_instruction="Confirm the qualifying payment window for the early-payment discount. If the source expresses this as a deadline (e.g., on/before March 31, 2026) or a window (e.g., Jan 1–Mar 31, 2026), treat them as equivalent if consistent."
    )

    # Leaf: QC_Discount_Eligibility_Requirement
    leaf_elig = evaluator.add_leaf(
        id="QC_Discount_Eligibility_Requirement",
        desc="States the key eligibility requirement: full payment is required to qualify for the discount.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Full payment is required to qualify for the QC early-payment discount (i.e., full settlement of the annual RPT).",
        node=leaf_elig,
        sources=qc.sources,
        additional_instruction="Verify that the policy requires full payment of the annual real property tax to avail the discount. Accept minor wording variations such as 'full settlement' or 'payment in full'."
    )


async def build_mm_rpt_section(evaluator: Evaluator, parent, data: GuideExtraction):
    node = evaluator.add_parallel(
        id="Metro_Manila_RPT",
        desc="Metro Manila RPT ceiling verification group.",
        parent=parent,
        critical=True
    )

    mm = data.mm_rpt_ceiling or MetroManilaRPTInfo()

    # Gates
    evaluator.add_custom_node(
        result=bool(mm.sources),
        id="MM_RPT_Sources_Provided",
        desc="Metro Manila RPT ceiling: Sources provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(mm.rate_ceiling and mm.rate_ceiling.strip()),
        id="MM_RPT_Rate_Provided",
        desc="Metro Manila RPT ceiling: Rate stated in the answer",
        parent=node,
        critical=True
    )

    # Leaf: Metro_Manila_RPT_Rate_Ceiling
    leaf = evaluator.add_leaf(
        id="Metro_Manila_RPT_Rate_Ceiling",
        desc="States the maximum RPT rate for cities within Metro Manila is up to 2% of assessed property value.",
        parent=node,
        critical=True
    )
    rate_text = mm.rate_ceiling or ""
    await evaluator.verify(
        claim=f"The maximum real property tax rate for cities within Metro Manila is up to {rate_text} of assessed property value.",
        node=leaf,
        sources=mm.sources,
        additional_instruction="Check that the source states a rate not exceeding 2% for cities/municipalities in Metro Manila. Accept equivalent phrasing like 'not exceeding 2%'."
    )


async def build_amnesty_section(evaluator: Evaluator, parent, data: GuideExtraction):
    node = evaluator.add_parallel(
        id="Property_Tax_Amnesty_Program",
        desc="Details about the currently available Philippines real property tax amnesty program.",
        parent=parent,
        critical=True
    )

    am = data.amnesty or AmnestyInfo()

    # Gates
    evaluator.add_custom_node(
        result=bool(am.sources),
        id="Amnesty_Sources_Provided",
        desc="Amnesty: Sources provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(am.application_deadline and am.application_deadline.strip()),
        id="Amnesty_Deadline_Provided",
        desc="Amnesty: Application deadline stated in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(am.coverage_period and am.coverage_period.strip()),
        id="Amnesty_Coverage_Provided",
        desc="Amnesty: Coverage period stated in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(am.relief and am.relief.strip()),
        id="Amnesty_Relief_Provided",
        desc="Amnesty: Relief provided stated in the answer",
        parent=node,
        critical=True
    )

    # Leaves
    leaf_deadline = evaluator.add_leaf(
        id="Amnesty_Application_Deadline",
        desc="States the amnesty application deadline is July 5, 2026.",
        parent=node,
        critical=True
    )
    deadline_text = am.application_deadline or ""
    await evaluator.verify(
        claim=f"The real property tax amnesty application deadline is {deadline_text}.",
        node=leaf_deadline,
        sources=am.sources,
        additional_instruction="Verify the amnesty application deadline date in the source."
    )

    leaf_coverage = evaluator.add_leaf(
        id="Amnesty_Coverage_Period",
        desc="States the amnesty covers unpaid real property taxes incurred prior to July 5, 2024.",
        parent=node,
        critical=True
    )
    coverage_text = am.coverage_period or ""
    await evaluator.verify(
        claim=f"The amnesty covers unpaid real property taxes incurred {coverage_text}.",
        node=leaf_coverage,
        sources=am.sources,
        additional_instruction="Confirm the coverage period (e.g., prior to a specified date) matches the source."
    )

    leaf_relief = evaluator.add_leaf(
        id="Amnesty_Relief_Provided",
        desc="States the amnesty relief is the waiver of penalties, surcharges, and interest on covered delinquent real property taxes.",
        parent=node,
        critical=True
    )
    # Use a canonical phrasing for relief; sources should confirm this
    await evaluator.verify(
        claim="The amnesty relief waives penalties, surcharges, and interest on covered delinquent real property taxes.",
        node=leaf_relief,
        sources=am.sources,
        additional_instruction="Allow equivalent terms such as 'condonation' for waiver, and verify it covers penalties, surcharges, and interest."
    )


async def build_sale_taxes_section(evaluator: Evaluator, parent, data: GuideExtraction):
    node = evaluator.add_parallel(
        id="Property_Sale_Taxes",
        desc="Lists applicable taxes for selling residential property in Metro Manila and states the rate (and base rule where specified) for each.",
        parent=parent,
        critical=True
    )

    taxes = data.sale_taxes or SaleTaxesInfo()

    # CGT
    cgt = taxes.cgt or TaxComponent()
    evaluator.add_custom_node(
        result=bool(cgt.sources),
        id="CGT_Sources_Provided",
        desc="CGT: Sources provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(cgt.rate_text and cgt.rate_text.strip()),
        id="CGT_Info_Provided",
        desc="CGT: Rate stated in the answer",
        parent=node,
        critical=True
    )
    leaf_cgt = evaluator.add_leaf(
        id="Capital_Gains_Tax",
        desc="Identifies CGT and states it is 6% of the gross selling price or current fair market value, whichever is higher (for residential property classified as a capital asset).",
        parent=node,
        critical=True
    )
    cgt_rate = cgt.rate_text or ""
    await evaluator.verify(
        claim=f"For residential property classified as a capital asset, the capital gains tax is {cgt_rate} of the gross selling price or the current fair market value, whichever is higher.",
        node=leaf_cgt,
        sources=cgt.sources,
        additional_instruction="Confirm the standard Philippine CGT rule for capital assets: a single rate applied to the higher of gross selling price or current FMV."
    )

    # DST
    dst = taxes.dst or TaxComponent()
    evaluator.add_custom_node(
        result=bool(dst.sources),
        id="DST_Sources_Provided",
        desc="DST: Sources provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(dst.rate_text and dst.rate_text.strip()),
        id="DST_Info_Provided",
        desc="DST: Rate stated in the answer",
        parent=node,
        critical=True
    )
    leaf_dst = evaluator.add_leaf(
        id="Documentary_Stamp_Tax",
        desc="Identifies DST and states it is 1.5% of the selling price or zonal value, whichever is higher.",
        parent=node,
        critical=True
    )
    dst_rate = dst.rate_text or ""
    await evaluator.verify(
        claim=f"The documentary stamp tax on the sale of real property is {dst_rate} of the selling price or zonal value, whichever is higher.",
        node=leaf_dst,
        sources=dst.sources,
        additional_instruction="Verify the DST rate basis for deeds of sale of real property. Accept equivalent expressions indicating 1.5% or P15 per P1,000."
    )

    # Local Transfer Tax
    ltt = taxes.local_transfer_tax or TaxComponent()
    evaluator.add_custom_node(
        result=bool(ltt.sources),
        id="LTT_Sources_Provided",
        desc="Local transfer tax: Sources provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ltt.rate_text and ltt.rate_text.strip()),
        id="LTT_Info_Provided",
        desc="Local transfer tax: Rate stated in the answer",
        parent=node,
        critical=True
    )
    leaf_ltt = evaluator.add_leaf(
        id="Local_Transfer_Tax",
        desc="Identifies local transfer tax and states the Metro Manila rate is 0.75% of the property's selling price, zonal value, or fair market value, whichever is higher.",
        parent=node,
        critical=True
    )
    ltt_rate = ltt.rate_text or ""
    await evaluator.verify(
        claim=f"In Metro Manila, the local transfer tax rate is {ltt_rate} of the property's selling price, zonal value, or fair market value, whichever is higher.",
        node=leaf_ltt,
        sources=ltt.sources,
        additional_instruction="Confirm that Metro Manila LGUs commonly impose a 0.75% transfer tax rate and that the tax base uses the highest of the stated valuation bases."
    )


async def build_strategy_recommendations_section(evaluator: Evaluator, parent, data: GuideExtraction):
    # Adjusted to non-critical parent to allow partial credit and a mix of critical/non-critical children
    node = evaluator.add_parallel(
        id="Strategic_Recommendations",
        desc="Actionable recommendations to optimize payment timing and utilize available benefits for 2026 and for a planned sale.",
        parent=parent,
        critical=False
    )

    strategies = data.strategies or StrategyInfo()

    # Discount timing strategy (critical under strategies)
    leaf_disc = evaluator.add_leaf(
        id="Discount_Program_Timing_Strategy",
        desc="Provides a timing strategy that leverages the QC early-payment discount program (i.e., advises action that would qualify for the discount based on the stated eligibility window/requirements).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer provides a timing strategy that leverages the QC early-payment discount by advising full payment within the qualifying window (e.g., by March 31, 2026) to obtain the discount.",
        node=leaf_disc,
        additional_instruction="Judge solely based on the answer text: it should explicitly or implicitly advise full payment within the Q1 2026 window to qualify for the discount. Minor wording differences are acceptable."
    )

    # Amnesty utilization strategy (critical under strategies)
    leaf_amnesty = evaluator.add_leaf(
        id="Amnesty_Program_Utilization_Strategy",
        desc="Provides guidance on when/how to use the RPT amnesty if applicable, including a timing consideration tied to the stated amnesty deadline and coverage period (without requiring a specific phrasing).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer provides guidance on how/when to use the RPT amnesty, referencing timing tied to the application deadline and the coverage period.",
        node=leaf_amnesty,
        additional_instruction="Judge solely based on the answer text: it should indicate when to apply (before the deadline) and whether the delinquency falls within the covered period. Accept concise, actionable guidance."
    )

    # Sale tax planning strategy (non-critical)
    leaf_sale = evaluator.add_leaf(
        id="Sale_Tax_Planning_Strategy",
        desc="Provides sale planning guidance that takes into account the listed sale-related taxes (e.g., budgeting/estimating net proceeds or cash-flow/timing considerations).",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The answer provides sale planning guidance that accounts for CGT, DST, and local transfer tax (e.g., budgeting for these taxes or estimating net proceeds).",
        node=leaf_sale,
        additional_instruction="Judge solely based on the answer text: it should acknowledge multiple tax items and provide a planning tip such as budgeting, cash flow timing, or net proceeds estimation."
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
    Evaluate an answer for the Metro Manila / QC 2026 property tax optimization task.
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
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_guide(),
        template_class=GuideExtraction,
        extraction_name="guide_extraction"
    )

    # Optional: add expected reference info to summary (non-binding ground truth hints)
    evaluator.add_ground_truth({
        "expected_qc_discount_highlights": {
            "percentage": "10%",
            "window_or_deadline": "Full payment by March 31, 2026 (Q1 window Jan 1–Mar 31, 2026)",
            "eligibility": "Full payment of annual RPT required"
        },
        "expected_mm_rpt_rate_ceiling": "Up to 2% of assessed value",
        "expected_amnesty": {
            "application_deadline": "July 5, 2026",
            "coverage_period": "Delinquencies incurred prior to July 5, 2024",
            "relief": "Waiver/condonation of penalties, surcharges, and interest"
        },
        "expected_sale_taxes": {
            "cgt": "6% of higher of gross selling price or current FMV (capital asset)",
            "dst": "1.5% of higher of selling price or zonal value",
            "local_transfer_tax_mm": "0.75% of higher of selling price, zonal value, or FMV"
        }
    })

    # Build top-level guide node (non-critical to allow partial credit, while sub-sections can be critical)
    guide_node = evaluator.add_parallel(
        id="Property_Tax_Optimization_Guide",
        desc="Guide covering: QC 2026 early-payment discount details, Philippines RPT amnesty details, taxes on sale of residential property in Metro Manila (types + rates), and timing/benefit optimization recommendations.",
        parent=root,
        critical=False
    )

    # Build sections
    await build_qc_discount_section(evaluator, guide_node, extracted)
    await build_mm_rpt_section(evaluator, guide_node, extracted)
    await build_amnesty_section(evaluator, guide_node, extracted)
    await build_sale_taxes_section(evaluator, guide_node, extracted)
    await build_strategy_recommendations_section(evaluator, guide_node, extracted)

    # Return structured evaluation result
    return evaluator.get_summary()