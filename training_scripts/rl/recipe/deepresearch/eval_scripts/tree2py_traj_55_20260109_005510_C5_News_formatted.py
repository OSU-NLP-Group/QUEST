import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "policy_jan_2025"
TASK_DESCRIPTION = """
In January 2025, multiple significant government policy announcements and updates were made in the United States. Provide detailed information about the following four items:

1. Federal Reserve Interest Rate Decision: On January 29, 2025, the Federal Open Market Committee (FOMC) made a decision regarding the federal funds rate. What specific action did the FOMC take regarding the target range, what is the current target range, and on what date was this decision announced?

2. California Statewide Minimum Wage: A new statewide minimum wage for all employers in California took effect on January 1, 2025. What is the new minimum wage rate per hour, and what was the previous rate in 2024?

3. California Healthcare Worker Minimum Wage: Under California Senate Bill 525, healthcare workers at the largest healthcare facilities have a higher minimum wage. For healthcare employers or integrated health systems with 10,000 or more employees, what is the minimum wage rate that applies during the period from October 16, 2024 to June 30, 2025, and when did these healthcare minimum wage increases first begin under SB 525?

4. California Fast Food Worker Minimum Wage: Fast food restaurant workers in California have a separate minimum wage. What is this minimum wage rate per hour, and when did it become effective?

For each item, provide the requested specific details along with reference URLs that verify the information.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FederalReserveInfo(BaseModel):
    action: Optional[str] = None
    target_range: Optional[str] = None
    announcement_date: Optional[str] = None
    vote_count: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CAStatewideMWInfo(BaseModel):
    new_rate_2025: Optional[str] = None
    previous_rate_2024: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CAHealthcareMWInfo(BaseModel):
    employer_category: Optional[str] = None
    rate_for_period: Optional[str] = None  # Rate for Oct 16, 2024 – Jun 30, 2025 for 10,000+ employer category
    sb525_start_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CAFastFoodMWInfo(BaseModel):
    rate: Optional[str] = None
    effective_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PolicyExtraction(BaseModel):
    federal_reserve: Optional[FederalReserveInfo] = None
    ca_statewide: Optional[CAStatewideMWInfo] = None
    ca_healthcare: Optional[CAHealthcareMWInfo] = None
    ca_fast_food: Optional[CAFastFoodMWInfo] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_policy_info() -> str:
    return """
    Extract the requested details for each of the four items from the provided answer. Return a single JSON with this schema:
    {
      "federal_reserve": {
        "action": string | null,                 // e.g., "maintained", "held steady", "raised", "cut", "lowered"
        "target_range": string | null,           // e.g., "5.25%–5.50%", "5.25% to 5.50%"
        "announcement_date": string | null,      // e.g., "January 29, 2025" (allow other formats, but extract exactly as stated in the answer)
        "vote_count": string | null,             // extract as string; if a number appears, keep it as a string, e.g., "12"
        "sources": string[]                      // reference URLs explicitly mentioned in the answer for this item
      },
      "ca_statewide": {
        "new_rate_2025": string | null,          // e.g., "$16.00 per hour"
        "previous_rate_2024": string | null,     // e.g., "$15.50 per hour"
        "sources": string[]
      },
      "ca_healthcare": {
        "employer_category": string | null,      // capture the category text mentioned (should be for 10,000+ employees systems)
        "rate_for_period": string | null,        // rate for period Oct 16, 2024 to Jun 30, 2025 for the 10,000+ category, as stated in the answer (e.g., "$23/hour")
        "sb525_start_date": string | null,       // e.g., "October 16, 2024"
        "sources": string[]
      },
      "ca_fast_food": {
        "rate": string | null,                   // e.g., "$20 per hour"
        "effective_date": string | null,         // e.g., "April 1, 2024" or other format as stated
        "sources": string[]
      }
    }

    IMPORTANT:
    - Extract ONLY what is explicitly present in the answer text. Do not infer or add missing data.
    - For all URL fields (the 'sources' lists), extract actual URLs explicitly present in the answer (including those in markdown link format).
    - If some requested detail is missing in the answer, set its value to null and still return the rest.
    - Preserve the exact textual formatting (e.g., percent signs, en-dashes/hyphens, currency symbols) as it appears in the answer.
    - For 'sources', deduplicate URLs and include only valid HTTP/HTTPS links.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_federal_reserve(
    evaluator: Evaluator,
    parent_node,
    fed: Optional[FederalReserveInfo],
) -> None:
    node = evaluator.add_parallel(
        id="federal_reserve_section",
        desc="Federal Reserve's January 2025 interest rate decision details",
        parent=parent_node,
        critical=False
    )

    sources = (fed.sources if fed else []) if fed else []

    # Reference URLs existence (critical)
    evaluator.add_custom_node(
        result=_has_sources(sources),
        id="federal_reserve_reference_urls",
        desc="Provide reference URL(s) verifying the Federal Reserve decision details",
        parent=node,
        critical=True
    )

    # Decision Action (critical)
    action_leaf = evaluator.add_leaf(
        id="federal_reserve_action",
        desc="State the specific action taken (maintain/hold vs. raise/cut) regarding the target range",
        parent=node,
        critical=True
    )
    action_val = fed.action if fed and fed.action else ""
    action_claim = f"The FOMC stated that it {action_val} the target range for the federal funds rate."
    await evaluator.verify(
        claim=action_claim,
        node=action_leaf,
        sources=sources,
        additional_instruction=(
            "Check the provided source(s) (ideally the FOMC statement/press release) to verify the described action. "
            "Treat 'maintained', 'kept', 'held steady', and 'left unchanged' as equivalent. "
            "Treat 'raised'/'increased'/'hiked' as equivalent, and 'cut'/'lowered' as equivalent. "
            "Confirm the action pertains specifically to the federal funds rate target range."
        )
    )

    # Target Range (critical)
    range_leaf = evaluator.add_leaf(
        id="federal_reserve_target_range",
        desc="Provide the target range that applies after the decision (as stated in constraints)",
        parent=node,
        critical=True
    )
    range_val = fed.target_range if fed and fed.target_range else ""
    range_claim = f"The target range for the federal funds rate is {range_val}."
    await evaluator.verify(
        claim=range_claim,
        node=range_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the source(s) explicitly state the target range after the decision. "
            "Allow minor formatting differences (e.g., hyphen vs. en-dash, with or without the word 'percent')."
        )
    )

    # Announcement Date (critical)
    date_leaf = evaluator.add_leaf(
        id="federal_reserve_announcement_date",
        desc="Provide the date the decision was announced (January 29, 2025)",
        parent=node,
        critical=True
    )
    date_val = fed.announcement_date if fed and fed.announcement_date else ""
    date_claim = f"The decision was announced on {date_val}."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the date on which the FOMC announcement/press release was issued. "
            "Allow common date format variants (e.g., 'Jan. 29, 2025' vs 'January 29, 2025')."
        )
    )

    # Vote Count (non-critical)
    vote_leaf = evaluator.add_leaf(
        id="federal_reserve_vote_count",
        desc="State the number of voting members for this decision (12)",
        parent=node,
        critical=False
    )
    vote_val = fed.vote_count if fed and fed.vote_count else ""
    vote_claim = f"There were {vote_val} voting members for this decision."
    await evaluator.verify(
        claim=vote_claim,
        node=vote_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the count of voting members as indicated in the statement. "
            "If the source lists names, you may count them to confirm the number."
        )
    )


async def verify_ca_statewide_min_wage(
    evaluator: Evaluator,
    parent_node,
    statewide: Optional[CAStatewideMWInfo],
) -> None:
    node = evaluator.add_parallel(
        id="ca_statewide_min_wage_section",
        desc="California statewide minimum wage effective January 1, 2025",
        parent=parent_node,
        critical=False
    )

    sources = (statewide.sources if statewide else []) if statewide else []

    # Reference URLs existence (critical)
    evaluator.add_custom_node(
        result=_has_sources(sources),
        id="ca_statewide_reference_urls",
        desc="Provide reference URL(s) verifying the statewide minimum wage information",
        parent=node,
        critical=True
    )

    # New 2025 Rate (critical)
    new_rate_leaf = evaluator.add_leaf(
        id="ca_statewide_new_rate_2025",
        desc="Provide the 2025 statewide minimum wage rate (as stated in constraints)",
        parent=node,
        critical=True
    )
    new_rate_val = statewide.new_rate_2025 if statewide and statewide.new_rate_2025 else ""
    new_rate_claim = (
        f"The California statewide minimum wage for all employers is {new_rate_val} per hour effective January 1, 2025."
    )
    await evaluator.verify(
        claim=new_rate_claim,
        node=new_rate_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the statewide (all employers) minimum wage rate effective on January 1, 2025. "
            "Ensure this is the general statewide rate, not a sector-specific rate."
        )
    )

    # Previous 2024 Rate (critical)
    prev_rate_leaf = evaluator.add_leaf(
        id="ca_statewide_previous_rate_2024",
        desc="Provide the 2024 statewide minimum wage rate (as stated in constraints)",
        parent=node,
        critical=True
    )
    prev_rate_val = statewide.previous_rate_2024 if statewide and statewide.previous_rate_2024 else ""
    prev_rate_claim = f"In 2024, the California statewide minimum wage was {prev_rate_val} per hour."
    await evaluator.verify(
        claim=prev_rate_claim,
        node=prev_rate_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the statewide minimum wage rate that applied during 2024 (the immediately prior year)."
        )
    )


async def verify_ca_healthcare_min_wage(
    evaluator: Evaluator,
    parent_node,
    hc: Optional[CAHealthcareMWInfo],
) -> None:
    node = evaluator.add_parallel(
        id="ca_healthcare_min_wage_section",
        desc="California healthcare worker minimum wage under SB 525 for the specified largest-employer category",
        parent=parent_node,
        critical=False
    )

    sources = (hc.sources if hc else []) if hc else []

    # Reference URLs existence (critical)
    evaluator.add_custom_node(
        result=_has_sources(sources),
        id="ca_healthcare_reference_urls",
        desc="Provide reference URL(s) verifying the SB 525 healthcare minimum wage details",
        parent=node,
        critical=True
    )

    # Employer Category (critical)
    category_leaf = evaluator.add_leaf(
        id="ca_healthcare_employer_category",
        desc="Answer is for healthcare employers or integrated health systems with 10,000+ employees",
        parent=node,
        critical=True
    )
    # We assert the required category explicitly per rubric
    cat_claim = (
        "This rate applies to healthcare employers or integrated healthcare systems with 10,000 or more employees."
    )
    await evaluator.verify(
        claim=cat_claim,
        node=category_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the cited category is specifically healthcare employers/integrated systems with 10,000+ employees "
            "as defined under SB 525. Allow minor wording variations (e.g., '10,000 or more')."
        )
    )

    # Rate for specified period (critical)
    rate_leaf = evaluator.add_leaf(
        id="ca_healthcare_rate_for_period",
        desc="Provide the minimum wage rate that applies for this category during Oct 16, 2024–Jun 30, 2025 (as stated in constraints)",
        parent=node,
        critical=True
    )
    rate_val = hc.rate_for_period if hc and hc.rate_for_period else ""
    rate_claim = (
        f"From October 16, 2024 through June 30, 2025, the minimum wage for healthcare employers or integrated "
        f"health systems with 10,000+ employees is {rate_val} per hour."
    )
    await evaluator.verify(
        claim=rate_claim,
        node=rate_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the stated rate corresponds to the specified timeframe (Oct 16, 2024 to Jun 30, 2025) "
            "and applies to the 10,000+ employee category under SB 525."
        )
    )

    # SB 525 increase start date (critical)
    start_leaf = evaluator.add_leaf(
        id="ca_healthcare_start_date",
        desc="Provide when the SB 525 healthcare minimum wage increases first began (October 16, 2024)",
        parent=node,
        critical=True
    )
    start_val = hc.sb525_start_date if hc and hc.sb525_start_date else ""
    start_claim = f"Under SB 525, the healthcare minimum wage increases first began on {start_val}."
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the initial start date for SB 525 healthcare minimum wage increases is October 16, 2024. "
            "Allow minor date format variations."
        )
    )


async def verify_ca_fast_food_min_wage(
    evaluator: Evaluator,
    parent_node,
    ff: Optional[CAFastFoodMWInfo],
) -> None:
    node = evaluator.add_parallel(
        id="ca_fast_food_min_wage_section",
        desc="California fast food restaurant worker minimum wage",
        parent=parent_node,
        critical=False
    )

    sources = (ff.sources if ff else []) if ff else []

    # Reference URLs existence (critical)
    evaluator.add_custom_node(
        result=_has_sources(sources),
        id="ca_fast_food_reference_urls",
        desc="Provide reference URL(s) verifying the fast food minimum wage information",
        parent=node,
        critical=True
    )

    # Rate (critical)
    rate_leaf = evaluator.add_leaf(
        id="ca_fast_food_rate",
        desc="Provide the fast food minimum wage rate (as stated in constraints)",
        parent=node,
        critical=True
    )
    rate_val = ff.rate if ff and ff.rate else ""
    rate_claim = f"The California minimum wage for fast food restaurant workers is {rate_val} per hour."
    await evaluator.verify(
        claim=rate_claim,
        node=rate_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the current minimum wage rate for fast food restaurant workers in California as specified in the source(s)."
        )
    )

    # Effective Date (critical)
    eff_leaf = evaluator.add_leaf(
        id="ca_fast_food_effective_date",
        desc="Provide when the fast food minimum wage became effective (as stated in constraints)",
        parent=node,
        critical=True
    )
    eff_val = ff.effective_date if ff and ff.effective_date else ""
    eff_claim = f"This fast food minimum wage became effective on {eff_val}."
    await evaluator.verify(
        claim=eff_claim,
        node=eff_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the effective date of the fast food minimum wage. Allow minor date format variations."
        )
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
) -> Dict:
    """
    Evaluate an answer for the January 2025 policy items task.
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

    # Extract structured information
    extracted: PolicyExtraction = await evaluator.extract(
        prompt=prompt_extract_policy_info(),
        template_class=PolicyExtraction,
        extraction_name="policy_items_extraction"
    )

    # Build and verify each section
    await verify_federal_reserve(evaluator, root, extracted.federal_reserve)
    await verify_ca_statewide_min_wage(evaluator, root, extracted.ca_statewide)
    await verify_ca_healthcare_min_wage(evaluator, root, extracted.ca_healthcare)
    await verify_ca_fast_food_min_wage(evaluator, root, extracted.ca_fast_food)

    # Return standardized summary
    return evaluator.get_summary()