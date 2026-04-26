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
TASK_ID = "multi_state_real_estate_requirements"
TASK_DESCRIPTION = """A prospective real estate professional is comparing licensing requirements across different U.S. states to make an informed decision about where to pursue their career. They need to compile specific information for their comparison chart:

1. Among Texas, Florida, California, and New York, which state requires the MOST pre-licensing education hours for a real estate salesperson license, and how many hours are required?

2. Which U.S. state has the highest effective property tax rate, and what is that rate (as a percentage)?

3. In New York, what is the minimum number of years of experience as a licensed real estate salesperson required to qualify for a real estate broker license?

4. How many total hours of qualifying education are required to obtain a real estate broker license in New York?

5. In California, how many hours of continuing education are required for real estate license renewal, and what is the renewal period in years?

Provide all five pieces of information with supporting reference URLs.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PreLicensingMost(BaseModel):
    state: Optional[str] = None
    hours: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PropertyTaxTop(BaseModel):
    state: Optional[str] = None
    rate_percent: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NYBrokerExperience(BaseModel):
    years: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NYBrokerEducation(BaseModel):
    hours: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CACERenewal(BaseModel):
    hours: Optional[str] = None
    renewal_period_years: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RealEstateExtraction(BaseModel):
    prelicensing_most: Optional[PreLicensingMost] = None
    property_tax_top: Optional[PropertyTaxTop] = None
    ny_broker_experience: Optional[NYBrokerExperience] = None
    ny_broker_education: Optional[NYBrokerEducation] = None
    ca_ce_renewal: Optional[CACERenewal] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_real_estate() -> str:
    return """
Extract the five requested items from the answer. For each item, capture the exact values as written in the answer and all explicitly cited reference URLs for that item.

Return a JSON object with the following structure:

{
  "prelicensing_most": {
    "state": string or null,
    "hours": string or null,
    "sources": [list of URLs explicitly cited for this item]
  },
  "property_tax_top": {
    "state": string or null,
    "rate_percent": string or null,  // keep as it appears (e.g., "2.23%" or "2.23 percent")
    "sources": [URLs]
  },
  "ny_broker_experience": {
    "years": string or null,         // keep wording like "2 years" or "two years"
    "sources": [URLs]
  },
  "ny_broker_education": {
    "hours": string or null,         // keep wording as written, e.g., "152 hours"
    "sources": [URLs]
  },
  "ca_ce_renewal": {
    "hours": string or null,               // e.g., "45 hours"
    "renewal_period_years": string or null // e.g., "4 years" or "four years"
    "sources": [URLs]
  }
}

Special instructions:
- Only extract URLs that are explicitly present in the answer text (including in markdown links). Do not invent URLs.
- If a field is missing in the answer, return null for that field.
- If no URLs are provided for an item, return an empty list for that item's "sources".
- Do not normalize numbers; keep them as written in the answer (e.g., include symbols like "%" and words like "years" if present).
- For item 1 (pre-licensing hours), the state must be selected among: Texas, Florida, California, New York, as stated by the answer.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_prelicensing_most(
    evaluator: Evaluator,
    parent_node,
    data: Optional[PreLicensingMost]
) -> None:
    """
    Verify: Among Texas, Florida, California, and New York, which state requires the MOST pre-licensing hours
    and the stated number of hours, with support from cited URLs.
    """
    node = evaluator.add_sequential(
        id="Highest_Prelicensing_Hours_State",
        desc="Correctly identify which state among Texas, Florida, California, and New York requires the MOST pre-licensing education hours for a real estate salesperson license, and provide the exact number of hours",
        parent=parent_node,
        critical=False
    )

    exists = evaluator.add_custom_node(
        result=(
            data is not None and
            data.state is not None and data.state.strip() != "" and
            data.hours is not None and data.hours.strip() != "" and
            isinstance(data.sources, list) and len(data.sources) > 0
        ),
        id="Highest_Prelicensing_Hours_State_exists",
        desc="Presence check: state, hours, and supporting URLs are provided for 'highest pre-licensing hours'",
        parent=node,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id="Highest_Prelicensing_Hours_State_supported",
        desc="The identified state and hours for the 'most pre-licensing education hours' claim are correct and supported by the cited sources",
        parent=node,
        critical=True
    )

    state = data.state if data and data.state else ""
    hours = data.hours if data and data.hours else ""
    claim = (
        f"Among Texas, Florida, California, and New York, the state with the highest required pre-licensing education hours "
        f"for a real estate salesperson license is {state}, requiring {hours}."
    )
    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=(data.sources if data else []),
        additional_instruction=(
            "Evaluate only the four states: Texas, Florida, California, and New York. "
            "Support can come from an official or authoritative comparison page that states the ranking explicitly, "
            "or a source that makes this clear. Minor phrasing or formatting differences (e.g., 'classroom hours', 'course hours') are acceptable. "
            "If the provided sources are individual state pages that do not support the 'most' claim explicitly, "
            "then treat the claim as not supported."
        )
    )


async def verify_property_tax_top(
    evaluator: Evaluator,
    parent_node,
    data: Optional[PropertyTaxTop]
) -> None:
    """
    Verify: Which U.S. state has the highest effective property tax rate and the stated rate, with support.
    """
    node = evaluator.add_sequential(
        id="Highest_Property_Tax_State",
        desc="Correctly identify which state has the highest effective property tax rate in the United States, and provide the rate as a percentage",
        parent=parent_node,
        critical=False
    )

    exists = evaluator.add_custom_node(
        result=(
            data is not None and
            data.state is not None and data.state.strip() != "" and
            data.rate_percent is not None and data.rate_percent.strip() != "" and
            isinstance(data.sources, list) and len(data.sources) > 0
        ),
        id="Highest_Property_Tax_State_exists",
        desc="Presence check: state, rate (percentage), and supporting URLs are provided for 'highest effective property tax rate'",
        parent=node,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id="Highest_Property_Tax_State_supported",
        desc="The identified state and rate for 'highest effective property tax rate' are correct and supported by the cited sources",
        parent=node,
        critical=True
    )

    state = data.state if data and data.state else ""
    rate = data.rate_percent if data and data.rate_percent else ""
    claim = f"The U.S. state with the highest effective property tax rate is {state}, with a rate of {rate}."
    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=(data.sources if data else []),
        additional_instruction=(
            "Confirm that the source explicitly discusses 'effective property tax rate' rankings by state. "
            "Allow minor rounding differences (e.g., 2.21% vs 2.2%). "
            "If the source does not clearly indicate the top state and its rate, treat as not supported."
        )
    )


async def verify_ny_broker_experience(
    evaluator: Evaluator,
    parent_node,
    data: Optional[NYBrokerExperience]
) -> None:
    """
    Verify: Minimum number of years as a licensed NY real estate salesperson to qualify for broker license.
    """
    node = evaluator.add_sequential(
        id="NY_Broker_Experience_Requirement",
        desc="Correctly state the minimum experience requirement (in years) to qualify for a real estate broker license in New York as a licensed salesperson",
        parent=parent_node,
        critical=False
    )

    exists = evaluator.add_custom_node(
        result=(
            data is not None and
            data.years is not None and data.years.strip() != "" and
            isinstance(data.sources, list) and len(data.sources) > 0
        ),
        id="NY_Broker_Experience_Requirement_exists",
        desc="Presence check: years and supporting URLs are provided for NY broker experience requirement",
        parent=node,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id="NY_Broker_Experience_Requirement_supported",
        desc="The minimum NY salesperson experience (years) to qualify for broker is correct and supported by cited sources",
        parent=node,
        critical=True
    )

    years = data.years if data and data.years else ""
    claim = (
        f"In New York State, the minimum number of years of experience as a licensed real estate salesperson to qualify for a real estate broker license is {years}."
    )
    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=(data.sources if data else []),
        additional_instruction=(
            "Prefer official New York Department of State (NYS DOS) or similarly authoritative pages. "
            "Treat wording variants like 'two years' and '2 years' as equivalent. "
            "If the source specifies alternative pathways, confirm the minimum salesperson-experience pathway stated in the answer."
        )
    )


async def verify_ny_broker_education(
    evaluator: Evaluator,
    parent_node,
    data: Optional[NYBrokerEducation]
) -> None:
    """
    Verify: Total qualifying education hours required for a NY real estate broker license.
    """
    node = evaluator.add_sequential(
        id="NY_Broker_Education_Requirement",
        desc="Correctly state the total number of qualifying education hours required for a real estate broker license in New York",
        parent=parent_node,
        critical=False
    )

    exists = evaluator.add_custom_node(
        result=(
            data is not None and
            data.hours is not None and data.hours.strip() != "" and
            isinstance(data.sources, list) and len(data.sources) > 0
        ),
        id="NY_Broker_Education_Requirement_exists",
        desc="Presence check: total broker qualifying education hours and supporting URLs are provided for New York",
        parent=node,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id="NY_Broker_Education_Requirement_supported",
        desc="The total qualifying education hours for a NY broker license are correct and supported by cited sources",
        parent=node,
        critical=True
    )

    hours = data.hours if data and data.hours else ""
    claim = f"New York requires {hours} of qualifying education to obtain a real estate broker license."
    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=(data.sources if data else []),
        additional_instruction=(
            "If the requirement is presented as a total that includes salesperson education, confirm that the total matches the stated number. "
            "Prefer official NYS DOS or other authoritative sources."
        )
    )


async def verify_ca_ce_renewal(
    evaluator: Evaluator,
    parent_node,
    data: Optional[CACERenewal]
) -> None:
    """
    Verify: California CE hours for license renewal and renewal period (years).
    """
    node = evaluator.add_sequential(
        id="CA_Continuing_Education_Requirement",
        desc="Correctly state the number of continuing education hours required for California real estate license renewal and the renewal period in years",
        parent=parent_node,
        critical=False
    )

    exists = evaluator.add_custom_node(
        result=(
            data is not None and
            data.hours is not None and data.hours.strip() != "" and
            data.renewal_period_years is not None and data.renewal_period_years.strip() != "" and
            isinstance(data.sources, list) and len(data.sources) > 0
        ),
        id="CA_Continuing_Education_Requirement_exists",
        desc="Presence check: CE hours, renewal period, and supporting URLs are provided for California license renewal",
        parent=node,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id="CA_Continuing_Education_Requirement_supported",
        desc="The California CE hours and renewal period are correct and supported by cited sources",
        parent=node,
        critical=True
    )

    hours = data.hours if data and data.hours else ""
    period = data.renewal_period_years if data and data.renewal_period_years else ""
    claim = (
        f"In California, real estate license renewal requires {hours} of continuing education and the renewal period is {period}."
    )
    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=(data.sources if data else []),
        additional_instruction=(
            "Prefer official California DRE sources. "
            "Treat 'every four years' and 'a 4-year renewal period' as equivalent. "
            "If the source distinguishes first-time vs subsequent renewals, ensure the claim matches the general rule stated in the answer."
        )
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
    Evaluate an answer for the multi-state real estate requirements comparison task.
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
        prompt=prompt_extract_real_estate(),
        template_class=RealEstateExtraction,
        extraction_name="real_estate_requirements_extraction"
    )

    # Build verification subtrees for each requirement
    await verify_prelicensing_most(evaluator, root, extracted.prelicensing_most)
    await verify_property_tax_top(evaluator, root, extracted.property_tax_top)
    await verify_ny_broker_experience(evaluator, root, extracted.ny_broker_experience)
    await verify_ny_broker_education(evaluator, root, extracted.ny_broker_education)
    await verify_ca_ce_renewal(evaluator, root, extracted.ca_ce_renewal)

    return evaluator.get_summary()