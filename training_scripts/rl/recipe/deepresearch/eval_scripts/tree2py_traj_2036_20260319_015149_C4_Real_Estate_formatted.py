import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fl_q12024_sfr_yield_profile"
TASK_DESCRIPTION = (
    "According to ATTOM's Q1 2024 Single-Family Rental Market Report, which Florida county had the highest annual gross rental yield for three-bedroom properties? "
    "Provide a comprehensive investment profile for this county, including the specific rental yield percentage, population data, property tax information, demographic growth trends, "
    "location description within Florida, and authoritative source citations for your information."
)

GROUND_TRUTH = {
    "expected_county": "Indian River County",
    "expected_yield": "14.6%",
    "context": "Q1 2024 Single-Family (three-bedroom) annual gross rental yield in Florida per ATTOM"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CountyProfileExtraction(BaseModel):
    # Core identification
    county_name: Optional[str] = None
    rental_yield: Optional[str] = None

    # Sources for ATTOM and yield
    rental_yield_sources: List[str] = Field(default_factory=list)
    attom_report_urls: List[str] = Field(default_factory=list)

    # Investment profile fields + sources
    population: Optional[str] = None
    population_sources: List[str] = Field(default_factory=list)

    property_tax_info: Optional[str] = None
    property_tax_sources: List[str] = Field(default_factory=list)

    growth_trends: Optional[str] = None
    growth_sources: List[str] = Field(default_factory=list)

    location_context: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_county_profile() -> str:
    return """
    Your goal is to extract a structured investment profile for the Florida county that the answer claims has the highest annual gross rental yield for single-family three-bedroom properties in Q1 2024, per ATTOM.

    Extract exactly and only what is explicitly present in the provided answer. Do not infer or fabricate any information.

    Required fields to extract:
    1) county_name: The Florida county the answer identifies as having the highest Q1 2024 annual gross rental yield for three-bedroom SFR properties.
    2) rental_yield: The exact rental yield percentage stated for that county (keep the percent sign if present, e.g., "14.6%").
    3) rental_yield_sources: All URLs cited in the answer for the rental yield figure. Include any URLs that the answer associates with this yield figure, even if indirect.
    4) attom_report_urls: URLs in the answer that specifically point to ATTOM's Q1 2024 Single-Family Rental Market Report or an official ATTOM press release/page summarizing that report. Only include URLs explicitly present in the answer text. If none, return an empty list.

    5) population: Any population figure or demographic statistic mentioned for the identified county (e.g., population count with year). Return the exact text snippet as presented.
    6) population_sources: URLs supporting the population/demographic data cited.

    7) property_tax_info: Any property tax information included for the county (e.g., tax rate, effective rate, per capita taxes). Return the exact phrasing used.
    8) property_tax_sources: URLs supporting the tax information cited.

    9) growth_trends: Any growth/demographic/economic trend statements about the county (e.g., population growth rates, migration trends, job/economic development notes). Return the exact phrasing used.
    10) growth_sources: URLs supporting the growth/demographic/economic trend statements.

    11) location_context: A geographic description of where the county sits within Florida (e.g., along the Atlantic coast, near specific cities/regions). Return the exact phrasing used.
    12) location_sources: URLs supporting the geographic/location description.

    URL extraction rules:
    - Extract only URLs actually present in the answer (including markdown links). Do not invent or infer URLs.
    - Include full valid URLs. If protocol is missing, prepend http://.
    - If a category has no URLs, return an empty list.

    If a specific text field is not mentioned in the answer, set it to null. For URL lists, return empty arrays if none were provided.
    """


# --------------------------------------------------------------------------- #
# Utility                                                                     #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str):
                if u not in seen and u.strip():
                    merged.append(u.strip())
                    seen.add(u.strip())
    return merged


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root, extracted: CountyProfileExtraction) -> None:
    # Create a top-level node to mirror the rubric
    rubric_node = evaluator.add_parallel(
        id="Investment_Profile_Completeness",
        desc="Evaluates whether the answer provides a complete investment profile for the Florida county with the highest rental yield in Q1 2024",
        parent=root,
        critical=False
    )

    # 1) Correct County Identification (CRITICAL)
    county_node = evaluator.add_leaf(
        id="Correct_County_Identification",
        desc="Identifies Indian River County as the Florida county with the highest annual gross rental yield for single-family three-bedroom properties in Q1 2024",
        parent=rubric_node,
        critical=True
    )
    county_claim = f"The extracted county name '{extracted.county_name or ''}' refers to Indian River County, Florida (allow case-insensitive and minor formatting variations)."
    await evaluator.verify(
        claim=county_claim,
        node=county_node,
        additional_instruction="This is a simple name equivalence check. Consider minor spelling/casing differences equivalent if they clearly refer to Indian River County, Florida."
    )

    # 2) Rental Yield Accuracy (CRITICAL) – must be 14.6% and supported by ATTOM Q1 2024 or equivalent ATTOM page
    ry_node = evaluator.add_leaf(
        id="Rental_Yield_Accuracy",
        desc="Provides the correct rental yield percentage of 14.6% for three-bedroom properties in Q1 2024",
        parent=rubric_node,
        critical=True
    )
    yield_sources = _merge_urls(extracted.attom_report_urls, extracted.rental_yield_sources)
    ry_claim = "According to ATTOM's Q1 2024 Single-Family Rental Market Report, the annual gross rental yield for single-family three-bedroom properties in Indian River County, Florida is 14.6%."
    await evaluator.verify(
        claim=ry_claim,
        node=ry_node,
        sources=yield_sources,
        additional_instruction=(
            "Verify that the provided page explicitly supports that, in Q1 2024, for three-bedroom single-family rentals in Florida, "
            "Indian River County's annual gross rental yield is 14.6%. Accept minor textual variants such as '14.6 percent'. "
            "A press release or official ATTOM page summarizing the Q1 2024 report is acceptable."
        )
    )

    # 3) Population Data (NON-CRITICAL)
    pop_node = evaluator.add_leaf(
        id="Population_Data",
        desc="Includes population data or demographic statistics for the identified county",
        parent=rubric_node,
        critical=False
    )
    pop_claim = f"The population or demographic statistic for Indian River County, Florida mentioned in the answer is: '{extracted.population or ''}'."
    await evaluator.verify(
        claim=pop_claim,
        node=pop_node,
        sources=extracted.population_sources,
        additional_instruction=(
            "Check whether the cited page provides the same or reasonably equivalent population/demographic statistic. "
            "Allow rounding differences or year qualifiers if consistent with what the answer states. If the answer provides no such figure, mark as unsupported."
        )
    )

    # 4) Property Tax Information (NON-CRITICAL)
    tax_node = evaluator.add_leaf(
        id="Property_Tax_Information",
        desc="Provides property tax rates, per capita taxes, or related tax information for the county",
        parent=rubric_node,
        critical=False
    )
    tax_claim = f"The property tax information for Indian River County, Florida mentioned in the answer is: '{extracted.property_tax_info or ''}'."
    await evaluator.verify(
        claim=tax_claim,
        node=tax_node,
        sources=extracted.property_tax_sources,
        additional_instruction=(
            "Verify that the cited page supports the specific tax detail(s) stated (e.g., effective tax rate, millage, per capita taxes). "
            "Minor phrasing differences are acceptable if the meaning matches."
        )
    )

    # 5) Growth / Demographics (NON-CRITICAL)
    growth_node = evaluator.add_leaf(
        id="Growth_Demographics",
        desc="Includes information about population growth rate, demographic trends, or economic development in the county",
        parent=rubric_node,
        critical=False
    )
    growth_claim = f"The growth or demographic/economic trend for Indian River County, Florida mentioned in the answer is: '{extracted.growth_trends or ''}'."
    await evaluator.verify(
        claim=growth_claim,
        node=growth_node,
        sources=extracted.growth_sources,
        additional_instruction=(
            "Verify that the cited page supports the stated growth/demographic/economic trend (e.g., growth rate, migration trends, employment/economic development)."
        )
    )

    # 6) Location Context (NON-CRITICAL)
    loc_node = evaluator.add_leaf(
        id="Location_Context",
        desc="Describes the geographic location of the county within Florida, including regional context or nearby cities",
        parent=rubric_node,
        critical=False
    )
    loc_claim = f"The geographic/location description for Indian River County, Florida stated in the answer is: '{extracted.location_context or ''}'."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=extracted.location_sources,
        additional_instruction=(
            "Verify that the cited page supports the stated geographic description (e.g., Atlantic coast, proximity to specific cities/regions)."
        )
    )

    # 7) Source Citations (CRITICAL) – specifically cites ATTOM Q1 2024
    src_node = evaluator.add_leaf(
        id="Source_Citations",
        desc="Provides authoritative source URLs for the rental yield data and other key information, particularly citing the ATTOM Q1 2024 report",
        parent=rubric_node,
        critical=True
    )
    src_claim = "This webpage is ATTOM's Q1 2024 Single-Family Rental Market Report or an official ATTOM page/press release that clearly cites the Q1 2024 SFR rental yield figures."
    await evaluator.verify(
        claim=src_claim,
        node=src_node,
        sources=extracted.attom_report_urls,
        additional_instruction=(
            "Confirm that at least one provided URL is on attomdata.com (or an official ATTOM property) and clearly references the Q1 2024 Single-Family Rental Market Report or its summarized data. "
            "If no such ATTOM Q1 2024 URL is provided, mark as not supported."
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
        default_model=model
    )

    # Extract structured profile
    extracted = await evaluator.extract(
        prompt=prompt_extract_county_profile(),
        template_class=CountyProfileExtraction,
        extraction_name="county_profile_extraction"
    )

    # Record ground truth for reference
    evaluator.add_ground_truth(
        {
            "expected_county": GROUND_TRUTH["expected_county"],
            "expected_rental_yield": GROUND_TRUTH["expected_yield"],
            "scope": "Florida, Q1 2024, single-family three-bedroom annual gross rental yield (ATTOM)"
        },
        gt_type="ground_truth_profile"
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, root, extracted)

    # Return standardized summary
    return evaluator.get_summary()