import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "florida_city_real_estate_criteria"
TASK_DESCRIPTION = (
    "Identify a city in Florida that meets ALL of the following criteria as of February 2026: "
    "(1) the city serves as the county seat of its respective county, "
    "(2) the county has a population exceeding 1.5 million residents, "
    "(3) the city has an international airport located within its county boundaries, "
    "(4) the county's effective property tax rate is between 0.80% and 0.90%, "
    "(5) the city or county allows short-term rentals of 7 nights or less in designated residential or commercial zones, "
    "(6) the city allows accessory dwelling units (ADUs) with defined regulations, "
    "(7) the county has zoning designations that permit multi-family residential development, "
    "(8) the city's median home price is between $400,000 and $500,000, "
    "(9) the city has a population exceeding 300,000 residents, "
    "(10) the county experienced population growth between 2023 and 2024, "
    "(11) the city is located in Florida, which is a landlord-friendly state with no rent control laws, and "
    "(12) the state requires seller's disclosure notices for previously occupied single-family residences. "
    "Provide the name of the city, the county it serves, the county's population, the name of the international airport, "
    "the county's property tax rate, and the city's approximate median home price."
)

# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class CityCriteriaExtraction(BaseModel):
    # Core identification
    city_name: Optional[str] = None
    county_name: Optional[str] = None
    international_airport_name: Optional[str] = None
    state_name: Optional[str] = None

    # Quantitative or summarized facts (prefer strings for flexibility)
    county_population: Optional[str] = None
    county_property_tax_rate: Optional[str] = None
    city_median_home_price: Optional[str] = None
    city_population: Optional[str] = None
    county_population_growth_2023_to_2024: Optional[str] = None  # e.g., "increased", "increased by X", etc.

    # Regulatory/policy summaries
    allows_short_term_rentals_7_nights_or_less: Optional[str] = None  # textual summary as stated in answer
    allows_adus: Optional[str] = None  # textual summary
    multifamily_zoning_permitted: Optional[str] = None  # textual summary

    # Source URLs per criterion
    sources_county_seat: List[str] = Field(default_factory=list)
    sources_county_population: List[str] = Field(default_factory=list)
    sources_international_airport: List[str] = Field(default_factory=list)
    sources_property_tax_rate: List[str] = Field(default_factory=list)
    sources_short_term_rentals: List[str] = Field(default_factory=list)
    sources_adu: List[str] = Field(default_factory=list)
    sources_multifamily_zoning: List[str] = Field(default_factory=list)
    sources_median_home_price: List[str] = Field(default_factory=list)
    sources_city_population: List[str] = Field(default_factory=list)
    sources_population_growth: List[str] = Field(default_factory=list)
    sources_landlord_state: List[str] = Field(default_factory=list)
    sources_seller_disclosure: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_city_criteria() -> str:
    return """
Extract the structured information that the answer provides about a Florida city satisfying the specified real-estate related criteria. Return exactly the following JSON fields:

Core identification
- city_name: The city name proposed in the answer.
- county_name: The name of the county that city serves as county seat.
- international_airport_name: The name of the international airport cited for the city/county.
- state_name: The U.S. state of the city (should be 'Florida' if provided in the answer).

Key quantitative/summary facts (keep as strings, exactly as stated in the answer)
- county_population: The stated population figure or description (e.g., '1,950,000 in 2024').
- county_property_tax_rate: The effective property tax rate for the county, as a percentage string if provided (e.g., '0.85%').
- city_median_home_price: The approximate median home price for the city (e.g., '$450,000').
- city_population: The city population described (e.g., '410,000').
- county_population_growth_2023_to_2024: A short description that the county population increased from 2023 to 2024 (e.g., 'increased', 'went up by 1.2%', etc.).

Regulatory/policy summaries (verbatim or summarized from the answer)
- allows_short_term_rentals_7_nights_or_less: Whether short-term rentals of 7 nights or less are allowed and under which designated zones (e.g., 'Allowed in designated zones' or a short paraphrase).
- allows_adus: Whether the city allows accessory dwelling units (ADUs) with defined regulations (short text).
- multifamily_zoning_permitted: Whether the county zoning permits multi-family residential development (short text).

For each criterion, extract all cited source URLs explicitly present in the answer text. If none are provided, return an empty list:
- sources_county_seat
- sources_county_population
- sources_international_airport
- sources_property_tax_rate
- sources_short_term_rentals
- sources_adu
- sources_multifamily_zoning
- sources_median_home_price
- sources_city_population
- sources_population_growth
- sources_landlord_state
- sources_seller_disclosure

Rules for URLs:
- Only include actual URLs explicitly present in the answer (plain URLs or markdown links).
- Do not invent URLs. If none are present for a field, return an empty array for that field.

If any requested field is not mentioned in the answer, set it to null (or an empty array for sources).
    """.strip()


# -----------------------------------------------------------------------------
# Verification helpers
# -----------------------------------------------------------------------------
def _safe(s: Optional[str]) -> str:
    return s or ""


# -----------------------------------------------------------------------------
# Build and verify the rubric tree
# -----------------------------------------------------------------------------
async def _build_and_verify(
    evaluator: Evaluator,
    root,
    data: CityCriteriaExtraction,
) -> None:
    """
    Build the verification nodes and run verifications according to the rubric.
    We add a critical parallel aggregator under the root to reflect 'ALL criteria must be met'.
    """
    # Critical aggregator under root (root created by Evaluator is non-critical by design)
    all_criteria = evaluator.add_parallel(
        id="all_criteria",
        desc="All required criteria for the selected Florida city are satisfied and properly sourced",
        parent=root,
        critical=True
    )

    # Existence checks (as required by rubric: city, county, airport provided)
    evaluator.add_custom_node(
        result=bool(data.city_name and data.city_name.strip()),
        id="city_name_provided",
        desc="The solution provides the name of the city",
        parent=all_criteria,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(data.county_name and data.county_name.strip()),
        id="county_name_provided",
        desc="The solution provides the name of the county the city serves",
        parent=all_criteria,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(data.international_airport_name and data.international_airport_name.strip()),
        id="airport_name_provided",
        desc="The solution provides the name of the international airport",
        parent=all_criteria,
        critical=True
    )

    city = _safe(data.city_name)
    county = _safe(data.county_name)
    airport = _safe(data.international_airport_name)

    # 1) County seat status
    n_county_seat = evaluator.add_leaf(
        id="county_seat_status",
        desc="The city is the county seat of its respective county",
        parent=all_criteria,
        critical=True
    )
    claim_seat = f"{city} is the county seat of {county} County, Florida."
    await evaluator.verify(
        claim=claim_seat,
        node=n_county_seat,
        sources=data.sources_county_seat,
        additional_instruction=(
            "Confirm that the city is the designated county seat for the named county in Florida. "
            "Allow minor variations (e.g., abbreviations)."
        ),
    )

    # 2) County population > 1.5 million
    n_county_pop = evaluator.add_leaf(
        id="county_population",
        desc="The county has a population exceeding 1.5 million residents as of 2024",
        parent=all_criteria,
        critical=True
    )
    claim_county_pop = f"As of 2024, the population of {county} County exceeds 1.5 million residents."
    await evaluator.verify(
        claim=claim_county_pop,
        node=n_county_pop,
        sources=data.sources_county_population,
        additional_instruction=(
            "Check the county population as close to 2024 as possible. "
            "Rounding differences are acceptable; confirm that the figure is above 1,500,000."
        ),
    )

    # 3) International airport within county boundaries
    n_intl_airport = evaluator.add_leaf(
        id="international_airport",
        desc="The city has an international airport located within its county boundaries",
        parent=all_criteria,
        critical=True
    )
    claim_airport = (
        f"'{airport}' is an international airport located within {county} County, Florida."
    )
    await evaluator.verify(
        claim=claim_airport,
        node=n_intl_airport,
        sources=data.sources_international_airport,
        additional_instruction=(
            "Verify that the named airport is designated 'international' and lies within the named county's boundaries. "
            "If the airport is not within the city limits but is within the county, that satisfies the requirement."
        ),
    )

    # 4) Property tax rate in [0.80%, 0.90%]
    n_tax = evaluator.add_leaf(
        id="property_tax_rate",
        desc="The county's property tax rate is between 0.80% and 0.90%",
        parent=all_criteria,
        critical=True
    )
    claim_tax = f"The effective property tax rate for {county} County is between 0.80% and 0.90%."
    await evaluator.verify(
        claim=claim_tax,
        node=n_tax,
        sources=data.sources_property_tax_rate,
        additional_instruction=(
            "Interpret 'effective property tax rate' as the overall rate borne by property owners. "
            "Use 2024–2026 data if available. Allow standard rounding; confirm it lies in [0.80%, 0.90%]."
        ),
    )

    # 5) Short-term rentals ≤ 7 nights allowed in designated zones
    n_str = evaluator.add_leaf(
        id="short_term_rentals",
        desc="The city or county allows short-term rentals (7 nights or less) in designated residential or commercial zones",
        parent=all_criteria,
        critical=True
    )
    claim_str = (
        f"In {city} or {county} County, short-term rentals of 7 nights or less are permitted within designated zones."
    )
    await evaluator.verify(
        claim=claim_str,
        node=n_str,
        sources=data.sources_short_term_rentals,
        additional_instruction=(
            "Look for official ordinance or policy language allowing vacation/short-term rentals of 7 nights or fewer "
            "in specific residential or commercial zones (synonyms acceptable)."
        ),
    )

    # 6) ADU regulations allowed
    n_adu = evaluator.add_leaf(
        id="adu_regulations",
        desc="The city allows accessory dwelling units (ADUs) with defined regulations",
        parent=all_criteria,
        critical=True
    )
    claim_adu = f"The city of {city} allows accessory dwelling units (ADUs) with defined regulations."
    await evaluator.verify(
        claim=claim_adu,
        node=n_adu,
        sources=data.sources_adu,
        additional_instruction=(
            "Confirm that the city's code explicitly allows ADUs (accessory dwelling units) and provides regulations "
            "or standards governing them."
        ),
    )

    # 7) Multifamily zoning permitted in county
    n_mf = evaluator.add_leaf(
        id="multifamily_zoning",
        desc="The county has zoning designations that permit multi-family residential development",
        parent=all_criteria,
        critical=True
    )
    claim_mf = f"{county} County's zoning code includes designations permitting multi-family residential development."
    await evaluator.verify(
        claim=claim_mf,
        node=n_mf,
        sources=data.sources_multifamily_zoning,
        additional_instruction=(
            "Verify that the county zoning designations (e.g., MR, RM, or similar) permit multi-family (multifamily) residential uses."
        ),
    )

    # 8) City median home price in [$400k, $500k]
    n_mhp = evaluator.add_leaf(
        id="median_home_price",
        desc="The city's median home price is between $400,000 and $500,000 as of 2024-2026",
        parent=all_criteria,
        critical=True
    )
    claim_mhp = (
        f"The approximate median home sale price in {city} is between $400,000 and $500,000 (2024–2026 timeframe)."
    )
    await evaluator.verify(
        claim=claim_mhp,
        node=n_mhp,
        sources=data.sources_median_home_price,
        additional_instruction=(
            "Use city-level market data around 2024–2026. Allow rounding and modest variance; confirm the estimate lies within "
            "$400k–$500k range."
        ),
    )

    # 9) City population > 300,000
    n_city_pop = evaluator.add_leaf(
        id="city_population",
        desc="The city has a population exceeding 300,000 residents",
        parent=all_criteria,
        critical=True
    )
    claim_city_pop = f"The population of {city} exceeds 300,000 residents."
    await evaluator.verify(
        claim=claim_city_pop,
        node=n_city_pop,
        sources=data.sources_city_population,
        additional_instruction=(
            "Confirm city population (latest or recent estimate). Allow rounding; ensure it is strictly greater than 300,000."
        ),
    )

    # 10) County population grew from 2023 to 2024
    n_growth = evaluator.add_leaf(
        id="population_growth",
        desc="The county experienced population growth between 2023 and 2024",
        parent=all_criteria,
        critical=True
    )
    claim_growth = f"Between 2023 and 2024, the population of {county} County increased."
    await evaluator.verify(
        claim=claim_growth,
        node=n_growth,
        sources=data.sources_population_growth,
        additional_instruction=(
            "Check official/credible sources (e.g., census estimates) indicating population increased from 2023 to 2024."
        ),
    )

    # 11) Florida has no rent control and is landlord-friendly (policy check)
    n_landlord = evaluator.add_leaf(
        id="landlord_friendly_state",
        desc="The city is located in Florida, which is a landlord-friendly state with no rent control laws",
        parent=all_criteria,
        critical=True
    )
    # We focus the verification on Florida's policy aspects (no statewide rent control, landlord-friendly stance).
    claim_landlord = "Florida has no statewide rent control laws and is considered landlord-friendly."
    await evaluator.verify(
        claim=claim_landlord,
        node=n_landlord,
        sources=data.sources_landlord_state,
        additional_instruction=(
            "Verify Florida policy: no statewide rent control; general characterization as landlord-friendly is acceptable with credible sources."
        ),
    )

    # 12) Seller disclosure requirement (state-level)
    n_disclosure = evaluator.add_leaf(
        id="seller_disclosure_requirement",
        desc="The state requires seller's disclosure notices for previously occupied single-family residences",
        parent=all_criteria,
        critical=True
    )
    claim_disclosure = (
        "Florida requires sellers of previously occupied single-family homes to provide disclosure notices of known defects or material facts."
    )
    await evaluator.verify(
        claim=claim_disclosure,
        node=n_disclosure,
        sources=data.sources_seller_disclosure,
        additional_instruction=(
            "Verify Florida's statutory or case law–based seller disclosure obligations for occupied single-family residences."
        ),
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extract structured information from the answer
    extracted: CityCriteriaExtraction = await evaluator.extract(
        prompt=prompt_extract_city_criteria(),
        template_class=CityCriteriaExtraction,
        extraction_name="city_criteria_extraction",
    )

    # Optional: record a brief summary for convenience
    evaluator.add_custom_info(
        {
            "city": extracted.city_name,
            "county": extracted.county_name,
            "airport": extracted.international_airport_name,
            "county_population": extracted.county_population,
            "county_property_tax_rate": extracted.county_property_tax_rate,
            "city_median_home_price": extracted.city_median_home_price,
            "city_population": extracted.city_population,
            "state": extracted.state_name,
        },
        info_type="selection_summary",
    )

    # Build and verify rubric
    await _build_and_verify(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()