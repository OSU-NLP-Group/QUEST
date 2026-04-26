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
TASK_ID = "cre_retax_vacancy_2025_2026"
TASK_DESCRIPTION = (
    "A commercial real estate investment firm is evaluating potential U.S. markets for retail property acquisition. "
    "Identify three U.S. states where the real estate property tax rate is below 0.60%. For each identified state, "
    "provide the following information supported by verifiable sources from 2025-2026: "
    "(1) The state's exact effective real estate property tax rate, "
    "(2) One major city located within that state, "
    "(3) The retail commercial real estate vacancy rate for that state or city market, "
    "(4) The standard closing cost percentage range that real estate buyers typically pay, and "
    "(5) The minimum dwelling coverage percentage that mortgage lenders require for property insurance. "
    "All data must be current as of 2025-2026 and supported by credible sources with accessible URLs."
)

YEARS_TEXT = "2025-2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateItem(BaseModel):
    state: Optional[str] = None
    property_tax_rate: Optional[str] = None  # e.g., "0.57%" or "0.57 percent"
    property_tax_urls: List[str] = Field(default_factory=list)

    city: Optional[str] = None
    city_urls: List[str] = Field(default_factory=list)

    retail_vacancy_rate: Optional[str] = None  # e.g., "4.8%" or "4.8 percent"
    retail_vacancy_geography: Optional[str] = None  # e.g., "statewide", "Austin", "Austin MSA"
    retail_vacancy_urls: List[str] = Field(default_factory=list)


class CREExtraction(BaseModel):
    # Up to three states; if more appear in the answer, still extract all (we will slice to first 3 in evaluation)
    states: List[StateItem] = Field(default_factory=list)

    # General requirements (typical across U.S.; if answer provides per-state values, consolidate to a single typical range)
    closing_cost_range: Optional[str] = None  # e.g., "2-6%" or "2%–5%"
    closing_cost_urls: List[str] = Field(default_factory=list)

    dwelling_coverage_min: Optional[str] = None  # e.g., "80% of replacement cost"
    dwelling_coverage_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cre() -> str:
    return """
Extract structured information from the answer for evaluating U.S. real estate markets with 2025-2026 sourcing.

Return JSON with:
- states: array of state entries (extract all that appear; we will only use the first 3)
  For each state entry, extract exactly these fields:
  - state: the U.S. state name (e.g., "Hawaii").
  - property_tax_rate: the state's effective real estate property tax rate provided in the answer (as a string, keep the exact formatting such as "0.57%").
  - property_tax_urls: an array of URLs that the answer cites as the source for that state's property tax rate (do not invent; include all that are explicitly linked in the answer).
  - city: one major city in that state mentioned in the answer (e.g., "Honolulu").
  - city_urls: an array of URLs used in the answer to support the city identification (if any are cited; otherwise empty array).
  - retail_vacancy_rate: the retail commercial real estate vacancy rate cited in the answer for either the state or a city market within that state (e.g., "4.8%").
  - retail_vacancy_geography: a short descriptor for the geography the vacancy rate refers to (e.g., "statewide", "Honolulu", "Phoenix MSA").
  - retail_vacancy_urls: an array of URLs cited for the retail vacancy rate (include all URLs explicitly linked in the answer for that vacancy figure).
  
- closing_cost_range: the standard closing cost percentage range buyers typically pay as stated in the answer (e.g., "2-6%").
- closing_cost_urls: array of URLs cited for closing costs (include those explicitly linked in the answer).
- dwelling_coverage_min: the minimum dwelling coverage % lenders require (e.g., "80% of replacement cost"), exactly as stated in the answer.
- dwelling_coverage_urls: array of URLs cited for dwelling coverage requirement.

Rules:
- Extract ONLY what is explicitly present in the answer.
- Do NOT infer or invent any URLs or values.
- Prefer strings for all numeric-like values (e.g., "0.57%" instead of 0.0057).
- If something is missing, set it to null (for single-value fields) or [] (for URL arrays).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["first", "second", "third", "fourth", "fifth"][n - 1] if 1 <= n <= 5 else f"#{n}"


def geography_label(si: StateItem) -> str:
    if si.retail_vacancy_geography:
        return si.retail_vacancy_geography
    if si.city and si.state:
        return f"{si.city}, {si.state}"
    return si.state or "the market"


# --------------------------------------------------------------------------- #
# Verification blocks                                                         #
# --------------------------------------------------------------------------- #
async def verify_state_block(evaluator: Evaluator, parent, si: StateItem, idx: int) -> None:
    """
    Build verification subtree for one state.
    Adjusted criticalities: the 'State_Selection_Requirement' parent is critical,
    so each 'State_i' node here is also set to critical to satisfy framework rules.
    """
    state_label = si.state or f"State #{idx + 1}"
    ord_txt = ordinal(idx + 1)

    # State node (critical under State_Selection_Requirement)
    state_node = evaluator.add_parallel(
        id=f"State_{idx + 1}",
        desc=f"{ord_txt.capitalize()} state identified with property tax rate below 0.60%",
        parent=parent,
        critical=True,
    )

    # 1) Property Tax block (critical)
    tax_node = evaluator.add_parallel(
        id=f"State_{idx + 1}_Property_Tax",
        desc=f"Property tax rate verification for the {ord_txt} state",
        parent=state_node,
        critical=True,
    )

    # 1.a) Value: below 0.60% (critical)
    tax_value_leaf = evaluator.add_leaf(
        id=f"State_{idx + 1}_Tax_Rate_Value",
        desc=f"The property tax rate for the {ord_txt} state is below 0.60%",
        parent=tax_node,
        critical=True,
    )
    tax_rate_text = si.property_tax_rate or ""
    await evaluator.verify(
        claim=f"The provided effective real estate property tax rate value '{tax_rate_text}' for {state_label} is below 0.60% (i.e., less than 0.60 percent). "
              f"If the value is expressed as a decimal (e.g., 0.0057), interpret it as 0.57%.",
        node=tax_value_leaf,
        additional_instruction="Focus only on whether the numeric value provided in the answer is < 0.60%. Tolerate rounding differences.",
    )

    # 1.b) Source presence (critical) – ensure at least one URL
    evaluator.add_custom_node(
        result=bool(si.property_tax_urls),
        id=f"State_{idx + 1}_Tax_Rate_Source_URLs_Present",
        desc=f"At least one source URL is provided for the property tax rate of {state_label}",
        parent=tax_node,
        critical=True,
    )

    # 1.c) Source verification (critical)
    tax_src_leaf = evaluator.add_leaf(
        id=f"State_{idx + 1}_Tax_Rate_Source",
        desc="The property tax rate is supported by a verifiable source from 2025-2026",
        parent=tax_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided webpage reports the effective real estate property tax rate for {state_label} as '{tax_rate_text}' "
              f"(or a very close rounded equivalent) and this figure explicitly pertains to the years {YEARS_TEXT} (published/updated in 2025 or 2026 or the dataset labeled 2025/2026).",
        node=tax_src_leaf,
        sources=si.property_tax_urls,
        additional_instruction=(
            "Only pass if the page clearly indicates the figure is an 'effective' property tax rate (average effective rate) and specifically for 2025 or 2026. "
            "Accept minor rounding differences (e.g., 0.57% vs 0.569%). "
            "If the page has only older data (e.g., 2024 or earlier) or is undated/ambiguous about 2025/2026, mark as not supported."
        ),
    )

    # 2) City block (critical)
    city_node = evaluator.add_parallel(
        id=f"State_{idx + 1}_City",
        desc=f"Major city identification for the {ord_txt} state",
        parent=state_node,
        critical=True,
    )

    # 2.a) City name (critical)
    city_value_leaf = evaluator.add_leaf(
        id=f"State_{idx + 1}_City_Name",
        desc=f"A major city within the {ord_txt} state is identified",
        parent=city_node,
        critical=True,
    )
    city_text = si.city or ""
    await evaluator.verify(
        claim=f"The city '{city_text}' is located in the state of {state_label}. Treat 'major city' flexibly (capital, top population, principal city, or well-known metropolitan hub).",
        node=city_value_leaf,
        additional_instruction="If the state name is missing, consider the claim failed. Allow reasonable spelling/formatting variations.",
    )

    # 2.b) City source presence (critical)
    evaluator.add_custom_node(
        result=bool(si.city_urls),
        id=f"State_{idx + 1}_City_Source_URLs_Present",
        desc=f"At least one source URL is provided to verify that {city_text} is in {state_label}",
        parent=city_node,
        critical=True,
    )

    # 2.c) City source verification (critical)
    city_src_leaf = evaluator.add_leaf(
        id=f"State_{idx + 1}_City_Source",
        desc="The city identification is verifiable",
        parent=city_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided webpage confirms that '{city_text}' is a city located within the state of {state_label}.",
        node=city_src_leaf,
        sources=si.city_urls,
        additional_instruction="Evidence can include state/city official pages, census, encyclopedia, or other credible references. The page must clearly indicate the city is in the given state.",
    )

    # 3) Retail vacancy block (critical)
    vac_node = evaluator.add_parallel(
        id=f"State_{idx + 1}_Retail_Vacancy",
        desc=f"Retail vacancy rate information for the {ord_txt} state or its identified city",
        parent=state_node,
        critical=True,
    )

    # 3.a) Vacancy value (critical)
    vac_value_leaf = evaluator.add_leaf(
        id=f"State_{idx + 1}_Vacancy_Value",
        desc=f"A retail vacancy rate is provided for the {ord_txt} state or its identified city",
        parent=vac_node,
        critical=True,
    )
    vac_text = si.retail_vacancy_rate or ""
    geo_text = geography_label(si)
    await evaluator.verify(
        claim=f"The provided vacancy value '{vac_text}' is a valid retail commercial real estate vacancy rate for {geo_text} (i.e., a percentage or rate).",
        node=vac_value_leaf,
        additional_instruction="This is specifically for the RETAIL sector (not office/industrial). Tolerate simple formats like '4.8%' or 'about 5%'.",
    )

    # 3.b) Vacancy source presence (critical)
    evaluator.add_custom_node(
        result=bool(si.retail_vacancy_urls),
        id=f"State_{idx + 1}_Vacancy_Source_URLs_Present",
        desc=f"At least one source URL is provided for the retail vacancy rate for {geo_text}",
        parent=vac_node,
        critical=True,
    )

    # 3.c) Vacancy source verification (critical)
    vac_src_leaf = evaluator.add_leaf(
        id=f"State_{idx + 1}_Vacancy_Source",
        desc="The retail vacancy rate is supported by a verifiable source from 2025-2026",
        parent=vac_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided webpage reports the RETAIL commercial real estate vacancy rate for {geo_text} as '{vac_text}' (or a reasonably close rounded figure), "
              f"and the figure clearly pertains to {YEARS_TEXT} (published/updated in 2025 or 2026 or a dataset labeled 2025/2026).",
        node=vac_src_leaf,
        sources=si.retail_vacancy_urls,
        additional_instruction=(
            "Verify that the metric is for RETAIL vacancy (not office, industrial, or overall non-retail). "
            "Ensure the page clearly indicates the data year is 2025 or 2026. If only older or ambiguous, mark as unsupported."
        ),
    )


async def verify_general_requirements(evaluator: Evaluator, parent, data: CREExtraction) -> None:
    # General Requirements (critical under root)
    gen_node = evaluator.add_parallel(
        id="General_Requirements",
        desc="General real estate financial and insurance requirements applicable across all states",
        parent=parent,
        critical=True,
    )

    # Closing Costs block (critical)
    close_node = evaluator.add_parallel(
        id="Closing_Costs",
        desc="Standard closing cost percentage range for real estate buyers",
        parent=gen_node,
        critical=True,
    )

    # Closing Costs range value (critical)
    close_value_leaf = evaluator.add_leaf(
        id="Closing_Costs_Range",
        desc="A percentage range for typical buyer closing costs is provided (typically 2-6% of loan amount)",
        parent=close_node,
        critical=True,
    )
    ccr = data.closing_cost_range or ""
    await evaluator.verify(
        claim=f"The provided closing costs range '{ccr}' represents a valid percentage range (e.g., '2-6%') for typical buyer closing costs.",
        node=close_value_leaf,
        additional_instruction="Do not require exact 2–6%; accept common ranges (e.g., 2–5%, 2–6%). This check is about the existence and formatting as a range.",
    )

    # Closing Costs source presence (critical)
    evaluator.add_custom_node(
        result=bool(data.closing_cost_urls),
        id="Closing_Costs_Source_URLs_Present",
        desc="At least one source URL is provided for closing costs",
        parent=close_node,
        critical=True,
    )

    # Closing Costs source verification (critical)
    close_src_leaf = evaluator.add_leaf(
        id="Closing_Costs_Source",
        desc="The closing costs information is supported by verifiable sources from 2025-2026",
        parent=close_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided webpage states that typical buyer closing costs are around '{ccr}' of the purchase price or loan amount "
              f"(or a clearly equivalent range), and the content is current in {YEARS_TEXT} (published/updated in 2025 or 2026).",
        node=close_src_leaf,
        sources=data.closing_cost_urls,
        additional_instruction="Only pass if the page clearly conveys the buyer's closing cost percentage range and indicates a 2025/2026 date or dataset. Accept reasonable range phrasing.",
    )

    # Dwelling Coverage block (critical)
    cover_node = evaluator.add_parallel(
        id="Dwelling_Coverage",
        desc="Minimum dwelling coverage percentage required by mortgage lenders",
        parent=gen_node,
        critical=True,
    )

    # Dwelling coverage percentage value (critical)
    cover_value_leaf = evaluator.add_leaf(
        id="Dwelling_Coverage_Percentage",
        desc="The minimum dwelling coverage percentage requirement is provided (typically 80% of replacement cost)",
        parent=cover_node,
        critical=True,
    )
    dcp = data.dwelling_coverage_min or ""
    await evaluator.verify(
        claim=f"The provided dwelling coverage minimum '{dcp}' is a valid percentage requirement that mortgage lenders require for property insurance (e.g., 80% of replacement cost or similar).",
        node=cover_value_leaf,
        additional_instruction="This check is about the existence and format as a % requirement. Typical value is 80% RC, but 100% RC also appears. Accept clear minimum % statements.",
    )

    # Dwelling coverage source presence (critical)
    evaluator.add_custom_node(
        result=bool(data.dwelling_coverage_urls),
        id="Dwelling_Coverage_Source_URLs_Present",
        desc="At least one source URL is provided for dwelling coverage requirement",
        parent=cover_node,
        critical=True,
    )

    # Dwelling coverage source verification (critical)
    cover_src_leaf = evaluator.add_leaf(
        id="Dwelling_Coverage_Source",
        desc="The dwelling coverage requirement is supported by verifiable sources from 2025-2026",
        parent=cover_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided webpage states that mortgage lenders require a minimum dwelling coverage of '{dcp}' (e.g., at least 80% of replacement cost), "
              f"and the content is current in {YEARS_TEXT} (published/updated in 2025 or 2026).",
        node=cover_src_leaf,
        sources=data.dwelling_coverage_urls,
        additional_instruction="Only pass if the page clearly indicates a lender requirement minimum % and provides a 2025/2026 publication/update marker or dataset year.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
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
    """
    Entry point to evaluate the agent's answer against the rubric tree for the CRE task.
    """
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
        default_model=model,
    )

    # Extract structured data
    extracted: CREExtraction = await evaluator.extract(
        prompt=prompt_extract_cre(),
        template_class=CREExtraction,
        extraction_name="cre_market_extraction",
    )

    # Root node is critical in the original rubric. Keep root critical with both child groups critical.
    # Note: In the original rubric, 'State_Selection_Requirement' was critical but its children 'State_1..3' were non-critical,
    # which violates the framework rule that children of a critical node must also be critical.
    # We adjust by making State_1..3 critical to satisfy consistency while preserving the intent that all three states must pass.

    # State selection requirement group (critical)
    state_sel_node = evaluator.add_parallel(
        id="State_Selection_Requirement",
        desc="Three distinct U.S. states are identified, each with property tax rate below 0.60%",
        parent=root,
        critical=True,
    )

    # Build state verifications for first three states (pad with empty entries if fewer provided)
    states = list(extracted.states[:3])
    while len(states) < 3:
        states.append(StateItem())  # ensures nodes are created and will fail if missing

    for i, si in enumerate(states):
        await verify_state_block(evaluator, state_sel_node, si, i)

    # General requirements (critical)
    await verify_general_requirements(evaluator, root, extracted)

    # Add some helpful custom info
    evaluator.add_custom_info(
        info={
            "years_required": YEARS_TEXT,
            "states_extracted_count": len(extracted.states),
        },
        info_type="meta",
        info_name="evaluation_context",
    )

    # Return summary
    return evaluator.get_summary()