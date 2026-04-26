import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "budget_airlines_hvn_florida"
TASK_DESCRIPTION = (
    "Identify budget airlines that operate a base at Tweed-New Haven Airport (HVN) in Connecticut and offer direct "
    "flights to Florida destinations. For each qualifying airline, provide: (1) the airline name, (2) confirmation that "
    "it is a budget/low-cost carrier, (3) confirmation that it operates a base at HVN, (4) at least one specific Florida "
    "destination city they serve directly from New Haven with the airport code, and (5) reference URLs supporting each "
    "piece of information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RouteInfo(BaseModel):
    """Information for a single Florida destination served directly from HVN."""
    city: Optional[str] = None
    airport_code: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AirlineEntry(BaseModel):
    """All extracted info for one airline."""
    name: Optional[str] = None
    budget_sources: List[str] = Field(default_factory=list)
    hvn_base_sources: List[str] = Field(default_factory=list)
    destinations: List[RouteInfo] = Field(default_factory=list)
    current_sources: List[str] = Field(default_factory=list)


class AirlinesExtraction(BaseModel):
    """Top-level extraction: list of up to 5 airlines."""
    airlines: List[AirlineEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_airlines() -> str:
    return """
    Extract up to five airline entries mentioned in the answer that aim to satisfy the task:
    Identify budget airlines that operate a base at Tweed–New Haven Airport (HVN) and offer direct flights to Florida destinations,
    providing required fields and sources.

    For each airline, extract the following fields:

    1) name: The airline's name exactly as presented in the answer. If not provided, return null.

    2) budget_sources: An array of reference URLs cited in the answer that support the classification of the airline as a
       budget/low-cost/ultra-low-cost carrier. Accept credible aviation sources (e.g., airline official site, airline or airport press
       releases, recognized industry publications, Wikipedia with citations). If none are provided, return an empty array.

    3) hvn_base_sources: An array of reference URLs cited in the answer that support the statement that the airline operates a base/hub
       at Tweed–New Haven Airport (HVN). If none are provided, return an empty array.

    4) destinations: An array of Florida destinations for which the answer claims the airline operates nonstop/direct flights from HVN.
       For EACH destination, extract:
         - city: The Florida destination city name (e.g., Orlando, Tampa, Fort Lauderdale). If not present, return null.
         - airport_code: The destination airport IATA code (e.g., MCO, TPA, FLL). If not present, return null.
         - sources: An array of URLs cited in the answer that support the direct HVN→Florida route and/or the destination details.
           If none are provided, return an empty array.
       Include all such destinations mentioned in the answer for this airline. If none are mentioned, return an empty array.

    5) current_sources: An array of URLs cited in the answer indicating that the HVN base and/or HVN→Florida route is current
       and operating in late 2025 (roughly Oct–Dec 2025). If none are provided, return an empty array.

    IMPORTANT:
    - Only extract URLs explicitly present in the answer. Do not invent URLs.
    - Include full URLs (prepend http:// if protocol is missing).
    - If more than five airlines are mentioned, only extract the first five in the order they appear.
    - If some fields are missing for an airline, set them to null or empty arrays as instructed above.

    Return a JSON object with a single field:
    {
      "airlines": [ ... up to five AirlineEntry objects ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _pick_primary_destination(airline: AirlineEntry) -> Optional[RouteInfo]:
    """Pick the first destination that has city, airport_code, and at least one source."""
    for dest in airline.destinations:
        if (dest.city and dest.city.strip()) and (dest.airport_code and dest.airport_code.strip()) and dest.sources:
            return dest
    return None


def _fail_leaf(node: VerificationNode) -> None:
    """Convenience to mark a leaf as failed without invoking LLM verification."""
    node.score = 0.0
    node.status = "failed"


# --------------------------------------------------------------------------- #
# Verification for one airline                                                #
# --------------------------------------------------------------------------- #
async def verify_airline(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    airline: AirlineEntry,
    idx: int,
) -> None:
    """
    Build verification subtree for a single airline entry.
    Each verification leaf corresponds to a single required check.
    """

    # Parent node for this airline (parallel; non-critical so partial is allowed)
    airline_node = evaluator.add_parallel(
        id=f"airline_{idx + 1}",
        desc=f"Airline entry #{idx + 1} evaluated against all constraints/required fields",
        parent=parent_node,
        critical=False,
    )

    # ---- 1) Airline_Name_Provided (critical existence check) ----
    name_present = airline.name is not None and airline.name.strip() != ""
    evaluator.add_custom_node(
        result=name_present,
        id=f"airline_{idx + 1}_name_provided",
        desc="Airline name is provided.",
        parent=airline_node,
        critical=True,
    )

    # ---- 2) Budget_Carrier_With_Source (critical; requires sources + verification) ----
    budget_leaf = evaluator.add_leaf(
        id=f"airline_{idx + 1}_budget_carrier_with_source",
        desc=(
            "Airline is classified as a budget/low-cost/ultra-low-cost carrier in the United States AND "
            "a supporting reference URL from an official airline website or credible aviation source is provided."
        ),
        parent=airline_node,
        critical=True,
    )
    if not airline.budget_sources:
        _fail_leaf(budget_leaf)
    else:
        claim_budget = (
            f"The airline '{airline.name}' is categorized as a budget/low-cost or ultra-low-cost carrier in the United States."
        )
        await evaluator.verify(
            claim=claim_budget,
            node=budget_leaf,
            sources=airline.budget_sources,
            additional_instruction=(
                "Use only the provided sources to judge. Accept credible classifications such as 'low-cost carrier' or "
                "'ultra-low-cost carrier (ULCC)'. Marketing phrases alone are insufficient unless the source explicitly "
                "classifies the airline as such."
            ),
        )

    # ---- 3) HVN_Base_With_Source (critical; requires sources + verification) ----
    base_leaf = evaluator.add_leaf(
        id=f"airline_{idx + 1}_hvn_base_with_source",
        desc=(
            "Airline operates an official base (hub) at Tweed–New Haven Airport (HVN) AND a supporting reference URL from an "
            "official airline website or credible aviation source is provided."
        ),
        parent=airline_node,
        critical=True,
    )
    if not airline.hvn_base_sources:
        _fail_leaf(base_leaf)
    else:
        claim_base = f"The airline '{airline.name}' operates an official base (hub) at Tweed–New Haven Airport (HVN)."
        await evaluator.verify(
            claim=claim_base,
            node=base_leaf,
            sources=airline.hvn_base_sources,
            additional_instruction=(
                "Confirm that the source explicitly indicates a base/hub or crew/aircraft stationing at HVN—mere service "
                "(i.e., operating flights to/from HVN) is not sufficient."
            ),
        )

    # ---- 4) Florida_Nonstop_Destination_With_Code_And_Source (critical; verify one destination) ----
    florida_leaf = evaluator.add_leaf(
        id=f"airline_{idx + 1}_florida_nonstop_with_code_and_source",
        desc=(
            "At least one nonstop/direct Florida destination city served from HVN is provided WITH the destination airport "
            "code AND a supporting reference URL from an official airline website or credible aviation source is provided."
        ),
        parent=airline_node,
        critical=True,
    )
    primary_dest = _pick_primary_destination(airline)
    if not primary_dest:
        _fail_leaf(florida_leaf)
    else:
        claim_route = (
            f"The airline '{airline.name}' operates nonstop/direct flights from Tweed–New Haven (HVN) to "
            f"{primary_dest.city}, Florida ({primary_dest.airport_code})."
        )
        await evaluator.verify(
            claim=claim_route,
            node=florida_leaf,
            sources=primary_dest.sources,
            additional_instruction=(
                "Verify that the destination is in Florida, the IATA code matches the city/airport, and the route is nonstop/direct "
                "from HVN. Use schedules, route maps, press releases, airport pages, and credible aviation sources."
            ),
        )

    # ---- 5) Current_As_Of_Late_2025_With_Source (critical; verify recency as of late 2025) ----
    current_leaf = evaluator.add_leaf(
        id=f"airline_{idx + 1}_current_as_of_late_2025_with_source",
        desc=(
            "Information indicates the base/route is currently operating as of late 2025 AND a supporting reference URL is provided."
        ),
        parent=airline_node,
        critical=True,
    )

    # Build sources for current verification: prefer explicit current_sources; otherwise, fall back to any provided route/base sources
    combined_sources: List[str] = []
    if airline.current_sources:
        combined_sources.extend(airline.current_sources)
    # Fall-back: augment with route/base sources if available
    if primary_dest and primary_dest.sources:
        combined_sources.extend(primary_dest.sources)
    if airline.hvn_base_sources:
        combined_sources.extend(airline.hvn_base_sources)

    # Deduplicate
    combined_sources = list(dict.fromkeys(combined_sources))

    if not combined_sources:
        _fail_leaf(current_leaf)
    else:
        # Prefer referencing the route if available; otherwise reference the base
        if primary_dest:
            current_claim = (
                f"As of late 2025 (around Oct–Dec 2025), the HVN–{primary_dest.city} ({primary_dest.airport_code}) route "
                f"operated by '{airline.name}' was active."
            )
        else:
            current_claim = f"As of late 2025 (around Oct–Dec 2025), the HVN base operated by '{airline.name}' was active."

        await evaluator.verify(
            claim=current_claim,
            node=current_leaf,
            sources=combined_sources,
            additional_instruction=(
                "Confirm recency explicitly. Pass only if the page content or date metadata indicates operation in late 2025 "
                "(approximately Oct–Dec 2025). If evidence is older or ambiguous with no late-2025 indication, judge as NOT SUPPORTED."
            ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the HVN budget airlines to Florida task.
    """
    # Initialize evaluator with a parallel root (independent airlines)
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

    # Root node matching rubric description
    airlines_root = evaluator.add_parallel(
        id="airlines_meeting_criteria",
        desc=(
            "Identify budget airlines that operate a base at Tweed–New Haven Airport (HVN) and offer direct flights to Florida "
            "destinations, providing required fields and sources."
        ),
        parent=root,
        critical=False,
    )

    # Extract structured content from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_airlines(),
        template_class=AirlinesExtraction,
        extraction_name="airlines_extraction",
    )

    # Record a quick summary of extraction
    evaluator.add_custom_info(
        info={"num_airlines_extracted": len(extracted.airlines)},
        info_type="extraction_stats",
        info_name="airlines_extraction_stats",
    )

    # Ensure we evaluate up to five entries (pad with empty placeholders if fewer are provided)
    airlines_to_check: List[AirlineEntry] = list(extracted.airlines[:5])
    while len(airlines_to_check) < 5:
        airlines_to_check.append(AirlineEntry())

    # Build verification subtrees for each airline
    for i, airline in enumerate(airlines_to_check):
        await verify_airline(evaluator, airlines_root, airline, i)

    # Return structured evaluation summary
    return evaluator.get_summary()