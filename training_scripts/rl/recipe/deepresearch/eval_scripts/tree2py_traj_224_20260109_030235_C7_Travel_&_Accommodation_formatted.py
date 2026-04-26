import asyncio
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_state_capitals_2020"
TASK_DESCRIPTION = (
    "You are a travel consultant planning a multi-city tour across major state capital cities in the United States. "
    "Your client wants to visit the three most populous state capitals based on the 2020 US Census data. "
    "Identify these three state capitals in order from largest to smallest, providing for each: "
    "(1) the capital city name, (2) the state, and (3) the 2020 Census population figure."
)

# Ground truth for the three largest state capitals by 2020 US Census
GROUND_TRUTH_CAPITALS = [
    {
        "rank": 1,
        "city": "Phoenix",
        "state": "Arizona",
        "population_2020": "1,608,139",
    },
    {
        "rank": 2,
        "city": "Austin",
        "state": "Texas",
        "population_2020": "961,855",
    },
    {
        "rank": 3,
        "city": "Columbus",
        "state": "Ohio",
        "population_2020": "905,748",
    },
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CapitalEntry(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    population_2020: Optional[str] = None


class CapitalsExtraction(BaseModel):
    first: Optional[CapitalEntry] = None
    second: Optional[CapitalEntry] = None
    third: Optional[CapitalEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_capitals() -> str:
    return (
        "Extract from the answer the three most populous official US state capital cities, in order from largest "
        "to smallest, as the user has presented them. For each of the first three entries, return:\n"
        "- city: the capital city name\n"
        "- state: the US state name for that capital\n"
        "- population_2020: the 2020 US Census population figure as written in the answer (keep formatting as-is; "
        "if the answer uses commas or spaces, preserve them; do not convert to numeric types).\n\n"
        "Map the first three entries explicitly to JSON fields:\n"
        "- first: {city, state, population_2020}\n"
        "- second: {city, state, population_2020}\n"
        "- third: {city, state, population_2020}\n\n"
        "Rules:\n"
        "1) Only extract values explicitly mentioned in the answer; do not infer or add.\n"
        "2) If any field for an entry is missing, set it to null.\n"
        "3) If the answer lists more than three entries, only extract the first three.\n"
        "4) If fewer than three are provided, return null for missing entries."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_capital_slot(
    evaluator: Evaluator,
    parent_node,
    slot_id_prefix: str,
    slot_desc_prefix: str,
    extracted: Optional[CapitalEntry],
    expected_city: str,
    expected_state: str,
    expected_population: str,
) -> None:
    """
    Build verification nodes for a single capital slot (first, second, third) and run checks.
    Parent node is a parallel aggregator. Each leaf is critical.
    """
    # Create the slot aggregator node (non-critical to allow partial credit across slots)
    slot_node = evaluator.add_parallel(
        id=slot_id_prefix,
        desc=f"{slot_desc_prefix} entry is correct and complete.",
        parent=parent_node,
        critical=False,
    )

    city_val = extracted.city if extracted and extracted.city else ""
    state_val = extracted.state if extracted and extracted.state else ""
    pop_val = extracted.population_2020 if extracted and extracted.population_2020 else ""

    # City check (critical leaf)
    city_leaf = evaluator.add_leaf(
        id=f"{slot_id_prefix}_City" if "Capital" in slot_id_prefix else f"{slot_id_prefix}_city",
        desc=f"{slot_desc_prefix} capital city name is {expected_city}.",
        parent=slot_node,
        critical=True,
    )
    city_claim = (
        f"The {slot_desc_prefix.lower()} capital city listed in the answer is '{city_val}', "
        f"and the expected city is '{expected_city}'. Determine if these refer to the same city."
    )
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        additional_instruction=(
            "Judge name equality with case-insensitive comparison and allow minor stylistic differences "
            "(e.g., punctuation or spacing). Do not accept a completely different city."
        ),
    )

    # State check (critical leaf)
    state_leaf = evaluator.add_leaf(
        id=f"{slot_id_prefix}_State" if "Capital" in slot_id_prefix else f"{slot_id_prefix}_state",
        desc=f"{slot_desc_prefix} capital state is {expected_state}.",
        parent=slot_node,
        critical=True,
    )
    state_claim = (
        f"The state for the {slot_desc_prefix.lower()} capital listed in the answer is '{state_val}', "
        f"and the expected state is '{expected_state}'. Determine if these refer to the same state."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        additional_instruction=(
            "Judge state name equality ignoring case and minor variations like 'State of Texas' vs 'Texas'. "
            "Do not accept a different state."
        ),
    )

    # Population check (critical leaf)
    pop_leaf = evaluator.add_leaf(
        id=f"{slot_id_prefix}_Population_2020_Census" if "Capital" in slot_id_prefix else f"{slot_id_prefix}_population",
        desc=f"{slot_desc_prefix} capital 2020 Census population is {expected_population}.",
        parent=slot_node,
        critical=True,
    )
    pop_claim = (
        f"The 2020 US Census population for the {slot_desc_prefix.lower()} capital is reported in the answer as "
        f"'{pop_val}', and the expected exact figure is '{expected_population}'. Decide whether the reported figure "
        f"matches the expected figure exactly."
    )
    await evaluator.verify(
        claim=pop_claim,
        node=pop_leaf,
        additional_instruction=(
            "Require an exact numeric match to the expected population figure. Accept differences in thousands "
            "separators (e.g., '1608139' vs '1,608,139'), but do NOT accept rounded values (e.g., '1.61 million') "
            "or different counts. If the field is missing or empty, this should fail."
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
    Evaluate an answer for the 'three largest state capitals by 2020 Census' task.
    """
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

    # Extract the three capitals from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_capitals(),
        template_class=CapitalsExtraction,
        extraction_name="capitals_extraction",
    )

    # Add a top-level aggregator corresponding to rubric's main node (set non-critical to allow partial credit)
    main_node = evaluator.add_parallel(
        id="Three_Largest_State_Capitals_Identified",
        desc="Identify the three most populous official US state capitals using 2020 US Census populations, "
             "in largest-to-smallest order, and provide city, state, and population for each.",
        parent=root,
        critical=False,
    )

    # Add ground truth information for transparency
    evaluator.add_ground_truth({
        "expected": GROUND_TRUTH_CAPITALS,
        "note": "Top three most populous US state capitals by 2020 Census: Phoenix (AZ) 1,608,139; "
                "Austin (TX) 961,855; Columbus (OH) 905,748."
    })

    # Verify each slot according to rubric
    # First (largest)
    await verify_capital_slot(
        evaluator=evaluator,
        parent_node=main_node,
        slot_id_prefix="First_Capital",
        slot_desc_prefix="First-ranked (largest) capital",
        extracted=extracted.first,
        expected_city=GROUND_TRUTH_CAPITALS[0]["city"],
        expected_state=GROUND_TRUTH_CAPITALS[0]["state"],
        expected_population=GROUND_TRUTH_CAPITALS[0]["population_2020"],
    )

    # Second
    await verify_capital_slot(
        evaluator=evaluator,
        parent_node=main_node,
        slot_id_prefix="Second_Capital",
        slot_desc_prefix="Second-ranked capital",
        extracted=extracted.second,
        expected_city=GROUND_TRUTH_CAPITALS[1]["city"],
        expected_state=GROUND_TRUTH_CAPITALS[1]["state"],
        expected_population=GROUND_TRUTH_CAPITALS[1]["population_2020"],
    )

    # Third
    await verify_capital_slot(
        evaluator=evaluator,
        parent_node=main_node,
        slot_id_prefix="Third_Capital",
        slot_desc_prefix="Third-ranked capital",
        extracted=extracted.third,
        expected_city=GROUND_TRUTH_CAPITALS[2]["city"],
        expected_state=GROUND_TRUTH_CAPITALS[2]["state"],
        expected_population=GROUND_TRUTH_CAPITALS[2]["population_2020"],
    )

    # Return evaluation summary
    return evaluator.get_summary()