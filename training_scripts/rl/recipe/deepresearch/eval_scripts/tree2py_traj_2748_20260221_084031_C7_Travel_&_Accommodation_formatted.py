import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "kalahari_specs_wisconsin_dells"
TASK_DESCRIPTION = (
    "I am preparing a detailed travel guide about Wisconsin Dells waterpark resorts and need to verify comprehensive "
    "specifications for Kalahari Resort. According to my preliminary research, Kalahari Resort in Wisconsin Dells, "
    "Wisconsin currently operates a 125,000 square-foot indoor waterpark and has a 75,000 square-foot expansion under "
    "construction that is scheduled to open in Fall 2026. Please provide and verify the following complete facility "
    "specifications for this resort: 1. Current indoor waterpark size (in square feet), 2. Expansion size (in square feet), "
    "3. Expansion opening timeframe, 4. Total indoor waterpark size after the expansion is complete (in square feet), "
    "5. Outdoor waterpark size (in square feet), 6. Indoor theme park size (in square feet), 7. Total resort square footage "
    "(in square feet), 8. Total number of waterslides, 9. Total number of pools and whirlpools, 10. Key design features of "
    "the expansion, 11. Resort location (city and state), 12. Current status claim regarding Wisconsin indoor waterparks, "
    "13. Total investment amount for the expansion. Each specification must be supported by reference URLs from official "
    "sources or reputable travel/news websites."
)

# Optional ground truth hints from preliminary research (for logging only; not enforced)
PRELIMINARY_EXPECTATIONS = {
    "current_indoor_waterpark_size": "125,000 square feet",
    "expansion_size": "75,000 square feet",
    "expansion_opening_timeframe": "Fall 2026",
    "total_indoor_waterpark_size_after_expansion": "200,000 square feet",
    "outdoor_waterpark_size": "77,000 square feet",
    "indoor_theme_park_size": "over 100,000 square feet",
    "total_resort_square_footage": "over 1 million square feet",
    "number_of_waterslides": "54",
    "number_of_pools_and_whirlpools": "20",
    "expansion_design_features": "glass-enclosed design with a retractable roof",
    "resort_location": "Wisconsin Dells, Wisconsin",
    "largest_indoor_waterpark_status": "Wisconsin's largest indoor waterpark",
    "expansion_investment_amount": "$85 million",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SpecField(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class KalahariSpecs(BaseModel):
    current_indoor_waterpark_size: Optional[SpecField] = None
    expansion_size: Optional[SpecField] = None
    expansion_opening_timeframe: Optional[SpecField] = None
    total_indoor_waterpark_size_after_expansion: Optional[SpecField] = None
    outdoor_waterpark_size: Optional[SpecField] = None
    indoor_theme_park_size: Optional[SpecField] = None
    total_resort_square_footage: Optional[SpecField] = None
    number_of_waterslides: Optional[SpecField] = None
    number_of_pools_and_whirlpools: Optional[SpecField] = None
    expansion_design_features: Optional[SpecField] = None
    resort_location: Optional[SpecField] = None
    largest_indoor_waterpark_status: Optional[SpecField] = None
    expansion_investment_amount: Optional[SpecField] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_kalahari_specs() -> str:
    return """
Extract the comprehensive facility specifications for Kalahari Resort in Wisconsin Dells, Wisconsin as stated in the answer.

For each item below, extract:
- value: the exact value as written in the answer (keep units/qualifiers like "square feet", "over", "+", "Fall 2026", "$85 million").
- sources: a list of all URLs explicitly cited in the answer that directly support this specific item. Only include URLs actually present in the answer (plain or markdown links). Do not invent URLs.

Return a JSON object with the following fields (each an object with `value` and `sources`):
- current_indoor_waterpark_size
- expansion_size
- expansion_opening_timeframe
- total_indoor_waterpark_size_after_expansion
- outdoor_waterpark_size
- indoor_theme_park_size
- total_resort_square_footage
- number_of_waterslides
- number_of_pools_and_whirlpools
- expansion_design_features
- resort_location
- largest_indoor_waterpark_status
- expansion_investment_amount

Rules:
1) If a value is missing from the answer, set it to null.
2) If no supporting URLs are cited for a field in the answer, return an empty list for that field’s `sources`.
3) Keep numbers as strings exactly as written (e.g., "125,000 square feet", "over 1 million square feet", "$85 million", "Fall 2026").
4) Prefer URLs that clearly reference the Wisconsin Dells, WI location (avoid other Kalahari locations). However, only extract from the answer; do not add new URLs.
"""


# --------------------------------------------------------------------------- #
# Spec definition and criticality mapping                                     #
# --------------------------------------------------------------------------- #
# Mapping from our field keys to tree node IDs, descriptions, and criticality (per rubric JSON)
SPEC_DEFS: List[Dict[str, Any]] = [
    {
        "field_key": "current_indoor_waterpark_size",
        "node_id": "Current_Indoor_Waterpark_Size",
        "node_desc": "The current indoor waterpark size is 125,000 square feet",
        "critical": True,
    },
    {
        "field_key": "expansion_size",
        "node_id": "Expansion_Size",
        "node_desc": "The expansion under construction is 75,000 square feet",
        "critical": True,
    },
    {
        "field_key": "expansion_opening_timeframe",
        "node_id": "Expansion_Opening_Date",
        "node_desc": "The expansion is scheduled to open in Fall 2026",
        "critical": True,
    },
    {
        "field_key": "total_indoor_waterpark_size_after_expansion",
        "node_id": "Total_Indoor_Size_After_Expansion",
        "node_desc": "The total indoor waterpark size after expansion will be 200,000 square feet",
        "critical": False,
    },
    {
        "field_key": "outdoor_waterpark_size",
        "node_id": "Outdoor_Waterpark_Size",
        "node_desc": "The outdoor waterpark size is 77,000 square feet",
        "critical": False,
    },
    {
        "field_key": "indoor_theme_park_size",
        "node_id": "Indoor_Theme_Park_Size",
        "node_desc": "The indoor theme park is over 100,000 square feet",
        "critical": False,
    },
    {
        "field_key": "total_resort_square_footage",
        "node_id": "Total_Resort_Square_Footage",
        "node_desc": "The total resort square footage is over 1 million square feet",
        "critical": False,
    },
    {
        "field_key": "number_of_waterslides",
        "node_id": "Number_of_Waterslides",
        "node_desc": "The resort has 54 waterslides",
        "critical": False,
    },
    {
        "field_key": "number_of_pools_and_whirlpools",
        "node_id": "Number_of_Pools_Whirlpools",
        "node_desc": "The resort has 20 pools and whirlpools",
        "critical": False,
    },
    {
        "field_key": "expansion_design_features",
        "node_id": "Expansion_Design_Features",
        "node_desc": "The expansion features a glass-enclosed design with a retractable roof",
        "critical": False,
    },
    {
        "field_key": "resort_location",
        "node_id": "Resort_Location",
        "node_desc": "The resort is located in Wisconsin Dells, Wisconsin",
        "critical": True,
    },
    {
        "field_key": "largest_indoor_waterpark_status",
        "node_id": "Largest_Indoor_Waterpark_Status",
        "node_desc": "The resort currently holds the status of Wisconsin's largest indoor waterpark",
        "critical": False,
    },
    {
        "field_key": "expansion_investment_amount",
        "node_id": "Expansion_Investment_Amount",
        "node_desc": "The expansion represents an $85 million investment",
        "critical": False,
    },
]


# --------------------------------------------------------------------------- #
# Claim construction                                                          #
# --------------------------------------------------------------------------- #
BASE_INSTRUCTION = (
    "Only consider the Kalahari Resort in Wisconsin Dells, Wisconsin (WI). "
    "Disregard pages about other Kalahari locations (e.g., Sandusky, OH; Pocono Mountains, PA; Round Rock, TX). "
    "Allow minor formatting variations (commas in numbers, 'square feet' vs 'sq ft', symbols like 'SF', '+' signs, "
    "and phrasings like 'over' or '~'). The claim is supported if at least one cited URL explicitly supports it."
)


def build_claim_and_instruction(field_key: str, value: str) -> Tuple[str, str]:
    if field_key == "current_indoor_waterpark_size":
        claim = f"According to the cited webpages, the current indoor waterpark size of Kalahari Resort in Wisconsin Dells, WI is {value}."
        ins = BASE_INSTRUCTION + " Focus on the indoor waterpark size currently in operation."
    elif field_key == "expansion_size":
        claim = f"According to the cited webpages, the indoor waterpark expansion under construction at Kalahari Resort in Wisconsin Dells is {value}."
        ins = BASE_INSTRUCTION + " The page should explicitly refer to the size of the expansion project."
    elif field_key == "expansion_opening_timeframe":
        claim = f"According to the cited webpages, the indoor waterpark expansion is scheduled to open {value}."
        ins = BASE_INSTRUCTION + " Accept timeframe expressions like 'Fall 2026' or more specific months in that window."
    elif field_key == "total_indoor_waterpark_size_after_expansion":
        claim = f"According to the cited webpages, once complete, the total indoor waterpark size at Kalahari Resort in Wisconsin Dells will be {value}."
        ins = BASE_INSTRUCTION + " This should describe the total indoor waterpark square footage after the expansion."
    elif field_key == "outdoor_waterpark_size":
        claim = f"According to the cited webpages, the outdoor waterpark size at Kalahari Resort in Wisconsin Dells is {value}."
        ins = BASE_INSTRUCTION + " Ensure the figure refers specifically to the outdoor waterpark area."
    elif field_key == "indoor_theme_park_size":
        claim = f"According to the cited webpages, the indoor theme park at Kalahari Resort in Wisconsin Dells is {value}."
        ins = BASE_INSTRUCTION + " The indoor theme park may be referred to as 'Tom Foolerys Adventure Park'."
    elif field_key == "total_resort_square_footage":
        claim = f"According to the cited webpages, the total resort square footage for Kalahari Resort in Wisconsin Dells is {value}."
        ins = BASE_INSTRUCTION + " This should refer to the total square footage of the entire resort."
    elif field_key == "number_of_waterslides":
        claim = f"According to the cited webpages, the Kalahari Resort in Wisconsin Dells has {value} waterslides."
        ins = BASE_INSTRUCTION + " Count should apply to the Wisconsin Dells location only."
    elif field_key == "number_of_pools_and_whirlpools":
        claim = f"According to the cited webpages, the Kalahari Resort in Wisconsin Dells has {value} pools and whirlpools."
        ins = BASE_INSTRUCTION + " Combined count of pools and whirlpools is acceptable if the source states it."
    elif field_key == "expansion_design_features":
        claim = f"According to the cited webpages, the indoor waterpark expansion at Kalahari Resort in Wisconsin Dells features {value}."
        ins = BASE_INSTRUCTION + " Focus on design elements like 'glass-enclosed' structures, 'retractable roof', or similar key features."
    elif field_key == "resort_location":
        claim = f"According to the cited webpages, Kalahari Resort is located in {value}."
        ins = BASE_INSTRUCTION + " The correct location should be 'Wisconsin Dells, Wisconsin' (or 'Wisconsin Dells, WI')."
    elif field_key == "largest_indoor_waterpark_status":
        claim = f"According to the cited webpages, Kalahari Resort in Wisconsin Dells currently holds the status of {value}."
        ins = BASE_INSTRUCTION + " The status should pertain to Wisconsin (e.g., 'Wisconsin's largest indoor waterpark')."
    elif field_key == "expansion_investment_amount":
        claim = f"According to the cited webpages, the indoor waterpark expansion represents an investment of {value}."
        ins = BASE_INSTRUCTION + " Monetary figures may include symbols like '$' or words like 'million'."
    else:
        # Fallback generic claim
        claim = f"According to the cited webpages, the following statement about Kalahari Resort in Wisconsin Dells is correct: {value}"
        ins = BASE_INSTRUCTION
    return claim, ins


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_spec_field(
    evaluator: Evaluator,
    parent_node,
    field_key: str,
    node_id: str,
    node_desc: str,
    field: Optional[SpecField],
    parent_critical: bool,
) -> None:
    """
    Create a spec group node, add existence check, and verify the claim against cited sources.
    """
    # Group node per spec
    group_node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=parent_critical
    )

    # Existence check: value present AND at least one source URL present
    has_value = bool(field and field.value and str(field.value).strip())
    has_sources = bool(field and field.sources and len(field.sources) > 0)

    evaluator.add_custom_node(
        result=has_value and has_sources,
        id=f"{node_id}_exists",
        desc=f"{node_desc} – value and supporting source URLs are provided in the answer",
        parent=group_node,
        critical=True  # Critical to gate verification; if missing, subsequent leaf will be skipped
    )

    # Verification leaf (source‑grounded)
    verify_leaf = evaluator.add_leaf(
        id=f"{node_id}_source_support",
        desc=f"{node_desc} – claim is supported by cited sources",
        parent=group_node,
        critical=True  # Keep critical to enforce gating within this spec group
    )

    # Build and run verification if data exists; auto‑preconditions will skip if existence fails
    value_str = field.value if field and field.value else ""
    sources_list = field.sources if field and field.sources else []
    claim, add_ins = build_claim_and_instruction(field_key, value_str)

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=sources_list,  # verify_by_urls is triggered when list has 2+ URLs; single URL triggers verify_by_url; empty list falls back but will be skipped by precondition
        additional_instruction=add_ins
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Kalahari Resort (Wisconsin Dells) comprehensive specifications verification.
    """
    # Initialize evaluator with a parallel root (each spec verified independently)
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

    # Extract structured specs from the answer
    extracted_specs = await evaluator.extract(
        prompt=prompt_extract_kalahari_specs(),
        template_class=KalahariSpecs,
        extraction_name="kalahari_specs_extraction"
    )

    # Add a top-level node for this resort verification (as in rubric)
    resort_node = evaluator.add_parallel(
        id="Kalahari_Resort_Verification",
        desc="Verify comprehensive facility specifications for Kalahari Resort in Wisconsin Dells",
        parent=root,
        critical=False
    )

    # Add preliminary expectations as GT info (for logging/reference only)
    evaluator.add_ground_truth(
        {
            "preliminary_expectations": PRELIMINARY_EXPECTATIONS,
            "note": "These are preliminary research hints and not enforced as ground truth."
        },
        gt_type="preliminary_research"
    )

    # Verify each spec field as per rubric
    for spec in SPEC_DEFS:
        field_key = spec["field_key"]
        node_id = spec["node_id"]
        node_desc = spec["node_desc"]
        parent_critical = spec["critical"]

        field_obj: Optional[SpecField] = getattr(extracted_specs, field_key, None)
        await verify_spec_field(
            evaluator=evaluator,
            parent_node=resort_node,
            field_key=field_key,
            node_id=node_id,
            node_desc=node_desc,
            field=field_obj,
            parent_critical=parent_critical
        )

    # Return summary with verification tree and extraction logs
    return evaluator.get_summary()