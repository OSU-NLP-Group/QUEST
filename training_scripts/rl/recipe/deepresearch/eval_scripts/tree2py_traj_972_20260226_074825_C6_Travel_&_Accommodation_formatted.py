import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "wi_dells_suite_eval"
TASK_DESCRIPTION = """
A family of 8 people (6 adults and 2 children) is planning a vacation to Wisconsin Dells and needs to book appropriate accommodation at a waterpark resort. They require a suite that meets the following specific criteria:

Sleeping Arrangements:
- Must accommodate a maximum of at least 8 guests
- Must have exactly 2 separate bedrooms: one with a king bed and one with two queen beds
- Must include a queen sofa sleeper in the living room

Bathrooms:
- Must have at least 1.75 bathrooms (or 2 full bathrooms)
- Must include at least one bathtub

Living Space:
- Must have a separate living room area with a fireplace
- Must include a large balcony or patio

Kitchen Amenities:
- Must include a microwave, mini-fridge (or full refrigerator), and coffee maker
- Must have a table and chairs for dining

Entertainment & Technology:
- Must have at least 3 televisions distributed throughout the suite
- Must include complimentary high-speed internet access

Included Services:
- Waterpark admission must be included for all registered guests
- Must include complimentary access to a fitness center

Additional Requirements:
- The resort must be located in Wisconsin Dells, Wisconsin
- The resort must have indoor waterpark facilities

Question: Which Wisconsin Dells waterpark resort offers a specific suite type that meets all of these requirements? Provide both the resort name and the exact suite type name, along with reference URLs that verify the suite specifications.
""".strip()


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class SuiteSelectionExtraction(BaseModel):
    resort_name: Optional[str] = None
    suite_name: Optional[str] = None
    suite_urls: List[str] = Field(default_factory=list)
    resort_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_suite_selection() -> str:
    return """
    Extract the key items identifying the recommended Wisconsin Dells waterpark resort and the specific suite type, plus all reference URLs cited in the answer text.

    Required fields:
    - resort_name: The exact resort name (e.g., "Kalahari Resorts & Conventions – Wisconsin Dells").
    - suite_name: The exact suite type name (e.g., "2 Bedroom 3 Bath Living Room Suite").
    - suite_urls: An array of all URLs in the answer that specifically describe the suite and/or list its in-room features, occupancy, bedroom configuration, amenities, etc. This typically includes the official resort’s suite page(s) or booking pages. Include all such URLs if multiple are provided.
    - resort_urls: An array of any URLs in the answer that describe the resort-level amenities or facts (e.g., waterpark access info, fitness center, location page, resort homepage). Include all such URLs if provided.

    Rules:
    - Only extract URLs explicitly present in the answer text (plain or markdown links). Do not invent URLs.
    - Normalize URLs to include http:// or https:// if missing.
    - If a field is not present, return null (for a single value) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


def _combined_sources(extracted: SuiteSelectionExtraction) -> List[str]:
    return _normalize_urls((extracted.suite_urls or []) + (extracted.resort_urls or []))


# --------------------------------------------------------------------------- #
# Verification tree construction & checks                                     #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: SuiteSelectionExtraction) -> None:
    """
    Build the verification tree according to the rubric and run all verifications.
    All criticality and structure mirror the provided rubric.
    """
    # Top-level critical node under evaluator root (root node in Evaluator is always non-critical)
    top = evaluator.add_parallel(
        id="Wisconsin_Dells_Suite_Identification",
        desc="Identify a Wisconsin Dells indoor-waterpark resort and a specific suite type that meets all stated requirements, and provide reference URL(s) verifying the suite specifications.",
        parent=evaluator.root,
        critical=True
    )

    # ------------------ Answer_Provides_Resort_And_Suite (critical) ------------------ #
    provides_node = evaluator.add_parallel(
        id="Answer_Provides_Resort_And_Suite",
        desc="Provide the required named entities.",
        parent=top,
        critical=True
    )

    resort_provided = evaluator.add_custom_node(
        result=(extracted.resort_name is not None and extracted.resort_name.strip() != ""),
        id="Resort_Name_Provided",
        desc="A specific resort name is provided.",
        parent=provides_node,
        critical=True
    )

    suite_provided = evaluator.add_custom_node(
        result=(extracted.suite_name is not None and extracted.suite_name.strip() != ""),
        id="Suite_Type_Name_Provided",
        desc="The exact suite type name is provided.",
        parent=provides_node,
        critical=True
    )

    # ------------------ Reference URLs existence (critical) ------------------ #
    ref_urls_leaf = evaluator.add_custom_node(
        result=(len(extracted.suite_urls) > 0),
        id="Reference_URLs_For_Suite_Specifications",
        desc="Provide reference URL(s) that verify the suite specifications/amenities used to justify that the suite meets the stated requirements.",
        parent=top,
        critical=True
    )

    # Prepare context strings for claims
    resort_name = extracted.resort_name or ""
    suite_name = extracted.suite_name or ""
    all_sources = _combined_sources(extracted)

    # ------------------ Resort_Requirements (critical) ------------------ #
    resort_req = evaluator.add_parallel(
        id="Resort_Requirements",
        desc="Verify resort-level constraints from the prompt.",
        parent=top,
        critical=True
    )

    # Resort located in Wisconsin Dells, WI
    rr_loc = evaluator.add_leaf(
        id="Resort_Located_In_Wisconsin_Dells_WI",
        desc="The resort is located in Wisconsin Dells, Wisconsin.",
        parent=resort_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The resort '{resort_name}' is located in Wisconsin Dells, Wisconsin.",
        node=rr_loc,
        sources=all_sources,
        additional_instruction="Verify the resort’s address/location mentions 'Wisconsin Dells, WI' (or 'Wisconsin Dells, Wisconsin')."
    )

    # Resort has indoor waterpark facilities
    rr_iwp = evaluator.add_leaf(
        id="Resort_Has_Indoor_Waterpark",
        desc="The resort has indoor waterpark facilities.",
        parent=resort_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The resort '{resort_name}' has indoor waterpark facilities.",
        node=rr_iwp,
        sources=all_sources,
        additional_instruction="Accept phrasing such as 'indoor waterpark', 'indoor water park', or equivalent language clearly indicating indoor water play areas."
    )

    # ------------------ Suite_Requirements (critical) ------------------ #
    suite_req = evaluator.add_parallel(
        id="Suite_Requirements",
        desc="Verify suite-level constraints from the prompt.",
        parent=top,
        critical=True
    )

    # Max occupancy at least 8
    sr_occ = evaluator.add_leaf(
        id="Suite_Max_Occupancy_At_Least_8",
        desc="The suite accommodates a maximum occupancy of at least 8 guests.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{suite_name}' suite at '{resort_name}' sleeps up to at least 8 guests.",
        node=sr_occ,
        sources=all_sources,
        additional_instruction="Check for phrases like 'sleeps up to 8' or occupancy >= 8."
    )

    # Exactly 2 separate bedrooms
    sr_2br = evaluator.add_leaf(
        id="Suite_Has_Exactly_2_Separate_Bedrooms",
        desc="The suite has exactly 2 separate bedrooms.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{suite_name}' at '{resort_name}' is a two Logs bedroom suite with exactly 2 separate bedrooms.",
        node=sr_2br,
        sources=all_sources,
        additional_instruction="Accept 'two-bedroom' or explicit listing of 2 bedrooms. It should be clear they are separate rooms."
    )

    # One bedroom includes a king bed
    sr_king = evaluator.add_leaf(
        id="Suite_Bedroom_Includes_King_Bed",
        desc="One bedroom includes a king bed.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the '{suite_name}' at '{resort_name}', one of the bedrooms has a king bed.",
        node=sr_king,
        sources=all_sources,
        additional_instruction="Look for 'king bed' listed within the bedroom configuration."
    )

    # One bedroom includes two queen beds
    sr_two_queens = evaluator.add_leaf(
        id="Suite_Bedroom_Includes_Two_Queen_Beds",
        desc="One bedroom includes two queen beds.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the '{suite_name}' at '{resort_name}', one bedroom includes two queen beds (2 queens).",
        node=sr_two_queens,
        sources=all_sources,
        additional_instruction="Look for 'two queen beds', '2 queen beds', or equivalent phrasing."
    )

    # Queen sofa sleeper in living room
    sr_sofa = evaluator.add_leaf(
        id="Suite_Has_Queen_Sofa_Sleeper",
        desc="The living room includes a queen sofa sleeper.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The living room of the '{suite_name}' at '{resort_name}' includes a queen sofa sleeper (queen-sized sleeper sofa).",
        node=sr_sofa,
        sources=all_sources,
        additional_instruction="Accept 'queen sofa sleeper', 'queen sleeper sofa', or equivalent terminology."
    )

    # Bathrooms: at least 1.75 (or 2 full)
    sr_baths = evaluator.add_leaf(
        id="Suite_Bathroom_Count",
        desc="The suite has at least 1.75 bathrooms (or 2 full bathrooms).",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{suite_name}' at '{resort_name}' has at least two bathrooms (or 1.75 baths or more).",
        node=sr_baths,
        sources=all_sources,
        additional_instruction="Treat '2 bathrooms' as satisfying the requirement. If the page explicitly says '1.75 baths', that also satisfies the criterion."
    )

    # At least one bathtub
    sr_tub = evaluator.add_leaf(
        id="Suite_Has_At_Least_One_Bathtub",
        desc="At least one bathroom includes a bathtub.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{suite_name}' at '{resort_name}' includes at least one bathtub.",
        node=sr_tub,
        sources=all_sources,
        additional_instruction="Look for 'bathtub' or 'tub' in the bathroom details."
    )

    # Separate living room area
    sr_lr = evaluator.add_leaf(
        id="Suite_Has_Separate_Living_Room",
        desc="The suite has a separate living room area.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{suite_name}' at '{resort_name}' has a separate living room area.",
        node=sr_lr,
        sources=all_sources,
        additional_instruction="Accept 'separate living room', 'separate living area', or similar phrasing indicating a distinct room."
    )

    # Living room fireplace
    sr_fireplace = evaluator.add_leaf(
        id="Suite_Living_Room_Has_Fireplace",
        desc="The living room includes a fireplace.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The living room of the '{suite_name}' at '{resort_name}' includes a fireplace.",
        node=sr_fireplace,
        sources=all_sources,
        additional_instruction="Look for 'fireplace' listed among room features."
    )

    # Large balcony or patio
    sr_balcony = evaluator.add_leaf(
        id="Suite_Has_Large_Balcony_Or_Patio",
        desc="The suite includes a large balcony or patio.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{suite_name}' at '{resort_name}' includes a large balcony or patio.",
        node=sr_balcony,
        sources=all_sources,
        additional_instruction="Prefer explicit 'large' or 'spacious'. If the page clearly states a balcony or patio and implies spaciousness, that may be acceptable."
    )

    # Microwave
    sr_mw = evaluator.add_leaf(
        id="Suite_Has_Microwave",
        desc="The suite includes a microwave.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{suite_name}' at '{resort_name}' includes a microwave.",
        node=sr_mw,
        sources=all_sources,
        additional_instruction="Look for in-room amenities lists containing 'microwave'."
    )

    # Refrigerator (mini-fridge or full)
    sr_fridge = evaluator.add_leaf(
        id="Suite_Has_Refrigerator_Or_Mini_Fridge",
        desc="The suite includes a mini-fridge or full refrigerator.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{suite_name}' at '{resort_name}' includes either a mini-fridge or a full refrigerator.",
        node=sr_fridge,
        sources=all_sources,
        additional_instruction="Accept 'mini-fridge', 'mini refrigerator', or 'refrigerator'."
    )

    # Coffee maker
    sr_coffee = evaluator.add_leaf(
        id="Suite_Has_Coffee_Maker",
        desc="The suite includes a coffee maker.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{suite_name}' at '{resort_name}' includes a coffee maker.",
        node=sr_coffee,
        sources=all_sources,
        additional_instruction="Accept 'coffee maker', 'coffee machine', 'Keurig', etc."
    )

    # Dining table and chairs
    sr_dining = evaluator.add_leaf(
        id="Suite_Has_Dining_Table_And_Chairs",
        desc="The suite has a table and chairs for dining.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{suite_name}' at '{resort_name}' includes a dining table with chairs.",
        node=sr_dining,
        sources=all_sources,
        additional_instruction="Look for 'dining table and chairs' or equivalent (e.g., 'dining area with table and chairs')."
    )

    # At least 3 televisions
    sr_tvs = evaluator.add_leaf(
        id="Suite_Has_At_Least_3_Televisions",
        desc="The suite includes at least 3 televisions.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{suite_name}' at '{resort_name}' includes at least three televisions (3 TVs).",
        node=sr_tvs,
        sources=all_sources,
        additional_instruction="Look for explicit counts or wording indicating three or more TVs."
    )

    # Complimentary high-speed internet access
    sr_internet = evaluator.add_leaf(
        id="Suite_Has_Complimentary_High_Speed_Internet",
        desc="The suite includes complimentary high-speed internet access.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{suite_name}' at '{resort_name}' includes complimentary high-speed internet (Wi‑Fi).",
        node=sr_internet,
        sources=all_sources,
        additional_instruction="Accept 'complimentary Wi‑Fi', 'free high-speed internet', or equivalent phrasing."
    )

    # Waterpark admission included for all registered guests
    sr_waterpark = evaluator.add_leaf(
        id="Waterpark_Admission_Included_For_All_Registered_Guests",
        desc="Waterpark admission is included for all registered guests.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"Waterpark admission is included for all registered overnight guests at '{resort_name}'.",
        node=sr_waterpark,
        sources=all_sources,
        additional_instruction="This is typically a resort-level benefit; verify language like 'waterpark passes included for all overnight guests'."
    )

    # Complimentary fitness center access
    sr_fitness = evaluator.add_leaf(
        id="Complimentary_Fitness_Center_Access",
        desc="The booking includes complimentary access to a fitness center.",
        parent=suite_req,
        critical=True
    )
    await evaluator.verify(
        claim=f"Guests booking the '{suite_name}' at '{resort_name}' receive complimentary access to a fitness center.",
        node=sr_fitness,
        sources=all_sources,
        additional_instruction="Accept resort-level amenity wording indicating access is included with stay."
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
    Entry point for evaluating an answer for the Wisconsin Dells suite identification task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract resort & suite identifiers and reference URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_suite_selection(),
        template_class=SuiteSelectionExtraction,
        extraction_name="suite_selection"
    )

    # Store some custom info for debugging
    evaluator.add_custom_info(
        {
            "resort_name": extracted.resort_name,
            "suite_name": extracted.suite_name,
            "suite_urls_count": len(extracted.suite_urls or []),
            "resort_urls_count": len(extracted.resort_urls or []),
        },
        info_type="extraction_summary",
    )

    # Build tree and run verifications
    await build_and_verify_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()