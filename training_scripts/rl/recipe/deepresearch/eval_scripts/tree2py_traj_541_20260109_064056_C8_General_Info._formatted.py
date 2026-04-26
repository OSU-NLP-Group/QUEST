import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_airports_amenities_2025_top10"
TASK_DESCRIPTION = (
    "From the 10 busiest US airports by 2025 passenger traffic, identify 4 airports that each offer all three of the following amenities in post-security areas: "
    "(1) dedicated nursing rooms or lactation facilities, "
    "(2) service animal/pet relief areas, and "
    "(3) either a yoga room/studio OR a rest & recharge area with seating and privacy partitions. "
    "Provide the three-letter airport code for each airport, describe each amenity with specific details, and include a reference URL from the airport's official website or trusted aviation source for each airport."
)

ALLOWED_TOP10_CODES = ["ATL", "DFW", "DEN", "ORD", "LAX", "JFK", "LAS", "PHX", "CLT", "SFO"]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AirportAmenity(BaseModel):
    code: Optional[str] = None
    nursing_desc: Optional[str] = None
    pet_relief_desc: Optional[str] = None
    wellness_type: Optional[str] = None  # e.g., "yoga" or "rest_recharge" (or textual variant)
    wellness_desc: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AirportsExtraction(BaseModel):
    airports: List[AirportAmenity] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_airports() -> str:
    return """
    Extract all airports mentioned in the answer that the user claims meet the three amenity requirements in post-security areas.
    For each airport, extract the following fields exactly as written in the answer:

    - code: The three-letter IATA airport code (e.g., ATL, DFW). If the answer uses a name instead of a code, convert to the code only if the code is explicitly stated in the answer; otherwise return null.
    - nursing_desc: The description text for the dedicated nursing room/lactation facility. Include the location and any specifics the answer provides (e.g., terminal/concourse/gate/near landmark, count, hours, facility type). If not described, return null.
    - pet_relief_desc: The description text for the service animal/pet relief area. Include the location and any specifics (e.g., terminal/concourse/gate/near landmark, count). If not described, return null.
    - wellness_type: If the answer describes a yoga room/studio, set to "yoga". If it describes a rest & recharge area with seating and privacy partitions (e.g., quiet room, lounge with partitions), set to "rest_recharge". If unclear or unspecified, return a short string that the answer used (e.g., "wellness space").
    - wellness_desc: The description text for the wellness amenity, including post-security location and key features. If not described, return null.
    - urls: All reference URLs mentioned for this airport. The URLs must be explicitly present in the answer (plain links or markdown form). Extract only valid URLs; if none provided, return an empty list.

    Return a JSON object with a single field:
    {
      "airports": [
        { "code": "...", "nursing_desc": "...", "pet_relief_desc": "...", "wellness_type": "...", "wellness_desc": "...", "urls": ["...", "..."] },
        ...
      ]
    }

    IMPORTANT:
    - Do not invent any information. Only extract what the answer actually states.
    - Keep the descriptions as free-form text. Do not normalize or rephrase.
    - Include all airports the answer lists, even if more than four; the evaluator will handle selecting the first four later.
    - If a field is missing, use null (or empty list for urls) as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_code(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    code = code.strip().upper()
    return code if len(code) == 3 else code


def select_first_k_airports(all_airports: List[AirportAmenity], k: int = 4) -> List[AirportAmenity]:
    # Filter out completely empty items, then select first k
    filtered = [a for a in all_airports if any([a.code, a.nursing_desc, a.pet_relief_desc, a.wellness_desc, a.urls])]
    selected = filtered[:k]
    # Pad to k with empty placeholders if fewer
    while len(selected) < k:
        selected.append(AirportAmenity())
    return selected


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_global_requirements(evaluator: Evaluator, parent_node, selected_airports: List[AirportAmenity]) -> None:
    """
    Create and verify the Global Requirements node:
    - Exactly 4 airports listed (with non-empty codes)
    - All 4 airports are distinct
    """
    global_node = evaluator.add_parallel(
        id="Global_Requirements",
        desc="Response-level requirements that apply to the full set of airports.",
        parent=parent_node,
        critical=True  # Critical gatekeeping
    )

    # Exactly 4 airports with codes present
    codes = [normalize_code(a.code) for a in selected_airports]
    num_with_codes = sum(1 for c in codes if c)
    exact_four = (len(selected_airports) == 4) and (num_with_codes == 4)

    evaluator.add_custom_node(
        result=exact_four,
        id="Exactly_4_Airports_Listed",
        desc="Response identifies exactly 4 airports (not fewer or more).",
        parent=global_node,
        critical=True
    )

    # Distinctness check (case-insensitive)
    unique_codes = set([c for c in codes if c])
    all_distinct = (len(unique_codes) == 4)

    evaluator.add_custom_node(
        result=all_distinct,
        id="All_4_Airports_Are_Distinct",
        desc="All identified airports are distinct (no duplicates).",
        parent=global_node,
        critical=True
    )


async def verify_airport(
    evaluator: Evaluator,
    parent_node,
    airport: AirportAmenity,
    idx: int
) -> None:
    """
    Verify a single airport item and its required attributes and citations.
    """
    airport_node = evaluator.add_parallel(
        id=f"Airport_{idx+1}",
        desc=f"{idx+1}st identified airport and its required attributes/citations." if idx == 0 else (
            f"{idx+1}nd identified airport and its required attributes/citations." if idx == 1 else (
                f"{idx+1}rd identified airport and its required attributes/citations." if idx == 2 else
                f"{idx+1}th identified airport and its required attributes/citations."
            )
        ),
        parent=parent_node,
        critical=False
    )

    code = normalize_code(airport.code)

    # 1) Airport code provided (critical)
    evaluator.add_custom_node(
        result=bool(code),
        id=f"Airport_{idx+1}_Code_Provided",
        desc=f"Provides a three-letter IATA airport code for Airport {idx+1}.",
        parent=airport_node,
        critical=True
    )

    # 2) Airport in allowed top-10 list (critical)
    in_allowed_node = evaluator.add_leaf(
        id=f"Airport_{idx+1}_In_Allowed_Top10_List",
        desc=f"Airport {idx+1} is one of the allowed top-10 airports listed in the constraints (ATL, DFW, DEN, ORD, LAX, JFK, LAS, PHX, CLT, SFO).",
        parent=airport_node,
        critical=True
    )
    claim_allowed = f"The airport code '{code or ''}' is one of the allowed top-10 codes: {', '.join(ALLOWED_TOP10_CODES)}."
    await evaluator.verify(
        claim=claim_allowed,
        node=in_allowed_node,
        additional_instruction="This is a simple membership check; treat codes case-insensitively."
    )

    # 3) Nursing post-security with details (critical)
    nursing_node = evaluator.add_leaf(
        id=f"Airport_{idx+1}_Nursing_PostSecurity_With_Details",
        desc=f"Describes a dedicated nursing room/lactation facility for Airport {idx+1} that is explicitly post-security and includes specific details.",
        parent=airport_node,
        critical=True
    )
    nursing_desc = (airport.nursing_desc or "").strip()
    claim_nursing = (
        f"Based on this amenity description for airport {code or 'UNKNOWN'}: '{nursing_desc}'. "
        f"The description clearly indicates a dedicated nursing room or lactation facility that is located in a post-security area "
        f"(e.g., behind security, after security, airside) and includes at least one specific detail such as terminal/concourse/gate/landmark, a count/number, operating hours, or facility type."
    )
    await evaluator.verify(
        claim=claim_nursing,
        node=nursing_node,
        additional_instruction="Focus on whether 'post-security' (or equivalent phrasing) is explicitly present and whether at least one concrete detail is included."
    )

    # 4) Pet relief post-security with details (critical)
    pet_node = evaluator.add_leaf(
        id=f"Airport_{idx+1}_PetRelief_PostSecurity_With_Details",
        desc=f"Describes a service animal/pet relief area for Airport {idx+1} that is explicitly post-security and includes specific details.",
        parent=airport_node,
        critical=True
    )
    pet_desc = (airport.pet_relief_desc or "").strip()
    claim_pet = (
        f"Based on this amenity description for airport {code or 'UNKNOWN'}: '{pet_desc}'. "
        f"The description clearly indicates a service animal/pet relief area located in a post-security area "
        f"(e.g., behind security, after security, airside) and includes at least one specific detail such as terminal/concourse/gate/landmark or count."
    )
    await evaluator.verify(
        claim=claim_pet,
        node=pet_node,
        additional_instruction="Allow common synonyms for 'post-security' (behind/after security, airside). Check for at least one concrete location/detail."
    )

    # 5) Wellness post-security with details (critical)
    wellness_node = evaluator.add_leaf(
        id=f"Airport_{idx+1}_Wellness_PostSecurity_With_Details",
        desc=f"Describes either a yoga room/studio OR a rest & recharge area with seating and privacy partitions for Airport {idx+1} that is explicitly post-security and includes specific details.",
        parent=airport_node,
        critical=True
    )
    wellness_type = (airport.wellness_type or "").strip().lower()
    wellness_desc = (airport.wellness_desc or "").strip()
    # Construct claim to accept either yoga or rest_recharge w/ seating & privacy partitions
    claim_wellness = (
        f"For airport {code or 'UNKNOWN'}, the wellness amenity described is '{airport.wellness_type or ''}' with details: '{wellness_desc}'. "
        f"The description clearly indicates the amenity is in a post-security area. "
        f"It satisfies ONE of the following: "
        f"(a) a yoga room/studio, OR "
        f"(b) a rest & recharge area that has both seating AND privacy partitions. "
        f"At least one specific detail (e.g., terminal/concourse/gate/landmark) is present."
    )
    await evaluator.verify(
        claim=claim_wellness,
        node=wellness_node,
        additional_instruction="Accept synonyms for yoga room (yoga studio/space) and for rest & recharge (quiet room, lounge with partitions). For rest & recharge, both seating and privacy partitions must be present."
    )

    # 6) Reference URLs present and acceptable (critical)
    refs_leaf = evaluator.add_leaf(
        id=f"Airport_{idx+1}_Reference_URLs_Present_And_Acceptable",
        desc=f"Includes at least one reference URL for Airport {idx+1} from the airport’s official website or a trusted aviation source that supports amenity information.",
        parent=airport_node,
        critical=True
    )
    urls_list = airport.urls if airport.urls else []
    # Check at least one official/trusted source exists (verify-by-urls; pass if any is acceptable)
    claim_refs_accept = (
        f"Among these URLs for airport {code or 'UNKNOWN'}, at least one is an official airport website or a trusted aviation source that provides amenity information: {urls_list}."
    )
    await evaluator.verify(
        claim=claim_refs_accept,
        node=refs_leaf,
        sources=urls_list,
        additional_instruction=(
            "Judge each page individually: consider airport official domains or recognized aviation sources as 'trusted'. "
            "If the page includes airport amenity info (e.g., terminal services pages, facilities pages), treat it as acceptable support."
        )
    )

    # Additional fine-grained source-supported verifications for each amenity (critical)
    # These ensure that the provided URLs collectively support each amenity claim, not necessarily all on one page.
    refs_group = evaluator.add_parallel(
        id=f"Airport_{idx+1}_References_Verification",
        desc=f"Source-backed verification that the listed amenities for Airport {idx+1} are supported by the provided URLs.",
        parent=airport_node,
        critical=True
    )

    # Nursing supported by URLs
    nursing_src_leaf = evaluator.add_leaf(
        id=f"Airport_{idx+1}_Nursing_Supported_By_URLs",
        desc=f"Nursing/lactation amenity for Airport {idx+1} is supported by the provided URLs.",
        parent=refs_group,
        critical=True
    )
    claim_nursing_src = (
        f"The provided URLs collectively support that airport {code or 'UNKNOWN'} has a dedicated nursing/lactation facility "
        f"in post-security areas consistent with the description: '{nursing_desc}'."
    )
    await evaluator.verify(
        claim=claim_nursing_src,
        node=nursing_src_leaf,
        sources=urls_list,
        additional_instruction="It is acceptable if different URLs cover different aspects (existence/location). Verify explicit support for nursing/lactation in post-security."
    )

    # Pet relief supported by URLs
    pet_src_leaf = evaluator.add_leaf(
        id=f"Airport_{idx+1}_PetRelief_Supported_By_URLs",
        desc=f"Service animal/pet relief amenity for Airport {idx+1} is supported by the provided URLs.",
        parent=refs_group,
        critical=True
    )
    claim_pet_src = (
        f"The provided URLs collectively support that airport {code or 'UNKNOWN'} has service animal/pet relief area(s) "
        f"in post-security areas consistent with the description: '{pet_desc}'."
    )
    await evaluator.verify(
        claim=claim_pet_src,
        node=pet_src_leaf,
        sources=urls_list,
        additional_instruction="Confirm explicit mention of pet/service animal relief located behind/after security or airside. Location details help."
    )

    # Wellness supported by URLs
    wellness_src_leaf = evaluator.add_leaf(
        id=f"Airport_{idx+1}_Wellness_Supported_By_URLs",
        desc=f"Wellness amenity (yoga room or rest & recharge with seating and privacy partitions) for Airport {idx+1} is supported by the provided URLs.",
        parent=refs_group,
        critical=True
    )
    claim_wellness_src = (
        f"The provided URLs collectively support that airport {code or 'UNKNOWN'} has "
        f"{'a yoga room/studio' if wellness_type == 'yoga' else 'a rest & recharge area with seating and privacy partitions' if wellness_type == 'rest_recharge' else 'the described wellness amenity'} "
        f"in post-security areas consistent with the description: '{wellness_desc}'."
    )
    await evaluator.verify(
        claim=claim_wellness_src,
        node=wellness_src_leaf,
        sources=urls_list,
        additional_instruction=(
            "For yoga: look for explicit 'yoga room/studio'. "
            "For rest & recharge: the page must indicate both seating and privacy partitions, and post-security."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the US airports amenities task and return a structured summary.
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
        default_model=model
    )

    # Extract airports data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_airports(),
        template_class=AirportsExtraction,
        extraction_name="airports_extraction"
    )

    # Select the first 4 airports (padding if fewer)
    selected = select_first_k_airports(extraction.airports, k=4)

    # Add ground truth / constraints info
    evaluator.add_ground_truth({
        "allowed_top10_codes": ALLOWED_TOP10_CODES,
        "amenity_requirements": [
            "Dedicated nursing/lactation facility in post-security with specific details",
            "Service animal/pet relief area in post-security with specific details",
            "Either a yoga room/studio OR a rest & recharge area with seating AND privacy partitions, in post-security, with specific details"
        ],
        "exact_airport_count_required": 4
    }, gt_type="constraints")

    # Global requirements (critical)
    await verify_global_requirements(evaluator, root, selected)

    # Verify each airport (parallel under root)
    tasks = []
    for idx, airport in enumerate(selected):
        tasks.append(verify_airport(evaluator, root, airport, idx))
    await asyncio.gather(*tasks)

    # Return summary
    return evaluator.get_summary()