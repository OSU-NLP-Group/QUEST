import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "eclipse_nps_2024"
TASK_DESCRIPTION = (
    "On April 8, 2024, a total solar eclipse crossed North America. The path of totality passed over multiple "
    "National Park Service units. Identify: (1) how many total National Park Service units were crossed by the path "
    "of totality, and (2) the two U.S. national parks (specifically parks, not other types of NPS units such as "
    "national recreation areas or historic sites) that were located within the path of totality. For the national park "
    "located in Arkansas, provide: (a) the entrance fee for a private vehicle, (b) the name of the visitor center, and "
    "(c) the street address where the visitor center is located. For the national park located in Ohio, provide its name."
)

# Ground truth expectations per rubric
EXPECTED_TOTAL_UNITS = "27"
EXPECTED_PARKS = ["Hot Springs National Park", "Cuyahoga Valley National Park"]
AR_PARK_NAME = "Hot Springs National Park"
AR_EXPECTED_VEHICLE_FEE = "$0"
AR_EXPECTED_VISITOR_CENTER_NAME = "Fordyce Bathhouse Visitor Center"
AR_EXPECTED_VISITOR_CENTER_ADDRESS = "369 Central Avenue, Hot Springs, Arkansas"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ArkansasParkDetails(BaseModel):
    park_name: Optional[str] = None
    private_vehicle_fee: Optional[str] = None
    visitor_center_name: Optional[str] = None
    visitor_center_address: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EclipseNPSExtraction(BaseModel):
    total_nps_units_crossed: Optional[str] = None
    total_units_sources: List[str] = Field(default_factory=list)

    national_parks_in_path: List[str] = Field(default_factory=list)
    parks_sources: List[str] = Field(default_factory=list)

    arkansas_park: Optional[ArkansasParkDetails] = None

    ohio_park_name: Optional[str] = None
    ohio_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_eclipse_nps_info() -> str:
    return """
    Extract the following structured information from the answer. Return JSON with these fields:

    1) total_nps_units_crossed: The total count of National Park Service (NPS) units crossed by the April 8, 2024 path of totality. Return as string exactly as stated (digits or words, e.g., "27" or "twenty-seven"). If not provided, return null.
    2) total_units_sources: Array of all URLs cited in the answer that specifically support the total count of NPS units crossed. If none are cited, return [].

    3) national_parks_in_path: Array of the names of U.S. national parks (parks only) that the answer claims were in the path of totality. Do not include other NPS unit types (e.g., national recreation areas, monuments, historic sites). If none are mentioned, return [].
    4) parks_sources: Array of URLs cited in the answer that support which national parks were in the path of totality. If none are cited, return [].

    5) arkansas_park: Object with details for the Arkansas national park in the path. If the park is not mentioned, return null. Otherwise include:
       - park_name: The park's name.
       - private_vehicle_fee: The private vehicle entrance fee text exactly as stated (e.g., "$0", "no fee", etc.).
       - visitor_center_name: The visitor center name.
       - visitor_center_address: The street address for the visitor center.
       - sources: Array of URLs cited in the answer that support any of these Arkansas park details. If none, return [].

    6) ohio_park_name: The name of the national park in Ohio claimed to be in the path of totality (e.g., "Cuyahoga Valley National Park"). If not mentioned, return null.
    7) ohio_sources: Array of URLs cited in the answer supporting the Ohio park identification. If none, return [].

    Rules:
    - Extract only what is explicitly present in the answer.
    - For URL fields, include only actual URLs present in the answer (plain, markdown, etc.). If no URL provided, return [] for that URL field.
    - Keep strings exactly as in the answer; do not normalize or invent.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_total_units(evaluator: Evaluator, parent_node, extracted: EclipseNPSExtraction) -> None:
    # Create a critical parallel node for total count verification
    total_node = evaluator.add_parallel(
        id="Total_NPS_Units_Crossed",
        desc="Correctly identifies how many total NPS units were crossed by the path of totality (27)",
        parent=parent_node,
        critical=True
    )

    # Existence gate: require a stated count and at least one supporting URL
    exist_total = evaluator.add_custom_node(
        result=bool(extracted.total_nps_units_crossed and extracted.total_nps_units_crossed.strip()) and bool(extracted.total_units_sources),
        id="Total_NPS_Units_Crossed_Exists",
        desc="Total NPS units count and supporting source(s) are provided",
        parent=total_node,
        critical=True
    )

    # Value match to 27 (simple, tolerant to words vs digits)
    total_match = evaluator.add_leaf(
        id="Total_NPS_Units_Crossed_Match_27",
        desc="The reported total equals 27",
        parent=total_node,
        critical=True
    )
    stated = extracted.total_nps_units_crossed or ""
    await evaluator.verify(
        claim=f"The reported total number of National Park Service units crossed by the April 8, 2024 eclipse path is '{stated}', and this equals 27.",
        node=total_match,
        additional_instruction="Interpret numbers expressed in digits or words (e.g., '27' vs 'twenty-seven'). Minor punctuation/formatting should not affect the equality."
    )

    # Source support: verify by cited URLs
    total_supported = evaluator.add_leaf(
        id="Total_NPS_Units_Crossed_Supported_By_Sources",
        desc="Sources support that the eclipse path crossed 27 NPS units in total",
        parent=total_node,
        critical=True
    )
    await evaluator.verify(
        claim="The path of totality of the April 8, 2024 solar eclipse crossed 27 National Park Service units in total.",
        node=total_supported,
        sources=extracted.total_units_sources,
        additional_instruction="Confirm the total count (27) using the provided sources."
    )


async def verify_parks_in_path(evaluator: Evaluator, parent_node, extracted: EclipseNPSExtraction) -> None:
    parks_node = evaluator.add_parallel(
        id="National_Parks_in_Path_of_Totality",
        desc="Correctly identifies exactly the two U.S. national parks (parks only) located within the path of totality: Hot Springs National Park (Arkansas) and Cuyahoga Valley National Park (Ohio), and no others",
        parent=parent_node,
        critical=True
    )

    # Existence gate: require at least two parks listed and at least one supporting URL
    exist_parks = evaluator.add_custom_node(
        result=(len(extracted.national_parks_in_path) >= 2) and bool(extracted.parks_sources),
        id="National_Parks_List_And_Sources_Exist",
        desc="National parks list and supporting source(s) are provided",
        parent=parks_node,
        critical=True
    )

    # Exact parks check (answer correctness; simple verify)
    parks_exact = evaluator.add_leaf(
        id="National_Parks_List_Exact",
        desc="Exactly the two national parks are Hot Springs National Park and Cuyahoga Valley National Park",
        parent=parks_node,
        critical=True
    )
    parks_list_str = ", ".join(extracted.national_parks_in_path) if extracted.national_parks_in_path else ""
    await evaluator.verify(
        claim=(
            f"Based on the extracted list of national parks in the path of totality: [{parks_list_str}], "
            "there are exactly two items and they are Hot Springs National Park (Arkansas) and "
            "Cuyahoga Valley National Park (Ohio), with no additional national parks."
        ),
        node=parks_exact,
        additional_instruction=(
            "This check is strictly about U.S. national parks. Ignore other NPS unit types. "
            "Allow minor naming variations such as 'NP' vs 'National Park' or case differences."
        )
    )

    # Source support for Hot Springs NP being in the path
    hs_supported = evaluator.add_leaf(
        id="Hot_Springs_NP_In_Path_Supported",
        desc="Hot Springs National Park was within the path of totality (supported by sources)",
        parent=parks_node,
        critical=True
    )
    await evaluator.verify(
        claim="Hot Springs National Park (Arkansas) was within the path of totality on April 8, 2024.",
        node=hs_supported,
        sources=extracted.parks_sources,
        additional_instruction="Verify using the provided sources that Hot Springs National Park lay within the April 8, 2024 eclipse path of totality."
    )

    # Source support for Cuyahoga Valley NP being in the path
    cv_supported = evaluator.add_leaf(
        id="Cuyahoga_Valley_NP_In_Path_Supported",
        desc="Cuyahoga Valley National Park was within the path of totality (supported by sources)",
        parent=parks_node,
        critical=True
    )
    await evaluator.verify(
        claim="Cuyahoga Valley National Park (Ohio) was within the path of totality on April 8, 2024.",
        node=cv_supported,
        sources=extracted.parks_sources,
        additional_instruction="Verify using the provided sources that Cuyahoga Valley National Park lay within the April 8, 2024 eclipse path of totality."
    )


async def verify_arkansas_details(evaluator: Evaluator, parent_node, extracted: EclipseNPSExtraction) -> None:
    ar_node = evaluator.add_parallel(
        id="Arkansas_Park_Details",
        desc="For Hot Springs National Park (Arkansas), provides the required entrance-fee and visitor-center details",
        parent=parent_node,
        critical=True
    )

    ar = extracted.arkansas_park

    # Existence gate: require park name, fee, visitor center name, address, and at least one source
    exist_ar = evaluator.add_custom_node(
        result=(
            ar is not None and
            bool(ar.park_name and ar.park_name.strip()) and
            bool(ar.private_vehicle_fee and ar.private_vehicle_fee.strip()) and
            bool(ar.visitor_center_name and ar.visitor_center_name.strip()) and
            bool(ar.visitor_center_address and ar.visitor_center_address.strip()) and
            bool(ar.sources)
        ),
        id="Arkansas_Park_Details_Exist",
        desc="Arkansas park details and supporting source(s) are provided",
        parent=ar_node,
        critical=True
    )

    # Entrance fee value match
    fee_match = evaluator.add_leaf(
        id="Entrance_Fee_Private_Vehicle_AR",
        desc="Provides the entrance fee for a private vehicle for Hot Springs National Park (no fee / $0)",
        parent=ar_node,
        critical=True
    )
    fee_str = ar.private_vehicle_fee if ar and ar.private_vehicle_fee else ""
    await evaluator.verify(
        claim=f"The extracted private vehicle entrance fee for Hot Springs National Park is '{fee_str}', which indicates $0 (no fee).",
        node=fee_match,
        additional_instruction="Treat phrases like 'no fee', 'no entrance fee', 'free', or '$0' as equivalent to $0 for a private vehicle entrance fee."
    )

    # Entrance fee source support
    fee_supported = evaluator.add_leaf(
        id="Entrance_Fee_Private_Vehicle_AR_Supported",
        desc="Sources support that the private vehicle entrance fee for Hot Springs National Park is $0 (no fee)",
        parent=ar_node,
        critical=True
    )
    await evaluator.verify(
        claim="Hot Springs National Park charges $0 (no fee) for private vehicle entrance.",
        node=fee_supported,
        sources=ar.sources if ar else [],
        additional_instruction="Verify the fee statement using the provided sources; accept explicit statements of 'no entrance fee' as equivalent to $0."
    )

    # Visitor center name match
    vc_name_match = evaluator.add_leaf(
        id="Visitor_Center_Name_AR",
        desc="Provides the visitor center name for Hot Springs National Park (Fordyce Bathhouse Visitor Center)",
        parent=ar_node,
        critical=True
    )
    vc_name = ar.visitor_center_name if ar and ar.visitor_center_name else ""
    await evaluator.verify(
        claim=f"The visitor center name for Hot Springs National Park is '{vc_name}', and it matches 'Fordyce Bathhouse Visitor Center'.",
        node=vc_name_match,
        additional_instruction="Allow minor variations such as inclusion of 'and Museum' or punctuation differences. The core name should match 'Fordyce Bathhouse Visitor Center'."
    )

    # Visitor center name source support
    vc_name_supported = evaluator.add_leaf(
        id="Visitor_Center_Name_AR_Supported",
        desc="Sources support that the visitor center is the Fordyce Bathhouse Visitor Center",
        parent=ar_node,
        critical=True
    )
    await evaluator.verify(
        claim="Hot Springs National Park's visitor center is the Fordyce Bathhouse Visitor Center.",
        node=vc_name_supported,
        sources=ar.sources if ar else [],
        additional_instruction="Confirm the official visitor center name using the provided sources."
    )

    # Visitor center address match
    vc_addr_match = evaluator.add_leaf(
        id="Visitor_Center_Street_Address_AR",
        desc="Provides the street address where the visitor center is located (369 Central Avenue, Hot Springs, Arkansas)",
        parent=ar_node,
        critical=True
    )
    vc_addr = ar.visitor_center_address if ar and ar.visitor_center_address else ""
    await evaluator.verify(
        claim=f"The visitor center street address is '{vc_addr}', and it matches '369 Central Avenue, Hot Springs, Arkansas'.",
        node=vc_addr_match,
        additional_instruction="Allow minor variations such as 'Ave' vs 'Avenue', inclusion of city/state abbreviations (e.g., 'Hot Springs, AR') or ZIP code."
    )

    # Visitor center address source support
    vc_addr_supported = evaluator.add_leaf(
        id="Visitor_Center_Street_Address_AR_Supported",
        desc="Sources support that the visitor center address is 369 Central Avenue, Hot Springs, Arkansas",
        parent=ar_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Fordyce Bathhouse Visitor Center (Hot Springs National Park) is located at 369 Central Avenue, Hot Springs, Arkansas.",
        node=vc_addr_supported,
        sources=ar.sources if ar else [],
        additional_instruction="Confirm the street address of the visitor center using the provided sources."
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
    # Initialize evaluator and root
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

    # Create a critical main rubric node under root (since root is non-critical by design)
    main = evaluator.add_parallel(
        id="Root",
        desc="Provides all required eclipse path-of-totality / NPS information from the question",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_eclipse_nps_info(),
        template_class=EclipseNPSExtraction,
        extraction_name="eclipse_nps_extraction"
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_total_nps_units_crossed": EXPECTED_TOTAL_UNITS,
        "expected_national_parks_in_path": EXPECTED_PARKS,
        "arkansas_expected_vehicle_fee": AR_EXPECTED_VEHICLE_FEE,
        "arkansas_expected_visitor_center_name": AR_EXPECTED_VISITOR_CENTER_NAME,
        "arkansas_expected_visitor_center_address": AR_EXPECTED_VISITOR_CENTER_ADDRESS,
    }, gt_type="rubric_expectations")

    # Build verification subtrees
    await verify_total_units(evaluator, main, extracted)
    await verify_parks_in_path(evaluator, main, extracted)
    await verify_arkansas_details(evaluator, main, extracted)

    # Return evaluation summary
    return evaluator.get_summary()