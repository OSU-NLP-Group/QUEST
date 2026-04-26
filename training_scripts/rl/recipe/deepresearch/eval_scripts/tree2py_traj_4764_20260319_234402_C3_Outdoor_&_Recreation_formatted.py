import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "canary_islands_winter_2026_teide_ecotax_and_gc_route"
TASK_DESCRIPTION = """
A non-resident outdoor enthusiast is planning a winter 2026 trip to the Canary Islands. They want to complete a specific pilgrimage-style walking route on Gran Canaria that starts at Maspalomas Lighthouse and ends at a historic church dedicated to Santiago in a northern town, where they can obtain their completion certificate. After finishing this trek, they plan to travel to Tenerife to hike the Teide National Park summit trail that requires the highest ecotax fee for non-residents under the new 2026 fee system. What is the exact date (in DD Month YYYY format, e.g., "19 January 2026") when the Teide ecotax system was implemented, and what is the name of the official online platform they must use to book their Teide trail permit?
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class GranCanariaRouteExtraction(BaseModel):
    distance_text: Optional[str] = None
    start_point: Optional[str] = None
    end_point: Optional[str] = None
    certificate_mentioned: Optional[bool] = None
    route_urls: List[str] = Field(default_factory=list)


class TeideInfoExtraction(BaseModel):
    trail_name: Optional[str] = None
    nonresident_fee_text: Optional[str] = None
    highest_fee_explicit: Optional[bool] = None
    implementation_date_text: Optional[str] = None
    platform_name: Optional[str] = None
    teide_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_gran_canaria_route() -> str:
    return """
    Extract, from the answer text only, the details about the Gran Canaria pilgrimage-style walking route.

    Return a JSON object with:
    - distance_text: the route's total distance as written in the answer (e.g., "67 km" or "67 kilometres"); null if not stated
    - start_point: the route's start location as stated (e.g., "Faro de Maspalomas" or "Maspalomas Lighthouse"); null if not stated
    - end_point: the route's end location as stated (e.g., "Church of Santiago de Los Caballeros in Gáldar"); null if not stated
    - certificate_mentioned: true if the answer explicitly mentions that a completion certificate (credential/certificado/compostela) is obtainable; false if explicitly denied; null if not mentioned
    - route_urls: all URLs in the answer that are specifically associated with this Gran Canaria route information (distance, start/end, certificate). Include only URLs explicitly present in the answer.

    If any field is missing, set it to null. Do not invent information.
    """


def prompt_extract_teide_info() -> str:
    return """
    Extract, from the answer text only, the details about the Teide National Park summit trail and ecotax.

    Return a JSON object with:
    - trail_name: the specific summit trail identified (e.g., "PNT 10", "Sendero 10", "Telesforo Bravo"); null if not stated
    - nonresident_fee_text: the non-resident ecotax fee amount for that trail as written (e.g., "€15", "15 EUR"); null if not stated
    - highest_fee_explicit: true if the answer explicitly claims this is the highest ecotax fee for non-residents among Teide trails; false if explicitly denied; null if not mentioned
    - implementation_date_text: the date string the answer uses for the 2026 Teide ecotax system implementation (e.g., "19 January 2026" or "January 19, 2026"); null if not stated
    - platform_name: the name of the official online platform that must be used to book the Teide trail permit (e.g., "Tenerife ON"); null if not stated
    - teide_urls: all URLs in the answer that are specifically associated with Teide trail, ecotax fee, date, or booking platform.

    If any field is missing, set it to null. Do not invent information.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_dd_month_yyyy(date_str: Optional[str]) -> bool:
    if not date_str:
        return False
    pattern = r"^\s*(0[1-9]|[12][0-9]|3[01])\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}\s*$"
    return re.match(pattern, date_str.strip(), flags=re.IGNORECASE) is not None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_gc_constraints_nodes(
    evaluator: Evaluator,
    parent_node,
) -> None:
    gc_node = evaluator.add_parallel(
        id="Gran_Canaria_Route_Constraints",
        desc="Describe the Gran Canaria pilgrimage-style route consistent with the constraints (distance, start/end points, certificate availability).",
        parent=parent_node,
        critical=True,
    )

    # Route total distance stated as 67 km
    n_dist = evaluator.add_leaf(
        id="Route_Total_Distance_67km",
        desc="States the walking route total distance is 67 km.",
        parent=gc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the Gran Canaria pilgrimage-style walking route total distance is explicitly given as 67 km (accept small variants like '67km' or '67 kilometres').",
        node=n_dist,
        additional_instruction="Focus only on whether the answer text itself clearly says 67 km for the route. If the answer gives a different number or a range, mark incorrect.",
    )

    # Route start at Faro de Maspalomas
    n_start = evaluator.add_leaf(
        id="Route_Start_Faro_de_Maspalomas",
        desc="States the route starts at Faro de Maspalomas (Maspalomas Lighthouse).",
        parent=gc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the route start point is explicitly stated as Faro de Maspalomas (Maspalomas Lighthouse).",
        node=n_start,
        additional_instruction="Accept synonyms and bilingual variants like 'Faro de Maspalomas' or 'Maspalomas Lighthouse'.",
    )

    # Route ends at Church of Santiago de Los Caballeros in Gáldar
    n_end = evaluator.add_leaf(
        id="Route_End_Church_Santiago_de_Los_Caballeros_Galdar",
        desc="States the route ends at the Church of Santiago de Los Caballeros in Gáldar.",
        parent=gc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the route endpoint is explicitly given as the Church of Santiago de Los Caballeros in Gáldar.",
        node=n_end,
        additional_instruction="Accept close variants such as 'Iglesia/Basílica de Santiago (de los Caballeros) en Gáldar' or 'Church of Santiago in Gáldar'.",
    )

    # Pilgrimage-style and certificate obtainable
    n_cert = evaluator.add_leaf(
        id="Pilgrimage_Style_With_Certificate",
        desc="States it is a pilgrimage-style route where a completion certificate is obtainable.",
        parent=gc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the route is described as a pilgrimage-style route and it is stated that a completion certificate (credential completion certificate) can be obtained.",
        node=n_cert,
        additional_instruction="Accept terms like 'pilgrimage', 'peregrinación', 'pilgrim route', and certificate terms such as 'certificate of completion', 'certificado', 'credencial' leading to a certificate.",
    )


async def build_teide_trail_nodes(
    evaluator: Evaluator,
    parent_node,
) -> None:
    teide_trail_node = evaluator.add_parallel(
        id="Teide_Trail_With_Highest_NonResident_Ecotax",
        desc="Correctly identifies the Teide summit trail referenced as the one with the highest non-resident ecotax fee under the 2026 system.",
        parent=parent_node,
        critical=True,
    )

    # Identify trail as PNT 10 (Telesforo Bravo)
    n_trail = evaluator.add_leaf(
        id="Trail_Is_PNT10_Telesforo_Bravo",
        desc="Identifies the relevant Teide summit trail as PNT 10 (Telesforo Bravo).",
        parent=teide_trail_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the Teide summit trail with the highest non-resident ecotax is identified as PNT 10 (Telesforo Bravo).",
        node=n_trail,
        additional_instruction="Accept variants such as 'Sendero 10', 'Route 10', 'PNT-10', and inclusion of 'Telesforo Bravo'.",
    )

    # State non-resident ecotax fee is €15
    n_fee = evaluator.add_leaf(
        id="Trail_NonResident_Fee_Is_15_EUR",
        desc="States the non-resident ecotax fee for PNT 10 is €15.",
        parent=teide_trail_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the non-resident ecotax fee for PNT 10 is stated as 15 euros.",
        node=n_fee,
        additional_instruction="Accept common notations like '€15', '15€', 'EUR 15', '15 EUR'. If a different amount is given, mark incorrect.",
    )

    # Explicitly states highest non-resident fee among Teide trails
    n_highest = evaluator.add_leaf(
        id="Explicitly_States_Highest_NonResident_Fee",
        desc="Explicitly states that this trail corresponds to the highest ecotax fee for non-residents among Teide trails.",
        parent=teide_trail_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, it is explicitly stated that this Teide summit trail carries the highest ecotax fee for non-residents among Teide trails.",
        node=n_highest,
        additional_instruction="Look for explicit phrases like 'highest', 'most expensive', or 'top tier' fee for non-residents compared to other Teide trails.",
    )


async def build_teide_date_and_platform_nodes(
    evaluator: Evaluator,
    parent_node,
    teide_info: TeideInfoExtraction,
) -> None:
    # Implementation date subtree
    date_node = evaluator.add_parallel(
        id="Teide_Ecotax_Implementation_Date",
        desc="Provide the Teide ecotax system implementation date and ensure it is formatted as required.",
        parent=parent_node,
        critical=True,
    )

    # Exact value given as 19 January 2026 (or January 19, 2026)
    n_date_exact = evaluator.add_leaf(
        id="Implementation_Date_Exact_Value",
        desc="Gives the implementation date value as 19 January 2026 (January 19, 2026).",
        parent=date_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the Teide ecotax system implementation date is given as 19 January 2026 (accept also 'January 19, 2026').",
        node=n_date_exact,
        additional_instruction="Focus strictly on what the answer states. Accept either '19 January 2026' or 'January 19, 2026'.",
    )

    # Format check DD Month YYYY
    date_string = teide_info.implementation_date_text
    format_ok = is_dd_month_yyyy(date_string)
    evaluator.add_custom_node(
        result=format_ok,
        id="Implementation_Date_Format_DD_Month_YYYY",
        desc='Formats the implementation date in DD Month YYYY format (e.g., "19 January 2026").',
        parent=date_node,
        critical=True,
    )

    # Official booking platform subtree
    platform_node = evaluator.add_parallel(
        id="Teide_Official_Booking_Platform",
        desc="Provide the official online platform required for Teide trail permit reservations.",
        parent=parent_node,
        critical=True,
    )

    n_platform = evaluator.add_leaf(
        id="Platform_Name_Tenerife_ON",
        desc='Gives the official booking platform name as "Tenerife ON".',
        parent=platform_node,
        critical=True,
    )
    await evaluator.verify(
        claim='In the answer, the name of the official online platform required to book the Teide trail permit is "Tenerife ON".',
        node=n_platform,
        additional_instruction="Allow minor casing or punctuation variants (e.g., 'TenerifeON'). The key idea: platform is explicitly named Tenerife ON.",
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

    # Add a top-level critical node to mirror the rubric "Task_Completion"
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Satisfy all stated constraints and provide the requested outputs (Teide ecotax implementation date in DD Month YYYY format and the official booking platform name).",
        parent=root,
        critical=True,
    )

    # Perform extractions (recorded in summary)
    gc_extraction, teide_extraction = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_gran_canaria_route(),
            template_class=GranCanariaRouteExtraction,
            extraction_name="gran_canaria_route_extraction",
        ),
        evaluator.extract(
            prompt=prompt_extract_teide_info(),
            template_class=TeideInfoExtraction,
            extraction_name="teide_info_extraction",
        ),
    )

    # Build verification trees according to rubric
    await build_gc_constraints_nodes(evaluator, task_node)
    await build_teide_trail_nodes(evaluator, task_node)
    await build_teide_date_and_platform_nodes(evaluator, task_node, teide_extraction)

    # Add ground truth info for reference
    evaluator.add_ground_truth({
        "gran_canaria": {
            "expected_distance": "67 km",
            "expected_start": "Faro de Maspalomas (Maspalomas Lighthouse)",
            "expected_end": "Church of Santiago de Los Caballeros in Gáldar",
            "certificate_obtainable": True,
        },
        "teide": {
            "expected_trail": "PNT 10 (Telesforo Bravo)",
            "expected_nonresident_fee": "€15",
            "ecotax_implementation_date": "19 January 2026",
            "official_platform": "Tenerife ON",
            "note": "Highest non-resident ecotax fee among Teide trails should be explicitly stated.",
        }
    }, gt_type="expected_answers")

    return evaluator.get_summary()