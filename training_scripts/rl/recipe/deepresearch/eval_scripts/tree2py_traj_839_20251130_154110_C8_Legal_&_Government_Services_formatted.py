import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nys_ag_regional_offices_5plus"
TASK_DESCRIPTION = """
Identify all regional offices of the New York State Attorney General that serve 5 or more counties. For each qualifying office, provide: (1) the office name and location city, (2) the complete street address including suite number, (3) the exact number of counties served, and (4) a complete list of all county names served by that office. Provide a reference URL supporting your information for each office.
"""

# Optional ground truth based on commonly referenced official listings
GROUND_TRUTH_OFFICES = {
    "Buffalo Regional Office": {
        "city": "Buffalo, NY",
        "count": "8",
        "counties": [
            "Allegany", "Cattaraugus", "Chautauqua", "Erie",
            "Genesee", "Niagara", "Orleans", "Wyoming"
        ],
        "address_example": "Main Place Tower, Suite 300A, 350 Main Street, Buffalo NY 14202"
    },
    "Binghamton Regional Office": {
        "city": "Binghamton, NY",
        "count": "8",
        "counties": [
            "Broome", "Chemung", "Chenango", "Delaware",
            "Otsego", "Schuyler", "Tioga", "Tompkins"
        ],
        "address_example": "State Office Building, 17th Floor, 44 Hawley Street, Binghamton NY 13901"
    },
    "Rochester Regional Office": {
        "city": "Rochester, NY",
        "count": "7",
        "counties": [
            "Livingston", "Monroe", "Ontario", "Seneca",
            "Steuben", "Wayne", "Yates"
        ],
        "address_example": "144 Exchange Blvd., Suite 200, Rochester NY 14614"
    },
    "Syracuse Regional Office": {
        "city": "Syracuse, NY",
        "count": "5",
        "counties": [
            "Cayuga", "Cortland", "Madison", "Oswego", "Onondaga"
        ],
        "address_example": "300 South State Street, Suite 300, Syracuse NY 13202"
    },
    "Utica Regional Office": {
        "city": "Utica, NY",
        "count": "6",
        "counties": [
            "Fulton", "Hamilton", "Herkimer", "Lewis", "Montgomery", "Oneida"
        ],
        "address_example": "207 Genesee Street, Room 508, Utica NY 13501"
    }
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OfficeInfo(BaseModel):
    office_name: Optional[str] = None  # e.g., "Buffalo Regional Office"
    location_city: Optional[str] = None  # e.g., "Buffalo, NY"
    street_address: Optional[str] = None  # full address including suite/floor if present
    counties_served_count: Optional[str] = None  # keep as string for robustness
    counties_served: List[str] = Field(default_factory=list)  # list of county names
    reference_urls: List[str] = Field(default_factory=list)  # one or more URLs supporting the info


class OfficesExtraction(BaseModel):
    buffalo: Optional[OfficeInfo] = None
    binghamton: Optional[OfficeInfo] = None
    rochester: Optional[OfficeInfo] = None
    syracuse: Optional[OfficeInfo] = None
    utica: Optional[OfficeInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_offices() -> str:
    return """
    Extract the regional office details for the New York State Attorney General that the answer claims serve 5 or more counties.
    Focus on the following five offices if they appear in the answer (any missing should be returned as null):
    - Buffalo Regional Office
    - Binghamton Regional Office
    - Rochester Regional Office
    - Syracuse Regional Office
    - Utica Regional Office

    For EACH office that is mentioned, extract the following fields exactly as stated in the answer:
    1) office_name: The office's name as written (e.g., "Buffalo Regional Office").
    2) location_city: The location city and state (e.g., "Buffalo, NY").
    3) street_address: The complete street address including suite/floor/room number if provided (e.g., "Main Place Tower, Suite 300A, 350 Main Street, Buffalo NY 14202").
    4) counties_served_count: The exact number of counties served (string).
    5) counties_served: An array of all county names served by the office. Include all counties listed by the answer. Do not invent or omit counties.
    6) reference_urls: All URLs the answer provides to support the office information (one or more). Extract only URLs explicitly present in the answer. Include full URLs with protocol.

    Return a JSON object with keys: buffalo, binghamton, rochester, syracuse, utica.
    Each key should contain an object with the specified fields for that office. If the office is not mentioned in the answer, set that key to null. If a specific field is missing for a mentioned office, set it to null or an empty list as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(x: Optional[str]) -> str:
    return x.strip() if isinstance(x, str) else ""


def _join_counties(counties: List[str]) -> str:
    cleaned = [c.strip() for c in counties if isinstance(c, str) and c.strip()]
    return ", ".join(cleaned)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_office(
    evaluator: Evaluator,
    parent_node,
    office_key: str,
    office_display_name: str,
    office_city_state: str,
    office_info: Optional[OfficeInfo],
) -> None:
    """
    Build verification nodes and run checks for a single regional office.
    """
    # Create office node (non-critical to allow partial credit across offices)
    office_node = evaluator.add_parallel(
        id=f"{office_display_name.replace(' ', '_')}",
        desc=f"{office_display_name} details are correct and complete per constraints.",
        parent=parent_node,
        critical=False
    )

    # Reference URL existence (critical prerequisite for evidence-backed checks)
    urls = office_info.reference_urls if office_info else []
    ref_url_exists = evaluator.add_custom_node(
        result=bool(urls),
        id=f"{office_key}_Reference_URL",
        desc="Provides a reference URL supporting the office information.",
        parent=office_node,
        critical=True
    )

    # Name and location city (critical)
    name_city_node = evaluator.add_leaf(
        id=f"{office_key}_Name_And_Location_City",
        desc=f"Provides the office name ({office_display_name}) AND the location city ({office_city_state}).",
        parent=office_node,
        critical=True
    )
    provided_name = _safe_str(office_info.office_name if office_info else None)
    provided_city = _safe_str(office_info.location_city if office_info else None)
    name_city_claim = (
        f"The New York State Attorney General regional office is identified as '{provided_name}' and is located in {provided_city}."
    )
    await evaluator.verify(
        claim=name_city_claim,
        node=name_city_node,
        sources=urls,
        additional_instruction=(
            f"Verify on the cited webpage(s) that the office's official name and its location city/state are as claimed."
            f" The expected canonical office for this check is '{office_display_name}' located in {office_city_state}."
            f" Allow minor phrasing or punctuation variations, but the meaning must match."
        ),
    )

    # Complete street address (critical)
    addr_node = evaluator.add_leaf(
        id=f"{office_key}_Complete_Street_Address",
        desc="Provides the complete street address including suite/floor/room number when applicable.",
        parent=office_node,
        critical=True
    )
    provided_address = _safe_str(office_info.street_address if office_info else None)
    addr_claim = (
        f"The office's complete street address (including unit details if applicable) is: '{provided_address}'."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=addr_node,
        sources=urls,
        additional_instruction=(
            "Confirm that the exact address string (including suite/floor/room details) is supported on the cited webpage(s). "
            "Allow minor formatting differences (commas, abbreviations like 'Ste.' vs 'Suite'), but the substantive content must match."
        ),
    )

    # Exact number of counties served (critical)
    count_node = evaluator.add_leaf(
        id=f"{office_key}_Exact_Number_Of_Counties_Served",
        desc="States the exact number of counties served.",
        parent=office_node,
        critical=True
    )
    provided_count = _safe_str(office_info.counties_served_count if office_info else None)
    count_claim = (
        f"This office serves exactly {provided_count} counties."
    )
    await evaluator.verify(
        claim=count_claim,
        node=count_node,
        sources=urls,
        additional_instruction=(
            "Check the cited webpage(s) to ensure the number of counties served equals the claimed number. "
            "If the page lists counties individually, counting them is acceptable to confirm the total."
        ),
    )

    # Complete county list (critical)
    counties_node = evaluator.add_leaf(
        id=f"{office_key}_Complete_County_List",
        desc="Lists all counties served (no omissions/incorrect additions).",
        parent=office_node,
        critical=True
    )
    provided_counties_list = office_info.counties_served if office_info else []
    counties_str = _join_counties(provided_counties_list)
    counties_claim = (
        f"This office serves the following counties: {counties_str}. The list is complete and contains no extra counties."
    )
    await evaluator.verify(
        claim=counties_claim,
        node=counties_node,
        sources=urls,
        additional_instruction=(
            "Verify that every listed county is supported by the cited webpage(s) and that no required county is missing. "
            "Treat county names case-insensitively and ignore minor punctuation differences. "
            "Order does not matter, but the set of counties must match the official listing for this office."
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the NYS Attorney General Regional Offices (serve >=5 counties) task.
    """
    # Initialize evaluator with parallel strategy (offices evaluated independently)
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

    # Extract office details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_offices(),
        template_class=OfficesExtraction,
        extraction_name="nys_ag_offices_5plus"
    )

    # Add ground truth info to summary (for transparency/debugging)
    evaluator.add_ground_truth({
        "expected_offices": GROUND_TRUTH_OFFICES,
        "note": "Ground truth included for reference; verification primarily checks claim support via provided URLs."
    })

    # Build verification nodes for each qualifying office in rubric
    offices_meta = [
        ("buffalo", "Buffalo Regional Office", "Buffalo, NY"),
        ("binghamton", "Binghamton Regional Office", "Binghamton, NY"),
        ("rochester", "Rochester Regional Office", "Rochester, NY"),
        ("syracuse", "Syracuse Regional Office", "Syracuse, NY"),
        ("utica", "Utica Regional Office", "Utica, NY"),
    ]

    for key, display_name, city_state in offices_meta:
        office_info: Optional[OfficeInfo] = getattr(extracted, key)
        await verify_office(
            evaluator=evaluator,
            parent_node=root,
            office_key=key,
            office_display_name=display_name,
            office_city_state=city_state,
            office_info=office_info
        )

    # Return structured summary with verification tree and score
    return evaluator.get_summary()