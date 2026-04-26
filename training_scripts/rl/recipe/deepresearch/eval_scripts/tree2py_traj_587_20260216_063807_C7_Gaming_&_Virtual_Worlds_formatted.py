import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "esports_venue_tx_a_criteria"
TASK_DESCRIPTION = (
    "Identify the esports venue in the United States that satisfies ALL of the following criteria: "
    "(1) Located in the state of Texas, "
    "(2) Located in a city whose name begins with the letter 'A', "
    "(3) Has a seating capacity between 2,000 and 3,000 people, "
    "(4) Has at least 90,000 square feet of total space, "
    "(5) Is specifically dedicated to esports as its primary purpose, "
    "(6) Opened between 2018 and 2019, "
    "(7) Was converted or renovated from existing convention center space, "
    "(8) Is owned and operated by the city government, "
    "(9) Has hosted major esports championships or tournaments, "
    "(10) Offers flexible seating configurations that can be adjusted for different event sizes, "
    "(11) Had a construction or renovation cost between 8 and 12 million dollars, "
    "(12) Is located within the Dallas-Fort Worth metropolitan area. "
    "Provide the name of the venue and a reference URL confirming these specifications."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """Structured info extracted from the agent's answer about the venue."""
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity_text: Optional[str] = None
    square_footage_text: Optional[str] = None
    opening_year_text: Optional[str] = None
    cost_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    Extract the key information about the esports venue identified in the answer.

    Return the following fields:
    - venue_name: The name of the esports venue identified.
    - city: The city where the venue is located (just the city name).
    - state: The state where the venue is located (full name or 2-letter code).
    - capacity_text: Any text describing the seating capacity (exact number or phrase).
    - square_footage_text: Any text describing total square footage (e.g., "100,000 square feet").
    - opening_year_text: Any text describing the opening year (e.g., "opened in 2018").
    - cost_text: Any text describing the construction or renovation cost (e.g., "$10 million").
    - source_urls: All URLs explicitly provided in the answer as references supporting the venue details. 
      Include only valid http/https URLs. If links are in markdown format, extract the actual URL.

    Important:
    - Do not invent information. If a field is not mentioned in the answer, set it to null (or empty list for source_urls).
    - For source_urls, include every URL that appears to support the venue info. If none, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _name_prefix(extracted: VenueExtraction) -> str:
    """Build a human-readable subject for claims."""
    if extracted.venue_name and extracted.venue_name.strip():
        return f"The esports venue '{extracted.venue_name.strip()}'"
    return "The esports venue"


def _city_phrase(extracted: VenueExtraction) -> str:
    if extracted.city and extracted.state:
        return f"in {extracted.city}, {extracted.state}"
    if extracted.city:
        return f"in {extracted.city}"
    return ""


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_venue(evaluator: Evaluator, parent_node, extracted: VenueExtraction) -> None:
    """
    Build the verification tree according to the rubric and perform checks.
    """
    # Create the main aggregation node as per rubric
    main_node = evaluator.add_parallel(
        id="Esports_Venue_Identification",
        desc="Identification of an esports venue in the United States that meets all specified criteria",
        parent=parent_node,
        critical=False  # Root aggregation is non-critical to allow proper gate-then-average at higher level
    )

    # Create a gating node for reference URL presence (critical), so downstream verifications get skipped if absent
    has_url = bool(extracted.source_urls)
    evaluator.add_custom_node(
        result=has_url,
        id="Reference_URL",
        desc="A reference URL is provided confirming the venue's specifications",
        parent=main_node,
        critical=True
    )

    # Build claims based on extracted info (name if available)
    subject = _name_prefix(extracted)
    city_phrase = _city_phrase(extracted)
    urls = extracted.source_urls if extracted.source_urls else None  # None will route to simple verify (not ideal)

    # Prepare all leaf nodes and corresponding claims
    leaves_and_claims: List[tuple] = []

    # 1) Located in Texas
    node_tx = evaluator.add_leaf(
        id="Located_in_Texas",
        desc="The venue is located in the state of Texas",
        parent=main_node,
        critical=True
    )
    claim_tx = f"{subject} is located in the state of Texas."
    leaves_and_claims.append((
        claim_tx,
        urls,
        node_tx,
        "Verify that the venue's address or location indicates it is in Texas (accept 'TX' as Texas)."
    ))

    # 2) City name starts with 'A'
    node_city_a = evaluator.add_leaf(
        id="City_Name_Starts_A",
        desc="The venue is located in a city whose name begins with the letter 'A'",
        parent=main_node,
        critical=True
    )
    if extracted.city and extracted.city.strip():
        claim_city_a = f"{subject} is located in the city of {extracted.city.strip()}, whose name begins with the letter 'A'."
    else:
        claim_city_a = f"{subject} is located in a city whose name begins with the letter 'A'."
    leaves_and_claims.append((
        claim_city_a,
        urls,
        node_city_a,
        "Check the venue's city name; accept examples like Arlington or Austin. The city name must begin with 'A'."
    ))

    # 3) Seating capacity between 2,000 and 3,000
    node_capacity = evaluator.add_leaf(
        id="Seating_Capacity_Range",
        desc="The venue has a seating capacity between 2,000 and 3,000 people",
        parent=main_node,
        critical=True
    )
    claim_capacity = f"{subject} has a seating capacity between 2,000 and 3,000 people."
    leaves_and_claims.append((
        claim_capacity,
        urls,
        node_capacity,
        "Verify that the capacity stated on the page is within the inclusive range [2000, 3000]; "
        "allow phrasing like 'up to 2,500' or 'approximately 2,500'."
    ))

    # 4) At least 90,000 square feet
    node_sqft = evaluator.add_leaf(
        id="Square_Footage_Minimum",
        desc="The venue has at least 90,000 square feet of total space",
        parent=main_node,
        critical=True
    )
    claim_sqft = f"{subject} has at least 90,000 square feet of total space."
    leaves_and_claims.append((
        claim_sqft,
        urls,
        node_sqft,
        "Check total area; values like 100,000 sq ft satisfy this requirement. Accept minor formatting variants like 'sf' or 'sq. ft.'."
    ))

    # 5) Dedicated to esports as primary purpose
    node_dedicated = evaluator.add_leaf(
        id="Dedicated_Esports",
        desc="The venue is specifically dedicated to esports as its primary purpose",
        parent=main_node,
        critical=True
    )
    claim_dedicated = f"{subject} is specifically dedicated to esports as its primary purpose."
    leaves_and_claims.append((
        claim_dedicated,
        urls,
        node_dedicated,
        "Look for wording like 'dedicated esports facility', 'purpose-built for esports', or similar phrases."
    ))

    # 6) Opened between 2018 and 2019
    node_open_year = evaluator.add_leaf(
        id="Opening_Year_Range",
        desc="The venue opened between 2018 and 2019",
        parent=main_node,
        critical=True
    )
    claim_open_year = f"{subject} opened between 2018 and 2019 (inclusive)."
    leaves_and_claims.append((
        claim_open_year,
        urls,
        node_open_year,
        "Confirm the opening year is 2018 or 2019."
    ))

    # 7) Converted or renovated from existing convention center space
    node_converted = evaluator.add_leaf(
        id="Converted_From_Convention_Center",
        desc="The venue was converted or renovated from existing convention center space",
        parent=main_node,
        critical=True
    )
    claim_converted = f"{subject} was converted or renovated from existing convention center space."
    leaves_and_claims.append((
        claim_converted,
        urls,
        node_converted,
        "Look for explicit mention of conversion/renovation of a convention center hall/space into the esports venue."
    ))

    # 8) City-owned and operated
    node_city_owned = evaluator.add_leaf(
        id="City_Owned",
        desc="The venue is owned and operated by the city government",
        parent=main_node,
        critical=True
    )
    claim_city_owned = f"{subject} is owned and operated by the city government."
    leaves_and_claims.append((
        claim_city_owned,
        urls,
        node_city_owned,
        "Verify that the owner and operator are the city government (e.g., City of Arlington). "
        "If only 'owned by city' but not operated by city is stated, this should be considered not fully supported."
    ))

    # 9) Hosted major esports championships/tournaments
    node_hosted = evaluator.add_leaf(
        id="Hosted_Major_Championships",
        desc="The venue has hosted major esports championships or tournaments",
        parent=main_node,
        critical=True
    )
    claim_hosted = f"{subject} has hosted major esports championships or tournaments."
    leaves_and_claims.append((
        claim_hosted,
        urls,
        node_hosted,
        "Check for examples of significant events (e.g., world championships, large-scale LANs, top-tier league finals)."
    ))

    # 10) Flexible seating configurations
    node_flex = evaluator.add_leaf(
        id="Flexible_Seating",
        desc="The venue offers flexible seating configurations that can be adjusted for different event sizes",
        parent=main_node,
        critical=True
    )
    claim_flex = f"{subject} offers flexible seating configurations that can be adjusted for different event sizes."
    leaves_and_claims.append((
        claim_flex,
        urls,
        node_flex,
        "Look for terms like 'reconfigurable seating', 'modular seating', 'flexible layout', or similar wording."
    ))

    # 11) Investment (cost) between $8M and $12M
    node_invest = evaluator.add_leaf(
        id="Investment_Range",
        desc="The venue's construction or renovation cost was between 8 and 12 million dollars",
        parent=main_node,
        critical=True
    )
    claim_invest = f"{subject}'s construction or renovation cost was between $8 million and $12 million."
    leaves_and_claims.append((
        claim_invest,
        urls,
        node_invest,
        "Confirm that the reported cost lies within [8, 12] million USD (e.g., '$10 million')."
    ))

    # 12) Within Dallas–Fort Worth metropolitan area
    node_dfw = evaluator.add_leaf(
        id="Dallas_Metropolitan_Area",
        desc="The venue is located within the Dallas-Fort Worth metropolitan area",
        parent=main_node,
        critical=True
    )
    claim_dfw = f"{subject} is located within the Dallas–Fort Worth metropolitan area."
    leaves_and_claims.append((
        claim_dfw,
        urls,
        node_dfw,
        "If the venue is in Arlington or another city recognized as part of DFW, this is supported."
    ))

    # Execute all verifications in parallel
    await evaluator.batch_verify(leaves_and_claims)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
) -> Dict:
    """
    Evaluate an answer for the esports venue identification task.
    """
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

    # Extract the venue info and references from the answer
    extracted: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # Add some custom info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "extracted_name": extracted.venue_name,
            "extracted_city": extracted.city,
            "extracted_state": extracted.state,
            "num_sources": len(extracted.source_urls)
        },
        info_type="extraction_summary",
        info_name="extraction_overview"
    )

    # Build tree and verify
    await build_and_verify_venue(evaluator, root, extracted)

    return evaluator.get_summary()