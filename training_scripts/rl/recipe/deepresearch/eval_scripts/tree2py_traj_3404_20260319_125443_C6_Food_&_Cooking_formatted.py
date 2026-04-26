import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "md_chain_restaurant_compliance"
TASK_DESCRIPTION = """
Identify a chain restaurant that operates in Maryland and meets ALL of the following requirements:

1. The restaurant must be a national chain with multiple locations across different states
2. The restaurant must operate 24 hours a day, 7 days a week, 365 days a year
3. The restaurant must remain open on both Thanksgiving Day and Christmas Day
4. The Maryland location must serve alcoholic beverages
5. The Maryland location must have a seating capacity of at least 50 people
6. The restaurant must meet ADA accessibility requirements with at least 5% of seating being wheelchair accessible (minimum 1 table)
7. The Maryland location must have outdoor seating available
8. The outdoor seating must maintain proper clearance requirements (minimum 5 feet from driveways, alleys, and handicap ramps)

For your answer, provide:
- The name of the restaurant chain
- The specific address of at least one Maryland location that meets all requirements
- Documentation (with reference URLs) confirming:
  - The chain's 24/7/365 operating policy
  - Holiday operating hours (Thanksgiving and Christmas)
  - Alcohol service at the Maryland location
  - Seating capacity of 50 or more
  - ADA accessibility features
  - Outdoor seating availability
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ChainLocationExtraction(BaseModel):
    """
    Structured extraction from the agent's answer.
    Only include URLs that are explicitly present in the answer.
    """
    chain_name: Optional[str] = None
    md_location_address: Optional[str] = None

    # Evidence URLs explicitly cited in the answer
    url_chain_multistate: List[str] = Field(default_factory=list)
    url_md_location: List[str] = Field(default_factory=list)

    url_247: List[str] = Field(default_factory=list)
    url_holiday: List[str] = Field(default_factory=list)
    url_alcohol_md: List[str] = Field(default_factory=list)
    url_seating_capacity: List[str] = Field(default_factory=list)
    url_ada: List[str] = Field(default_factory=list)
    url_outdoor_seating: List[str] = Field(default_factory=list)

    # Additional categories potentially cited in the answer
    url_outdoor_clearance: List[str] = Field(default_factory=list)
    url_certified_food_manager: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
    Extract the following fields strictly from the provided answer (do not invent anything):
    1) chain_name: The name of the restaurant chain.
    2) md_location_address: A specific street address for at least one Maryland location (include city and state if present).
    
    Also extract arrays of reference URLs cited in the answer (extract only URLs explicitly present in the text; include full URLs):
    3) url_chain_multistate: URLs supporting that the brand operates in multiple U.S. states (national chain).
    4) url_md_location: URLs for the specific Maryland location page or listing confirming the address and/or details for that specific site.
    5) url_247: URLs supporting a 24/7/365 operating policy (chain-wide or for the specific MD location).
    6) url_holiday: URLs supporting being open on both Thanksgiving Day and Christmas Day (chain-wide or MD location specific).
    7) url_alcohol_md: URLs supporting alcohol service at the Maryland location and/or compliance with Maryland/local alcohol service hours.
    8) url_seating_capacity: URLs supporting that the Maryland location has seating capacity of at least 50 people.
    9) url_ada: URLs supporting ADA accessibility features at the Maryland location (including accessible seating; may be chain policy or location-specific if clearly applicable).
    10) url_outdoor_seating: URLs supporting outdoor seating availability at the Maryland location.
    11) url_outdoor_clearance: URLs supporting that the outdoor seating maintains minimum 5-foot clearance from driveways, alleys, and handicap ramps (could be location documentation or permitting/municipal guidance explicitly tied to the location).
    12) url_certified_food_manager: URLs supporting that the restaurant has at least one certified food manager on staff (chain policy or location-specific).

    Return JSON exactly with these fields. For any field not present in the answer, return null (for strings) or an empty array (for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nz(s: Optional[str]) -> str:
    return s or ""


def _combine_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    combined.append(uu)
                    seen.add(uu)
    return combined


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: ChainLocationExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Top-level critical sequential node representing the rubric "Root"
    task_root = evaluator.add_sequential(
        id="Root",
        desc="Identify a qualifying national chain restaurant and a specific Maryland location meeting all stated requirements, and provide the requested documentation URLs.",
        parent=evaluator.root,
        critical=True
    )

    # ------------------------------------------------------------------ #
    # 1) Provide Requested Identification (critical, parallel)           #
    # ------------------------------------------------------------------ #
    provide_ident = evaluator.add_parallel(
        id="Provide_Requested_Identification",
        desc="Provide the requested identification details (chain name and a specific Maryland address).",
        parent=task_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.chain_name and extracted.chain_name.strip()),
        id="Restaurant_Chain_Name_Provided",
        desc="The name of the restaurant chain is provided.",
        parent=provide_ident,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.md_location_address and extracted.md_location_address.strip()),
        id="Maryland_Location_Address_Provided",
        desc="A specific street address for at least one Maryland location is provided.",
        parent=provide_ident,
        critical=True
    )

    # ------------------------------------------------------------------ #
    # 2) Meets All Constraints (critical, parallel)                      #
    # ------------------------------------------------------------------ #
    constraints = evaluator.add_parallel(
        id="Meets_All_Constraints",
        desc="The identified chain/location meets all operational, holiday, alcohol, seating, ADA, outdoor seating, and other listed constraints.",
        parent=task_root,
        critical=True
    )

    name = _nz(extracted.chain_name)
    addr = _nz(extracted.md_location_address)

    # National_Chain_MultiState
    node_multistate = evaluator.add_leaf(
        id="National_Chain_MultiState",
        desc="Restaurant is a chain with multiple locations across different states.",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The restaurant brand '{name}' operates locations in multiple U.S. states (i.e., it is a national multistate chain).",
        node=node_multistate,
        sources=extracted.url_chain_multistate,
        additional_instruction="Accept evidence such as a company locations page or reputable source (e.g., Wikipedia/company profile) that clearly shows presence across multiple states."
    )

    # Has_Maryland_Location
    node_md_loc = evaluator.add_leaf(
        id="Has_Maryland_Location",
        desc="At least one location is in Maryland (the provided address is in Maryland).",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The address '{addr}' is located in the state of Maryland, United States, and corresponds to a '{name}' restaurant location.",
        node=node_md_loc,
        sources=_combine_urls(extracted.url_md_location),
        additional_instruction="Allow 'MD' as equivalent to 'Maryland'. The evidence should show this exact location belongs to the named chain and is in Maryland."
    )

    # Operates_24_7_365
    node_247 = evaluator.add_leaf(
        id="Operates_24_7_365",
        desc="Restaurant operates 24 hours a day, 7 days a week, 365 days a year.",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{name}' operates 24 hours a day, 7 days a week, 365 days a year (24/7/365) at least as a chain-wide policy or specifically at the Maryland location at '{addr}'.",
        node=node_247,
        sources=_combine_urls(extracted.url_247, extracted.url_md_location),
        additional_instruction="The page should make clear the 24/7/365 nature; if the claim is chain-wide, that is acceptable provided it applies to the selected MD location as an instance or explicit exception is not present."
    )

    # Open_On_Thanksgiving_And_Christmas
    node_holiday = evaluator.add_leaf(
        id="Open_On_Thanksgiving_And_Christmas",
        desc="Restaurant remains open on both Thanksgiving Day and Christmas Day.",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{name}' remains open on both Thanksgiving Day and Christmas Day (limited hours acceptable).",
        node=node_holiday,
        sources=_combine_urls(extracted.url_holiday, extracted.url_md_location),
        additional_instruction="Evidence can be chain policy or specific location hours for those holidays; limited hours still count as open."
    )

    # Alcohol_Service_And_MD_Hours_Compliance
    node_alcohol = evaluator.add_leaf(
        id="Alcohol_Service_And_MD_Hours_Compliance",
        desc="The Maryland location serves alcoholic beverages and complies with Maryland alcohol service hour restrictions.",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{name}' Maryland location at '{addr}' serves alcoholic beverages and its alcoholic-beverage service hours comply with Maryland (or the relevant Maryland county/municipal) alcohol service hour restrictions.",
        node=node_alcohol,
        sources=_combine_urls(extracted.url_alcohol_md, extracted.url_md_location),
        additional_instruction="Look for evidence like a menu showing beer/wine/cocktails, a liquor license listing, or posted hours cross-referenced with Maryland or county rules indicating compliance."
    )

    # Seating_Capacity_At_Least_50
    node_seating = evaluator.add_leaf(
        id="Seating_Capacity_At_Least_50",
        desc="The Maryland location has a seating capacity of at least 50 people.",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Maryland location at '{addr}' has a seating capacity of at least 50 people.",
        node=node_seating,
        sources=_combine_urls(extracted.url_seating_capacity, extracted.url_md_location),
        additional_instruction="Evidence could be occupancy or seating statements on the location page, a permit, or other authoritative documentation explicitly indicating seating capacity ≥ 50."
    )

    # ADA_Accessibility_Compliance
    node_ada = evaluator.add_leaf(
        id="ADA_Accessibility_Compliance",
        desc="The location complies with ADA accessibility requirements as specified: at least 5% of seating (minimum 1 table) is wheelchair accessible, and accessible tables meet the stated height (28–34 inches) and knee-clearance (27 inches) specs.",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Maryland location at '{addr}' meets ADA seating accessibility: at least 5% (minimum 1 table) is wheelchair accessible; accessible tables are 28–34 inches high with ≥27 inches knee clearance.",
        node=node_ada,
        sources=_combine_urls(extracted.url_ada, extracted.url_md_location),
        additional_instruction="Accept explicit ADA policy or location-specific documentation that clearly asserts compliance with these seating/table specifications."
    )

    # Certified_Food_Manager_On_Staff
    node_cfm = evaluator.add_leaf(
        id="Certified_Food_Manager_On_Staff",
        desc="The restaurant has at least one certified food manager on staff.",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The restaurant '{name}' (chain policy or the specific Maryland location) has at least one certified food manager (e.g., ServSafe-certified Food Protection Manager) on staff.",
        node=node_cfm,
        sources=_combine_urls(extracted.url_certified_food_manager),
        additional_instruction="Accept chain policy statements or authoritative documentation that a certified food manager is on staff; location-specific proof also acceptable."
    )

    # Outdoor_Seating_Available
    node_outdoor = evaluator.add_leaf(
        id="Outdoor_Seating_Available",
        desc="The Maryland location has outdoor seating available.",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Maryland location at '{addr}' has outdoor seating available.",
        node=node_outdoor,
        sources=_combine_urls(extracted.url_outdoor_seating, extracted.url_md_location),
        additional_instruction="Look for explicit mention or visuals confirming outdoor seating at the specific Maryland site."
    )

    # Outdoor_Clearance_At_Least_5ft
    node_clearance = evaluator.add_leaf(
        id="Outdoor_Clearance_At_Least_5ft",
        desc="Outdoor seating maintains a minimum 5-foot clearance from driveways, alleys, and handicap ramps.",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The outdoor seating area at the Maryland location maintains at least 5 feet of clearance from driveways, alleys, and handicap ramps.",
        node=node_clearance,
        sources=_combine_urls(extracted.url_outdoor_clearance, extracted.url_outdoor_seating, extracted.url_md_location),
        additional_instruction="Accept authoritative permitting/inspection documents or location-specific compliance statements explicitly confirming the ≥5 ft clearance."
    )

    # ------------------------------------------------------------------ #
    # 3) Provide Requested Documentation URLs (critical, parallel)       #
    # ------------------------------------------------------------------ #
    doc_urls = evaluator.add_parallel(
        id="Provide_Requested_Documentation_URLs",
        desc="Provide reference URLs for each documentation category explicitly requested in the proposed question.",
        parent=task_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(extracted.url_247) > 0,
        id="URL_24_7_365",
        desc="At least one reference URL supports the 24/7/365 operating claim.",
        parent=doc_urls,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(extracted.url_holiday) > 0,
        id="URL_Holiday_Operations",
        desc="At least one reference URL supports being open on Thanksgiving Day and Christmas Day.",
        parent=doc_urls,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(extracted.url_alcohol_md) > 0,
        id="URL_Alcohol_Service",
        desc="At least one reference URL supports alcohol service at the Maryland location.",
        parent=doc_urls,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(extracted.url_seating_capacity) > 0,
        id="URL_Seating_Capacity",
        desc="At least one reference URL supports seating capacity of 50 or more.",
        parent=doc_urls,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(extracted.url_ada) > 0,
        id="URL_ADA_Accessibility",
        desc="At least one reference URL supports ADA accessibility features (including seating accessibility).",
        parent=doc_urls,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(extracted.url_outdoor_seating) > 0,
        id="URL_Outdoor_Seating",
        desc="At least one reference URL supports outdoor seating availability at the Maryland location.",
        parent=doc_urls,
        critical=True
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
    Evaluate an answer for the Maryland chain restaurant compliance task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=ChainLocationExtraction,
        extraction_name="chain_location_extraction",
    )

    # Build and run verification
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()