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
TASK_ID = "historic_broadway_theater_identification"
TASK_DESCRIPTION = (
    "Identify the Broadway theater that was home to the longest-running show in Broadway history "
    "(which had 13,981 performances before closing in April 2023). For this theater, provide the "
    "following verified information: (1) The name of the longest-running show and the theater where it "
    "performed, (2) The complete street address of the theater (including street number, street name, "
    "city, and state), (3) The theater's seating capacity, (4) Confirmation that the theater meets the "
    "official Broadway classification requirement (state what the minimum seating requirement is and "
    "confirm the theater meets it), and (5) The name of the organization that operates this theater. "
    "All information must be supported by reliable reference URLs."
)

# Expected facts (used only as context/ground-truth info)
EXPECTED = {
    "show_name": "The Phantom of the Opera",
    "theater_name": "Majestic Theatre",
    "performance_count": "13,981",
    "closing_date": "April 16, 2023",
    "address_full": "245 West 44th Street, New York, NY",
    "capacity": "1,681",
    "broadway_min_seats": "500",
    "operator_name": "Shubert Organization",
    "theater_district_definition": "West 41st–54th Streets, between Sixth and Eighth Avenues",
}

# --------------------------------------------------------------------------- #
# Data model for extracted info                                               #
# --------------------------------------------------------------------------- #
class TheaterInfoExtraction(BaseModel):
    # Core facts
    show_name: Optional[str] = None
    theater_name: Optional[str] = None
    performance_count: Optional[str] = None
    closing_date: Optional[str] = None

    address_full: Optional[str] = None
    capacity: Optional[str] = None

    min_seating_requirement: Optional[str] = None
    theater_district_definition: Optional[str] = None
    operator_name: Optional[str] = None

    # URL buckets
    urls_show_and_run: List[str] = Field(default_factory=list)               # show identity + run facts (performances, closing)
    urls_show_at_theater: List[str] = Field(default_factory=list)            # show performed at the theater
    urls_address: List[str] = Field(default_factory=list)                    # theater address
    urls_capacity: List[str] = Field(default_factory=list)                   # theater seating capacity
    urls_min_seat_req: List[str] = Field(default_factory=list)               # Broadway minimum seat requirement (500)
    urls_official_designation: List[str] = Field(default_factory=list)       # official Broadway designation for the venue
    urls_district_definition: List[str] = Field(default_factory=list)        # Theater District boundary definition references
    urls_operator: List[str] = Field(default_factory=list)                   # operator (Shubert Organization) references


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theater_info() -> str:
    return """
    Extract the following fields EXACTLY as stated in the provided answer. Do NOT invent or infer anything that is not explicitly present. 
    If a field is missing, set it to null. For each category of fact, also extract the list of explicit URL(s) cited that directly support that fact.
    
    Required fields:
    - show_name: the name of the longest-running Broadway show.
    - theater_name: the Broadway theater where that show performed.
    - performance_count: the number of performances (e.g., "13,981").
    - closing_date: the closing date on Broadway (e.g., "April 16, 2023" or "April 2023").
    - address_full: the complete street address of the theater (include street number, street name, city, and state; ZIP code optional).
    - capacity: the seating capacity of the theater (string, keep commas if provided).
    - min_seating_requirement: the stated minimum seating requirement for a Broadway theater (e.g., "500").
    - theater_district_definition: the stated boundary definition for NYC's Theater District (e.g., "West 41st–54th Streets, between Sixth and Eighth Avenues").
    - operator_name: the organization that operates the theater (e.g., "Shubert Organization").
    
    For each of the following URL lists, extract ALL URLs explicitly shown in the answer text for that purpose:
    - urls_show_and_run: supports the show identity and its run facts (13,981 performances; April 2023 closing date).
    - urls_show_at_theater: supports that the show performed at the Majestic Theatre.
    - urls_address: supports the theater address.
    - urls_capacity: supports the theater capacity.
    - urls_min_seat_req: supports the Broadway minimum seating requirement (500).
    - urls_official_designation: supports that the theater is officially designated as a Broadway venue.
    - urls_district_definition: supports the Theater District boundary definition cited.
    - urls_operator: supports the theater operator (Shubert Organization).
    
    Return a single JSON object with exactly these fields. Ensure all URL fields are arrays of strings; if no URL is provided, return an empty array for that field.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_urls(*url_lists: List[str]) -> List[str]:
    """Combine and de-duplicate multiple URL lists while preserving order."""
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if u and u not in seen:
                combined.append(u)
                seen.add(u)
    return combined


async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    additional_instruction: str = "None",
    critical: bool = True,
):
    """
    Create a leaf node and verify the claim using the provided URLs.
    If no URLs are provided, mark the node as failed immediately (URL-backed requirement).
    """
    if not urls:
        # Hard fail due to missing sources for this URL-backed verification
        evaluator.add_leaf(
            id=node_id,
            desc=f"{desc} (FAILED: no supporting URLs provided)",
            parent=parent,
            critical=critical,
            score=0.0,
            status="failed",
        )
        return

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=additional_instruction,
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
    Evaluate an answer for the Historic Broadway Theater Identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # top-level grouping can be parallel
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

    # Extract structured info from the answer
    extracted: TheaterInfoExtraction = await evaluator.extract(
        prompt=prompt_extract_theater_info(),
        template_class=TheaterInfoExtraction,
        extraction_name="theater_info_extraction",
    )

    # Record ground truth context (for transparency only; not enforced automatically)
    evaluator.add_ground_truth(
        {
            "expected": EXPECTED,
            "notes": "These are commonly accepted facts; the evaluation still relies on the answer's cited URLs.",
        }
    )

    # Build main critical node
    main_node = evaluator.add_parallel(
        id="Historic_Broadway_Theater_Identification",
        desc="Identify the Broadway theater that hosted the longest-running show and provide the required verified attributes with supporting reliable reference URLs.",
        parent=root,
        critical=True,
    )

    # ------------------------------------------------------------------ #
    # 1) Show + Theater group                                            #
    # ------------------------------------------------------------------ #
    show_theater_node = evaluator.add_parallel(
        id="Show_And_Theater",
        desc="Provide the name of the longest-running show and the theater where it performed, consistent with the constraints.",
        parent=main_node,
        critical=True,
    )

    # Show name (verify with URLs that the longest-running show is Phantom)
    await _verify_with_urls_or_fail(
        evaluator,
        parent=show_theater_node,
        node_id="Show_Name",
        desc="Identifies the longest-running show as 'The Phantom of the Opera'.",
        claim="The longest-running show in Broadway history is 'The Phantom of the Opera'.",
        urls=_combine_urls(extracted.urls_show_and_run),
        additional_instruction="Accept equivalent capitalization and punctuation. Verify that the page explicitly states Phantom is the longest-running Broadway show.",
    )

    # Theater name (verify that Phantom's Broadway production was at the Majestic Theatre)
    await _verify_with_urls_or_fail(
        evaluator,
        parent=show_theater_node,
        node_id="Theater_Name",
        desc="Identifies the theater as the 'Majestic Theatre'.",
        claim="The Broadway production of 'The Phantom of the Opera' played at the Majestic Theatre in New York City.",
        urls=_combine_urls(extracted.urls_show_at_theater, extracted.urls_show_and_run),
        additional_instruction="Minor formatting differences in the theater name are acceptable (e.g., 'Majestic Theatre (Broadway)').",
    )

    # Show performed at theater (redundant but explicit)
    await _verify_with_urls_or_fail(
        evaluator,
        parent=show_theater_node,
        node_id="Show_Performed_At_Theater",
        desc="States/clearly indicates that 'The Phantom of the Opera' performed at the Majestic Theatre.",
        claim="'The Phantom of the Opera' performed at the Majestic Theatre (Broadway) in NYC.",
        urls=_combine_urls(extracted.urls_show_at_theater, extracted.urls_show_and_run),
        additional_instruction="Look for explicit statements that the Majestic Theatre hosted the Broadway run of Phantom.",
    )

    # Performance count (13,981)
    await _verify_with_urls_or_fail(
        evaluator,
        parent=show_theater_node,
        node_id="Performance_Count",
        desc="States the show had 13,981 performances.",
        claim="'The Phantom of the Opera' had 13,981 performances on Broadway.",
        urls=_combine_urls(extracted.urls_show_and_run),
        additional_instruction="Allow thousand separators ('13,981'). The page should clearly state this performance count for the Broadway run.",
    )

    # Closing date (April 16, 2023 / April 2023)
    await _verify_with_urls_or_fail(
        evaluator,
        parent=show_theater_node,
        node_id="Closing_Date",
        desc="States the show closed on April 16, 2023 (April 2023).",
        claim="'The Phantom of the Opera' on Broadway closed on April 16, 2023 (i.e., April 2023).",
        urls=_combine_urls(extracted.urls_show_and_run),
        additional_instruction="Count as supported if the page explicitly gives April 16, 2023 or otherwise clearly states closure in April 2023.",
    )

    # ------------------------------------------------------------------ #
    # 2) Theater Address                                                 #
    # ------------------------------------------------------------------ #
    address_node = evaluator.add_parallel(
        id="Theater_Address",
        desc="Provide the complete street address of the theater (street number, street name, city, state).",
        parent=main_node,
        critical=True,
    )

    await _verify_with_urls_or_fail(
        evaluator,
        parent=address_node,
        node_id="Complete_Address",
        desc="Gives the full address as 245 West 44th Street, New York, NY (street number + street name + city + state).",
        claim="The address of the Majestic Theatre is 245 West 44th Street, New York, NY.",
        urls=_combine_urls(extracted.urls_address),
        additional_instruction="Allow optional ZIP code (e.g., 10036) and minor punctuation variants.",
    )

    # ------------------------------------------------------------------ #
    # 3) Theater Capacity                                                #
    # ------------------------------------------------------------------ #
    capacity_node = evaluator.add_parallel(
        id="Theater_Capacity",
        desc="Provide the theater's seating capacity.",
        parent=main_node,
        critical=True,
    )

    await _verify_with_urls_or_fail(
        evaluator,
        parent=capacity_node,
        node_id="Capacity_Value",
        desc="States the Majestic Theatre seating capacity as 1,681 seats.",
        claim="The seating capacity of the Majestic Theatre is 1,681 seats.",
        urls=_combine_urls(extracted.urls_capacity),
        additional_instruction="Accept minor formatting such as '1,681' vs '1681'. The page should clearly indicate the stated capacity.",
    )

    # ------------------------------------------------------------------ #
    # 4) Broadway Classification                                         #
    # ------------------------------------------------------------------ #
    classification_node = evaluator.add_parallel(
        id="Broadway_Classification",
        desc="Confirm the theater meets Broadway classification requirements specified in the constraints.",
        parent=main_node,
        critical=True,
    )

    # Minimum seat requirement (500)
    await _verify_with_urls_or_fail(
        evaluator,
        parent=classification_node,
        node_id="Minimum_Seat_Requirement",
        desc="States the minimum seating requirement for a Broadway theater is 500 seats.",
        claim="The minimum seating requirement for a Broadway theatre is 500 seats.",
        urls=_combine_urls(extracted.urls_min_seat_req),
        additional_instruction="Accept equivalent wording like 'Broadway theaters have 500 or more seats'.",
    )

    # Meets minimum seat requirement (>= 500) based on capacity source(s)
    await _verify_with_urls_or_fail(
        evaluator,
        parent=classification_node,
        node_id="Meets_Minimum_Seat_Requirement",
        desc="Confirms the theater meets/exceeds the 500-seat minimum (consistent with the stated capacity).",
        claim="The Majestic Theatre has at least 500 seats (its capacity is 1,681, which is >= 500).",
        urls=_combine_urls(extracted.urls_capacity),
        additional_instruction="It is sufficient that the page shows a capacity above 500; simple numeric comparison establishes compliance.",
    )

    # Official Broadway venue designation
    await _verify_with_urls_or_fail(
        evaluator,
        parent=classification_node,
        node_id="Official_Broadway_Venue",
        desc="Confirms the theater is an officially designated Broadway venue (not Off-Broadway/other).",
        claim="The Majestic Theatre is an officially designated Broadway theatre (not Off-Broadway).",
        urls=_combine_urls(extracted.urls_official_designation),
        additional_instruction="Prefer official or authoritative sources (e.g., theater owner/operator or Broadway League).",
    )

    # Theater District definition
    await _verify_with_urls_or_fail(
        evaluator,
        parent=classification_node,
        node_id="Theater_District_Definition",
        desc="States the Theater District boundary definition: West 41st–54th Streets, between 6th–8th Avenues.",
        claim="New York City's Theater District is commonly defined as West 41st–54th Streets, between Sixth and Eighth Avenues.",
        urls=_combine_urls(extracted.urls_district_definition),
        additional_instruction="Accept minor variants such as hyphen or en dash, and '6th' vs 'Sixth'.",
    )

    # Theater in Theater District (use address + district definition pages)
    await _verify_with_urls_or_fail(
        evaluator,
        parent=classification_node,
        node_id="Theater_In_Theater_District",
        desc="Confirms the theater’s address is consistent with being within the stated Theater District boundaries.",
        claim="The address 245 West 44th Street, New York, NY lies within the Theater District boundaries (West 41st–54th Streets, between Sixth and Eighth Avenues).",
        urls=_combine_urls(extracted.urls_address, extracted.urls_district_definition, extracted.urls_official_designation),
        additional_instruction="Use the address page and the district-definition page together. If an official page states the Majestic Theatre is in the Theater District, that also suffices.",
    )

    # ------------------------------------------------------------------ #
    # 5) Theater Operator                                                #
    # ------------------------------------------------------------------ #
    operator_node = evaluator.add_parallel(
        id="Theater_Operator",
        desc="Provide the name of the organization that operates the theater.",
        parent=main_node,
        critical=True,
    )

    await _verify_with_urls_or_fail(
        evaluator,
        parent=operator_node,
        node_id="Operator_Name",
        desc="Identifies the operator as the Shubert Organization.",
        claim="The Majestic Theatre is operated by the Shubert Organization.",
        urls=_combine_urls(extracted.urls_operator),
        additional_instruction="Prefer the operator's official site or the Broadway League; reputable references are acceptable.",
    )

    # ------------------------------------------------------------------ #
    # 6) Reference URLs completeness/reliability checks                  #
    # ------------------------------------------------------------------ #
    refs_node = evaluator.add_parallel(
        id="Reference_URLs",
        desc="All required facts are supported by reliable reference URLs that contain the asserted information.",
        parent=main_node,
        critical=True,
    )

    all_urls = _combine_urls(
        extracted.urls_show_and_run,
        extracted.urls_show_at_theater,
        extracted.urls_address,
        extracted.urls_capacity,
        extracted.urls_min_seat_req,
        extracted.urls_official_designation,
        extracted.urls_district_definition,
        extracted.urls_operator,
    )

    # URL reliability (at least one provided page is from a reliable/authoritative source)
    await _verify_with_urls_or_fail(
        evaluator,
        parent=refs_node,
        node_id="URL_Reliability",
        desc="Provided URLs are from generally reliable sources (official organizations/sites, reputable news outlets, or recognized reference publishers).",
        claim="This page is from an official organization, a reputable news outlet, or a recognized reference publisher—not an unverifiable personal blog or forum post.",
        urls=all_urls,
        additional_instruction="Judge each page's reliability by its domain/ownership and editorial standards. Passing any reliable page suffices.",
    )

    # Per-category URL support checks (redundant to above fact-claims but explicitly required)
    await _verify_with_urls_or_fail(
        evaluator,
        parent=refs_node,
        node_id="URL_For_Show_And_Run_Facts",
        desc="Provides at least one URL supporting the show identity and the run facts (13,981 performances and closing date).",
        claim="The page supports that 'The Phantom of the Opera' is the longest-running Broadway show, closed in April 2023 (specifically April 16, 2023), and had 13,981 performances.",
        urls=_combine_urls(extracted.urls_show_and_run),
        additional_instruction="The page should explicitly include the run facts (performances and closing date).",
    )

    await _verify_with_urls_or_fail(
        evaluator,
        parent=refs_node,
        node_id="URL_For_Show_At_Theater",
        desc="Provides at least one URL supporting that the show performed at the Majestic Theatre.",
        claim="The page supports that 'The Phantom of the Opera' performed at the Majestic Theatre (Broadway).",
        urls=_combine_urls(extracted.urls_show_at_theater, extracted.urls_show_and_run),
        additional_instruction="Look for explicit phrasing linking Phantom to the Majestic Theatre.",
    )

    await _verify_with_urls_or_fail(
        evaluator,
        parent=refs_node,
        node_id="URL_For_Address",
        desc="Provides at least one URL supporting the theater address.",
        claim="The page lists the Majestic Theatre's address as 245 West 44th Street, New York, NY.",
        urls=_combine_urls(extracted.urls_address),
        additional_instruction="ZIP code may be present or absent; minor formatting differences acceptable.",
    )

    await _verify_with_urls_or_fail(
        evaluator,
        parent=refs_node,
        node_id="URL_For_Capacity",
        desc="Provides at least one URL supporting the theater seating capacity.",
        claim="The page states that the Majestic Theatre has a seating capacity of 1,681.",
        urls=_combine_urls(extracted.urls_capacity),
        additional_instruction="Accept '1,681' vs '1681'.",
    )

    await _verify_with_urls_or_fail(
        evaluator,
        parent=refs_node,
        node_id="URL_For_Broadway_Minimum_Seats",
        desc="Provides at least one URL supporting the Broadway minimum seating requirement (500 seats).",
        claim="The page states that Broadway theaters have a minimum of 500 seats.",
        urls=_combine_urls(extracted.urls_min_seat_req),
        additional_instruction="Equivalent wording acceptable, e.g., 'Broadway houses have 500 or more seats'.",
    )

    await _verify_with_urls_or_fail(
        evaluator,
        parent=refs_node,
        node_id="URL_For_Official_Broadway_Designation",
        desc="Provides at least one URL supporting that the venue is an officially designated Broadway theater.",
        claim="The page confirms that the Majestic Theatre is an officially designated Broadway theater.",
        urls=_combine_urls(extracted.urls_official_designation),
        additional_instruction="Prefer authoritative/official sources.",
    )

    await _verify_with_urls_or_fail(
        evaluator,
        parent=refs_node,
        node_id="URL_For_Theater_District_Definition",
        desc="Provides at least one URL supporting the Theater District boundary definition used.",
        claim="The page describes the Theater District as West 41st–54th Streets, between Sixth and Eighth Avenues.",
        urls=_combine_urls(extracted.urls_district_definition),
        additional_instruction="Minor variants (hyphen vs en dash, '6th' vs 'Sixth') acceptable.",
    )

    await _verify_with_urls_or_fail(
        evaluator,
        parent=refs_node,
        node_id="URL_For_Operator",
        desc="Provides at least one URL supporting the theater operator (Shubert Organization).",
        claim="The page indicates that the Majestic Theatre is operated by the Shubert Organization.",
        urls=_combine_urls(extracted.urls_operator),
        additional_instruction="Prefer the operator's official site or authoritative industry sources.",
    )

    # Return structured summary
    return evaluator.get_summary()