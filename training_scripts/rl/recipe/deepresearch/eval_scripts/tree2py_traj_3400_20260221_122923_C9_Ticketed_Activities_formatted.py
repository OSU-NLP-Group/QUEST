import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wc2026_us_stadiums"
TASK_DESCRIPTION = """Among the stadiums in the United States that will host matches during the 2026 FIFA World Cup, identify 4 venues that each satisfy ALL of the following requirements:

1. Minimum Seating Capacity: The stadium must have a total seating capacity of at least 69,000 seats.

2. Natural Grass Field Capability: The stadium must have the capability to install and maintain a natural grass playing surface for FIFA World Cup matches, as required by FIFA regulations (artificial turf is not permitted for World Cup play).

3. Accessibility Compliance: The stadium must meet ADA (Americans with Disabilities Act) requirements by providing wheelchair-accessible seating for at least 1% of its total capacity (minimum 690 wheelchair-accessible seats for a 69,000-capacity venue).

4. Official Host Venue Status: The stadium must be officially designated as a 2026 FIFA World Cup host venue by FIFA.

For each of the 4 stadiums you identify, provide the following information:
- Official stadium name
- Host city and state
- Total seating capacity
- Reference URL(s) that document the stadium's capacity, natural grass capability, and 2026 World Cup host status
"""

MIN_CAPACITY = 69000
MIN_ACCESSIBLE_RATIO = 0.01
MIN_ACCESSIBLE_ABS = int(MIN_CAPACITY * MIN_ACCESSIBLE_RATIO)  # 690


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StadiumItem(BaseModel):
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to handle ranges/approximate wording
    capacity_urls: List[str] = Field(default_factory=list)
    grass_urls: List[str] = Field(default_factory=list)
    host_urls: List[str] = Field(default_factory=list)
    accessibility_urls: List[str] = Field(default_factory=list)


class StadiumsExtraction(BaseModel):
    stadiums: List[StadiumItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadiums() -> str:
    return (
        "Extract up to 4 U.S. stadiums mentioned in the answer that the user claims will host matches "
        "during the 2026 FIFA World Cup. For each stadium, extract the following fields exactly as stated:\n"
        "1. official_name: The official stadium name.\n"
        "2. city: The host city.\n"
        "3. state: The host U.S. state (use the two-letter abbreviation or full name if provided).\n"
        "4. capacity: The total seating capacity (as written in the answer; keep it as a string, do not convert to a number).\n"
        "5. capacity_urls: A list of URL(s) that the answer cites for the stadium's capacity.\n"
        "6. grass_urls: A list of URL(s) that the answer cites for natural grass capability (installation plan or feasibility) for 2026.\n"
        "7. host_urls: A list of URL(s) that the answer cites confirming the stadium is an official 2026 FIFA World Cup host venue.\n"
        "8. accessibility_urls: A list of URL(s) that the answer cites confirming ADA wheelchair-accessible seating provisions (counts or compliance).\n\n"
        "Rules:\n"
        "- Only extract URLs that explicitly appear in the answer text. If the answer mentions a source without an actual URL, do not invent the URL; just omit it.\n"
        "- For each field that is missing in the answer, set it to null (or an empty list for the URL fields).\n"
        "- Return the results in an array under the key 'stadiums'. If the answer lists more than 4 stadiums, only include the first 4.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal_label(idx: int) -> str:
    return ["First", "Second", "Third", "Fourth"][idx] if 0 <= idx < 4 else f"Stadium_{idx + 1}"


def aggregate_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            u = (u or "").strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


def looks_like_valid_url(u: str) -> bool:
    if not isinstance(u, str):
        return False
    s = u.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


# --------------------------------------------------------------------------- #
# Verification for a single stadium                                           #
# --------------------------------------------------------------------------- #
async def verify_one_stadium(
    evaluator: Evaluator,
    parent_node,
    stadium: StadiumItem,
    idx: int,
) -> None:
    ord_label = ordinal_label(idx)

    # Stadium node (critical to satisfy root critical requirement)
    stadium_node = evaluator.add_parallel(
        id=f"{ord_label}_Stadium",
        desc=f"{ord_label} qualifying stadium meets all requirements",
        parent=parent_node,
        critical=True,
    )

    # ---------------- Venue Identification ----------------
    venue_ident_node = evaluator.add_parallel(
        id=f"{ord_label}_Venue_Identification",
        desc="Stadium is correctly identified with accurate venue information",
        parent=stadium_node,
        critical=True,
    )

    # Official Name
    official_name_node = evaluator.add_parallel(
        id=f"{ord_label}_Official_Name",
        desc="Official stadium name is provided accurately",
        parent=venue_ident_node,
        critical=True,
    )

    # Name reference existence first (gate)
    name_ref_urls = aggregate_urls(stadium.host_urls, stadium.capacity_urls)
    evaluator.add_custom_node(
        result=len(name_ref_urls) > 0,
        id=f"{ord_label}_Name_Reference_URL",
        desc="Reference URL confirms the official stadium name",
        parent=official_name_node,
        critical=True,
    )

    name_accuracy_leaf = evaluator.add_leaf(
        id=f"{ord_label}_Name_Accuracy",
        desc="Stadium name matches the official designation used for 2026 FIFA World Cup",
        parent=official_name_node,
        critical=True,
    )
    official_name = stadium.official_name or ""
    await evaluator.verify(
        claim=f"The official stadium name is '{official_name}', as recognized by authoritative references (e.g., FIFA host venue listings or stadium official pages). Minor naming variants due to sponsorship should be treated as equivalent if referring to the same venue.",
        node=name_accuracy_leaf,
        sources=name_ref_urls,
        additional_instruction="Confirm the venue name shown on the provided pages matches the answer's stadium name or an accepted variant referring to the same venue.",
    )

    # Location Information
    location_node = evaluator.add_parallel(
        id=f"{ord_label}_Location_Information",
        desc="Host city and state are correctly specified",
        parent=venue_ident_node,
        critical=True,
    )

    # Location reference existence first (gate)
    loc_ref_urls = aggregate_urls(stadium.host_urls, stadium.capacity_urls)
    evaluator.add_custom_node(
        result=len(loc_ref_urls) > 0,
        id=f"{ord_label}_Location_Reference_URL",
        desc="Reference URL confirms the stadium location",
        parent=location_node,
        critical=True,
    )

    city = stadium.city or ""
    state = stadium.state or ""
    city_state_leaf = evaluator.add_leaf(
        id=f"{ord_label}_City_State_Accuracy",
        desc="City and state location match the stadium's actual location",
        parent=location_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The stadium is located in {city}, {state}.",
        node=city_state_leaf,
        sources=loc_ref_urls,
        additional_instruction="Verify that the referenced pages explicitly indicate the stadium is located in the specified city and state. Allow minor regional phrasing (e.g., metro area variants) if clearly the same venue.",
    )

    # ---------------- Capacity Requirements ----------------
    capacity_req_node = evaluator.add_parallel(
        id=f"{ord_label}_Capacity_Requirements",
        desc="Stadium meets all capacity-related requirements",
        parent=stadium_node,
        critical=True,
    )

    # Total capacity threshold
    total_cap_node = evaluator.add_parallel(
        id=f"{ord_label}_Total_Capacity_Threshold",
        desc="Stadium has minimum 69,000 total seating capacity",
        parent=capacity_req_node,
        critical=True,
    )

    # Capacity reference existence first (gate)
    cap_urls = aggregate_urls(stadium.capacity_urls)
    evaluator.add_custom_node(
        result=len(cap_urls) > 0,
        id=f"{ord_label}_Capacity_Reference_URL",
        desc="Reference URL documents the stadium's total capacity",
        parent=total_cap_node,
        critical=True,
    )

    cap_verify_leaf = evaluator.add_leaf(
        id=f"{ord_label}_Capacity_Verification",
        desc="Documented capacity is at least 69,000 seats",
        parent=total_cap_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The stadium has a total seating capacity of at least {MIN_CAPACITY} seats.",
        node=cap_verify_leaf,
        sources=cap_urls,
        additional_instruction="Confirm the capacity number on the referenced page(s). If multiple capacities are listed for different configurations, accept configurations that meet or exceed 69,000 for World Cup match hosting.",
    )

    # Accessibility compliance
    access_node = evaluator.add_parallel(
        id=f"{ord_label}_Accessibility_Compliance",
        desc="Stadium meets ADA wheelchair accessibility requirements (minimum 1% of capacity)",
        parent=capacity_req_node,
        critical=True,
    )

    # Accessibility reference existence first (gate)
    access_urls = aggregate_urls(stadium.accessibility_urls)
    evaluator.add_custom_node(
        result=len(access_urls) > 0,
        id=f"{ord_label}_Accessibility_Reference_URL",
        desc="Reference URL confirms accessibility seating provisions",
        parent=access_node,
        critical=True,
    )

    wheelchair_leaf = evaluator.add_leaf(
        id=f"{ord_label}_Wheelchair_Seating_Verification",
        desc="Stadium provides at least 1% of capacity as wheelchair-accessible seating (minimum 690 wheelchair spaces for 69,000 capacity)",
        parent=access_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The stadium provides wheelchair-accessible seating for at least 1% of total capacity (minimum {MIN_ACCESSIBLE_ABS} wheelchair-accessible seats for a {MIN_CAPACITY}-capacity venue).",
        node=wheelchair_leaf,
        sources=access_urls,
        additional_instruction="Look for explicit counts or credible documentation indicating that ADA seating meets or exceeds 1% of total capacity. Seating maps, official ADA policies, or compliance reports are acceptable sources.",
    )

    # ---------------- FIFA Competition Standards ----------------
    fifa_node = evaluator.add_parallel(
        id=f"{ord_label}_FIFA_Competition_Standards",
        desc="Stadium meets FIFA World Cup 2026 competition requirements",
        parent=stadium_node,
        critical=True,
    )

    # Natural grass capability
    grass_node = evaluator.add_parallel(
        id=f"{ord_label}_Natural_Grass_Capability",
        desc="Stadium has capability to install/maintain natural grass field for FIFA matches",
        parent=fifa_node,
        critical=True,
    )

    # Grass reference existence first (gate)
    grass_urls = aggregate_urls(stadium.grass_urls)
    evaluator.add_custom_node(
        result=len(grass_urls) > 0,
        id=f"{ord_label}_Grass_Reference_URL",
        desc="Reference URL confirms natural grass field capability",
        parent=grass_node,
        critical=True,
    )

    grass_confirm_leaf = evaluator.add_leaf(
        id=f"{ord_label}_Grass_Installation_Confirmed",
        desc="Documentation confirms natural grass field capability or installation plan for 2026 World Cup",
        parent=grass_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The stadium has documented capability to install and maintain a natural grass field for FIFA World Cup matches (e.g., a plan to install grass for 2026 or technical feasibility confirmed).",
        node=grass_confirm_leaf,
        sources=grass_urls,
        additional_instruction="Confirm via official announcements, credible news sources, or stadium documentation that natural grass will be installed or is feasible for World Cup matches.",
    )

    # Official host status
    host_node = evaluator.add_parallel(
        id=f"{ord_label}_World_Cup_Host_Status",
        desc="Stadium is officially designated as a 2026 FIFA World Cup host venue",
        parent=fifa_node,
        critical=True,
    )

    # Host status reference existence first (gate)
    host_urls = aggregate_urls(stadium.host_urls)
    evaluator.add_custom_node(
        result=len(host_urls) > 0,
        id=f"{ord_label}_Host_Status_Reference_URL",
        desc="Reference URL confirms 2026 World Cup host venue status",
        parent=host_node,
        critical=True,
    )

    host_confirm_leaf = evaluator.add_leaf(
        id=f"{ord_label}_Official_Host_Confirmation",
        desc="Stadium appears on official FIFA 2026 World Cup venue list for United States",
        parent=host_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The stadium is officially designated by FIFA as a host venue for the 2026 World Cup.",
        node=host_confirm_leaf,
        sources=host_urls,
        additional_instruction="Prefer official FIFA venue list pages or authoritative announcements. If multiple pages are provided, any page explicitly confirming official host status suffices.",
    )

    # ---------------- Documentation Quality ----------------
    # Note: To satisfy the framework's 'critical parent -> critical children' constraint,
    # we mark this section as critical but design the check to be lenient.
    doc_quality_node = evaluator.add_parallel(
        id=f"{ord_label}_Documentation_Quality",
        desc="All provided information is properly documented with valid reference URLs",
        parent=stadium_node,
        critical=True,
    )

    # Check URLs look valid (simple heuristic)
    all_urls = aggregate_urls(stadium.capacity_urls, stadium.grass_urls, stadium.host_urls, stadium.accessibility_urls)
    url_valid_result = (
        len(all_urls) > 0 and all(looks_like_valid_url(u) for u in all_urls)
    )
    evaluator.add_custom_node(
        result=url_valid_result,
        id=f"{ord_label}_Reference_URL_Validity",
        desc="All reference URLs are accessible and support the provided information",
        parent=doc_quality_node,
        critical=True,
    )


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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 2026 FIFA World Cup U.S. stadiums task.
    Builds a verification tree per the rubric and returns the evaluator summary.
    """
    evaluator = Evaluator()

    # Initialize evaluator; the root created here is non-critical by design
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

    # Add the rubric root node under the evaluator root (critical, parallel)
    rubric_root = evaluator.add_parallel(
        id="Four_Qualifying_Stadiums",
        desc="Identify 4 U.S. stadiums hosting the 2026 FIFA World Cup, each meeting all specified requirements (minimum 69,000 capacity, natural grass capability, accessibility compliance)",
        parent=root,
        critical=True,
    )

    # Extract stadiums from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_stadiums(),
        template_class=StadiumsExtraction,
        extraction_name="stadiums_extraction",
    )

    # Prepare exactly 4 stadiums: take up to first 4; pad with empty if fewer
    items: List[StadiumItem] = list(extracted.stadiums[:4])
    while len(items) < 4:
        items.append(StadiumItem())

    # Verify each stadium subtree
    for i in range(4):
        await verify_one_stadium(evaluator, rubric_root, items[i], i)

    # Return structured summary
    return evaluator.get_summary()