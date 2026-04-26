import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "midwest_historic_theater"
TASK_DESCRIPTION = (
    "Identify a historic theater in the Midwest United States that meets all of the following criteria: "
    "(1) The theater must be listed on the National Register of Historic Places; "
    "(2) It must have been built during the 1920s (between 1920 and 1929); "
    "(3) It must have a seating capacity of at least 2,000 seats; "
    "(4) It must be currently operational and actively hosting live performances; "
    "(5) It must have a proscenium stage with a width of at least 30 feet; "
    "(6) It must provide ADA-compliant wheelchair accessible seating and facilities. "
    "Provide the theater's name, city location, construction year, seating capacity, "
    "and documentation of its National Register status, stage specifications, and accessibility features. "
    "Include reference URLs for all key information."
)


# --------------------------------------------------------------------------- #
# Extraction model                                                            #
# --------------------------------------------------------------------------- #
class TheaterExtraction(BaseModel):
    # Identity
    theater_name: Optional[str] = None
    theater_name_urls: List[str] = Field(default_factory=list)

    # Location
    city: Optional[str] = None
    state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    major_city_support_urls: List[str] = Field(default_factory=list)

    # Historical / NRHP / Construction year
    nrhp_listed_statement: Optional[str] = None
    nrhp_urls: List[str] = Field(default_factory=list)

    construction_year: Optional[str] = None
    construction_year_urls: List[str] = Field(default_factory=list)

    original_features_statement: Optional[str] = None
    original_features_urls: List[str] = Field(default_factory=list)

    # Capacity and operations
    seating_capacity: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)

    operational_status_statement: Optional[str] = None
    operational_urls: List[str] = Field(default_factory=list)

    # Stage / technical / accessibility
    proscenium_stage_statement: Optional[str] = None
    proscenium_width: Optional[str] = None  # keep as string (e.g., "50 ft", "15m", "32 feet")
    stage_urls: List[str] = Field(default_factory=list)

    broadway_infra_statement: Optional[str] = None
    broadway_urls: List[str] = Field(default_factory=list)

    ada_wheelchair_statement: Optional[str] = None
    ada_wheelchair_urls: List[str] = Field(default_factory=list)

    accessible_restrooms_statement: Optional[str] = None
    accessible_restrooms_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theater() -> str:
    return """
You must extract exactly one theater (the first one that the answer claims meets the constraints). Return a single JSON object with the following fields. Extract only what is explicitly present in the answer text; do not invent anything.

Required identity and location:
- theater_name: string (the theater's proper name as written)
- theater_name_urls: array of URLs that explicitly identify the theater by name
- city: string (city name as written)
- state: string (full state name or 2-letter abbreviation as written)
- location_urls: array of URLs that support the specific city/state location of the theater
- major_city_support_urls: array of URLs used in the answer to support that the city is a "major" U.S. city (e.g., authoritative population/metro sources or pages that explicitly describe it as a major city)

Historical and construction:
- nrhp_listed_statement: string (answer's statement about NRHP listing, if any; otherwise null)
- nrhp_urls: array of URLs supporting the NRHP listing
- construction_year: string (the construction/opening year as written; do NOT compute)
- construction_year_urls: array of URLs supporting the construction/opening year
- original_features_statement: string (answer's statement that the theater retains significant original architectural features; otherwise null)
- original_features_urls: array of URLs supporting that original features are retained

Capacity and operations:
- seating_capacity: string (seating capacity value as written, e.g., "2,279", "about 2,500", "2k+")
- capacity_urls: array of URLs supporting the seating capacity
- operational_status_statement: string (answer's statement that it is currently operational and hosting live performances; otherwise null)
- operational_urls: array of URLs supporting current operations (e.g., schedules, events pages, seasons, box office)

Stage / technical / accessibility:
- proscenium_stage_statement: string (answer's statement that it has a proscenium stage; otherwise null)
- proscenium_width: string (stage/proscenium width as written, including unit if provided; do NOT convert)
- stage_urls: array of URLs supporting proscenium presence and width
- broadway_infra_statement: string (answer's statement about touring-Broadway-suitable technical infrastructure; otherwise null)
- broadway_urls: array of URLs supporting technical infrastructure (fly system, rigging, lighting)
- ada_wheelchair_statement: string (answer's statement about ADA wheelchair accessible seating; otherwise null)
- ada_wheelchair_urls: array of URLs supporting wheelchair accessible seating
- accessible_restrooms_statement: string (answer's statement about accessible restrooms/entrances; otherwise null)
- accessible_restrooms_urls: array of URLs supporting accessible restrooms and accessible entrances

Rules:
- Extract only URLs that are explicitly present in the answer (plain links or markdown). If none are present for a field, return an empty array for that field.
- Do not deduplicate or transform values; return them as written.
- Do not infer or compute values (e.g., do not convert meters to feet); just extract as is.
- If multiple theaters are mentioned, extract details for the first theater that the answer claims satisfies all constraints.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def nonempty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def merge_urls(*lists: List[str]) -> List[str]:
    merged = []
    for lst in lists:
        for u in lst:
            if u and u not in merged:
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_identity_and_location_nodes(evaluator: Evaluator, parent, info: TheaterExtraction) -> None:
    node = evaluator.add_parallel(
        id="Theater_Identity_And_Location",
        desc="The theater is identified and its location satisfies the Midwest + major-city constraint, with required fields and sourcing.",
        parent=parent,
        critical=True,
    )

    # Theater name provided
    evaluator.add_custom_node(
        result=nonempty(info.theater_name),
        id="Theater_Name_Provided",
        desc="The theater name is provided.",
        parent=node,
        critical=True
    )

    # Theater name URLs exist (explicit existence check to enforce 'with URL')
    evaluator.add_custom_node(
        result=nonempty_urls(info.theater_name_urls),
        id="Theater_Name_URL_Provided",
        desc="At least one theater-name reference URL is provided.",
        parent=node,
        critical=True
    )

    # Theater name supported by URL
    leaf = evaluator.add_leaf(
        id="Theater_Name_Supported_By_URL",
        desc="At least one reference URL is provided that identifies the theater by name (i.e., supports the theater identity).",
        parent=node,
        critical=True
    )
    name_claim = f"This webpage identifies the theater by the name '{info.theater_name}'."
    await evaluator.verify(
        claim=name_claim,
        node=leaf,
        sources=info.theater_name_urls,
        additional_instruction="Verify that the page explicitly names the theater as stated (allow for minor punctuation/case variations). A venue home page, Wikipedia, or official NRHP listing page are acceptable if they clearly identify the theater by this name."
    )

    # City/state provided
    evaluator.add_custom_node(
        result=nonempty(info.city) and nonempty(info.state),
        id="City_State_Location_Provided",
        desc="The theater's city and state location is provided.",
        parent=node,
        critical=True
    )

    # City/state location URLs existence
    evaluator.add_custom_node(
        result=nonempty_urls(info.location_urls) or nonempty_urls(info.theater_name_urls),
        id="City_State_Location_URL_Provided",
        desc="At least one reference URL is provided that can support the theater's city/state (location or identity URL).",
        parent=node,
        critical=True
    )

    # City/state supported by URL
    loc_leaf = evaluator.add_leaf(
        id="City_State_Location_Supported_By_URL",
        desc="At least one reference URL is provided that supports the theater's city/state location.",
        parent=node,
        critical=True
    )
    loc_sources = merge_urls(info.location_urls, info.theater_name_urls)
    loc_claim = f"The theater '{info.theater_name}' is located in {info.city}, {info.state}."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=loc_sources,
        additional_instruction="Confirm that the referenced page explicitly or clearly indicates the theater's city and state as provided. Allow common abbreviations (e.g., IL for Illinois) or well-known city nicknames if unambiguous."
    )

    # Midwest location constraint satisfied (non-web factual check allowed)
    midwest_leaf = evaluator.add_leaf(
        id="Midwest_Location_Constraint_Satisfied",
        desc="The provided city/state location is in the U.S. Midwest.",
        parent=node,
        critical=True
    )
    midwest_states_list = "Illinois (IL), Indiana (IN), Iowa (IA), Kansas (KS), Michigan (MI), Minnesota (MN), Missouri (MO), Nebraska (NE), North Dakota (ND), Ohio (OH), South Dakota (SD), Wisconsin (WI)"
    midwest_claim = f"The U.S. state '{info.state}' is in the Midwest region (commonly including: {midwest_states_list})."
    await evaluator.verify(
        claim=midwest_claim,
        node=midwest_leaf,
        additional_instruction="Use general U.S. regional knowledge. The Midwest is typically considered the 12 states listed. If the provided state matches one of them, mark as supported."
    )

    # Major city: existence of URL
    evaluator.add_custom_node(
        result=nonempty_urls(info.major_city_support_urls),
        id="Major_City_URL_Provided",
        desc='At least one supporting URL is provided for "major city" qualification.',
        parent=node,
        critical=True
    )

    # Major city with URL
    major_leaf = evaluator.add_leaf(
        id="Major_City_Constraint_Satisfied_With_URL",
        desc='A supporting reference URL is provided that substantiates the city qualifies as a "major" city.',
        parent=node,
        critical=True
    )
    major_claim = (
        f"The city of {info.city}, {info.state} is a major U.S. city or a principal city of a major metropolitan area "
        f"(e.g., clearly described as a major city, or population/metro size indicates major-city status)."
    )
    await evaluator.verify(
        claim=major_claim,
        node=major_leaf,
        sources=info.major_city_support_urls,
        additional_instruction=(
            "Accept if the page clearly describes the city as a major city, or shows population/metro statistics that "
            "reasonably indicate major-city status (e.g., population well over 100,000 or principal city of a large metro). "
            "Census, Wikipedia, or city government pages are acceptable if they support the claim."
        )
    )


async def build_historical_and_construction_nodes(evaluator: Evaluator, parent, info: TheaterExtraction) -> None:
    node = evaluator.add_parallel(
        id="Historical_And_Construction_Requirements",
        desc="The theater meets NRHP and 1920s construction requirements, with documentation.",
        parent=parent,
        critical=True,
    )

    # NRHP URL provided
    evaluator.add_custom_node(
        result=nonempty_urls(info.nrhp_urls),
        id="NRHP_URL_Provided",
        desc="At least one NRHP support URL is provided.",
        parent=node,
        critical=True
    )

    # NRHP listed with URL
    nrhp_leaf = evaluator.add_leaf(
        id="NRHP_Listed_With_URL",
        desc="The theater is listed on the National Register of Historic Places and a supporting reference URL is provided.",
        parent=node,
        critical=True
    )
    nrhp_claim = f"The theater '{info.theater_name}' is listed on the National Register of Historic Places (NRHP)."
    await evaluator.verify(
        claim=nrhp_claim,
        node=nrhp_leaf,
        sources=info.nrhp_urls,
        additional_instruction="Look for explicit NRHP listing language on official NPS pages, State Historic Preservation Office pages, NRHP nomination PDFs, or reliable references (e.g., Wikipedia with citations)."
    )

    # Construction year provided
    evaluator.add_custom_node(
        result=nonempty(info.construction_year),
        id="Construction_Year_Provided",
        desc="A specific construction/opening year is provided.",
        parent=node,
        critical=True
    )

    # Construction year URL provided
    evaluator.add_custom_node(
        result=nonempty_urls(info.construction_year_urls),
        id="Construction_Year_URL_Provided",
        desc="A supporting construction/opening year URL is provided.",
        parent=node,
        critical=True
    )

    # Construction year in the 1920s with URL
    year_leaf = evaluator.add_leaf(
        id="Construction_Year_Provided_And_1920s_With_URL",
        desc="A specific construction/opening year is provided; it falls between 1920 and 1929 (inclusive); and a supporting reference URL is provided.",
        parent=node,
        critical=True
    )
    year_claim = (
        f"The theater '{info.theater_name}' was constructed/opened in {info.construction_year}, "
        f"which is between 1920 and 1929 inclusive."
    )
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=info.construction_year_urls,
        additional_instruction=(
            "Confirm that the year given is explicitly stated on the referenced page and that it lies within 1920–1929 inclusive. "
            "Allow reasonable synonyms like 'opened', 'built', or 'completed' when referring to the original construction year."
        )
    )

    # Original architectural features retained: treat as critical due to framework constraints (critical parent cannot have non-critical child)
    # Existence URL check
    evaluator.add_custom_node(
        result=nonempty_urls(info.original_features_urls),
        id="Original_Architectural_Features_URL_Provided",
        desc="A supporting reference URL is provided for retaining significant original architectural features.",
        parent=node,
        critical=True
    )
    features_leaf = evaluator.add_leaf(
        id="Original_Architectural_Features_Retained_With_URL",
        desc="A supporting reference URL is provided indicating the theater retains significant original architectural features from its construction period.",
        parent=node,
        critical=True
    )
    features_claim = (
        f"The theater '{info.theater_name}' retains significant original architectural features from its original construction period "
        f"(e.g., original interior decor, ornamental plaster, historic proscenium, or other defining elements)."
    )
    await evaluator.verify(
        claim=features_claim,
        node=features_leaf,
        sources=info.original_features_urls,
        additional_instruction="Accept mentions of preserved/restored original features documented by NRHP nominations, official pages, reputable histories, or venue guides."
    )


async def build_capacity_and_operations_nodes(evaluator: Evaluator, parent, info: TheaterExtraction) -> None:
    node = evaluator.add_parallel(
        id="Capacity_And_Operations_Requirements",
        desc="The theater meets seating capacity and operational/live-performance requirements, with documentation.",
        parent=parent,
        critical=True,
    )

    # Seating capacity provided
    evaluator.add_custom_node(
        result=nonempty(info.seating_capacity),
        id="Seating_Capacity_Provided",
        desc="A specific seating capacity value is provided.",
        parent=node,
        critical=True
    )

    # Seating capacity URL provided
    evaluator.add_custom_node(
        result=nonempty_urls(info.capacity_urls),
        id="Seating_Capacity_URL_Provided",
        desc="A supporting URL for seating capacity is provided.",
        parent=node,
        critical=True
    )

    # Seating capacity >= 2000 with URL
    capacity_leaf = evaluator.add_leaf(
        id="Seating_Capacity_Provided_And_AtLeast_2000_With_URL",
        desc="A specific seating capacity is provided; it is at least 2,000; and a supporting reference URL is provided.",
        parent=node,
        critical=True
    )
    capacity_claim = (
        f"The theater '{info.theater_name}' has a seating capacity of {info.seating_capacity}, "
        f"which is at least 2,000 seats."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=info.capacity_urls,
        additional_instruction="Extract or interpret the seat count on the page. If the value is >= 2000 (allowing formatting like '2,200' or 'about 2,100'), mark supported."
    )

    # Operational URLs provided
    evaluator.add_custom_node(
        result=nonempty_urls(info.operational_urls),
        id="Operational_URL_Provided",
        desc="A supporting URL for current operations/live performances is provided.",
        parent=node,
        critical=True
    )

    # Currently operational and hosting live performances with URL
    operational_leaf = evaluator.add_leaf(
        id="Currently_Operational_Live_Performances_With_URL",
        desc="The theater is currently operational and actively hosting live performances, and a supporting reference URL is provided.",
        parent=node,
        critical=True
    )
    operational_claim = (
        f"The theater '{info.theater_name}' is currently operational and actively hosts live performances "
        f"(e.g., shows, concerts, or a current season schedule)."
    )
    await evaluator.verify(
        claim=operational_claim,
        node=operational_leaf,
        sources=info.operational_urls,
        additional_instruction="Look for current or recent schedules, events pages, box office/ticketing, or season announcements demonstrating ongoing live performances."
    )


async def build_stage_and_accessibility_nodes(evaluator: Evaluator, parent, info: TheaterExtraction) -> None:
    node = evaluator.add_parallel(
        id="Stage_Technical_And_Accessibility_Requirements",
        desc="The theater meets stage, technical infrastructure, and accessibility requirements, with documentation.",
        parent=parent,
        critical=True,
    )

    # Proscenium width and stage URLs provided
    evaluator.add_custom_node(
        result=nonempty(info.proscenium_width) and nonempty_urls(info.stage_urls),
        id="Proscenium_Stage_Width_And_URL_Provided",
        desc="Proscenium/stage width is provided and at least one stage-specification URL is provided.",
        parent=node,
        critical=True
    )

    # Proscenium stage and width >= 30 ft with URL
    prosc_leaf = evaluator.add_leaf(
        id="Proscenium_Stage_And_Width_AtLeast_30ft_With_URL",
        desc="The theater has a proscenium stage; the proscenium/stage width is at least 30 feet; and a supporting reference URL is provided.",
        parent=node,
        critical=True
    )
    prosc_claim = (
        f"The theater '{info.theater_name}' has a proscenium stage with width at least 30 feet. "
        f"The reported width is '{info.proscenium_width}'."
    )
    await evaluator.verify(
        claim=prosc_claim,
        node=prosc_leaf,
        sources=info.stage_urls,
        additional_instruction=(
            "Confirm two things on the page: (1) it is a proscenium stage; and (2) the width is >= 30 ft. "
            "If width is given in meters, convert approximately (30 ft ≈ 9.14 m). "
            "Accept if the width clearly meets or exceeds 30 ft (e.g., 32', 40', 10 m)."
        )
    )

    # Broadway touring infrastructure URL provided
    evaluator.add_custom_node(
        result=nonempty_urls(info.broadway_urls),
        id="Broadway_Infrastructure_URL_Provided",
        desc="A supporting URL is provided indicating touring-Broadway-suitable technical infrastructure.",
        parent=node,
        critical=True
    )

    # Broadway touring technical infrastructure with URL
    broad_leaf = evaluator.add_leaf(
        id="Broadway_Touring_Technical_Infrastructure_With_URL",
        desc="A supporting reference URL is provided indicating the theater has touring-Broadway-suitable technical infrastructure (e.g., fly system, rigging, lighting).",
        parent=node,
        critical=True
    )
    broad_claim = (
        f"The theater '{info.theater_name}' has technical infrastructure suitable for touring Broadway "
        f"(e.g., fly system, rigging capacity, and professional lighting)."
    )
    await evaluator.verify(
        claim=broad_claim,
        node=broad_leaf,
        sources=info.broadway_urls,
        additional_instruction="Look for venue tech specs, stage manuals, riders, or official descriptions indicating fly system/rigging/lighting suitable for touring Broadway shows."
    )

    # ADA wheelchair seating URL provided
    evaluator.add_custom_node(
        result=nonempty_urls(info.ada_wheelchair_urls),
        id="ADA_Wheelchair_URL_Provided",
        desc="A supporting URL for ADA-compliant wheelchair accessible seating is provided.",
        parent=node,
        critical=True
    )

    # ADA wheelchair seating with URL
    ada_seat_leaf = evaluator.add_leaf(
        id="ADA_Wheelchair_Seating_With_URL",
        desc="The theater provides ADA-compliant wheelchair accessible seating and a supporting reference URL is provided.",
        parent=node,
        critical=True
    )
    ada_seat_claim = f"The theater '{info.theater_name}' provides ADA-compliant wheelchair accessible seating."
    await evaluator.verify(
        claim=ada_seat_claim,
        node=ada_seat_leaf,
        sources=info.ada_wheelchair_urls,
        additional_instruction="Look for explicit mentions of wheelchair seating, ADA seating locations, companion seating, or accessibility statements on official pages."
    )

    # Accessible restrooms and entrances URL provided
    evaluator.add_custom_node(
        result=nonempty_urls(info.accessible_restrooms_urls),
        id="Accessible_Restrooms_Entrances_URL_Provided",
        desc="A supporting URL for accessible restrooms and entrances is provided.",
        parent=node,
        critical=True
    )

    # Accessible restrooms and entrances with URL
    access_leaf = evaluator.add_leaf(
        id="Accessible_Restrooms_And_Entrances_With_URL",
        desc="The theater has accessible restrooms and entrances and a supporting reference URL is provided.",
        parent=node,
        critical=True
    )
    access_claim = f"The theater '{info.theater_name}' has accessible restrooms and accessible entrances."
    await evaluator.verify(
        claim=access_claim,
        node=access_leaf,
        sources=info.accessible_restrooms_urls,
        additional_instruction="Confirm explicit statements about accessible/ADA restrooms and accessible entrances. Accept venue accessibility pages or policy pages that clearly state these features."
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
    Evaluate an answer for the 'midwest_historic_theater' task using the Mind2Web2 evaluation framework.
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
        default_model=model,
    )

    # Create a critical task root under the framework root (framework root itself is non-critical by design)
    task_root = evaluator.add_parallel(
        id="Root",
        desc="Identify one historic theater in a major Midwest U.S. city that satisfies all stated constraints and provide required fields with supporting reference URLs.",
        parent=root,
        critical=True,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_theater(),
        template_class=TheaterExtraction,
        extraction_name="theater_extraction",
    )

    # Build verification subtrees
    await build_identity_and_location_nodes(evaluator, task_root, extracted)
    await build_historical_and_construction_nodes(evaluator, task_root, extracted)
    await build_capacity_and_operations_nodes(evaluator, task_root, extracted)
    await build_stage_and_accessibility_nodes(evaluator, task_root, extracted)

    # Return summary
    return evaluator.get_summary()