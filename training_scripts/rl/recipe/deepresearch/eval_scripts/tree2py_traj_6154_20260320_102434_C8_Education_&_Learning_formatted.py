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
TASK_ID = "ohio_universities_facilities"
TASK_DESCRIPTION = """I am researching universities in Ohio for a prospective undergraduate student who values comprehensive campus facilities and services. Please identify three (3) universities located in Ohio that meet all of the following criteria:

1. **Library Access**: The university has at least one library location that offers extended study hours, specifically operating past midnight or providing 24-hour access during regular academic terms.

2. **Recreation Facilities**: The university operates a campus recreation center that includes swimming pool facilities (such as lap pools, aquatic complexes, or similar aquatic amenities).

3. **Student Housing**: The university provides on-campus residential housing options that are available to first-year (freshman) students.

4. **Dining Services**: The university has at least two (2) separate dining hall locations that offer meal plan options for students.

5. **Parking**: The university offers parking permits that students can purchase to park on campus.

For each of the three universities you identify, please provide:
- The university's official name
- The name and location of the library with extended/24-hour access, along with a link to the official page confirming the hours
- The name of the recreation center and confirmation that it has pool facilities, along with a link to the official page
- Confirmation of first-year student housing availability, along with a link to the official housing information page
- The names of at least two dining hall locations with meal plans, along with a link to the official dining services page
- Confirmation that student parking permits are available, along with a link to the official parking services page
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LibraryInfo(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    hours_url: Optional[str] = None


class RecreationInfo(BaseModel):
    name: Optional[str] = None
    rec_url: Optional[str] = None
    pool_statement: Optional[str] = None


class HousingInfo(BaseModel):
    url: Optional[str] = None
    statement: Optional[str] = None


class DiningInfo(BaseModel):
    halls: List[str] = Field(default_factory=list)
    dining_url: Optional[str] = None
    meal_plan_statement: Optional[str] = None


class ParkingInfo(BaseModel):
    url: Optional[str] = None
    statement: Optional[str] = None


class UniversityItem(BaseModel):
    name: Optional[str] = None
    library: Optional[LibraryInfo] = None
    recreation: Optional[RecreationInfo] = None
    housing: Optional[HousingInfo] = None
    dining: Optional[DiningInfo] = None
    parking: Optional[ParkingInfo] = None
    extra_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract up to the first three (3) universities presented in the answer that are located in Ohio and for each, extract the following structured fields exactly as provided in the answer. Do not invent information. If a field is missing, set it to null (or [] for lists).

For each university, extract:
- name: The university's official name as stated in the answer.
- library:
    - name: The library (or library location/space) that offers late-night or 24-hour access.
    - location: The stated physical location of that library if provided (building, address, campus area).
    - hours_url: The direct official URL that confirms extended hours past midnight or 24-hour/overnight access (24/5, 24/7, etc.). Prefer a page clearly listing hours or 24-hour study areas.
- recreation:
    - name: The campus recreation center name.
    - rec_url: The direct official URL for the recreation center.
    - pool_statement: Any explicit mention in the answer about pool/aquatic facilities.
- housing:
    - url: The official housing information page URL where first-year/freshman housing availability is described.
    - statement: Any explicit mention in the answer confirming first-year housing availability.
- dining:
    - halls: A list of at least two dining hall location names if provided; if more are listed, include them all; if only one is present, include that one.
    - dining_url: The official dining services page URL that lists locations and/or meal plan information.
    - meal_plan_statement: Any explicit mention of meal plans in the answer.
- parking:
    - url: The official parking services/permits URL indicating student permits can be purchased.
    - statement: Any explicit mention confirming student parking permits are available.
- extra_urls: Any additional official URLs in the answer for this university (e.g., about pages, location pages). Include only URLs explicitly present.

Rules:
- Only extract URLs that are explicitly present in the answer. If a URL is missing a protocol, prepend http://.
- Prefer official university domains (often .edu) if multiple links are given; choose the most specific page for the required confirmation (e.g., an hours page for library hours).
- Do not add universities beyond those actually mentioned in the answer. Preserve order of appearance.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    s = u.strip()
    if not s:
        return None
    if not (s.startswith("http://") or s.startswith("https://")):
        s = "http://" + s
    return s


def _unique_nonempty(urls: List[Optional[str]]) -> List[str]:
    seq = [x for x in ( _norm_url(u) for u in urls ) if x]
    # de-duplicate preserving order
    out: List[str] = []
    seen = set()
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _ordinal(i: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][i] if i < 5 else f"#{i+1}"


# --------------------------------------------------------------------------- #
# University verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx_zero_based: int,
) -> None:
    ord_word = _ordinal(idx_zero_based)
    uni_node = evaluator.add_parallel(
        id=f"university_{idx_zero_based + 1}",
        desc=f"{ord_word} university meeting all criteria",
        parent=parent_node,
        critical=False,  # allow partial credit across universities
    )

    # Prepare URL references
    lib = uni.library or LibraryInfo()
    rec = uni.recreation or RecreationInfo()
    housing = uni.housing or HousingInfo()
    dining = uni.dining or DiningInfo()
    parking = uni.parking or ParkingInfo()

    lib_url = _norm_url(lib.hours_url)
    rec_url = _norm_url(rec.rec_url)
    housing_url = _norm_url(housing.url)
    dining_url = _norm_url(dining.dining_url)
    parking_url = _norm_url(parking.url)

    # Create "reference provided" custom nodes first (critical)
    evaluator.add_custom_node(
        result=bool(lib_url),
        id=f"library_reference_{idx_zero_based + 1}",
        desc="Provides official URL confirming library hours",
        parent=uni_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(rec_url),
        id=f"pool_reference_{idx_zero_based + 1}",
        desc="Provides official URL confirming pool facilities",
        parent=uni_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(housing_url),
        id=f"housing_reference_{idx_zero_based + 1}",
        desc="Provides official URL confirming freshman housing availability",
        parent=uni_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(dining_url),
        id=f"dining_reference_{idx_zero_based + 1}",
        desc="Provides official URL confirming dining locations and meal plans",
        parent=uni_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(parking_url),
        id=f"parking_reference_{idx_zero_based + 1}",
        desc="Provides official URL confirming student parking permit availability",
        parent=uni_node,
        critical=True,
    )

    # Build leaf nodes for factual verification
    ohio_node = evaluator.add_leaf(
        id=f"ohio_location_{idx_zero_based + 1}",
        desc="University is located in Ohio",
        parent=uni_node,
        critical=True,
    )
    library_hours_node = evaluator.add_leaf(
        id=f"library_hours_{idx_zero_based + 1}",
        desc="Has at least one library with extended hours (past midnight or 24-hour access)",
        parent=uni_node,
        critical=True,
    )
    pool_node = evaluator.add_leaf(
        id=f"pool_facility_{idx_zero_based + 1}",
        desc="Has a campus recreation center with swimming pool facilities",
        parent=uni_node,
        critical=True,
    )
    housing_node = evaluator.add_leaf(
        id=f"freshman_housing_{idx_zero_based + 1}",
        desc="Provides on-campus housing available for first-year students",
        parent=uni_node,
        critical=True,
    )
    dining_node = evaluator.add_leaf(
        id=f"dining_halls_{idx_zero_based + 1}",
        desc="Has at least two dining hall locations with meal plan options",
        parent=uni_node,
        critical=True,
    )
    parking_node = evaluator.add_leaf(
        id=f"parking_permits_{idx_zero_based + 1}",
        desc="Offers parking permits available for student purchase",
        parent=uni_node,
        critical=True,
    )

    # Construct claims and sources
    # 1) Ohio location
    all_sources_for_location = _unique_nonempty([lib_url, rec_url, housing_url, dining_url, parking_url] + uni.extra_urls)
    uni_name = uni.name or "the university"
    ohio_claim = f"The university '{uni_name}' is located in the state of Ohio (OH), United States."
    ohio_add_ins = (
        "Confirm that the referenced official page indicates the institution is in Ohio. "
        "Accept evidence such as an address containing ', OH' or 'Ohio', phrases like 'in City, Ohio', "
        "or a clearly Ohio-based campus description. Minor variations are acceptable."
    )

    # 2) Library extended hours
    lib_name = lib.name or "the library"
    loc_fragment = f" located at '{lib.location}'" if (lib.location and lib.location.strip()) else ""
    lib_claim = (
        f"The library '{lib_name}'{loc_fragment} offers extended study hours past midnight or provides 24-hour access "
        f"during regular academic terms."
    )
    lib_add_ins = (
        "Determine whether the page shows overnight/extended access such as '24/7', '24/5', 'open until 2 a.m.', "
        "'24-hour reading room', or similar during regular academic terms. If a dedicated 24-hour study space within "
        "the library is provided, that satisfies the requirement."
    )

    # 3) Recreation pool facilities
    rec_name = rec.name or "the campus recreation center"
    pool_claim = (
        f"The campus recreation facility '{rec_name}' includes a swimming pool or aquatic facility "
        f"(e.g., lap pool, leisure pool, natatorium, aquatics center)."
    )
    pool_add_ins = (
        "Verify that the referenced recreation facility includes a swimming pool or aquatic facility. "
        "Accept wording such as 'pool', 'natatorium', 'lap pool', 'leisure pool', or 'aquatics'."
    )

    # 4) First-year housing available
    housing_claim = (
        "The university provides on-campus residential housing that is available to first-year (freshman) students."
    )
    housing_add_ins = (
        "Look for statements like 'first-year students live on campus', 'freshmen are required/eligible to live in residence halls', "
        "'guaranteed housing for first-year students', or equivalent language."
    )

    # 5) Dining halls (>=2) with meal plans
    halls = dining.halls or []
    halls_preview = ", ".join(halls[:2]) if halls else ""
    if halls_preview:
        dining_claim = (
            f"The university has at least two distinct dining hall locations that participate in student meal plans, "
            f"including: {halls_preview}."
        )
    else:
        dining_claim = (
            "The university has at least two distinct dining hall locations that participate in student meal plans."
        )
    dining_add_ins = (
        "Verify that the referenced page lists at least two distinct dining halls/locations and indicates they accept or are part of "
        "student meal plans. Accept synonyms like 'dining center', 'dining commons', 'market', 'cafeteria', and meal-plan terms like "
        "'board plan' or 'meal swipes'. The specific names in the claim should be supported by the page."
    )

    # 6) Student parking permits
    parking_claim = (
        "Students can purchase parking permits to park on campus (e.g., resident/commuter permits, semester/annual or virtual permits)."
    )
    parking_add_ins = (
        "Confirm that the referenced page indicates parking permits are available for students to purchase for on-campus parking."
    )

    # Batch verify factual leaves
    claims_and_sources = [
        (ohio_claim, all_sources_for_location if all_sources_for_location else None, ohio_node, ohio_add_ins),
        (lib_claim, lib_url, library_hours_node, lib_add_ins),
        (pool_claim, rec_url, pool_node, pool_add_ins),
        (housing_claim, housing_url, housing_node, housing_add_ins),
        (dining_claim, dining_url, dining_node, dining_add_ins),
        (parking_claim, parking_url, parking_node, parking_add_ins),
    ]
    await evaluator.batch_verify(claims_and_sources)


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
    # Initialize evaluator (root as parallel aggregator; root kept non-critical to allow partial credit)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify three Ohio universities that meet all specified facility and service criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured university information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Ensure exactly three slots (pad with empty if fewer)
    universities = list(extracted.universities)[:3] if extracted and extracted.universities else []
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Record simple GT/context info (optional)
    evaluator.add_ground_truth({
        "requirements": {
            "state": "Ohio",
            "library_hours": "Past midnight or 24-hour access during regular terms",
            "recreation_pool": "Swimming pool/aquatic facility available",
            "first_year_housing": True,
            "dining_halls_min": 2,
            "parking_permits": True
        },
        "num_universities_expected": 3
    })

    # Build and verify each university subtree
    for i, uni in enumerate(universities):
        await verify_university(evaluator, root, uni, i)

    # Return final structured summary
    return evaluator.get_summary()