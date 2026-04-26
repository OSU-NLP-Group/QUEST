import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sk24j_osaa_cvc_one_school"
TASK_DESCRIPTION = (
    "Identify one high school in the Salem-Keizer School District 24J that competes in the OSAA 6A-6 Central Valley "
    "Conference and has a documented home football stadium or field facility. Provide the following information about "
    "this school: (1) The school's full official name, (2) The complete street address (including street number, street "
    "name, city, state, and ZIP code), (3) The official school mascot or team nickname, (4) Confirmation of athletic "
    "conference affiliation (OSAA 6A-6 Central Valley Conference), (5) The name of the school's home football stadium "
    "or field facility, (6) The grades served by the school, (7) The city location (Salem or Keizer), (8) The stadium's "
    "seating capacity, if this information is available in documented sources."
)

ROOT_NODE_DESC = (
    "Identify one qualifying Salem-Keizer SD 24J high school in OSAA 6A-6 Central Valley Conference with a documented "
    "home football stadium/field facility, and provide all required school/stadium details."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SchoolURLs(BaseModel):
    school_urls: List[str] = Field(default_factory=list)
    osaa_urls: List[str] = Field(default_factory=list)
    district_urls: List[str] = Field(default_factory=list)
    stadium_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class SchoolExtraction(BaseModel):
    school_name: Optional[str] = None
    district: Optional[str] = None
    conference: Optional[str] = None

    grades: Optional[str] = None

    address_full: Optional[str] = None
    address_street: Optional[str] = None
    address_city: Optional[str] = None
    address_state: Optional[str] = None
    address_zip: Optional[str] = None

    mascot: Optional[str] = None

    stadium_name: Optional[str] = None
    stadium_capacity: Optional[str] = None  # keep as string; may include separators or approx text

    city_location: Optional[str] = None  # expected "Salem" or "Keizer"

    urls: SchoolURLs = Field(default_factory=SchoolURLs)

    # fallback generic sources extracted from answer (if any general list is provided)
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_school_info() -> str:
    return """
Extract exactly one high school described in the answer that belongs to the Salem-Keizer School District 24J and competes in the OSAA 6A-6 Central Valley Conference. If the answer mentions multiple schools, extract the first one only.

Return the following fields:
- school_name: The school's full official name as written in the answer.
- district: The school district name as stated (e.g., "Salem-Keizer Public Schools", "Salem-Keizer School District 24J").
- conference: The conference affiliation as stated (e.g., "OSAA 6A-6 Central Valley Conference", "6A-6 Central Valley", "Central Valley Conference (6A)").
- grades: The grades served string as stated (prefer normalized "9-12" or "9–12" if applicable).
- address_full: The full street address string (if present) as written in the answer.
- address_street: The street number and street name (e.g., "4700 Keubler Blvd SE").
- address_city: The city (must be "Salem" or "Keizer" if provided).
- address_state: The state abbreviation (e.g., "OR").
- address_zip: The 5-digit ZIP code (e.g., "97302").
- mascot: The official school mascot or team nickname as stated.
- stadium_name: The name of the school's home football stadium or field facility (must be a specific facility name if provided).
- stadium_capacity: The stadium seating capacity value or string if provided in the answer (otherwise null).
- city_location: Explicitly set to "Salem" or "Keizer" based on the answer's statement of city location. If not explicitly stated, try to infer from the address; otherwise set to null.

Also extract URLs explicitly present in the answer into categorized lists:
- urls.school_urls: URLs from the school's official site or athletics subpages for this school.
- urls.osaa_urls: URLs from the OSAA website relevant to this school.
- urls.district_urls: URLs from the district website about this school.
- urls.stadium_urls: URLs specifically about the stadium/field facility (could be school/district pages or other reputable sources).
- urls.capacity_urls: URLs that explicitly mention the stadium seating capacity.
- urls.other_urls: Any other relevant URLs cited in the answer but not fitting above categories.

Additionally, if the answer has a general "sources" section or additional URLs, include them in the top-level 'sources' array.

Important notes:
- Only extract data explicitly mentioned in the answer. Do not invent or infer values beyond what is stated.
- For URL fields, extract actual URLs (including protocol). For markdown links, extract the target URL.
- If any field is not present in the answer, set it to null. For arrays, return empty arrays if none are provided.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _looks_like_complete_address(info: SchoolExtraction) -> bool:
    return all([
        _nonempty(info.address_street),
        _nonempty(info.address_city),
        _nonempty(info.address_state),
        _nonempty(info.address_zip),
    ])


def _norm_city(s: Optional[str]) -> Optional[str]:
    if not _nonempty(s):
        return None
    t = s.strip().lower()
    if t in ("salem", "city of salem", "salem, or", "salem oregon"):
        return "Salem"
    if t in ("keizer", "city of keizer", "keizer, or", "keizer oregon"):
        return "Keizer"
    return None


def _merge_sources(info: SchoolExtraction, prefer_order: Optional[List[str]] = None) -> List[str]:
    """
    Merge all extracted source URLs into a single list, preserving priority then order, deduplicated.
    prefer_order is a list of attribute names on info.urls or "sources" to front-load.
    """
    seen = set()
    merged: List[str] = []

    def add_list(urls: List[str]):
        for u in urls:
            if not _nonempty(u):
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)

    # Default priority
    default_order = [
        "urls.osaa_urls",
        "urls.school_urls",
        "urls.district_urls",
        "urls.stadium_urls",
        "urls.capacity_urls",
        "sources",
        "urls.other_urls",
    ]
    order = prefer_order if prefer_order else default_order

    # Resolve attribute paths
    for key in order:
        try:
            if key.startswith("urls."):
                field = key.split(".", 1)[1]
                add_list(getattr(info.urls, field, []))
            elif key == "sources":
                add_list(info.sources or [])
            else:
                # Unknown key -> ignore silently
                pass
        except Exception:
            pass

    return merged


def _sources_for(info: SchoolExtraction, focus: str) -> List[str]:
    """
    Get prioritized sources for a particular verification focus.
    focus options:
      - "district" -> district + school + osaa
      - "conference" -> osaa + school + other
      - "grades" -> school + district + osaa
      - "address" -> school + district + osaa
      - "mascot" -> school + osaa + district
      - "stadium" -> stadium + school + district + other
      - "capacity" -> capacity + stadium + school + other
      - "location" -> school + district + osaa
      - default -> merge all
    """
    mapping = {
        "district": ["urls.district_urls", "urls.school_urls", "urls.osaa_urls", "sources", "urls.other_urls"],
        "conference": ["urls.osaa_urls", "urls.school_urls", "urls.district_urls", "sources", "urls.other_urls"],
        "grades": ["urls.school_urls", "urls.district_urls", "urls.osaa_urls", "sources", "urls.other_urls"],
        "address": ["urls.school_urls", "urls.district_urls", "urls.osaa_urls", "sources", "urls.other_urls"],
        "mascot": ["urls.school_urls", "urls.osaa_urls", "urls.district_urls", "sources", "urls.other_urls"],
        "stadium": ["urls.stadium_urls", "urls.school_urls", "urls.district_urls", "sources", "urls.other_urls"],
        "capacity": ["urls.capacity_urls", "urls.stadium_urls", "urls.school_urls", "sources", "urls.other_urls"],
        "location": ["urls.school_urls", "urls.district_urls", "urls.osaa_urls", "sources", "urls.other_urls"],
    }
    prefer = mapping.get(focus, None)
    return _merge_sources(info, prefer)


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, info: SchoolExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    All child nodes under the critical aggregator are also critical to satisfy consistency constraints.
    """
    # Top-level critical aggregator under root (root itself is non-critical by design of Evaluator.initialize)
    task_node = evaluator.add_parallel(
        id="task_main",
        desc=ROOT_NODE_DESC,
        parent=None,
        critical=True  # critical parent → all children must be critical
    )

    school_name = info.school_name or ""

    # 1) School name provided (existence)
    evaluator.add_custom_node(
        result=_nonempty(info.school_name),
        id="school_name_provided",
        desc="The school's full official name is provided.",
        parent=task_node,
        critical=True
    )

    # 2) District membership
    district_node = evaluator.add_leaf(
        id="district_membership",
        desc="The identified school is an officially recognized high school in the Salem-Keizer School District 24J.",
        parent=task_node,
        critical=True
    )
    district_claim = (
        f"The school named '{school_name}' is a high school in the Salem-Keizer School District 24J "
        f"(also known as Salem-Keizer Public Schools or SKPS)."
    )
    await evaluator.verify(
        claim=district_claim,
        node=district_node,
        sources=_sources_for(info, "district"),
        additional_instruction=(
            "Confirm that the school belongs to the Salem-Keizer Public Schools (Salem-Keizer School District 24J). "
            "Accept mentions like 'Salem-Keizer Public Schools', 'Salem-Keizer SD', or '24J'. "
            "It should clearly be a high school in this district."
        ),
    )

    # 3) Conference affiliation (OSAA 6A-6 Central Valley Conference)
    conf_node = evaluator.add_leaf(
        id="conference_affiliation",
        desc="The school competes in the OSAA 6A-6 Central Valley Conference (conference affiliation is stated/confirmed).",
        parent=task_node,
        critical=True
    )
    conf_claim = f"'{school_name}' competes in the OSAA 6A-6 Central Valley Conference."
    await evaluator.verify(
        claim=conf_claim,
        node=conf_node,
        sources=_sources_for(info, "conference"),
        additional_instruction=(
            "Verify that the school is in the 'Central Valley Conference' at the 6A classification, "
            "often denoted as '6A-6'. Accept reasonable variants like '6A-6 Central Valley', 'Central Valley Conference (6A)', "
            "or 'CVC' on OSAA or school/district pages."
        ),
    )

    # 4) Grade levels (ensure 9–12)
    grades_node = evaluator.add_leaf(
        id="grade_levels",
        desc="The school serves grades 9–12 as a traditional comprehensive high school (grades served are provided and match 9–12).",
        parent=task_node,
        critical=True
    )
    # We validate the requirement by checking explicit 9–12 against sources.
    grades_claim = f"'{school_name}' serves grades 9–12 (a standard comprehensive high school)."
    await evaluator.verify(
        claim=grades_claim,
        node=grades_node,
        sources=_sources_for(info, "grades"),
        additional_instruction=(
            "Confirm that the school is a standard high school serving grades 9–12. "
            "Allow small formatting variants like '9-12' or 'grades 9 through 12'. "
            "If the evidence shows a different grade span, mark as incorrect."
        ),
    )

    # 5) Complete address provided (existence of all components)
    evaluator.add_custom_node(
        result=_looks_like_complete_address(info),
        id="complete_address",
        desc="A complete street address is provided, including street number, street name, city, state, and ZIP code.",
        parent=task_node,
        critical=True
    )

    # 6) Mascot / nickname
    mascot_node = evaluator.add_leaf(
        id="mascot_nickname",
        desc="The official school mascot or team nickname is provided.",
        parent=task_node,
        critical=True
    )
    mascot_val = info.mascot or ""
    mascot_claim = f"The official mascot or team nickname of '{school_name}' is '{mascot_val}'."
    await evaluator.verify(
        claim=mascot_claim,
        node=mascot_node,
        sources=_sources_for(info, "mascot"),
        additional_instruction=(
            "Verify the team's official mascot or nickname on school, district, or OSAA pages. "
            "Allow pluralization (e.g., 'Titans' vs 'Titan') and typical naming conventions."
        ),
    )

    # 7) Home football stadium/field name
    stadium_node = evaluator.add_leaf(
        id="home_stadium_name",
        desc="The name of the school's home football stadium or field facility is provided (and is a documented/identifiable facility name, not an unspecified/generic description).",
        parent=task_node,
        critical=True
    )
    stadium_name = info.stadium_name or ""
    stadium_claim = (
        f"The home football stadium or field facility for '{school_name}' is named '{stadium_name}'."
    )
    await evaluator.verify(
        claim=stadium_claim,
        node=stadium_node,
        sources=_sources_for(info, "stadium"),
        additional_instruction=(
            "Confirm that the named facility is the school's home football stadium or field. "
            "Accept variants like 'Field', 'Stadium', or named complex fields if they clearly serve as the football home field."
        ),
    )

    # 8) Location requirement (Salem or Keizer in Marion County)
    location_node = evaluator.add_leaf(
        id="location_requirement",
        desc="The school is physically located in Marion County, Oregon within Salem or Keizer city limits.",
        parent=task_node,
        critical=True
    )
    city_for_claim = _norm_city(info.city_location) or _norm_city(info.address_city) or "Salem"
    location_claim = (
        f"'{school_name}' is located in {city_for_claim}, Oregon, within that city's limits (Marion County)."
    )
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=_sources_for(info, "location"),
        additional_instruction=(
            "Verify that the school's official address indicates the city is either Salem or Keizer, Oregon. "
            "If the page explicitly mentions the county as Marion County, use that; if county is not mentioned, "
            "focus on confirming the city (Salem/Keizer). Fail only if the evidence contradicts Salem/Keizer city location."
        ),
    )

    # 9) City location explicitly provided in the answer (existence)
    city_norm = _norm_city(info.city_location) or _norm_city(info.address_city)
    evaluator.add_custom_node(
        result=bool(city_norm in ("Salem", "Keizer")),
        id="city_location_provided",
        desc="The answer explicitly states whether the school is in Salem or Keizer.",
        parent=task_node,
        critical=True
    )

    # 10) Stadium capacity if available
    if _nonempty(info.stadium_capacity):
        cap_node = evaluator.add_leaf(
            id="stadium_capacity_if_available",
            desc="If the stadium seating capacity is available in documented sources, it is provided.",
            parent=task_node,
            critical=True
        )
        cap_str = (info.stadium_capacity or "").strip()
        cap_claim = (
            f"The seating capacity of the home football stadium/field for '{school_name}'"
            f" ('{stadium_name}') is {cap_str}."
        )
        await evaluator.verify(
            claim=cap_claim,
            node=cap_node,
            sources=_sources_for(info, "capacity"),
            additional_instruction=(
                "Verify that a seating capacity value is stated for the stadium/field in the provided sources. "
                "Allow reasonable formatting differences (e.g., commas, approximate wording like '~5,000'). "
                "If multiple numbers exist, ensure the number corresponds to the stated seating capacity."
            ),
        )
    else:
        # If not available/provided, the requirement says 'if available'. In such case, pass this check
        # because the answer is not penalized if capacity is not available in sources.
        evaluator.add_custom_node(
            result=True,
            id="stadium_capacity_if_available",
            desc="If the stadium seating capacity is available in documented sources, it is provided.",
            parent=task_node,
            critical=True
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
) -> Dict:
    """
    Evaluate an answer for the Salem-Keizer SD 24J high school + OSAA 6A-6 CVC + stadium details task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # children independent at top level
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

    # Extract structured information about the selected school from the answer
    extracted: SchoolExtraction = await evaluator.extract(
        prompt=prompt_extract_school_info(),
        template_class=SchoolExtraction,
        extraction_name="school_extraction",
    )

    # Build verification nodes and run verifications
    await build_and_verify(evaluator, extracted)

    # Return evaluator summary with verification tree and scores
    return evaluator.get_summary()