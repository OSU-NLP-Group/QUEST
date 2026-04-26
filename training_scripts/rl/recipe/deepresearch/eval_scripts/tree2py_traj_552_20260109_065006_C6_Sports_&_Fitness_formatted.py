import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "olympic_stadium_constraints"
TASK_DESCRIPTION = (
    "Identify an Olympic athletics stadium that meets ALL of the following requirements:\n\n"
    "1. The stadium must have hosted Summer Olympic Games athletics events\n"
    "2. The stadium must have hosted World Athletics Championships at least once after 1980\n"
    "3. The stadium's current permanent capacity must be between 35,000 and 80,000 seats\n"
    "4. The stadium must have undergone a major renovation or reconstruction completed in the 21st century (between 2000 and 2025)\n"
    "5. The stadium must have a World Athletics certified athletic track\n\n"
    "Provide the following information:\n"
    "- Stadium name and location (city and country)\n"
    "- Year the stadium hosted Summer Olympics\n"
    "- Year(s) the stadium hosted World Athletics Championships after 1980\n"
    "- Current permanent seating capacity\n"
    "- Years when the major 21st-century renovation took place\n"
    "- Confirmation that the stadium has World Athletics certified track"
)


class StadiumExtraction(BaseModel):
    stadium_name: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None

    olympic_year: Optional[str] = None  # Keep as string to be flexible
    worlds_years: List[str] = Field(default_factory=list)

    capacity: Optional[str] = None  # Keep as string; range or formatted values allowed
    renovation_years: List[str] = Field(default_factory=list)
    track_certified_statement: Optional[str] = None  # Any confirmation text present in the answer

    sources_general: List[str] = Field(default_factory=list)

    sources_olympics: List[str] = Field(default_factory=list)
    sources_worlds: List[str] = Field(default_factory=list)
    sources_capacity: List[str] = Field(default_factory=list)
    sources_renovation: List[str] = Field(default_factory=list)
    sources_track: List[str] = Field(default_factory=list)


def prompt_extract_stadium() -> str:
    return (
        "Extract exactly the stadium information presented in the answer. Do not invent or infer any details. "
        "If multiple stadiums are mentioned, treat the first one as the selected stadium and still extract all fields "
        "based on that first stadium.\n\n"
        "Return the following fields:\n"
        "1. stadium_name: The stadium's name.\n"
        "2. city: The city where the stadium is located.\n"
        "3. country: The country where the stadium is located.\n"
        "4. olympic_year: The year the stadium hosted Summer Olympic Games athletics events (as stated in the answer). If not provided, return null.\n"
        "5. worlds_years: An array of year strings for the World Athletics Championships hosted at the stadium after 1980 (as stated). If none given, return an empty array.\n"
        "6. capacity: The current permanent seating capacity as stated in the answer (string; keep formatting). If not provided, return null.\n"
        "7. renovation_years: An array of year strings for major renovation/reconstruction completed in the 21st century (2000–2025) as stated in the answer. If not provided, return an empty array.\n"
        "8. track_certified_statement: The confirmation or statement that the track is World Athletics certified (e.g., 'World Athletics Class 1 certified'), if mentioned. If not present, return null.\n"
        "9. sources_general: All general source URLs cited for this stadium info in the answer.\n"
        "10. sources_olympics: Source URLs specifically supporting the Summer Olympics athletics hosting claim (if any).\n"
        "11. sources_worlds: Source URLs specifically supporting the World Athletics Championships hosting claim (if any).\n"
        "12. sources_capacity: Source URLs specifically supporting the capacity claim (if any).\n"
        "13. sources_renovation: Source URLs specifically supporting the renovation/reconstruction claim (if any).\n"
        "14. sources_track: Source URLs specifically supporting the World Athletics certified track claim (if any).\n\n"
        "Special rules for URL sources extraction:\n"
        "- Extract only URLs explicitly present in the answer (plain URL, markdown link destination, etc.).\n"
        "- If no specific URLs for a field are provided, leave the corresponding array empty.\n"
        "- Include full URLs, with http:// or https://.\n"
    )


def extract_integers(text: Optional[str]) -> List[int]:
    if not text:
        return []
    nums = re.findall(r"\d{1,6}", text)
    out = []
    for n in nums:
        try:
            out.append(int(n))
        except Exception:
            pass
    return out


def extract_years_from_list_str(years: List[str]) -> List[int]:
    out: List[int] = []
    for y in years:
        out.extend([n for n in extract_integers(y) if 1800 <= n <= 2100])
    return out


def any_year_in_range(years: List[int], start: int, end: int) -> bool:
    return any(start <= y <= end for y in years)


def pick_sources(primary: List[str], fallback: List[str]) -> Optional[List[str]]:
    if primary and len(primary) > 0:
        return primary
    if fallback and len(fallback) > 0:
        return fallback
    return None


async def build_verification_tree(evaluator: Evaluator, extracted: StadiumExtraction) -> None:
    top = evaluator.add_parallel(
        id="Root",
        desc="Evaluate whether exactly one Olympic athletics stadium is identified that satisfies all stated constraints and includes all requested information",
        parent=evaluator.root,
        critical=True
    )

    basic = evaluator.add_parallel(
        id="Stadium_Basic_Identification",
        desc="Stadium name and location are provided",
        parent=top,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.stadium_name and extracted.stadium_name.strip()),
        id="Stadium_Name_Provided",
        desc="Stadium name is provided",
        parent=basic,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.city and extracted.city.strip()),
        id="City_Provided",
        desc="City is provided",
        parent=basic,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.country and extracted.country.strip()),
        id="Country_Provided",
        desc="Country is provided",
        parent=basic,
        critical=True
    )

    hosting = evaluator.add_parallel(
        id="Event_Hosting_Requirements",
        desc="Stadium hosting-history constraints are satisfied and required years are provided",
        parent=top,
        critical=True
    )

    summer_olympics = evaluator.add_parallel(
        id="Summer_Olympics_Athletics",
        desc="Stadium hosted Summer Olympic Games athletics events and the Olympic hosting year is provided",
        parent=hosting,
        critical=True
    )

    olympics_leaf = evaluator.add_leaf(
        id="Hosted_Summer_Olympics_Athletics",
        desc="Stadium hosted Summer Olympic Games athletics events",
        parent=summer_olympics,
        critical=True
    )
    olympics_claim = (
        f"The stadium '{extracted.stadium_name}' hosted athletics events at the Summer Olympic Games"
        + (f" in {extracted.olympic_year}." if extracted.olympic_year else ".")
    )
    olympics_sources = pick_sources(extracted.sources_olympics, extracted.sources_general)
    await evaluator.verify(
        claim=olympics_claim,
        node=olympics_leaf,
        sources=olympics_sources,
        additional_instruction=(
            "Verify that this venue was the athletics (track and field) stadium for the indicated Summer Olympics year, "
            "not merely a football stadium or a different sport. Use provided sources."
        )
    )

    evaluator.add_custom_node(
        result=bool(extracted.olympic_year and extracted.olympic_year.strip()),
        id="Olympic_Year_Provided",
        desc="Year the stadium hosted the Summer Olympics is provided",
        parent=summer_olympics,
        critical=True
    )

    worlds = evaluator.add_parallel(
        id="World_Athletics_Championships_Post_1980",
        desc="Stadium hosted World Athletics Championships at least once after 1980 and the year(s) are provided",
        parent=hosting,
        critical=True
    )

    worlds_leaf = evaluator.add_leaf(
        id="Hosted_World_Athletics_Championships_After_1980",
        desc="Stadium hosted the World Athletics Championships at least once after 1980",
        parent=worlds,
        critical=True
    )
    worlds_years_int = extract_years_from_list_str(extracted.worlds_years)
    worlds_claim = (
        f"The stadium '{extracted.stadium_name}' hosted the outdoor World Athletics Championships "
        f"at least once after 1980."
    )
    worlds_sources = pick_sources(extracted.sources_worlds, extracted.sources_general)
    await evaluator.verify(
        claim=worlds_claim,
        node=worlds_leaf,
        sources=worlds_sources,
        additional_instruction=(
            "Confirm that this is the outdoor World Championships in Athletics (IAAF/World Athletics), not the World Indoor Championships. "
            "At least one hosting year must be after 1980."
        )
    )

    years_provided_ok = len(worlds_years_int) > 0 and any(y > 1980 for y in worlds_years_int)
    evaluator.add_custom_node(
        result=years_provided_ok,
        id="World_Championships_Years_Provided",
        desc="Year(s) the stadium hosted World Athletics Championships after 1980 are provided",
        parent=worlds,
        critical=True
    )

    attrs = evaluator.add_parallel(
        id="Stadium_Attribute_Requirements",
        desc="Stadium capacity, renovation timeframe, and track certification constraints are satisfied and required details are provided",
        parent=top,
        critical=True
    )

    capacity_node = evaluator.add_parallel(
        id="Capacity",
        desc="Current permanent capacity is provided and within the required range",
        parent=attrs,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.capacity and extracted.capacity.strip()),
        id="Capacity_Value_Provided",
        desc="Current permanent seating capacity is provided",
        parent=capacity_node,
        critical=True
    )

    capacity_ints = extract_integers(extracted.capacity)
    within_range = any(35000 <= n <= 80000 for n in capacity_ints)
    evaluator.add_custom_node(
        result=within_range,
        id="Capacity_Within_35000_80000",
        desc="Current permanent seating capacity is between 35,000 and 80,000 seats",
        parent=capacity_node,
        critical=True
    )

    renovation = evaluator.add_parallel(
        id="Renovation_21st_Century",
        desc="Major renovation/reconstruction completed between 2000 and 2025, and renovation years are provided",
        parent=attrs,
        critical=True
    )

    renovation_leaf = evaluator.add_leaf(
        id="Renovation_Completed_2000_2025",
        desc="Stadium underwent a major renovation or reconstruction completed between 2000 and 2025",
        parent=renovation,
        critical=True
    )
    ren_years_int = extract_years_from_list_str(extracted.renovation_years)
    renovation_claim = (
        f"The stadium '{extracted.stadium_name}' underwent a major renovation or reconstruction with completion year(s) in the range 2000–2025."
    )
    renovation_sources = pick_sources(extracted.sources_renovation, extracted.sources_general)
    await evaluator.verify(
        claim=renovation_claim,
        node=renovation_leaf,
        sources=renovation_sources,
        additional_instruction=(
            "The evidence should explicitly indicate a major renovation or reconstruction and that it was completed in 2000–2025. "
            "If multiple phases exist, at least one completion year within 2000–2025 is acceptable."
        )
    )

    evaluator.add_custom_node(
        result=bool(extracted.renovation_years and len(extracted.renovation_years) > 0),
        id="Renovation_Years_Provided",
        desc="Years when the major 21st-century renovation took place are provided",
        parent=renovation,
        critical=True
    )

    track = evaluator.add_parallel(
        id="World_Athletics_Certified_Track",
        desc="Stadium has a World Athletics certified athletic track and this is stated",
        parent=attrs,
        critical=True
    )

    track_leaf = evaluator.add_leaf(
        id="Track_Is_WA_Certified",
        desc="Stadium has a World Athletics certified athletic track",
        parent=track,
        critical=True
    )
    track_claim = (
        f"The stadium '{extracted.stadium_name}' has a World Athletics certified athletics track "
        f"(e.g., Class 1 or Class 2 certification)."
    )
    track_sources = pick_sources(extracted.sources_track, extracted.sources_general)
    await evaluator.verify(
        claim=track_claim,
        node=track_leaf,
        sources=track_sources,
        additional_instruction=(
            "Look for explicit mention of World Athletics (formerly IAAF) certification, including Class 1 or Class 2 track certifications. "
            "Facility directories or official certification listings are acceptable."
        )
    )

    evaluator.add_custom_node(
        result=bool(extracted.track_certified_statement and extracted.track_certified_statement.strip()),
        id="Track_Certification_Confirmation_Provided",
        desc="Confirmation that the track is World Athletics certified is provided",
        parent=track,
        critical=True
    )


async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
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

    extracted: StadiumExtraction = await evaluator.extract(
        prompt=prompt_extract_stadium(),
        template_class=StadiumExtraction,
        extraction_name="stadium_extraction"
    )

    evaluator.add_custom_info(
        info={
            "parsed_capacity_numbers": extract_integers(extracted.capacity),
            "parsed_worlds_years": extract_years_from_list_str(extracted.worlds_years),
            "parsed_renovation_years": extract_years_from_list_str(extracted.renovation_years)
        },
        info_type="debug",
        info_name="parsed_numbers_summary"
    )

    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()