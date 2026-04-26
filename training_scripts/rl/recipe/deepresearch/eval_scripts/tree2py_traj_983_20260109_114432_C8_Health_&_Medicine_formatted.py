import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "usnews_hospitals_2025_2026_top3"
TASK_DESCRIPTION = (
    "Identify three hospitals in the United States that meet all of the following criteria based on the "
    "U.S. News & World Report 2025-2026 Best Hospitals rankings:\n\n"
    "First Hospital:\n"
    "• Ranked in the top 3 nationally for Cardiology, Heart & Vascular Surgery\n"
    "• Ranked in the top 3 nationally for Neurology & Neurosurgery\n"
    "• Listed on the U.S. News Best Hospitals Honor Roll 2025-2026\n"
    "• Located in New York City, New York\n\n"
    "Second Hospital:\n"
    "• Ranked in the top 3 nationally for Cardiology, Heart & Vascular Surgery\n"
    "• Ranked in the top 3 nationally for Geriatrics\n"
    "• Listed on the U.S. News Best Hospitals Honor Roll 2025-2026\n"
    "• Located in New York City, New York\n\n"
    "Third Hospital:\n"
    "• Ranked in the top 3 nationally for Gastroenterology & GI Surgery\n"
    "• Ranked in the top 3 nationally for Pulmonology & Lung Surgery\n"
    "• Listed on the U.S. News Best Hospitals Honor Roll 2025-2026\n"
    "• Located in Rochester, Minnesota\n\n"
    "For each hospital, provide:\n"
    "1. The full official hospital name\n"
    "2. The specific ranking position (#1, #2, or #3) for each of the two required specialties\n"
    "3. Confirmation of Honor Roll status\n"
    "4. The city and state location\n"
    "5. Reference URLs from the U.S. News & World Report 2025-2026 Best Hospitals rankings that verify each ranking position, Honor Roll status, and location"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HospitalInfo(BaseModel):
    name: Optional[str] = None
    name_urls: List[str] = Field(default_factory=list)

    cardiology_rank: Optional[str] = None
    cardiology_urls: List[str] = Field(default_factory=list)

    neurology_rank: Optional[str] = None
    neurology_urls: List[str] = Field(default_factory=list)

    geriatrics_rank: Optional[str] = None
    geriatrics_urls: List[str] = Field(default_factory=list)

    gastroenterology_rank: Optional[str] = None
    gastroenterology_urls: List[str] = Field(default_factory=list)

    pulmonology_rank: Optional[str] = None
    pulmonology_urls: List[str] = Field(default_factory=list)

    honor_roll_status: Optional[str] = None
    honor_roll_urls: List[str] = Field(default_factory=list)

    city: Optional[str] = None
    state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)


class HospitalsExtraction(BaseModel):
    hospital1: Optional[HospitalInfo] = None
    hospital2: Optional[HospitalInfo] = None
    hospital3: Optional[HospitalInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hospitals() -> str:
    return (
        "From the answer, extract structured information for exactly three hospitals that are claimed to meet the criteria "
        "based on U.S. News & World Report 2025-2026 Best Hospitals rankings. For each hospital (hospital1, hospital2, hospital3), "
        "return the following fields:\n\n"
        "Required fields per hospital:\n"
        "- name: Full official hospital name as stated in the answer (string or null).\n"
        "- name_urls: A list of URLs from U.S. News 2025-2026 pages that support the hospital name (array; empty if none).\n"
        "- cardiology_rank: The ranking position for Cardiology, Heart & Vascular Surgery (string like '#1', '#2', '#3', '1st', '2nd', '3rd'; null if not applicable or not mentioned).\n"
        "- cardiology_urls: U.S. News 2025-2026 URLs verifying the Cardiology ranking position (array; empty if none).\n"
        "- neurology_rank: The ranking position for Neurology & Neurosurgery (string as above; null if not applicable or not mentioned).\n"
        "- neurology_urls: U.S. News 2025-2026 URLs verifying the Neurology ranking position (array; empty if none).\n"
        "- geriatrics_rank: The ranking position for Geriatrics (string as above; null if not applicable or not mentioned).\n"
        "- geriatrics_urls: U.S. News 2025-2026 URLs verifying the Geriatrics ranking position (array; empty if none).\n"
        "- gastroenterology_rank: The ranking position for Gastroenterology & GI Surgery (string as above; null if not applicable or not mentioned).\n"
        "- gastroenterology_urls: U.S. News 2025-2026 URLs verifying the Gastroenterology ranking position (array; empty if none).\n"
        "- pulmonology_rank: The ranking position for Pulmonology & Lung Surgery (string as above; null if not applicable or not mentioned).\n"
        "- pulmonology_urls: U.S. News 2025-2026 URLs verifying the Pulmonology ranking position (array; empty if none).\n"
        "- honor_roll_status: 'Yes'/'No' or similar indicating Honor Roll status for 2025-2026 (string; null if not mentioned).\n"
        "- honor_roll_urls: U.S. News 2025-2026 URLs verifying Honor Roll status (array; empty if none).\n"
        "- city: City name for the hospital location (string; null if not mentioned).\n"
        "- state: State name for the hospital location (string; null if not mentioned).\n"
        "- location_urls: U.S. News 2025-2026 URLs verifying the hospital location (array; empty if none).\n\n"
        "Rules:\n"
        "1) Extract ONLY what is explicitly present in the answer. Do not invent any values.\n"
        "2) Extract URLs only if they appear as explicit URLs in the answer (including markdown links). If not present, leave the corresponding URL list empty.\n"
        "3) If the answer lists more than three hospitals, only extract the first three that appear to meet the criteria; if fewer, extract whatever is present for hospital1, hospital2, hospital3 (missing fields should be null or empty accordingly).\n"
        "4) Use the 2025-2026 U.S. News pages when possible; if the answer cites other years, still extract them as provided, but keep the URL lists exactly as in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_top3_rank(rank: Optional[str]) -> bool:
    if not rank:
        return False
    s = rank.strip().lower()
    # Accept forms like "#1", "1", "1st", "no. 1", "number 1"
    return bool(re.search(r"\b(#?\s*(?:1|2|3)(?:st|nd|rd)?|no\.?\s*(?:1|2|3)|number\s*(?:1|2|3))\b", s))


def is_affirmative(text: Optional[str]) -> bool:
    if not text:
        return False
    s = text.strip().lower()
    return s in {"yes", "true", "y", "honor roll", "on honor roll"} or "yes" in s or "honor roll" in s or "true" in s


def non_empty_str(text: Optional[str]) -> bool:
    return bool(text and text.strip())


def has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0 and all(isinstance(u, str) and u.strip() for u in urls))


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_name_group(
    evaluator: Evaluator,
    parent,
    hospital_idx_label: str,
    info: HospitalInfo,
) -> None:
    name_group = evaluator.add_sequential(
        id=f"{hospital_idx_label}_Name",
        desc="The full official hospital name is provided.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_str(info.name),
        id=f"{hospital_idx_label}_Name_Provided",
        desc="Hospital name string is present in the answer.",
        parent=name_group,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_urls(info.name_urls),
        id=f"{hospital_idx_label}_Name_URL_Present",
        desc="At least one U.S. News 2025-2026 reference URL is provided for the hospital name.",
        parent=name_group,
        critical=True
    )

    name_leaf = evaluator.add_leaf(
        id=f"{hospital_idx_label}_Name_URL",
        desc="A reference URL is provided that supports the hospital name (U.S. News 2025-2026 rankings page).",
        parent=name_group,
        critical=True
    )
    claim = f"The page shows the hospital named '{info.name}'."
    await evaluator.verify(
        claim=claim,
        node=name_leaf,
        sources=info.name_urls,
        additional_instruction=(
            "Verify the official hospital name on the U.S. News 2025-2026 page. "
            "Allow minor formatting variations (e.g., punctuation, hyphens, abbreviations)."
        )
    )


async def _verify_specialty_group(
    evaluator: Evaluator,
    parent,
    hospital_idx_label: str,
    info: HospitalInfo,
    specialty_label: str,          # e.g., "Cardiology"
    specialty_full_name: str,      # e.g., "Cardiology, Heart & Vascular Surgery"
    rank_field: str,               # e.g., "cardiology_rank"
    urls_field: str,               # e.g., "cardiology_urls"
) -> None:
    group_id = f"{hospital_idx_label}_{specialty_label}_Ranking"
    group_desc = f"The specific ranking position (#1, #2, or #3) for {specialty_full_name} (U.S. News 2025-2026) is provided."
    spec_group = evaluator.add_sequential(
        id=group_id,
        desc=group_desc,
        parent=parent,
        critical=True
    )

    rank_val: Optional[str] = getattr(info, rank_field, None)
    urls_val: List[str] = getattr(info, urls_field, [])

    evaluator.add_custom_node(
        result=is_top3_rank(rank_val),
        id=f"{hospital_idx_label}_{specialty_label}_Rank_Provided",
        desc=f"Specific ranking position (#1/#2/#3) is provided for {specialty_full_name}.",
        parent=spec_group,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_urls(urls_val),
        id=f"{hospital_idx_label}_{specialty_label}_URL_Present",
        desc=f"At least one U.S. News 2025-2026 reference URL is provided for {specialty_full_name} ranking verification.",
        parent=spec_group,
        critical=True
    )

    rank_leaf = evaluator.add_leaf(
        id=f"{hospital_idx_label}_{specialty_label}_URL",
        desc=f"A U.S. News 2025-2026 reference URL is provided verifying the {specialty_full_name} ranking position.",
        parent=spec_group,
        critical=True
    )
    claim = (
        f"{info.name} is ranked {rank_val} nationally for {specialty_full_name} in the U.S. News Best Hospitals 2025-2026."
    )
    await evaluator.verify(
        claim=claim,
        node=rank_leaf,
        sources=urls_val,
        additional_instruction=(
            "Check the U.S. News 2025-2026 page to confirm that the hospital is ranked within the top three (#1/#2/#3) "
            f"for {specialty_full_name}. Allow reasonable phrasing variants, but the ranking must explicitly indicate a top-3 position."
        )
    )


async def _verify_honor_roll_group(
    evaluator: Evaluator,
    parent,
    hospital_idx_label: str,
    info: HospitalInfo,
) -> None:
    hr_group = evaluator.add_sequential(
        id=f"{hospital_idx_label}_Honor_Roll",
        desc="Honor Roll 2025-2026 status is confirmed.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_affirmative(info.honor_roll_status),
        id=f"{hospital_idx_label}_Honor_Roll_Provided",
        desc="Honor Roll status ('Yes'/'true' or equivalent) is present in the answer for 2025-2026.",
        parent=hr_group,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_urls(info.honor_roll_urls),
        id=f"{hospital_idx_label}_Honor_Roll_URL_Present",
        desc="At least one U.S. News 2025-2026 reference URL is provided for Honor Roll verification.",
        parent=hr_group,
        critical=True
    )

    hr_leaf = evaluator.add_leaf(
        id=f"{hospital_idx_label}_Honor_Roll_URL",
        desc="A U.S. News 2025-2026 reference URL is provided verifying Honor Roll status.",
        parent=hr_group,
        critical=True
    )
    claim = f"{info.name} is listed on the U.S. News Best Hospitals Honor Roll 2025-2026."
    await evaluator.verify(
        claim=claim,
        node=hr_leaf,
        sources=info.honor_roll_urls,
        additional_instruction=(
            "Confirm that the hospital appears on the U.S. News Best Hospitals Honor Roll for the 2025-2026 cycle."
        )
    )


async def _verify_location_group(
    evaluator: Evaluator,
    parent,
    hospital_idx_label: str,
    info: HospitalInfo,
    expected_city: str,
    expected_state: str
) -> None:
    loc_group = evaluator.add_sequential(
        id=f"{hospital_idx_label}_Location",
        desc=f"City and state location is provided as {expected_city}, {expected_state}.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_str(info.city) and non_empty_str(info.state),
        id=f"{hospital_idx_label}_Location_Provided",
        desc="City and state location strings are present in the answer.",
        parent=loc_group,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_urls(info.location_urls),
        id=f"{hospital_idx_label}_Location_URL_Present",
        desc="At least one U.S. News 2025-2026 reference URL is provided for location verification.",
        parent=loc_group,
        critical=True
    )

    loc_leaf = evaluator.add_leaf(
        id=f"{hospital_idx_label}_Location_URL",
        desc="A U.S. News 2025-2026 reference URL is provided verifying the hospital location.",
        parent=loc_group,
        critical=True
    )
    claim = f"{info.name} is located in {expected_city}, {expected_state}."
    await evaluator.verify(
        claim=claim,
        node=loc_leaf,
        sources=info.location_urls,
        additional_instruction=(
            "Verify the city and state for the hospital from the U.S. News 2025-2026 page. "
            "Allow reasonable variants such as 'NY, NY', 'New York, NY', 'NYC' for New York City, and 'MN' for Minnesota."
        )
    )


async def verify_hospital(
    evaluator: Evaluator,
    parent,
    hospital_node_id: str,
    hospital_desc: str,
    info: HospitalInfo,
    expected_city: str,
    expected_state: str,
    specialties: List[Tuple[str, str, str]]  # list of (short_label, full_name, (rank_field, urls_field))
) -> None:
    # Hospital top-level node (parallel, non-critical)
    hosp_node = evaluator.add_parallel(
        id=hospital_node_id,
        desc=hospital_desc,
        parent=parent,
        critical=False
    )

    # Name verification group
    await _verify_name_group(evaluator, hosp_node, hospital_node_id, info)

    # Specialty ranking groups (two specialties per hospital)
    for short_label, full_name, fields in specialties:
        rank_field, urls_field = fields
        await _verify_specialty_group(
            evaluator=evaluator,
            parent=hosp_node,
            hospital_idx_label=hospital_node_id,
            info=info,
            specialty_label=short_label,
            specialty_full_name=full_name,
            rank_field=rank_field,
            urls_field=urls_field
        )

    # Honor Roll group
    await _verify_honor_roll_group(evaluator, hosp_node, hospital_node_id, info)

    # Location group
    await _verify_location_group(evaluator, hosp_node, hospital_node_id, info, expected_city, expected_state)


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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Hospitals evaluated independently
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_hospitals(),
        template_class=HospitalsExtraction,
        extraction_name="hospitals_extraction"
    )

    # Record constraints in summary for clarity
    evaluator.add_ground_truth({
        "requirements": {
            "year_cycle": "U.S. News Best Hospitals 2025-2026",
            "hospital_1": {
                "specialties": ["Cardiology, Heart & Vascular Surgery", "Neurology & Neurosurgery"],
                "location": "New York City, New York",
                "honor_roll": True
            },
            "hospital_2": {
                "specialties": ["Cardiology, Heart & Vascular Surgery", "Geriatrics"],
                "location": "New York City, New York",
                "honor_roll": True
            },
            "hospital_3": {
                "specialties": ["Gastroenterology & GI Surgery", "Pulmonology & Lung Surgery"],
                "location": "Rochester, Minnesota",
                "honor_roll": True
            }
        }
    }, gt_type="constraints")

    # Build the root-level task node (parallel, non-critical) per rubric
    top_node = evaluator.add_parallel(
        id="Identify_Three_Top_Ranked_US_Hospitals",
        desc="Find three hospitals in the United States that meet the specified specialty top-3 ranking, Honor Roll, and location requirements using U.S. News & World Report 2025-2026 Best Hospitals rankings, and provide verifying URLs.",
        parent=root,
        critical=False
    )

    # Prepare hospital info objects with safe fallbacks
    h1 = extracted.hospital1 or HospitalInfo()
    h2 = extracted.hospital2 or HospitalInfo()
    h3 = extracted.hospital3 or HospitalInfo()

    # Hospital #1: NYC, NY, Cardiology + Neurology
    await verify_hospital(
        evaluator=evaluator,
        parent=top_node,
        hospital_node_id="Hospital_1",
        hospital_desc="Hospital #1: Top 3 in Cardiology, Heart & Vascular Surgery AND Top 3 in Neurology & Neurosurgery; on Honor Roll 2025-2026; located in New York City, NY.",
        info=h1,
        expected_city="New York City",
        expected_state="New York",
        specialties=[
            ("Cardiology", "Cardiology, Heart & Vascular Surgery", ("cardiology_rank", "cardiology_urls")),
            ("Neurology", "Neurology & Neurosurgery", ("neurology_rank", "neurology_urls")),
        ]
    )

    # Hospital #2: NYC, NY, Cardiology + Geriatrics
    await verify_hospital(
        evaluator=evaluator,
        parent=top_node,
        hospital_node_id="Hospital_2",
        hospital_desc="Hospital #2: Top 3 in Cardiology, Heart & Vascular Surgery AND Top 3 in Geriatrics; on Honor Roll 2025-2026; located in New York City, NY.",
        info=h2,
        expected_city="New York City",
        expected_state="New York",
        specialties=[
            ("Cardiology", "Cardiology, Heart & Vascular Surgery", ("cardiology_rank", "cardiology_urls")),
            ("Geriatrics", "Geriatrics", ("geriatrics_rank", "geriatrics_urls")),
        ]
    )

    # Hospital #3: Rochester, MN, Gastroenterology + Pulmonology
    await verify_hospital(
        evaluator=evaluator,
        parent=top_node,
        hospital_node_id="Hospital_3",
        hospital_desc="Hospital #3: Top 3 in Gastroenterology & GI Surgery AND Top 3 in Pulmonology & Lung Surgery; on Honor Roll 2025-2026; located in Rochester, MN.",
        info=h3,
        expected_city="Rochester",
        expected_state="Minnesota",
        specialties=[
            ("Gastroenterology", "Gastroenterology & GI Surgery", ("gastroenterology_rank", "gastroenterology_urls")),
            ("Pulmonology", "Pulmonology & Lung Surgery", ("pulmonology_rank", "pulmonology_urls")),
        ]
    )

    return evaluator.get_summary()