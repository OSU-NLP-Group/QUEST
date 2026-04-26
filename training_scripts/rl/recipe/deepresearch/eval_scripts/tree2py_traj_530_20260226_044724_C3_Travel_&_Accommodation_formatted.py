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
TASK_ID = "tsa_precheck_orlando_mlk_2026"
TASK_DESCRIPTION = (
    "A traveler is planning a trip to the Orlando, Florida area over the Martin Luther King Jr. Day 2026 holiday "
    "weekend and wants to enroll in TSA PreCheck upon arrival at the destination airport. Provide the following information: "
    "(1) The exact date of Martin Luther King Jr. Day in 2026, including the day of the week; "
    "(2) The TSA PreCheck enrollment location at Orlando International Airport, including the enrollment provider name, "
    "complete street address, and the specific terminal location(s) where enrollment is available; "
    "(3) The identification requirements for domestic air travel that will be in effect during January 2026, specifically "
    "whether REAL ID-compliant licenses are required."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MLKDateInfo(BaseModel):
    date_text: Optional[str] = None  # e.g., "January 19, 2026"
    weekday: Optional[str] = None    # e.g., "Monday"
    sources: List[str] = Field(default_factory=list)


class TSALocationInfo(BaseModel):
    provider_name: Optional[str] = None  # e.g., "IdentoGO (IDEMIA)" or similar
    street_address: Optional[str] = None # complete address string
    terminal_locations: List[str] = Field(default_factory=list)  # e.g., ["Terminal A", "Terminal B", "Main Terminal Landside"]
    sources: List[str] = Field(default_factory=list)


class IDRequirementsInfo(BaseModel):
    requirement_text: Optional[str] = None  # free text statement the answer made
    real_id_required_text: Optional[str] = None  # e.g., "required", "not required", "yes", "no"
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_mlk_date() -> str:
    return (
        "From the answer, extract the information about Martin Luther King Jr. Day in 2026.\n"
        "Return a JSON with fields:\n"
        "  - date_text: the exact date string mentioned for MLK Day 2026 (e.g., 'January 19, 2026').\n"
        "  - weekday: the day of the week mentioned for MLK Day 2026 (e.g., 'Monday').\n"
        "  - sources: an array of any URLs cited in the answer that support the date or weekday; if none are present, return an empty array.\n"
        "If any field isn't explicitly present in the answer, set it to null (or empty array for sources)."
    )


def prompt_extract_tsa_location() -> str:
    return (
        "From the answer, extract the TSA PreCheck enrollment location information for Orlando International Airport (MCO).\n"
        "Return a JSON with fields:\n"
        "  - provider_name: the enrollment provider name explicitly stated (e.g., 'IdentoGO', 'IDEMIA').\n"
        "  - street_address: the complete street address for the enrollment location.\n"
        "  - terminal_locations: an array listing the terminal(s) or checkpoint area(s) where enrollment is available (e.g., 'Terminal A', 'Terminal B', 'Main Terminal', 'Landside').\n"
        "  - sources: an array of URLs cited in the answer that support the provider name, address, and terminal locations.\n"
        "If any field isn't explicitly present in the answer, set it to null (or empty array for arrays)."
    )


def prompt_extract_id_requirements() -> str:
    return (
        "From the answer, extract the identification requirements for domestic air travel in effect during January 2026, "
        "specifically whether REAL ID-compliant licenses are required.\n"
        "Return a JSON with fields:\n"
        "  - requirement_text: the statement in the answer about ID requirements as of January 2026.\n"
        "  - real_id_required_text: normalize the answer's statement into a concise value among: 'required', 'not required', or 'unknown'. "
        "    Use 'required' if the answer asserts REAL ID-compliant licenses (or other TSA-accepted ID) are required as of January 2026; "
        "    use 'not required' if the answer asserts they are not required; use 'unknown' otherwise.\n"
        "  - sources: an array of URLs cited in the answer that support this identification requirement.\n"
        "If any field isn't explicitly present in the answer, set it to null (or empty array for sources)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_non_empty_string(s: Optional[str]) -> bool:
    return bool(s) and bool(str(s).strip())


def _has_non_empty_list(lst: Optional[List[str]]) -> bool:
    return bool(lst) and len(lst) > 0


def _normalize_bool_from_text(text: Optional[str]) -> Optional[bool]:
    """
    Convert 'required'/'not required'/'yes'/'no' textual forms to boolean.
    Returns True if required, False if not required, None if unknown.
    """
    if not _has_non_empty_string(text):
        return None
    t = text.lower().strip()
    if t in {"required", "yes", "true"}:
        return True
    if t in {"not required", "no", "false"}:
        return False
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_mlk(
    evaluator: Evaluator,
    parent_node,
    mlk: MLKDateInfo,
) -> None:
    """
    Build and verify the MLK Day 2026 information subtree.
    """
    mlk_node = evaluator.add_sequential(
        id="MLK_Day_2026_Date_Info",
        desc="Provide the exact date of Martin Luther King Jr. Day in 2026, including the day of the week",
        parent=parent_node,
        critical=True,
    )

    # Existence check: both date_text and weekday should be provided in the answer
    exist_ok = _has_non_empty_string(mlk.date_text) and _has_non_empty_string(mlk.weekday)
    evaluator.add_custom_node(
        result=exist_ok,
        id="MLK_Day_2026_Date_Provided",
        desc="MLK Day 2026 date and weekday are provided in the answer",
        parent=mlk_node,
        critical=True,
    )

    # Verification 1: Date corresponds to MLK Day (third Monday of January 2026)
    date_claim_leaf = evaluator.add_leaf(
        id="MLK_Day_2026_Date_Correct",
        desc="The provided date corresponds to MLK Day 2026 (third Monday in January)",
        parent=mlk_node,
        critical=True,
    )
    date_claim = (
        f"The date '{mlk.date_text}' is the observance of Martin Luther King Jr. Day in 2026 "
        f"(the third Monday in January)."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_claim_leaf,
        additional_instruction=(
            "Use calendar knowledge: MLK Day is observed on the third Monday in January each year. "
            "Confirm the provided date is the third Monday of January 2026."
        ),
    )

    # Verification 2: The weekday for the provided date matches the answer and is Monday
    weekday_leaf = evaluator.add_leaf(
        id="MLK_Day_2026_Weekday_Correct",
        desc="The provided weekday matches the actual weekday for the provided date and is Monday",
        parent=mlk_node,
        critical=True,
    )
    weekday_claim = (
        f"The date '{mlk.date_text}' falls on a {mlk.weekday}, and MLK Day is always observed on a Monday."
    )
    await evaluator.verify(
        claim=weekday_claim,
        node=weekday_leaf,
        additional_instruction=(
            "Confirm the weekday of the provided date and ensure it aligns with Monday."
        ),
    )


async def build_and_verify_tsa_location(
    evaluator: Evaluator,
    parent_node,
    tsa: TSALocationInfo,
) -> None:
    """
    Build and verify the TSA PreCheck enrollment location information subtree for MCO.
    """
    tsa_node = evaluator.add_parallel(
        id="TSA_PreCheck_Location_Info",
        desc="Identify the TSA PreCheck enrollment location at Orlando International Airport with complete details",
        parent=parent_node,
        critical=True,
    )

    # Sub-node: Provider and Address (sequential)
    provider_node = evaluator.add_sequential(
        id="Provider_and_Address",
        desc="Provide the enrollment provider name and complete street address",
        parent=tsa_node,
        critical=True,
    )

    # Existence check: provider, address, and at least one source
    provider_exist_ok = _has_non_empty_string(tsa.provider_name) and _has_non_empty_string(tsa.street_address) and _has_non_empty_list(tsa.sources)
    evaluator.add_custom_node(
        result=provider_exist_ok,
        id="Provider_Address_Provided",
        desc="Enrollment provider and complete street address are provided, with cited sources",
        parent=provider_node,
        critical=True,
    )

    # Verify provider name via sources
    provider_leaf = evaluator.add_leaf(
        id="Provider_Name_Supported",
        desc="Enrollment provider name is supported by cited sources",
        parent=provider_node,
        critical=True,
    )
    provider_claim = (
        f"The TSA PreCheck enrollment provider at Orlando International Airport (MCO) is '{tsa.provider_name}'."
    )
    await evaluator.verify(
        claim=provider_claim,
        node=provider_leaf,
        sources=tsa.sources,
        additional_instruction=(
            "Verify that the cited source(s) explicitly identify the TSA PreCheck enrollment provider for MCO. "
            "Allow minor naming variations (e.g., 'IdentoGO', 'IDEMIA')."
        ),
    )

    # Verify street address via sources
    address_leaf = evaluator.add_leaf(
        id="Street_Address_Supported",
        desc="Enrollment street address is supported by cited sources",
        parent=provider_node,
        critical=True,
    )
    address_claim = (
        f"The TSA PreCheck enrollment location address at MCO is '{tsa.street_address}'."
    )
    await evaluator.verify(
        claim=address_claim,
        node=address_leaf,
        sources=tsa.sources,
        additional_instruction=(
            "Verify that the cited source(s) clearly list the address for the TSA PreCheck enrollment center at MCO. "
            "Allow minor formatting differences in address representation."
        ),
    )

    # Sub-node: Terminal Locations (sequential)
    terminals_node = evaluator.add_sequential(
        id="Terminal_Locations",
        desc="Specify the terminal(s) or checkpoint area(s) where enrollment is available",
        parent=tsa_node,
        critical=True,
    )

    # Existence check: terminal locations list and sources present
    terminals_exist_ok = _has_non_empty_list(tsa.terminal_locations) and _has_non_empty_list(tsa.sources)
    evaluator.add_custom_node(
        result=terminals_exist_ok,
        id="Terminal_Locations_Provided",
        desc="Specific terminal or checkpoint locations are provided, with cited sources",
        parent=terminals_node,
        critical=True,
    )

    # Verify terminal locations via sources
    terminals_leaf = evaluator.add_leaf(
        id="Terminal_Locations_Supported",
        desc="Terminal/area locations for enrollment are supported by cited sources",
        parent=terminals_node,
        critical=True,
    )
    terminals_list_str = ", ".join(tsa.terminal_locations) if tsa.terminal_locations else ""
    terminals_claim = (
        f"TSA PreCheck enrollment is available at the following location(s) within MCO: {terminals_list_str}."
    )
    await evaluator.verify(
        claim=terminals_claim,
        node=terminals_leaf,
        sources=tsa.sources,
        additional_instruction=(
            "Confirm that the source(s) identify the specific terminal(s) or area(s) (e.g., Terminal A/B, Main Terminal, Landside) "
            "where enrollment is available. Allow minor naming variations such as 'Side A/Side B' vs 'Terminal A/Terminal B'."
        ),
    )


async def build_and_verify_id_requirements(
    evaluator: Evaluator,
    parent_node,
    ids: IDRequirementsInfo,
) -> None:
    """
    Build and verify the ID requirements subtree, focusing on REAL ID status for January 2026.
    """
    id_node = evaluator.add_sequential(
        id="ID_Requirements_Info",
        desc="State the identification requirements for domestic air travel in effect during January 2026, specifically addressing whether REAL ID-compliant licenses are required",
        parent=parent_node,
        critical=True,
    )

    # Existence check: requirement text and sources must be present
    id_exist_ok = _has_non_empty_string(ids.requirement_text) and _has_non_empty_list(ids.sources)
    evaluator.add_custom_node(
        result=id_exist_ok,
        id="ID_Requirements_Provided",
        desc="ID requirements statement and sources are provided in the answer",
        parent=id_node,
        critical=True,
    )

    # Verification leaf: REAL ID requirement status via sources
    real_id_leaf = evaluator.add_leaf(
        id="REAL_ID_Status_Supported",
        desc="REAL ID requirement status as of January 2026 is supported by cited sources",
        parent=id_node,
        critical=True,
    )
    normalized = _normalize_bool_from_text(ids.real_id_required_text)
    if normalized is True:
        real_id_claim = (
            "As of January 2026, REAL ID-compliant driver's licenses (or other TSA-accepted identification such as a passport) "
            "are required to board domestic flights in the United States."
        )
    elif normalized is False:
        real_id_claim = (
            "As of January 2026, REAL ID-compliant driver's licenses are not required to board domestic flights in the United States."
        )
    else:
        real_id_claim = (
            "The answer indicates uncertainty about whether REAL ID-compliant licenses are required as of January 2026."
        )
    await evaluator.verify(
        claim=real_id_claim,
        node=real_id_leaf,
        sources=ids.sources,
        additional_instruction=(
            "Verify the requirement status specifically for January 2026 using the cited sources (e.g., DHS/TSA official pages). "
            "Focus solely on whether REAL ID-compliant licenses are required by that time."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the answer for TSA PreCheck enrollment at Orlando during MLK Day 2026 weekend.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level aggregation
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

    # Create critical task root under the framework root to respect critical consistency
    task_root = evaluator.add_parallel(
        id="TSA_PreCheck_Orlando_Information",
        desc="Provide accurate and complete information for TSA PreCheck enrollment at Orlando International Airport during the Martin Luther King Jr. Day 2026 weekend",
        parent=root,
        critical=True,
    )

    # Extract information in parallel
    mlk_task = evaluator.extract(
        prompt=prompt_extract_mlk_date(),
        template_class=MLKDateInfo,
        extraction_name="mlk_day_2026",
    )
    tsa_task = evaluator.extract(
        prompt=prompt_extract_tsa_location(),
        template_class=TSALocationInfo,
        extraction_name="tsa_location_mco",
    )
    id_task = evaluator.extract(
        prompt=prompt_extract_id_requirements(),
        template_class=IDRequirementsInfo,
        extraction_name="id_requirements_jan_2026",
    )
    mlk_info, tsa_info, id_info = await asyncio.gather(mlk_task, tsa_task, id_task)

    # Build and verify each subtree (all critical under task_root)
    await build_and_verify_mlk(evaluator, task_root, mlk_info)
    await build_and_verify_tsa_location(evaluator, task_root, tsa_info)
    await build_and_verify_id_requirements(evaluator, task_root, id_info)

    # Summary
    return evaluator.get_summary()