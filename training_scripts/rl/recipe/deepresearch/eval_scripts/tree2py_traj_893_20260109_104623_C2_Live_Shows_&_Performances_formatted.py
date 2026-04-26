import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "broadway_max_capacity"
TASK_DESCRIPTION = (
    "A major theatrical production company is planning to stage a new large-scale Broadway musical in New York City and "
    "needs to identify the theater with the maximum seating capacity to maximize ticket sales potential. Identify which "
    "Broadway theater has the highest seating capacity, and provide the following information: (1) The name of the theater, "
    "(2) Its exact seating capacity, (3) Its complete street address in Manhattan, (4) Confirmation that it meets the official "
    "Broadway theater criteria (location within the Theater District between 41st-54th Streets and 6th-8th Avenues, and minimum "
    "500-seat capacity), (5) Reference URLs to support your findings. Your answer should help the production company verify they "
    "are considering the genuinely largest Broadway venue for their production."
)


class BroadwayTheaterCandidate(BaseModel):
    theater_name: Optional[str] = None
    seating_capacity: Optional[str] = None
    street_address: Optional[str] = None
    largest_claim_text: Optional[str] = None
    official_broadway_designation_statement: Optional[str] = None
    theatre_district_location_statement: Optional[str] = None
    minimum_500_seats_statement: Optional[str] = None
    capacity_source_urls: List[str] = Field(default_factory=list)
    largest_claim_source_urls: List[str] = Field(default_factory=list)
    address_source_urls: List[str] = Field(default_factory=list)
    criteria_source_urls: List[str] = Field(default_factory=list)


def prompt_extract_theater_candidate() -> str:
    return """
    Extract the single Broadway theater the answer identifies as having the highest seating capacity, along with required details and supporting URLs.

    Return a JSON object with fields:
    - theater_name: the name of the identified theater
    - seating_capacity: the exact seating capacity as written in the answer (keep formatting like commas or hyphens if present; do not infer)
    - street_address: the complete street address in Manhattan as written in the answer (include street, number, city, state, ZIP if provided)
    - largest_claim_text: the statement or sentence in the answer asserting this theater has the highest seating capacity among Broadway theaters (or null if not explicitly stated)
    - official_broadway_designation_statement: any statement in the answer confirming it is an officially designated Broadway theater (or null)
    - theatre_district_location_statement: any statement in the answer confirming it is within the Theater District bounds (41st–54th Streets; 6th–8th Avenues) (or null)
    - minimum_500_seats_statement: any statement confirming it meets the minimum 500-seat requirement (or null)
    - capacity_source_urls: list of URL(s) in the answer that support the seating capacity value
    - largest_claim_source_urls: list of URL(s) in the answer that support the claim that it has the highest capacity among Broadway theaters
    - address_source_urls: list of URL(s) in the answer that support the complete street address in Manhattan
    - criteria_source_urls: list of URL(s) in the answer that support Broadway designation and/or the district and 500-seat criteria confirmation

    IMPORTANT:
    - Extract only URLs explicitly present in the answer, including plain URLs or markdown links. Do not invent URLs.
    - If a field is missing in the answer, set it to null (or empty array for URL lists).
    """


def _strip_and_nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if isinstance(u, str):
            u2 = u.strip()
            if u2:
                # Basic normalization: ensure protocol if obviously missing
                if not re.match(r"^https?://", u2):
                    u2 = "http://" + u2
                cleaned.append(u2)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def _parse_capacity_to_int(capacity_text: Optional[str]) -> Optional[int]:
    if not _strip_and_nonempty(capacity_text):
        return None
    # Extract the first integer-like token from the string
    # Handles "1,933", "1933", "approx. 1,900", "1,900–1,933" -> choose the largest found if range is present
    numbers = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", capacity_text)]
    if not numbers:
        return None
    # If there's a range, use the max as the capacity for >=500 check
    return max(numbers)


async def _build_verification_tree(evaluator: Evaluator, candidate: BroadwayTheaterCandidate, root) -> None:
    # Child group 1: theater_identification (critical, parallel)
    identification_node = evaluator.add_parallel(
        id="theater_identification",
        desc="Identify the Broadway theater that has the highest seating capacity among Broadway theaters",
        parent=root,
        critical=True,
    )

    # theater_name_provided (existence check)
    evaluator.add_custom_node(
        result=_strip_and_nonempty(candidate.theater_name),
        id="theater_name_provided",
        desc="Provide the name of the identified theater",
        parent=identification_node,
        critical=True,
    )

    # highest_capacity_among_broadway (claim verification)
    highest_capacity_leaf = evaluator.add_leaf(
        id="highest_capacity_among_broadway",
        desc="Claim/indicate that the identified theater has the highest seating capacity among Broadway theaters",
        parent=identification_node,
        critical=True,
    )
    name_for_claim = candidate.theater_name or ""
    claim_highest = (
        f"The theater '{name_for_claim}' has the highest seating capacity among Broadway theaters in New York City."
    )
    largest_urls = _normalize_urls(candidate.largest_claim_source_urls)
    await evaluator.verify(
        claim=claim_highest,
        node=highest_capacity_leaf,
        sources=largest_urls if largest_urls else None,
        additional_instruction=(
            "Verify strictly based on the provided sources if available. Confirm that the sources explicitly state "
            f"that '{name_for_claim}' is the largest Broadway theater by seating capacity. If no URLs are provided, "
            "still evaluate the claim against the answer text, but do not rely on external knowledge."
        ),
    )

    # Child group 2: required_details_and_criteria (critical, parallel)
    details_node = evaluator.add_parallel(
        id="required_details_and_criteria",
        desc="Provide required factual details and confirm the official Broadway theater criteria specified in the prompt",
        parent=root,
        critical=True,
    )

    # exact_seating_capacity_provided (existence check)
    evaluator.add_custom_node(
        result=_strip_and_nonempty(candidate.seating_capacity),
        id="exact_seating_capacity_provided",
        desc="Provide the theater's exact seating capacity as a specific numeric value",
        parent=details_node,
        critical=True,
    )

    # complete_street_address_provided (existence check)
    evaluator.add_custom_node(
        result=_strip_and_nonempty(candidate.street_address),
        id="complete_street_address_provided",
        desc="Provide the theater's complete street address in Manhattan",
        parent=details_node,
        critical=True,
    )

    # official_broadway_designation_confirmed (verification by URLs)
    official_bway_leaf = evaluator.add_leaf(
        id="official_broadway_designation_confirmed",
        desc="Confirm the venue is an officially designated Broadway theater",
        parent=details_node,
        critical=True,
    )
    criteria_urls = _normalize_urls(candidate.criteria_source_urls)
    claim_bway = f"'{name_for_claim}' is an officially designated Broadway theater."
    await evaluator.verify(
        claim=claim_bway,
        node=official_bway_leaf,
        sources=criteria_urls if criteria_urls else None,
        additional_instruction=(
            "Prefer authoritative sources (e.g., The Broadway League, IBDB, Playbill, official theater site). "
            "The verification should confirm that this venue is classified as a Broadway theater (not Off-Broadway)."
        ),
    )

    # theater_district_location_requirement_confirmed (verification of bounds)
    location_leaf = evaluator.add_leaf(
        id="theater_district_location_requirement_confirmed",
        desc="Confirm the venue is located between 41st–54th Streets and 6th–8th Avenues in Manhattan",
        parent=details_node,
        critical=True,
    )
    address_urls = _normalize_urls(candidate.address_source_urls)
    combined_loc_urls = _normalize_urls(address_urls + criteria_urls)
    address_text = candidate.street_address or ""
    claim_location = (
        f"'{name_for_claim}' is located within Manhattan's Theater District boundaries (between 41st–54th Streets "
        f"and 6th–8th Avenues). Its address is '{address_text}'."
    )
    await evaluator.verify(
        claim=claim_location,
        node=location_leaf,
        sources=combined_loc_urls if combined_loc_urls else None,
        additional_instruction=(
            "Use the provided address sources to determine if the street number lies between 41st and 54th Streets "
            "and the avenues between 6th and 8th. If explicit phrasing is present in sources, that also suffices."
        ),
    )

    # minimum_500_seats_requirement_confirmed (verification by URLs or numeric check)
    min500_leaf = evaluator.add_leaf(
        id="minimum_500_seats_requirement_confirmed",
        desc="Confirm the venue meets the minimum 500-seat capacity requirement",
        parent=details_node,
        critical=True,
    )
    capacity_urls = _normalize_urls(candidate.capacity_source_urls)
    parsed_capacity = _parse_capacity_to_int(candidate.seating_capacity)
    claim_min500 = (
        f"'{name_for_claim}' has at least 500 seats."
    )
    add_ins_500 = (
        "Confirm using the seating capacity stated in the provided sources if available. "
        "If multiple capacity figures appear, treat the most credible or most recent. "
        "Minor formatting differences (commas) do not affect numeric value."
    )
    # If we have a parsed capacity and it is >= 500, we can mention it to help the judge
    if parsed_capacity is not None:
        add_ins_500 += f" The answer states a capacity of {parsed_capacity}, which should meet the 500-seat cutoff."
    await evaluator.verify(
        claim=claim_min500,
        node=min500_leaf,
        sources=capacity_urls if capacity_urls else None,
        additional_instruction=add_ins_500,
    )

    # Child group 3: supporting_references (critical, parallel)
    refs_node = evaluator.add_parallel(
        id="supporting_references",
        desc="Provide supporting reference URL(s) for the key claims",
        parent=root,
        critical=True,
    )

    # references_support_seating_capacity
    refs_capacity_leaf = evaluator.add_leaf(
        id="references_support_seating_capacity",
        desc="Provide reference URL(s) that support the stated exact seating capacity",
        parent=refs_node,
        critical=True,
    )
    claim_capacity_exact = f"The seating capacity of '{name_for_claim}' is '{candidate.seating_capacity or ''}'."
    add_ins_cap = (
        "Verify whether at least one of the provided URLs explicitly states the same seating capacity value as in the answer. "
        "If no URLs are provided, you must treat the claim as not supported."
    )
    await evaluator.verify(
        claim=claim_capacity_exact,
        node=refs_capacity_leaf,
        sources=capacity_urls if capacity_urls else None,
        additional_instruction=add_ins_cap,
    )

    # references_support_largest_claim
    refs_largest_leaf = evaluator.add_leaf(
        id="references_support_largest_claim",
        desc="Provide reference URL(s) that support the claim that this theater has the highest seating capacity among Broadway theaters",
        parent=refs_node,
        critical=True,
    )
    add_ins_largest = (
        "Verify that the provided URLs explicitly support the claim that this is the largest Broadway theater by seating capacity. "
        "If no URLs are provided, you must treat the claim as not supported."
    )
    await evaluator.verify(
        claim=claim_highest,
        node=refs_largest_leaf,
        sources=largest_urls if largest_urls else None,
        additional_instruction=add_ins_largest,
    )

    # references_support_street_address
    refs_address_leaf = evaluator.add_leaf(
        id="references_support_street_address",
        desc="Provide reference URL(s) that support the stated complete Manhattan street address",
        parent=refs_node,
        critical=True,
    )
    claim_address_exact = f"The complete street address of '{name_for_claim}' is '{address_text}'."
    add_ins_address = (
        "Verify whether at least one of the provided URLs explicitly shows the same address string. "
        "If no URLs are provided, you must treat the claim as not supported."
    )
    await evaluator.verify(
        claim=claim_address_exact,
        node=refs_address_leaf,
        sources=address_urls if address_urls else None,
        additional_instruction=add_ins_address,
    )

    # references_support_broadway_criteria_confirmation
    refs_criteria_leaf = evaluator.add_leaf(
        id="references_support_broadway_criteria_confirmation",
        desc="Provide reference URL(s) that support the Broadway-theater/criteria confirmation (official Broadway designation and/or district/location/500-seat criterion)",
        parent=refs_node,
        critical=True,
    )
    claim_criteria = (
        f"'{name_for_claim}' is an officially designated Broadway theater located within the Theater District boundaries and has at least 500 seats."
    )
    add_ins_criteria = (
        "At least one provided URL must explicitly confirm the Broadway designation and/or directly support the district boundaries and 500-seat criterion. "
        "If no URLs are provided, you must treat the claim as not supported."
    )
    await evaluator.verify(
        claim=claim_criteria,
        node=refs_criteria_leaf,
        sources=criteria_urls if criteria_urls else None,
        additional_instruction=add_ins_criteria,
    )


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
    evaluator = Evaluator()
    root = evaluator.initialize(
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

    candidate = await evaluator.extract(
        prompt=prompt_extract_theater_candidate(),
        template_class=BroadwayTheaterCandidate,
        extraction_name="broadway_theater_candidate",
    )

    await _build_verification_tree(evaluator, candidate, root)

    return evaluator.get_summary()