import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "radio_city_accessibility_safety"
TASK_DESCRIPTION = (
    "Radio City Music Hall in New York City is preparing for a major concert series and must verify its compliance with accessibility and safety regulations. "
    "Answer the following questions in sequence: "
    "(1) What is Radio City Music Hall's total seating capacity? "
    "(2) Based on that capacity, what is the minimum number of wheelchair-accessible seating spaces required under ADA Table 221.2.1.1 for assembly areas? "
    "(3) Under ADA requirements, what is the minimum number of companion seats that must be provided alongside those wheelchair-accessible spaces (expressed as a ratio)? "
    "(4) Under ADA ticketing requirements, how many additional companion seat tickets (beyond the required companion seat) may a patron purchasing one wheelchair-accessible seat ticket buy for seats located next to the wheelchair space? "
    "(5) To ensure safe emergency egress for all patrons, what is the minimum illumination level (in foot-candles) that emergency exit signs must maintain according to OSHA standard 1910.37?"
)


class AccessibilitySafetyExtraction(BaseModel):
    seating_capacity: Optional[str] = None
    seating_capacity_sources: List[str] = Field(default_factory=list)

    ada_wheelchair_min: Optional[str] = None
    ada_wheelchair_sources: List[str] = Field(default_factory=list)

    ada_companion_ratio: Optional[str] = None
    ada_companion_ratio_sources: List[str] = Field(default_factory=list)

    ada_ticketing_additional_companion: Optional[str] = None
    ada_ticketing_sources: List[str] = Field(default_factory=list)

    osha_exit_sign_min_foot_candles: Optional[str] = None
    osha_exit_sign_sources: List[str] = Field(default_factory=list)


def prompt_extract_all() -> str:
    return """
    Extract the specific answers and their cited URLs for the five requested items, based strictly on what is stated in the answer text. Return exactly the phrasing used by the answer for values (do not normalize), and extract all URLs the answer cites for each item.

    Fields to extract:
    1) seating_capacity: The stated total seating capacity of Radio City Music Hall in NYC (as a string exactly as in the answer).
    2) seating_capacity_sources: All URLs the answer cites to support the capacity figure.

    3) ada_wheelchair_min: The stated minimum number of wheelchair-accessible seating spaces required, derived from ADA Table 221.2.1.1 (as a string exactly as in the answer).
    4) ada_wheelchair_sources: All URLs the answer cites for ADA Table 221.2.1.1 or other authoritative ADA sources used for the wheelchair-space requirement.

    5) ada_companion_ratio: The stated minimum companion-seat requirement expressed as a ratio relative to wheelchair-accessible spaces (e.g., "1:1", "one companion seat per wheelchair space").
    6) ada_companion_ratio_sources: All URLs the answer cites for the ADA companion-seat requirement.

    7) ada_ticketing_additional_companion: The stated number of additional companion seat tickets (beyond the required companion seat) that may be purchased with one wheelchair-accessible seat ticket for adjacent seats.
    8) ada_ticketing_sources: All URLs the answer cites for ADA ticketing requirements.

    9) osha_exit_sign_min_foot_candles: The stated minimum illumination level (foot-candles) for emergency exit signs per OSHA 1910.37.
    10) osha_exit_sign_sources: All URLs the answer cites for OSHA 1910.37 exit sign illumination.

    Notes:
    - Return null for any field not stated in the answer.
    - Only extract URLs that appear explicitly in the answer text (including markdown links). If there are no URLs for a field, return an empty array.
    """


def parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    s = text.strip()
    s = s.replace(",", "")
    match = re.search(r"(\d+)", s)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def compute_ada_min_wheelchair_spaces_5001_plus(capacity: Optional[int]) -> Optional[int]:
    """
    ADA 2010 Standards, Table 221.2.1.1 – For assembly areas with 5,001+ seats:
    Minimum wheelchair spaces = 36 + 1 per each 200 seats (or fraction) over 5,000.
    """
    if capacity is None:
        return None
    if capacity >= 5001:
        over = capacity - 5000
        additional = (over + 199) // 200
        return 36 + additional
    return None


async def verify_q1_q3_sequence(evaluator: Evaluator, parent_node, extracted: AccessibilitySafetyExtraction) -> None:
    seq_node = evaluator.add_sequential(
        id="ADA_Accessibility_Steps_Q1_to_Q3",
        desc="Verify the sequential ADA accessibility determinations that depend on seating capacity (Q1→Q2→Q3).",
        parent=parent_node,
        critical=True
    )

    # Q1: Seating capacity
    q1_node = evaluator.add_parallel(
        id="Q1_Seating_Capacity",
        desc="State the total seating capacity of Radio City Music Hall (NYC), determined from current venue specifications.",
        parent=seq_node,
        critical=True
    )

    q1_exists = evaluator.add_custom_node(
        result=bool(extracted.seating_capacity and extracted.seating_capacity.strip()),
        id="Q1_capacity_exists",
        desc="Q1: The seating capacity value is provided",
        parent=q1_node,
        critical=True
    )

    q1_supported = evaluator.add_leaf(
        id="Q1_capacity_supported_by_sources",
        desc="Q1: The stated seating capacity is supported by cited sources",
        parent=q1_node,
        critical=True
    )
    cap_str = extracted.seating_capacity or ""
    await evaluator.verify(
        claim=f"Radio City Music Hall's total seating capacity is '{cap_str}'.",
        node=q1_supported,
        sources=extracted.seating_capacity_sources,
        additional_instruction=(
            "Verify that at least one cited source explicitly states the same capacity figure as claimed. "
            "Allow minor formatting differences (e.g., commas, the word 'seats')."
        )
    )

    cap_int = parse_int_from_text(extracted.seating_capacity)
    q1_parsed = evaluator.add_custom_node(
        result=cap_int is not None and cap_int > 0,
        id="Q1_capacity_parsed_int",
        desc=f"Q1: Seating capacity contains a parseable positive integer ({cap_int if cap_int is not None else 'None'})",
        parent=q1_node,
        critical=True
    )

    # Q2: Minimum wheelchair-accessible seating spaces
    q2_node = evaluator.add_parallel(
        id="Q2_Min_Wheelchair_Accessible_Spaces",
        desc="Using the stated seating capacity, compute the minimum required wheelchair-accessible seating spaces per ADA Table 221.2.1.1.",
        parent=seq_node,
        critical=True
    )

    q2_exists = evaluator.add_custom_node(
        result=bool(extracted.ada_wheelchair_min and extracted.ada_wheelchair_min.strip()),
        id="Q2_wheelchair_min_exists",
        desc="Q2: The minimum wheelchair-accessible spaces value is provided",
        parent=q2_node,
        critical=True
    )

    # Formula check for 5001+ seats branch
    provided_wheel_min_int = parse_int_from_text(extracted.ada_wheelchair_min)
    expected_wheel_min_int = compute_ada_min_wheelchair_spaces_5001_plus(cap_int)
    formula_ok = (expected_wheel_min_int is not None) and (provided_wheel_min_int == expected_wheel_min_int)

    q2_formula_check = evaluator.add_custom_node(
        result=formula_ok,
        id="Q2_wheelchair_min_formula_5001plus_check",
        desc=(
            f"Q2: Formula check (for capacities ≥5001): expected {expected_wheel_min_int}, "
            f"provided {provided_wheel_min_int}"
        ),
        parent=q2_node,
        critical=True
    )

    q2_source_support = evaluator.add_leaf(
        id="Q2_ada_table_rule_supported",
        desc="Q2: ADA Table 221.2.1.1 states 36 + 1 per each 200 seats (or fraction) over 5,000 for 5,001+ seats",
        parent=q2_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "ADA 2010 Standards Table 221.2.1.1 for assembly areas requires, for facilities with more than 5,000 seats, "
            "a minimum of 36 wheelchair spaces plus one additional space for each 200 seats (or fraction thereof) over 5,000."
        ),
        node=q2_source_support,
        sources=extracted.ada_wheelchair_sources,
        additional_instruction=(
            "Confirm the exact rule text on the cited ADA standard page(s). The verification focuses on the 5,001+ seats formula."
        )
    )

    # Q3: Minimum companion seat ratio
    q3_node = evaluator.add_parallel(
        id="Q3_Min_Companion_Seat_Ratio",
        desc="State the ADA minimum companion-seat requirement as a ratio relative to wheelchair-accessible spaces (at least 1 companion seat per wheelchair space).",
        parent=seq_node,
        critical=True
    )

    q3_exists = evaluator.add_custom_node(
        result=bool(extracted.ada_companion_ratio and extracted.ada_companion_ratio.strip()),
        id="Q3_companion_ratio_exists",
        desc="Q3: The companion-seat ratio is provided",
        parent=q3_node,
        critical=True
    )

    q3_source_support = evaluator.add_leaf(
        id="Q3_companion_ratio_supported",
        desc="Q3: ADA requires at least one companion seat per wheelchair space (1:1 minimum)",
        parent=q3_node,
        critical=True
    )
    ratio_str = extracted.ada_companion_ratio or ""
    await evaluator.verify(
        claim=(
            "Under the ADA 2010 Standards for assembly seating, at least one companion seat must be provided for each "
            "wheelchair-accessible space (i.e., a minimum 1:1 ratio)."
        ),
        node=q3_source_support,
        sources=extracted.ada_companion_ratio_sources,
        additional_instruction=(
            "Verify the ADA companion seating requirement. Minor variations in wording are acceptable as long as the rule "
            "clearly states at least one companion seat per wheelchair space."
        )
    )


async def verify_q4_q5(evaluator: Evaluator, parent_node, extracted: AccessibilitySafetyExtraction) -> None:
    # Q4: ADA ticketing - additional companion seats purchasable
    q4_node = evaluator.add_parallel(
        id="Q4_ADA_Ticketing_Additional_Companion_Tickets",
        desc="State, under ADA ticketing requirements, how many additional companion seat tickets (beyond the required companion seat) can be purchased for adjacent seats.",
        parent=parent_node,
        critical=True
    )

    q4_exists = evaluator.add_custom_node(
        result=bool(extracted.ada_ticketing_additional_companion and extracted.ada_ticketing_additional_companion.strip()),
        id="Q4_additional_companion_exists",
        desc="Q4: The number of additional companion seat tickets is provided",
        parent=q4_node,
        critical=True
    )

    q4_supported = evaluator.add_leaf(
        id="Q4_additional_companion_supported",
        desc="Q4: ADA ticketing rule about additional companion seats is supported by cited sources",
        parent=q4_node,
        critical=True
    )
    add_comp_str = extracted.ada_ticketing_additional_companion or ""
    await evaluator.verify(
        claim=(
            f"Under ADA ticketing requirements, a patron purchasing one wheelchair-accessible seat ticket may purchase "
            f"{add_comp_str} additional companion seat ticket(s) for seats located next to the wheelchair space (subject to availability)."
        ),
        node=q4_supported,
        sources=extracted.ada_ticketing_sources,
        additional_instruction=(
            "Verify against ADA Title III ticketing guidance (2010 revisions). The claim concerns how many additional companion seats "
            "beyond the required one companion seat may be purchased together with a wheelchair seat."
        )
    )

    # Q5: OSHA exit sign illumination
    q5_node = evaluator.add_parallel(
        id="Q5_OSHA_Exit_Sign_Illumination",
        desc="State the minimum emergency-exit-sign illumination level (in foot-candles) required by OSHA 1910.37.",
        parent=parent_node,
        critical=True
    )

    q5_exists = evaluator.add_custom_node(
        result=bool(extracted.osha_exit_sign_min_foot_candles and extracted.osha_exit_sign_min_foot_candles.strip()),
        id="Q5_exit_sign_illumination_exists",
        desc="Q5: The minimum foot-candles value for exit sign illumination is provided",
        parent=q5_node,
        critical=True
    )

    q5_supported = evaluator.add_leaf(
        id="Q5_exit_sign_illumination_supported",
        desc="Q5: OSHA 1910.37 minimum exit sign illumination level (foot-candles) is supported by cited sources",
        parent=q5_node,
        critical=True
    )
    fc_str = extracted.osha_exit_sign_min_foot_candles or ""
    await evaluator.verify(
        claim=(
            f"OSHA standard 1910.37 requires emergency exit signs to be illuminated to at least {fc_str} foot-candles."
        ),
        node=q5_supported,
        sources=extracted.osha_exit_sign_sources,
        additional_instruction=(
            "Confirm the specified minimum foot-candles on the cited OSHA (or authoritative) page. "
            "Allow minor unit formatting variations (e.g., 'foot-candles', 'fc')."
        )
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AccessibilitySafetyExtraction,
        extraction_name="accessibility_safety_extraction"
    )

    comp_root = evaluator.add_parallel(
        id="Root_Compliance_Verification",
        desc="Verify that all five requested accessibility and safety determinations for Radio City Music Hall are correctly provided.",
        parent=root,
        critical=True
    )

    await verify_q1_q3_sequence(evaluator, comp_root, extracted)
    await verify_q4_q5(evaluator, comp_root, extracted)

    # Record helpful computed info
    cap_int = parse_int_from_text(extracted.seating_capacity)
    expected_wheel_min_int = compute_ada_min_wheelchair_spaces_5001_plus(cap_int)
    evaluator.add_custom_info(
        info={
            "parsed_seating_capacity_int": cap_int,
            "expected_wheelchair_min_5001plus": expected_wheel_min_int,
            "provided_wheelchair_min_raw": extracted.ada_wheelchair_min
        },
        info_type="computed_helpers",
        info_name="ada_computation_helpers"
    )

    return evaluator.get_summary()