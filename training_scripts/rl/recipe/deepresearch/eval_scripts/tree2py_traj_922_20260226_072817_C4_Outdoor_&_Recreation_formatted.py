import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "modern_outdoor_recreation_and_travel_infra_2026"
TASK_DESCRIPTION = (
    "A travel and recreation researcher is compiling a comprehensive reference guide about modern outdoor recreation "
    "infrastructure and contactless travel technology in North America and the Caribbean. The guide aims to help travelers "
    "understand the latest developments in digital travel systems, outdoor airport amenities, transit payment innovations, "
    "and national park access programs as of February 2026.\n\n"
    "Please research and provide the following specific factual information:\n\n"
    "1. TSA Digital ID Program: How many U.S. states and territories currently have mobile driver's licenses or digital IDs "
    "that are accepted by TSA at airport security checkpoints?\n\n"
    "2. Denver International Airport Outdoor Amenities:\n"
    "   - How many outdoor decks with firepits are available for passengers at Denver International Airport?\n"
    "   - What are the specific gate numbers (or closest gate designations) for each of these outdoor deck locations?\n\n"
    "3. NYC OMNY Fare Cap System:\n"
    "   - Under the OMNY contactless payment system in New York City, how many rides must a full-fare passenger pay for within a "
    "7-day period before earning free rides for the remainder of that week?\n"
    "   - What is the total dollar amount (weekly fare cap) that a full-fare OMNY user will pay before qualifying for free rides?\n\n"
    "4. America the Beautiful Pass (2026):\n"
    "   - What is the cost of a 2026 America the Beautiful Resident Annual Pass for U.S. citizens and residents?\n"
    "   - What is the cost of a 2026 America the Beautiful Non-Resident Annual Pass?\n\n"
    "5. Arikok National Park (Aruba):\n"
    "   - In what year was Arikok National Park officially established?\n"
    "   - Approximately what percentage of Aruba's total land area does Arikok National Park cover?\n\n"
    "For each piece of information, provide the specific factual answer along with a reference URL that supports your answer."
)

# Ground-truth expectations per rubric (used for simple logical checks)
EXPECTED = {
    "tsa_count": 21,
    "denver_decks_count": 3,
    "denver_decks_gates": {"A15", "B7", "C67"},
    "omny_rides_to_cap": 12,
    "omny_cap_amount_allowed": {34, 35},  # per rubric: $35 (or $34) both acceptable
    "atb_resident_cost": 80,
    "atb_nonresident_cost": 250,
    "arikok_established_year": 2000,
    "arikok_coverage_percent_center": 20.0,  # approximately 20%
    "arikok_coverage_percent_tolerance": 2.0
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TSAInfo(BaseModel):
    count: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DIAInfo(BaseModel):
    deck_count: Optional[str] = None
    gate_locations: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class OMNYInfo(BaseModel):
    rides_to_cap: Optional[str] = None
    cap_amount: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ATBInfo(BaseModel):
    resident_cost: Optional[str] = None
    nonresident_cost: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ArikokInfo(BaseModel):
    establishment_year: Optional[str] = None
    land_area_percent: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AllFactsExtraction(BaseModel):
    tsa: Optional[TSAInfo] = None
    dia: Optional[DIAInfo] = None
    omny: Optional[OMNYInfo] = None
    atb: Optional[ATBInfo] = None
    arikok: Optional[ArikokInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all_facts() -> str:
    return """
Extract the specific factual items requested, exactly as stated in the answer, and list the supporting source URLs explicitly provided in the answer text for each sub-item. Do NOT invent values or URLs. If an item is not provided, return null (or an empty list for arrays).

Return a JSON object with the following structure:

{
  "tsa": {
    "count": string | null,
    "sources": string[]  // All URLs cited for TSA digital ID acceptance
  },
  "dia": {
    "deck_count": string | null,
    "gate_locations": string[],  // Extract the gate designations only (e.g., "A15", "B7", "C67"). If the answer says "near gate A15", extract "A15".
    "sources": string[]          // All URLs cited for Denver outdoor decks with firepits
  },
  "omny": {
    "rides_to_cap": string | null,   // e.g., "12"
    "cap_amount": string | null,     // e.g., "$35" or "$34"
    "sources": string[]              // All URLs cited for OMNY weekly fare cap
  },
  "atb": {
    "resident_cost": string | null,      // e.g., "$80"
    "nonresident_cost": string | null,   // e.g., "$250"
    "sources": string[]                  // All URLs cited for America the Beautiful 2026 pass pricing
  },
  "arikok": {
    "establishment_year": string | null,     // e.g., "2000"
    "land_area_percent": string | null,      // e.g., "20%" or "approximately 20%"
    "sources": string[]                      // All URLs cited for Arikok facts
  }
}

Special rules:
- For URL extraction, include only actual URLs present in the answer (including markdown links).
- For gate_locations, extract only concise gate IDs like "A15", "B7", "C67".
- Preserve the exact strings from the answer for counts, money, and percentages (do not normalize).
"""


# --------------------------------------------------------------------------- #
# Helper parsing utilities                                                    #
# --------------------------------------------------------------------------- #
def parse_first_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.findall(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m[0])
    except Exception:
        return None


def parse_money_to_int_dollars(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    # Extract something like 34, 35, 80, 250 (ignore decimals)
    m = re.findall(r"\d+(?:\.\d+)?", s.replace(",", ""))
    if not m:
        return None
    try:
        val = float(m[0])
        return int(round(val))
    except Exception:
        return None


def parse_percent_to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = re.findall(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def normalize_gate_label(s: str) -> Optional[str]:
    """
    Try to extract a gate ID like 'A15', 'B7', 'C67' from a string.
    """
    if not s:
        return None
    # Common cleanup
    s_clean = s.strip().upper()
    # Direct match like A15, B7, C67
    m = re.search(r"\b([A-Z]\d{1,3})\b", s_clean)
    if m:
        return m.group(1)
    # Try removing common words then search again
    s_clean = re.sub(r"\b(NEAR|GATE|BY|CLOSEST|TO|AT|CONCOURSE|LEVEL|OUTDOOR|PATIO|DECK|FIREPITS?)\b", " ", s_clean)
    m = re.search(r"\b([A-Z]\d{1,3})\b", s_clean)
    if m:
        return m.group(1)
    return None


def normalize_gate_set(items: List[str]) -> Set[str]:
    out: Set[str] = set()
    for it in items or []:
        lab = normalize_gate_label(it)
        if lab:
            out.add(lab)
    return out


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_and_verify_tsa(evaluator: Evaluator, parent_node, tsa: Optional[TSAInfo]) -> None:
    node = evaluator.add_parallel(
        id="tsa_digital_id_participating_states_count",
        desc="The answer correctly states that 21 U.S. states and territories have digital IDs accepted by TSA at airport checkpoints",
        parent=parent_node,
        critical=False
    )

    count_present = bool(tsa and tsa.count and tsa.count.strip())
    has_sources = bool(tsa and tsa.sources and len(tsa.sources) > 0)

    # Existence (critical gate)
    evaluator.add_custom_node(
        result=(count_present and has_sources),
        id="tsa_count_exists",
        desc="TSA digital ID acceptance count and sources are provided",
        parent=node,
        critical=True
    )

    # Expected value check (simple, custom)
    parsed_count = parse_first_int(tsa.count if tsa else None)
    evaluator.add_custom_node(
        result=(parsed_count == EXPECTED["tsa_count"]),
        id="tsa_count_expected_match",
        desc=f"Count matches expected value {EXPECTED['tsa_count']}",
        parent=node,
        critical=True  # Require both correctness and support
    )

    # Source support check
    leaf = evaluator.add_leaf(
        id="tsa_count_source_supported",
        desc="Claim is supported by cited sources (TSA accepted digital ID count)",
        parent=node,
        critical=True
    )
    claim = f"TSA accepts mobile driver's licenses or digital IDs from {tsa.count} U.S. states and territories at airport security checkpoints."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=tsa.sources if tsa else [],
        additional_instruction="Look for official TSA pages or announcements regarding accepted digital IDs (mDLs). If the page lists individual states/territories rather than a number, counting them is acceptable to verify the total."
    )


async def build_and_verify_dia_deck_count(evaluator: Evaluator, parent_node, dia: Optional[DIAInfo]) -> None:
    node = evaluator.add_parallel(
        id="denver_airport_outdoor_deck_count",
        desc="The answer correctly states that Denver International Airport has 3 outdoor decks with firepits available for passengers",
        parent=parent_node,
        critical=False
    )

    count_present = bool(dia and dia.deck_count and dia.deck_count.strip())
    has_sources = bool(dia and dia.sources)

    evaluator.add_custom_node(
        result=(count_present and has_sources),
        id="dia_deck_count_exists",
        desc="Denver outdoor deck count and sources are provided",
        parent=node,
        critical=True
    )

    parsed = parse_first_int(dia.deck_count if dia else None)
    evaluator.add_custom_node(
        result=(parsed == EXPECTED["denver_decks_count"]),
        id="dia_deck_count_expected_match",
        desc=f"Deck count matches expected value {EXPECTED['denver_decks_count']}",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="dia_deck_count_supported",
        desc="Claim is supported by cited sources (DIA outdoor decks with firepits count)",
        parent=node,
        critical=True
    )
    claim = f"Denver International Airport has {dia.deck_count} outdoor decks with firepits available for passengers."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=dia.sources if dia else [],
        additional_instruction="Check official DEN (Denver International Airport) pages or authoritative travel resources specifying the number of outdoor decks with firepits for passengers."
    )


async def build_and_verify_dia_gate_locations(evaluator: Evaluator, parent_node, dia: Optional[DIAInfo]) -> None:
    node = evaluator.add_parallel(
        id="denver_airport_outdoor_deck_gate_locations",
        desc="The answer correctly identifies all three specific gate locations for Denver Airport's outdoor decks: near gate A15, near gate B7, and near gate C67",
        parent=parent_node,
        critical=False
    )

    gates = dia.gate_locations if dia else []
    norm_gates = normalize_gate_set(gates)
    has_all_gates = len(norm_gates) >= 3
    has_sources = bool(dia and dia.sources)

    evaluator.add_custom_node(
        result=(has_all_gates and has_sources),
        id="dia_gate_locations_exist",
        desc="Gate locations and sources are provided",
        parent=node,
        critical=True
    )

    # Expected set match
    evaluator.add_custom_node(
        result=(norm_gates == EXPECTED["denver_decks_gates"]),
        id="dia_gate_locations_expected_match",
        desc="Gate locations match expected set {A15, B7, C67}",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="dia_gate_locations_supported",
        desc="Claim is supported by cited sources (DIA outdoor deck gate locations)",
        parent=node,
        critical=True
    )
    claim = "Denver International Airport's outdoor decks with firepits are located near gates A15, B7, and C67."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=dia.sources if dia else [],
        additional_instruction="Verify that the pages explicitly indicate outdoor deck locations near gates A15, B7, and C67 (or equivalent nearest gate designations)."
    )


async def build_and_verify_omny_rides_cap(evaluator: Evaluator, parent_node, omny: Optional[OMNYInfo]) -> None:
    node = evaluator.add_parallel(
        id="omny_weekly_fare_cap_rides",
        desc="The answer correctly states that a full-fare passenger must pay for 12 rides within a 7-day period to earn free rides under OMNY",
        parent=parent_node,
        critical=False
    )

    rides_present = bool(omny and omny.rides_to_cap and omny.rides_to_cap.strip())
    has_sources = bool(omny and omny.sources)

    evaluator.add_custom_node(
        result=(rides_present and has_sources),
        id="omny_rides_to_cap_exists",
        desc="OMNY rides-to-cap value and sources are provided",
        parent=node,
        critical=True
    )

    parsed = parse_first_int(omny.rides_to_cap if omny else None)
    evaluator.add_custom_node(
        result=(parsed == EXPECTED["omny_rides_to_cap"]),
        id="omny_rides_to_cap_expected_match",
        desc=f"OMNY rides-to-cap matches expected value {EXPECTED['omny_rides_to_cap']}",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="omny_rides_to_cap_supported",
        desc="Claim is supported by cited sources (OMNY rides required for weekly cap)",
        parent=node,
        critical=True
    )
    claim = f"Under OMNY, a full-fare passenger must pay for {omny.rides_to_cap} rides within a 7-day period to earn free rides for the remainder of that week."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=omny.sources if omny else [],
        additional_instruction="Confirm OMNY's weekly 'tap-to-cap' policy for full-fare riders, specifically the number of paid rides within a rolling 7-day period after which rides are free."
    )


async def build_and_verify_omny_cap_amount(evaluator: Evaluator, parent_node, omny: Optional[OMNYInfo]) -> None:
    node = evaluator.add_parallel(
        id="omny_weekly_fare_cap_amount",
        desc="The answer correctly states that the full-fare OMNY weekly cap amount is $35 (or $34, as both values appear in official OMNY documentation)",
        parent=parent_node,
        critical=False
    )

    amt_present = bool(omny and omny.cap_amount and omny.cap_amount.strip())
    has_sources = bool(omny and omny.sources)

    evaluator.add_custom_node(
        result=(amt_present and has_sources),
        id="omny_cap_amount_exists",
        desc="OMNY weekly cap amount and sources are provided",
        parent=node,
        critical=True
    )

    parsed = parse_money_to_int_dollars(omny.cap_amount if omny else None)
    allowed = EXPECTED["omny_cap_amount_allowed"]
    evaluator.add_custom_node(
        result=(parsed in allowed),
        id="omny_cap_amount_expected_match",
        desc=f"OMNY weekly cap amount matches an allowed value {sorted(list(allowed))}",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="omny_cap_amount_supported",
        desc="Claim is supported by cited sources (OMNY weekly cap dollar amount)",
        parent=node,
        critical=True
    )
    claim = f"The weekly fare cap amount for a full-fare OMNY user is {omny.cap_amount}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=omny.sources if omny else [],
        additional_instruction="Verify the weekly cap price for a full-fare OMNY rider. Some official materials may state $34 while others state $35; accept either if the page supports the claimed amount."
    )


async def build_and_verify_atb_resident(evaluator: Evaluator, parent_node, atb: Optional[ATBInfo]) -> None:
    node = evaluator.add_parallel(
        id="atb_resident_pass_cost_2026",
        desc="The answer correctly states that the 2026 America the Beautiful Resident Annual Pass costs $80",
        parent=parent_node,
        critical=False
    )

    present = bool(atb and atb.resident_cost and atb.resident_cost.strip())
    has_sources = bool(atb and atb.sources)

    evaluator.add_custom_node(
        result=(present and has_sources),
        id="atb_resident_cost_exists",
        desc="Resident pass cost and sources are provided",
        parent=node,
        critical=True
    )

    parsed = parse_money_to_int_dollars(atb.resident_cost if atb else None)
    evaluator.add_custom_node(
        result=(parsed == EXPECTED["atb_resident_cost"]),
        id="atb_resident_cost_expected_match",
        desc=f"Resident pass cost matches expected value ${EXPECTED['atb_resident_cost']}",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="atb_resident_cost_supported",
        desc="Claim is supported by cited sources (ATB 2026 Resident Annual Pass cost)",
        parent=node,
        critical=True
    )
    claim = f"The 2026 America the Beautiful Resident Annual Pass costs {atb.resident_cost}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=atb.sources if atb else [],
        additional_instruction="Verify the price for the 'America the Beautiful – The National Parks & Federal Recreational Lands Pass' for 2026 (Resident Annual Pass)."
    )


async def build_and_verify_atb_nonresident(evaluator: Evaluator, parent_node, atb: Optional[ATBInfo]) -> None:
    node = evaluator.add_parallel(
        id="atb_nonresident_pass_cost_2026",
        desc="The answer correctly states that the 2026 America the Beautiful Non-Resident Annual Pass costs $250",
        parent=parent_node,
        critical=False
    )

    present = bool(atb and atb.nonresident_cost and atb.nonresident_cost.strip())
    has_sources = bool(atb and atb.sources)

    evaluator.add_custom_node(
        result=(present and has_sources),
        id="atb_nonresident_cost_exists",
        desc="Non-resident pass cost and sources are provided",
        parent=node,
        critical=True
    )

    parsed = parse_money_to_int_dollars(atb.nonresident_cost if atb else None)
    evaluator.add_custom_node(
        result=(parsed == EXPECTED["atb_nonresident_cost"]),
        id="atb_nonresident_cost_expected_match",
        desc=f"Non-resident pass cost matches expected value ${EXPECTED['atb_nonresident_cost']}",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="atb_nonresident_cost_supported",
        desc="Claim is supported by cited sources (ATB 2026 Non-Resident Annual Pass cost)",
        parent=node,
        critical=True
    )
    claim = f"The 2026 America the Beautiful Non-Resident Annual Pass costs {atb.nonresident_cost}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=atb.sources if atb else [],
        additional_instruction="Verify the stated 2026 price for a 'Non-Resident Annual Pass' associated with America the Beautiful program, if applicable per cited sources."
    )


async def build_and_verify_arikok_year(evaluator: Evaluator, parent_node, arikok: Optional[ArikokInfo]) -> None:
    node = evaluator.add_parallel(
        id="arikok_establishment_year",
        desc="The answer correctly states that Arikok National Park was officially established in 2000",
        parent=parent_node,
        critical=False
    )

    present = bool(arikok and arikok.establishment_year and arikok.establishment_year.strip())
    has_sources = bool(arikok and arikok.sources)

    evaluator.add_custom_node(
        result=(present and has_sources),
        id="arikok_year_exists",
        desc="Arikok establishment year and sources are provided",
        parent=node,
        critical=True
    )

    parsed = parse_first_int(arikok.establishment_year if arikok else None)
    evaluator.add_custom_node(
        result=(parsed == EXPECTED["arikok_established_year"]),
        id="arikok_year_expected_match",
        desc=f"Establishment year matches expected value {EXPECTED['arikok_established_year']}",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="arikok_year_supported",
        desc="Claim is supported by cited sources (Arikok establishment year)",
        parent=node,
        critical=True
    )
    claim = f"Arikok National Park was officially established in {arikok.establishment_year}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=arikok.sources if arikok else [],
        additional_instruction="Check Arikok National Park's official site or authoritative government/tourism resources for the official establishment year."
    )


async def build_and_verify_arikok_percent(evaluator: Evaluator, parent_node, arikok: Optional[ArikokInfo]) -> None:
    node = evaluator.add_parallel(
        id="arikok_coverage_percentage",
        desc="The answer correctly states that Arikok National Park covers approximately 20% of Aruba's land area",
        parent=parent_node,
        critical=False
    )

    present = bool(arikok and arikok.land_area_percent and arikok.land_area_percent.strip())
    has_sources = bool(arikok and arikok.sources)

    evaluator.add_custom_node(
        result=(present and has_sources),
        id="arikok_percent_exists",
        desc="Arikok land area percentage and sources are provided",
        parent=node,
        critical=True
    )

    parsed = parse_percent_to_float(arikok.land_area_percent if arikok else None)
    target = EXPECTED["arikok_coverage_percent_center"]
    tol = EXPECTED["arikok_coverage_percent_tolerance"]
    within_range = (parsed is not None) and (abs(parsed - target) <= tol)

    evaluator.add_custom_node(
        result=within_range,
        id="arikok_percent_expected_match",
        desc=f"Coverage percentage is approximately {target}% (±{tol}%)",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="arikok_percent_supported",
        desc="Claim is supported by cited sources (Arikok coverage percentage)",
        parent=node,
        critical=True
    )
    claim = f"Arikok National Park covers approximately {arikok.land_area_percent} of Aruba's land area."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=arikok.sources if arikok else [],
        additional_instruction="Verify that the percentage of Aruba’s land covered by Arikok is approximately 20% (minor variation acceptable)."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict:
    """
    Evaluate an answer for the Modern Outdoor Recreation and Travel Infrastructure facts task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # rubric root is parallel
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

    # Add a top-level grouping node to mirror rubric section
    facts_root = evaluator.add_parallel(
        id="modern_outdoor_recreation_and_travel_infra_facts",
        desc="Verifies that the answer provides accurate, specific factual information about modern outdoor recreation and travel infrastructure across multiple domains, as requested in the question",
        parent=root,
        critical=False
    )

    # Extract structured facts from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all_facts(),
        template_class=AllFactsExtraction,
        extraction_name="facts_extraction"
    )

    # Add ground truth/expectations for transparency
    evaluator.add_ground_truth({
        "tsa_count_expected": EXPECTED["tsa_count"],
        "denver_decks_count_expected": EXPECTED["denver_decks_count"],
        "denver_decks_gates_expected": sorted(list(EXPECTED["denver_decks_gates"])),
        "omny_rides_to_cap_expected": EXPECTED["omny_rides_to_cap"],
        "omny_cap_amount_allowed": sorted(list(EXPECTED["omny_cap_amount_allowed"])),
        "atb_resident_cost_expected": EXPECTED["atb_resident_cost"],
        "atb_nonresident_cost_expected": EXPECTED["atb_nonresident_cost"],
        "arikok_established_year_expected": EXPECTED["arikok_established_year"],
        "arikok_coverage_percent_expected": f"~{EXPECTED['arikok_coverage_percent_center']}% ±{EXPECTED['arikok_coverage_percent_tolerance']}%"
    })

    # Build and run verification subtrees
    await build_and_verify_tsa(evaluator, facts_root, extracted.tsa if extracted else None)
    await build_and_verify_dia_deck_count(evaluator, facts_root, extracted.dia if extracted else None)
    await build_and_verify_dia_gate_locations(evaluator, facts_root, extracted.dia if extracted else None)
    await build_and_verify_omny_rides_cap(evaluator, facts_root, extracted.omny if extracted else None)
    await build_and_verify_omny_cap_amount(evaluator, facts_root, extracted.omny if extracted else None)
    await build_and_verify_atb_resident(evaluator, facts_root, extracted.atb if extracted else None)
    await build_and_verify_atb_nonresident(evaluator, facts_root, extracted.atb if extracted else None)
    await build_and_verify_arikok_year(evaluator, facts_root, extracted.arikok if extracted else None)
    await build_and_verify_arikok_percent(evaluator, facts_root, extracted.arikok if extracted else None)

    # Return structured summary
    return evaluator.get_summary()