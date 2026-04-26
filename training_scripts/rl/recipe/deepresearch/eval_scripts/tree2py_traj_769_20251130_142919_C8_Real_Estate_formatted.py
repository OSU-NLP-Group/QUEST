import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "leed_energy_star_offices_portfolio"
TASK_DESCRIPTION = (
    "A commercial real estate investment firm specializing in sustainable properties is expanding its portfolio and has "
    "tasked their research team with identifying premium office buildings that demonstrate top-tier environmental performance. "
    "The firm requires buildings with both LEED and Energy Star certifications to ensure comprehensive sustainability credentials.\n\n"
    "Identify 4 commercial office buildings that meet ALL of the following criteria:\n\n"
    "Location & Building Type:\n"
    "- Located in one of these U.S. cities: Chicago (Illinois), Seattle (Washington), Denver (Colorado), Atlanta (Georgia), or Phoenix (Arizona)\n"
    "- Commercial office building (office use as primary function)\n\n"
    "Physical Specifications:\n"
    "- Minimum 50,000 square feet of total building area\n"
    "- Minimum 5 stories in height\n\n"
    "Environmental Certifications (BOTH required):\n"
    "- LEED certification at Gold level (60-79 points) OR Platinum level (80+ points)\n"
    "- Energy Star certification with a score of 75 or higher\n\n"
    "Code Compliance (standard for commercial office buildings):\n"
    "- Fire protection systems installed\n"
    "- Elevator access to all floors\n"
    "- ADA accessibility compliance\n"
    "- Multiple means of egress\n\n"
    "Provide the following information for each building:\n"
    "1. Official building name\n"
    "2. Complete street address (street, city, state, ZIP code)\n"
    "3. Total building square footage\n"
    "4. Number of stories\n"
    "5. LEED certification level (Gold or Platinum) and points achieved\n"
    "6. Energy Star performance score\n"
    "7. Reference URL(s) that document the building information and certifications\n\n"
    "Requirements: All 4 buildings must be distinct properties (not multiple buildings within the same development). "
    "They may be in the same city or spread across multiple cities from the specified list."
)

ALLOWED_CITIES: List[Tuple[str, str]] = [
    ("chicago", "il"),
    ("seattle", "wa"),
    ("denver", "co"),
    ("atlanta", "ga"),
    ("phoenix", "az"),
]


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class Building(BaseModel):
    name: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None

    total_sqft: Optional[str] = None
    stories: Optional[str] = None

    leed_level: Optional[str] = None  # Expected: "Gold" or "Platinum"
    leed_points: Optional[str] = None  # Points achieved (e.g., "72", "85", etc.)

    energy_star_score: Optional[str] = None  # e.g., "82"

    primary_use: Optional[str] = None  # Should indicate "office" as primary

    reference_urls: List[str] = Field(default_factory=list)


class BuildingsExtraction(BaseModel):
    buildings: List[Building] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_buildings() -> str:
    return """
    Extract up to 6 commercial office buildings mentioned in the answer that are proposed as candidates. For each building, extract the following fields exactly as stated in the answer (do not invent):

    - name: Official building name
    - street: Street address (exclude city/state/ZIP)
    - city: City name
    - state: Two-letter U.S. state abbreviation
    - zip_code: 5-digit ZIP code (include 4-digit extension if present)
    - total_sqft: Total building square footage (e.g., "520,000", "520k sq ft")
    - stories: Number of stories (e.g., "5", "12 floors")
    - leed_level: LEED certification level (expect "Gold" or "Platinum")
    - leed_points: LEED points achieved (numeric if available; extract number string)
    - energy_star_score: The Energy Star score (numeric string if available)
    - primary_use: The primary use of the building as described (e.g., "office", "mixed-use with primary office")
    - reference_urls: An array of all URLs cited for this building that document building details and/or certifications.
    
    Special rules:
    - If a field is missing in the answer, set it to null (or [] for the URL list).
    - Normalize URLs where possible (include http:// or https://).
    - Do not infer or fabricate values; only extract explicitly stated content from the provided answer.
    - If the answer provides more than 4 buildings, still extract all you find (we will later select the first 4).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", s.strip().lower()).strip()


def parse_int_relaxed(s: Optional[str]) -> Optional[int]:
    """
    Parse a relaxed integer from strings like:
    - "50,000", "50000", "50k", "1.2M", "approx. 55000", "82+"
    Returns None if no numeric information can be extracted.
    """
    if not s:
        return None
    x = s.strip().lower()
    x = x.replace(",", "").replace("+", "").strip()

    # Match formats with suffix k/m
    m = re.search(r"(\d+(?:\.\d+)?)\s*([km])\b", x)
    if m:
        val = float(m.group(1))
        suf = m.group(2)
        if suf == "k":
            return int(val * 1_000)
        if suf == "m":
            return int(val * 1_000_000)

    # Plain integer (possibly inside text)
    m2 = re.search(r"(\d{1,9})", x)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    return None


def is_valid_url(u: str) -> bool:
    return bool(re.match(r"^https?://", u.strip()))


def normalize_state(us_state: Optional[str]) -> Optional[str]:
    if not us_state:
        return None
    return us_state.strip().upper()


def location_in_allowed(city: Optional[str], state: Optional[str]) -> bool:
    if not city or not state:
        return False
    c = normalize_text(city)
    s = normalize_state(state)
    if not s:
        return False
    for ac, as_ in ALLOWED_CITIES:
        if c == ac and s == as_.upper():
            return True
    return False


def build_full_address(b: Building) -> str:
    parts = [p for p in [b.street, b.city, b.state, b.zip_code] if p and p.strip()]
    return ", ".join(parts)


def unique_building_key(b: Building) -> str:
    """
    Construct a canonical uniqueness key combining name + street + zip.
    """
    name = normalize_text(b.name)
    street = normalize_text(b.street)
    zipc = normalize_text(b.zip_code)
    if name or street or zipc:
        return f"{name}|{street}|{zipc}"
    # If nothing is available, return empty key to mark as unusable for uniqueness
    return ""


# --------------------------------------------------------------------------- #
# Verification per building                                                   #
# --------------------------------------------------------------------------- #
async def verify_building(evaluator: Evaluator, parent_node, b: Building, idx: int) -> None:
    """
    Build the verification sub-tree for one building.
    """
    building_node = evaluator.add_parallel(
        id=f"building_{idx+1}",
        desc=f"{idx+1}st building meets all criteria and required fields are provided" if idx == 0 else (
            f"{idx+1}nd building meets all criteria and required fields are provided" if idx == 1 else (
                f"{idx+1}rd building meets all criteria and required fields are provided" if idx == 2 else
                f"{idx+1}th building meets all criteria and required fields are provided"
            )
        ),
        parent=parent_node,
        critical=False,
    )

    # 1) Official building name provided (critical)
    name_ok = bool(b.name and b.name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id=f"b{idx+1}_name_provided",
        desc="Official building name is provided",
        parent=building_node,
        critical=True,
    )

    # 2) Address provided: street, city, state, ZIP (critical)
    addr_ok = all([b.street and b.street.strip(), b.city and b.city.strip(), b.state and b.state.strip(), b.zip_code and b.zip_code.strip()])
    evaluator.add_custom_node(
        result=addr_ok,
        id=f"b{idx+1}_address_provided",
        desc="Complete street address (street, city, state, ZIP) is provided",
        parent=building_node,
        critical=True,
    )

    # 3) Location in allowed cities (critical)
    location_ok = location_in_allowed(b.city, b.state)
    evaluator.add_custom_node(
        result=location_ok,
        id=f"b{idx+1}_location_allowed",
        desc="Building is located in Chicago (IL), Seattle (WA), Denver (CO), Atlanta (GA), or Phoenix (AZ)",
        parent=building_node,
        critical=True,
    )

    # 4) Office use is primary (critical) - verify via sources
    office_leaf = evaluator.add_leaf(
        id=f"b{idx+1}_office_use_primary",
        desc="Building is a commercial office building (office use is primary function)",
        parent=building_node,
        critical=True,
    )
    office_claim = "This property is a commercial office building and its primary use is office."
    await evaluator.verify(
        claim=office_claim,
        node=office_leaf,
        sources=b.reference_urls,
        additional_instruction="Treat 'office building', 'office tower', 'Class A office', or equivalent as primary office use. "
                               "Allow minor wording variants. The webpage must clearly indicate office as the primary use.",
    )

    # 5) Total square footage provided and >= 50,000 (critical)
    sqft_leaf = evaluator.add_leaf(
        id=f"b{idx+1}_sqft_min",
        desc="Total building square footage is provided and is at least 50,000 sq ft",
        parent=building_node,
        critical=True,
    )
    sqft_val = parse_int_relaxed(b.total_sqft)
    if sqft_val is None or sqft_val < 50000:
        # Fail early if not provided or below threshold
        sqft_leaf.score = 0.0
        sqft_leaf.status = "failed"
    else:
        sqft_claim = f"The building's total floor area is {sqft_val} square feet, which is at least 50,000 square feet."
        await evaluator.verify(
            claim=sqft_claim,
            node=sqft_leaf,
            sources=b.reference_urls,
            additional_instruction="Confirm the total building area (gross floor area). Allow minor rounding differences or approximate ranges "
                                   "as long as it clearly meets >=50,000 sq ft.",
        )

    # 6) Stories provided and >= 5 (critical)
    stories_leaf = evaluator.add_leaf(
        id=f"b{idx+1}_stories_min",
        desc="Number of stories is provided and is at least 5 stories",
        parent=building_node,
        critical=True,
    )
    stories_val = parse_int_relaxed(b.stories)
    if stories_val is None or stories_val < 5:
        stories_leaf.score = 0.0
        stories_leaf.status = "failed"
    else:
        stories_claim = f"The building has {stories_val} stories, which is at least 5."
        await evaluator.verify(
            claim=stories_claim,
            node=stories_leaf,
            sources=b.reference_urls,
            additional_instruction="The number of stories may be phrased as 'floors'. Minor discrepancies such as counting mezzanines should still be acceptable "
                                   "if the page shows the building is >= 5 stories.",
        )

    # 7) LEED certification level (Gold/Platinum) and points provided, and eligible (critical)
    leed_leaf = evaluator.add_leaf(
        id=f"b{idx+1}_leed_level_points",
        desc="LEED certification level (Gold or Platinum) AND points achieved are provided, and level is Gold (60–79) or Platinum (80+)",
        parent=building_node,
        critical=True,
    )
    level_norm = (b.leed_level or "").strip().lower()
    points_val = parse_int_relaxed(b.leed_points)
    level_ok = level_norm in {"gold", "platinum"}
    points_ok = points_val is not None
    eligibility_ok = False
    if level_norm == "gold" and points_val is not None:
        eligibility_ok = 60 <= points_val <= 79
    if level_norm == "platinum" and points_val is not None:
        eligibility_ok = points_val >= 80

    if not (level_ok and points_ok and eligibility_ok):
        leed_leaf.score = 0.0
        leed_leaf.status = "failed"
    else:
        # Formulate the claim for verification by URLs
        level_title = "Gold" if level_norm == "gold" else "Platinum"
        threshold_text = "60–79 points for Gold; 80+ for Platinum"
        leed_claim = f"This building holds a LEED {level_title} certification with {points_val} points."
        await evaluator.verify(
            claim=leed_claim,
            node=leed_leaf,
            sources=b.reference_urls,
            additional_instruction=f"Confirm that the page states LEED {level_title} and shows {points_val} points. "
                                   f"Consider LEED versions (v3/v4/v4.1) acceptable. The level/points should be consistent with {threshold_text}.",
        )

    # 8) Energy Star score provided and >= 75 (critical)
    es_leaf = evaluator.add_leaf(
        id=f"b{idx+1}_energystar_min",
        desc="Energy Star performance score is provided and is at least 75",
        parent=building_node,
        critical=True,
    )
    es_val = parse_int_relaxed(b.energy_star_score)
    if es_val is None or es_val < 75:
        es_leaf.score = 0.0
        es_leaf.status = "failed"
    else:
        es_claim = f"The building has an Energy Star score of {es_val}, which is at least 75."
        await evaluator.verify(
            claim=es_claim,
            node=es_leaf,
            sources=b.reference_urls,
            additional_instruction="Confirm the stated Energy Star score for this building. Allow minor rounding differences. "
                                   "If the page shows certification year with the score, that's acceptable.",
        )

    # 9) Code compliance claims (critical parent with 4 critical leaves)
    code_parent = evaluator.add_parallel(
        id=f"b{idx+1}_code_main",
        desc="Building is stated/documented to meet the code-compliance requirements listed in the question",
        parent=building_node,
        critical=True,
    )
    # 9.a Fire protection systems
    code_fire = evaluator.add_leaf(
        id=f"b{idx+1}_code_fire",
        desc="Fire protection systems installed",
        parent=code_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The building is equipped with fire protection systems (e.g., sprinklers and/or fire alarm systems) appropriate for commercial office buildings.",
        node=code_fire,
        sources=b.reference_urls,
        additional_instruction="Look for explicit mentions like 'sprinklered', 'automatic fire sprinkler system', 'NFPA compliance', or similar. "
                               "If the page content clearly implies installed fire protection systems for the office building, accept.",
    )

    # 9.b Elevator access to all floors
    code_elev = evaluator.add_leaf(
        id=f"b{idx+1}_code_elevator",
        desc="Elevator access to all floors",
        parent=code_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The building has elevator access to all floors.",
        node=code_elev,
        sources=b.reference_urls,
        additional_instruction="Accept phrasing like 'elevators serve all floors', 'full elevator access', or implicit indications on the building's specifications page.",
    )

    # 9.c ADA accessibility compliance
    code_ada = evaluator.add_leaf(
        id=f"b{idx+1}_code_ada",
        desc="ADA accessibility compliance",
        parent=code_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The building complies with ADA accessibility requirements.",
        node=code_ada,
        sources=b.reference_urls,
        additional_instruction="Evidence might include 'ADA compliant', 'accessible entrances/restrooms/elevators', or equivalent accessibility statements on official or credible sources.",
    )

    # 9.d Multiple means of egress
    code_egress = evaluator.add_leaf(
        id=f"b{idx+1}_code_egress",
        desc="Multiple means of egress",
        parent=code_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The building has multiple means of egress consistent with commercial office building codes.",
        node=code_egress,
        sources=b.reference_urls,
        additional_instruction="Look for statements like 'multiple stairwells', 'two or more means of egress', or building plans/specifications indicating more than one egress path.",
    )

    # 10) Reference URLs provided (critical)
    urls_ok = any(is_valid_url(u) for u in (b.reference_urls or []))
    evaluator.add_custom_node(
        result=urls_ok,
        id=f"b{idx+1}_refs_provided",
        desc="At least one reference URL is provided that documents the building information and certifications",
        parent=building_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Distinct properties across the portfolio                                    #
# --------------------------------------------------------------------------- #
def check_distinct_properties(buildings: List[Building]) -> bool:
    """
    Ensure all 4 buildings are distinct properties.
    We require 4 non-empty uniqueness keys and no duplicates among them.
    """
    keys = []
    for b in buildings:
        k = unique_building_key(b)
        keys.append(k)

    # Require that each building has a non-empty key
    if any(k == "" for k in keys):
        return False

    return len(set(keys)) == 4


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the answer for the LEED + Energy Star commercial office buildings portfolio task.
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

    # Extract candidate buildings from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_buildings(),
        template_class=BuildingsExtraction,
        extraction_name="buildings_extraction",
    )

    # Select exactly 4 buildings (first 4 if more; pad with empty if fewer)
    buildings = list(extracted.buildings[:4])
    while len(buildings) < 4:
        buildings.append(Building())

    # Portfolio evaluation node (root already parallel and non-critical)
    portfolio_node = root  # alias for clarity

    # Verify each building subtree
    for i in range(4):
        await verify_building(evaluator, portfolio_node, buildings[i], i)

    # Distinct properties across all 4 (critical)
    distinct_result = check_distinct_properties(buildings)
    evaluator.add_custom_node(
        result=distinct_result,
        id="distinct_properties_all_4",
        desc="All 4 buildings are distinct properties (no duplicates / not the same building repeated; not multiple buildings within the same development as a single property)",
        parent=portfolio_node,
        critical=True,
    )

    # Optionally add custom info for debugging
    evaluator.add_custom_info(
        info={
            "allowed_cities": [{"city": c.title(), "state": s.upper()} for c, s in ALLOWED_CITIES],
            "total_extracted": len(extracted.buildings),
            "considered_buildings": 4,
        },
        info_type="context",
        info_name="portfolio_context",
    )

    return evaluator.get_summary()