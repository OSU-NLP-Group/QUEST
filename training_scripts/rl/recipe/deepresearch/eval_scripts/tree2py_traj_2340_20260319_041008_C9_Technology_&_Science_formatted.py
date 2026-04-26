import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "hyperscale_ai_dc_2026"
TASK_DESCRIPTION = (
    "Identify three operational hyperscale data centers in the United States that are specifically equipped "
    "to support artificial intelligence workloads and meet the following comprehensive requirements:\n\n"
    "Location Requirements:\n"
    "- Each facility must be located in the United States\n"
    "- The three facilities must be distributed across at least 3 different US states\n\n"
    "Technical Specifications:\n"
    "- Minimum power capacity of 50 megawatts (MW) per facility\n"
    "- Power Usage Effectiveness (PUE) of 1.3 or lower\n"
    "- Confirmed operational status as of Q1 2026 (not merely planned or under construction)\n\n"
    "Certification and Standards:\n"
    "- Each facility must hold at least one recognized green building certification: LEED (any level), "
    "Energy Star (score ≥ 75), or ISO 14001\n"
    "- Each facility must meet Tier III or higher standards (Uptime Institute or TIA-942)\n\n"
    "Infrastructure Capabilities:\n"
    "- Deployment of AI-specific chips/accelerators (e.g., NVIDIA GPUs, AMD/Intel AI processors, TPUs, "
    "AWS Trainium/Inferentia, etc.)\n"
    "- Advanced cooling (e.g., liquid cooling, free cooling, evaporative cooling)\n"
    "- Network infrastructure with N+1 or higher redundancy\n"
    "- ≥50% of power from renewable sources or on-site renewable generation\n\n"
    "For each facility, provide:\n"
    "1) Facility name and location (state and city/region)\n"
    "2) Verification of each technical specification\n"
    "3) Evidence of certification/standards compliance\n"
    "4) Confirmation of infrastructure capabilities\n"
    "5) Reference URLs from official/credible sources for each major category"
)

# -----------------------------------------------------------------------------
# US states helpers
# -----------------------------------------------------------------------------
US_STATES_FULL = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware",
    "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
    "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming", "district of columbia"
}
US_ABBR_TO_STATE = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia"
}
US_STATE_CANONICAL_MAP = {s.lower(): s for s in US_STATES_FULL}
US_ABBR_LOWER_TO_STATE = {k.lower(): v for k, v in US_ABBR_TO_STATE.items()}


def canonicalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip()
    if not s:
        return None
    # Try abbreviation
    abbr = s.upper()
    if abbr in US_ABBR_TO_STATE:
        return US_ABBR_TO_STATE[abbr]
    # Try full name
    lower = s.lower()
    if lower in US_STATE_CANONICAL_MAP:
        return US_STATE_CANONICAL_MAP[lower]
    # Handle common variations (e.g., "N. Carolina", "Wash.")
    normalized = re.sub(r"[^\w\s]", "", lower)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in US_STATE_CANONICAL_MAP:
        return US_STATE_CANONICAL_MAP[normalized]
    return None


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class GreenCertification(BaseModel):
    kind: Optional[str] = None  # e.g., "LEED", "Energy Star", "ISO 14001"
    detail: Optional[str] = None  # e.g., "LEED Gold v4", "Energy Star score 82", "ISO 14001:2015"
    score_or_level: Optional[str] = None  # e.g., "Gold", "82", "2015"


class TierStandard(BaseModel):
    framework: Optional[str] = None  # e.g., "Uptime Institute", "TIA-942"
    level: Optional[str] = None      # e.g., "Tier III", "Tier IV"


class Facility(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    city_region: Optional[str] = None

    location_urls: List[str] = Field(default_factory=list)

    # Technical specs
    power_capacity: Optional[str] = None  # e.g., "100 MW", "90+ MW"
    pue: Optional[str] = None             # e.g., "1.2", "~1.3"
    operational_status: Optional[str] = None  # e.g., "Operational as of 2026-02", "Live", etc.
    tech_urls: List[str] = Field(default_factory=list)

    # Certifications and standards
    green_certs: List[GreenCertification] = Field(default_factory=list)
    tier_standard: Optional[TierStandard] = None
    cert_urls: List[str] = Field(default_factory=list)

    # Infrastructure
    ai_chips: List[str] = Field(default_factory=list)   # e.g., ["NVIDIA H100", "TPU v5e"]
    cooling: List[str] = Field(default_factory=list)    # e.g., ["liquid cooling", "free cooling"]
    network_redundancy: Optional[str] = None            # e.g., "N+1", "2N"
    renewable_energy: Optional[str] = None              # e.g., "60% renewables", "on-site solar covers 50%"
    infra_urls: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    facilities: List[Facility] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_facilities() -> str:
    return """
    Extract up to three specific, operational hyperscale data center facilities in the United States that the answer claims are AI-enabled and meet the task requirements.

    For each facility mentioned in the answer (take the first three if there are more), extract the following fields exactly as written in the answer (do NOT invent information):
    - name: The facility name (e.g., campus or site name). If not explicitly provided, return null.
    - state: The U.S. state for the facility (full name or 2-letter abbreviation if used in the answer). If not provided, return null.
    - city_region: The city or region (if provided). If not provided, return null.

    - location_urls: All URLs in the answer that support the location/identification of this facility. If none are provided, return an empty list.

    Technical specifications (as text from the answer, do not normalize or infer):
    - power_capacity: The stated site/facility power capacity (e.g., "100 MW", "≥ 80 MW", "200MW planned but 80MW live"). If not provided, return null.
    - pue: The stated PUE (Power Usage Effectiveness) (e.g., "1.2", "<=1.3"). If not provided, return null.
    - operational_status: A phrase indicating the facility is operational (e.g., "operational", "in production", "live", "commissioned in 2025"); include any date/quarter if present (e.g., "operational as of Q1 2026"). If not provided, return null.
    - tech_urls: All URLs that support any of the above technical specifications.

    Certification and standards:
    - green_certs: A list where each item has:
        * kind: "LEED", "Energy Star", or "ISO 14001" (or the closest string appearing in the answer)
        * detail: The descriptive text as shown in the answer (e.g., "LEED Gold v4", "Energy Star score 82", "ISO 14001:2015")
        * score_or_level: The score/level/year noted (if any), otherwise null
    - tier_standard: An object with:
        * framework: "Uptime Institute" or "TIA-942" (or as in the answer)
        * level: e.g., "Tier III", "Tier IV" (as in the answer)
      If the answer does not provide tier information, return null.
    - cert_urls: All URLs supporting certifications or tier standards.

    Infrastructure capabilities:
    - ai_chips: List of the AI-specific chips/accelerators the answer mentions for this facility (e.g., "NVIDIA H100", "Google TPU", "AWS Trainium"). If not provided, return an empty list.
    - cooling: List of advanced cooling technologies mentioned (e.g., "liquid cooling", "immersion cooling", "free cooling", "evaporative cooling"). If not provided, return an empty list.
    - network_redundancy: The redundancy description if mentioned (e.g., "N+1", "2N", "N+2"). If not provided, return null.
    - renewable_energy: Any statement about renewable energy share (e.g., "60% renewable", "on-site solar covers 50%"). If not provided, return null.
    - infra_urls: All URLs supporting any of the infrastructure capabilities.

    Return a JSON object with:
    { "facilities": [ Facility, Facility, Facility ] }

    Important:
    - Only include information explicitly present in the answer text.
    - For each URL field, include only actual URLs explicitly shown in the answer (plain or markdown links). Do not invent URLs.
    - If fewer than three facilities are present, return only those available.
    """


# -----------------------------------------------------------------------------
# Helper parsing utilities
# -----------------------------------------------------------------------------
def parse_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(\d+)", text.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def any_url_present(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls or [])


def join_items(items: List[str], limit: int = 5) -> str:
    items_clean = [x.strip() for x in items if x and x.strip()]
    if not items_clean:
        return ""
    if len(items_clean) <= limit:
        return ", ".join(items_clean)
    return ", ".join(items_clean[:limit]) + f", and {len(items_clean) - limit} more"


# -----------------------------------------------------------------------------
# Verification subroutines for a single facility
# -----------------------------------------------------------------------------
async def verify_facility(
    evaluator: Evaluator,
    parent_node,
    facility: Facility,
    index: int,
) -> None:
    """
    Build verification tree and run verifications for one facility.
    index: 1-based index (1, 2, 3)
    """
    # Facility node (non-critical; overall passing depends on its critical children)
    fac_node = evaluator.add_parallel(
        id=f"facility_{index}",
        desc=f"{['First','Second','Third'][index-1] if 1 <= index <= 3 else f'Facility {index}'} qualifying hyperscale AI data center",
        parent=parent_node,
        critical=False,
    )

    # ---------------- Identification & Location ----------------
    ident_node = evaluator.add_parallel(
        id=f"f{index}_identification",
        desc=f"Facility {index} identification and location",
        parent=fac_node,
        critical=True  # critical category
    )

    # Facility name provided (existence)
    evaluator.add_custom_node(
        result=bool(facility and facility.name and facility.name.strip()),
        id=f"f{index}_facility_name",
        desc=f"Facility {index} name is provided",
        parent=ident_node,
        critical=True
    )

    # US state provided AND is a valid US state (existence + validation)
    state_canon = canonicalize_state(facility.state) if facility else None
    evaluator.add_custom_node(
        result=bool(state_canon),
        id=f"f{index}_us_state",
        desc="Located in a US state",
        parent=ident_node,
        critical=True
    )

    # Optional city/region (non-critical) — to satisfy critical-parent rule, attach under facility node
    evaluator.add_custom_node(
        result=bool(facility and facility.city_region and facility.city_region.strip()),
        id=f"f{index}_city_region",
        desc="Specific city or region identified",
        parent=fac_node,  # attach here to allow non-critical child
        critical=False
    )

    # Identification verification group
    ident_ver_node = evaluator.add_parallel(
        id=f"f{index}_identification_verification",
        desc="Identification and location verification",
        parent=ident_node,
        critical=True
    )

    # Location claim supported by URLs
    loc_leaf = evaluator.add_leaf(
        id=f"f{index}_location_url",
        desc="Location information supported by reference URL from official or credible source",
        parent=ident_ver_node,
        critical=True
    )

    # Build a strong claim for location
    loc_parts = []
    if facility and facility.name:
        loc_parts.append(f"the data center facility named '{facility.name}'")
    else:
        loc_parts.append("the data center facility")

    if state_canon:
        if facility and facility.city_region and facility.city_region.strip():
            loc_parts.append(f"is located in {facility.city_region.strip()}, {state_canon}, United States")
        else:
            loc_parts.append(f"is located in {state_canon}, United States")
    else:
        loc_parts.append("is located in the United States")

    location_claim = " ".join(loc_parts) + "."
    loc_additional_instruction = (
        "Use the provided URLs to confirm the facility name-to-location association. "
        "If no URLs are provided or the pages do not clearly support this location, judge as NOT supported."
    )
    await evaluator.verify(
        claim=location_claim,
        node=loc_leaf,
        sources=facility.location_urls if facility else None,
        additional_instruction=loc_additional_instruction
    )

    # ---------------- Technical Specifications ----------------
    tech_node = evaluator.add_parallel(
        id=f"f{index}_technical_specs",
        desc=f"Facility {index} technical specifications",
        parent=fac_node,
        critical=True
    )

    # Power capacity >= 50 MW
    power_leaf = evaluator.add_leaf(
        id=f"f{index}_power_capacity",
        desc="Minimum 50 MW power capacity",
        parent=tech_node,
        critical=True
    )
    power_claim = "The facility has a power capacity of at least 50 MW."
    await evaluator.verify(
        claim=power_claim,
        node=power_leaf,
        sources=facility.tech_urls if facility else None,
        additional_instruction=(
            "Confirm the stated or implied power capacity is ≥ 50 MW. "
            "If exact MW isn't stated but an equivalent capacity is, it still counts. "
            "If no URL evidence, judge as NOT supported."
        )
    )

    # PUE <= 1.3
    pue_leaf = evaluator.add_leaf(
        id=f"f{index}_pue_rating",
        desc="PUE of 1.3 or lower",
        parent=tech_node,
        critical=True
    )
    pue_claim = "The facility's Power Usage Effectiveness (PUE) is 1.3 or lower."
    await evaluator.verify(
        claim=pue_claim,
        node=pue_leaf,
        sources=facility.tech_urls if facility else None,
        additional_instruction=(
            "Look for PUE explicitly stated as ≤ 1.3 (allow forms like 1.2, ~1.3, <=1.3). "
            "If PUE is >1.3 or not stated, judge as NOT supported. "
            "If no URL evidence, judge as NOT supported."
        )
    )

    # Operational as of Q1 2026
    oper_leaf = evaluator.add_leaf(
        id=f"f{index}_operational_status",
        desc="Operational as of Q1 2026",
        parent=tech_node,
        critical=True
    )
    oper_claim = (
        "The facility is operational (in production, live, commissioned, or serving customers) "
        "as of March 31, 2026 (Q1 2026), and is not merely planned or under construction."
    )
    await evaluator.verify(
        claim=oper_claim,
        node=oper_leaf,
        sources=facility.tech_urls if facility else None,
        additional_instruction=(
            "Accept synonyms of operational (e.g., 'live', 'in production', 'commissioned'). "
            "If the facility is only planned/under construction or operational status is not confirmed by the page, "
            "judge as NOT supported. If no URL evidence, judge as NOT supported."
        )
    )

    # Technical URLs existence (as required support)
    tech_ver_node = evaluator.add_parallel(
        id=f"f{index}_tech_verification",
        desc="Technical specifications verification",
        parent=tech_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=any_url_present(facility.tech_urls if facility else []),
        id=f"f{index}_tech_url",
        desc="Technical specifications supported by reference URL from official or credible source",
        parent=tech_ver_node,
        critical=True
    )

    # ---------------- Certifications & Standards ----------------
    cert_node = evaluator.add_parallel(
        id=f"f{index}_certifications",
        desc=f"Facility {index} certification and standards compliance",
        parent=fac_node,
        critical=True
    )

    # Green certification: LEED(any) OR Energy Star (>=75) OR ISO 14001
    green_leaf = evaluator.add_leaf(
        id=f"f{index}_green_cert",
        desc="LEED, Energy Star (75+), or ISO 14001 certification",
        parent=cert_node,
        critical=True
    )

    # Build the best-available green cert claim using extracted info
    green_claim, green_urls = build_green_cert_claim_and_sources(facility)
    await evaluator.verify(
        claim=green_claim,
        node=green_leaf,
        sources=green_urls,
        additional_instruction=(
            "Accept any LEED certification (any level), or ENERGY STAR with score ≥ 75, or ISO 14001 certification. "
            "If none are clearly supported by the provided URLs, judge as NOT supported. "
            "If no URL evidence, judge as NOT supported."
        )
    )

    # Tier III or higher (Uptime or TIA-942)
    tier_leaf = evaluator.add_leaf(
        id=f"f{index}_tier_standard",
        desc="Tier III or higher standard (Uptime Institute or TIA-942)",
        parent=cert_node,
        critical=True
    )
    tier_claim, tier_urls = build_tier_claim_and_sources(facility)
    await evaluator.verify(
        claim=tier_claim,
        node=tier_leaf,
        sources=tier_urls,
        additional_instruction=(
            "Confirm that the facility meets Tier III or higher as defined by the Uptime Institute or TIA-942. "
            "If the page does not clearly indicate Tier III or above, judge as NOT supported. "
            "If no URL evidence, judge as NOT supported."
        )
    )

    cert_ver_node = evaluator.add_parallel(
        id=f"f{index}_cert_verification",
        desc="Certification information verification",
        parent=cert_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=any_url_present(facility.cert_urls if facility else []),
        id=f"f{index}_cert_url",
        desc="Certification information supported by reference URL from official or credible source",
        parent=cert_ver_node,
        critical=True
    )

    # ---------------- Infrastructure Capabilities ----------------
    infra_node = evaluator.add_parallel(
        id=f"f{index}_infrastructure",
        desc=f"Facility {index} infrastructure capabilities",
        parent=fac_node,
        critical=True
    )

    # AI chips/accelerators
    ai_leaf = evaluator.add_leaf(
        id=f"f{index}_ai_chips",
        desc="Deploys AI-specific chips or accelerators",
        parent=infra_node,
        critical=True
    )
    ai_claim = build_ai_chip_claim(facility)
    await evaluator.verify(
        claim=ai_claim,
        node=ai_leaf,
        sources=facility.infra_urls if facility else None,
        additional_instruction=(
            "Look for deployment of AI accelerators such as NVIDIA/AMD/Intel AI GPUs, Google TPUs, "
            "AWS Trainium/Inferentia, or equivalent. "
            "If unclear or not stated, judge as NOT supported. If no URL evidence, judge as NOT supported."
        )
    )

    # Advanced cooling
    cool_leaf = evaluator.add_leaf(
        id=f"f{index}_cooling",
        desc="Advanced cooling technologies employed",
        parent=infra_node,
        critical=True
    )
    cool_claim = build_cooling_claim(facility)
    await evaluator.verify(
        claim=cool_claim,
        node=cool_leaf,
        sources=facility.infra_urls if facility else None,
        additional_instruction=(
            "Accept advanced cooling approaches like liquid cooling, immersion, free cooling, or evaporative systems. "
            "If unclear or not stated, judge as NOT supported. If no URL evidence, judge as NOT supported."
        )
    )

    # Network redundancy N+1 or higher
    net_leaf = evaluator.add_leaf(
        id=f"f{index}_network_redundancy",
        desc="N+1 or higher network redundancy",
        parent=infra_node,
        critical=True
    )
    net_claim = "The facility's critical systems (including networking) implement N+1 or higher redundancy (e.g., N+1, N+2, 2N)."
    await evaluator.verify(
        claim=net_claim,
        node=net_leaf,
        sources=facility.infra_urls if facility else None,
        additional_instruction=(
            "Look for redundancy descriptions N+1, N+2, or 2N for critical systems (especially networking). "
            "If unclear or not stated, judge as NOT supported. If no URL evidence, judge as NOT supported."
        )
    )

    # Renewable energy ≥ 50%
    ren_leaf = evaluator.add_leaf(
        id=f"f{index}_renewable_energy",
        desc="At least 50% renewable energy sourcing",
        parent=infra_node,
        critical=True
    )
    ren_claim = "At least 50% of the facility's power is sourced from renewable energy or on-site renewable generation."
    await evaluator.verify(
        claim=ren_claim,
        node=ren_leaf,
        sources=facility.infra_urls if facility else None,
        additional_instruction=(
            "Check for explicit statements that ≥50% of energy is renewable (including on-site generation or PPAs). "
            "If renewable share is <50% or not stated, judge as NOT supported. If no URL evidence, judge as NOT supported."
        )
    )

    infra_ver_node = evaluator.add_parallel(
        id=f"f{index}_infra_verification",
        desc="Infrastructure capabilities verification",
        parent=infra_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=any_url_present(facility.infra_urls if facility else []),
        id=f"f{index}_infra_url",
        desc="Infrastructure details supported by reference URL from official or credible source",
        parent=infra_ver_node,
        critical=True
    )


# -----------------------------------------------------------------------------
# Claim builders
# -----------------------------------------------------------------------------
def build_green_cert_claim_and_sources(fac: Optional[Facility]) -> Tuple[str, List[str]]:
    urls = (fac.cert_urls if fac else []) or []
    default_claim = (
        "The facility holds at least one recognized green certification: "
        "LEED (any level), or ENERGY STAR with a score of 75 or higher, or ISO 14001."
    )
    if not fac or not fac.green_certs:
        return default_claim, urls

    # Prefer ENERGY STAR >= 75, else LEED(any), else ISO 14001, else default
    # ENERGY STAR path
    for gc in fac.green_certs:
        kind = (gc.kind or "").strip().lower()
        if "energy" in kind and "star" in kind:
            # Try to parse score
            score_text = gc.score_or_level or gc.detail or ""
            score = parse_first_int(score_text)
            if score is not None and score >= 75:
                return f"The facility holds an ENERGY STAR certification with a score of at least 75 (e.g., {score}).", urls
            # If Energy Star is present but score not stated, keep generic Energy Star 75+ claim
            return "The facility holds an ENERGY STAR certification with a score of at least 75.", urls

    # LEED path
    for gc in fac.green_certs:
        kind = (gc.kind or "").strip().lower()
        if "leed" in kind:
            level = (gc.score_or_level or gc.detail or "").strip()
            if level:
                return f"The facility holds a LEED certification ({level}).", urls
            return "The facility holds a LEED certification.", urls

    # ISO 14001 path
    for gc in fac.green_certs:
        kind = (gc.kind or "").strip().lower()
        if "iso" in kind and "14001" in kind:
            return "The facility is certified to ISO 14001.", urls

    return default_claim, urls


def build_tier_claim_and_sources(fac: Optional[Facility]) -> Tuple[str, List[str]]:
    urls = (fac.cert_urls if fac else []) or []
    if not fac or not fac.tier_standard:
        return (
            "The facility meets Tier III or higher data center standards as defined by the Uptime Institute or TIA-942.",
            urls
        )
    framework = fac.tier_standard.framework or "Uptime Institute or TIA-942"
    level = fac.tier_standard.level or "Tier III or higher"
    return f"The facility meets {level} data center standards as defined by {framework}.", urls


def build_ai_chip_claim(fac: Optional[Facility]) -> str:
    if fac and fac.ai_chips:
        listed = join_items(fac.ai_chips)
        return f"The facility deploys AI-specific chips or accelerators including {listed}."
    return "The facility deploys AI-specific chips or accelerators (e.g., NVIDIA/AMD/Intel AI GPUs, Google TPUs, AWS Trainium/Inferentia, or equivalent)."


def build_cooling_claim(fac: Optional[Facility]) -> str:
    if fac and fac.cooling:
        listed = join_items(fac.cooling)
        return f"The facility employs advanced cooling technologies such as {listed}."
    return "The facility employs advanced cooling technologies such as liquid cooling, immersion cooling, free cooling, or evaporative cooling."


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the hyperscale AI-enabled data centers task.
    """
    evaluator = Evaluator()
    # IMPORTANT: set root non-critical to allow mixed criticality children under it
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

    # Extract facilities (up to 3)
    extracted = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction",
    )

    facilities: List[Facility] = list(extracted.facilities or [])
    # Keep only first 3; pad with empties if fewer
    facilities = facilities[:3]
    while len(facilities) < 3:
        facilities.append(Facility())

    # Verify each facility subtree
    for i in range(3):
        await verify_facility(evaluator, root, facilities[i], i + 1)

    # Geographic diversity: at least 3 different states across the facilities
    states_norm = []
    for fac in facilities:
        states_norm.append(canonicalize_state(fac.state) if fac else None)
    unique_states = {s for s in states_norm if s}

    geo_result = len(unique_states) >= 3
    evaluator.add_custom_node(
        result=geo_result,
        id="geographic_diversity",
        desc="The three facilities are distributed across at least 3 different US states",
        parent=root,
        critical=True
    )

    # Record some custom info
    evaluator.add_custom_info(
        {
            "normalized_states": list(unique_states),
            "all_states_extracted": states_norm,
        },
        info_type="debug",
        info_name="state_aggregation"
    )

    return evaluator.get_summary()