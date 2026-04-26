import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# =============================================================================
# Task-specific constants
# =============================================================================
TASK_ID = "ai_colocation_facilities_us"
TASK_DESCRIPTION = (
    "Identify three colocation data centers in the United States that are suitable for deploying AI training "
    "workloads. For each facility, provide the following information: (1) Facility Name/Identifier: The specific "
    "name or code of the data center facility, (2) Provider: The colocation provider operating the facility, "
    "(3) State Location: The US state where the facility is located, (4) Power Density: Confirmation that the "
    "facility supports at least 40 kW per rack, (5) Tier Certification: Confirmation of Tier III or Tier IV "
    "certification, (6) Cooling Technology: Confirmation of liquid cooling capability, (7) Energy Efficiency (PUE): "
    "The facility's Power Usage Effectiveness rating (must be 1.5 or lower), (8) AI-Ready Status: Evidence that the "
    "facility is explicitly marketed or certified as AI-ready or suitable for GPU/HPC workloads, (9) Reference URL: "
    "A supporting URL from the provider's official website or a reputable data center industry source. Requirements: "
    "All three facilities must be from established major colocation providers (such as Equinix, Digital Realty, "
    "CoreSite, CyrusOne, Flexential, or equivalent tier-1 providers). Each facility must meet ALL of the "
    "specifications listed above. Facilities must be in different US states. Each facility must have clear "
    "documentation or official statements supporting its AI-ready capabilities."
)

# A non-exhaustive, defensible list of major U.S. colocation providers (tier-1 / widely established)
MAJOR_PROVIDERS = {
    "equinix",
    "digital realty", "digital realty trust",
    "coresite",
    "cyrusone",
    "flexential",
    "qts", "qts data centers",
    "switch",
    "ntt", "ntt global data centers", "ragingwire",
    "iron mountain", "iron mountain data centers",
    "aligned", "aligned data centers",
    "stack infrastructure",
    "sabey", "sabey data centers",
    "compass datacenters", "compass data centers",
    "edgeconnex",
    "vantage", "vantage data centers",
    "cologix",
    "tierpoint",
}

US_STATE_ABBR = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
    "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
    "va", "wa", "wv", "wi", "wy", "dc",
}
US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware",
    "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
    "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming", "district of columbia",
}

# =============================================================================
# Data models for extraction
# =============================================================================
class FacilityItem(BaseModel):
    facility_name: Optional[str] = None
    provider: Optional[str] = None
    state: Optional[str] = None

    # Free-form fields captured from the answer (strings to maximize compatibility)
    power_density: Optional[str] = None          # e.g., "40 kW per rack", "50kW+ per rack"
    tier_certification: Optional[str] = None     # e.g., "Uptime Tier III"
    liquid_cooling: Optional[str] = None         # e.g., "Direct-to-Chip", "RDHx", "Immersion", or "Yes"
    pue: Optional[str] = None                    # e.g., "1.3", "as low as 1.25"
    ai_ready_status: Optional[str] = None        # e.g., "GPU-ready", "AI-ready", "HPC capable"

    # URLs - category-specific if provided; otherwise, the extractor may copy general references into these
    reference_urls: List[str] = Field(default_factory=list)
    url_power: List[str] = Field(default_factory=list)
    url_cooling: List[str] = Field(default_factory=list)
    url_certification: List[str] = Field(default_factory=list)
    url_ai_ready: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    facilities: List[FacilityItem] = Field(default_factory=list)


# =============================================================================
# Extraction prompt
# =============================================================================
def prompt_extract_facilities() -> str:
    return """
    Extract up to three U.S. colocation data center facilities that the answer claims are suitable for AI training workloads.
    For each facility, extract the following fields from the answer text exactly as stated:

    - facility_name: Specific facility name or identifier (e.g., "Equinix DA11", "CoreSite SV1", "Digital Realty IAD39").
    - provider: The colocation provider operating the facility (e.g., Equinix, Digital Realty, CoreSite, CyrusOne, Flexential, QTS, etc.).
    - state: The U.S. state where the facility is located (use the state name or 2-letter abbreviation if the answer provides it).
    - power_density: Any phrase that indicates the per-rack power density and explicitly shows at least 40 kW/rack (e.g., "40 kW per rack", "50 kW+").
    - tier_certification: Any phrase describing Tier certification (Uptime Institute Tier III or Tier IV).
    - liquid_cooling: Any phrase indicating liquid cooling capability (e.g., "direct-to-chip", "immersion", "rear door heat exchanger/RDHx", "liquid-ready").
    - pue: The facility's PUE value if provided (accept ranges or "as low as" language).
    - ai_ready_status: Any phrase indicating the facility is explicitly marketed as AI-ready, GPU-ready, or suitable for HPC/GPU workloads.

    URLs:
    - reference_urls: All URLs cited for this facility (provider official site or reputable industry sources).
    - url_power: URLs that specifically support the power density claim. If the answer only gives general references, copy the most relevant URL(s) here too.
    - url_cooling: URLs that specifically support the liquid cooling capability. If only general references are given, copy them here as well.
    - url_certification: URLs that specifically support Tier certification and/or PUE metrics. If only general references are given, copy them here as well.
    - url_ai_ready: URLs that specifically support the AI-/GPU-/HPC-ready designation. If only general references are given, copy them here as well.

    IMPORTANT RULES:
    - Do not invent URLs or data; extract only what is explicitly present in the answer. However, it is acceptable to copy the same cited URL(s) into multiple category-specific URL fields if the answer uses one page to support multiple claims.
    - If a field is missing in the answer for a facility, set it to null (or empty list for URL arrays).
    - Return a JSON object with a single key 'facilities' which is an array of up to three FacilityItem objects as defined.
    """


# =============================================================================
# Helper utilities
# =============================================================================
def _nonempty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _canon_provider(name: Optional[str]) -> str:
    if not _nonempty(name):
        return ""
    s = name.strip().lower()
    # Normalize common suffixes
    for token in [", inc.", " inc.", " llc", ", llc", " corp.", " corporation", " data centers", " data centre",
                  " holdings", " trust"]:
        if s.endswith(token):
            s = s.replace(token, "")
    s = s.replace("|", " ").replace("  ", " ").strip()
    return s


def is_major_provider(name: Optional[str]) -> bool:
    if not _nonempty(name):
        return False
    s = _canon_provider(name)
    return s in MAJOR_PROVIDERS


def is_us_state(state: Optional[str]) -> bool:
    if not _nonempty(state):
        return False
    s = state.strip().lower()
    if s in US_STATE_NAMES:
        return True
    if len(s) == 2 and s in US_STATE_ABBR:
        return True
    # Also allow "ca - california" like strings; take first token or try split by comma
    first_token = s.replace(".", "").split(",")[0].strip()
    if len(first_token) == 2 and first_token in US_STATE_ABBR:
        return True
    if first_token in US_STATE_NAMES:
        return True
    return False


def get_state_key(state: Optional[str]) -> Optional[str]:
    if not _nonempty(state):
        return None
    s = state.strip().lower()
    # Normalize to 2-letter if possible; else use lower-case full name
    if len(s) == 2 and s in US_STATE_ABBR:
        return s
    # try to map full names to an internal key (use the name directly)
    if s in US_STATE_NAMES:
        return s
    # try common patterns like "CA - California"
    parts = [p.strip() for p in s.split("-")]
    if parts:
        cand = parts[0].lower()
        if len(cand) == 2 and cand in US_STATE_ABBR:
            return cand
        if cand in US_STATE_NAMES:
            return cand
    return s  # fallback (still useful for non-exact equality)


def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not _nonempty(u):
            continue
        uu = u.strip()
        if uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out


def choose_sources(f: FacilityItem, category: str) -> List[str]:
    if category == "power":
        base = f.url_power
    elif category == "cooling":
        base = f.url_cooling
    elif category == "cert":
        base = f.url_certification
    elif category == "ai":
        base = f.url_ai_ready
    else:
        base = []
    fallback = f.reference_urls or []
    urls = _unique_urls([*(base or []), *(fallback or [])])
    return urls


# =============================================================================
# Verification builders
# =============================================================================
async def verify_facility(
    evaluator: Evaluator,
    parent_node,
    facility: FacilityItem,
    index_one_based: int,
) -> None:
    """
    Build the verification subtree for a single facility.
    """

    # ---------------------------------------------------------------------
    # Facility root (non-critical to allow partial scoring across facilities)
    # ---------------------------------------------------------------------
    fac_node = evaluator.add_parallel(
        id=f"facility_{index_one_based}",
        desc=(
            "First colocation facility meeting all AI-ready specifications"
            if index_one_based == 1 else
            ("Second colocation facility meeting all AI-ready specifications" if index_one_based == 2
             else "Third colocation facility meeting all AI-ready specifications")
        ),
        parent=parent_node,
        critical=False,
    )

    # ---------------------------------------------------------------------
    # Basic identification (critical)
    # ---------------------------------------------------------------------
    basic_node = evaluator.add_parallel(
        id=f"facility_{index_one_based}_basic_identification",
        desc="Basic facility identification and location information",
        parent=fac_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(facility.facility_name),
        id=f"facility_{index_one_based}_facility_name",
        desc="Specific facility name or identifier is provided",
        parent=basic_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=is_us_state(facility.state),
        id=f"facility_{index_one_based}_us_location",
        desc="Facility is located in the United States",
        parent=basic_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(facility.state),
        id=f"facility_{index_one_based}_state_specification",
        desc="US state where the facility is located is specified",
        parent=basic_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=is_major_provider(facility.provider),
        id=f"facility_{index_one_based}_provider_identity",
        desc="Facility is operated by an established major colocation provider",
        parent=basic_node,
        critical=True,
    )

    # ---------------------------------------------------------------------
    # Technical specifications (critical)
    # ---------------------------------------------------------------------
    tech_node = evaluator.add_parallel(
        id=f"facility_{index_one_based}_technical_specifications",
        desc="Technical infrastructure specifications for AI workloads",
        parent=fac_node,
        critical=True,
    )

    # Power infrastructure (critical)
    power_node = evaluator.add_parallel(
        id=f"facility_{index_one_based}_power_infrastructure",
        desc="Power density and capacity specifications",
        parent=tech_node,
        critical=True,
    )

    # Ensure we have at least one supporting URL for power
    power_urls = choose_sources(facility, "power")
    evaluator.add_custom_node(
        result=len(power_urls) > 0,
        id=f"facility_{index_one_based}_url_power",
        desc="URL reference supporting power density specifications",
        parent=power_node,
        critical=True,
    )

    # Rack power density verification (>= 40 kW per rack)
    rack_power_leaf = evaluator.add_leaf(
        id=f"facility_{index_one_based}_rack_power_density",
        desc="Facility supports minimum 40 kW per rack for AI workloads",
        parent=power_node,
        critical=True,
    )
    rp_name = facility.facility_name or "the facility"
    rp_claim = (
        f"{rp_name} supports at least 40 kW per rack (or higher) for AI/high-density deployments. "
        f"Accept equivalent phrasing like '>=40 kW per rack', '50 kW per rack', or '40kW+ per rack'."
    )
    await evaluator.verify(
        claim=rp_claim,
        node=rack_power_leaf,
        sources=power_urls,
        additional_instruction=(
            "Consider 'per rack' and 'per cabinet' as equivalent. Accept explicit mentions of 40kW or higher per-rack "
            "capacity, including values like 45kW, 50kW, 60kW, 100kW, etc. The page must clearly apply to this facility."
        ),
    )

    # Cooling infrastructure (critical)
    cooling_node = evaluator.add_parallel(
        id=f"facility_{index_one_based}_cooling_infrastructure",
        desc="Cooling technology specifications",
        parent=tech_node,
        critical=True,
    )

    cooling_urls = choose_sources(facility, "cooling")
    evaluator.add_custom_node(
        result=len(cooling_urls) > 0,
        id=f"facility_{index_one_based}_url_cooling",
        desc="URL reference supporting cooling technology specifications",
        parent=cooling_node,
        critical=True,
    )

    liquid_leaf = evaluator.add_leaf(
        id=f"facility_{index_one_based}_liquid_cooling",
        desc="Facility offers liquid cooling capability for high-density workloads",
        parent=cooling_node,
        critical=True,
    )
    lc_claim = (
        f"{rp_name} offers liquid cooling capability for high-density workloads, such as direct-to-chip (D2C), "
        f"immersion cooling, liquid-to-rack/CDU, or rear door heat exchangers (RDHx)."
    )
    await evaluator.verify(
        claim=lc_claim,
        node=liquid_leaf,
        sources=cooling_urls,
        additional_instruction=(
            "Accept synonyms: 'liquid cooling', 'direct-to-chip', 'immersion', 'RDHx', 'liquid-to-rack', 'CDU'. "
            "The page must make it clear that liquid cooling is supported for this facility (or this campus that includes this facility)."
        ),
    )

    # ---------------------------------------------------------------------
    # Certifications and efficiency (critical)
    # ---------------------------------------------------------------------
    cert_node = evaluator.add_parallel(
        id=f"facility_{index_one_based}_certifications_efficiency",
        desc="Facility certifications and efficiency metrics",
        parent=fac_node,
        critical=True,
    )

    cert_urls = choose_sources(facility, "cert")
    evaluator.add_custom_node(
        result=len(cert_urls) > 0,
        id=f"facility_{index_one_based}_url_certification",
        desc="URL reference supporting tier certification and PUE metrics",
        parent=cert_node,
        critical=True,
    )

    tier_leaf = evaluator.add_leaf(
        id=f"facility_{index_one_based}_tier_certification",
        desc="Facility holds Tier III or Tier IV certification",
        parent=cert_node,
        critical=True,
    )
    tier_claim = (
        f"{rp_name} is Uptime Institute Tier III or Tier IV (Design and/or Constructed Facility) certified."
    )
    await evaluator.verify(
        claim=tier_claim,
        node=tier_leaf,
        sources=cert_urls,
        additional_instruction=(
            "Look for explicit 'Uptime Institute Tier III' or 'Tier IV' mentions. Accept 'Tier III Design', "
            "'Tier III Constructed Facility', or similar official phrasing. If tier is not mentioned, the claim is unsupported."
        ),
    )

    pue_leaf = evaluator.add_leaf(
        id=f"facility_{index_one_based}_energy_efficiency",
        desc="Facility has documented PUE of 1.5 or lower",
        parent=cert_node,
        critical=True,
    )
    pue_str = facility.pue or "N/A"
    pue_claim = (
        f"{rp_name} has a documented PUE of 1.5 or lower. The answer cites PUE '{pue_str}' which must be <= 1.5 if provided."
    )
    await evaluator.verify(
        claim=pue_claim,
        node=pue_leaf,
        sources=cert_urls,
        additional_instruction=(
            "Verify the page shows a PUE value at or below 1.5. Accept 'as low as' or 'design PUE' if explicitly <= 1.5. "
            "If the PUE is not provided or clearly above 1.5, mark as not supported."
        ),
    )

    # ---------------------------------------------------------------------
    # AI readiness (critical)
    # ---------------------------------------------------------------------
    ai_node = evaluator.add_parallel(
        id=f"facility_{index_one_based}_ai_readiness",
        desc="AI-ready designation and supporting documentation",
        parent=fac_node,
        critical=True,
    )

    ai_urls = choose_sources(facility, "ai")
    evaluator.add_custom_node(
        result=len(ai_urls) > 0,
        id=f"facility_{index_one_based}_url_ai_ready",
        desc="URL reference supporting AI-ready designation",
        parent=ai_node,
        critical=True,
    )

    ai_leaf = evaluator.add_leaf(
        id=f"facility_{index_one_based}_ai_ready_status",
        desc="Facility is explicitly designated as AI-ready or suitable for GPU/HPC workloads",
        parent=ai_node,
        critical=True,
    )
    ai_claim = (
        f"{rp_name} is explicitly described as AI-ready, GPU-ready, or suitable for HPC/GPU workloads."
    )
    await evaluator.verify(
        claim=ai_claim,
        node=ai_leaf,
        sources=ai_urls,
        additional_instruction=(
            "Accept explicit phrasing like 'AI-ready', 'GPU-ready', 'HPC-ready', 'supports GPU clusters', "
            "'NVIDIA DGX-Ready Data Center', or clear marketing targeting AI/HPC/GPU workloads for this facility/campus."
        ),
    )


# =============================================================================
# Main evaluation function
# =============================================================================
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
    Entry point for evaluating an answer to the AI-ready U.S. colocation facilities task.
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

    # Extract structured facilities information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction",
    )

    # Select exactly three facilities (pad with empty FacilityItem if fewer provided)
    facilities: List[FacilityItem] = list(extracted.facilities[:3])
    while len(facilities) < 3:
        facilities.append(FacilityItem())

    # Build per-facility verification trees
    for i in range(3):
        await verify_facility(
            evaluator=evaluator,
            parent_node=root,
            facility=facilities[i],
            index_one_based=i + 1,
        )

    # Add a critical node to enforce that all three facilities are in different U.S. states
    state_unique_node = evaluator.add_parallel(
        id="state_uniqueness",
        desc="Verification that all three facilities are in different US states",
        parent=root,
        critical=True,
    )

    # Compute normalized state keys
    s1 = get_state_key(facilities[0].state)
    s2 = get_state_key(facilities[1].state)
    s3 = get_state_key(facilities[2].state)

    # Pairwise different-state checks (critical leaves)
    evaluator.add_custom_node(
        result=(s1 is not None and s2 is not None and s1 != s2 and is_us_state(facilities[0].state) and is_us_state(facilities[1].state)),
        id="facility_1_2_different_states",
        desc="Facility 1 and Facility 2 are located in different US states",
        parent=state_unique_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(s1 is not None and s3 is not None and s1 != s3 and is_us_state(facilities[0].state) and is_us_state(facilities[2].state)),
        id="facility_1_3_different_states",
        desc="Facility 1 and Facility 3 are located in different US states",
        parent=state_unique_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(s2 is not None and s3 is not None and s2 != s3 and is_us_state(facilities[1].state) and is_us_state(facilities[2].state)),
        id="facility_2_3_different_states",
        desc="Facility 2 and Facility 3 are located in different US states",
        parent=state_unique_node,
        critical=True,
    )

    # Optionally record helper info for transparency
    evaluator.add_custom_info(
        info={
            "major_providers_list_used": sorted(list(MAJOR_PROVIDERS)),
            "extracted_states": [fac.state for fac in facilities],
            "extracted_providers": [fac.provider for fac in facilities],
        },
        info_type="debug_info",
        info_name="normalization_rules_and_extracted_overview",
    )

    return evaluator.get_summary()