import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "three_hospitals_flu_pharmacy"
TASK_DESCRIPTION = """
Identify three hospitals that meet all of the following criteria:

1. Each hospital must be part of one of these three major U.S. health systems: HCA Healthcare, CommonSpirit Health, or Ascension Health
2. All three hospitals must be from different health systems (one from each of the three systems)
3. Each hospital must be located in a state that is currently reporting "high" or "very high" influenza activity according to the most recent CDC FluView weekly surveillance report
4. Each hospital must have either a CVS Pharmacy or Walgreens location within 5 miles

For each of the three hospitals, provide:
- The hospital's official name
- Complete physical address (street address, city, state, ZIP code)
- Confirmation of which major health system it belongs to
- The current flu activity level in that hospital's state according to CDC data
- Name and address of a CVS or Walgreens pharmacy within 5 miles of the hospital, along with the distance between the hospital and pharmacy
- Direct URL to the hospital's official website
"""

REQUIRED_SYSTEMS = {"HCA Healthcare", "CommonSpirit Health", "Ascension Health"}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PharmacyInfo(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None  # Expect "CVS" or "Walgreens"
    address: Optional[str] = None
    distance_miles: Optional[str] = None  # Keep as string to allow "4.8 mi", "approx. 5 miles"
    sources: List[str] = Field(default_factory=list)  # e.g., store page + Google Maps link(s)


class HospitalItem(BaseModel):
    name: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    full_address: Optional[str] = None

    health_system: Optional[str] = None
    health_system_sources: List[str] = Field(default_factory=list)

    flu_activity_level: Optional[str] = None  # e.g., "high", "very high"
    flu_sources: List[str] = Field(default_factory=list)  # CDC FluView URL(s) explicitly provided in the answer

    pharmacy: Optional[PharmacyInfo] = None

    website: Optional[str] = None
    name_address_sources: List[str] = Field(default_factory=list)  # URLs that show name/address (hospital site, location page, etc.)


class HospitalsExtraction(BaseModel):
    hospitals: List[HospitalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hospitals() -> str:
    return """
    Extract up to the first three hospitals described in the answer along with the specific verification sources the answer cites.

    For each hospital, extract the following fields (return as an array named 'hospitals'):
    - name: Official hospital name as written in the answer.
    - street: Street address line (if present).
    - city: City (if present).
    - state: State abbreviation or full name as shown (if present).
    - zip: ZIP code (if present).
    - full_address: The full address string if the answer provides it in one line; otherwise null.
    - health_system: The major health system the hospital belongs to (as claimed in the answer).
    - health_system_sources: URLs cited in the answer that directly support the hospital's membership in that health system. Include health system site pages or the hospital's page that states the affiliation.
    - flu_activity_level: The flu activity level claimed for the hospital's state (e.g., "high" or "very high").
    - flu_sources: URLs cited in the answer that directly support the CDC FluView weekly surveillance classification for that state (most recent). Only extract URLs explicitly provided in the answer.
    - pharmacy: An object describing a nearby retail pharmacy (CVS or Walgreens) within 5 miles of the hospital (as claimed in the answer):
        * name: Pharmacy name as written in the answer.
        * brand: Either "CVS" or "Walgreens" if clearly stated; otherwise null.
        * address: Pharmacy address string as shown in the answer.
        * distance_miles: The distance between hospital and pharmacy as stated (string; keep units if present).
        * sources: All URLs cited for the pharmacy proximity (e.g., pharmacy store page and/or Google Maps directions link).
    - website: Direct URL to the official hospital website or the hospital's page on the health system's website, as provided in the answer.
    - name_address_sources: URLs cited that show the hospital's official name and physical address (e.g., hospital location page, contact page). If the answer only provides the hospital website, include that here too.

    RULES:
    - Only include URLs that are explicitly present in the answer. Do not invent or infer URLs.
    - Normalize URL formats; if a URL is missing a protocol, prepend "http://".
    - If any field is missing in the answer, set it to null (or empty list for URL lists).
    - Return a JSON object with a single property 'hospitals' which is an array of up to 3 items. If more than 3 hospitals are mentioned, only include the first 3.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_health_system_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    # HCA Healthcare and common brand variants
    if "hca" in s:
        return "HCA Healthcare"
    if "commonspirit" in s or "dignity health" in s or "dignityhealth" in s or " chi " in f" {s} " or s.startswith("chi "):
        return "CommonSpirit Health"
    if "ascension" in s:
        return "Ascension Health"
    # Sometimes the health system name equals the expected label already
    if "hca healthcare" in s:
        return "HCA Healthcare"
    if "commonspirit health" in s:
        return "CommonSpirit Health"
    if "ascension health" in s or s == "ascension":
        return "Ascension Health"
    return None


def build_full_address(h: HospitalItem) -> Optional[str]:
    if h.full_address and h.full_address.strip():
        return h.full_address.strip()
    parts = [p for p in [h.street, h.city, h.state, h.zip] if p and str(p).strip()]
    if parts:
        # Join with comma spaces as appropriate; typical format "street, city, state zip"
        # If we have all 4 fields, format nicely
        street = h.street.strip() if h.street else None
        city = h.city.strip() if h.city else None
        state = h.state.strip() if h.state else None
        zipc = h.zip.strip() if h.zip else None
        if street and city and state and zipc:
            return f"{street}, {city}, {state} {zipc}"
        # Fallback concatenation
        return ", ".join(parts)
    return None


def unique_urls(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in lst or []:
            if not u:
                continue
            u = u.strip()
            if not u:
                continue
            # Basic validation: must contain a dot and no spaces
            if " " in u or "." not in u:
                continue
            if not (u.startswith("http://") or u.startswith("https://")):
                u = "http://" + u
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def brand_from_name_or_brand(pharmacy: Optional[PharmacyInfo]) -> Optional[str]:
    if not pharmacy:
        return None
    if pharmacy.brand and pharmacy.brand.strip():
        b = pharmacy.brand.strip().lower()
        if "cvs" in b:
            return "CVS"
        if "walgreens" in b:
            return "Walgreens"
    if pharmacy.name:
        n = pharmacy.name.strip().lower()
        if "cvs" in n:
            return "CVS"
        if "walgreens" in n:
            return "Walgreens"
    return None


def systems_for_first_three(hospitals: List[HospitalItem]) -> List[Optional[str]]:
    systems: List[Optional[str]] = []
    for h in hospitals[:3]:
        systems.append(normalize_health_system_name(h.health_system))
    return systems


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_hospital(
    evaluator: Evaluator,
    parent_node,
    hospital: HospitalItem,
    idx: int
) -> None:
    """
    Build verification leaves for a single hospital.
    """
    # Use IDs aligned with the rubric JSON to aid debugging.
    hosp_group_id = f"Hospital_{idx + 1}"
    hosp_desc = [
        "First hospital meeting all criteria",
        "Second hospital meeting all criteria from a different health system than Hospital 1",
        "Third hospital meeting all criteria from a different health system than Hospitals 1 and 2",
    ][idx] if idx < 3 else f"Hospital #{idx + 1}"

    hospital_node = evaluator.add_parallel(
        id=hosp_group_id,
        desc=hosp_desc,
        parent=parent_node,
        critical=False  # Allow partial credit per hospital
    )

    # 1) Name + Location leaf
    name_loc_leaf = evaluator.add_leaf(
        id=f"H{idx + 1}_Name_Location",
        desc="Hospital's official name and complete physical address including street, city, state, and ZIP code",
        parent=hospital_node,
        critical=True
    )

    # Build claim and sources
    name_ok = bool(hospital.name and hospital.name.strip())
    addr_str = build_full_address(hospital)
    addr_ok = bool(addr_str and addr_str.strip())
    sources_name_addr = unique_urls(hospital.name_address_sources, [hospital.website] if hospital.website else [])

    if not name_ok or not addr_ok or not sources_name_addr:
        # Missing essential info or sources => fail this critical leaf
        name_loc_leaf.score = 0.0
        name_loc_leaf.status = "failed"
    else:
        claim = (
            f"The hospital's official name is '{hospital.name}'. The complete physical address is '{addr_str}'. "
            f"The provided webpage(s) explicitly show both this official name and this full address for the hospital."
        )
        await evaluator.verify(
            claim=claim,
            node=name_loc_leaf,
            sources=sources_name_addr,
            additional_instruction=(
                "Confirm the page(s) explicitly state the hospital's official name and its full mailing or street address. "
                "Accept minor formatting differences (e.g., 'St.' vs 'Saint', 'Rd' vs 'Road', ZIP+4). The page should clearly "
                "correspond to the specified hospital."
            )
        )

    # 2) Health system membership leaf
    health_system_leaf = evaluator.add_leaf(
        id=f"H{idx + 1}_Health_System",
        desc="Hospital is part of one of the three specified major health systems: HCA Healthcare, CommonSpirit Health, or Ascension Health",
        parent=hospital_node,
        critical=True
    )

    normalized_sys = normalize_health_system_name(hospital.health_system)
    sys_sources = unique_urls(hospital.health_system_sources, [hospital.website] if hospital.website else [])

    if not normalized_sys or normalized_sys not in REQUIRED_SYSTEMS or not sys_sources or not name_ok:
        health_system_leaf.score = 0.0
        health_system_leaf.status = "failed"
    else:
        claim = f"The hospital '{hospital.name}' is part of {normalized_sys}."
        await evaluator.verify(
            claim=claim,
            node=health_system_leaf,
            sources=sys_sources,
            additional_instruction=(
                "Verify the page indicates the hospital's affiliation with the stated major system. "
                "Allow brand variants: e.g., 'HCA Houston Healthcare', 'HCA Florida Healthcare' -> HCA Healthcare; "
                "'Dignity Health' or 'CHI Health' -> CommonSpirit Health; any 'Ascension' branded hospitals -> Ascension Health."
            )
        )

    # 3) State flu activity leaf (CDC FluView)
    state_activity_leaf = evaluator.add_leaf(
        id=f"H{idx + 1}_State_Activity",
        desc="Hospital is located in a state with 'high' or 'very high' flu activity level according to the most recent CDC FluView weekly surveillance report",
        parent=hospital_node,
        critical=True
    )

    state_ok = bool(hospital.state and hospital.state.strip())
    flu_level = (hospital.flu_activity_level or "").strip().lower()
    flu_sources = unique_urls(hospital.flu_sources)

    if not state_ok or not flu_sources or flu_level not in {"high", "very high"}:
        state_activity_leaf.score = 0.0
        state_activity_leaf.status = "failed"
    else:
        claim = (
            f"According to the most recent CDC FluView weekly surveillance, the state {hospital.state} is classified as "
            f"'{hospital.flu_activity_level}'."
        )
        await evaluator.verify(
            claim=claim,
            node=state_activity_leaf,
            sources=flu_sources,
            additional_instruction=(
                "Focus on whether CDC FluView shows the state's influenza activity level as 'High' or 'Very High'. "
                "Case-insensitive matching is acceptable. If the page displays a color-coded map or a text table, either suffices."
            )
        )

    # 4) Pharmacy within 5 miles (CVS or Walgreens)
    pharmacy_leaf = evaluator.add_leaf(
        id=f"H{idx + 1}_Pharmacy_Access",
        desc="Identification of a retail pharmacy (CVS or Walgreens) within 5 miles of the hospital, including pharmacy name, address, and distance",
        parent=hospital_node,
        critical=True
    )

    ph = hospital.pharmacy
    brand = brand_from_name_or_brand(ph)
    pharmacy_sources = unique_urls(ph.sources if ph and ph.sources else [])

    if not (brand in {"CVS", "Walgreens"} and ph and ph.name and ph.address and pharmacy_sources and addr_ok):
        pharmacy_leaf.score = 0.0
        pharmacy_leaf.status = "failed"
    else:
        # Construct a conservative claim that can be checked via store page and/or Google Maps link
        claim = (
            f"There is a {brand} pharmacy named '{ph.name}' at '{ph.address}' within 5 miles of the hospital at '{build_full_address(hospital)}'."
        )
        await evaluator.verify(
            claim=claim,
            node=pharmacy_leaf,
            sources=pharmacy_sources,
            additional_instruction=(
                "Use the provided store page and/or Google Maps link to confirm that the pharmacy is a CVS or Walgreens and "
                "that the distance from the hospital to this pharmacy is no more than 5.0 miles. "
                "Allow minor rounding differences (e.g., 5.0 vs 4.9)."
            )
        )

    # 5) Official website leaf
    website_leaf = evaluator.add_leaf(
        id=f"H{idx + 1}_Website",
        desc="Direct URL to the hospital's official website or its page on the health system's website",
        parent=hospital_node,
        critical=True
    )

    if not hospital.website or not hospital.website.strip():
        website_leaf.score = 0.0
        website_leaf.status = "failed"
    else:
        claim = (
            f"The provided URL is the official website for '{hospital.name}' or the hospital's official page on the "
            f"{normalized_sys or (hospital.health_system or 'health system')} website."
        )
        await evaluator.verify(
            claim=claim,
            node=website_leaf,
            sources=hospital.website,
            additional_instruction=(
                "Confirm the page appears official (hospital or health system domain) and specifically represents this hospital, "
                "showing its proper name and/or location information."
            )
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 'three hospitals from different systems with CDC high/very high flu activity and nearby CVS/Walgreens' task.
    """
    # Initialize evaluator (root is non-critical to avoid critical-child constraint; we'll add a critical cross-system node)
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hospitals(),
        template_class=HospitalsExtraction,
        extraction_name="hospitals_extraction"
    )

    # Build the working list of exactly 3 hospitals (pad with empty items if fewer)
    hospitals = (extracted.hospitals or [])[:3]
    while len(hospitals) < 3:
        hospitals.append(HospitalItem())

    # Record the systems that were extracted (normalized) for cross-checking
    normalized_systems = systems_for_first_three(hospitals)
    evaluator.add_ground_truth(
        {
            "required_systems": sorted(list(REQUIRED_SYSTEMS)),
            "extracted_systems_normalized": normalized_systems
        },
        gt_type="required_health_systems"
    )

    # Cross-system constraints node (critical)
    cross_node = evaluator.add_parallel(
        id="Cross_System_Constraints",
        desc="All three hospitals must be from different health systems and cover one from each of the three specified systems",
        parent=root,
        critical=True
    )

    # Leaf 1: Distinct systems among the three hospitals (custom logic)
    systems_present = [s for s in normalized_systems if s is not None]
    distinct_ok = (len(systems_present) == 3) and (len(set(systems_present)) == 3)
    evaluator.add_custom_node(
        result=distinct_ok,
        id="Distinct_Health_Systems",
        desc="The three hospitals are from three different health systems (no duplicates)",
        parent=cross_node,
        critical=True
    )

    # Leaf 2: Coverage includes exactly the three required systems (custom logic)
    coverage_ok = set(systems_present) == REQUIRED_SYSTEMS
    evaluator.add_custom_node(
        result=coverage_ok,
        id="Coverage_All_Three_Specified_Systems",
        desc="The three hospitals collectively cover one from each of HCA Healthcare, CommonSpirit Health, and Ascension Health",
        parent=cross_node,
        critical=True
    )

    # Build verification subtrees for each hospital
    for i in range(3):
        await verify_hospital(evaluator, root, hospitals[i], i)

    # Return structured summary
    return evaluator.get_summary()