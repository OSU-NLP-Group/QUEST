import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sma_hospitals_us_mda"
TASK_DESCRIPTION = (
    "For families seeking comprehensive spinal muscular atrophy (SMA) treatment options across the United States, "
    "identify 4 major pediatric hospitals, each located in a different US state, that meet the following criteria: "
    "(1) The hospital must be part of the Muscular Dystrophy Association (MDA) Care Center Network, "
    "(2) The hospital must offer at least 2 of the 4 FDA-approved SMA treatments: Zolgensma (onasemnogene abeparvovec), "
    "Spinraza (nusinersen), Evrysdi (risdiplam), or Itvisma (onasemnogene abeparvovec-brve), "
    "(3) For each treatment offered, specify the FDA-approved administration route (intravenous, intrathecal, or oral), and "
    "(4) Provide URL references that verify both the hospital's MDA Care Center status and the availability of each specified treatment."
)

# --------------------------------------------------------------------------- #
# FDA-approved treatment specs and utilities                                  #
# --------------------------------------------------------------------------- #

TREATMENT_SPECS: Dict[str, Dict[str, Any]] = {
    "Zolgensma": {
        "synonyms": ["zolgensma", "onasemnogene abeparvovec"],
        "route": "intravenous"
    },
    "Spinraza": {
        "synonyms": ["spinraza", "nusinersen"],
        "route": "intrathecal"
    },
    "Evrysdi": {
        "synonyms": ["evrysdi", "risdiplam"],
        "route": "oral"
    },
    "Itvisma": {
        "synonyms": ["itvisma", "onasemnogene abeparvovec-brve"],
        "route": "intrathecal"
    },
}

ROUTE_SYNONYMS: Dict[str, str] = {
    "iv": "intravenous",
    "iv infusion": "intravenous",
    "intravenous": "intravenous",
    "intravenous infusion": "intravenous",
    "infusion": "intravenous",  # Often implies IV in drug context

    "intrathecal": "intrathecal",
    "intrathecal injection": "intrathecal",
    "spinal injection": "intrathecal",
    "lumbar puncture": "intrathecal",

    "oral": "oral",
    "by mouth": "oral",
    "oral solution": "oral",
    "oral liquid": "oral",
    "oral suspension": "oral",
}

# US States normalization
US_STATE_MAP: Dict[str, str] = {
    # Abbreviations
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    # Full names
    "alabama": "Alabama", "alaska": "Alaska", "arizona": "Arizona", "arkansas": "Arkansas", "california": "California",
    "colorado": "Colorado", "connecticut": "Connecticut", "delaware": "Delaware", "florida": "Florida",
    "georgia": "Georgia", "hawaii": "Hawaii", "idaho": "Idaho", "illinois": "Illinois", "indiana": "Indiana",
    "iowa": "Iowa", "kansas": "Kansas", "kentucky": "Kentucky", "louisiana": "Louisiana", "maine": "Maine",
    "maryland": "Maryland", "massachusetts": "Massachusetts", "michigan": "Michigan", "minnesota": "Minnesota",
    "mississippi": "Mississippi", "missouri": "Missouri", "montana": "Montana", "nebraska": "Nebraska",
    "nevada": "Nevada", "new hampshire": "New Hampshire", "new jersey": "New Jersey", "new mexico": "New Mexico",
    "new york": "New York", "north carolina": "North Carolina", "north dakota": "North Dakota", "ohio": "Ohio",
    "oklahoma": "Oklahoma", "oregon": "Oregon", "pennsylvania": "Pennsylvania", "rhode island": "Rhode Island",
    "south carolina": "South Carolina", "south dakota": "South Dakota", "tennessee": "Tennessee", "texas": "Texas",
    "utah": "Utah", "vermont": "Vermont", "virginia": "Virginia", "washington": "Washington",
    "west virginia": "West Virginia", "wisconsin": "Wisconsin", "wyoming": "Wyoming",
    "district of columbia": "District of Columbia",
}

def _norm_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def canonical_treatment(name: Optional[str]) -> Optional[str]:
    nm = _norm_text(name)
    if not nm:
        return None
    for canonical, spec in TREATMENT_SPECS.items():
        for syn in spec["synonyms"]:
            if nm == syn:
                return canonical
        # allow minor punctuation differences and case-insensitive containment
        for syn in spec["synonyms"]:
            if nm.replace("-", " ").replace("/", " ").strip() == syn.replace("-", " ").replace("/", " ").strip():
                return canonical
    # fuzzy match: exact canonical names case-insensitive
    for canonical in TREATMENT_SPECS.keys():
        if nm == canonical.lower():
            return canonical
    return None

def normalize_route(route_str: Optional[str]) -> Optional[str]:
    r = _norm_text(route_str)
    if not r:
        return None
    # direct
    if r in ROUTE_SYNONYMS:
        return ROUTE_SYNONYMS[r]
    # try to simplify
    simplified = r.replace("-", " ").replace("/", " ").replace("  ", " ").strip()
    if simplified in ROUTE_SYNONYMS:
        return ROUTE_SYNONYMS[simplified]
    # contain keywords
    if "intrathecal" in simplified:
        return "intrathecal"
    if "iv" in simplified or "intravenous" in simplified or "infusion" in simplified:
        return "intravenous"
    if "oral" in simplified or "by mouth" in simplified:
        return "oral"
    return None

def expected_route_for(canonical: Optional[str]) -> Optional[str]:
    if canonical and canonical in TREATMENT_SPECS:
        return TREATMENT_SPECS[canonical]["route"]
    return None

def normalize_state(state_str: Optional[str]) -> Optional[str]:
    s = _norm_text(state_str)
    if not s:
        return None
    # Try abbreviation uppercase
    if s.upper() in US_STATE_MAP:
        return US_STATE_MAP[s.upper()]
    if s in US_STATE_MAP:
        return US_STATE_MAP[s]
    # common formats like "CA (California)" -> extract part before space or bracket
    s2 = s.split("(")[0].strip()
    if s2.upper() in US_STATE_MAP:
        return US_STATE_MAP[s2.upper()]
    if s2 in US_STATE_MAP:
        return US_STATE_MAP[s2]
    return None

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #

class TreatmentEntry(BaseModel):
    name: Optional[str] = None
    route: Optional[str] = None
    urls: List[str] = Field(default_factory=list)

class HospitalEntry(BaseModel):
    hospital_name: Optional[str] = None
    state: Optional[str] = None
    mda_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)
    treatments: List[TreatmentEntry] = Field(default_factory=list)

class HospitalsExtraction(BaseModel):
    hospitals: List[HospitalEntry] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #

def prompt_extract_hospitals() -> str:
    return (
        "Extract up to four pediatric hospitals from the answer that relate to spinal muscular atrophy (SMA). "
        "For each hospital, return a JSON object with the following fields:\n"
        "- hospital_name: The hospital's name exactly as in the answer.\n"
        "- state: The US state where the hospital is located (either full name or 2-letter abbreviation).\n"
        "- mda_urls: A list of URLs that explicitly verify the hospital is part of the Muscular Dystrophy Association (MDA) Care Center Network. "
        "Only include URLs that are present in the answer text.\n"
        "- location_urls: A list of URLs that explicitly verify the hospital's location/state. "
        "Only include URLs present in the answer.\n"
        "- treatments: An array of at least two treatment entries (if available). For each treatment, include:\n"
        "  * name: The treatment name as given in the answer. It should be one of: Zolgensma (onasemnogene abeparvovec), "
        "Spinraza (nusinersen), Evrysdi (risdiplam), or Itvisma (onasemnogene abeparvovec-brve). Generic names are acceptable.\n"
        "  * route: The administration route as described in the answer (e.g., 'intravenous', 'intrathecal', 'oral', or reasonable synonyms like 'IV infusion', 'by mouth').\n"
        "  * urls: A list of URLs present in the answer that verify this hospital offers this specific treatment.\n\n"
        "Return the result as a JSON object with a single field 'hospitals' that is an array of these hospital objects. "
        "If any field is missing for a hospital or treatment, set it to null or an empty list as appropriate. "
        "Do not invent URLs—extract only those explicitly included in the answer."
    )

# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #

def treatment_name_valid(name: Optional[str]) -> bool:
    return canonical_treatment(name) is not None

def treatment_route_matches(name: Optional[str], route_str: Optional[str]) -> bool:
    can = canonical_treatment(name)
    if not can:
        return False
    exp = expected_route_for(can)
    got = normalize_route(route_str)
    return (exp is not None) and (got == exp)

def distinct_from(first_name: Optional[str], second_name: Optional[str]) -> bool:
    c1 = canonical_treatment(first_name)
    c2 = canonical_treatment(second_name)
    return (c1 is not None) and (c2 is not None) and (c1 != c2)

# --------------------------------------------------------------------------- #
# Verification functions per hospital                                         #
# --------------------------------------------------------------------------- #

async def verify_hospital(
    evaluator: Evaluator,
    parent_node,
    hospital: HospitalEntry,
    hospital_index: int,
    prior_states: List[str],
) -> None:
    """
    Build and verify the tree for a single hospital.
    """
    # Hospital node (non-critical to allow partial scoring across hospitals)
    hosp_node = evaluator.add_parallel(
        id=f"hospital_{hospital_index+1}",
        desc=f"Hospital #{hospital_index+1} verification",
        parent=parent_node,
        critical=False
    )

    # Extract key fields
    hosp_name = hospital.hospital_name or ""
    raw_state = hospital.state or ""
    normalized_state = normalize_state(raw_state)

    # 1) Location checks (use a parallel aggregator under the named node)
    loc_node = evaluator.add_parallel(
        id=f"H{hospital_index+1}_Location",
        desc=f"Hospital {hospital_index+1} is located in a valid and distinct US state",
        parent=hosp_node,
        critical=True
    )

    # 1.a) State validity custom check
    evaluator.add_custom_node(
        result=(normalized_state is not None),
        id=f"H{hospital_index+1}_State_Valid",
        desc=f"Hospital {hospital_index+1} state value is a valid US state",
        parent=loc_node,
        critical=True
    )

    # 1.b) State supported by URL(s)
    state_supported_node = evaluator.add_leaf(
        id=f"H{hospital_index+1}_State_Supported",
        desc=f"Hospital {hospital_index+1} location/state is supported by cited URLs",
        parent=loc_node,
        critical=True
    )
    # Use both location_urls and mda_urls (MDA listings often include address/state)
    state_sources = list(hospital.location_urls or [])
    state_sources.extend(hospital.mda_urls or [])

    state_claim = (
        f"The hospital '{hosp_name}' is located in the US state '{raw_state}'. "
        f"Treat 'CA' as equivalent to 'California', 'NY' to 'New York', etc."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_supported_node,
        sources=state_sources,
        additional_instruction="Verify the US state information from the URLs. Allow reasonable equivalence between state abbreviations and full names."
    )

    # 1.c) State uniqueness among hospitals (except first one)
    if hospital_index >= 1:
        prev_set = set(prior_states)
        evaluator.add_custom_node(
            result=(normalized_state is not None and normalized_state not in prev_set),
            id=f"H{hospital_index+1}_State_Unique",
            desc=f"Hospital {hospital_index+1} is in a different US state from previously listed hospitals",
            parent=loc_node,
            critical=True
        )

    # Update prior states list for subsequent hospitals
    if normalized_state:
        prior_states.append(normalized_state)

    # 2) MDA Care Center status verified by URL
    mda_status_node = evaluator.add_leaf(
        id=f"H{hospital_index+1}_MDA_Status",
        desc=f"Hospital {hospital_index+1} is part of the MDA Care Center Network",
        parent=hosp_node,
        critical=True
    )
    mda_claim = f"The hospital '{hosp_name}' is part of the Muscular Dystrophy Association (MDA) Care Center Network."
    await evaluator.verify(
        claim=mda_claim,
        node=mda_status_node,
        sources=hospital.mda_urls or [],
        additional_instruction="Confirm that the hospital is listed as an MDA Care Center. Pages from mda.org or official hospital pages explicitly stating MDA Care Center status should support this claim."
    )

    # 3) Treatment count node (requires two treatments verified)
    t_count_node = evaluator.add_parallel(
        id=f"H{hospital_index+1}_Treatment_Count",
        desc=f"Hospital {hospital_index+1} offers at least 2 FDA-approved SMA treatments",
        parent=hosp_node,
        critical=True
    )

    # Take first two treatments from extraction
    t_entries: List[TreatmentEntry] = hospital.treatments[:2] if hospital.treatments else []
    while len(t_entries) < 2:
        t_entries.append(TreatmentEntry())

    # Treatment 1
    t1_node = evaluator.add_parallel(
        id=f"H{hospital_index+1}_Treatment_1",
        desc=f"Hospital {hospital_index+1} Treatment #1 verification",
        parent=t_count_node,
        critical=True
    )

    # 3.1.a Name valid for T1
    evaluator.add_custom_node(
        result=treatment_name_valid(t_entries[0].name),
        id=f"H{hospital_index+1}_T1_Name",
        desc=f"Name of the first treatment is one of the allowed FDA-approved SMA treatments",
        parent=t1_node,
        critical=True
    )

    # 3.1.b Route matches FDA-approved route for T1
    evaluator.add_custom_node(
        result=treatment_route_matches(t_entries[0].name, t_entries[0].route),
        id=f"H{hospital_index+1}_T1_Route",
        desc=f"Administration route of the first treatment matches the FDA-approved route for that treatment",
        parent=t1_node,
        critical=True
    )

    # 3.1.c URLs verify offering of T1
    t1_url_node = evaluator.add_leaf(
        id=f"H{hospital_index+1}_T1_URL",
        desc=f"URL reference verifying Hospital {hospital_index+1} offers the first treatment",
        parent=t1_node,
        critical=True
    )
    t1_can = canonical_treatment(t_entries[0].name) or (t_entries[0].name or "")
    t1_claim = f"The hospital '{hosp_name}' offers the SMA treatment '{t1_can}'."
    await evaluator.verify(
        claim=t1_claim,
        node=t1_url_node,
        sources=t_entries[0].urls or [],
        additional_instruction="Verify that the provided URLs explicitly indicate this hospital offers the specified treatment. Allow brand or generic name equivalence."
    )

    # Treatment 2
    t2_node = evaluator.add_parallel(
        id=f"H{hospital_index+1}_Treatment_2",
        desc=f"Hospital {hospital_index+1} Treatment #2 verification",
        parent=t_count_node,
        critical=True
    )

    # 3.2.a Name valid and different from T1
    evaluator.add_custom_node(
        result=(treatment_name_valid(t_entries[1].name) and distinct_from(t_entries[0].name, t_entries[1].name)),
        id=f"H{hospital_index+1}_T2_Name",
        desc=f"Name of the second treatment is allowed and different from the first treatment",
        parent=t2_node,
        critical=True
    )

    # 3.2.b Route matches FDA-approved route for T2
    evaluator.add_custom_node(
        result=treatment_route_matches(t_entries[1].name, t_entries[1].route),
        id=f"H{hospital_index+1}_T2_Route",
        desc=f"Administration route of the second treatment matches the FDA-approved route for that treatment",
        parent=t2_node,
        critical=True
    )

    # 3.2.c URLs verify offering of T2
    t2_url_node = evaluator.add_leaf(
        id=f"H{hospital_index+1}_T2_URL",
        desc=f"URL reference verifying Hospital {hospital_index+1} offers the second treatment",
        parent=t2_node,
        critical=True
    )
    t2_can = canonical_treatment(t_entries[1].name) or (t_entries[1].name or "")
    t2_claim = f"The hospital '{hosp_name}' offers the SMA treatment '{t2_can}'."
    await evaluator.verify(
        claim=t2_claim,
        node=t2_url_node,
        sources=t_entries[1].urls or [],
        additional_instruction="Verify that the provided URLs explicitly indicate this hospital offers the specified treatment. Allow brand or generic name equivalence."
    )

    # 4) URL reference existence check (both MDA and location references present)
    evaluator.add_custom_node(
        result=(bool(hospital.mda_urls) and bool(hospital.location_urls)),
        id=f"H{hospital_index+1}_URL_Reference",
        desc=f"URL references exist confirming Hospital {hospital_index+1}'s MDA Care Center status and location",
        parent=hosp_node,
        critical=True
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the SMA hospitals task.
    """
    # Initialize evaluator
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

    # Record ground truth-like info for routes
    evaluator.add_ground_truth({
        "fda_routes": {
            "Zolgensma": "intravenous",
            "Itvisma": "intrathecal",
            "Spinraza": "intrathecal",
            "Evrysdi": "oral",
        },
        "allowed_treatments": list(TREATMENT_SPECS.keys()),
        "synonyms": {k: v["synonyms"] for k, v in TREATMENT_SPECS.items()}
    })

    # Extract hospitals data
    extracted = await evaluator.extract(
        prompt=prompt_extract_hospitals(),
        template_class=HospitalsExtraction,
        extraction_name="hospitals_extraction"
    )

    # Prepare states tracking to enforce uniqueness across hospitals
    prior_states: List[str] = []

    # Root-level: four hospital nodes (non-critical children) to allow partial credit
    # Filter to first 4 hospitals; pad with empty entries if fewer
    hospitals = extracted.hospitals[:4]
    while len(hospitals) < 4:
        hospitals.append(HospitalEntry())

    # Build and verify each hospital
    for i in range(4):
        await verify_hospital(
            evaluator=evaluator,
            parent_node=root,
            hospital=hospitals[i],
            hospital_index=i,
            prior_states=prior_states
        )

    # Return structured summary
    return evaluator.get_summary()