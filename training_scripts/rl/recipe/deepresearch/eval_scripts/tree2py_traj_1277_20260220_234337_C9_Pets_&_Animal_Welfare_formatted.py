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
TASK_ID = "veccs_level1_top_tier_4"
TASK_DESCRIPTION = (
    "Identify four veterinary emergency facilities that hold VECCS (Veterinary Emergency and Critical Care Society) "
    "Level I certification. Each facility must be located in a different state, and all four states must be ranked in "
    "the Top Tier (ranks 1-15) of the Animal Legal Defense Fund's 2025 U.S. Animal Protection Laws Rankings. For each "
    "facility, provide: (1) The facility name and location (city and state), (2) Confirmation that it holds VECCS Level I "
    "certification (which requires meeting all Level III and Level II requirements, plus specific Level I requirements including: "
    "backup power supply, invasive blood pressure monitoring equipment, bronchoscopy equipment, CT scanner, echocardiography equipment, "
    "ICU ventilator, total parenteral nutrition capability, at least one full-time Emergency and Critical Care specialist, and at least "
    "two full-time VTS(ECC) technicians), (3) Confirmation that the facility's state is in the ALDF 2025 Top Tier (the Top Tier states are: "
    "Oregon, Massachusetts, Maine, Illinois, Colorado, California, Florida, Washington, Rhode Island, Louisiana, Arizona, Connecticut, "
    "Virginia, New Jersey, and Texas), and (4) A reference URL that verifies the facility's VECCS Level I certification and location."
)

TOP_TIER_STATES = [
    "Oregon", "Massachusetts", "Maine", "Illinois", "Colorado",
    "California", "Florida", "Washington", "Rhode Island", "Louisiana",
    "Arizona", "Connecticut", "Virginia", "New Jersey", "Texas"
]

US_STATE_ABBR_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
}
US_STATE_NAMES = set(US_STATE_ABBR_TO_NAME.values())

LEVEL_DEF_INSTRUCTION = (
    "This is a logical implication check based on the VECCS Hospital Certification program definitions. "
    "If a facility is VECCS Level I certified, it must meet all VECCS Level III baseline requirements, all VECCS Level II requirements, "
    "and the specific Level I items: backup power supply; invasive blood pressure monitoring equipment; bronchoscopy equipment; CT scanner; "
    "echocardiography equipment; ICU ventilator; total parenteral nutrition capability; at least one full-time Emergency and Critical Care (ECC) specialist; "
    "and at least two full-time VTS(ECC) technicians. You are not asked to find these items in the webpage; instead, verify the logical implication "
    "given that certification is confirmed. Allow minor wording variations (e.g., 'Level 1' vs 'Level I')."
)

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class FacilityItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    veccs_level_text: Optional[str] = None
    reference_url: Optional[str] = None
    support_urls: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    facilities: List[FacilityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return (
        "Extract up to four veterinary emergency facilities mentioned in the answer that are claimed to hold VECCS Level I certification. "
        "For each facility, return a JSON object with the following fields:\n"
        "- name: the facility name\n"
        "- city: the city\n"
        "- state: the U.S. state as a full name (e.g., 'Texas'); if an abbreviation is used in the answer (e.g., 'TX'), still return it as given\n"
        "- veccs_level_text: the certification level text as presented (e.g., 'VECCS Level I' or 'Level 1')\n"
        "- reference_url: a single URL explicitly cited in the answer that is intended to verify BOTH the facility's VECCS Level I certification and its location\n"
        "- support_urls: any additional URLs explicitly cited in the answer for this facility (do not include duplicates of reference_url)\n\n"
        "Rules:\n"
        "1. Only extract information explicitly present in the answer; do not invent URLs.\n"
        "2. If more than four facilities are presented, return only the first four in the order they appear.\n"
        "3. If the answer does not provide enough facilities, include as many as available, and set missing fields to null.\n"
        "4. For URLs, accept plain URLs or markdown links; always return the actual URL string.\n"
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def normalize_state_name(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    st = s.strip()
    if not st:
        return None
    # Try exact match ignoring case
    for full in US_STATE_NAMES:
        if full.lower() == st.lower():
            return full
    # Try abbreviation
    abbr = st.upper()
    if abbr in US_STATE_ABBR_TO_NAME:
        return US_STATE_ABBR_TO_NAME[abbr]
    # Some common variations
    variations = {
        "District of Columbia": None,  # not a state
    }
    if st in variations:
        return variations[st]
    return None


def compose_sources(f: FacilityItem) -> List[str]:
    urls = []
    if f.reference_url and f.reference_url.strip():
        urls.append(f.reference_url.strip())
    for u in f.support_urls:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for u in urls:
        if u not in seen:
            ordered.append(u)
            seen.add(u)
    return ordered


# --------------------------------------------------------------------------- #
# Verification per facility                                                   #
# --------------------------------------------------------------------------- #
async def verify_facility(
    evaluator: Evaluator,
    root_parent,
    facility: FacilityItem,
    index: int,
    previous_states: List[str],
) -> None:
    fac_idx = index + 1
    fac_node = evaluator.add_parallel(
        id=f"Facility_{fac_idx}",
        desc=("First veterinary emergency facility meeting all requirements" if fac_idx == 1 else
              "Second veterinary emergency facility meeting all requirements" if fac_idx == 2 else
              "Third veterinary emergency facility meeting all requirements" if fac_idx == 3 else
              "Fourth veterinary emergency facility meeting all requirements"),
        parent=root_parent,
        critical=False
    )

    # Identification: name, city, state provided (critical)
    ident_ok = bool(facility.name and facility.name.strip()) and bool(facility.city and facility.city.strip()) and bool(facility.state and facility.state.strip())
    evaluator.add_custom_node(
        result=ident_ok,
        id=f"Facility_{fac_idx}_Identification",
        desc=f"Facility {fac_idx}'s name, city, and state are provided",
        parent=fac_node,
        critical=True
    )

    # State validity / different
    normalized_state = normalize_state_name(facility.state)
    if fac_idx == 1:
        state_valid = normalized_state in US_STATE_NAMES
        evaluator.add_custom_node(
            result=state_valid,
            id=f"Facility_{fac_idx}_State_Different",
            desc=f"Facility {fac_idx} is located in a U.S. state",
            parent=fac_node,
            critical=True
        )
    else:
        # Validity check (non-JSON node; helpful)
        state_valid = normalized_state in US_STATE_NAMES
        evaluator.add_custom_node(
            result=state_valid,
            id=f"Facility_{fac_idx}_State_Valid",
            desc=f"Facility {fac_idx} has a valid U.S. state name",
            parent=fac_node,
            critical=True
        )
        # Different from previous facilities
        is_unique = (normalized_state is not None) and (normalized_state not in previous_states)
        evaluator.add_custom_node(
            result=is_unique,
            id=f"Facility_{fac_idx}_State_Different",
            desc=(f"Facility {fac_idx} is located in a U.S. state different from Facility 1" if fac_idx == 2 else
                  f"Facility {fac_idx} is located in a U.S. state different from Facilities 1 and 2" if fac_idx == 3 else
                  f"Facility {fac_idx} is located in a U.S. state different from Facilities 1, 2, and 3"),
            parent=fac_node,
            critical=True
        )

    # ALDF Top Tier check (critical) via simple verification
    aldf_node = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_ALDF_Top_Tier",
        desc=(f"Facility {fac_idx}'s state is ranked in ALDF 2025 Top Tier "
              "(ranks 1-15: Oregon, Massachusetts, Maine, Illinois, Colorado, California, Florida, "
              "Washington, Rhode Island, Louisiana, Arizona, Connecticut, Virginia, New Jersey, or Texas)"),
        parent=fac_node,
        critical=True
    )
    state_for_claim = normalized_state or (facility.state or "").strip()
    aldf_claim = f"The state '{state_for_claim}' is in the ALDF 2025 Top Tier list: {', '.join(TOP_TIER_STATES)}."
    await evaluator.verify(
        claim=aldf_claim,
        node=aldf_node,
        additional_instruction=(
            "Judge this claim by comparing the provided state to the explicit Top Tier list given in the claim. "
            "Allow common abbreviations (e.g., 'TX' for 'Texas') or minor casing differences. "
            "Do not use any external sources; evaluate only the statement itself."
        )
    )

    # Reference URL leaf (critical): verify both certification and location via source(s)
    ref_sources = compose_sources(facility)
    ref_url_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_Reference_URL",
        desc=f"A reference URL is provided that verifies Facility {fac_idx}'s VECCS Level I certification and state location",
        parent=fac_node,
        critical=True
    )
    ref_claim = (
        f"The veterinary emergency facility named '{facility.name or ''}' is located in {facility.city or ''}, {facility.state or ''}, "
        f"and holds VECCS Level I certification."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_url_leaf,
        sources=ref_sources if ref_sources else None,
        additional_instruction=(
            "Use the provided webpage(s) to confirm BOTH parts of the claim: "
            "(1) the location (city and state), and (2) VECCS Level I certification. "
            "If either piece is missing or unclear, mark as not supported. "
            "Accept minor formatting differences (e.g., 'Level 1' vs 'Level I')."
        )
    )

    # VECCS Certification subtree (critical, sequential)
    cert_node = evaluator.add_sequential(
        id=f"Facility_{fac_idx}_VECCS_Certification",
        desc=f"Facility {fac_idx} holds VECCS Level I certification",
        parent=fac_node,
        critical=True
    )

    # Level III baseline requirements (critical leaf) - logical implication
    lvl3_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_Level_III_Requirements",
        desc=f"Facility {fac_idx} meets all VECCS Level III baseline requirements",
        parent=cert_node,
        critical=True
    )
    await evaluator.verify(
        claim=("A VECCS Level I certified facility necessarily satisfies all VECCS Level III baseline requirements."),
        node=lvl3_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # Level II requirements subtree (critical, parallel)
    lvl2_node = evaluator.add_parallel(
        id=f"Facility_{fac_idx}_Level_II_Requirements",
        desc=f"Facility {fac_idx} meets all VECCS Level II requirements",
        parent=cert_node,
        critical=True
    )

    # 24/7 operations (critical leaf)
    ops_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_24_7_Operations",
        desc=f"Facility {fac_idx} is open to receive emergency patients 24 hours a day, 7 days a week, 365 days a year",
        parent=lvl2_node,
        critical=True
    )
    await evaluator.verify(
        claim=("A VECCS Level I facility must meet VECCS Level II requirements, including 24/7/365 emergency operations."),
        node=ops_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # Additional medications (critical leaf)
    meds_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_Level_II_Medications",
        desc=("Facility {idx} has all required additional medications readily available "
              "(magnesium sulfate or chloride, sodium or potassium phosphate, diltiazem, norepinephrine, "
              "procainamide, sodium nitroprusside or hydralazine)").format(idx=fac_idx),
        parent=lvl2_node,
        critical=True
    )
    await evaluator.verify(
        claim=("Level II requirements include having the additional medications: magnesium sulfate/chloride, "
               "sodium/potassium phosphate, diltiazem, norepinephrine, procainamide, sodium nitroprusside/hydralazine; "
               "therefore a Level I facility meets this requirement."),
        node=meds_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # Level II equipment (critical leaf)
    equip_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_Level_II_Equipment",
        desc=("Facility {idx} has all required Level II equipment (anesthesia ventilator, central venous catheters, "
              "crystalloid maintenance fluids, defibrillator, endoscopy equipment, dedicated ER/ICU monitoring)").format(idx=fac_idx),
        parent=lvl2_node,
        critical=True
    )
    await evaluator.verify(
        claim=("Level II requires specified equipment including an anesthesia ventilator, central venous catheters, "
               "crystalloid maintenance fluids, defibrillator, endoscopy equipment, and ER/ICU monitoring; "
               "a Level I facility meets this requirement."),
        node=equip_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # Level II capabilities (critical leaf)
    caps_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_Level_II_Capabilities",
        desc=("Facility {idx} has blood gas testing capability, high-flow oxygen capability, and partial parenteral nutrition capability").format(idx=fac_idx),
        parent=lvl2_node,
        critical=True
    )
    await evaluator.verify(
        claim=("Level II capability requirements include blood gas testing, high-flow oxygen, and partial parenteral nutrition; "
               "a Level I facility meets this requirement."),
        node=caps_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # Level II staffing (critical leaf)
    staff_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_Level_II_Staffing",
        desc=("Facility {idx} has at least one full-time Internal Medicine specialist and at least one full-time Surgery specialist").format(idx=fac_idx),
        parent=lvl2_node,
        critical=True
    )
    await evaluator.verify(
        claim=("Level II staffing requires at least one full-time Internal Medicine specialist and at least one full-time Surgery specialist; "
               "a Level I facility meets this requirement."),
        node=staff_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # Level I specific requirements subtree (critical, parallel)
    lvl1_node = evaluator.add_parallel(
        id=f"Facility_{fac_idx}_Level_I_Requirements",
        desc=f"Facility {fac_idx} meets all VECCS Level I specific requirements",
        parent=cert_node,
        critical=True
    )

    # Backup power
    bp_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_Backup_Power",
        desc=f"Facility {fac_idx} has backup power supply",
        parent=lvl1_node,
        critical=True
    )
    await evaluator.verify(
        claim=("VECCS Level I requires a backup power supply; a Level I facility must have it."),
        node=bp_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # Invasive BP monitoring
    ibp_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_Invasive_BP_Monitoring",
        desc=f"Facility {fac_idx} has invasive blood pressure monitoring equipment",
        parent=lvl1_node,
        critical=True
    )
    await evaluator.verify(
        claim=("VECCS Level I requires invasive blood pressure monitoring equipment; a Level I facility must have it."),
        node=ibp_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # Bronchoscopy
    bronc_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_Bronchoscopy",
        desc=f"Facility {fac_idx} has bronchoscopy equipment",
        parent=lvl1_node,
        critical=True
    )
    await evaluator.verify(
        claim=("VECCS Level I requires bronchoscopy equipment; a Level I facility must have it."),
        node=bronc_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # CT scanner
    ct_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_CT_Scanner",
        desc=f"Facility {fac_idx} has CT scanner",
        parent=lvl1_node,
        critical=True
    )
    await evaluator.verify(
        claim=("VECCS Level I requires a CT scanner; a Level I facility must have it."),
        node=ct_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # Echocardiography
    echo_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_Echocardiography",
        desc=f"Facility {fac_idx} has echocardiography equipment",
        parent=lvl1_node,
        critical=True
    )
    await evaluator.verify(
        claim=("VECCS Level I requires echocardiography equipment; a Level I facility must have it."),
        node=echo_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # ICU ventilator
    icu_vent_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_ICU_Ventilator",
        desc=f"Facility {fac_idx} has ICU ventilator",
        parent=lvl1_node,
        critical=True
    )
    await evaluator.verify(
        claim=("VECCS Level I requires an ICU ventilator; a Level I facility must have it."),
        node=icu_vent_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # TPN capability
    tpn_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_TPN_Capability",
        desc=f"Facility {fac_idx} has capability to provide total parenteral nutrition",
        parent=lvl1_node,
        critical=True
    )
    await evaluator.verify(
        claim=("VECCS Level I requires total parenteral nutrition capability; a Level I facility must have it."),
        node=tpn_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # ECC specialist
    ecc_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_ECC_Specialist",
        desc=f"Facility {fac_idx} has at least one full-time Emergency and Critical Care specialist employed",
        parent=lvl1_node,
        critical=True
    )
    await evaluator.verify(
        claim=("VECCS Level I requires at least one full-time ECC specialist; a Level I facility must have it."),
        node=ecc_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # VTS(ECC) technicians
    vts_leaf = evaluator.add_leaf(
        id=f"Facility_{fac_idx}_VTS_ECC_Technicians",
        desc=f"Facility {fac_idx} has at least two full-time VTS(ECC) technicians employed",
        parent=lvl1_node,
        critical=True
    )
    await evaluator.verify(
        claim=("VECCS Level I requires at least two full-time VTS(ECC) technicians; a Level I facility must have them."),
        node=vts_leaf,
        additional_instruction=LEVEL_DEF_INSTRUCTION,
        extra_prerequisites=[ref_url_leaf]
    )

    # Update previous states list for uniqueness checks outside
    if normalized_state:
        previous_states.append(normalized_state)


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
    Evaluate an answer for the VECCS Level I + ALDF Top Tier multi-facility task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Parallel: each facility evaluated independently
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

    # Extract facilities from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction"
    )

    # Add ground truth info about ALDF top tier states
    evaluator.add_ground_truth({
        "aldf_2025_top_tier_states": TOP_TIER_STATES,
        "requirement": "All four facilities must be in distinct states from this Top Tier list."
    })

    # Prepare up to four facilities
    facilities: List[FacilityItem] = list(extracted.facilities[:4])
    while len(facilities) < 4:
        facilities.append(FacilityItem())

    # Track previously used states to enforce distinctness
    previous_states: List[str] = []

    # Verify each facility
    for idx in range(4):
        await verify_facility(evaluator, root, facilities[idx], idx, previous_states)

    # Return structured result
    return evaluator.get_summary()