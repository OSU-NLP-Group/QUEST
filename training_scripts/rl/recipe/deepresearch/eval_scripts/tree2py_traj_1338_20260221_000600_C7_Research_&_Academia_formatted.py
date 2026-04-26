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
TASK_ID = "artemis_ii_mission_info"
TASK_DESCRIPTION = (
    "Provide comprehensive information about NASA's Artemis II mission, including: "
    "(1) the complete crew roster with each member's role (commander, pilot, or mission specialist), "
    "(2) notable background information for each crew member (such as prior spaceflight experience, records held, or space agency affiliation), "
    "(3) the mission duration, "
    "(4) key distance milestones including total mission distance and how far beyond the lunar far side the crew will travel, "
    "(5) the Orion spacecraft's habitable volume, "
    "(6) the SLS Block 1 vehicle's height and maximum thrust, and "
    "(7) the splashdown location. "
    "For each piece of information, provide supporting reference URLs from official sources."
)

# Ground truth context (for logging only; not used for verification checks)
GROUND_TRUTH_CONTEXT = {
    "crew_expected": {
        "Commander": "Reid Wiseman",
        "Pilot": "Victor Glover",
        "Mission Specialist": ["Christina Koch", "Jeremy Hansen"]
    },
    "canonical_values_notes": {
        "mission_duration": "approximately 10 days",
        "total_distance_miles": "≈685,000+ miles",
        "far_side_beyond_miles": "≈4,700 miles beyond the lunar far side",
        "orion_habitable_volume": "330 ft^3 (≈9.3 m^3)",
        "sls_block1_height": "322.4 feet",
        "sls_max_thrust": "8.8 million pounds",
        "splashdown": "Pacific Ocean"
    }
}

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CrewMember(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    # URLs that support the identification of this member and their role on Artemis II
    sources: List[str] = Field(default_factory=list)


class ArtemisIIExtraction(BaseModel):
    # Crew roster
    crew: List[CrewMember] = Field(default_factory=list)

    # Background claims – booleans indicate whether the answer explicitly stated the claim;
    # sources should include URLs used in the answer to support the claim.
    wiseman_iss_4041_claimed: Optional[bool] = None
    wiseman_iss_sources: List[str] = Field(default_factory=list)

    glover_extended_iss_first_african_american_claimed: Optional[bool] = None
    glover_extended_iss_sources: List[str] = Field(default_factory=list)

    koch_longest_single_spaceflight_328d_claimed: Optional[bool] = None
    koch_record_sources: List[str] = Field(default_factory=list)

    hansen_csa_astronaut_claimed: Optional[bool] = None
    hansen_csa_sources: List[str] = Field(default_factory=list)

    # Mission parameters and sources
    mission_duration: Optional[str] = None
    mission_duration_sources: List[str] = Field(default_factory=list)

    total_mission_distance: Optional[str] = None
    total_distance_sources: List[str] = Field(default_factory=list)

    far_side_beyond_distance: Optional[str] = None
    far_side_beyond_sources: List[str] = Field(default_factory=list)

    orion_habitable_volume: Optional[str] = None
    orion_volume_sources: List[str] = Field(default_factory=list)

    sls_block1_height: Optional[str] = None
    sls_height_sources: List[str] = Field(default_factory=list)

    sls_maximum_thrust: Optional[str] = None
    sls_thrust_sources: List[str] = Field(default_factory=list)

    splashdown_location: Optional[str] = None
    splashdown_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_artemis_ii() -> str:
    return """
    Extract structured information that the answer provides about NASA's Artemis II mission. 
    IMPORTANT: Only extract information explicitly stated in the answer. Do not infer or invent.

    1) Crew roster and roles:
       - Return up to 4 crew members, each with:
         • name (e.g., "Reid Wiseman", "Victor Glover", "Christina Koch", "Jeremy Hansen")
         • role as one of: "Commander", "Pilot", or "Mission Specialist" (normalize common variants like "Mission Commander"→"Commander", "Orion pilot"→"Pilot")
         • sources: all URLs in the answer that support that member's role on Artemis II.
       - If the answer lists more than 4 people, include only those clearly identified as Artemis II crew.

    2) Background claims for each named astronaut (mark TRUE only if the answer explicitly states the claim):
       - wiseman_iss_4041_claimed: true/false (Does the answer state Reid Wiseman served on ISS Expeditions 40/41?)
         • wiseman_iss_sources: URLs used to support this
       - glover_extended_iss_first_african_american_claimed: true/false (Does the answer state Victor Glover was the first African American (or Black) astronaut to live on ISS on a long-duration/extended mission?)
         • glover_extended_iss_sources: URLs used to support this
       - koch_longest_single_spaceflight_328d_claimed: true/false (Does the answer state Christina Koch holds the record for the longest single spaceflight by a woman at ~328 days?)
         • koch_record_sources: URLs used to support this
       - hansen_csa_astronaut_claimed: true/false (Does the answer state Jeremy Hansen is a Canadian Space Agency (CSA) astronaut?)
         • hansen_csa_sources: URLs used to support this

    3) Mission parameters (strings as they appear in the answer; DO NOT normalize units):
       - mission_duration (e.g., "about 10 days") + mission_duration_sources
       - total_mission_distance (e.g., "about 685,000 miles") + total_distance_sources
       - far_side_beyond_distance (e.g., "about 4,700 miles beyond the far side") + far_side_beyond_sources
       - orion_habitable_volume (e.g., "330 cubic feet (9.3 m^3)") + orion_volume_sources
       - sls_block1_height (e.g., "322.4 feet") + sls_height_sources
       - sls_maximum_thrust (e.g., "8.8 million pounds of thrust") + sls_thrust_sources
       - splashdown_location (e.g., "Pacific Ocean") + splashdown_sources

    SPECIAL RULES FOR URL FIELDS:
    - Extract only URLs explicitly present in the answer text (plain or markdown links).
    - Include all URLs that were used to justify each specific piece of information.
    - If no sources are provided for a field, return an empty list for that field's sources.

    Return a single JSON object matching the target schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_role(role: Optional[str]) -> Optional[str]:
    if role is None:
        return None
    r = role.strip().lower()
    if "commander" in r:
        return "Commander"
    if "pilot" in r:
        return "Pilot"
    if "mission specialist" in r or "specialist" in r:
        return "Mission Specialist"
    return role.strip()


def _name_matches(candidate: Optional[str], target: str) -> bool:
    if not candidate:
        return False
    return candidate.strip().lower() == target.strip().lower()


def _role_matches(candidate_role: Optional[str], expected_role: str) -> bool:
    return _normalize_role(candidate_role) == expected_role


def _find_member_by_name(crew: List[CrewMember], target_name: str) -> Optional[CrewMember]:
    for m in crew:
        if _name_matches(m.name, target_name):
            return m
    return None


def _find_member_by_role(crew: List[CrewMember], target_role: str) -> Optional[CrewMember]:
    for m in crew:
        if _role_matches(m.role, target_role):
            return m
    return None


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    res = []
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if uu and uu not in seen:
            seen.add(uu)
            res.append(uu)
    return res


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_identity(
    evaluator: Evaluator,
    root,
    extracted: ArtemisIIExtraction,
    expected_name: str,
    expected_role: str,
    group_id: str,
    leaf_id: str,
    leaf_desc: str,
    group_desc: str,
    critical: bool = True,
):
    """
    Build a small parallel group for a single identity item.
    Structure:
      - existence (custom, critical): answer contains expected name+role with at least one source
      - source-supported (leaf, critical): verify "X is the <role> for Artemis II" with provided URLs
    """
    group = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=root,
        critical=critical
    )

    # Try to find member by name first, then by role
    member = _find_member_by_name(extracted.crew, expected_name) or _find_member_by_role(extracted.crew, expected_role)
    member_sources = _dedupe_urls(member.sources if member else [])

    existence_ok = (
        member is not None and
        _name_matches(member.name, expected_name) and
        _role_matches(member.role, expected_role) and
        len(member_sources) > 0
    )

    evaluator.add_custom_node(
        result=existence_ok,
        id=f"{group_id}_exists",
        desc=f"{expected_name} is identified as {expected_role} with supporting sources in the answer",
        parent=group,
        critical=True
    )

    # Leaf: verify with sources
    leaf = evaluator.add_leaf(
        id=leaf_id,
        desc=leaf_desc,
        parent=group,
        critical=True
    )

    claim = f"For Artemis II, {expected_name} is the {expected_role}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=member_sources,
        additional_instruction=(
            "Verify that the cited page(s) explicitly state that this person holds the specified role on Artemis II. "
            "Allow reasonable phrasing variants (e.g., 'mission commander' for Commander, 'Orion pilot' for Pilot). "
            "Prefer official NASA/CSA sources if provided."
        )
    )


async def verify_background_claim(
    evaluator: Evaluator,
    root,
    claimed: Optional[bool],
    sources: List[str],
    group_id: str,
    leaf_id: str,
    group_desc: str,
    leaf_desc: str,
    claim_text: str,
    additional_instruction: str
):
    """
    Group:
      - existence (custom, critical): answer explicitly states the background fact and provided at least one source
      - supported (leaf, critical): verify the background fact with the provided URLs
    The group itself is non-critical (partial credit).
    """
    group = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=root,
        critical=False
    )
    srcs = _dedupe_urls(sources or [])
    existence_ok = bool(claimed) and len(srcs) > 0

    evaluator.add_custom_node(
        result=existence_ok,
        id=f"{group_id}_exists",
        desc="The answer explicitly states this background fact and provides source URL(s)",
        parent=group,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=leaf_id,
        desc=leaf_desc,
        parent=group,
        critical=True
    )

    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=srcs,
        additional_instruction=additional_instruction
    )


async def verify_simple_fact_with_sources(
    evaluator: Evaluator,
    root,
    value: Optional[str],
    sources: List[str],
    group_id: str,
    leaf_id: str,
    group_desc: str,
    leaf_desc: str,
    make_claim_fn,
    critical: bool = True,
    additional_instruction: str = "None"
):
    """
    Generic pattern for mission parameters.
    Group:
      - existence (custom, critical): value present and at least one source URL
      - supported (leaf, critical): verify claim built from the value against the URLs
    """
    group = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=root,
        critical=critical
    )
    srcs = _dedupe_urls(sources or [])
    existence_ok = (value is not None and str(value).strip() != "" and len(srcs) > 0)

    evaluator.add_custom_node(
        result=existence_ok,
        id=f"{group_id}_exists",
        desc="The answer states this information and provides supporting reference URL(s)",
        parent=group,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=leaf_id,
        desc=leaf_desc,
        parent=group,
        critical=True
    )

    claim = make_claim_fn(value or "")
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction=additional_instruction
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
    Evaluate an answer for the Artemis II mission information task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation of criteria
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

    # Record ground-truth context info for easier debugging (not used for scoring)
    evaluator.add_ground_truth(GROUND_TRUTH_CONTEXT, gt_type="context_notes")

    # 1) Extract all relevant information from the answer
    extracted: ArtemisIIExtraction = await evaluator.extract(
        prompt=prompt_extract_artemis_ii(),
        template_class=ArtemisIIExtraction,
        extraction_name="artemis_ii_extraction"
    )

    # 2) Crew identity verifications (critical)
    await verify_identity(
        evaluator,
        root,
        extracted,
        expected_name="Reid Wiseman",
        expected_role="Commander",
        group_id="commander_identity_group",
        leaf_id="Commander_Identity",
        leaf_desc="Correctly identifies Reid Wiseman as the mission Commander",
        group_desc="Verify Commander identity with sources",
        critical=True
    )

    await verify_identity(
        evaluator,
        root,
        extracted,
        expected_name="Victor Glover",
        expected_role="Pilot",
        group_id="pilot_identity_group",
        leaf_id="Pilot_Identity",
        leaf_desc="Correctly identifies Victor Glover as the mission Pilot",
        group_desc="Verify Pilot identity with sources",
        critical=True
    )

    await verify_identity(
        evaluator,
        root,
        extracted,
        expected_name="Christina Koch",
        expected_role="Mission Specialist",
        group_id="ms_koch_identity_group",
        leaf_id="Mission_Specialist_Koch",
        leaf_desc="Correctly identifies Christina Koch as a Mission Specialist",
        group_desc="Verify Mission Specialist identity (Koch) with sources",
        critical=True
    )

    await verify_identity(
        evaluator,
        root,
        extracted,
        expected_name="Jeremy Hansen",
        expected_role="Mission Specialist",
        group_id="ms_hansen_identity_group",
        leaf_id="Mission_Specialist_Hansen",
        leaf_desc="Correctly identifies Jeremy Hansen as a Mission Specialist",
        group_desc="Verify Mission Specialist identity (Hansen) with sources",
        critical=True
    )

    # 3) Background claims (non-critical)
    await verify_background_claim(
        evaluator,
        root,
        claimed=extracted.wiseman_iss_4041_claimed,
        sources=extracted.wiseman_iss_sources,
        group_id="wiseman_iss_background",
        leaf_id="Wiseman_ISS_Experience",
        group_desc="Background: Reid Wiseman ISS Expeditions 40/41",
        leaf_desc="States that Reid Wiseman served on ISS Expedition 40/41",
        claim_text="Reid Wiseman served on International Space Station Expeditions 40 and 41.",
        additional_instruction="Allow phrasing variants like 'Expedition 40/41' or 'Expeditions 40 and 41'. Prefer NASA biography pages if provided."
    )

    await verify_background_claim(
        evaluator,
        root,
        claimed=extracted.glover_extended_iss_first_african_american_claimed,
        sources=extracted.glover_extended_iss_sources,
        group_id="glover_historic_background",
        leaf_id="Glover_Historic_Achievement",
        group_desc="Background: Victor Glover historic ISS achievement",
        leaf_desc="States that Victor Glover was the first African American to spend extended time on the ISS",
        claim_text=(
            "Victor Glover was the first Black/African American astronaut to live on the International Space Station "
            "for a long-duration (extended) mission."
        ),
        additional_instruction=(
            "Allow phrasing variants: 'first Black astronaut to live on ISS for an extended stay', 'first African American on a long-duration ISS mission'. "
            "Prefer official NASA pages if provided."
        )
    )

    await verify_background_claim(
        evaluator,
        root,
        claimed=extracted.koch_longest_single_spaceflight_328d_claimed,
        sources=extracted.koch_record_sources,
        group_id="koch_record_background",
        leaf_id="Koch_Spaceflight_Record",
        group_desc="Background: Christina Koch single-flight duration record",
        leaf_desc="States that Christina Koch holds the record for longest single spaceflight by a woman at 328 days",
        claim_text="Christina Koch holds the record for the longest single spaceflight by a woman at about 328 days.",
        additional_instruction="Allow 'single continuous mission' phrasing and minor rounding (e.g., 328 days). Prefer NASA pages."
    )

    await verify_background_claim(
        evaluator,
        root,
        claimed=extracted.hansen_csa_astronaut_claimed,
        sources=extracted.hansen_csa_sources,
        group_id="hansen_csa_background",
        leaf_id="Hansen_Nationality",
        group_desc="Background: Jeremy Hansen CSA affiliation",
        leaf_desc="States that Jeremy Hansen is a Canadian Space Agency astronaut",
        claim_text="Jeremy Hansen is an astronaut with the Canadian Space Agency (CSA).",
        additional_instruction="Prefer CSA or NASA official pages if provided; allow minor phrasing differences."
    )

    # 4) Mission duration (critical)
    await verify_simple_fact_with_sources(
        evaluator,
        root,
        value=extracted.mission_duration,
        sources=extracted.mission_duration_sources,
        group_id="mission_duration_group",
        leaf_id="Mission_Duration",
        group_desc="Mission Duration",
        leaf_desc="States the mission duration as approximately 10 days (as stated in the answer, verified by sources)",
        make_claim_fn=lambda v: f"The Artemis II mission duration is {v}.",
        critical=True,
        additional_instruction=(
            "Verify that the page indicates the Artemis II mission duration matches the stated value. "
            "Allow 'about/approximately' phrasing. Prefer NASA Artemis II pages."
        )
    )

    # 5) Total mission distance (critical)
    await verify_simple_fact_with_sources(
        evaluator,
        root,
        value=extracted.total_mission_distance,
        sources=extracted.total_distance_sources,
        group_id="total_distance_group",
        leaf_id="Total_Mission_Distance",
        group_desc="Total mission distance",
        leaf_desc="States the total mission distance (as in the answer) with source support",
        make_claim_fn=lambda v: f"The total Artemis II mission distance is {v}.",
        critical=True,
        additional_instruction=(
            "Allow approximate values and unit conversions (miles↔kilometers). "
            "The claim should be explicitly supported. Prefer NASA pages."
        )
    )

    # 6) Far-side beyond distance (critical)
    await verify_simple_fact_with_sources(
        evaluator,
        root,
        value=extracted.far_side_beyond_distance,
        sources=extracted.far_side_beyond_sources,
        group_id="far_side_beyond_group",
        leaf_id="Lunar_Farside_Distance",
        group_desc="Distance beyond lunar far side",
        leaf_desc="States how far beyond the lunar far side the crew will travel (as in the answer) with source support",
        make_claim_fn=lambda v: f"The Artemis II trajectory takes the crew {v} beyond the Moon's far side.",
        critical=True,
        additional_instruction=(
            "Allow approximate values and conversions (miles↔kilometers). "
            "Look for language like 'beyond the far side' or 'distant retrograde orbit excursion'. Prefer NASA pages."
        )
    )

    # 7) Orion habitable volume (critical)
    await verify_simple_fact_with_sources(
        evaluator,
        root,
        value=extracted.orion_habitable_volume,
        sources=extracted.orion_volume_sources,
        group_id="orion_volume_group",
        leaf_id="Orion_Habitable_Volume",
        group_desc="Orion spacecraft habitable volume",
        leaf_desc="States the Orion spacecraft habitable volume (as in the answer) with source support",
        make_claim_fn=lambda v: f"The Orion spacecraft habitable volume is {v}.",
        critical=True,
        additional_instruction=(
            "Allow equivalent units (cubic feet vs cubic meters) and minor rounding. Prefer NASA Orion fact pages."
        )
    )

    # 8) SLS Block 1 height (critical)
    await verify_simple_fact_with_sources(
        evaluator,
        root,
        value=extracted.sls_block1_height,
        sources=extracted.sls_height_sources,
        group_id="sls_height_group",
        leaf_id="SLS_Vehicle_Height",
        group_desc="SLS Block 1 vehicle height",
        leaf_desc="States the SLS Block 1 vehicle height (as in the answer) with source support",
        make_claim_fn=lambda v: f"The SLS Block 1 vehicle height is {v}.",
        critical=True,
        additional_instruction=(
            "Allow equivalent metric conversions (e.g., ≈98 m) and minor rounding. Prefer NASA SLS pages."
        )
    )

    # 9) SLS maximum thrust (critical)
    await verify_simple_fact_with_sources(
        evaluator,
        root,
        value=extracted.sls_maximum_thrust,
        sources=extracted.sls_thrust_sources,
        group_id="sls_thrust_group",
        leaf_id="SLS_Maximum_Thrust",
        group_desc="SLS maximum thrust",
        leaf_desc="States the SLS maximum thrust (as in the answer) with source support",
        make_claim_fn=lambda v: f"The SLS maximum thrust is {v}.",
        critical=True,
        additional_instruction=(
            "Allow phrasing variants like '8.8 million pounds of thrust' or '8.8 million lbf'. Prefer NASA SLS pages."
        )
    )

    # 10) Splashdown location (critical)
    await verify_simple_fact_with_sources(
        evaluator,
        root,
        value=extracted.splashdown_location,
        sources=extracted.splashdown_sources,
        group_id="splashdown_location_group",
        leaf_id="Splashdown_Location",
        group_desc="Splashdown location",
        leaf_desc="States that the mission ends with a Pacific Ocean splashdown (as in the answer) with source support",
        make_claim_fn=lambda v: f"The Artemis II mission ends with a {v} splashdown.",
        critical=True,
        additional_instruction=(
            "Verify that the splashdown location matches the stated value. "
            "If 'Pacific Ocean' is stated, confirm the cited page indicates Pacific splashdown. Prefer NASA."
        )
    )

    # 11) Return evaluation summary
    return evaluator.get_summary()