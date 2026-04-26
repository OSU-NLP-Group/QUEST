import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "artemis_ii_2026"
TASK_DESCRIPTION = """
Provide comprehensive information about NASA's Artemis II mission scheduled for 2026. Your response must include:
(1) The names and roles of all four crew members (commander, pilot, and two mission specialists), specifying their space agency affiliation;
(2) For each crew member, identify the university or institution where they earned their bachelor's degree;
(3) For each crew member who has a master's degree, identify the university or institution where they earned it;
(4) The specific launch complex from which the mission will launch;
(5) The earliest possible launch date;
(6) The name of the rocket system being used;
(7) The name of the spacecraft;
(8) The approximate duration of the mission.
All information must be verifiable through official sources.
"""

# Ground-truth expectations for value-matching checks
EXPECTED = {
    "mission_name": "Artemis II",
    "crew": [
        {"name": "Reid Wiseman", "role": "commander", "agency": "NASA"},
        {"name": "Victor Glover", "role": "pilot", "agency": "NASA"},
        {"name": "Christina Koch", "role": "mission specialist", "agency": "NASA"},
        {"name": "Jeremy Hansen", "role": "mission specialist", "agency": "CSA"},
    ],
    "education": {
        "Reid Wiseman": {
            "bachelor_institution": "Rensselaer Polytechnic Institute",
            "master_institution": "Johns Hopkins University",
        },
        "Victor Glover": {
            "bachelor_institution": "California Polytechnic State University",
            # Masters: multiple possible correct institutions (e.g., Naval Postgraduate School, Air University).
            # We will validate from sources rather than enforcing a single school name here.
        },
        "Christina Koch": {
            "bachelor_institution": "North Carolina State University",
            "master_institution": "North Carolina State University",
        },
        "Jeremy Hansen": {
            "bachelor_institution": "Royal Military College of Canada",
            "master_institution": "Royal Military College of Canada",
        },
    },
    "launch_complex": "Kennedy Space Center Launch Complex 39B",
    "earliest_launch_date": "April 1, 2026",
    "rocket_system": "Space Launch System",
    "spacecraft": "Orion",
    "mission_duration_approx_days": "10",  # accept approximate "10 days"
}


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class DegreeInfo(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None  # optional label (e.g., "B.S. in Electrical Engineering")


class CrewMember(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None  # commander, pilot, mission specialist
    agency: Optional[str] = None  # NASA or CSA
    bachelors: List[DegreeInfo] = Field(default_factory=list)
    masters: List[DegreeInfo] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)  # URLs provided in the answer for this member


class ArtemisExtraction(BaseModel):
    mission_name: Optional[str] = None
    crew: List[CrewMember] = Field(default_factory=list)
    launch_complex: Optional[str] = None
    earliest_launch_date: Optional[str] = None
    rocket_system: Optional[str] = None
    spacecraft: Optional[str] = None
    mission_duration: Optional[str] = None
    mission_sources: List[str] = Field(default_factory=list)  # URLs cited for mission-level facts


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_artemis_info() -> str:
    return """
    Extract structured information about NASA's Artemis II mission as presented in the answer.

    Fields to extract:
    - mission_name: The mission name as stated in the answer (e.g., "Artemis II").
    - crew: An array of all crew members mentioned in the answer (keep the original order).
        For each crew member, extract:
            • name
            • role (e.g., commander, pilot, mission specialist)
            • agency (NASA or CSA)
            • bachelors: list of objects with:
                - institution (university/college name for each bachelor’s degree mentioned)
                - degree (optional textual degree name or major, if present)
            • masters: list of objects with:
                - institution (university/college name for each master’s degree mentioned)
                - degree (optional textual degree name or major, if present)
            • sources: list of URLs that the answer explicitly cites for this crew member’s information
              (bios, official agency pages, university pages, etc.). Only include URLs explicitly present in the answer.
    - launch_complex: The launch site/complex as written (e.g., "Kennedy Space Center Launch Complex 39B" or "LC-39B").
    - earliest_launch_date: The earliest possible launch date exactly as stated in the answer (e.g., "April 1, 2026" or "NET April 2026").
    - rocket_system: The rocket system name (e.g., "Space Launch System (SLS)").
    - spacecraft: The spacecraft name (e.g., "Orion").
    - mission_duration: The mission duration phrasing (e.g., "about 10 days", "approximately ten days").
    - mission_sources: list of URLs expressly cited for mission-level facts (launch complex, date, rocket, spacecraft, duration).
      Only include URLs explicitly present in the answer. Prefer official sources (nasa.gov, csa-asc.gc.ca), if provided.

    URL handling rules:
    - Extract only URLs explicitly present in the answer. Do not fabricate URLs.
    - Include full URLs with protocol (prepend http:// if missing).
    - Keep sources organized: member.sources for person-level facts; mission_sources for mission-level facts.

    If any field is missing in the answer, set it to null (for singular fields) or an empty list (for list fields).
    """


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _contains_any(s: str, keywords: List[str]) -> bool:
    s_norm = _norm(s)
    return any(k.lower() in s_norm for k in keywords)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _has_official_source(urls: List[str], allowed_domains: List[str]) -> bool:
    for u in urls:
        d = _domain(u)
        if any(ad in d for ad in allowed_domains):
            return True
    return False


def _find_member_by_name(crew: List[CrewMember], expected_name: str) -> Optional[CrewMember]:
    en = _norm(expected_name)
    # exact-like match first
    for m in crew:
        if en and en in _norm(m.name):
            return m
    # fallback: return None
    return None


def _find_member_by_role(crew: List[CrewMember], expected_role_keywords: List[str]) -> Optional[CrewMember]:
    for m in crew:
        if _contains_any(m.role or "", expected_role_keywords):
            return m
    return None


def _institutions_from_degrees(deg_list: List[DegreeInfo]) -> List[str]:
    return [d.institution for d in deg_list if d.institution]


def _any_institution_matches(institutions: List[str], expected_keywords: List[str]) -> bool:
    if not institutions:
        return False
    for inst in institutions:
        if _contains_any(inst, expected_keywords):
            return True
    return False


# -----------------------------------------------------------------------------
# Verification builders
# -----------------------------------------------------------------------------
async def verify_mission_identification(evaluator: Evaluator, parent, ex: ArtemisExtraction):
    node = evaluator.add_parallel(
        id="Mission_Identified_As_Artemis_II",
        desc="Response clearly identifies the mission as Artemis II",
        parent=parent,
        critical=True,
    )

    # 1) Match expected mission name
    mission_name = ex.mission_name or ""
    leaf_match = evaluator.add_leaf(
        id="mission_name_match_expected",
        desc="Mission name in the answer matches 'Artemis II'",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The mission name '{mission_name}' refers to NASA's 'Artemis II' mission. Treat 'Artemis II', 'Artemis 2', and minor punctuation/casing variants as equivalent.",
        node=leaf_match,
        additional_instruction="Use tolerant matching for roman numerals vs digits and hyphenation/punctuation variants."
    )

    # 2) Supported by sources (prefer mission_sources)
    leaf_src = evaluator.add_leaf(
        id="mission_name_supported_by_sources",
        desc="Artemis II identification is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This page is an official page about NASA's Artemis II mission.",
        node=leaf_src,
        sources=ex.mission_sources,
        additional_instruction="Accept NASA official Artemis II pages, press releases, or mission overviews that clearly indicate the mission name."
    )


async def verify_crew_count(evaluator: Evaluator, parent, ex: ArtemisExtraction):
    # Per framework guidance, if the answer has more, we filter to first four for downstream checks;
    # but this node ensures the provided answer overall aimed for exactly four crew.
    result = len(ex.crew[:4]) == 4
    evaluator.add_custom_node(
        result=result,
        id="Crew_Count_Exactly_4",
        desc="Response specifies exactly four crew members (no missing and no additional crew beyond the four)",
        parent=parent,
        critical=True
    )


async def _verify_single_crew_role(
    evaluator: Evaluator,
    parent,
    ex: ArtemisExtraction,
    expected_name: str,
    role_keywords: List[str],
    expected_agency: str,
    node_id: str,
    node_desc: str,
):
    # Build a sequential node: (1) answer lists correct mapping; (2) sources support it
    role_node = evaluator.add_sequential(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=True
    )

    member = _find_member_by_name(ex.crew, expected_name) or _find_member_by_role(ex.crew, role_keywords)
    name_ok = member is not None and member.name is not None and _norm(expected_name) in _norm(member.name)
    role_ok = member is not None and member.role is not None and _contains_any(member.role, role_keywords)
    agency_ok = member is not None and member.agency is not None and _contains_any(member.agency, [expected_agency])

    evaluator.add_custom_node(
        result=bool(name_ok and role_ok and agency_ok),
        id=f"{node_id}_answer_has_mapping",
        desc=f"Answer lists {expected_name} with correct role and agency",
        parent=role_node,
        critical=True
    )

    # Source support
    leaf_support = evaluator.add_leaf(
        id=f"{node_id}_source_supported",
        desc=f"Official sources support that {expected_name} is {role_keywords[0]} and affiliated with {expected_agency}",
        parent=role_node,
        critical=True
    )

    sources = member.sources if member and member.sources else []
    claim = f"{expected_name} is the {role_keywords[0]} for NASA's Artemis II mission and is a {expected_agency} astronaut."
    await evaluator.verify(
        claim=claim,
        node=leaf_support,
        sources=sources,
        additional_instruction="Verify the crew role and agency on official sources (NASA or CSA). Allow minor variants like 'mission commander' vs 'commander'."
    )


async def verify_crew_roles_affiliations(evaluator: Evaluator, parent, ex: ArtemisExtraction):
    crew_node = evaluator.add_parallel(
        id="Crew_Roles_And_Affiliations",
        desc="Crew names, roles, and space agency affiliations are correct",
        parent=parent,
        critical=True
    )

    # Commander - Reid Wiseman (NASA)
    await _verify_single_crew_role(
        evaluator, crew_node, ex,
        expected_name="Reid Wiseman",
        role_keywords=["commander", "mission commander"],
        expected_agency="nasa",
        node_id="Mission_Commander",
        node_desc="Identifies Reid Wiseman as mission commander and as affiliated with NASA",
    )

    # Pilot - Victor Glover (NASA)
    await _verify_single_crew_role(
        evaluator, crew_node, ex,
        expected_name="Victor Glover",
        role_keywords=["pilot"],
        expected_agency="nasa",
        node_id="Mission_Pilot",
        node_desc="Identifies Victor Glover as the pilot and as affiliated with NASA",
    )

    # Mission Specialist - Christina Koch (NASA)
    await _verify_single_crew_role(
        evaluator, crew_node, ex,
        expected_name="Christina Koch",
        role_keywords=["mission specialist", "specialist"],
        expected_agency="nasa",
        node_id="Mission_Specialist_NASA",
        node_desc="Identifies Christina Koch as a mission specialist and as affiliated with NASA",
    )

    # Mission Specialist - Jeremy Hansen (CSA)
    await _verify_single_crew_role(
        evaluator, crew_node, ex,
        expected_name="Jeremy Hansen",
        role_keywords=["mission specialist", "specialist"],
        expected_agency="csa",
        node_id="Mission_Specialist_CSA",
        node_desc="Identifies Jeremy Hansen as a mission specialist and as affiliated with the Canadian Space Agency (CSA)",
    )


async def _verify_degree_expected(
    evaluator: Evaluator,
    parent,
    ex: ArtemisExtraction,
    person_name: str,
    degree_level: str,  # 'bachelor' or 'master'
    expected_institution: str,
    node_id: str,
    node_desc: str,
):
    # Sequential: (1) answer includes expected institution; (2) official sources support it
    deg_node = evaluator.add_sequential(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=True
    )

    member = _find_member_by_name(ex.crew, person_name)
    if degree_level == "bachelor":
        insts = _institutions_from_degrees(member.bachelors if member else [])
    else:
        insts = _institutions_from_degrees(member.masters if member else [])

    # Match expected institution (allow common short names)
    expected_keywords = [expected_institution]
    # Add common alias keywords
    aliases = {
        "Rensselaer Polytechnic Institute": ["rpi", "rensselaer"],
        "Johns Hopkins University": ["johns hopkins", "jhu"],
        "California Polytechnic State University": ["california polytechnic", "cal poly", "cal poly san luis obispo", "california polytechnic state university, san luis obispo"],
        "North Carolina State University": ["ncsu", "north carolina state"],
        "Royal Military College of Canada": ["rmc", "royal military college"],
    }
    if expected_institution in aliases:
        expected_keywords.extend(aliases[expected_institution])

    match_ok = _any_institution_matches(insts, expected_keywords)

    evaluator.add_custom_node(
        result=bool(match_ok),
        id=f"{node_id}_match_expected",
        desc=f"Answer lists {person_name}'s {degree_level} institution as {expected_institution}",
        parent=deg_node,
        critical=True
    )

    # Source support from that member's sources
    leaf_support = evaluator.add_leaf(
        id=f"{node_id}_source_supported",
        desc=f"Official sources support {person_name}'s {degree_level} institution",
        parent=deg_node,
        critical=True
    )
    sources = member.sources if member and member.sources else []
    claim = f"{person_name} earned a {degree_level}'s degree from {expected_institution}."
    await evaluator.verify(
        claim=claim,
        node=leaf_support,
        sources=sources,
        additional_instruction="Prefer NASA/CSA official bios or the official university pages that list degrees."
    )


async def _verify_master_info_generic_for_pilot(
    evaluator: Evaluator, parent, ex: ArtemisExtraction
):
    """
    For Victor Glover: verify correctness of master's-degree info present in the answer.
    If the answer lists master(s), verify via sources that he has those master’s degrees from the stated institutions.
    """
    node = evaluator.add_sequential(
        id="Pilot_Master_Degree_Info_Correct_If_Applicable",
        desc="Victor Glover master’s degree information (whether he has one, and the granting institution if he does) is correct",
        parent=parent,
        critical=True
    )

    member = _find_member_by_name(ex.crew, "Victor Glover")
    masters_insts = _institutions_from_degrees(member.masters if member else [])
    has_masters_in_answer = len(masters_insts) > 0

    # We expect he DOES have master's degrees; ensure the answer did not claim otherwise.
    evaluator.add_custom_node(
        result=has_masters_in_answer,
        id="pilot_master_presence_in_answer",
        desc="Answer lists at least one master's degree for Victor Glover (he does hold master's degrees)",
        parent=node,
        critical=True
    )

    # Verify with sources that the master's information is accurate.
    leaf_support = evaluator.add_leaf(
        id="pilot_master_source_supported",
        desc="Official sources support Victor Glover's master's degree information as stated",
        parent=node,
        critical=True
    )
    # Build a conservative claim that can be supported by NASA bios:
    if masters_insts:
        inst_clause = "; ".join(sorted(set(masters_insts)))
        claim = f"Victor Glover holds one or more master's degrees from the following institution(s): {inst_clause}."
    else:
        # If not listed, make a falsifiable claim (should fail against official pages that list his master's degrees)
        claim = "Victor Glover does not hold any master's degree."

    await evaluator.verify(
        claim=claim,
        node=leaf_support,
        sources=member.sources if member and member.sources else [],
        additional_instruction="Check official NASA bio or other official pages listing Victor Glover’s education."
    )


async def verify_education_backgrounds(evaluator: Evaluator, parent, ex: ArtemisExtraction):
    edu_node = evaluator.add_parallel(
        id="Education_Backgrounds",
        desc="Educational background requirements (bachelor’s for all; master’s where applicable) are satisfied",
        parent=parent,
        critical=True
    )

    # Commander - Wiseman: Bachelor (RPI) and Master (JHU)
    await _verify_degree_expected(
        evaluator, edu_node, ex,
        person_name="Reid Wiseman",
        degree_level="bachelor",
        expected_institution=EXPECTED["education"]["Reid Wiseman"]["bachelor_institution"],
        node_id="Commander_Bachelor_Institution",
        node_desc="Reid Wiseman bachelor's institution is Rensselaer Polytechnic Institute",
    )
    await _verify_degree_expected(
        evaluator, edu_node, ex,
        person_name="Reid Wiseman",
        degree_level="master",
        expected_institution=EXPECTED["education"]["Reid Wiseman"]["master_institution"],
        node_id="Commander_Master_Institution",
        node_desc="Reid Wiseman master's institution is Johns Hopkins University",
    )

    # Pilot - Glover: Bachelor (Cal Poly); Masters info generic correctness
    await _verify_degree_expected(
        evaluator, edu_node, ex,
        person_name="Victor Glover",
        degree_level="bachelor",
        expected_institution=EXPECTED["education"]["Victor Glover"]["bachelor_institution"],
        node_id="Pilot_Bachelor_Institution",
        node_desc="Victor Glover bachelor's institution is California Polytechnic State University",
    )
    await _verify_master_info_generic_for_pilot(evaluator, edu_node, ex)

    # Koch - Bachelor(s) NCSU, Master NCSU
    await _verify_degree_expected(
        evaluator, edu_node, ex,
        person_name="Christina Koch",
        degree_level="bachelor",
        expected_institution=EXPECTED["education"]["Christina Koch"]["bachelor_institution"],
        node_id="Koch_Bachelor_Institution",
        node_desc="Christina Koch bachelor's institution(s) are North Carolina State University",
    )
    await _verify_degree_expected(
        evaluator, edu_node, ex,
        person_name="Christina Koch",
        degree_level="master",
        expected_institution=EXPECTED["education"]["Christina Koch"]["master_institution"],
        node_id="Koch_Master_Institution",
        node_desc="Christina Koch master's institution is North Carolina State University",
    )

    # Hansen - Bachelor RMC, Master RMC
    await _verify_degree_expected(
        evaluator, edu_node, ex,
        person_name="Jeremy Hansen",
        degree_level="bachelor",
        expected_institution=EXPECTED["education"]["Jeremy Hansen"]["bachelor_institution"],
        node_id="Hansen_Bachelor_Institution",
        node_desc="Jeremy Hansen bachelor's institution is Royal Military College of Canada",
    )
    await _verify_degree_expected(
        evaluator, edu_node, ex,
        person_name="Jeremy Hansen",
        degree_level="master",
        expected_institution=EXPECTED["education"]["Jeremy Hansen"]["master_institution"],
        node_id="Hansen_Master_Institution",
        node_desc="Jeremy Hansen master's institution is Royal Military College of Canada",
    )


async def verify_mission_specs(evaluator: Evaluator, parent, ex: ArtemisExtraction):
    # Launch Site
    launch_node = evaluator.add_sequential(
        id="Launch_Site",
        desc="Identifies Kennedy Space Center Launch Complex 39B as the launch complex",
        parent=parent,
        critical=True
    )
    lc_match = evaluator.add_custom_node(
        result=_contains_any(ex.launch_complex or "", ["kennedy space center launch complex 39b", "lc-39b", "launch complex 39b"]),
        id="launch_site_match_expected",
        desc="Answer lists launch site as KSC LC-39B (allow LC-39B variants)",
        parent=launch_node,
        critical=True
    )
    leaf_lc_src = evaluator.add_leaf(
        id="launch_site_supported",
        desc="Official sources support that Artemis II launches from LC-39B",
        parent=launch_node,
        critical=True
    )
    await evaluator.verify(
        claim="Artemis II will launch from Launch Complex 39B (LC-39B) at NASA's Kennedy Space Center.",
        node=leaf_lc_src,
        sources=ex.mission_sources,
        additional_instruction="Prefer NASA official Artemis II pages or press releases mentioning LC-39B."
    )

    # Earliest launch date
    date_node = evaluator.add_sequential(
        id="Earliest_Launch_Date",
        desc="Identifies April 1, 2026 as the earliest possible launch date",
        parent=parent,
        critical=True
    )
    date_match = evaluator.add_custom_node(
        result=_contains_any(ex.earliest_launch_date or "", ["april 1, 2026", "1 april 2026"]),
        id="earliest_date_match_expected",
        desc="Answer lists the earliest possible launch date as April 1, 2026",
        parent=date_node,
        critical=True
    )
    leaf_date_src = evaluator.add_leaf(
        id="earliest_date_supported",
        desc="Official sources support the stated earliest possible launch date",
        parent=date_node,
        critical=True
    )
    await evaluator.verify(
        claim="The earliest possible launch date for Artemis II is April 1, 2026 (or 'no earlier than' April 1, 2026).",
        node=leaf_date_src,
        sources=ex.mission_sources,
        additional_instruction="Confirm with official NASA schedules or press releases; accept 'no earlier than April 1, 2026' phrasing."
    )

    # Rocket System
    rocket_node = evaluator.add_sequential(
        id="Rocket_System",
        desc="Identifies Space Launch System (SLS) as the rocket system",
        parent=parent,
        critical=True
    )
    rocket_match = evaluator.add_custom_node(
        result=_contains_any(ex.rocket_system or "", ["space launch system", "sls"]),
        id="rocket_system_match_expected",
        desc="Answer lists the rocket system as Space Launch System (SLS)",
        parent=rocket_node,
        critical=True
    )
    leaf_rocket_src = evaluator.add_leaf(
        id="rocket_system_supported",
        desc="Official sources support that Artemis II uses SLS",
        parent=rocket_node,
        critical=True
    )
    await evaluator.verify(
        claim="Artemis II will use NASA's Space Launch System (SLS) rocket.",
        node=leaf_rocket_src,
        sources=ex.mission_sources,
        additional_instruction="Verify on NASA Artemis pages that the rocket is SLS."
    )

    # Spacecraft
    craft_node = evaluator.add_sequential(
        id="Spacecraft",
        desc="Identifies Orion as the spacecraft",
        parent=parent,
        critical=True
    )
    craft_match = evaluator.add_custom_node(
        result=_contains_any(ex.spacecraft or "", ["orion"]),
        id="spacecraft_match_expected",
        desc="Answer lists the spacecraft as Orion",
        parent=craft_node,
        critical=True
    )
    leaf_craft_src = evaluator.add_leaf(
        id="spacecraft_supported",
        desc="Official sources support that Artemis II will fly Orion",
        parent=craft_node,
        critical=True
    )
    await evaluator.verify(
        claim="Artemis II will use the Orion spacecraft.",
        node=leaf_craft_src,
        sources=ex.mission_sources,
        additional_instruction="Confirm on official NASA Artemis pages that the spacecraft is Orion."
    )

    # Mission Duration
    dur_node = evaluator.add_sequential(
        id="Mission_Duration",
        desc="Identifies mission duration as approximately 10 days",
        parent=parent,
        critical=True
    )
    dur_match = evaluator.add_custom_node(
        result=_contains_any(ex.mission_duration or "", ["10 day", "ten day"]),
        id="duration_match_expected",
        desc="Answer lists the duration as approximately 10 days (allow 'ten days' variants)",
        parent=dur_node,
        critical=True
    )
    leaf_dur_src = evaluator.add_leaf(
        id="duration_supported",
        desc="Official sources support the ~10 days mission duration",
        parent=dur_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Artemis II mission duration is approximately 10 days.",
        node=leaf_dur_src,
        sources=ex.mission_sources,
        additional_instruction="Accept approximate phrasing like 'about 10 days' on official NASA Artemis pages."
    )


async def verify_official_source_verifiability(evaluator: Evaluator, parent, ex: ArtemisExtraction):
    off_node = evaluator.add_parallel(
        id="Official_Source_Verifiability",
        desc="Claims are supported by official sources (as required by the question)",
        parent=parent,
        critical=True
    )

    nasa_domains = ["nasa.gov"]
    csa_domains = ["asc-csa.gc.ca", "csa-asc.gc.ca"]  # bilingual domain variants
    uni_domains = [
        "rpi.edu", "jhu.edu", "ncsu.edu",
        "rmc.ca", "rmc-cmr.ca", "forces.gc.ca"
    ]

    # Crew + roles official sources: ensure each member has at least one NASA/CSA official source
    crew_ok = True
    for m in ex.crew[:4]:
        if not m or not m.sources:
            crew_ok = False
            break
        allowed = nasa_domains + csa_domains
        if not _has_official_source(m.sources, allowed):
            crew_ok = False
            break

    evaluator.add_custom_node(
        result=crew_ok,
        id="Official_Sources_For_Crew_And_Roles",
        desc="Crew names/roles/agency affiliations are supported via official sources (e.g., NASA/CSA official pages or official press releases)",
        parent=off_node,
        critical=True
    )

    # Education official sources: require NASA/CSA bios or official university sources present for each person’s degrees
    edu_ok = True
    for person_name in ["Reid Wiseman", "Victor Glover", "Christina Koch", "Jeremy Hansen"]:
        m = _find_member_by_name(ex.crew, person_name)
        if not m or not m.sources:
            edu_ok = False
            break
        # For education, allow NASA/CSA or university official domains
        allowed = nasa_domains + csa_domains + uni_domains
        if not _has_official_source(m.sources, allowed):
            edu_ok = False
            break

    evaluator.add_custom_node(
        result=edu_ok,
        id="Official_Sources_For_Education",
        desc="Each stated degree institution is supported via official sources (e.g., NASA/CSA bios, official university/agency bios), not solely non-official aggregators",
        parent=off_node,
        critical=True
    )

    # Mission specs official sources: mission_sources should include NASA
    mission_ok = _has_official_source(ex.mission_sources, nasa_domains)
    evaluator.add_custom_node(
        result=mission_ok,
        id="Official_Sources_For_Mission_Specs",
        desc="Launch complex/date, rocket system, spacecraft, and mission duration are supported via official sources (e.g., NASA mission pages/press kits)",
        parent=off_node,
        critical=True
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    # Initialize evaluator (root is non-critical by design)
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

    # Extract structured info
    ex: ArtemisExtraction = await evaluator.extract(
        prompt=prompt_extract_artemis_info(),
        template_class=ArtemisExtraction,
        extraction_name="artemis_extraction"
    )

    # Normalize: keep at most 4 crew entries for downstream checks (per framework guidance)
    if len(ex.crew) > 4:
        ex.crew = ex.crew[:4]

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_mission_name": EXPECTED["mission_name"],
        "expected_crew": EXPECTED["crew"],
        "expected_education": EXPECTED["education"],
        "expected_launch_complex": EXPECTED["launch_complex"],
        "expected_earliest_launch_date": EXPECTED["earliest_launch_date"],
        "expected_rocket_system": EXPECTED["rocket_system"],
        "expected_spacecraft": EXPECTED["spacecraft"],
        "expected_mission_duration_approx_days": EXPECTED["mission_duration_approx_days"]
    }, gt_type="expected_values")

    # Build rubric root as a critical parallel node
    artemis_root = evaluator.add_parallel(
        id="Artemis_II_Mission_Information",
        desc="Comprehensive information about NASA's Artemis II mission (2026) including crew, education, mission specs, and official-source verifiability",
        parent=root,
        critical=True
    )

    # 1) Mission identified as Artemis II
    await verify_mission_identification(evaluator, artemis_root, ex)

    # 2) Crew count exactly 4
    await verify_crew_count(evaluator, artemis_root, ex)

    # 3) Crew roles and affiliations
    await verify_crew_roles_affiliations(evaluator, artemis_root, ex)

    # 4) Education backgrounds
    await verify_education_backgrounds(evaluator, artemis_root, ex)

    # 5-9) Mission specs (launch site, date, rocket, spacecraft, duration)
    await verify_mission_specs(evaluator, artemis_root, ex)

    # 10) Official source verifiability checks
    await verify_official_source_verifiability(evaluator, artemis_root, ex)

    # Return structured summary
    return evaluator.get_summary()