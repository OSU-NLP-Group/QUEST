import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "four_us_districts_compliance"
TASK_DESCRIPTION = """
Identify four public school districts, each from a different U.S. state (Connecticut, Ohio, Texas, and Minnesota), where each district must meet all of the following comprehensive criteria:

For the Connecticut district:
1. Must conduct exactly 7 fire drills per school year as required by state law
2. Must conduct exactly 3 crisis response drills per school year as required by state law
3. Must have documented emergency operations procedures
4. Must provide special education services under IDEA Part B for students ages 3-21

For the Ohio district:
1. Must have at least one high school competing in OHSAA Division I football, which requires an adjusted enrollment of 592 or more students in grades 9-11 for the 2025-26 school year
2. Must report official enrollment figures to OHSAA
3. Must provide special education services under IDEA Part B for students ages 3-21
4. Must have elected school board members

For the Texas district:
1. Must have schools participating in UIL athletic competitions
2. Must provide special education services under IDEA Part B for students ages 3-21
3. Must have had a school board election in 2025
4. Must have multiple board member positions (places/seats) on the board of trustees

For the Minnesota district:
1. Must have a student enrollment of at least 35,000 students
2. Must have had teacher contract negotiations that concluded with a ratified agreement in 2026
3. Must provide special education services under IDEA Part B for students ages 3-21
4. Must conduct required emergency safety drills

For each of the four districts identified, provide: (1) the official district name, (2) the specific state, (3) verification of how each requirement is met, and (4) official reference URLs documenting each criterion.
"""


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _normalize_list(lst: Optional[List[str]]) -> List[str]:
    return [u for u in (lst or []) if isinstance(u, str) and u.strip()]


def _state_matches(extracted: Optional[str], expected_full: str, expected_abbr: str) -> bool:
    if not extracted:
        return False
    val = extracted.strip().lower()
    return val in {
        expected_full.lower(),
        expected_abbr.lower(),
        f"state of {expected_full.lower()}",
        f"{expected_full.lower()} (usa)",
        f"{expected_abbr.lower()}.",
    }


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CTDistrict(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    fire_drills_urls: List[str] = Field(default_factory=list)
    crisis_drills_urls: List[str] = Field(default_factory=list)
    emergency_procedures_urls: List[str] = Field(default_factory=list)
    special_education_urls: List[str] = Field(default_factory=list)


class OHDistrict(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    d1_school_name: Optional[str] = None
    ohsaa_division_urls: List[str] = Field(default_factory=list)        # Division I assignment pages for 2025-26 football
    ohsaa_enrollment_urls: List[str] = Field(default_factory=list)      # Pages listing school adjusted enrollments (grades 9–11)
    enrollment_reporting_urls: List[str] = Field(default_factory=list)  # Evidence district/schools report enrollment to OHSAA
    special_education_urls: List[str] = Field(default_factory=list)
    elected_board_urls: List[str] = Field(default_factory=list)


class TXDistrict(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    uil_urls: List[str] = Field(default_factory=list)
    special_education_urls: List[str] = Field(default_factory=list)
    board_election_2025_urls: List[str] = Field(default_factory=list)
    board_structure_urls: List[str] = Field(default_factory=list)  # pages showing multiple places/seats


class MNDistrict(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)
    contract_urls: List[str] = Field(default_factory=list)  # bargaining updates and ratification in 2026
    special_education_urls: List[str] = Field(default_factory=list)
    drills_urls: List[str] = Field(default_factory=list)


class FourDistrictsExtraction(BaseModel):
    connecticut: Optional[CTDistrict] = None
    ohio: Optional[OHDistrict] = None
    texas: Optional[TXDistrict] = None
    minnesota: Optional[MNDistrict] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    Extract exactly one public school district for each of the following states as presented in the answer: Connecticut, Ohio, Texas, and Minnesota.
    For each state, extract the official district name, the state string as written, and the URLs cited in the answer that document each required criterion. Only include URLs that are explicitly present in the answer (plain or markdown links). If a specific URL is not provided, leave the corresponding list empty.

    Return a JSON object with the following structure and fields:

    {
      "connecticut": {
        "name": str|null,
        "state": str|null,
        "fire_drills_urls": [url...],              // URLs that the answer uses to support "exactly 7 fire drills annually" (CGS §10-231)
        "crisis_drills_urls": [url...],            // URLs that support "exactly 3 crisis response drills annually" (CGS §10-231)
        "emergency_procedures_urls": [url...],     // URLs to documented emergency operations procedures (EOP, safety plan, etc.)
        "special_education_urls": [url...]         // URLs supporting IDEA Part B services ages 3-21
      },
      "ohio": {
        "name": str|null,
        "state": str|null,
        "d1_school_name": str|null,                // a high school in the district that the answer claims is OHSAA Division I for football in 2025-26
        "ohsaa_division_urls": [url...],           // URLs to OHSAA pages/lists showing 2025-26 football Division I assignments
        "ohsaa_enrollment_urls": [url...],         // URLs listing adjusted enrollments for grades 9-11 by school (2025-26)
        "enrollment_reporting_urls": [url...],     // URLs evidencing district/schools report enrollment to OHSAA
        "special_education_urls": [url...],        // URLs supporting IDEA Part B services ages 3-21
        "elected_board_urls": [url...]             // URLs showing the district has elected board members
      },
      "texas": {
        "name": str|null,
        "state": str|null,
        "uil_urls": [url...],                      // URLs evidencing participation in UIL athletics
        "special_education_urls": [url...],        // URLs supporting IDEA Part B services ages 3-21
        "board_election_2025_urls": [url...],      // URLs evidencing a school board election in 2025
        "board_structure_urls": [url...]           // URLs showing multiple board places/seats
      },
      "minnesota": {
        "name": str|null,
        "state": str|null,
        "enrollment_urls": [url...],               // URLs evidencing total enrollment >= 35,000 students
        "contract_urls": [url...],                 // URLs evidencing teacher negotiations and ratified agreement in 2026
        "special_education_urls": [url...],        // URLs supporting IDEA Part B services ages 3-21
        "drills_urls": [url...]                    // URLs evidencing required emergency safety drills conducted
      }
    }

    Rules:
    - Do not invent URLs; extract only those explicitly present in the answer.
    - If multiple districts are mentioned for a state, pick the first one and its matching URLs.
    - Normalize any URLs missing a protocol by prepending http://
    - If any field is not mentioned in the answer, set it to null or [] accordingly.
    """


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_connecticut_subtree(evaluator: Evaluator, parent, ct: Optional[CTDistrict]) -> None:
    # Node: Connecticut district identification and verification (sequential)
    ct_root = evaluator.add_sequential(
        id="connecticut_district",
        desc="Connecticut district identification and verification",
        parent=parent,
        critical=False
    )

    # Leaf: identification (critical)
    ct_identified = evaluator.add_custom_node(
        result=(
            ct is not None and
            isinstance(ct.name, str) and ct.name.strip() != "" and
            _state_matches(ct.state, "Connecticut", "CT")
        ),
        id="ct_district_identification",
        desc="Identified a specific public school district in Connecticut",
        parent=ct_root,
        critical=True
    )

    # Parallel critical requirements
    ct_reqs = evaluator.add_parallel(
        id="ct_requirements",
        desc="Connecticut district meets all state-specific and federal requirements",
        parent=ct_root,
        critical=True
    )

    # 1) Fire drills exactly 7 per year (sequential)
    ct_fire_seq = evaluator.add_sequential(
        id="ct_fire_drills",
        desc="District conducts exactly 7 fire drills per school year as required by CGS §10-231",
        parent=ct_reqs,
        critical=True
    )

    # 1a) requirement verification (leaf)
    node_ct_fire_req = evaluator.add_leaf(
        id="ct_fire_drills_requirement",
        desc="Verified the district conducts 7 fire drills annually",
        parent=ct_fire_seq,
        critical=True
    )
    fire_urls = _normalize_list(ct.fire_drills_urls if ct else [])
    fire_claim = f"The district '{(ct.name if ct else 'the district')}' states that exactly 7 fire drills are conducted each school year (as required by Connecticut law CGS §10-231)."
    await evaluator.verify(
        claim=fire_claim,
        node=node_ct_fire_req,
        sources=fire_urls,
        additional_instruction="Look for explicit mention of '7 fire drills' per school year or a direct citation to CGS §10-231 specifying seven fire drills."
    )

    # 1b) url presence (leaf as custom existence)
    evaluator.add_custom_node(
        result=len(fire_urls) > 0,
        id="ct_fire_drills_url",
        desc="Provided official URL documenting the fire drill requirement or policy",
        parent=ct_fire_seq,
        critical=True
    )

    # 2) Crisis response drills exactly 3 per year (sequential)
    ct_crisis_seq = evaluator.add_sequential(
        id="ct_crisis_drills",
        desc="District conducts exactly 3 crisis response drills per school year as required by CGS §10-231",
        parent=ct_reqs,
        critical=True
    )

    node_ct_crisis_req = evaluator.add_leaf(
        id="ct_crisis_drills_requirement",
        desc="Verified the district conducts 3 crisis response drills annually",
        parent=ct_crisis_seq,
        critical=True
    )
    crisis_urls = _normalize_list(ct.crisis_drills_urls if ct else [])
    crisis_claim = f"The district '{(ct.name if ct else 'the district')}' conducts exactly 3 crisis response drills each school year (per Connecticut CGS §10-231)."
    await evaluator.verify(
        claim=crisis_claim,
        node=node_ct_crisis_req,
        sources=crisis_urls,
        additional_instruction="Confirm explicit language about '3 crisis response drills' per school year, or a policy page that spells this out."
    )

    evaluator.add_custom_node(
        result=len(crisis_urls) > 0,
        id="ct_crisis_drills_url",
        desc="Provided official URL documenting the crisis response drill requirement or policy",
        parent=ct_crisis_seq,
        critical=True
    )

    # 3) Emergency operations procedures documented (sequential)
    ct_eop_seq = evaluator.add_sequential(
        id="ct_emergency_procedures",
        desc="District has documented emergency operations procedures",
        parent=ct_reqs,
        critical=True
    )

    node_ct_eop_exist = evaluator.add_leaf(
        id="ct_emergency_procedures_existence",
        desc="Verified the district has emergency operations procedures documented",
        parent=ct_eop_seq,
        critical=True
    )
    eop_urls = _normalize_list(ct.emergency_procedures_urls if ct else [])
    eop_claim = f"The district '{(ct.name if ct else 'the district')}' publishes documented emergency operations procedures (e.g., Emergency Operations Plan or School Safety Plan)."
    await evaluator.verify(
        claim=eop_claim,
        node=node_ct_eop_exist,
        sources=eop_urls,
        additional_instruction="Accept EOPs, Emergency Operations Plans, Safety Plans, or similar official documents/pages that clearly describe emergency procedures."
    )

    evaluator.add_custom_node(
        result=len(eop_urls) > 0,
        id="ct_emergency_procedures_url",
        desc="Provided official URL to the emergency procedures documentation",
        parent=ct_eop_seq,
        critical=True
    )

    # 4) Special education services Part B ages 3-21 (sequential)
    ct_sped_seq = evaluator.add_sequential(
        id="ct_special_education",
        desc="District provides special education services under IDEA Part B for ages 3-21",
        parent=ct_reqs,
        critical=True
    )

    node_ct_sped = evaluator.add_leaf(
        id="ct_special_education_services",
        desc="Verified the district provides IDEA Part B services for students ages 3-21",
        parent=ct_sped_seq,
        critical=True
    )
    ct_sped_urls = _normalize_list(ct.special_education_urls if ct else [])
    sped_claim = f"The district '{(ct.name if ct else 'the district')}' provides special education services under IDEA Part B for eligible students ages 3–21."
    await evaluator.verify(
        claim=sped_claim,
        node=node_ct_sped,
        sources=ct_sped_urls,
        additional_instruction="Look for phrases like 'ages 3-21', 'IDEA-B', or equivalent wording indicating special education for students ages 3 through 21."
    )

    evaluator.add_custom_node(
        result=len(ct_sped_urls) > 0,
        id="ct_special_education_url",
        desc="Provided official URL documenting special education services",
        parent=ct_sped_seq,
        critical=True
    )


async def build_ohio_subtree(evaluator: Evaluator, parent, oh: Optional[OHDistrict]) -> None:
    oh_root = evaluator.add_sequential(
        id="ohio_district",
        desc="Ohio district identification and verification",
        parent=parent,
        critical=False
    )

    # Identification (critical)
    evaluator.add_custom_node(
        result=(
            oh is not None and
            isinstance(oh.name, str) and oh.name.strip() != "" and
            _state_matches(oh.state, "Ohio", "OH")
        ),
        id="oh_district_identification",
        desc="Identified a specific public school district in Ohio",
        parent=oh_root,
        critical=True
    )

    oh_reqs = evaluator.add_parallel(
        id="oh_requirements",
        desc="Ohio district meets all state-specific and federal requirements",
        parent=oh_root,
        critical=True
    )

    # 1) OHSAA Division I (sequential)
    oh_div_seq = evaluator.add_sequential(
        id="oh_ohsaa_division_i",
        desc="District has at least one high school in OHSAA Division I football with enrollment of 592+ students in grades 9-11 for 2025-26",
        parent=oh_reqs,
        critical=True
    )

    node_oh_d1_school = evaluator.add_leaf(
        id="oh_division_i_school",
        desc="Identified at least one high school in the district competing in OHSAA Division I football",
        parent=oh_div_seq,
        critical=True
    )
    d1_urls = _normalize_list(oh.ohsaa_division_urls if oh else [])
    d1_claim = f"For the 2025–26 school year, at least one high school from the '{(oh.name if oh else 'district')}' district competes in OHSAA Division I football."
    await evaluator.verify(
        claim=d1_claim,
        node=node_oh_d1_school,
        sources=d1_urls,
        additional_instruction="Look for OHSAA division assignment lists or pages that explicitly show a school from the district in 'Division I' for football in 2025–26."
    )

    node_oh_enroll = evaluator.add_leaf(
        id="oh_enrollment_verification",
        desc="Verified the school meets the Division I enrollment threshold of 592 or more students in grades 9-11",
        parent=oh_div_seq,
        critical=True
    )
    enroll_urls = _normalize_list((oh.ohsaa_enrollment_urls if oh else []) or d1_urls)
    enroll_claim = f"For 2025–26 OHSAA football, Division I requires an adjusted enrollment of at least 592 students in grades 9–11; the Division I school from '{(oh.name if oh else 'the district')}' meets this threshold."
    await evaluator.verify(
        claim=enroll_claim,
        node=node_oh_enroll,
        sources=enroll_urls,
        additional_instruction="Prefer pages that list adjusted enrollments per school. If the school's exact adjusted enrollment is shown and is >=592, that satisfies this check."
    )

    evaluator.add_custom_node(
        result=len(d1_urls) > 0,
        id="oh_ohsaa_url",
        desc="Provided official OHSAA URL documenting the school's Division I classification for 2025-26",
        parent=oh_div_seq,
        critical=True
    )

    # 2) District reports enrollment to OHSAA (sequential)
    oh_rep_seq = evaluator.add_sequential(
        id="oh_enrollment_reporting",
        desc="District reports official enrollment figures to OHSAA",
        parent=oh_reqs,
        critical=True
    )

    node_oh_reporting = evaluator.add_leaf(
        id="oh_enrollment_reported",
        desc="Verified the district reports enrollment data to OHSAA",
        parent=oh_rep_seq,
        critical=True
    )
    rep_urls = _normalize_list(oh.enrollment_reporting_urls if oh else [])
    rep_claim = f"The '{(oh.name if oh else 'district')}' district (or its high schools) reports official enrollment figures to OHSAA for classification purposes."
    await evaluator.verify(
        claim=rep_claim,
        node=node_oh_reporting,
        sources=rep_urls,
        additional_instruction="Accept official OHSAA pages or district/athletics pages that explicitly reference OHSAA-reported enrollments, ADM, or classification enrollment submissions."
    )

    evaluator.add_custom_node(
        result=len(rep_urls) > 0,
        id="oh_enrollment_url",
        desc="Provided official URL documenting enrollment reporting",
        parent=oh_rep_seq,
        critical=True
    )

    # 3) Special education services ages 3–21 (sequential)
    oh_sped_seq = evaluator.add_sequential(
        id="oh_special_education",
        desc="District provides special education services under IDEA Part B for ages 3-21",
        parent=oh_reqs,
        critical=True
    )

    node_oh_sped = evaluator.add_leaf(
        id="oh_special_education_services",
        desc="Verified the district provides IDEA Part B services for students ages 3-21",
        parent=oh_sped_seq,
        critical=True
    )
    oh_sped_urls = _normalize_list(oh.special_education_urls if oh else [])
    oh_sped_claim = f"The '{(oh.name if oh else 'district')}' district provides special education services under IDEA Part B to eligible students ages 3–21."
    await evaluator.verify(
        claim=oh_sped_claim,
        node=node_oh_sped,
        sources=oh_sped_urls,
        additional_instruction="Look for clear language about ages 3 through 21 or 'IDEA-B' on the district's special education pages."
    )

    evaluator.add_custom_node(
        result=len(oh_sped_urls) > 0,
        id="oh_special_education_url",
        desc="Provided official URL documenting special education services",
        parent=oh_sped_seq,
        critical=True
    )

    # 4) Elected school board members (sequential)
    oh_board_seq = evaluator.add_sequential(
        id="oh_elected_board",
        desc="District has elected school board members",
        parent=oh_reqs,
        critical=True
    )

    node_oh_board = evaluator.add_leaf(
        id="oh_board_elected",
        desc="Verified the district has elected school board members",
        parent=oh_board_seq,
        critical=True
    )
    board_urls = _normalize_list(oh.elected_board_urls if oh else [])
    board_claim = f"The board of education for '{(oh.name if oh else 'the district')}' is composed of elected members (not appointed)."
    await evaluator.verify(
        claim=board_claim,
        node=node_oh_board,
        sources=board_urls,
        additional_instruction="Accept official district governance/board pages that explicitly say 'elected' or show election terms/vote history."
    )

    evaluator.add_custom_node(
        result=len(board_urls) > 0,
        id="oh_board_url",
        desc="Provided official URL documenting the elected school board structure",
        parent=oh_board_seq,
        critical=True
    )


async def build_texas_subtree(evaluator: Evaluator, parent, tx: Optional[TXDistrict]) -> None:
    tx_root = evaluator.add_sequential(
        id="texas_district",
        desc="Texas district identification and verification",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=(
            tx is not None and
            isinstance(tx.name, str) and tx.name.strip() != "" and
            _state_matches(tx.state, "Texas", "TX")
        ),
        id="tx_district_identification",
        desc="Identified a specific public school district in Texas",
        parent=tx_root,
        critical=True
    )

    tx_reqs = evaluator.add_parallel(
        id="tx_requirements",
        desc="Texas district meets all state-specific and federal requirements",
        parent=tx_root,
        critical=True
    )

    # 1) UIL participation (sequential)
    tx_uil_seq = evaluator.add_sequential(
        id="tx_uil_participation",
        desc="District has schools participating in UIL athletic competitions",
        parent=tx_reqs,
        critical=True
    )

    node_tx_uil = evaluator.add_leaf(
        id="tx_uil_athletics",
        desc="Verified the district participates in UIL athletics",
        parent=tx_uil_seq,
        critical=True
    )
    tx_uil_urls = _normalize_list(tx.uil_urls if tx else [])
    tx_uil_claim = f"The '{(tx.name if tx else 'district')}' district (or its schools) participates in UIL athletic competitions."
    await evaluator.verify(
        claim=tx_uil_claim,
        node=node_tx_uil,
        sources=tx_uil_urls,
        additional_instruction="Accept UIL district alignment pages, school athletics pages referencing UIL, or UIL results pages showing the district's schools."
    )

    evaluator.add_custom_node(
        result=len(tx_uil_urls) > 0,
        id="tx_uil_url",
        desc="Provided official URL documenting UIL athletic participation",
        parent=tx_uil_seq,
        critical=True
    )

    # 2) Special education services ages 3–21 (sequential)
    tx_sped_seq = evaluator.add_sequential(
        id="tx_special_education",
        desc="District provides special education services under IDEA Part B for ages 3-21",
        parent=tx_reqs,
        critical=True
    )

    node_tx_sped = evaluator.add_leaf(
        id="tx_special_education_services",
        desc="Verified the district provides IDEA Part B services for students ages 3-21",
        parent=tx_sped_seq,
        critical=True
    )
    tx_sped_urls = _normalize_list(tx.special_education_urls if tx else [])
    tx_sped_claim = f"The '{(tx.name if tx else 'district')}' district provides special education services under IDEA Part B for eligible students ages 3–21."
    await evaluator.verify(
        claim=tx_sped_claim,
        node=node_tx_sped,
        sources=tx_sped_urls,
        additional_instruction="Look for 'ages 3–21', 'IDEA-B', or equivalent statements on the special education pages."
    )

    evaluator.add_custom_node(
        result=len(tx_sped_urls) > 0,
        id="tx_special_education_url",
        desc="Provided official URL documenting special education services",
        parent=tx_sped_seq,
        critical=True
    )

    # 3) Board election in 2025 (sequential)
    tx_elect_seq = evaluator.add_sequential(
        id="tx_board_election_2025",
        desc="District had a school board election in 2025",
        parent=tx_reqs,
        critical=True
    )

    node_tx_election = evaluator.add_leaf(
        id="tx_election_held",
        desc="Verified the district held a school board election in 2025",
        parent=tx_elect_seq,
        critical=True
    )
    tx_elect_urls = _normalize_list(tx.board_election_2025_urls if tx else [])
    tx_elect_claim = f"The '{(tx.name if tx else 'district')}' district held a school board election in 2025."
    await evaluator.verify(
        claim=tx_elect_claim,
        node=node_tx_election,
        sources=tx_elect_urls,
        additional_instruction="Accept district notices, board minutes, canvass reports, or county election pages explicitly showing a 2025 board election for this district."
    )

    evaluator.add_custom_node(
        result=len(tx_elect_urls) > 0,
        id="tx_election_url",
        desc="Provided official URL documenting the 2025 school board election",
        parent=tx_elect_seq,
        critical=True
    )

    # 4) Multiple board positions/places (sequential)
    tx_boardpos_seq = evaluator.add_sequential(
        id="tx_multiple_board_positions",
        desc="District has multiple board member positions (places/seats) on the board of trustees",
        parent=tx_reqs,
        critical=True
    )

    node_tx_positions = evaluator.add_leaf(
        id="tx_board_positions",
        desc="Verified the district has multiple board member positions or places",
        parent=tx_boardpos_seq,
        critical=True
    )
    tx_boardpos_urls = _normalize_list(tx.board_structure_urls if tx else [])
    tx_boardpos_claim = f"The '{(tx.name if tx else 'district')}' board of trustees uses multiple numbered positions/places (e.g., Place 1, Place 2, etc.)."
    await evaluator.verify(
        claim=tx_boardpos_claim,
        node=node_tx_positions,
        sources=tx_boardpos_urls,
        additional_instruction="Accept district governance/board pages that list trustees by distinct numbered places or seats."
    )

    evaluator.add_custom_node(
        result=len(tx_boardpos_urls) > 0,
        id="tx_board_structure_url",
        desc="Provided official URL documenting the board structure with multiple positions",
        parent=tx_boardpos_seq,
        critical=True
    )


async def build_minnesota_subtree(evaluator: Evaluator, parent, mn: Optional[MNDistrict]) -> None:
    mn_root = evaluator.add_sequential(
        id="minnesota_district",
        desc="Minnesota district identification and verification",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=(
            mn is not None and
            isinstance(mn.name, str) and mn.name.strip() != "" and
            _state_matches(mn.state, "Minnesota", "MN")
        ),
        id="mn_district_identification",
        desc="Identified a specific public school district in Minnesota",
        parent=mn_root,
        critical=True
    )

    mn_reqs = evaluator.add_parallel(
        id="mn_requirements",
        desc="Minnesota district meets all state-specific and federal requirements",
        parent=mn_root,
        critical=True
    )

    # 1) Enrollment >= 35,000 (sequential)
    mn_enroll_seq = evaluator.add_sequential(
        id="mn_enrollment_35000",
        desc="District has a student enrollment of at least 35,000 students",
        parent=mn_reqs,
        critical=True
    )

    node_mn_enroll = evaluator.add_leaf(
        id="mn_enrollment_count",
        desc="Verified the district has at least 35,000 students enrolled",
        parent=mn_enroll_seq,
        critical=True
    )
    mn_enroll_urls = _normalize_list(mn.enrollment_urls if mn else [])
    mn_enroll_claim = f"The '{(mn.name if mn else 'district')}' district has total student enrollment of at least 35,000."
    await evaluator.verify(
        claim=mn_enroll_claim,
        node=node_mn_enroll,
        sources=mn_enroll_urls,
        additional_instruction="Look for district fast facts, enrollment reports, or official statistics showing a number >= 35,000 students."
    )

    evaluator.add_custom_node(
        result=len(mn_enroll_urls) > 0,
        id="mn_enrollment_url",
        desc="Provided official URL documenting the district enrollment figures",
        parent=mn_enroll_seq,
        critical=True
    )

    # 2) Teacher contract negotiations concluded/ratified in 2026 (sequential)
    mn_contract_seq = evaluator.add_sequential(
        id="mn_teacher_contract_2026",
        desc="District had teacher contract negotiations that concluded with a ratified agreement in 2026",
        parent=mn_reqs,
        critical=True
    )

    node_mn_neg = evaluator.add_leaf(
        id="mn_contract_negotiations",
        desc="Verified the district had teacher contract negotiations in 2026",
        parent=mn_contract_seq,
        critical=True
    )
    mn_contract_urls = _normalize_list(mn.contract_urls if mn else [])
    mn_neg_claim = f"In 2026, the '{(mn.name if mn else 'district')}' district engaged in teacher contract negotiations."
    await evaluator.verify(
        claim=mn_neg_claim,
        node=node_mn_neg,
        sources=mn_contract_urls,
        additional_instruction="Accept official district/union updates, board agendas/minutes, or news releases indicating active bargaining/negotiations during 2026."
    )

    node_mn_rat = evaluator.add_leaf(
        id="mn_contract_ratification",
        desc="Verified the contract was ratified in 2026",
        parent=mn_contract_seq,
        critical=True
    )
    mn_rat_claim = f"The teacher contract for '{(mn.name if mn else 'the district')}' was ratified in 2026."
    await evaluator.verify(
        claim=mn_rat_claim,
        node=node_mn_rat,
        sources=mn_contract_urls,
        additional_instruction="Look for language like 'ratified', 'approved', or 'final agreement' in 2026 by the union and/or the school board."
    )

    evaluator.add_custom_node(
        result=len(mn_contract_urls) > 0,
        id="mn_contract_url",
        desc="Provided official URL documenting the 2026 contract ratification",
        parent=mn_contract_seq,
        critical=True
    )

    # 3) Special education services ages 3–21 (sequential)
    mn_sped_seq = evaluator.add_sequential(
        id="mn_special_education",
        desc="District provides special education services under IDEA Part B for ages 3-21",
        parent=mn_reqs,
        critical=True
    )

    node_mn_sped = evaluator.add_leaf(
        id="mn_special_education_services",
        desc="Verified the district provides IDEA Part B services for students ages 3-21",
        parent=mn_sped_seq,
        critical=True
    )
    mn_sped_urls = _normalize_list(mn.special_education_urls if mn else [])
    mn_sped_claim = f"The '{(mn.name if mn else 'district')}' district provides special education services under IDEA Part B for students ages 3–21."
    await evaluator.verify(
        claim=mn_sped_claim,
        node=node_mn_sped,
        sources=mn_sped_urls,
        additional_instruction="Look for 'ages 3–21' or equivalent wording on special education service pages."
    )

    evaluator.add_custom_node(
        result=len(mn_sped_urls) > 0,
        id="mn_special_education_url",
        desc="Provided official URL documenting special education services",
        parent=mn_sped_seq,
        critical=True
    )

    # 4) Emergency drills conducted (sequential)
    mn_drills_seq = evaluator.add_sequential(
        id="mn_emergency_drills",
        desc="District conducts required emergency safety drills",
        parent=mn_reqs,
        critical=True
    )

    node_mn_drills = evaluator.add_leaf(
        id="mn_drills_conducted",
        desc="Verified the district conducts required emergency safety drills",
        parent=mn_drills_seq,
        critical=True
    )
    mn_drills_urls = _normalize_list(mn.drills_urls if mn else [])
    mn_drills_claim = f"The '{(mn.name if mn else 'district')}' district conducts required emergency safety drills (e.g., fire, lockdown, severe weather)."
    await evaluator.verify(
        claim=mn_drills_claim,
        node=node_mn_drills,
        sources=mn_drills_urls,
        additional_instruction="Accept district policy manuals or safety pages that describe required drills conducted each year."
    )

    evaluator.add_custom_node(
        result=len(mn_drills_urls) > 0,
        id="mn_drills_url",
        desc="Provided official URL documenting emergency drill requirements or policies",
        parent=mn_drills_seq,
        critical=True
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the four-districts compliance task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root per rubric
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=FourDistrictsExtraction,
        extraction_name="districts_extraction",
    )

    # Record ground-truth constraints for transparency (not used for scoring)
    evaluator.add_ground_truth({
        "CT_requirements": {
            "fire_drills_per_year": 7,
            "crisis_drills_per_year": 3,
            "needs_EOP_documentation": True,
            "IDEA_Part_B_ages": "3-21"
        },
        "OH_requirements": {
            "OHSAA_D1_football_2025_26_enrollment_threshold": ">=592 (grades 9-11 adjusted enrollment)",
            "reports_enrollment_to_OHSAA": True,
            "IDEA_Part_B_ages": "3-21",
            "elected_board": True
        },
        "TX_requirements": {
            "UIL_participation": True,
            "IDEA_Part_B_ages": "3-21",
            "board_election_year": 2025,
            "multiple_board_places": True
        },
        "MN_requirements": {
            "enrollment_minimum": 35000,
            "teacher_contract_ratified_year": 2026,
            "IDEA_Part_B_ages": "3-21",
            "conducts_emergency_drills": True
        }
    })

    # Build each state's verification subtree
    await build_connecticut_subtree(evaluator, root, extracted.connecticut if extracted else None)
    await build_ohio_subtree(evaluator, root, extracted.ohio if extracted else None)
    await build_texas_subtree(evaluator, root, extracted.texas if extracted else None)
    await build_minnesota_subtree(evaluator, root, extracted.minnesota if extracted else None)

    # Return structured summary
    return evaluator.get_summary()