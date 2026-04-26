import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "triangle_woodworking_facilities"
TASK_DESCRIPTION = """
I am researching woodworking facilities in North Carolina's Research Triangle region to support a community guide for woodworkers with different needs. Please identify three distinct woodworking facilities that meet the following specific criteria:

Facility 1: Find a woodworking facility located in Durham, Raleigh, or Cary that offers semester-based memberships with a three-tier pricing structure (such as student, affiliate, and general public rates). The facility must allow members independent access to woodworking equipment outside of scheduled classes. Among all facilities you identify, this one should have the lowest general public semester membership rate. Provide the facility name, complete address, the general public semester rate, the membership tier structure, and a URL reference to the official membership pricing page.

Facility 2: Find a woodworking facility located in Durham, Raleigh, or Cary that explicitly permits teenagers aged 12-17 to access woodworking equipment through classes, studio passes, or memberships. The facility must publish specific operating hours or class schedules showing when teenagers can participate. Provide the facility name, complete address, the minimum age requirement, specific operating hours or days/times when teenagers can access woodworking, and a URL reference to the official page documenting age policies and schedules.

Facility 3: Find a North Carolina community college that offers a formal woodworking certificate, diploma, or associate degree program (not just standalone classes) with hands-on instruction in woodworking techniques. The program must have published course fees or tuition information. Provide the college name, location in North Carolina, the specific credential offered (certificate/diploma/degree), at least one specific woodworking course fee or program cost in dollars, and a URL reference to the official program page showing curriculum and fees.

All three facilities must be different institutions, and all pricing and factual claims must be supported by official URL references from the facility's or institution's website.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityBase(BaseModel):
    name: Optional[str] = None
    # Address fields: prefer structured, but also allow a full string
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    full_address: Optional[str] = None

    # Optional supporting URLs the answer may include (must be official)
    equipment_urls: List[str] = Field(default_factory=list)
    other_official_urls: List[str] = Field(default_factory=list)  # contact/about/home pages, etc.


class Facility1Extraction(FacilityBase):
    membership_pricing_url: Optional[str] = None
    pricing_tiers_description: Optional[str] = None
    general_public_semester_rate: Optional[str] = None
    semester_periods_description: Optional[str] = None
    independent_access_description: Optional[str] = None
    # Optional: if answer mentions any other general-public semester rates for comparison in other facilities
    # (not required, but helps "lowest rate" check in edge cases)
    # Keep base as-is.


class Facility2Extraction(FacilityBase):
    minimum_age_requirement: Optional[str] = None
    age_policy_url: Optional[str] = None
    teen_schedule_days_times: Optional[str] = None
    teen_schedule_url: Optional[str] = None
    # Optional: in case the answer also mentions any general-public semester rate (not required)
    general_public_semester_rate: Optional[str] = None


class Facility3Extraction(FacilityBase):
    # College info uses base.name, base.city, base.state (should be NC)
    program_credential: Optional[str] = None  # certificate/diploma/associate degree
    program_url: Optional[str] = None
    fees_url: Optional[str] = None  # may be same as program_url; if not, separate
    specific_cost_amount: Optional[str] = None  # at least one $ amount mentioned


class AllFacilitiesExtraction(BaseModel):
    facility_1: Optional[Facility1Extraction] = None
    facility_2: Optional[Facility2Extraction] = None
    facility_3: Optional[Facility3Extraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract structured information for three facilities described in the answer. Follow strictly:

    COMMON RULES:
    - Extract exactly what appears in the answer; do NOT invent.
    - For URLs, extract the actual URLs present in the answer text (including markdown links).
    - If a field is missing, set it to null; for URL lists, return an empty array if none.

    FACILITY 1 (Triangle facility with semester-based 3-tier pricing and independent access):
    Fields:
    - name
    - street (street portion of address; null if not provided)
    - city (null if not provided)
    - state (null if not provided)
    - full_address (as a single string if given)
    - membership_pricing_url (URL to the official membership pricing page)
    - pricing_tiers_description (the answer’s description of the three tier categories, e.g., "student / affiliate / general public")
    - general_public_semester_rate (a dollar string as presented, e.g., "$240 per semester")
    - semester_periods_description (any mention like Fall/Spring/Summer semesters or “semester-based” phrasing as stated in the answer)
    - independent_access_description (answer’s wording that indicates open studio/independent equipment access outside classes)
    - equipment_urls (list of official URLs, if the answer cites any, that describe available equipment)
    - other_official_urls (any additional official URLs provided in the answer for this facility, e.g., contact/home/about)

    FACILITY 2 (Triangle facility explicitly allowing teens 12–17; publishes specific days/times):
    Fields:
    - name
    - street
    - city
    - state
    - full_address
    - minimum_age_requirement (e.g., "12+", "13–17", etc.)
    - age_policy_url (official URL that documents teen access policy)
    - teen_schedule_days_times (the specific days/times quoted in the answer for teen participation)
    - teen_schedule_url (official URL that shows operating hours or class schedules relevant to teen access)
    - equipment_urls (list of official URLs for equipment, if present)
    - other_official_urls (any other official URLs provided)
    - general_public_semester_rate (if the answer mentions a general-public semester rate for this facility; else null)

    FACILITY 3 (NC community college with formal woodworking credential and published fees):
    Fields:
    - name
    - street (if provided; else null)
    - city
    - state
    - full_address (if provided; else null)
    - program_credential (e.g., "Certificate", "Diploma", "Associate degree")
    - program_url (official program page URL showing curriculum; fees may be here or separate)
    - fees_url (official page URL that shows fees/tuition if separate; null if program_url shows fees)
    - specific_cost_amount (a dollar string exactly as shown in the answer for a woodworking course/program cost)
    - equipment_urls (list of official URLs about equipment, if provided)
    - other_official_urls (any additional official URLs provided)

    Ensure that:
    - city/state fields reflect what the answer claims.
    - URLs must be extracted exactly as presented in the answer; do not add new URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
ALLOWED_TRIANGLE_CITIES = {"durham", "raleigh", "cary"}


def has_complete_address(entity: FacilityBase) -> bool:
    # Consider complete if street, city, and state are all non-empty.
    if entity.street and entity.street.strip() and entity.city and entity.city.strip() and entity.state and entity.state.strip():
        return True
    # Fallback: if full_address includes city and state
    if entity.full_address and entity.city and entity.state:
        fa = entity.full_address.lower()
        if entity.city.lower() in fa and entity.state.lower() in fa:
            return True
    return False


def parse_dollar_amount(amount_str: Optional[str]) -> Optional[float]:
    if not amount_str:
        return None
    s = amount_str.strip().lower()
    if "free" in s:
        return 0.0
    # Extract first dollar-like number
    m = re.search(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,2})|[0-9]+(?:\.[0-9]{1,2})?)", s)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return float(num)
    except Exception:
        return None


def combine_urls(*args: Optional[List[str] | str]) -> List[str]:
    urls: List[str] = []
    seen = set()
    for item in args:
        if not item:
            continue
        if isinstance(item, str):
            u = item.strip()
            if u and u not in seen:
                urls.append(u)
                seen.add(u)
        elif isinstance(item, list):
            for u in item:
                if u and isinstance(u, str):
                    uu = u.strip()
                    if uu and uu not in seen:
                        urls.append(uu)
                        seen.add(uu)
    return urls


def city_in_allowed(city: Optional[str], state: Optional[str]) -> bool:
    if not city:
        return False
    if state and state.strip().lower() not in {"nc", "north carolina"}:
        # Still allow if state omitted in answer; strict if provided wrong state
        return False
    return city.strip().lower() in ALLOWED_TRIANGLE_CITIES


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_facility_1(
    evaluator: Evaluator,
    parent_node,
    f1: Facility1Extraction,
    f2: Optional[Facility2Extraction],
    f3: Optional[Facility3Extraction],
) -> None:
    # Facility 1 node
    f1_node = evaluator.add_parallel(
        id="facility_1",
        desc="Facility 1: Triangle woodworking facility with semester-based membership, 3-tier pricing, independent access, and lowest general-public semester rate.",
        parent=parent_node,
        critical=False
    )

    # 1) Identity & location (critical parallel)
    id_loc = evaluator.add_parallel(
        id="facility_1_identity_and_location",
        desc="Provide facility name and complete address; address confirms Durham/Raleigh/Cary, NC.",
        parent=f1_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(f1 and f1.name and f1.name.strip()),
        id="facility_1_name_provided",
        desc="Facility name is provided.",
        parent=id_loc,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(f1 and has_complete_address(f1)),
        id="facility_1_complete_address_provided",
        desc="A complete physical address is provided (street + city + state).",
        parent=id_loc,
        critical=True
    )

    city_leaf = evaluator.add_leaf(
        id="facility_1_city_in_allowed_set",
        desc="The address city is Durham, Raleigh, or Cary (NC).",
        parent=id_loc,
        critical=True
    )
    city_claim = f"The provided city '{(f1.city or '').strip()}' is one of Durham, Raleigh, or Cary in North Carolina."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        additional_instruction="This is a simple logical check against the allowed set. Accept if the city equals Durham, Raleigh, or Cary (case-insensitive)."
    )

    # 2) Semester membership & independent access (critical parallel)
    sem_and_access = evaluator.add_parallel(
        id="facility_1_semester_membership_and_access",
        desc="Facility offers semester-based memberships and allows independent access outside scheduled classes.",
        parent=f1_node,
        critical=True
    )

    sem_periods = evaluator.add_leaf(
        id="facility_1_semester_periods",
        desc="Official info shows membership is sold for a semester-based period (e.g., Fall/Spring/Summer).",
        parent=sem_and_access,
        critical=True
    )
    await evaluator.verify(
        claim="The official membership page indicates that woodworking membership is sold on a semester-based period (e.g., Fall, Spring, Summer).",
        node=sem_periods,
        sources=f1.membership_pricing_url,
        additional_instruction="Look for the word 'semester' or explicit semester labels like 'Fall/Spring/Summer' on the page."
    )

    indep_access = evaluator.add_leaf(
        id="facility_1_independent_access_outside_classes",
        desc="Official policy indicates members can access/use woodworking equipment outside scheduled classes (e.g., open studio/independent use).",
        parent=sem_and_access,
        critical=True
    )
    access_sources = combine_urls(f1.membership_pricing_url, f1.other_official_urls)
    await evaluator.verify(
        claim="The official page(s) indicate that members can independently access woodworking equipment outside scheduled classes (e.g., open studio/member hours).",
        node=indep_access,
        sources=access_sources,
        additional_instruction="Accept synonyms such as 'open studio', 'member shop hours', or 'independent use'. If the page clearly states independent access beyond classes, pass."
    )

    # 3) Three-tier pricing & deliverables (critical parallel)
    tiers = evaluator.add_parallel(
        id="facility_1_three_tier_pricing_and_deliverables",
        desc="Facility has a three-tier pricing structure distinguishing student, affiliate, and general public; response provides tier structure and general public semester rate.",
        parent=f1_node,
        critical=True
    )

    tiers_exist = evaluator.add_leaf(
        id="facility_1_pricing_tiers_exist_officially",
        desc="Official pricing shows three distinct tiers that correspond to student, affiliate, and general public categories.",
        parent=tiers,
        critical=True
    )
    await evaluator.verify(
        claim="The official membership pricing page shows three distinct membership tiers that correspond to student, affiliate (e.g., faculty/staff/alumni/campus), and general public/community.",
        node=tiers_exist,
        sources=f1.membership_pricing_url,
        additional_instruction="Allow reasonable synonyms for 'affiliate' (e.g., faculty/staff/alumni/fellows). The page should clearly distinguish three categories including a general public/community tier."
    )

    evaluator.add_custom_node(
        result=bool(f1 and f1.pricing_tiers_description and f1.pricing_tiers_description.strip()),
        id="facility_1_membership_tier_structure_stated",
        desc="Answer states the membership tier structure (i.e., describes the three tiers and how they differ).",
        parent=tiers,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(parse_dollar_amount(getattr(f1, "general_public_semester_rate", None)) is not None),
        id="facility_1_general_public_semester_rate_provided",
        desc="A general public semester membership rate (dollar amount) is provided.",
        parent=tiers,
        critical=True
    )

    # 4) Lowest general public semester rate (critical single leaf)
    # Compare f1 against any other provided general-public semester rates (if any).
    other_rates: List[float] = []
    if f2 and f2.general_public_semester_rate:
        r = parse_dollar_amount(f2.general_public_semester_rate)
        if r is not None:
            other_rates.append(r)
    # Facility 3 is a college program, not a membership; do not include its costs here.

    f1_rate_val = parse_dollar_amount(getattr(f1, "general_public_semester_rate", None))
    lowest_ok = False
    if f1_rate_val is not None:
        if not other_rates:
            lowest_ok = True
        else:
            lowest_ok = all(f1_rate_val <= r for r in other_rates)
    else:
        lowest_ok = False

    evaluator.add_custom_node(
        result=lowest_ok,
        id="facility_1_lowest_general_public_semester_rate",
        desc="Facility 1’s general-public semester membership rate is the lowest among the identified facilities that provide a general-public semester membership rate.",
        parent=f1_node,
        critical=True
    )

    # 5) Equipment minimum (critical parallel)
    equip_node = evaluator.add_parallel(
        id="facility_1_equipment_minimum",
        desc="Facility provides access to professional woodworking equipment including saws, sanders, and (lathes OR planers).",
        parent=f1_node,
        critical=True
    )
    equip_sources = combine_urls(f1.membership_pricing_url, f1.equipment_urls, f1.other_official_urls)

    saws = evaluator.add_leaf(
        id="facility_1_has_saws",
        desc="Official info indicates saw(s) are available.",
        parent=equip_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official page(s) indicate that saws (e.g., table saw, band saw, miter saw) are available for use.",
        node=saws,
        sources=equip_sources,
        additional_instruction="Look for specific tools such as 'table saw', 'band saw', 'miter saw', or 'saw'."
    )

    sanders = evaluator.add_leaf(
        id="facility_1_has_sanders",
        desc="Official info indicates sander(s) are available.",
        parent=equip_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official page(s) indicate that sanders (e.g., belt sander, disc sander, spindle sander) are available for use.",
        node=sanders,
        sources=equip_sources,
        additional_instruction="Look for 'sander', 'belt sander', 'disc sander', 'spindle sander', or similar."
    )

    lathe_planer = evaluator.add_leaf(
        id="facility_1_has_lathes_or_planers",
        desc="Official info indicates lathe(s) or planer(s) are available.",
        parent=equip_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official page(s) indicate that either wood lathes or planers are available for use.",
        node=lathe_planer,
        sources=equip_sources,
        additional_instruction="Pass if either 'lathe' or 'planer' is present on the official equipment list/page."
    )

    # 6) Official URL pricing page (critical, single)
    pricing_url_leaf = evaluator.add_leaf(
        id="facility_1_official_url_pricing_page",
        desc="Provide an official-website URL that documents membership pricing/tiers and the semester rate for Facility 1.",
        parent=f1_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page is an official facility website page that documents woodworking membership pricing and tiers including a semester rate.",
        node=pricing_url_leaf,
        sources=f1.membership_pricing_url,
        additional_instruction="Confirm the page is on the facility's official domain and mentions membership pricing/tiers and a semester-based rate."
    )


async def verify_facility_2(
    evaluator: Evaluator,
    parent_node,
    f2: Facility2Extraction
) -> None:
    f2_node = evaluator.add_parallel(
        id="facility_2",
        desc="Facility 2: Triangle facility allowing teens 12–17 with published schedules; required deliverables and official URLs.",
        parent=parent_node,
        critical=False
    )

    # 1) Identity & location
    id_loc = evaluator.add_parallel(
        id="facility_2_identity_and_location",
        desc="Provide facility name and complete address; address confirms Durham/Raleigh/Cary, NC.",
        parent=f2_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(f2 and f2.name and f2.name.strip()),
        id="facility_2_name_provided",
        desc="Facility name is provided.",
        parent=id_loc,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(f2 and has_complete_address(f2)),
        id="facility_2_complete_address_provided",
        desc="A complete physical address is provided (street + city + state).",
        parent=id_loc,
        critical=True
    )
    city_leaf = evaluator.add_leaf(
        id="facility_2_city_in_allowed_set",
        desc="The address city is Durham, Raleigh, or Cary (NC).",
        parent=id_loc,
        critical=True
    )
    city_claim = f"The provided city '{(f2.city or '').strip()}' is one of Durham, Raleigh, or Cary in North Carolina."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        additional_instruction="Simple logical check; accept if the city is one of Durham, Raleigh, or Cary (case-insensitive)."
    )

    # 2) Teen permission ages 12–17 (critical parallel)
    teen_perm = evaluator.add_parallel(
        id="facility_2_teen_permission_12_17",
        desc="Facility explicitly permits teenagers ages 12–17 to access woodworking equipment.",
        parent=f2_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(f2 and f2.minimum_age_requirement and f2.minimum_age_requirement.strip()),
        id="facility_2_minimum_age_requirement_provided",
        desc="A minimum age requirement is stated in the answer.",
        parent=teen_perm,
        critical=True
    )

    allow_12_17 = evaluator.add_leaf(
        id="facility_2_policy_allows_ages_12_17",
        desc="Official policy explicitly permits teen participation covering ages 12–17.",
        parent=teen_perm,
        critical=True
    )
    await evaluator.verify(
        claim="The official page explicitly permits teen participation covering ages 12–17 via classes, studio passes, or memberships.",
        node=allow_12_17,
        sources=f2.age_policy_url,
        additional_instruction="Accept if policy states 'ages 12–17', '12 and up' (which includes 12–17), or similar clear teen allowance language."
    )

    # 3) Published schedule for teens (critical parallel)
    teen_sched = evaluator.add_parallel(
        id="facility_2_published_schedule_for_teens",
        desc="Facility publishes specific operating hours or class schedules (days/times) for teen participation.",
        parent=f2_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(f2 and f2.teen_schedule_days_times and f2.teen_schedule_days_times.strip()),
        id="facility_2_days_times_provided",
        desc="Answer includes specific days/times when teens can participate.",
        parent=teen_sched,
        critical=True
    )

    sched_official = evaluator.add_leaf(
        id="facility_2_schedule_is_officially_published",
        desc="An official URL documents the operating hours or class schedule relevant to teen participation.",
        parent=teen_sched,
        critical=True
    )
    await evaluator.verify(
        claim="The official page shows operating hours or class schedules that indicate when teenagers can participate in woodworking.",
        node=sched_official,
        sources=f2.teen_schedule_url,
        additional_instruction="Look for specific days and times; the page should explicitly show schedule/hours relevant to teen participation."
    )

    # 4) Equipment minimum (critical parallel)
    equip_node = evaluator.add_parallel(
        id="facility_2_equipment_minimum",
        desc="Facility provides access to professional woodworking equipment including saws, sanders, and (lathes OR planers).",
        parent=f2_node,
        critical=True
    )
    equip_sources = combine_urls(f2.equipment_urls, f2.other_official_urls, f2.age_policy_url, f2.teen_schedule_url)

    saws = evaluator.add_leaf(
        id="facility_2_has_saws",
        desc="Official info indicates saw(s) are available.",
        parent=equip_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official page(s) indicate that saws (e.g., table saw, band saw, miter saw) are available for use.",
        node=saws,
        sources=equip_sources,
        additional_instruction="Look for 'table saw', 'band saw', 'miter saw', or 'saw'."
    )

    sanders = evaluator.add_leaf(
        id="facility_2_has_sanders",
        desc="Official info indicates sander(s) are available.",
        parent=equip_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official page(s) indicate that sanders (e.g., belt sander, disc sander, spindle sander) are available for use.",
        node=sanders,
        sources=equip_sources,
        additional_instruction="Look for 'sander', 'belt sander', 'disc sander', 'spindle sander', or similar."
    )

    lathe_planer = evaluator.add_leaf(
        id="facility_2_has_lathes_or_planers",
        desc="Official info indicates lathe(s) or planer(s) are available.",
        parent=equip_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official page(s) indicate that either wood lathes or planers are available for use.",
        node=lathe_planer,
        sources=equip_sources,
        additional_instruction="Pass if either 'lathe' or 'planer' appears on official equipment pages."
    )

    # 5) Official URLs for age policy and schedule (critical single)
    official_urls_leaf = evaluator.add_leaf(
        id="facility_2_official_urls_age_and_schedule",
        desc="Provide official-website URL(s) documenting teen age policy and schedule/hours.",
        parent=f2_node,
        critical=True
    )
    both_urls = combine_urls(f2.age_policy_url, f2.teen_schedule_url)
    await evaluator.verify(
        claim="These URLs are official facility pages documenting teen age policy (covering ages 12–17) and the schedule/hours relevant to teen participation.",
        node=official_urls_leaf,
        sources=both_urls,
        additional_instruction="Confirm that at least one URL clearly shows the teen age policy and another (or the same) shows specific days/times."
    )


async def verify_facility_3(
    evaluator: Evaluator,
    parent_node,
    f3: Facility3Extraction
) -> None:
    f3_node = evaluator.add_parallel(
        id="facility_3",
        desc="Facility 3: NC community college offering a formal woodworking credential with published fees; required deliverables and official program URL.",
        parent=parent_node,
        critical=False
    )

    # 1) College identity and NC location (critical parallel)
    college_id = evaluator.add_parallel(
        id="facility_3_college_identity_and_nc_location",
        desc="Provide college name and NC location; institution is a North Carolina community college.",
        parent=f3_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(f3 and f3.name and f3.name.strip()),
        id="facility_3_college_name_provided",
        desc="College name is provided.",
        parent=college_id,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(f3 and ((f3.city and f3.city.strip()) and (f3.state and f3.state.strip()))),
        id="facility_3_nc_location_provided",
        desc="Location in North Carolina is provided (city + state).",
        parent=college_id,
        critical=True
    )

    is_nc_cc = evaluator.add_leaf(
        id="facility_3_is_nc_community_college",
        desc="Official information confirms the institution is a North Carolina community college.",
        parent=college_id,
        critical=True
    )
    is_nc_sources = combine_urls(f3.program_url, f3.other_official_urls)
    await evaluator.verify(
        claim="The official college page indicates this institution is a North Carolina community college.",
        node=is_nc_cc,
        sources=is_nc_sources,
        additional_instruction="Look for mentions of 'Community College', 'NC Community College System', or clear indications the institution is a NC community college."
    )

    # 2) Formal credential program (critical parallel)
    formal_prog = evaluator.add_parallel(
        id="facility_3_formal_credential_program",
        desc="College offers a formal credential program (certificate/diploma/associate degree) in woodworking or professional crafts-wood.",
        parent=f3_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(f3 and f3.program_credential and f3.program_credential.strip()),
        id="facility_3_credential_type_provided",
        desc="Answer specifies the credential offered (certificate/diploma/associate degree).",
        parent=formal_prog,
        critical=True
    )

    formal_leaf = evaluator.add_leaf(
        id="facility_3_program_is_formal_not_standalone",
        desc="Official program page indicates a credential-bearing program rather than only standalone courses.",
        parent=formal_prog,
        critical=True
    )
    await evaluator.verify(
        claim="The official program page indicates a credential-bearing program (certificate, diploma, or associate degree) in woodworking/professional crafts (not just standalone enrichment classes).",
        node=formal_leaf,
        sources=f3.program_url,
        additional_instruction="Look for explicit credential naming (Certificate/Diploma/Associate) and program structure."
    )

    # 3) Hands-on curriculum (critical single)
    hands_on = evaluator.add_leaf(
        id="facility_3_hands_on_curriculum",
        desc="Official curriculum/course descriptions show hands-on woodworking instruction (e.g., joinery, finishing, design, construction).",
        parent=f3_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official program page shows hands-on woodworking instruction (e.g., joinery, finishing, design, construction, shop practice).",
        node=hands_on,
        sources=f3.program_url,
        additional_instruction="Accept if course descriptions clearly involve practical woodworking techniques or shop work."
    )

    # 4) Published fees (critical parallel)
    fees_node = evaluator.add_parallel(
        id="facility_3_published_fees",
        desc="Program has published fees/tuition information with at least one specific dollar amount stated.",
        parent=f3_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(parse_dollar_amount(f3.specific_cost_amount) is not None),
        id="facility_3_one_specific_cost_in_dollars",
        desc="At least one specific woodworking course fee or program cost is stated in dollars.",
        parent=fees_node,
        critical=True
    )

    fee_src = evaluator.add_leaf(
        id="facility_3_official_fee_or_tuition_source",
        desc="An official URL documents the cited fee/tuition/cost information.",
        parent=fees_node,
        critical=True
    )
    fee_sources = combine_urls(f3.fees_url, f3.program_url)
    await evaluator.verify(
        claim="The official college page(s) document fees/tuition/costs for the woodworking program or its courses.",
        node=fee_src,
        sources=fee_sources,
        additional_instruction="The page should show a dollar amount relevant to the program or link clearly to tuition/fees for it."
    )

    # 5) Equipment minimum (critical parallel)
    equip_node = evaluator.add_parallel(
        id="facility_3_equipment_minimum",
        desc="Institution/program provides access to professional woodworking equipment including saws, sanders, and (lathes OR planers).",
        parent=f3_node,
        critical=True
    )
    equip_sources = combine_urls(f3.program_url, f3.equipment_urls, f3.other_official_urls)

    saws = evaluator.add_leaf(
        id="facility_3_has_saws",
        desc="Official info indicates saw(s) are available for hands-on instruction.",
        parent=equip_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official page(s) indicate that saws (e.g., table saw, band saw, miter saw) are available for hands-on instruction.",
        node=saws,
        sources=equip_sources,
        additional_instruction="Look for 'table saw', 'band saw', 'miter saw', or 'saw' in facilities or course descriptions."
    )

    sanders = evaluator.add_leaf(
        id="facility_3_has_sanders",
        desc="Official info indicates sander(s) are available for hands-on instruction.",
        parent=equip_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official page(s) indicate that sanders (e.g., belt sander, disc sander, spindle sander) are available for hands-on instruction.",
        node=sanders,
        sources=equip_sources,
        additional_instruction="Look for common sander types."
    )

    lathe_planer = evaluator.add_leaf(
        id="facility_3_has_lathes_or_planers",
        desc="Official info indicates lathe(s) or planer(s) are available for hands-on instruction.",
        parent=equip_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official page(s) indicate that either wood lathes or planers are available for hands-on instruction.",
        node=lathe_planer,
        sources=equip_sources,
        additional_instruction="Pass if either 'lathe' or 'planer' appears on an official equipment or program page."
    )

    # 6) Official program URL with curriculum & fees (critical single)
    prog_url_leaf = evaluator.add_leaf(
        id="facility_3_official_program_url_curriculum_and_fees",
        desc="Provide an official-website URL to the program page showing curriculum/course descriptions and fees/tuition.",
        parent=f3_node,
        critical=True
    )
    await evaluator.verify(
        claim="This official program page shows curriculum/course descriptions and also includes fees/tuition information for the program.",
        node=prog_url_leaf,
        sources=f3.program_url,
        additional_instruction="Pass only if the program page itself shows both curriculum/course details and fees/tuition; if it only links elsewhere for fees without showing them, do not pass."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: three facilities evaluated independently
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

    # Extract all facilities information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllFacilitiesExtraction,
        extraction_name="extracted_facilities"
    )

    f1 = extracted.facility_1 or Facility1Extraction()
    f2 = extracted.facility_2 or Facility2Extraction()
    f3 = extracted.facility_3 or Facility3Extraction()

    # Build facility-specific verification trees
    await verify_facility_1(evaluator, root, f1, f2, f3)
    await verify_facility_2(evaluator, root, f2)
    await verify_facility_3(evaluator, root, f3)

    # Cross-facility: distinctness (critical, single leaf)
    distinct = evaluator.add_custom_node(
        result=(
            bool(f1.name and f2.name and f3.name) and
            (f1.name.strip().lower() != f2.name.strip().lower()) and
            (f1.name.strip().lower() != f3.name.strip().lower()) and
            (f2.name.strip().lower() != f3.name.strip().lower())
        ),
        id="facilities_distinctness",
        desc="All three facilities are distinct institutions (no reuse of the same institution).",
        parent=root,
        critical=True
    )

    # Root-level: official source requirement (critical, single leaf) – presence of required official URLs
    # Facility 1 requires membership_pricing_url
    f1_urls_ok = bool(f1.membership_pricing_url and f1.membership_pricing_url.strip())
    # Facility 2 requires age_policy_url and teen_schedule_url
    f2_urls_ok = bool(f2.age_policy_url and f2.age_policy_url.strip() and f2.teen_schedule_url and f2.teen_schedule_url.strip())
    # Facility 3 requires program_url
    f3_urls_ok = bool(f3.program_url and f3.program_url.strip())

    evaluator.add_custom_node(
        result=(f1_urls_ok and f2_urls_ok and f3_urls_ok),
        id="official_source_requirement",
        desc="All factual claims are supported by URL references from the official website(s) of the respective facility/institution.",
        parent=root,
        critical=True
    )

    # Return summary
    return evaluator.get_summary()