import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "four_universities_programs"
TASK_DESCRIPTION = """Identify four universities in the United States that each meet ALL of the following criteria:

For the first two universities:
1. The university must be located in a state in the Northeastern United States.
2. The university must participate in the IvyPlus Exchange Scholar Program for doctoral students.
3. The IvyPlus Exchange Scholar Program at the university must require doctoral students to complete at least one full academic year in residence before becoming eligible to participate in exchange programs at other institutions.
4. The university must have had an educational partnership or collaboration with another institution. Provide details about the partner institution, the nature of the partnership, and whether it is current or has ended.

For the third university:
1. The university must be located in a state in the Midwestern United States.
2. The university must participate in at least one NSF-funded Industry-University Cooperative Research Center (IUCRC) program that facilitates industry-university partnerships.
3. The university must offer graduate certificate programs. Specify the minimum credit hour requirement for these graduate certificates, which should fall within the 9-12 credit hour standard range.

For the fourth university:
1. The university must be located in a state on the West Coast of the United States (California, Oregon, or Washington).
2. The university must have an extension or continuing education division that offers certificate programs. The division should offer multiple certificate programs.
3. The university must participate in the IvyPlus Exchange Scholar Program for doctoral students, with a requirement that students complete at least one full academic year in residence before exchange eligibility.

For each university, provide:
- The university name
- The state and region where it is located
- Evidence of IvyPlus participation (if applicable) with residency requirements
- Evidence of partnership or collaboration (for universities 1 and 2)
- Evidence of IUCRC participation and graduate certificate programs (for university 3)
- Evidence of extension programs and IvyPlus participation (for university 4)
- URL references supporting each claim
"""

# Region definitions (US Census conventions; West Coast explicitly defined)
NORTHEAST_STATES = {
    "Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont",  # New England
    "New Jersey", "New York", "Pennsylvania"  # Mid-Atlantic (Census Northeast)
}
MIDWEST_STATES = {
    "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota",
    "Missouri", "Nebraska", "North Dakota", "Ohio", "South Dakota", "Wisconsin"
}
WEST_COAST_STATES = {"California", "Oregon", "Washington"}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    # Common identity and location
    name: Optional[str] = None
    location_state: Optional[str] = None
    location_region: Optional[str] = None  # If the answer explicitly mentions a region; else null
    location_sources: List[str] = Field(default_factory=list)

    # IvyPlus (u1, u2, u4)
    ivyplus_membership: Optional[str] = None  # Statement or snippet in the answer indicating membership
    ivyplus_doctoral_eligibility: Optional[str] = None  # Statement showing doctoral students eligible
    ivyplus_residency_requirement: Optional[str] = None  # Statement about residency (≥ 1 full academic year before exchange)
    ivyplus_sources: List[str] = Field(default_factory=list)

    # Partnership (u1, u2)
    partnership_partner: Optional[str] = None
    partnership_nature: Optional[str] = None
    partnership_status: Optional[str] = None  # "current" / "ended" / similar phrasing
    partnership_sources: List[str] = Field(default_factory=list)

    # IUCRC (u3)
    iucrc_center_names: List[str] = Field(default_factory=list)  # At least one center name
    iucrc_statements: Optional[str] = None  # Statements indicating NSF-funded IUCRC and industry-university partnerships
    iucrc_sources: List[str] = Field(default_factory=list)

    # Graduate certificates (u3)
    certificates_offered_statement: Optional[str] = None
    certificates_grad_level_statement: Optional[str] = None
    certificates_min_credit_hours: Optional[str] = None  # e.g., "12", "9-12", "at least 12"
    certificates_sources: List[str] = Field(default_factory=list)

    # Extension / Continuing education (u4)
    extension_division_name: Optional[str] = None
    extension_operational_statement: Optional[str] = None
    extension_certificate_offerings_statement: Optional[str] = None
    extension_multiple_programs_statement: Optional[str] = None
    extension_sources: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to FOUR universities as they appear in the answer, in order. For each university, output the following fields.
    If a field is not mentioned in the answer, set it to null (for strings) or [] (for lists).
    Extract only information explicitly present in the answer. For all URL lists, include only actual URLs (not domain mentions without links).

    For each university object, extract:
    - name: The full university name.
    - location_state: The U.S. state where the university is located (e.g., "Pennsylvania").
    - location_region: If the answer explicitly mentions a U.S. region (e.g., "Northeast", "Midwest", "West Coast"), put it here; else null.
    - location_sources: All URLs cited in the answer that support the location of the university.

    IvyPlus-related (for universities that discuss IvyPlus):
    - ivyplus_membership: A snippet or sentence from the answer stating that the university participates in the IvyPlus Exchange Scholar Program.
    - ivyplus_doctoral_eligibility: A snippet indicating the program is for doctoral students.
    - ivyplus_residency_requirement: A snippet indicating students must complete at least one full academic year in residence before exchange eligibility.
    - ivyplus_sources: All URLs cited that support IvyPlus participation and/or residency requirements.

    Partnership-related (for the first two universities if present in the answer):
    - partnership_partner: The partner institution name.
    - partnership_nature: The nature of the partnership (e.g., "dual degree", "joint research initiative", "MOU", "exchange").
    - partnership_status: Whether the partnership is current or ended, as described in the answer.
    - partnership_sources: All URLs documenting the partnership.

    IUCRC-related (for the third university if present in the answer):
    - iucrc_center_names: Names of IUCRC center(s) in which the university participates.
    - iucrc_statements: Snippet(s) indicating the center is NSF-funded and that it facilitates industry-university partnerships.
    - iucrc_sources: All URLs documenting IUCRC participation.

    Graduate certificates (for the third university if present in the answer):
    - certificates_offered_statement: Snippet showing graduate certificates are offered.
    - certificates_grad_level_statement: Snippet confirming certificates are at the graduate level.
    - certificates_min_credit_hours: The minimum credit hour requirement mentioned in the answer (e.g., "9", "12", "9-12").
    - certificates_sources: All URLs documenting the graduate certificate programs and credit hour requirements.

    Extension / Continuing Education (for the fourth university if present in the answer):
    - extension_division_name: Name of the extension or continuing education division.
    - extension_operational_statement: Snippet indicating the division is currently active/operational (e.g., current offerings).
    - extension_certificate_offerings_statement: Snippet confirming the division offers certificate programs.
    - extension_multiple_programs_statement: Snippet indicating the division offers multiple certificate programs (two or more).
    - extension_sources: All URLs documenting the extension division and certificates.

    Return JSON:
    {
      "universities": [ { ... up to 4 items ... } ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def norm_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip()
    # Normalize common abbreviations if the answer used them (best-effort)
    abbr = {
        "PA": "Pennsylvania", "NY": "New York", "NJ": "New Jersey", "MA": "Massachusetts",
        "CT": "Connecticut", "RI": "Rhode Island", "VT": "Vermont", "NH": "New Hampshire", "ME": "Maine",
        "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "MI": "Michigan", "MN": "Minnesota",
        "MO": "Missouri", "NE": "Nebraska", "ND": "North Dakota", "OH": "Ohio", "SD": "South Dakota", "WI": "Wisconsin",
        "CA": "California", "OR": "Oregon", "WA": "Washington"
    }
    return abbr.get(s, s)


def is_in_region(state: Optional[str], region: str) -> bool:
    st = norm_state(state)
    if not st:
        return False
    if region == "Northeast":
        return st in NORTHEAST_STATES
    if region == "Midwest":
        return st in MIDWEST_STATES
    if region == "WestCoast":
        return st in WEST_COAST_STATES
    return False


def parse_min_credits(credit_str: Optional[str]) -> Optional[int]:
    """
    Extract a plausible minimum credit value from a textual field.
    Handles formats like "12", "9-12", "at least 12 credits", "minimum of 9", etc.
    Returns the minimum numeric value if identifiable; else None.
    """
    if not credit_str:
        return None
    s = credit_str.strip().lower()

    # Range like "9-12"
    m_range = re.search(r'(\d+)\s*[-–]\s*(\d+)', s)
    if m_range:
        try:
            a = int(m_range.group(1))
            b = int(m_range.group(2))
            return min(a, b)
        except Exception:
            pass

    # "at least 12" or "minimum of 9"
    m_at_least = re.search(r'(?:at\s+least|min(?:imum)?\s+of)\s*(\d+)', s)
    if m_at_least:
        try:
            return int(m_at_least.group(1))
        except Exception:
            pass

    # First number occurrence
    m_num = re.search(r'(\d+)', s)
    if m_num:
        try:
            return int(m_num.group(1))
        except Exception:
            pass

    return None


def first_or_blank(values: List[str]) -> str:
    return values[0] if values else ""


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_geography(
    evaluator: Evaluator,
    parent_node,
    u_item: UniversityItem,
    u_index: int,
    target_region_label: str  # "Northeast" | "Midwest" | "WestCoast"
) -> None:
    """
    Build the 'geographic_location_u#' sequential subtree with region identification and location URL verification.
    """
    uni_name = u_item.name or f"University #{u_index}"
    state = norm_state(u_item.location_state)

    geo_node = evaluator.add_sequential(
        id=f"geographic_location_u{u_index}",
        desc="University geographic location verification",
        parent=parent_node,
        critical=True
    )

    # Region identification (parallel)
    region_node = evaluator.add_parallel(
        id=f"region_identification_u{u_index}",
        desc=f"University is located in a {'Northeastern' if target_region_label=='Northeast' else ('Midwestern' if target_region_label=='Midwest' else 'West Coast')} state",
        parent=geo_node,
        critical=True
    )

    # State specified (existence)
    evaluator.add_custom_node(
        result=bool(state),
        id=f"state_specification_u{u_index}",
        desc="Identify the specific state",
        parent=region_node,
        critical=True
    )

    # Region verification (logical check)
    region_leaf = evaluator.add_leaf(
        id=f"{'northeast' if target_region_label=='Northeast' else ('midwest' if target_region_label=='Midwest' else 'west_coast')}_verification_u{u_index}",
        desc=f"Verify state is in {'Northeastern United States' if target_region_label=='Northeast' else ('Midwestern United States' if target_region_label=='Midwest' else 'West Coast')}",
        parent=region_node,
        critical=True
    )
    claim_region = f"The state '{state or ''}' is in the {('Northeastern United States' if target_region_label=='Northeast' else ('Midwestern United States' if target_region_label=='Midwest' else 'West Coast (California, Oregon, or Washington)'))}."
    additional_instr_region = (
        "Use standard U.S. Census Bureau regional grouping for Northeast (CT, ME, MA, NH, RI, VT, NJ, NY, PA)."
        if target_region_label == "Northeast" else
        ("Use standard U.S. Census Bureau Midwest states (IL, IN, IA, KS, MI, MN, MO, NE, ND, OH, SD, WI)."
         if target_region_label == "Midwest" else
         "West Coast refers strictly to California, Oregon, or Washington.")
    )
    await evaluator.verify(
        claim=claim_region,
        node=region_leaf,
        additional_instruction=additional_instr_region
    )

    # Location reference via URL(s)
    loc_ref_leaf = evaluator.add_leaf(
        id=f"location_reference_u{u_index}",
        desc="Provide URL reference for location",
        parent=geo_node,
        critical=True
    )
    claim_loc = f"{uni_name} is located in the U.S. state of {state or ''}."
    await evaluator.verify(
        claim=claim_loc,
        node=loc_ref_leaf,
        sources=u_item.location_sources,
        additional_instruction="Verify the university's state from the provided URL(s). City mentions implying the state are acceptable if the page clearly indicates the state."
    )


async def verify_ivyplus(
    evaluator: Evaluator,
    parent_node,
    u_item: UniversityItem,
    u_index: int
) -> None:
    uni_name = u_item.name or f"University #{u_index}"
    ivy_node = evaluator.add_sequential(
        id=f"ivyplus_participation_u{u_index}",
        desc="IvyPlus Exchange Scholar Program participation verification",
        parent=parent_node,
        critical=True
    )

    # Program membership (parallel)
    membership_node = evaluator.add_parallel(
        id=f"program_membership_u{u_index}",
        desc="University participates in IvyPlus Exchange Scholar Program",
        parent=ivy_node,
        critical=True
    )

    # Confirm membership
    membership_leaf = evaluator.add_leaf(
        id=f"program_confirmation_u{u_index}",
        desc="Confirm IvyPlus Exchange Scholar Program membership",
        parent=membership_node,
        critical=True
    )
    claim_membership = f"{uni_name} participates in the IvyPlus Exchange Scholar Program."
    await evaluator.verify(
        claim=claim_membership,
        node=membership_leaf,
        sources=u_item.ivyplus_sources,
        additional_instruction="Check for official participation or inclusion in IvyPlus Exchange Scholar Program materials, membership lists, or institutional pages. Synonyms like 'Exchange Scholar Program' or 'IvyPlus Exchange Scholars' are acceptable."
    )

    # Doctoral students eligibility
    doctoral_leaf = evaluator.add_leaf(
        id=f"doctoral_student_eligibility_u{u_index}",
        desc="Program is available to doctoral students",
        parent=membership_node,
        critical=True
    )
    claim_doctoral = f"The IvyPlus Exchange Scholar Program at {uni_name} is available to doctoral (PhD) students."
    await evaluator.verify(
        claim=claim_doctoral,
        node=doctoral_leaf,
        sources=u_item.ivyplus_sources,
        additional_instruction="Accept phrases like 'doctoral students', 'PhD students', or equivalent. If the program is exclusively for doctoral-level students, that satisfies this criterion."
    )

    # Residency requirements (parallel)
    residency_node = evaluator.add_parallel(
        id=f"residency_requirement_u{u_index}",
        desc="Program has residency requirement before exchange eligibility",
        parent=ivy_node,
        critical=True
    )

    # Minimum one full academic year in residence
    one_year_leaf = evaluator.add_leaf(
        id=f"minimum_residency_period_u{u_index}",
        desc="Students must complete at least one full academic year in residence",
        parent=residency_node,
        critical=True
    )
    claim_one_year = f"Doctoral students at {uni_name} must complete at least one full academic year in residence before they are eligible for IvyPlus exchanges."
    await evaluator.verify(
        claim=claim_one_year,
        node=one_year_leaf,
        sources=u_item.ivyplus_sources,
        additional_instruction="Accept equivalent phrasing such as 'first academic year in residence' or 'one year of full-time study in residence'."
    )

    # Requirement before exchange participation
    before_exchange_leaf = evaluator.add_leaf(
        id=f"requirement_before_exchange_u{u_index}",
        desc="Residency must be completed before exchange program participation",
        parent=residency_node,
        critical=True
    )
    claim_before_exchange = f"The required residency must be completed before participating in an exchange at another institution under the IvyPlus program at {uni_name}."
    await evaluator.verify(
        claim=claim_before_exchange,
        node=before_exchange_leaf,
        sources=u_item.ivyplus_sources,
        additional_instruction="The page should imply the residency is a prerequisite to exchange eligibility; explicit 'before' language or equivalent prerequisite wording is acceptable."
    )

    # Reference existence for IvyPlus
    evaluator.add_custom_node(
        result=bool(u_item.ivyplus_sources),
        id=f"ivyplus_reference_u{u_index}",
        desc="Provide URL reference for IvyPlus participation",
        parent=ivy_node,
        critical=True
    )


async def verify_partnership(
    evaluator: Evaluator,
    parent_node,
    u_item: UniversityItem,
    u_index: int
) -> None:
    uni_name = u_item.name or f"University #{u_index}"
    partner = (u_item.partnership_partner or "").strip()
    nature = (u_item.partnership_nature or "").strip()
    status = (u_item.partnership_status or "").strip()

    part_node = evaluator.add_sequential(
        id=f"educational_partnership_u{u_index}",
        desc="Educational partnership or collaboration verification",
        parent=parent_node,
        critical=True
    )

    existence_node = evaluator.add_parallel(
        id=f"partnership_existence_u{u_index}",
        desc="University has had an educational partnership with another institution",
        parent=part_node,
        critical=True
    )

    # Partner identification - verify via URLs
    partner_leaf = evaluator.add_leaf(
        id=f"partner_identification_u{u_index}",
        desc="Identify the partner institution",
        parent=existence_node,
        critical=True
    )
    claim_partner = f"{uni_name} has or had an educational partnership or collaboration with {partner}."
    await evaluator.verify(
        claim=claim_partner,
        node=partner_leaf,
        sources=u_item.partnership_sources,
        additional_instruction="The source should mention the partner institution explicitly and indicate a formal collaboration, partnership, MOU, joint program, or similar arrangement."
    )

    # Nature of the partnership - verify via URLs
    nature_leaf = evaluator.add_leaf(
        id=f"partnership_nature_u{u_index}",
        desc="Describe the nature of the partnership",
        parent=existence_node,
        critical=True
    )
    claim_nature = f"The partnership between {uni_name} and {partner} involved: {nature}."
    await evaluator.verify(
        claim=claim_nature,
        node=nature_leaf,
        sources=u_item.partnership_sources,
        additional_instruction="Accept paraphrases of the partnership type (e.g., joint research, exchange, dual-degree, MOU). The essence should match."
    )

    # Status - verify via URLs (current vs ended)
    status_leaf = evaluator.add_leaf(
        id=f"partnership_status_u{u_index}",
        desc="Indicate whether partnership is current or has ended",
        parent=part_node,
        critical=True
    )
    claim_status = f"The partnership between {uni_name} and {partner} is {status}."
    await evaluator.verify(
        claim=claim_status,
        node=status_leaf,
        sources=u_item.partnership_sources,
        additional_instruction="Infer 'current' if the page shows ongoing activities or current-year references; infer 'ended' if dates clearly indicate it concluded or is historical."
    )

    # Reference existence for partnership
    evaluator.add_custom_node(
        result=bool(u_item.partnership_sources),
        id=f"partnership_reference_u{u_index}",
        desc="Provide URL reference documenting partnership",
        parent=part_node,
        critical=True
    )


async def verify_iucrc(
    evaluator: Evaluator,
    parent_node,
    u_item: UniversityItem,
    u_index: int
) -> None:
    uni_name = u_item.name or f"University #{u_index}"
    center_name = u_item.iucrc_center_names[0] if u_item.iucrc_center_names else ""

    iucrc_node = evaluator.add_sequential(
        id=f"iucrc_participation_u{u_index}",
        desc="IUCRC program participation verification",
        parent=parent_node,
        critical=True
    )

    membership_node = evaluator.add_parallel(
        id=f"iucrc_membership_u{u_index}",
        desc="University participates in NSF-funded IUCRC centers that facilitate industry-university partnerships",
        parent=iucrc_node,
        critical=True
    )

    # Center identification
    center_leaf = evaluator.add_leaf(
        id=f"center_identification_u{u_index}",
        desc="Identify at least one specific IUCRC center",
        parent=membership_node,
        critical=True
    )
    claim_center = f"{uni_name} participates in the IUCRC center named '{center_name}'."
    await evaluator.verify(
        claim=claim_center,
        node=center_leaf,
        sources=u_item.iucrc_sources,
        additional_instruction="The source should mention the IUCRC name and the university as a site, partner, or member."
    )

    # NSF funding confirmation
    nsf_leaf = evaluator.add_leaf(
        id=f"nsf_funding_confirmation_u{u_index}",
        desc="Confirm the center is NSF-funded",
        parent=membership_node,
        critical=True
    )
    claim_nsf = f"The IUCRC center '{center_name}' is funded by the National Science Foundation (NSF)."
    await evaluator.verify(
        claim=claim_nsf,
        node=nsf_leaf,
        sources=u_item.iucrc_sources,
        additional_instruction="Accept if the page explicitly indicates IUCRC and NSF support/funding."
    )

    # Industry-university collaboration
    collab_leaf = evaluator.add_leaf(
        id=f"industry_university_collaboration_u{u_index}",
        desc="Confirm center facilitates industry-university partnerships",
        parent=membership_node,
        critical=True
    )
    claim_collab = f"The IUCRC center '{center_name}' facilitates industry-university partnerships."
    await evaluator.verify(
        claim=claim_collab,
        node=collab_leaf,
        sources=u_item.iucrc_sources,
        additional_instruction="Look for IUCRC descriptions noting industry membership, industry-driven research agenda, and academia-industry collaboration."
    )

    # Reference existence for IUCRC
    evaluator.add_custom_node(
        result=bool(u_item.iucrc_sources),
        id=f"iucrc_reference_u{u_index}",
        desc="Provide URL reference for IUCRC participation",
        parent=iucrc_node,
        critical=True
    )


async def verify_graduate_certificates(
    evaluator: Evaluator,
    parent_node,
    u_item: UniversityItem,
    u_index: int
) -> None:
    uni_name = u_item.name or f"University #{u_index}"
    cert_node = evaluator.add_sequential(
        id=f"graduate_certificate_programs_u{u_index}",
        desc="Graduate certificate program offerings verification",
        parent=parent_node,
        critical=True
    )

    # Existence (parallel)
    exist_node = evaluator.add_parallel(
        id=f"certificate_existence_u{u_index}",
        desc="University offers graduate certificate programs",
        parent=cert_node,
        critical=True
    )

    # Availability
    avail_leaf = evaluator.add_leaf(
        id=f"certificate_availability_u{u_index}",
        desc="Confirm graduate certificates are available",
        parent=exist_node,
        critical=True
    )
    claim_avail = f"{uni_name} offers graduate certificate programs."
    await evaluator.verify(
        claim=claim_avail,
        node=avail_leaf,
        sources=u_item.certificates_sources,
        additional_instruction="The page should mention 'graduate certificate(s)' or equivalent terminology for post-bachelor graduate-level certificates."
    )

    # Graduate level confirmation
    grad_level_leaf = evaluator.add_leaf(
        id=f"graduate_level_confirmation_u{u_index}",
        desc="Verify certificates are at graduate level",
        parent=exist_node,
        critical=True
    )
    claim_grad = f"The certificate programs at {uni_name} are at the graduate level."
    await evaluator.verify(
        claim=claim_grad,
        node=grad_level_leaf,
        sources=u_item.certificates_sources,
        additional_instruction="Look for explicit 'graduate certificate' wording or an indication these are for graduate students/post-baccalaureate with graduate credit."
    )

    # Credit requirements (parallel)
    credit_node = evaluator.add_parallel(
        id=f"credit_requirements_u{u_index}",
        desc="Graduate certificate credit hour requirements",
        parent=cert_node,
        critical=True
    )

    # Minimum credit hours (source-verified)
    min_hrs_leaf = evaluator.add_leaf(
        id=f"minimum_credit_hours_u{u_index}",
        desc="Specify the minimum credit hour requirement",
        parent=credit_node,
        critical=True
    )
    credit_text = u_item.certificates_min_credit_hours or ""
    claim_minhrs = f"At {uni_name}, a graduate certificate requires a minimum of {credit_text} credit hours."
    await evaluator.verify(
        claim=claim_minhrs,
        node=min_hrs_leaf,
        sources=u_item.certificates_sources,
        additional_instruction="Accept if any official page indicates a minimum graduate certificate credit requirement matching the provided number/range."
    )

    # Standard compliance 9-12 (logic check)
    min_val = parse_min_credits(u_item.certificates_min_credit_hours)
    evaluator.add_custom_node(
        result=(min_val is not None and 9 <= min_val <= 12),
        id=f"standard_compliance_u{u_index}",
        desc="Credit requirement falls within 9-12 hour standard range",
        parent=credit_node,
        critical=True
    )

    # Reference existence
    evaluator.add_custom_node(
        result=bool(u_item.certificates_sources),
        id=f"certificate_reference_u{u_index}",
        desc="Provide URL reference for certificate programs",
        parent=cert_node,
        critical=True
    )


async def verify_extension(
    evaluator: Evaluator,
    parent_node,
    u_item: UniversityItem,
    u_index: int
) -> None:
    uni_name = u_item.name or f"University #{u_index}"
    division_name = (u_item.extension_division_name or "").strip()

    ext_node = evaluator.add_sequential(
        id=f"extension_programs_u{u_index}",
        desc="Extension or continuing education program verification",
        parent=parent_node,
        critical=True
    )

    # Division existence (parallel)
    div_node = evaluator.add_parallel(
        id=f"extension_division_existence_u{u_index}",
        desc="University has extension or continuing education division",
        parent=ext_node,
        critical=True
    )

    # Division identification (URL-verified)
    division_leaf = evaluator.add_leaf(
        id=f"division_identification_u{u_index}",
        desc="Identify the specific extension or continuing education division",
        parent=div_node,
        critical=True
    )
    claim_division = f"{uni_name} has an extension or continuing education division named '{division_name}'."
    await evaluator.verify(
        claim=claim_division,
        node=division_leaf,
        sources=u_item.extension_sources,
        additional_instruction="The page should name the division (e.g., 'University Extension', 'Continuing Studies', 'Continuing and Professional Education')."
    )

    # Division operational status (URL-verified)
    operational_leaf = evaluator.add_leaf(
        id=f"division_operational_status_u{u_index}",
        desc="Verify division is currently operational",
        parent=div_node,
        critical=True
    )
    claim_operational = f"The extension/continuing education division at {uni_name} is currently operational."
    await evaluator.verify(
        claim=claim_operational,
        node=operational_leaf,
        sources=u_item.extension_sources,
        additional_instruction="Evidence includes current offerings, 'Apply' links, recent dates, or active program listings."
    )

    # Certificate offerings (parallel)
    cert_off_leaf_node = evaluator.add_parallel(
        id=f"certificate_offerings_u{u_index}",
        desc="Extension division offers certificate programs",
        parent=ext_node,
        critical=True
    )

    # Certificates available
    cert_avail_leaf = evaluator.add_leaf(
        id=f"certificate_programs_available_u{u_index}",
        desc="Confirm certificate programs are offered through extension",
        parent=cert_off_leaf_node,
        critical=True
    )
    claim_ext_certs = f"The extension division at {uni_name} offers certificate programs."
    await evaluator.verify(
        claim=claim_ext_certs,
        node=cert_avail_leaf,
        sources=u_item.extension_sources,
        additional_instruction="Look for 'certificate programs' in the extension/continuing education division pages."
    )

    # Multiple certificate programs
    multiple_leaf = evaluator.add_leaf(
        id=f"program_quantity_u{u_index}",
        desc="Extension offers multiple certificate programs",
        parent=cert_off_leaf_node,
        critical=True
    )
    claim_multiple = f"The extension division at {uni_name} offers multiple (two or more) certificate programs."
    await evaluator.verify(
        claim=claim_multiple,
        node=multiple_leaf,
        sources=u_item.extension_sources,
        additional_instruction="Accept if the page lists at least two distinct certificate programs."
    )

    # Reference existence
    evaluator.add_custom_node(
        result=bool(u_item.extension_sources),
        id=f"extension_reference_u{u_index}",
        desc="Provide URL reference for extension programs",
        parent=ext_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# University-level verification builders                                      #
# --------------------------------------------------------------------------- #
async def build_university_1(
    evaluator: Evaluator,
    root_node,
    u_item: UniversityItem,
    idx: int
) -> None:
    u_node = evaluator.add_parallel(
        id=f"university_{idx}",
        desc="First university meeting specified criteria" if idx == 1 else "Second university meeting specified criteria",
        parent=root_node,
        critical=False
    )

    # Geographic: Northeast
    await verify_geography(evaluator, u_node, u_item, idx, target_region_label="Northeast")

    # IvyPlus
    await verify_ivyplus(evaluator, u_node, u_item, idx)

    # Partnership
    await verify_partnership(evaluator, u_node, u_item, idx)


async def build_university_3(
    evaluator: Evaluator,
    root_node,
    u_item: UniversityItem,
    idx: int
) -> None:
    u_node = evaluator.add_parallel(
        id=f"university_{idx}",
        desc="Third university meeting specified criteria",
        parent=root_node,
        critical=False
    )

    # Geographic: Midwest
    await verify_geography(evaluator, u_node, u_item, idx, target_region_label="Midwest")

    # IUCRC
    await verify_iucrc(evaluator, u_node, u_item, idx)

    # Graduate certificates
    await verify_graduate_certificates(evaluator, u_node, u_item, idx)


async def build_university_4(
    evaluator: Evaluator,
    root_node,
    u_item: UniversityItem,
    idx: int
) -> None:
    u_node = evaluator.add_parallel(
        id=f"university_{idx}",
        desc="Fourth university meeting specified criteria",
        parent=root_node,
        critical=False
    )

    # Geographic: West Coast
    await verify_geography(evaluator, u_node, u_item, idx, target_region_label="WestCoast")

    # Extension/Continuing Education
    await verify_extension(evaluator, u_node, u_item, idx)

    # IvyPlus
    await verify_ivyplus(evaluator, u_node, u_item, idx)


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
    Evaluate an answer for the four-universities program participation, geography, and partnerships task.
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Prepare exactly 4 universities (pad if fewer, clip if more)
    universities = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Build verification trees for the four universities
    # u1 (index 1): Northeast + IvyPlus + Partnership
    await build_university_1(evaluator, root, universities[0], 1)

    # u2 (index 2): Northeast + IvyPlus + Partnership
    await build_university_1(evaluator, root, universities[1], 2)

    # u3 (index 3): Midwest + IUCRC + Graduate Certificates
    await build_university_3(evaluator, root, universities[2], 3)

    # u4 (index 4): West Coast + Extension + IvyPlus
    await build_university_4(evaluator, root, universities[3], 4)

    return evaluator.get_summary()