import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "public_k12_admin_hiring_march_2026"
TASK_DESCRIPTION = (
    "Find three public school districts in three different U.S. states that are currently hiring for administrative or leadership positions in March 2026. "
    "For each district, provide the following information: (1) The complete name of the school district, (2) The U.S. state where the district is located, "
    "(3) The URL of the district's online job application system or career portal, and (4) At least one specific administrative or leadership position title "
    "that is currently posted (such as Principal, Assistant Principal, Superintendent, Director, or similar administrative role). Ensure that all three districts "
    "are public K-12 school districts (not private schools or universities), and each district is located in a different state."
)

CURRENT_MONTH_YEAR = "March 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictItem(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    application_url: Optional[str] = None
    positions: List[str] = Field(default_factory=list)
    supporting_urls: List[str] = Field(default_factory=list)


class DistrictsExtraction(BaseModel):
    districts: List[DistrictItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    Extract at most the first three school districts presented in the answer that claim to meet the task requirements.
    For each district, return an object with:
    - name: The complete district name as written in the answer (e.g., "Springfield Public Schools", "Mesa Unified School District").
    - state: The U.S. state for the district as written in the answer (can be full name or 2-letter abbreviation).
    - application_url: A URL explicitly provided in the answer that points to the district's online job application system or careers/job portal.
    - positions: A list of one or more specific administrative/leadership position titles that the answer claims are currently posted (e.g., "Principal", "Assistant Principal", "Superintendent", "Director of Special Education", "Athletic Director", "HR Director"). Use only titles explicitly written in the answer.
    - supporting_urls: Any additional URLs in the answer that are relevant to verifying the district identity, state, or job postings (e.g., district homepage, about pages, job posting detail pages). If none, return an empty list.
    
    Rules:
    - Only extract URLs that appear in the answer.
    - application_url must be a valid URL string; if not provided, set it to null.
    - If the answer includes more than three districts, keep only the first three mentioned.
    - If any field is missing, set it to null (or [] for lists).
    """


# --------------------------------------------------------------------------- #
# Helper: US state normalization                                              #
# --------------------------------------------------------------------------- #
US_STATE_ABBREV = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia",
}

FULL_TO_FULL = {v.lower(): v for v in US_STATE_ABBREV.values()}


def normalize_state_name(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().replace(".", "")
    s_up = s.upper()
    if s_up in US_STATE_ABBREV:
        return US_STATE_ABBREV[s_up]
    s_low = s.lower()
    if s_low in FULL_TO_FULL:
        return FULL_TO_FULL[s_low]
    # Accept common variants like "Wash" -> Washington only if unique; keep raw otherwise
    return s.title()


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_district(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    district: DistrictItem,
) -> None:
    """
    Build verification sub-tree for one district with leaves mirroring the rubric.
    """
    dnode = evaluator.add_parallel(
        id=f"District_{idx+1}",
        desc=f"Verification of the {'first' if idx==0 else 'second' if idx==1 else 'third'} district and all its required attributes",
        parent=parent_node,
        critical=False
    )

    # Prepare sources: prefer application_url plus any supporting_urls
    sources: List[str] = []
    if district.application_url:
        sources.append(district.application_url)
    if district.supporting_urls:
        # Avoid duplicates and empties
        extras = [u for u in district.supporting_urls if isinstance(u, str) and u.strip()]
        for u in extras:
            if u not in sources:
                sources.append(u)

    # 1) District_i_Name_And_State (CRITICAL)
    name_state_leaf = evaluator.add_leaf(
        id=f"District_{idx+1}_Name_And_State",
        desc="The complete district name and U.S. state location are correctly provided",
        parent=dnode,
        critical=True,
        score=0.0,
        status="initialized"
    )
    if district.name and district.state and sources:
        claim = (
            f"The provided page(s) belong to the public school district named '{district.name}', and the district is located in the U.S. state of "
            f"'{district.state}'. Minor naming variants are acceptable (e.g., 'USD', 'ISD', 'Public Schools', 'School District'). "
            f"State abbreviations vs. full names should be treated as equivalent."
        )
        await evaluator.verify(
            claim=claim,
            node=name_state_leaf,
            sources=sources,
            additional_instruction=(
                "Check the page header, organization name, footer, or 'About' references to confirm the district identity and state. "
                "Allow reasonable variants like 'XYZ USD' vs 'XYZ Unified School District'. Accept 'CA' == 'California', etc."
            )
        )
    else:
        # Missing required info or sources; fail this critical check
        name_state_leaf.status = "failed"
        name_state_leaf.score = 0.0

    # 2) District_i_Is_Public_K12_District (CRITICAL)
    public_k12_leaf = evaluator.add_leaf(
        id=f"District_{idx+1}_Is_Public_K12_District",
        desc="The district is verified to be a public K-12 school district (not a private school or university)",
        parent=dnode,
        critical=True,
        score=0.0,
        status="initialized"
    )
    if district.name and sources:
        claim = (
            f"'{district.name}' is a public K-12 school district (not a private school and not a university). "
            "The sources should indicate it operates public elementary/secondary schools or is explicitly called a 'school district', 'public schools', 'USD', 'ISD', or similar."
        )
        await evaluator.verify(
            claim=claim,
            node=public_k12_leaf,
            sources=sources,
            additional_instruction=(
                "Focus on whether the entity is a public K-12 district. Phrases like 'Public Schools', 'School District', 'Unified School District', 'Independent School District', "
                "or state/county public district designations are acceptable. Do not accept universities or private/charter organizations unless they are explicitly a public district."
            )
        )
    else:
        public_k12_leaf.status = "failed"
        public_k12_leaf.score = 0.0

    # 3) District_i_Has_Administrative_Hiring (CRITICAL)
    hiring_leaf = evaluator.add_leaf(
        id=f"District_{idx+1}_Has_Administrative_Hiring",
        desc=f"The district has active job postings for administrative or leadership positions as of {CURRENT_MONTH_YEAR}",
        parent=dnode,
        critical=True,
        score=0.0,
        status="initialized"
    )
    if district.application_url:
        claim = (
            f"As of {CURRENT_MONTH_YEAR}, there is at least one currently open administrative or leadership job posting for this district on its careers or application portal. "
            "Examples include principal, assistant principal, superintendent, director, coordinator, or other district/school-level administrative roles."
        )
        await evaluator.verify(
            claim=claim,
            node=hiring_leaf,
            sources=district.application_url,
            additional_instruction=(
                "Confirm at least one admin/leadership opening is active in March 2026. Use any of: explicit post date within March 2026; "
                "closing date in March/April 2026 or later; 'Open until filled' with no closed status; or an 'Active'/'Open' indicator. "
                "Accept reasonable variants of admin titles. If only teacher or classified (non-admin) roles are present, this should fail."
            )
        )
    else:
        hiring_leaf.status = "failed"
        hiring_leaf.score = 0.0

    # 4) District_i_Application_URL (CRITICAL)
    app_url_leaf = evaluator.add_leaf(
        id=f"District_{idx+1}_Application_URL",
        desc="A valid, accessible URL to the district's online job application portal or system is provided",
        parent=dnode,
        critical=True,
        score=0.0,
        status="initialized"
    )
    if district.application_url:
        claim = (
            f"This URL is the job application system or careers portal used by '{district.name or 'the district'}' for job applications "
            "(e.g., Frontline/AppliTrack, NEOGOV, TalentEd, Workday, Greenhouse, or a district-hosted careers page with an Apply function)."
        )
        await evaluator.verify(
            claim=claim,
            node=app_url_leaf,
            sources=district.application_url,
            additional_instruction=(
                "Verify that the page is clearly a jobs/careers portal for the district (job listings and/or 'Apply' functionality). "
                "It should not be a generic unrelated page. Vendor-hosted portals are acceptable if specifically for this district."
            )
        )
    else:
        app_url_leaf.status = "failed"
        app_url_leaf.score = 0.0

    # 5) District_i_Specific_Position (NON-CRITICAL)
    position_leaf = evaluator.add_leaf(
        id=f"District_{idx+1}_Specific_Position",
        desc="At least one specific administrative/leadership position title currently posted is listed",
        parent=dnode,
        critical=False,
        score=0.0,
        status="initialized"
    )
    if district.positions and district.positions[0].strip() and sources:
        pos_title = district.positions[0].strip()
        claim = (
            f"The careers/application portal currently lists an administrative or leadership opening titled '{pos_title}', "
            "or a close variant (e.g., includes school level or department), for this district in March 2026."
        )
        await evaluator.verify(
            claim=claim,
            node=position_leaf,
            sources=sources,
            additional_instruction=(
                "Allow equivalent or extended titles such as 'Assistant Principal - High School', 'Director of Special Education', "
                "'Elementary School Principal', etc. The posting must be an admin/leadership role and currently active in March 2026."
            )
        )
    else:
        # Non-critical but fail if insufficient info to verify
        position_leaf.status = "failed"
        position_leaf.score = 0.0


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the public K-12 administrative hiring task (March 2026).
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

    # Create the main root node (as per rubric)
    top = evaluator.add_parallel(
        id="Three_Districts_Identified",
        desc="Evaluation of whether three public school districts meeting all specified criteria have been correctly identified with complete information",
        parent=root,
        critical=False,
    )

    # 1) Extract district data
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction",
    )

    # Keep only the first three
    districts: List[DistrictItem] = (extracted.districts or [])[:3]
    # Pad with empty entries if fewer than 3
    while len(districts) < 3:
        districts.append(DistrictItem())

    # 2) Build district verification subtrees
    for i in range(3):
        await verify_single_district(evaluator, top, i, districts[i])

    # 3) Global constraint: all in different states (CRITICAL)
    states_raw = [normalize_state_name(d.state) for d in districts]
    non_null_states = [s for s in states_raw if s]
    unique_states = set(non_null_states)
    different_states_ok = len(non_null_states) == 3 and len(unique_states) == 3

    evaluator.add_custom_info(
        info={"states_raw": [d.state for d in districts], "normalized_states": states_raw},
        info_type="diagnostic",
        info_name="state_normalization_debug",
    )

    evaluator.add_custom_node(
        result=different_states_ok,
        id="Three_Different_States",
        desc="Each of the three districts is located in a different U.S. state (no state appears more than once)",
        parent=top,
        critical=True,
    )

    # 4) Return summary
    return evaluator.get_summary()