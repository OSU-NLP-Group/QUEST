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
TASK_ID = "athletic_admin_tri_level"
TASK_DESCRIPTION = """Identify three athletic administration leadership positions currently filled (as of 2024-2025 academic year or most recently appointed), one at each of the following organizational levels, meeting all specified requirements:

Position 1 - High School Athletic Director:
- Serves at a public high school (not middle school, not district-wide position) located in Texas
- Holds or is required to hold NIAAA certification at any level (RAA, CAA, RMSAA, or CMAA)
- Has prior teaching experience or holds teaching certification
- Reports directly to a building principal or school superintendent
- Provide the person's name, school name, and a reference URL

Position 2 - NCAA Division I FBS Athletic Director:
- Serves at an NCAA Division I FBS (Football Bowl Subdivision) institution
- The institution is a member of a Power Five conference (ACC, Big Ten, Big 12, Pac-12, or SEC) as of the 2024-2025 academic year
- Holds the title of Director of Athletics or Athletic Director (not assistant or associate)
- Holds at least a Master's degree
- Has documented previous experience in athletic administration at the collegiate level (e.g., associate AD, senior associate AD, or equivalent position)
- Provide the person's name, institution name, and a reference URL

Position 3 - State Athletic Association Executive Leadership:
- Serves as executive director, commissioner, or equivalent top leadership position at a state-level high school athletic association in the United States
- The organization governs high school athletics at the state level (covering an entire state, not regional or local)
- Has documented prior experience in athletic administration at either the high school or collegiate level before assuming this state association role
- Provide the person's name, organization name, and a reference URL

For each position, provide: name, current title, organization/institution, and a valid reference URL that confirms their position and qualifications.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HSPosition(BaseModel):
    person_name: Optional[str] = None
    current_title: Optional[str] = None
    school_name: Optional[str] = None
    reference_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class NCAAADPosition(BaseModel):
    person_name: Optional[str] = None
    current_title: Optional[str] = None
    institution_name: Optional[str] = None
    reference_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class StateAssociationPosition(BaseModel):
    person_name: Optional[str] = None
    current_title: Optional[str] = None
    organization_name: Optional[str] = None
    reference_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class PositionsExtraction(BaseModel):
    high_school: Optional[HSPosition] = None
    ncaa_ad: Optional[NCAAADPosition] = None
    state_association: Optional[StateAssociationPosition] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
Extract exactly one candidate for each of the following three categories from the answer text. If multiple are mentioned, pick the first clearly valid one. If a field is missing, set it to null. Also extract all URLs cited for that candidate; designate the first as `reference_url` and put any others in `additional_urls`.

1) high_school (a public high school athletic director in Texas):
- person_name
- current_title (e.g., "Athletic Director")
- school_name (public high school)
- reference_url (primary URL that confirms position/qualifications)
- additional_urls (list of any other URLs cited for this high school candidate)

2) ncaa_ad (an NCAA Division I FBS athletic director from a Power Five institution in 2024–2025):
- person_name
- current_title (e.g., "Director of Athletics" / "Athletic Director")
- institution_name
- reference_url (primary URL that confirms position/qualifications)
- additional_urls (list of any other URLs cited for this NCAA candidate)

3) state_association (state-level high school athletic association executive leader):
- person_name
- current_title (e.g., "Executive Director" or "Commissioner")
- organization_name (state high school athletic association)
- reference_url (primary URL that confirms position/qualifications)
- additional_urls (list of any other URLs cited for this state-association candidate)
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(primary: Optional[str], extras: Optional[List[str]]) -> List[str]:
    urls: List[str] = []
    if primary and isinstance(primary, str) and primary.strip():
        urls.append(primary.strip())
    if extras:
        for u in extras:
            if isinstance(u, str) and u.strip():
                if u.strip() not in urls:
                    urls.append(u.strip())
    return urls


def non_empty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and bool(s.strip())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_high_school_position(evaluator: Evaluator, parent) -> None:
    """
    Build and verify nodes for the High School Athletic Director position.
    """
    hs_node = evaluator.add_parallel(
        id="high_school_athletic_director",
        desc="Identify a high school athletic director position meeting specified requirements",
        parent=parent,
        critical=False
    )

    # Get extracted data
    extraction: PositionsExtraction = evaluator.find_node("root")  # just to clarify type; we will access from context
    # Actually retrieve the last recorded extraction result from evaluator._extraction_results
    # But we don't have a public API for that; we rely on closure: we will pass the object in outer scope
    # To keep the function self-contained, we will attach the extracted object to parent via a custom info.
    # However, instead, we will have the caller pass the extracted object.
    # So we refactor signature to accept hs data.
    # This function remains for structure; actual implementation is in the refactored function below.
    return


async def verify_high_school_position_with_data(evaluator: Evaluator, parent, hs: Optional[HSPosition]) -> None:
    node = evaluator.add_parallel(
        id="high_school_athletic_director",
        desc="Identify a high school athletic director position meeting specified requirements",
        parent=parent,
        critical=False
    )

    # Existence: provided fields (non-critical)
    evaluator.add_custom_node(
        result=non_empty_str(hs.person_name if hs else None),
        id="hs_person_name_provided",
        desc="The person's full name is provided",
        parent=node,
        critical=False
    )
    evaluator.add_custom_node(
        result=non_empty_str(hs.current_title if hs else None),
        id="hs_position_title_provided",
        desc="The person's current title is provided",
        parent=node,
        critical=False
    )
    evaluator.add_custom_node(
        result=non_empty_str(hs.school_name if hs else None),
        id="hs_school_name_provided",
        desc="The school name is provided",
        parent=node,
        critical=False
    )

    # Critical: valid reference URL provided (gatekeeper)
    ref_ok = non_empty_str(hs.reference_url if hs else None) and (hs.reference_url.strip().lower().startswith("http") if hs and hs.reference_url else False)
    evaluator.add_custom_node(
        result=bool(ref_ok),
        id="reference_url_hs",
        desc="Valid reference URL provided for the high school athletic director position",
        parent=node,
        critical=True
    )

    # Build common info
    name = hs.person_name if hs and hs.person_name else "the person"
    title = hs.current_title if hs and hs.current_title else "Athletic Director"
    school = hs.school_name if hs and hs.school_name else "the school"
    sources = merge_sources(hs.reference_url if hs else None, hs.additional_urls if hs else [])

    # Critical verifications
    # 1) Position level: public HS, not middle, not district-wide; AD at that school
    pos_level_node = evaluator.add_leaf(
        id="position_level_verification",
        desc="Position is at a public high school level (not middle school, not district-wide)",
        parent=node,
        critical=True
    )
    claim_pos_level = (
        f"This webpage indicates that {name} serves as {title} for {school}, "
        f"which is a public high school (not a middle school and not a district-level position)."
    )
    await evaluator.verify(
        claim=claim_pos_level,
        node=pos_level_node,
        sources=sources,
        additional_instruction="Confirm that the page explicitly ties the person to a high school (e.g., 'High School') and indicates it is a public school. Reject if it appears to be a district office, central office, or middle school role."
    )

    # 2) State location: Texas
    state_node = evaluator.add_leaf(
        id="state_location",
        desc="The high school is located in Texas",
        parent=node,
        critical=True
    )
    claim_state = f"The school {school} is located in Texas."
    await evaluator.verify(
        claim=claim_state,
        node=state_node,
        sources=sources,
        additional_instruction="Accept if the page clearly indicates the school is in Texas (e.g., address, city/state, 'TX', 'Texas', UIL affiliation)."
    )

    # 3) NIAAA certification held or required
    niaaa_node = evaluator.add_leaf(
        id="niaaa_certification",
        desc="The person holds or is required to hold NIAAA certification (RAA, CAA, RMSAA, or CMAA level)",
        parent=node,
        critical=True
    )
    claim_niaaa = (
        f"The page indicates that {name} either holds or is required to hold an NIAAA certification at any level "
        f"(RAA, CAA, RMSAA, or CMAA)."
    )
    await evaluator.verify(
        claim=claim_niaaa,
        node=niaaa_node,
        sources=sources,
        additional_instruction="Accept if the page mentions NIAAA and any of: RAA, CAA, RMSAA, CMAA, or plainly 'NIAAA certification required'."
    )

    # 4) Teaching background
    teaching_node = evaluator.add_leaf(
        id="teaching_background",
        desc="The person has prior teaching experience or holds teaching certification",
        parent=node,
        critical=True
    )
    claim_teaching = (
        f"The page indicates that {name} has prior teaching experience or holds a teaching certification or license."
    )
    await evaluator.verify(
        claim=claim_teaching,
        node=teaching_node,
        sources=sources,
        additional_instruction="Look for phrases like 'teaching experience', 'teacher', 'classroom experience', 'teaching certification', 'teaching license', or equivalent."
    )

    # 5) Reporting structure
    reporting_node = evaluator.add_leaf(
        id="reporting_structure",
        desc="The athletic director reports to a building principal or superintendent",
        parent=node,
        critical=True
    )
    claim_reporting = (
        f"The page indicates that the athletic director reports to a building principal or a superintendent."
    )
    await evaluator.verify(
        claim=claim_reporting,
        node=reporting_node,
        sources=sources,
        additional_instruction="Accept if the page says 'reports to the principal', 'principal's designee', 'superintendent', or similar language indicating direct reporting."
    )


async def verify_ncaa_ad_position_with_data(evaluator: Evaluator, parent, ncaa: Optional[NCAAADPosition]) -> None:
    node = evaluator.add_parallel(
        id="ncaa_division_i_athletic_director",
        desc="Identify an NCAA Division I FBS athletic director position meeting specified requirements",
        parent=parent,
        critical=False
    )

    # Existence: provided fields (non-critical)
    evaluator.add_custom_node(
        result=non_empty_str(ncaa.person_name if ncaa else None),
        id="ncaa_person_name_provided",
        desc="The person's full name is provided",
        parent=node,
        critical=False
    )
    evaluator.add_custom_node(
        result=non_empty_str(ncaa.current_title if ncaa else None),
        id="ncaa_position_title_provided",
        desc="The person's current title is provided",
        parent=node,
        critical=False
    )
    evaluator.add_custom_node(
        result=non_empty_str(ncaa.institution_name if ncaa else None),
        id="ncaa_institution_name_provided",
        desc="The institution name is provided",
        parent=node,
        critical=False
    )

    # Critical: valid reference URL provided (gatekeeper)
    ref_ok = non_empty_str(ncaa.reference_url if ncaa else None) and (ncaa.reference_url.strip().lower().startswith("http") if ncaa and ncaa.reference_url else False)
    evaluator.add_custom_node(
        result=bool(ref_ok),
        id="reference_url_ncaa",
        desc="Valid reference URL provided for the NCAA Division I FBS athletic director",
        parent=node,
        critical=True
    )

    name = ncaa.person_name if ncaa and ncaa.person_name else "the person"
    title = ncaa.current_title if ncaa and ncaa.current_title else "Athletic Director"
    inst = ncaa.institution_name if ncaa and ncaa.institution_name else "the institution"
    sources = merge_sources(ncaa.reference_url if ncaa else None, ncaa.additional_urls if ncaa else [])

    # 1) Administrative title (must be AD/Director of Athletics; not assistant/associate)
    admin_title_node = evaluator.add_leaf(
        id="administrative_title",
        desc="The person holds the title of Director of Athletics or Athletic Director (not assistant or associate)",
        parent=node,
        critical=True
    )
    claim_admin_title = (
        f"The page indicates that {name} holds the top athletics role with the title 'Director of Athletics' or 'Athletic Director' at {inst}, "
        f"and is not labeled as assistant or associate."
    )
    await evaluator.verify(
        claim=claim_admin_title,
        node=admin_title_node,
        sources=sources,
        additional_instruction="Reject if the title includes 'Assistant', 'Associate', or 'Interim Associate'. Accept synonyms like 'Director of Athletics' or 'Athletic Director'."
    )

    # 2) Division classification: NCAA Division I FBS
    division_node = evaluator.add_leaf(
        id="division_classification",
        desc="The institution is classified as NCAA Division I FBS (Football Bowl Subdivision)",
        parent=node,
        critical=True
    )
    claim_division = f"The page indicates that {inst} competes in NCAA Division I FBS (Football Bowl Subdivision)."
    await evaluator.verify(
        claim=claim_division,
        node=division_node,
        sources=sources,
        additional_instruction="Accept if the page explicitly mentions 'FBS', 'Football Bowl Subdivision', or equivalent wording that clearly indicates FBS status."
    )

    # 3) Conference affiliation: Power Five (ACC, Big Ten, Big 12, Pac-12, SEC) as of 2024–2025
    conference_node = evaluator.add_leaf(
        id="conference_affiliation",
        desc="The institution is a member of a Power Five conference (ACC, Big Ten, Big 12, Pac-12, or SEC) as of 2024-2025",
        parent=node,
        critical=True
    )
    claim_conf = (
        f"The page indicates that {inst} is a member of one of these conferences: ACC, Big Ten, Big 12, Pac-12, or SEC."
    )
    await evaluator.verify(
        claim=claim_conf,
        node=conference_node,
        sources=sources,
        additional_instruction="Consider the claim satisfied if the page explicitly states membership in ACC, Big Ten, Big 12, Pac-12, or SEC. Focus on explicit mentions on the page."
    )

    # 4) Education level: at least Master's
    edu_node = evaluator.add_leaf(
        id="education_level",
        desc="The athletic director holds at least a Master's degree",
        parent=node,
        critical=True
    )
    claim_edu = (
        f"The page indicates that {name} holds at least a Master's degree (e.g., M.A., M.S., MBA, M.Ed., or a doctorate/Ph.D.)."
    )
    await evaluator.verify(
        claim=claim_edu,
        node=edu_node,
        sources=sources,
        additional_instruction="Accept any Master's or higher degree. Reject if only a Bachelor's is mentioned without any higher degree."
    )

    # 5) Previous collegiate athletic administration experience
    prev_exp_node = evaluator.add_leaf(
        id="previous_experience",
        desc="The athletic director has documented previous experience in athletic administration (associate AD, senior associate AD, or equivalent)",
        parent=node,
        critical=True
    )
    claim_prev = (
        f"The page indicates that {name} previously held a collegiate athletic administration role such as associate AD, senior associate AD, deputy AD, or an equivalent position."
    )
    await evaluator.verify(
        claim=claim_prev,
        node=prev_exp_node,
        sources=sources,
        additional_instruction="Look for terms like 'Associate AD', 'Senior Associate AD', 'Deputy AD', 'Compliance Director', 'External Operations' at the collegiate level before current role."
    )


async def verify_state_association_position_with_data(evaluator: Evaluator, parent, st: Optional[StateAssociationPosition]) -> None:
    node = evaluator.add_parallel(
        id="state_association_leadership",
        desc="Identify a state high school athletic association executive director or commissioner meeting specified requirements",
        parent=parent,
        critical=False
    )

    # Existence: provided fields (non-critical)
    evaluator.add_custom_node(
        result=non_empty_str(st.person_name if st else None),
        id="state_person_name_provided",
        desc="The person's full name is provided",
        parent=node,
        critical=False
    )
    evaluator.add_custom_node(
        result=non_empty_str(st.current_title if st else None),
        id="state_position_title_provided",
        desc="The person's current title is provided",
        parent=node,
        critical=False
    )
    evaluator.add_custom_node(
        result=non_empty_str(st.organization_name if st else None),
        id="organization_name_provided",
        desc="The organization name is provided",
        parent=node,
        critical=False
    )

    # Critical: valid reference URL provided (gatekeeper)
    ref_ok = non_empty_str(st.reference_url if st else None) and (st.reference_url.strip().lower().startswith("http") if st and st.reference_url else False)
    evaluator.add_custom_node(
        result=bool(ref_ok),
        id="reference_url_state",
        desc="Valid reference URL provided for the state association leadership position",
        parent=node,
        critical=True
    )

    name = st.person_name if st and st.person_name else "the person"
    title = st.current_title if st and st.current_title else "Executive Director"
    org = st.organization_name if st and st.organization_name else "the organization"
    sources = merge_sources(st.reference_url if st else None, st.additional_urls if st else [])

    # 1) Organization type: state-level HS athletic association
    org_type_node = evaluator.add_leaf(
        id="organization_type",
        desc="The position is with a state-level high school athletic association (e.g., UIL Texas, CIF California, NFHS state association)",
        parent=node,
        critical=True
    )
    claim_org_type = (
        f"The page indicates that {org} is a state-level high school athletic association (covering an entire state)."
    )
    await evaluator.verify(
        claim=claim_org_type,
        node=org_type_node,
        sources=sources,
        additional_instruction="Accept if the page clearly states the organization is a statewide governing body for high school athletics (e.g., UIL, OHSAA, GHSA). Reject if it's regional or district-level."
    )

    # 2) Leadership role: top executive (Executive Director / Commissioner)
    leader_node = evaluator.add_leaf(
        id="leadership_role",
        desc="The person holds an executive director, commissioner, or equivalent top leadership position",
        parent=node,
        critical=True
    )
    claim_leader = (
        f"The page indicates that {name} is the top executive leader (e.g., Executive Director, Commissioner, or equivalent) of {org}."
    )
    await evaluator.verify(
        claim=claim_leader,
        node=leader_node,
        sources=sources,
        additional_instruction="Accept synonyms such as 'Executive Director', 'Commissioner', 'Chief Executive' if it is clearly the top leadership role."
    )

    # 3) Geographic scope: statewide governance
    scope_node = evaluator.add_leaf(
        id="geographic_scope",
        desc="The organization governs high school athletics at the state level (not regional or local)",
        parent=node,
        critical=True
    )
    claim_scope = f"The page indicates that {org} governs high school athletics statewide (state-level scope)."
    await evaluator.verify(
        claim=claim_scope,
        node=scope_node,
        sources=sources,
        additional_instruction="Look for language indicating statewide jurisdiction, statewide governance, or statewide membership of high schools."
    )

    # 4) Prior athletic administration experience (HS or collegiate)
    prior_exp_node = evaluator.add_leaf(
        id="prior_athletic_admin_experience",
        desc="The person has documented prior experience in athletic administration at the high school or collegiate level",
        parent=node,
        critical=True
    )
    claim_priorexp = (
        f"The page indicates that {name} previously worked in athletic administration at either the high school or collegiate level before this role."
    )
    await evaluator.verify(
        claim=claim_priorexp,
        node=prior_exp_node,
        sources=sources,
        additional_instruction="Accept roles such as AD, assistant/associate AD, athletic administrator, director of athletics at HS or college, or similar administrative posts before this appointment."
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
    Evaluate an answer for the tri-level athletic administration leadership task.
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
        default_model=model
    )

    # Extract structured info
    positions: PositionsExtraction = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Build verification subtrees
    await verify_high_school_position_with_data(evaluator, root, positions.high_school if positions else None)
    await verify_ncaa_ad_position_with_data(evaluator, root, positions.ncaa_ad if positions else None)
    await verify_state_association_position_with_data(evaluator, root, positions.state_association if positions else None)

    # Return summary
    return evaluator.get_summary()