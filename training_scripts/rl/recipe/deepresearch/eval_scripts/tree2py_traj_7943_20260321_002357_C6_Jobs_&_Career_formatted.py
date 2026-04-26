import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "osu_admin_team_march_2026"
TASK_DESCRIPTION = (
    "For The Ohio State University, a Big Ten Conference FBS institution, identify the current holders "
    "(as of March 2026) of the following five key administrative positions:\n\n"
    "1. Athletic Director - the chief administrator of the athletics department\n"
    "2. Head Football Coach - the head coach of the varsity football program\n"
    "3. Provost - the chief academic officer (may have title Provost, Executive Vice President and Provost, or Vice President for Academic Affairs)\n"
    "4. Dean of the College of Engineering - the chief administrator of the engineering college\n"
    "5. Senior Associate Athletic Director for Compliance or equivalent - the senior administrator responsible for NCAA compliance within the athletics department\n\n"
    "For each of the five positions, provide the following information:\n"
    "- Full name of the current position holder\n"
    "- Highest educational degree attained (Bachelor's, Master's, Doctoral/PhD, JD, etc.)\n"
    "- Documented prior relevant experience - provide specific evidence of their career background in relevant fields\n"
    "- Direct supervisor/reporting relationship - identify the position to which this role reports\n"
    "- Reference URL - provide at least one official Ohio State University webpage or credible news source that confirms the person holds this position and provides biographical information\n\n"
    "All five positions must be correctly identified with complete information as specified above. "
    "Each position's information must be supported by verifiable reference URLs from official university sources or credible media outlets."
)


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class PositionInfo(BaseModel):
    name: Optional[str] = None
    degree_highest_level: Optional[str] = None  # e.g., Bachelor's, Master's, Doctoral/PhD, JD, MD, EdD
    degree_detail: Optional[str] = None         # e.g., PhD in X from Y University (include field/institution if provided)
    prior_experience: List[str] = Field(default_factory=list)  # list of role summaries
    reporting_to: Optional[str] = None          # e.g., University President, Athletic Director, Provost
    urls: List[str] = Field(default_factory=list)  # reference URLs explicitly cited in the answer


class OSULeadershipExtraction(BaseModel):
    athletic_director: Optional[PositionInfo] = None
    head_football_coach: Optional[PositionInfo] = None
    provost: Optional[PositionInfo] = None
    engineering_dean: Optional[PositionInfo] = None
    compliance_director: Optional[PositionInfo] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_osu_leadership() -> str:
    return """
Extract structured information for five Ohio State University roles as they appear in the answer text (as of March 2026). Do NOT invent any details—only extract what is explicitly present.

For each role, extract:
- name: Full name of the current position holder (string; null if not given).
- degree_highest_level: One of [Bachelor's, Master's, Doctoral/PhD, JD, MD, EdD, Other] if stated (string; null if not given).
- degree_detail: The most specific phrasing of the highest degree including field and institution if present (string; null if not given).
- prior_experience: An array of bullet-like strings summarizing prior positions or roles relevant to the role (may be empty if not given).
- reporting_to: The position this role reports to if explicitly stated (string; null if not present).
- urls: An array of URLs explicitly cited in the answer that substantiate identity and/or bio details for this person. Include only valid URLs from the answer (can be empty if none cited).

Roles (use these exact JSON keys):
- athletic_director
- head_football_coach
- provost
- engineering_dean
- compliance_director

Important:
- Only extract URLs explicitly present in the answer. If none are present for a role, return an empty list for urls.
- Preserve names and degree wording exactly as in the answer (aside from normalizing whitespace).
- If multiple URLs are listed, include all of them.
- If the answer provides multiple candidates for a role, select the one clearly indicated as current; otherwise return the first one presented.

Return a single JSON object with these five top-level fields, each mapping to a PositionInfo-like object as described.
""".strip()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _norm_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if isinstance(u, str):
            s = u.strip()
            if s:
                out.append(s)
    return out


def _short_list(items: List[str], max_items: int = 2) -> str:
    return "; ".join(items[:max_items]) if items else ""


def _has_minimal_info(info: Optional[PositionInfo]) -> bool:
    return bool(info and info.name and info.name.strip() and info.urls and len(info.urls) > 0)


# -----------------------------------------------------------------------------
# Verification logic per role
# -----------------------------------------------------------------------------
async def verify_role(
    evaluator: Evaluator,
    root_node,
    *,
    position_parent_id: str,
    position_parent_desc: str,
    id_prefix: str,
    display_title: str,
    info: Optional[PositionInfo],
    identification_leaf_id: str,
    qualifications_node_id: str,
    education_group_id: str,
    education_leaves: List[Dict[str, str]],  # [{"id": str, "mode": "atleast_bachelor"|"terminal"|"specific_detail"|"specific_field_institution", "desc": str}]
    experience_group_id: str,
    experience_leaves: List[Dict[str, str]],  # role-specific experience checks with id and "mode"
    reporting_leaf_id: str,
    reporting_expected_text: Optional[str],   # expected supervisor position text for normative check; can be None to skip normative phrasing
    url_reference_leaf_id: str,
) -> None:
    """
    Build the verification subtree for a single role using the given configuration.
    """
    # Create the sequential parent node for this position
    pos_node = evaluator.add_sequential(
        id=position_parent_id,
        desc=position_parent_desc,
        parent=root_node,
        critical=False  # allow partial credit across positions at root level
    )

    # Gate: ensure we have at least a name and one URL, otherwise block subsequent checks.
    info_present_node = evaluator.add_custom_node(
        result=_has_minimal_info(info),
        id=f"{id_prefix}_info_present",
        desc=f"{display_title}: name and at least one source URL are provided in the answer",
        parent=pos_node,
        critical=True
    )

    # Identification (critical)
    ident_leaf = evaluator.add_leaf(
        id=identification_leaf_id,
        desc=f"Correct identification of current {display_title} by name",
        parent=pos_node,
        critical=True
    )
    name = info.name if info and info.name else ""
    ident_claim = (
        f"{name} is the current {display_title} at The Ohio State University as of March 2026."
    )
    await evaluator.verify(
        claim=ident_claim,
        node=ident_leaf,
        sources=_norm_urls(info.urls if info else []),
        additional_instruction=(
            "Confirm that the page clearly states this person currently holds the specified role at The Ohio State University. "
            "Accept equivalent titles (e.g., 'Executive Vice President and Provost' for Provost; 'Senior Associate AD for Compliance' for Compliance head). "
            "If the page is outdated or does not clearly confirm the appointment, mark as not supported."
        ),
    )

    # Qualifications container (non-critical, parallel) to allow partial scoring inside
    qual_node = evaluator.add_parallel(
        id=qualifications_node_id,
        desc=f"Verification of {display_title}'s qualifications and biographical information",
        parent=pos_node,
        critical=False
    )

    # Education (critical group)
    edu_node = evaluator.add_parallel(
        id=education_group_id,
        desc=f"Verification of {display_title}'s educational credentials",
        parent=qual_node,
        critical=True
    )

    urls = _norm_urls(info.urls if info else [])
    degree_level = (info.degree_highest_level or "").strip() if info else ""
    degree_detail = (info.degree_detail or "").strip() if info else ""

    # Build education leaves based on config
    for leaf_cfg in education_leaves:
        leaf = evaluator.add_leaf(
            id=leaf_cfg["id"],
            desc=leaf_cfg.get("desc", ""),
            parent=edu_node,
            critical=True
        )
        mode = leaf_cfg.get("mode", "")
        if mode == "atleast_bachelor":
            claim = f"{name} holds at least a bachelor's degree."
            add_ins = (
                "Verify from the provided sources that the person has completed a bachelor's degree or higher "
                "(e.g., BA/BS, MA/MS, MBA, JD, MD, PhD/EdD)."
            )
        elif mode == "terminal":
            claim = f"{name} holds a doctoral or terminal degree (e.g., PhD, EdD, JD, MD, DSc)."
            add_ins = (
                "Confirm that the sources indicate a doctoral/terminal degree such as PhD, EdD, JD, MD, DSc, or equivalent."
            )
        elif mode == "specific_detail":
            # Check that a specific highest degree is provided/confirmable
            specific_text = degree_detail if degree_detail else degree_level
            claim = f"The sources specify {name}'s highest degree: {specific_text}."
            add_ins = (
                "Confirm that the sources provide the specific highest degree (e.g., level and, if available, major/field and/or awarding institution). "
                "Minor wording differences are acceptable."
            )
        elif mode == "specific_field_institution":
            claim = f"The sources specify the doctoral degree field and awarding institution for {name}."
            add_ins = (
                "Check that the pages explicitly mention the doctoral (or terminal) degree field/discipline and the awarding institution."
            )
        else:
            claim = f"The sources provide adequate confirmation of {name}'s educational credentials."
            add_ins = "Use the sources to confirm the stated educational detail."

        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=add_ins
        )

    # Experience (critical group)
    exp_node = evaluator.add_parallel(
        id=experience_group_id,
        desc=f"Verification of {display_title}'s professional experience",
        parent=qual_node,
        critical=True
    )
    exp_list_text = _short_list(info.prior_experience if info else [], 3)

    for leaf_cfg in experience_leaves:
        leaf = evaluator.add_leaf(
            id=leaf_cfg["id"],
            desc=leaf_cfg.get("desc", ""),
            parent=exp_node,
            critical=True
        )
        mode = leaf_cfg.get("mode", "")
        if mode == "prior_positions_generic":
            claim = (
                f"The sources document at least one prior role for {name} relevant to {display_title} responsibilities. "
                f"Examples cited: {exp_list_text}"
            )
            add_ins = (
                "Look for a 'Bio' or 'About' page listing prior positions. At least one substantive prior role should be clearly stated."
            )
        elif mode == "prior_positions_ath_admin":
            claim = (
                f"The sources document prior positions for {name} in athletics administration or a closely related field. "
                f"Examples cited: {exp_list_text}"
            )
            add_ins = (
                "Confirm the biography lists roles such as AD, deputy/associate AD, NCAA/governance roles, or comparable athletics administration experience."
            )
        elif mode == "experience_relevance_ath_admin":
            claim = f"{name}'s prior experience is relevant to athletics administration."
            add_ins = "Confirm the nature of prior roles aligns with athletics administration responsibilities."
        elif mode == "prior_positions_coaching":
            claim = (
                f"The sources document prior collegiate/professional coaching roles for {name}. "
                f"Examples cited: {exp_list_text}"
            )
            add_ins = "Confirm coaching titles (e.g., head coach, coordinator, position coach) and institutions/teams."
        elif mode == "experience_details_coaching":
            claim = f"The sources specify coaching roles and institutions previously held by {name}."
            add_ins = "Look for explicit role titles and the corresponding institutions/teams."
        elif mode == "faculty_background":
            claim = f"{name} has documented faculty experience or an academic rank."
            add_ins = "Check for mentions of professorships, faculty appointments, tenure, or similar academic ranks."
        elif mode == "admin_experience_academic":
            claim = f"{name} has prior academic administrative roles."
            add_ins = "Look for roles such as chair, dean, associate/vice provost/president, or similar academic administration positions."
        elif mode == "academic_background_generic":
            claim = f"{name} has documented prior academic experience."
            add_ins = "Confirm faculty roles, research leadership, or similar academic activities."
        elif mode == "leadership_experience_generic":
            claim = f"{name} has prior administrative or leadership experience relevant to serving as {display_title}."
            add_ins = "Check for prior management/leadership titles in academic or related organizations."
        elif mode == "prior_positions_compliance":
            claim = (
                f"The sources document prior roles for {name} in NCAA compliance or athletics administration. "
                f"Examples cited: {exp_list_text}"
            )
            add_ins = "Confirm compliance titles (e.g., Director of Compliance, Associate AD for Compliance) or closely related roles."
        elif mode == "experience_relevance_compliance":
            claim = f"{name}'s prior experience is relevant to NCAA compliance or athletics administration."
            add_ins = "Check alignment of past roles with compliance responsibilities."
        else:
            claim = f"The sources document relevant prior experience for {name}."
            add_ins = "Use the sources to confirm the experience."

        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=add_ins
        )

    # Reporting relationship (critical leaf under qualifications)
    rep_leaf = evaluator.add_leaf(
        id=reporting_leaf_id,
        desc=f"{display_title} reporting relationship is correct",
        parent=qual_node,
        critical=True
    )
    if reporting_expected_text:
        rep_claim = (
            f"The {display_title} reports to the {reporting_expected_text} at The Ohio State University."
        )
        rep_add_ins = (
            "Prefer explicit statements in official org charts or biographies. "
            "If the reporting line is not clearly stated on the sources, mark as not supported."
        )
    else:
        # Fall back to the extracted reporting_to if provided
        extracted_rep = (info.reporting_to or "").strip() if info else ""
        rep_claim = (
            f"According to the sources, the {display_title} reports to '{extracted_rep}'."
        )
        rep_add_ins = (
            "Check that the sources state this reporting line. If not explicitly stated, mark as not supported."
        )

    await evaluator.verify(
        claim=rep_claim,
        node=rep_leaf,
        sources=urls,
        additional_instruction=rep_add_ins
    )

    # URL reference validity (critical leaf under qualifications)
    url_leaf = evaluator.add_leaf(
        id=url_reference_leaf_id,
        desc=f"Valid URL reference provided for {display_title} from official or credible source",
        parent=qual_node,
        critical=True
    )
    url_claim = (
        f"This webpage confirms that {name} is the {display_title} at The Ohio State University and provides biographical information."
    )
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=urls,
        additional_instruction=(
            "Accept OSU official websites (e.g., osu.edu, ohiostatebuckeyes.com) and credible media outlets. "
            "The page must both confirm the role and include some bio details (education, experience)."
        )
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
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
    """
    Evaluate the agent's answer for OSU administrative leadership (as of March 2026).
    """
    evaluator = Evaluator()

    # Important: set root as non-critical to allow partial credit across positions
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_osu_leadership(),
        template_class=OSULeadershipExtraction,
        extraction_name="osu_leadership_extraction"
    )

    # Build verification subtrees for each role

    # 1) Athletic Director
    await verify_role(
        evaluator,
        root,
        position_parent_id="athletic_director_position",
        position_parent_desc="Verification of Athletic Director position holder and qualifications",
        id_prefix="ad",
        display_title="Athletic Director",
        info=extracted.athletic_director,
        identification_leaf_id="ad_identification",
        qualifications_node_id="ad_qualifications",
        education_group_id="ad_education",
        education_leaves=[
            {
                "id": "ad_degree_level",
                "mode": "atleast_bachelor",
                "desc": "Athletic Director holds at least a bachelor's degree"
            },
            {
                "id": "ad_degree_specification",
                "mode": "specific_detail",
                "desc": "Specific degree level is provided (bachelor's, master's, or doctoral)"
            }
        ],
        experience_group_id="ad_experience",
        experience_leaves=[
            {
                "id": "ad_prior_positions",
                "mode": "prior_positions_ath_admin",
                "desc": "Athletic Director has documented prior positions in athletics administration or related field"
            },
            {
                "id": "ad_experience_relevance",
                "mode": "experience_relevance_ath_admin",
                "desc": "Prior experience is relevant to athletic administration"
            }
        ],
        reporting_leaf_id="ad_reporting",
        reporting_expected_text="University President",
        url_reference_leaf_id="ad_url_reference"
    )

    # 2) Head Football Coach
    await verify_role(
        evaluator,
        root,
        position_parent_id="head_football_coach_position",
        position_parent_desc="Verification of Head Football Coach position holder and qualifications",
        id_prefix="coach",
        display_title="Head Football Coach",
        info=extracted.head_football_coach,
        identification_leaf_id="coach_identification",
        qualifications_node_id="coach_qualifications",
        education_group_id="coach_education",
        education_leaves=[
            {
                "id": "coach_degree_level",
                "mode": "atleast_bachelor",
                "desc": "Head Football Coach holds at least a bachelor's degree"
            },
            {
                "id": "coach_degree_specification",
                "mode": "specific_detail",
                "desc": "Specific degree level and field are provided"
            }
        ],
        experience_group_id="coach_experience",
        experience_leaves=[
            {
                "id": "coach_prior_positions",
                "mode": "prior_positions_coaching",
                "desc": "Head Football Coach has documented prior coaching positions at collegiate or professional level"
            },
            {
                "id": "coach_experience_details",
                "mode": "experience_details_coaching",
                "desc": "Specific coaching roles and institutions are provided"
            }
        ],
        reporting_leaf_id="coach_reporting",
        reporting_expected_text="Athletic Director",
        url_reference_leaf_id="coach_url_reference"
    )

    # 3) Provost
    await verify_role(
        evaluator,
        root,
        position_parent_id="provost_position",
        position_parent_desc="Verification of Provost or Chief Academic Officer position holder and qualifications",
        id_prefix="provost",
        display_title="Provost",
        info=extracted.provost,
        identification_leaf_id="provost_identification",
        qualifications_node_id="provost_qualifications",
        education_group_id="provost_education",
        education_leaves=[
            {
                "id": "provost_terminal_degree",
                "mode": "terminal",
                "desc": "Provost holds a doctoral or terminal degree"
            },
            {
                "id": "provost_degree_field",
                "mode": "specific_field_institution",
                "desc": "Specific doctoral degree field and institution are provided"
            }
        ],
        experience_group_id="provost_experience",
        experience_leaves=[
            {
                "id": "provost_faculty_background",
                "mode": "faculty_background",
                "desc": "Provost has documented faculty experience or held academic rank"
            },
            {
                "id": "provost_admin_experience",
                "mode": "admin_experience_academic",
                "desc": "Provost has documented prior academic administrative roles"
            }
        ],
        reporting_leaf_id="provost_reporting",
        reporting_expected_text="University President",
        url_reference_leaf_id="provost_url_reference"
    )

    # 4) Dean of the College of Engineering
    await verify_role(
        evaluator,
        root,
        position_parent_id="dean_position",
        position_parent_desc="Verification of College of Engineering Dean position holder and qualifications",
        id_prefix="dean",
        display_title="Dean of the College of Engineering",
        info=extracted.engineering_dean,
        identification_leaf_id="dean_identification",
        qualifications_node_id="dean_qualifications",
        education_group_id="dean_education",
        education_leaves=[
            {
                "id": "dean_terminal_degree",
                "mode": "terminal",
                "desc": "Dean holds a doctoral or terminal degree"
            },
            {
                "id": "dean_degree_field",
                "mode": "specific_field_institution",
                "desc": "Specific doctoral degree field and institution are provided"
            }
        ],
        experience_group_id="dean_experience",
        experience_leaves=[
            {
                "id": "dean_academic_background",
                "mode": "academic_background_generic",
                "desc": "Dean has documented prior academic experience"
            },
            {
                "id": "dean_leadership_experience",
                "mode": "leadership_experience_generic",
                "desc": "Dean has documented prior administrative or leadership experience"
            }
        ],
        reporting_leaf_id="dean_reporting",
        reporting_expected_text="Provost",
        url_reference_leaf_id="dean_url_reference"
    )

    # 5) Senior Associate AD for Compliance (or equivalent)
    await verify_role(
        evaluator,
        root,
        position_parent_id="compliance_director_position",
        position_parent_desc="Verification of Senior Associate Athletic Director for Compliance position holder and qualifications",
        id_prefix="compliance",
        display_title="Senior Associate Athletic Director for Compliance",
        info=extracted.compliance_director,
        identification_leaf_id="compliance_identification",
        qualifications_node_id="compliance_qualifications",
        education_group_id="compliance_education",
        education_leaves=[
            {
                "id": "compliance_degree_level",
                "mode": "atleast_bachelor",
                "desc": "Compliance Director holds at least a bachelor's degree"
            },
            {
                "id": "compliance_degree_specification",
                "mode": "specific_detail",
                "desc": "Specific degree level is provided"
            }
        ],
        experience_group_id="compliance_experience",
        experience_leaves=[
            {
                "id": "compliance_prior_positions",
                "mode": "prior_positions_compliance",
                "desc": "Compliance Director has documented prior positions in NCAA compliance or athletics administration"
            },
            {
                "id": "compliance_experience_relevance",
                "mode": "experience_relevance_compliance",
                "desc": "Prior experience is relevant to NCAA compliance or athletics administration"
            }
        ],
        reporting_leaf_id="compliance_reporting",
        reporting_expected_text="Athletic Director",
        url_reference_leaf_id="compliance_url_reference"
    )

    return evaluator.get_summary()