import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "career_min_qual_eval_k12_college"
TASK_DESCRIPTION = (
    "A professional currently working in college athletics has the following credentials: "
    "a bachelor's degree in Physical Education (earned 2001), a master's degree in Athletic Administration (earned 2004), "
    "12 years of college football coaching experience at the Division I FCS level including 5 years as offensive coordinator, "
    "no K-12 teaching certification, and no principal/school administrator certification. "
    "This person is exploring career opportunities and needs to determine which of the following five positions they currently meet the minimum qualifications for, "
    "based on standard requirements in the United States: "
    "(1) Assistant Principal at a public K-12 school, "
    "(2) Principal at a public K-12 school, "
    "(3) Athletic Director at a public school district, "
    "(4) Assistant Football Coach at a Division I FCS college program, and "
    "(5) Head Football Coach at a Division I FCS college program. "
    "For each position, indicate whether the person meets the minimum qualifications (YES or NO) and identify which specific requirements they meet or lack "
    "(education, experience, certification). Provide supporting evidence with reference URLs for the requirements of each position type."
)

PERSON_CREDENTIALS_SUMMARY = (
    "Person's credentials: Bachelor's in Physical Education (2001); "
    "Master's in Athletic Administration (2004); "
    "12 years of Division I FCS college football coaching experience including 5 years as offensive coordinator; "
    "no K-12 teaching certification; no principal/administrator certification."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AssistantPrincipalInfo(BaseModel):
    decision: Optional[str] = None  # "YES" or "NO"
    masters_edu_related: Optional[str] = None  # "YES" or "NO" (meets master's-in-education-related requirement)
    k12_teaching_cert: Optional[str] = None  # "YES" or "NO" (has valid K-12 teaching certification)
    k12_teaching_experience: Optional[str] = None  # "YES" or "NO" (≥3 years K-12 teaching experience)
    reference_urls: List[str] = Field(default_factory=list)


class PrincipalInfo(BaseModel):
    decision: Optional[str] = None  # "YES" or "NO"
    masters_field: Optional[str] = None  # "YES" or "NO" (master's in educational leadership/administration)
    k12_teaching_cert: Optional[str] = None  # "YES" or "NO"
    k12_teaching_experience: Optional[str] = None  # "YES" or "NO" (3–5 years K-12 teaching experience)
    admin_cert: Optional[str] = None  # "YES" or "NO" (principal/administrator certification)
    reference_urls: List[str] = Field(default_factory=list)


class AthleticDirectorInfo(BaseModel):
    decision: Optional[str] = None  # "YES" or "NO"
    bachelors_min: Optional[str] = None  # "YES" or "NO" (at least a bachelor's as minimum)
    relevant_experience: Optional[str] = None  # "YES" or "NO" (relevant coaching/athletics admin experience)
    teaching_cert_variability_noted: Optional[bool] = None  # whether the answer explicitly notes variability and impact
    reference_urls: List[str] = Field(default_factory=list)


class AssistantCoachInfo(BaseModel):
    decision: Optional[str] = None  # "YES" or "NO"
    bachelors: Optional[str] = None  # "YES" or "NO"
    coaching_or_playing_experience: Optional[str] = None  # "YES" or "NO" (≥1–2 yrs coll/pro coaching OR significant college playing)
    certification_addressed: Optional[str] = None  # "YES" or "NO" (answer addresses certification/licensure requirement per sources)
    reference_urls: List[str] = Field(default_factory=list)


class HeadCoachInfo(BaseModel):
    decision: Optional[str] = None  # "YES" or "NO"
    bachelors: Optional[str] = None  # "YES" or "NO"
    experience_including_coordinator_or_head: Optional[str] = None  # "YES" or "NO"
    certification_addressed: Optional[str] = None  # "YES" or "NO"
    reference_urls: List[str] = Field(default_factory=list)


class CareerQualificationsExtraction(BaseModel):
    assistant_principal: Optional[AssistantPrincipalInfo] = None
    principal: Optional[PrincipalInfo] = None
    athletic_director: Optional[AthleticDirectorInfo] = None
    assistant_coach: Optional[AssistantCoachInfo] = None
    head_coach: Optional[HeadCoachInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all_positions() -> str:
    return """
    Extract, from the provided answer, the explicit YES/NO decisions and requirement-by-requirement evaluations for each of the five positions below.
    IMPORTANT:
    - decision values must be exactly "YES" or "NO" (uppercase).
    - For each requirement that asks if it is met, return exactly "YES" or "NO".
    - reference_urls must contain ONLY URLs explicitly included in the answer; if none are present, return an empty array.
    - For 'teaching_cert_variability_noted', return a boolean: true if the answer explicitly states that teaching certification requirements vary by district and explains how lack of certification could affect eligibility; otherwise false.

    Return a single JSON object using this schema:

    {
      "assistant_principal": {
        "decision": "YES" | "NO" | null,
        "masters_edu_related": "YES" | "NO" | null,
        "k12_teaching_cert": "YES" | "NO" | null,
        "k12_teaching_experience": "YES" | "NO" | null,
        "reference_urls": [<urls>]
      },
      "principal": {
        "decision": "YES" | "NO" | null,
        "masters_field": "YES" | "NO" | null,
        "k12_teaching_cert": "YES" | "NO" | null,
        "k12_teaching_experience": "YES" | "NO" | null,
        "admin_cert": "YES" | "NO" | null,
        "reference_urls": [<urls>]
      },
      "athletic_director": {
        "decision": "YES" | "NO" | null,
        "bachelors_min": "YES" | "NO" | null,
        "relevant_experience": "YES" | "NO" | null,
        "teaching_cert_variability_noted": true | false | null,
        "reference_urls": [<urls>]
      },
      "assistant_coach": {
        "decision": "YES" | "NO" | null,
        "bachelors": "YES" | "NO" | null,
        "coaching_or_playing_experience": "YES" | "NO" | null,
        "certification_addressed": "YES" | "NO" | null,
        "reference_urls": [<urls>]
      },
      "head_coach": {
        "decision": "YES" | "NO" | null,
        "bachelors": "YES" | "NO" | null,
        "experience_including_coordinator_or_head": "YES" | "NO" | null,
        "certification_addressed": "YES" | "NO" | null,
        "reference_urls": [<urls>]
      }
    }

    Notes:
    - Do not infer any URLs. Only include URLs that are explicitly present in the answer, including markdown links.
    - If a position is not discussed in the answer, set its object to null or include nulls for its fields and an empty array for reference_urls.
    - Normalize any 'met'/'not met' or similar wording to strict YES/NO values as requested above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and len(u.strip()) > 0]


def _yn(value: Optional[str]) -> str:
    v = (value or "").strip().upper()
    return v if v in {"YES", "NO"} else "UNKNOWN"


def _claim_for_requirement(position_name: str, requirement_label: str, extracted_eval: Optional[str]) -> str:
    yn = _yn(extracted_eval)
    return (
        f"For the {position_name} role, the answer evaluates the minimum requirement '{requirement_label}' as '{yn}'. "
        f"{PERSON_CREDENTIALS_SUMMARY} "
        f"Judge whether this evaluation is correct based on the referenced source page(s) that state the minimum requirements for this role."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_assistant_principal(evaluator: Evaluator, parent_node, ap: Optional[AssistantPrincipalInfo]) -> None:
    node = evaluator.add_parallel(
        id="position_1_assistant_principal",
        desc="K-12 Assistant Principal (public school): decide YES/NO against the stated minimum requirements and cite sources.",
        parent=parent_node,
        critical=False
    )

    # Decision presence (critical)
    evaluator.add_custom_node(
        result=bool(ap and ap.decision and ap.decision.strip().upper() in {"YES", "NO"}),
        id="ap_decision_yes_no",
        desc="States an explicit qualification decision (YES or NO) for Assistant Principal.",
        parent=node,
        critical=True
    )

    # Reference URLs presence (critical)
    ap_urls = _safe_urls(ap.reference_urls if ap else [])
    evaluator.add_custom_node(
        result=len(ap_urls) > 0,
        id="ap_reference_urls",
        desc="Provides at least one reference URL supporting the Assistant Principal minimum requirements used.",
        parent=node,
        critical=True
    )

    # Requirement: Master's in an education-related field
    leaf = evaluator.add_leaf(
        id="ap_requirement_masters_edu_related",
        desc="Correctly evaluates whether the person meets the stated requirement: master's degree in an education-related field (and notes if not met).",
        parent=node,
        critical=True
    )
    claim = _claim_for_requirement(
        "public K-12 Assistant Principal",
        "Master's in education/educational leadership/administration or closely related field",
        ap.masters_edu_related if ap else None
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ap_urls,
        additional_instruction=(
            "From the provided source URL(s), identify the minimum education requirement for public K-12 Assistant Principals. "
            "Then, evaluate whether a 'Master's in Athletic Administration (2004)' reasonably satisfies a 'master's in education/educational leadership/administration or a closely related field' requirement. "
            "If the sources state an explicit master's requirement in education/edu leadership/administration (or equivalent), judge the answer's YES/NO evaluation accordingly. "
            "Base your judgment strictly on the URLs; use the person's credentials (given in the task) to assess fit."
        )
    )

    # Requirement: Valid K-12 teaching certification
    leaf = evaluator.add_leaf(
        id="ap_requirement_k12_teaching_cert",
        desc="Correctly evaluates whether the person meets the stated requirement: valid K-12 teaching certification (and notes if not met).",
        parent=node,
        critical=True
    )
    claim = _claim_for_requirement(
        "public K-12 Assistant Principal",
        "Valid K-12 teaching certification/license",
        ap.k12_teaching_cert if ap else None
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ap_urls,
        additional_instruction=(
            "Use the URLs to determine whether a K-12 teaching license is typically required for Assistant Principal roles in public schools. "
            "Given the person has no K-12 teaching certification, judge whether the answer's evaluation (YES/NO) is correct."
        )
    )

    # Requirement: ≥3 years K-12 teaching experience
    leaf = evaluator.add_leaf(
        id="ap_requirement_k12_teaching_experience",
        desc="Correctly evaluates whether the person meets the stated requirement: ≥3 years of K-12 teaching experience (and notes if not met).",
        parent=node,
        critical=True
    )
    claim = _claim_for_requirement(
        "public K-12 Assistant Principal",
        "At least 3 years of K-12 classroom teaching experience",
        ap.k12_teaching_experience if ap else None
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ap_urls,
        additional_instruction=(
            "From the sources, identify whether public K-12 Assistant Principal roles typically require several years of K-12 teaching experience (often 3+). "
            "The person has college coaching but no K-12 teaching experience; judge if the answer's evaluation (YES/NO) is correct."
        )
    )


async def verify_principal(evaluator: Evaluator, parent_node, prin: Optional[PrincipalInfo]) -> None:
    node = evaluator.add_parallel(
        id="position_2_principal",
        desc="K-12 Principal (public school): decide YES/NO against the stated minimum requirements and cite sources.",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(prin and prin.decision and prin.decision.strip().upper() in {"YES", "NO"}),
        id="prin_decision_yes_no",
        desc="States an explicit qualification decision (YES or NO) for Principal.",
        parent=node,
        critical=True
    )

    prin_urls = _safe_urls(prin.reference_urls if prin else [])
    evaluator.add_custom_node(
        result=len(prin_urls) > 0,
        id="prin_reference_urls",
        desc="Provides at least one reference URL supporting the Principal minimum requirements used.",
        parent=node,
        critical=True
    )

    # Master's in educational leadership/administration
    leaf = evaluator.add_leaf(
        id="prin_requirement_masters_field",
        desc="Correctly evaluates whether the person meets the stated requirement: master's degree in educational leadership or educational administration (and notes if not met).",
        parent=node,
        critical=True
    )
    claim = _claim_for_requirement(
        "public K-12 Principal",
        "Master's in educational leadership/administration (or explicitly required field stated by sources)",
        prin.masters_field if prin else None
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=prin_urls,
        additional_instruction=(
            "From the source URL(s), determine the minimum education field required for public K-12 principals (often educational leadership/administration). "
            "Assess whether a 'Master's in Athletic Administration' satisfies that field when sources specify a particular specialty. Judge the answer's evaluation accordingly."
        )
    )

    # Valid K-12 teaching certification
    leaf = evaluator.add_leaf(
        id="prin_requirement_k12_teaching_cert",
        desc="Correctly evaluates whether the person meets the stated requirement: valid K-12 teaching certification (and notes if not met).",
        parent=node,
        critical=True
    )
    claim = _claim_for_requirement(
        "public K-12 Principal",
        "Valid K-12 teaching certification/license",
        prin.k12_teaching_cert if prin else None
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=prin_urls,
        additional_instruction=(
            "Use the URLs to confirm whether a state K-12 teaching license is typically required for public K-12 principals. "
            "Given the person lacks such certification, judge whether the answer's evaluation is correct."
        )
    )

    # 3–5 years K-12 teaching experience
    leaf = evaluator.add_leaf(
        id="prin_requirement_k12_teaching_experience",
        desc="Correctly evaluates whether the person meets the stated requirement: minimum 3–5 years of K-12 teaching experience (and notes if not met).",
        parent=node,
        critical=True
    )
    claim = _claim_for_requirement(
        "public K-12 Principal",
        "Minimum 3–5 years of K-12 teaching experience",
        prin.k12_teaching_experience if prin else None
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=prin_urls,
        additional_instruction=(
            "From the URLs, identify if principals typically must have several years (e.g., 3–5+) of K-12 classroom experience. "
            "Given the person has no K-12 teaching experience, judge whether the answer's evaluation is correct."
        )
    )

    # Administrator/Principal certification
    leaf = evaluator.add_leaf(
        id="prin_requirement_admin_cert",
        desc="Correctly evaluates whether the person meets the stated requirement: principal/administrator certification (and notes if not met).",
        parent=node,
        critical=True
    )
    claim = _claim_for_requirement(
        "public K-12 Principal",
        "Administrator/Principal certification/license",
        prin.admin_cert if prin else None
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=prin_urls,
        additional_instruction=(
            "Use the URLs to determine if state principal/administrator certification is a minimum requirement. "
            "Given the person lacks administrator certification, judge whether the answer's evaluation is correct."
        )
    )


async def verify_athletic_director(evaluator: Evaluator, parent_node, ad: Optional[AthleticDirectorInfo]) -> None:
    node = evaluator.add_parallel(
        id="position_3_athletic_director",
        desc="K-12 Athletic Director (public school district): decide YES/NO against stated typical requirements; note teaching-cert variability; cite sources.",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(ad and ad.decision and ad.decision.strip().upper() in {"YES", "NO"}),
        id="ad_decision_yes_no",
        desc="States an explicit qualification decision (YES or NO) for Athletic Director, using the stated 'typical' minimum requirements as the standard; any variability is explained separately.",
        parent=node,
        critical=True
    )

    ad_urls = _safe_urls(ad.reference_urls if ad else [])
    evaluator.add_custom_node(
        result=len(ad_urls) > 0,
        id="ad_reference_urls",
        desc="Provides at least one reference URL supporting the Athletic Director typical requirements used.",
        parent=node,
        critical=True
    )

    # Bachelor's as minimum education
    leaf = evaluator.add_leaf(
        id="ad_requirement_bachelors_min",
        desc="Correctly evaluates whether the person meets the typical minimum education requirement: at least a bachelor's degree (treating a master's as preferred, not required, unless sources say otherwise).",
        parent=node,
        critical=True
    )
    claim = _claim_for_requirement(
        "public school district Athletic Director",
        "At least a bachelor's degree as minimum education",
        ad.bachelors_min if ad else None
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ad_urls,
        additional_instruction=(
            "From the source URL(s), determine the typical minimum education for K-12 district Athletic Directors. "
            "If sources indicate a bachelor's minimum (with master's often preferred), judge the answer's evaluation using the person's bachelor's in Physical Education (2001) and master's in Athletic Administration (2004)."
        )
    )

    # Relevant coaching/athletics admin experience
    leaf = evaluator.add_leaf(
        id="ad_requirement_relevant_experience",
        desc="Correctly evaluates whether the person meets the typical experience requirement: relevant coaching or athletic administrative experience.",
        parent=node,
        critical=True
    )
    claim = _claim_for_requirement(
        "public school district Athletic Director",
        "Relevant athletics/administrative/coaching experience",
        ad.relevant_experience if ad else None
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ad_urls,
        additional_instruction=(
            "From the URLs, identify the typical experience requirements (e.g., athletics administration or coaching experience). "
            "Given the person has 12 years of college coaching including 5 as OC, judge whether the answer's evaluation is correct."
        )
    )

    # Variability note (textual check within the answer)
    leaf = evaluator.add_leaf(
        id="ad_teaching_cert_variability_noted",
        desc="Explicitly notes that teaching certification requirements vary by district and explains how the person's lack of teaching certification could affect eligibility in some districts.",
        parent=node,
        critical=True
    )
    claim = (
        "The answer explicitly states that K-12 district Athletic Director teaching-certification requirements vary by district "
        "and explains that the person's lack of a teaching certificate could disqualify them in some districts."
    )
    # This is about the answer's content; verify without URLs
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=None,
        additional_instruction=(
            "Judge strictly from the answer text (provided as context). "
            "Pass only if the answer clearly notes variability across districts and connects it to the person's lack of K-12 teaching certification potentially affecting eligibility."
        )
    )


async def verify_assistant_coach(evaluator: Evaluator, parent_node, ac: Optional[AssistantCoachInfo]) -> None:
    node = evaluator.add_parallel(
        id="position_4_assistant_coach",
        desc="Division I FCS Assistant Football Coach: decide YES/NO against stated typical requirements and cite sources.",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(ac and ac.decision and ac.decision.strip().upper() in {"YES", "NO"}),
        id="ac_decision_yes_no",
        desc="States an explicit qualification decision (YES or NO) for Division I FCS Assistant Football Coach.",
        parent=node,
        critical=True
    )

    ac_urls = _safe_urls(ac.reference_urls if ac else [])
    evaluator.add_custom_node(
        result=len(ac_urls) > 0,
        id="ac_reference_urls",
        desc="Provides at least one reference URL supporting the Assistant Coach typical requirements used.",
        parent=node,
        critical=True
    )

    # Bachelor's degree
    leaf = evaluator.add_leaf(
        id="ac_requirement_bachelors",
        desc="Correctly evaluates whether the person meets the typical education requirement: bachelor's degree.",
        parent=node,
        critical=True
    )
    claim = _claim_for_requirement(
        "Division I FCS Assistant Football Coach",
        "Bachelor's degree",
        ac.bachelors if ac else None
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ac_urls,
        additional_instruction=(
            "From the source URL(s), determine if a bachelor's degree is a typical minimum for NCAA Division I (FCS) assistant coaches. "
            "Given the person has a bachelor's, judge whether the answer's evaluation is correct."
        )
    )

    # Coaching or playing experience
    leaf = evaluator.add_leaf(
        id="ac_requirement_coaching_or_playing_experience",
        desc="Correctly evaluates whether the person meets the typical experience requirement: ≥1–2 years collegiate/pro coaching experience OR significant college playing experience (and notes if not met).",
        parent=node,
        critical=True
    )
    claim = _claim_for_requirement(
        "Division I FCS Assistant Football Coach",
        "At least 1–2 years collegiate/pro coaching experience OR significant college playing experience",
        ac.coaching_or_playing_experience if ac else None
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ac_urls,
        additional_instruction=(
            "Use the URLs to confirm the typical experience expectation for FCS assistant coaches. "
            "Given the person has 12 years of college coaching with 5 as OC, judge whether the answer's evaluation is correct."
        )
    )

    # Certification/licensure addressed per sources
    leaf = evaluator.add_leaf(
        id="ac_certification_addressed",
        desc="Addresses certification/licensure requirements for this role per cited sources (explicitly stating none if none are required) and evaluates the person accordingly.",
        parent=node,
        critical=True
    )
    claim = (
        "For Division I FCS Assistant Football Coach, the answer addresses certification/licensure requirements and its statement is consistent with the provided source page(s) "
        "(e.g., no state K-12 teaching license is required for university coaching roles; institutional HR background checks/compliance may exist but are not teaching licenses)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ac_urls,
        additional_instruction=(
            "Check the URLs for any certification/licensure requirements for collegiate assistant coaches. "
            "If none are required (beyond general HR/compliance), accept statements that explicitly note 'none required'. "
            "Judge whether the answer correctly addressed and evaluated this requirement."
        )
    )


async def verify_head_coach(evaluator: Evaluator, parent_node, hc: Optional[HeadCoachInfo]) -> None:
    node = evaluator.add_parallel(
        id="position_5_head_coach",
        desc="Division I FCS Head Football Coach: decide YES/NO against stated typical requirements and cite sources.",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(hc and hc.decision and hc.decision.strip().upper() in {"YES", "NO"}),
        id="hc_decision_yes_no",
        desc="States an explicit qualification decision (YES or NO) for Division I FCS Head Football Coach.",
        parent=node,
        critical=True
    )

    hc_urls = _safe_urls(hc.reference_urls if hc else [])
    evaluator.add_custom_node(
        result=len(hc_urls) > 0,
        id="hc_reference_urls",
        desc="Provides at least one reference URL supporting the Head Coach typical requirements used.",
        parent=node,
        critical=True
    )

    # Bachelor's degree
    leaf = evaluator.add_leaf(
        id="hc_requirement_bachelors",
        desc="Correctly evaluates whether the person meets the typical education requirement: bachelor's degree (treating a master's as common/preferred unless stated as a minimum).",
        parent=node,
        critical=True
    )
    claim = _claim_for_requirement(
        "Division I FCS Head Football Coach",
        "Bachelor's degree as minimum education",
        hc.bachelors if hc else None
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=hc_urls,
        additional_instruction=(
            "From the URLs, determine if a bachelor's degree is typically required for head coaches; treat master's as preferred unless the source states it's a minimum. "
            "Given the person has a bachelor's (and a master's), judge whether the answer's evaluation is correct."
        )
    )

    # Substantial experience including coordinator or head coaching
    leaf = evaluator.add_leaf(
        id="hc_requirement_experience_including_coordinator_or_head",
        desc="Correctly evaluates whether the person meets the stated typical experience requirement: substantial coaching experience including coordinator experience or prior head coaching experience (without inventing extra thresholds beyond constraints/sources).",
        parent=node,
        critical=True
    )
    claim = _claim_for_requirement(
        "Division I FCS Head Football Coach",
        "Substantial coaching experience including coordinator or prior head coaching experience",
        hc.experience_including_coordinator_or_head if hc else None
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=hc_urls,
        additional_instruction=(
            "Use the URLs to identify the typical experience expectation for head coaches (e.g., extensive coaching background, coordinator or prior head coaching). "
            "Given the person has 12 years of college coaching with 5 years as offensive coordinator, judge whether the answer's evaluation is correct."
        )
    )

    # Certification/licensure addressed per sources
    leaf = evaluator.add_leaf(
        id="hc_certification_addressed",
        desc="Addresses certification/licensure requirements for this role per cited sources (explicitly stating none if none are required) and evaluates the person accordingly.",
        parent=node,
        critical=True
    )
    claim = (
        "For Division I FCS Head Football Coach, the answer addresses certification/licensure requirements and its statement is consistent with the provided source page(s) "
        "(e.g., no state K-12 teaching license is required for university coaching roles; institutional HR background checks/compliance may exist but are not teaching licenses)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=hc_urls,
        additional_instruction=(
            "Check the URLs for any certification/licensure requirement specific to collegiate head coaches. "
            "If none are required (beyond HR/compliance), accept statements that note 'none required'. "
            "Judge whether the answer correctly addressed and evaluated this requirement."
        )
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
    # Initialize evaluator (root is non-critical parallel by default per framework)
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

    # Record the person's credentials as GT context info (not used for scoring directly)
    evaluator.add_ground_truth({
        "person_credentials": {
            "bachelor": "Physical Education (2001)",
            "master": "Athletic Administration (2004)",
            "experience": "12 years DI FCS coaching, 5 years as Offensive Coordinator",
            "k12_teaching_cert": "None",
            "principal_admin_cert": "None"
        }
    })

    # Extract all positions data in one pass
    extracted = await evaluator.extract(
        prompt=prompt_extract_all_positions(),
        template_class=CareerQualificationsExtraction,
        extraction_name="positions_extraction"
    )

    # Build verification subtrees for each position
    await verify_assistant_principal(evaluator, root, extracted.assistant_principal)
    await verify_principal(evaluator, root, extracted.principal)
    await verify_athletic_director(evaluator, root, extracted.athletic_director)
    await verify_assistant_coach(evaluator, root, extracted.assistant_coach)
    await verify_head_coach(evaluator, root, extracted.head_coach)

    # Return full summary
    return evaluator.get_summary()