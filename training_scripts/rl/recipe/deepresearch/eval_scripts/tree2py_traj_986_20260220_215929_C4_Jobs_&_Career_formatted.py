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
TASK_ID = "fl_athletic_director_requirements"
TASK_DESCRIPTION = """
What are the complete certification, educational, and training requirements for becoming a high school athletic director in Florida? Provide a comprehensive list that includes: (1) all Florida coaching certification requirements as specified by the Florida Department of Education, including required coursework and certifications, and (2) the typical educational and experience requirements for high school athletic director positions. For each requirement, include the specific details (such as semester hour amounts for courses) and provide reference URLs from official or authoritative sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FloridaCoaching(BaseModel):
    # Prerequisite: other subject certification
    prerequisite_statement: Optional[str] = None
    prerequisite_urls: List[str] = Field(default_factory=list)

    # Coursework (Florida Rule 6A-4.0282)
    coursework_total_sh: Optional[str] = None
    injury_drug_sh: Optional[str] = None
    coaching_theory_sh: Optional[str] = None
    sport_specific_course: Optional[str] = None
    coursework_urls: List[str] = Field(default_factory=list)

    # CPR requirement
    cpr_requirement: Optional[str] = None
    cpr_urls: List[str] = Field(default_factory=list)


class TypicalADQuals(BaseModel):
    # Education
    bachelors_required: Optional[str] = None
    masters_preferred: Optional[str] = None
    education_urls: List[str] = Field(default_factory=list)

    # Experience
    head_coach_pref: Optional[str] = None
    experience_urls: List[str] = Field(default_factory=list)


class RequirementsExtraction(BaseModel):
    fl_coaching: Optional[FloridaCoaching] = FloridaCoaching()
    typical_quals: Optional[TypicalADQuals] = TypicalADQuals()


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract from the answer all statements and cited URLs related to:
    (A) Florida Athletic Coaching Endorsement / coaching certification requirements as specified by Florida authorities;
    (B) Typical high school athletic director (AD) education and experience qualifications.

    Return a JSON object matching this schema:
    {
      "fl_coaching": {
        "prerequisite_statement": string or null,   // Statement that certification in another subject is a prerequisite for the Florida Athletic Coaching Endorsement (add-on/endorsement)
        "prerequisite_urls": [urls...],             // The specific URLs cited in the answer for that prerequisite
        "coursework_total_sh": string or null,      // Total semester hours stated (e.g., "9 semester hours")
        "injury_drug_sh": string or null,           // Semester hours for injury/drug dangers coursework (e.g., "3 semester hours")
        "coaching_theory_sh": string or null,       // Semester hours for coaching theory coursework (e.g., "3 semester hours")
        "sport_specific_course": string or null,    // Mention of theory/practice of coaching a specific sport (no SH required, but presence must be noted)
        "coursework_urls": [urls...],               // URLs cited specifically for the coursework requirements
        "cpr_requirement": string or null,          // Statement describing CPR certification requirement (AHA/ARC or FL Department of Health equivalent)
        "cpr_urls": [urls...]                       // URLs cited specifically for the CPR requirement
      },
      "typical_quals": {
        "bachelors_required": string or null,       // Statement that a bachelor's degree is typically required
        "masters_preferred": string or null,        // Statement that a master's (athletic admin/educational leadership) is often preferred
        "education_urls": [urls...],                // URLs cited for education expectations
        "head_coach_pref": string or null,          // Statement that head coaching experience is especially preferred
        "experience_urls": [urls...]                // URLs cited for experience expectations
      }
    }

    Rules:
    - Extract only information explicitly present in the answer text.
    - For each URLs array, include exactly the URLs the answer provided for that item (plain URLs or URLs in markdown links).
    - Do not invent URLs. If the answer did not provide a URL for an item, return an empty array for that item's URLs.
    - If a statement is not present in the answer, return null for the corresponding field.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def has_any_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0 and any(isinstance(u, str) and u.strip() for u in urls)


def nonempty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls or []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_fl_coaching_endorsement_checks(evaluator: Evaluator, parent):
    """
    Build the subtree for Florida Coaching Endorsement/coaching requirements.
    Child nodes are all critical since this block is critical.
    """
    node = evaluator.add_parallel(
        id="fl_coaching_endorsement",
        desc="Florida Athletic Coaching Endorsement/coaching requirements are fully documented with authoritative citations",
        parent=parent,
        critical=True
    )
    return node


async def verify_fl_coaching_endorsement(evaluator: Evaluator, parent, extracted: FloridaCoaching):
    """
    Create sequential gates for each requirement:
    1) Prerequisite other-subject certification
    2) Coursework 9 semester hours with specified components
    3) CPR certification requirement
    """
    # --- Prerequisite other-subject certification ---
    prereq_seq = evaluator.add_sequential(
        id="fl_prerequisite_seq",
        desc="Prerequisite: certification in another subject is required (add-on endorsement)",
        parent=parent,
        critical=True
    )
    prereq_exists = evaluator.add_custom_node(
        result=(extracted is not None and nonempty(getattr(extracted, "prerequisite_statement", None)) and has_any_urls(getattr(extracted, "prerequisite_urls", None))),
        id="fl_prerequisite_exists",
        desc="Answer includes a prerequisite statement and cites at least one authoritative URL",
        parent=prereq_seq,
        critical=True
    )
    prereq_leaf = evaluator.add_leaf(
        id="fl_prerequisite_supported",
        desc="Florida Athletic Coaching Endorsement is an add-on that requires certification in another subject",
        parent=prereq_seq,
        AtlPlaceholderKeyIfNeeded=True,
        critical=True
    )
    prereq_claim = (
        "Under Florida's Athletic Coaching Endorsement (add-on), certification in another subject (a valid Florida Educator's Certificate Atlantic) "
        "is a prerequisite; the endorsement is attached to another subject certificate."
    )
    await AventGuard_named_api_noop_iframe_harmony_call_if_needed  # This is a placeholder

    await evaluator.verify(
        claim= confession  # Another placeholder pipeline placeholder
    )

    # The above placeholders were incorrect additions; rewrite the block cleanly below.


# --------------------------------------------------------------------------- #
# Rewriting verification builders cleanly (no placeholders)                   #
# --------------------------------------------------------------------------- #
async def verify_fl_coaching_endorsement(evaluator: Evaluator, parent, extracted: FloridaCoaching):
    """
    Create sequential gates for each requirement:
    1) Prerequisite other-subject certification
    2) Coursework 9 semester hours with specified components
    3) CPR certification requirement
    """
    # --- Prerequisite other-subject certification ---
    prereq_seq = evaluator.add_sequential(
        id="fl_prerequisite_seq",
        desc="Prerequisite: certification in another subject is required (add-on endorsement)",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=(extracted is not None and nonempty(extracted.prerequisite_statement) and has_any_urls(extracted.prerequisite_urls)),
        id="fl_prerequisite_exists",
        desc="Answer includes a prerequisite statement and cites at least one authoritative URL",
        parent=prereq_seq,
        critical=True
    )
    prereq_leaf = evaluator.add_leaf(
        id="fl_prerequisite_supported",
        desc="Florida Athletic Coaching Endorsement is an add-on that requires certification in another subject",
        parent=prereq_seq,
        critical=True
    )
    prereq_claim = (
        "Florida's Athletic Coaching Endorsement is an add-on endorsement that requires the educator to hold certification in another subject area "
        "(i.e., it is attached to a valid Florida Educator's Certificate in any subject)."
    )
    await evaluator.verify(
        claim=prereq_claim,
        node=prereq_leaf,
        sources=safe_urls(extracted.prerequisite_urls),
        additional_instruction="Treat official Florida sources (Florida Administrative Code 6A-4.0282, Florida Department of Education, or other Florida government sources) as authoritative. The page must explicitly state that the endorsement is added to another valid certificate or that certification in another subject is required."
    )

    # --- Coursework: 9 semester hours + components ---
    coursework_seq = evaluator.add_sequential(
        id="fl_coursework_seq",
        desc="Coursework requirement: 9 semester hours with specified components",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=(
            extracted is not None
            and nonempty(extracted.coursework_total_sh)
            and nonempty(extracted.injury_drug_sh)
            and nonempty(extracted.coaching_theory_sh)
            and nonempty(extracted.sport_specific_course)
            and has_any_urls(extracted.coursework_urls)
        ),
        id="fl_coursework_exists",
        desc="Answer includes coursework total SH and specified components (with SH where required) with at least one authoritative URL",
        parent=coursework_seq,
        critical=True
    )
    coursework_leaf = evaluator.add_leaf(
        id="fl_coursework_supported",
        desc="Rule 6A-4.0282 requires 9 SH: 3 SH injury/drug dangers; 3 SH coaching theory; and theory/practice of a specific sport",
        parent=coursework_seq,
        critical=True
    )
    coursework_claim = (
        "Florida Administrative Code Rule 6A-4.0282 requires a total of 9 semester hours of athletic-coaching coursework, comprising: "
        "3 semester hours on the prevention of athletic injuries and the dangers of drugs in sports; "
        "3 semester hours in coaching theory; and "
        "the theory and practice of coaching a specific sport."
    )
    await evaluator.verify(
        claim=coursework_claim,
        node=coursework_leaf,
        sources=safe_urls(extracted.coursework_urls),
        additional_instruction="Verify directly against the Florida Administrative Code or Florida Department of Education materials. The page should clearly show the 9 semester-hour total and the component breakdown (3 SH injury/drugs, 3 SH coaching theory, plus theory/practice of a specific sport)."
    )

    # --- CPR requirement ---
    cpr_seq = evaluator.add_sequential(
        id="fl_cpr_seq",
        desc="CPR certification requirement (AHA/ARC or FL DoH equivalent)",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=(extracted is not None and nonempty(extracted.cpr_requirement) and has_any_urls(extracted.cpr_urls)),
        id="fl_cpr_exists",
        desc="Answer includes a CPR requirement statement and cites at least one authoritative URL",
        parent=cpr_seq,
        critical=True
    )
    cpr_leaf = evaluator.add_leaf(
        id="fl_cpr_supported",
        desc="Valid AHA/ARC CPR certification or FL Department of Health–approved equivalent is required",
        parent=cpr_seq,
        critical=True
    )
    cpr_claim = (
        "For the Florida Athletic Coaching Endorsement, a valid CPR certification by the American Heart Association (AHA) or American Red Cross (ARC), "
        "or an equivalent certification approved by the Florida Department of Health, is required."
    )
    await evaluator.verify(
        claim=cpr_claim,
        node=cpr_leaf,
        sources=safe_urls(extracted.cpr_urls),
        additional_instruction="Prefer Florida Administrative Code 6A-4.0282, Florida Department of Education, or Florida Department of Health pages. The page must state AHA/ARC CPR or equivalent approved by FL DoH (or an equivalently strict phrasing)."
    )


# --------------------------------------------------------------------------- #
# Typical AD qualifications verification                                      #
# --------------------------------------------------------------------------- #
async def build_typical_qualifications_checks(evaluator: Evaluator, parent):
    """
    Build the subtree for typical AD education/experience requirements.
    Child nodes are all critical since this block is critical per rubric.
    """
    node = evaluator.add_parallel(
        id="typical_ad_qualifications",
        desc="Typical high school AD education and experience qualifications documented with authoritative citations",
        parent=parent,
        critical=True
    )
    return node


async def verify_typical_qualifications(evaluator: Evaluator, parent, extracted: TypicalADQuals):
    """
    Verify two aspects with gating:
    - Education: bachelor's typically required; master's often preferred
    - Experience: head coaching experience especially preferred
    """
    # --- Education checks: gate existence, then parallel verify sub-facts ---
    edu_seq = evaluator.add_sequential(
        id="typical_edu_seq",
        desc="Typical education requirements (bachelor's required; master's preferred)",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=(
            extracted is not None
            and nonempty(extracted.bachelors_required)
            and nonempty(extracted.masters_preferred)
            and has_any_urls(extracted.education_urls)
        ),
        id="typical_edu_exists",
        desc="Answer includes bachelor's-required and master's-preferred statements with at least one authoritative URL",
        parent=edu_seq,
        critical=True
    )
    edu_par = evaluator.add_parallel(
        id="typical_edu_parallel",
        desc="Parallel verification of education subrequirements",
        parent=edu_seq,
        critical=True
    )
    # Bachelor's typically required
    edu_bach_leaf = evaluator.add_leaf(
        id="typical_bachelors_required_supported",
        desc="Bachelor's degree is typically required for high school athletic directors",
        parent=edu_par,
        critical=True
    )
    bach_claim = "For high school athletic director positions, a bachelor's degree is typically required."
    await evaluator.verify(
        claim=bach_claim,
        node=edu_bach_leaf,
        sources=safe_urls(extracted.education_urls),
        additional_instruction="Accept authoritative sources such as Florida school district job postings, official HR documents, or reputable professional organizations. The page must explicitly indicate that a bachelor's degree is typically required."
    )
    # Master's often preferred
    edu_mast_leaf = evaluator.add_leaf(
        id="typical_masters_preferred_supported",
        desc="A master's degree (e.g., athletic administration or educational leadership) is often preferred",
        parent=edu_par,
        critical=True
    )
    mast_claim = (
        "For high school athletic director positions, a master's degree—such as in athletic administration or educational leadership—is often preferred."
    )
    await evaluator.verify(
        claim=mast_claim,
        node=edu_mast_leaf,
        sources=safe_urls(extracted.education_urls),
        additional_instruction="Accept wording like 'preferred', 'strongly preferred', or 'desired'. Sources should be authoritative (e.g., Florida district postings, state association guidance, or reputable professional/educational organizations)."
    )

    # --- Experience checks: gate existence then verify ---
    exp_seq = evaluator.add_sequential(
        id="typical_exp_seq",
        desc="Typical experience requirements (head coaching experience preferred)",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=(extracted is not None and nonempty(extracted.head_coach_pref) and has_any_urls(extracted.experience_urls)),
        id="typical_exp_exists",
        desc="Answer includes head-coaching-experience preferred statement with at least one authoritative URL",
        parent=exp_seq,
        critical=True
    )
    exp_leaf = evaluator.add_leaf(
        id="typical_head_coach_preferred_supported",
        desc="Head coaching experience is especially preferred for AD positions",
        parent=exp_seq,
        critical=True
    )
    exp_claim = "Head coaching experience is especially preferred for high school athletic director positions."
    await evaluator.verify(
        claim=exp_claim,
        node=exp_leaf,
        sources=safe_urls(extracted.experience_urls),
        additional_instruction="Accept phrasing like 'head coaching experience preferred', 'strongly preferred', or 'highly desired'. Sources should be authoritative (e.g., Florida district job postings, official HR documents, or well-recognized professional/education sources)."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the answer for Florida high school athletic director requirements.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator; top-level children are critical to simulate root criticality
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

    # Extract structured information
    extracted: RequirementsExtraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementsExtraction,
        extraction_name="requirements_extraction"
    )

    # Build top-level critical branches
    fl_node = await build_fl_coaching_endorsement_checks(evaluator, root)
    typical_node = await build_typical_qualifications_checks(evaluator, root)

    # Verify Florida Coaching Endorsement/coaching requirements
    await verify_fl_coaching_endorsement(
        evaluator,
        fl_node,
        extracted.fl_coaching or FloridaCoaching()
    )

    # Verify typical AD qualifications
    await verify_typical_qualifications(
        evaluator,
        typical_node,
        extracted.typical_quals or TypicalADQuals()
    )

    return evaluator.get_summary()