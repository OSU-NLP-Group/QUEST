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
TASK_ID = "cs_ms_prereq_gpa_univ_requirements"
TASK_DESCRIPTION = (
    "Identify and document the prerequisite course requirements and minimum GPA requirements for master's degree "
    "programs in Computer Science at the following three universities: University of Tennessee Knoxville (UTK), "
    "University of Houston (UH), and University of Texas at Austin (UT Austin). For each program, provide: "
    "(1) the required mathematics prerequisite courses, specifying the number of semesters or specific course names required; "
    "(2) the required computer science prerequisite courses or proficiency requirements, listing specific courses or topics; "
    "(3) the minimum GPA requirements, including the specific threshold values and the scope to which they apply (such as overall GPA, "
    "upper-division GPA, or senior year GPA); and (4) the official university webpage URL where these requirements are documented."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UTKRequirements(BaseModel):
    source_urls: List[str] = Field(default_factory=list)
    math_prereqs: Optional[str] = None
    cs_prereqs: Optional[str] = None
    gpa_overall: Optional[str] = None
    gpa_senior_year: Optional[str] = None


class UHRequirements(BaseModel):
    source_urls: List[str] = Field(default_factory=list)
    math_prereqs: Optional[str] = None
    cs_proficiency: Optional[str] = None
    gpa_info: Optional[str] = None


class UTARequirements(BaseModel):
    source_urls: List[str] = Field(default_factory=list)
    prereq_disclosure: Optional[str] = None
    gpa_upper_division: Optional[str] = None
    gpa_graduate_work: Optional[str] = None


class ProgramsExtraction(BaseModel):
    utk: Optional[UTKRequirements] = None
    uh: Optional[UHRequirements] = None
    ut_austin: Optional[UTARequirements] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program_requirements() -> str:
    return """
    Extract, from the provided answer, the CS MS program requirements for each university (UTK, UH, and UT Austin).
    For each university, collect the following fields exactly as stated in the answer:

    For UTK (University of Tennessee Knoxville):
      - source_urls: All official UTK webpages (full URLs) cited in the answer that document the requirements.
      - math_prereqs: The math prerequisites as stated in the answer (e.g., "two semesters of calculus plus two additional semesters of college mathematics").
      - cs_prereqs: The CS prerequisites as stated in the answer (e.g., "a formal languages course and a systems programming course").
      - gpa_overall: The minimum overall GPA requirement with scope if the answer states it (e.g., "Minimum 3.0 overall GPA").
      - gpa_senior_year: The minimum senior year GPA requirement with scope if the answer states it (e.g., "Minimum 3.0 GPA in senior year").
    
    For UH (University of Houston):
      - source_urls: All official UH webpages (full URLs) cited in the answer that document the requirements.
      - math_prereqs: The math prerequisites to be completed before admission as stated in the answer (e.g., "Calculus I, Calculus II, and Linear Algebra").
      - cs_proficiency: The CS proficiency requirements as stated, including course numbers/topics if present (e.g., "COSC 6305, COSC 6306, COSC 6308, COSC 6309, COSC 6310" or their topics/equivalents).
      - gpa_info: The GPA requirement information as stated in the answer, including the applicable scope; if the answer states that no explicit minimum threshold/scope is specified (holistic), capture that wording.

    For UT Austin (University of Texas at Austin):
      - source_urls: All official UT Austin webpages (full URLs) cited in the answer that document the requirements.
      - prereq_disclosure: The statement in the answer about prerequisites not being explicitly listed on the main CS MS admissions page (if stated).
      - gpa_upper_division: The minimum GPA requirement for upper-division coursework (scope: upper-division) if stated (e.g., "Minimum 3.0 in upper-division courses").
      - gpa_graduate_work: The minimum GPA requirement for any completed graduate work (scope: graduate work) if stated (e.g., "Minimum 3.0 in any completed graduate work").

    RULES:
    - Extract exactly what is written in the answer. Do not invent or infer missing information.
    - If a field is not present in the answer, set it to null (and for source_urls return an empty list).
    - Only include official university webpages in source_urls (for UTK: *.utk.edu, UH: *.uh.edu, UT Austin: *.utexas.edu or *.cs.utexas.edu). If the answer cites non-official sources, do not include them here.
    - URLs can be plain or markdown links; return the full URL strings.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_join(items: Optional[List[str]]) -> List[str]:
    return items or []


# --------------------------------------------------------------------------- #
# Verification functions per university                                       #
# --------------------------------------------------------------------------- #
async def verify_utk_program(evaluator: Evaluator, parent_node, utk: Optional[UTKRequirements]) -> None:
    program_node = evaluator.add_parallel(
        id="UTK_Program",
        desc="UTK CS MS requirements as documented on an official UTK webpage.",
        parent=parent_node,
        critical=True
    )

    sources = _safe_join(utk.source_urls if utk else None)

    # UTK Source URL – official page check
    url_node = evaluator.add_leaf(
        id="UTK_Source_URL",
        desc="Provide an official UTK webpage URL that documents the UTK requirements stated.",
        parent=program_node,
        critical=True
    )
    await evaluator.verify(
        claim="This is an official UTK webpage that documents CS MS prerequisites and/or GPA requirements.",
        node=url_node,
        sources=sources,
        additional_instruction=(
            "Treat a page as official UTK only if its URL domain ends with 'utk.edu' "
            "(including subdomains like eecs.utk.edu, catalog.utk.edu). "
            "It must be relevant to CS MS admissions/prerequisites/GPA requirements."
        ),
    )

    # UTK Math Prerequisites
    math_node = evaluator.add_leaf(
        id="UTK_Math_Prerequisites",
        desc="State UTK math prerequisites: two semesters of calculus plus two additional semesters of college mathematics (e.g., linear algebra, differential equations, probability).",
        parent=program_node,
        critical=True
    )
    math_claim = (
        f"UTK CS MS math prerequisites as stated in the answer: '{(utk.math_prereqs or 'unspecified')}'. "
        "The official UTK page states that two semesters of calculus plus two additional semesters of college mathematics are required."
    )
    await evaluator.verify(
        claim=math_claim,
        node=math_node,
        sources=sources,
        additional_instruction=(
            "Confirm that UTK requires two semesters of calculus PLUS two additional semesters of college mathematics "
            "(examples: linear algebra, differential equations, probability). Mark supported only if the official page clearly states this."
        ),
    )

    # UTK CS Prerequisites
    cs_node = evaluator.add_leaf(
        id="UTK_CS_Prerequisites",
        desc="State UTK CS prerequisites: a formal languages course and a systems programming course.",
        parent=program_node,
        critical=True
    )
    cs_claim = (
        f"UTK CS MS CS prerequisites as stated in the answer: '{(utk.cs_prereqs or 'unspecified')}'. "
        "The official page lists a formal languages course and a systems programming course as prerequisites."
    )
    await evaluator.verify(
        claim=cs_claim,
        node=cs_node,
        sources=sources,
        additional_instruction=(
            "Verify the presence of both: a 'Formal Languages' course (often phrased as 'Formal Languages and Automata Theory') "
            "AND a 'Systems Programming' course among prerequisites. Accept reasonable synonyms if the page clarifies equivalence."
        ),
    )

    # UTK GPA Requirements group
    gpa_group = evaluator.add_parallel(
        id="UTK_GPA_Requirements",
        desc="State UTK minimum GPA requirements with scope.",
        parent=program_node,
        critical=True
    )

    # UTK Overall GPA
    overall_node = evaluator.add_leaf(
        id="UTK_Overall_GPA",
        desc="Minimum 3.0 overall GPA requirement (scope: overall).",
        parent=gpa_group,
        critical=True
    )
    overall_claim = (
        f"UTK requires at least a 3.0 overall GPA for CS MS admission. "
        f"Answer mentions: '{(utk.gpa_overall or 'unspecified')}'."
    )
    await evaluator.verify(
        claim=overall_claim,
        node=overall_node,
        sources=sources,
        additional_instruction=(
            "Confirm the official UTK page states a minimum 3.0 GPA requirement with scope 'overall'. "
            "Accept phrasing variations like 'at least 3.0 GPA overall'."
        ),
    )

    # UTK Senior Year GPA
    senior_node = evaluator.add_leaf(
        id="UTK_Senior_Year_GPA",
        desc="Minimum 3.0 GPA in senior year requirement (scope: senior year).",
        parent=gpa_group,
        critical=True
    )
    senior_claim = (
        f"UTK requires at least a 3.0 GPA in the senior year. "
        f"Answer mentions: '{(utk.gpa_senior_year or 'unspecified')}'."
    )
    await evaluator.verify(
        claim=senior_claim,
        node=senior_node,
        sources=sources,
        additional_instruction=(
            "Confirm the official UTK page explicitly states a minimum 3.0 GPA requirement scoped to 'senior year'."
        ),
    )


async def verify_uh_program(evaluator: Evaluator, parent_node, uh: Optional[UHRequirements]) -> None:
    program_node = evaluator.add_parallel(
        id="UH_Program",
        desc="UH CS MS requirements as documented on an official UH webpage.",
        parent=parent_node,
        critical=True
    )

    sources = _safe_join(uh.source_urls if uh else None)

    # UH Source URL – official page check
    url_node = evaluator.add_leaf(
        id="UH_Source_URL",
        desc="Provide an official UH webpage URL that documents the UH requirements stated.",
        parent=program_node,
        critical=True
    )
    await evaluator.verify(
        claim="This is an official University of Houston webpage that documents CS MS prerequisites/proficiency and/or GPA requirements.",
        node=url_node,
        sources=sources,
        additional_instruction=(
            "Treat a page as official UH only if its URL domain ends with 'uh.edu' "
            "(including subdomains like uh.edu/nsm/computer-science/). "
            "The page must be relevant to CS MS admissions/prerequisites/proficiency/GPA."
        ),
    )

    # UH Math Prerequisites
    math_node = evaluator.add_leaf(
        id="UH_Math_Prerequisites",
        desc="State UH math prerequisites to be completed before admission: Calculus I, Calculus II, and Linear Algebra.",
        parent=program_node,
        critical=True
    )
    math_claim = (
        f"UH CS MS math prerequisites as stated in the answer: '{(uh.math_prereqs or 'unspecified')}'. "
        "The official page requires Calculus I, Calculus II, and Linear Algebra to be completed before admission."
    )
    await evaluator.verify(
        claim=math_claim,
        node=math_node,
        sources=sources,
        additional_instruction=(
            "Confirm the UH page lists Calculus I, Calculus II, and Linear Algebra as prerequisites to be completed before admission."
        ),
    )

    # UH CS Proficiency Requirements
    cs_prof_node = evaluator.add_leaf(
        id="UH_CS_Proficiency_Requirements",
        desc="State UH CS proficiency requirements: COSC 6305, COSC 6306, COSC 6308, COSC 6309, COSC 6310 (or their stated topics/equivalents as documented).",
        parent=program_node,
        critical=True
    )
    cs_prof_claim = (
        f"UH CS MS proficiency requirements as stated in the answer: '{(uh.cs_proficiency or 'unspecified')}'. "
        "The official page documents proficiency in five core graduate courses (COSC 6305, 6306, 6308, 6309, 6310) or their topic equivalents."
    )
    await evaluator.verify(
        claim=cs_prof_claim,
        node=cs_prof_node,
        sources=sources,
        additional_instruction=(
            "Verify the UH page lists proficiency requirements in COSC 6305, 6306, 6308, 6309, 6310, or explicitly lists their topics/equivalents. "
            "It is acceptable if the page allows satisfying proficiency via prior coursework, exams, or leveling courses, provided these specific cores are documented."
        ),
    )

    # UH GPA Requirements (holistic or explicit)
    gpa_node = evaluator.add_leaf(
        id="UH_GPA_Requirements",
        desc="State UH GPA requirement information as documented, including the applicable scope; if no minimum threshold/scope is specified, explicitly disclose that (e.g., GPA considered holistically/no explicit minimum stated).",
        parent=program_node,
        critical=True
    )
    gpa_claim = (
        f"UH CS MS GPA requirement information as stated in the answer: '{(uh.gpa_info or 'unspecified')}'. "
        "The official page either specifies a minimum threshold and scope or explicitly states no minimum threshold (holistic)."
    )
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_node,
        sources=sources,
        additional_instruction=(
            "Check the UH page for GPA requirement details. If the page specifies a minimum GPA threshold and scope (e.g., overall, upper-division), verify that. "
            "If the page states no explicit minimum and describes a holistic review, confirm that wording. "
            "Mark supported only if the claim aligns with the official page."
        ),
    )


async def verify_ut_austin_program(evaluator: Evaluator, parent_node, uta: Optional[UTARequirements]) -> None:
    program_node = evaluator.add_parallel(
        id="UT_Austin_Program",
        desc="UT Austin CS MS requirements as documented on an official UT Austin webpage.",
        parent=parent_node,
        critical=True
    )

    sources = _safe_join(uta.source_urls if uta else None)

    # UT Austin Source URL – official page check
    url_node = evaluator.add_leaf(
        id="UT_Austin_Source_URL",
        desc="Provide an official UT Austin webpage URL that documents the UT Austin requirements stated.",
        parent=program_node,
        critical=True
    )
    await evaluator.verify(
        claim="This is an official UT Austin webpage that documents CS MS admissions/prerequisites and/or GPA requirements.",
        node=url_node,
        sources=sources,
        additional_instruction=(
            "Treat a page as official UT Austin only if its URL domain ends with 'utexas.edu' or 'cs.utexas.edu'. "
            "The page must be relevant to CS MS admissions/prerequisites/GPA requirements."
        ),
    )

    # UT Austin Prereq Disclosure
    prereq_node = evaluator.add_leaf(
        id="UT_Austin_Prereq_Disclosure",
        desc="Explicitly state that specific prerequisite courses are not explicitly listed on the main UT Austin CS MS admissions page (per official documentation used).",
        parent=program_node,
        critical=True
    )
    prereq_claim = (
        f"The main UT Austin CS MS admissions page does not explicitly list specific prerequisite course names. "
        f"Answer states: '{(uta.prereq_disclosure or 'unspecified')}'."
    )
    await evaluator.verify(
        claim=prereq_claim,
        node=prereq_node,
        sources=sources,
        additional_instruction=(
            "Verify the page does NOT provide a concrete list of prerequisite course names. "
            "If it only describes expected background/competency without enumerating specific courses, treat as 'not explicitly listed'."
        ),
    )

    # UT Austin GPA Requirements group
    gpa_group = evaluator.add_parallel(
        id="UT_Austin_GPA_Requirements",
        desc="State UT Austin minimum GPA requirements with scope.",
        parent=program_node,
        critical=True
    )

    # UT Austin Upper Division GPA
    upper_node = evaluator.add_leaf(
        id="UT_Austin_Upper_Division_GPA",
        desc="Minimum 3.0 GPA in upper-division courses (scope: upper-division).",
        parent=gpa_group,
        critical=True
    )
    upper_claim = (
        f"UT Austin requires at least a 3.0 GPA in upper-division coursework for CS MS admission consideration. "
        f"Answer mentions: '{(uta.gpa_upper_division or 'unspecified')}'."
    )
    await evaluator.verify(
        claim=upper_claim,
        node=upper_node,
        sources=sources,
        additional_instruction=(
            "Confirm the UT Austin page explicitly states a minimum 3.0 GPA for upper-division coursework (scope: upper-division). "
            "Accept equivalent wording indicating the same threshold and scope."
        ),
    )

    # UT Austin Graduate Work GPA
    grad_node = evaluator.add_leaf(
        id="UT_Austin_Graduate_Work_GPA",
        desc="Minimum 3.0 GPA in any completed graduate work (scope: graduate work).",
        parent=gpa_group,
        critical=True
    )
    grad_claim = (
        f"UT Austin requires at least a 3.0 GPA in any completed graduate work. "
        f"Answer mentions: '{(uta.gpa_graduate_work or 'unspecified')}'."
    )
    await evaluator.verify(
        claim=grad_claim,
        node=grad_node,
        sources=sources,
        additional_instruction=(
            "Confirm the UT Austin page explicitly states a minimum 3.0 GPA requirement in any completed graduate work (scope: graduate work)."
        ),
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
    Evaluate an answer for CS MS prerequisites and GPA requirements across UTK, UH, and UT Austin.
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
        default_model=model,
    )

    # Top-level critical node mirroring the rubric root
    top = evaluator.add_parallel(
        id="CS_Masters_Program_Requirements",
        desc="Document prerequisite (math + CS) and GPA requirements for CS MS programs at UTK, UH, and UT Austin, each with an official source URL.",
        parent=root,
        critical=True
    )

    # Extract program requirements from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_program_requirements(),
        template_class=ProgramsExtraction,
        extraction_name="program_requirements",
    )

    # Optional: Record ground truth expectations (as guidance)
    evaluator.add_ground_truth({
        "UTK_expected_math": "Two semesters of calculus plus two additional semesters of college mathematics (e.g., linear algebra, differential equations, probability).",
        "UTK_expected_cs": "A formal languages course and a systems programming course.",
        "UTK_expected_gpa": {
            "overall": "Minimum 3.0 overall GPA",
            "senior_year": "Minimum 3.0 GPA in senior year"
        },
        "UH_expected_math": "Calculus I, Calculus II, and Linear Algebra completed before admission.",
        "UH_expected_cs_proficiency": "Proficiency in COSC 6305, COSC 6306, COSC 6308, COSC 6309, COSC 6310 (or documented equivalents/topics).",
        "UT_Austin_prereq_disclosure": "Specific prerequisite courses are not explicitly listed on the main CS MS admissions page.",
        "UT_Austin_expected_gpa": {
            "upper_division": "Minimum 3.0 GPA in upper-division coursework",
            "graduate_work": "Minimum 3.0 GPA in any completed graduate work"
        }
    }, gt_type="expected_requirements")

    # Build verification subtrees per university
    await verify_utk_program(evaluator, top, extracted.utk)
    await verify_uh_program(evaluator, top, extracted.uh)
    await verify_ut_austin_program(evaluator, top, extracted.ut_austin)

    # Return structured evaluation summary
    return evaluator.get_summary()