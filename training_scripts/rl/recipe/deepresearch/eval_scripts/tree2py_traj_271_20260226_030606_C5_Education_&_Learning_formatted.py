import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "unc_vs_ivy_eligibility_financial_aid"
TASK_DESCRIPTION = (
    "A high school senior who plays football is considering applying to either the University of North Carolina "
    "(under head coach Bill Belichick) or Harvard/Yale. Provide a detailed comparison of: (1) The NCAA Division I "
    "academic eligibility requirements that would apply at UNC, including both initial eligibility requirements for "
    "incoming student-athletes and continuing eligibility requirements once enrolled; (2) The Ivy League academic "
    "eligibility requirements that would apply at Harvard or Yale, including any additional requirements or systems "
    "beyond NCAA Division I standards; (3) The key difference in financial aid and scholarship policies between NCAA "
    "Division I institutions like UNC and Ivy League schools. Include specific GPA requirements, core course "
    "requirements, credit hour requirements, and any unique systems (such as the Academic Index) used by these institutions."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Claim(BaseModel):
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class UNCInitial(BaseModel):
    core_courses_requirement: Optional[Claim] = None
    minimum_core_course_gpa: Optional[Claim] = None
    high_school_graduation: Optional[Claim] = None


class UNCContinuing(BaseModel):
    minimum_cumulative_gpa_unc: Optional[Claim] = None
    semester_credit_hours_unc: Optional[Claim] = None
    annual_credit_hours_unc: Optional[Claim] = None


class AcademicIndex(BaseModel):
    ai_inputs: Optional[Claim] = None
    ai_range: Optional[Claim] = None
    ai_minimum_score: Optional[Claim] = None
    team_average_ai_rule: Optional[Claim] = None


class IvyRequirements(BaseModel):
    ivy_ncaa_baseline_alignment: Optional[Claim] = None
    ivy_continuing_eligibility_credit_hours: Optional[Claim] = None
    academic_index_system: Optional[AcademicIndex] = None


class FinancialAid(BaseModel):
    ncaa_d1_athletic_scholarships: Optional[Claim] = None
    ivy_need_based_only_policy: Optional[Claim] = None
    ivy_1954_agreement_policy_basis: Optional[Claim] = None


class EligibilityExtraction(BaseModel):
    initial_eligibility_unc: Optional[UNCInitial] = None
    continuing_eligibility_unc: Optional[UNCContinuing] = None
    ivy_league_requirements_harvard_yale: Optional[IvyRequirements] = None
    financial_aid_and_scholarships: Optional[FinancialAid] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract, exactly as asserted in the answer, the specific claims and their cited URL sources for each of the following items. For every item, return:
- statement: the exact claim or concise paraphrase made by the answer text.
- sources: a list of URL(s) explicitly cited in the answer for that claim (markdown links are OK; extract actual URLs). If no URL is provided in the answer for a claim, return an empty list.

Organize your JSON as follows:

{
  "initial_eligibility_unc": {
    "core_courses_requirement": { "statement": string | null, "sources": [urls...] },
    "minimum_core_course_gpa": { "statement": string | null, "sources": [urls...] },
    "high_school_graduation": { "statement": string | null, "sources": [urls...] }
  },
  "continuing_eligibility_unc": {
    "minimum_cumulative_gpa_unc": { "statement": string | null, "sources": [urls...] },
    "semester_credit_hours_unc": { "statement": string | null, "sources": [urls...] },
    "annual_credit_hours_unc": { "statement": string | null, "sources": [urls...] }
  },
  "ivy_league_requirements_harvard_yale": {
    "ivy_ncaa_baseline_alignment": { "statement": string | null, "sources": [urls...] },
    "ivy_continuing_eligibility_credit_hours": { "statement": string | null, "sources": [urls...] },
    "academic_index_system": {
      "ai_inputs": { "statement": string | null, "sources": [urls...] },
      "ai_range": { "statement": string | null, "sources": [urls...] },
      "ai_minimum_score": { "statement": string | null, "sources": [urls...] },
      "team_average_ai_rule": { "statement": string | null, "sources": [urls...] }
    }
  },
  "financial_aid_and_scholarships": {
    "ncaa_d1_athletic_scholarships": { "statement": string | null, "sources": [urls...] },
    "ivy_need_based_only_policy": { "statement": string | null, "sources": [urls...] },
    "ivy_1954_agreement_policy_basis": { "statement": string | null, "sources": [urls...] }
  }
}

Detailed guidance for each item:
- UNC initial eligibility:
  • core_courses_requirement: The answer’s description of the NCAA Division I 16 core-course requirement and the subject-area distribution as stated.
  • minimum_core_course_gpa: The answer’s stated NCAA Division I minimum core-course GPA (e.g., 2.3).
  • high_school_graduation: The claim that DI initial eligibility requires high school graduation.
- UNC continuing eligibility:
  • minimum_cumulative_gpa_unc: The answer’s stated minimum cumulative GPA to remain eligible at UNC (often aligned with “good academic standing”).
  • semester_credit_hours_unc: The answer’s stated “6 degree-applicable hours each semester” or equivalent per-term credit-hour requirement.
  • annual_credit_hours_unc: The answer’s stated “18 degree-applicable hours each academic year (fall+spring)” or equivalent per-year requirement.
- Ivy League:
  • ivy_ncaa_baseline_alignment: The claim that Ivy League eligibility uses NCAA DI initial eligibility as baseline.
  • ivy_continuing_eligibility_credit_hours: The answer’s stated credit-hour/progress-toward-degree requirements that apply once enrolled (as DI athletes).
  • academic_index_system:
     – ai_inputs: The components used to compute AI (e.g., GPA/class rank/standardized tests) as described in the answer.
     – ai_range: The stated range (e.g., 60–240).
     – ai_minimum_score: Any stated minimum score (e.g., 176) required or referenced.
     – team_average_ai_rule: The team-average AI rule (e.g., within one standard deviation of the school’s admitted class).
- Financial aid/scholarships:
  • ncaa_d1_athletic_scholarships: The claim that DI schools (like UNC) may award athletic scholarships.
  • ivy_need_based_only_policy: The claim that Ivy League schools provide only need-based aid (no athletic or merit scholarships).
  • ivy_1954_agreement_policy_basis: The claim that the Ivy no-athletic-scholarship policy is mandated by the 1954 Ivy Group Agreement.

Rules:
- Extract only URLs actually present in the answer.
- Do not infer or create data not present in the answer.
- If the answer does not cover an item, set its statement to null and sources to [].
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _claim_and_sources(c: Optional[Claim]) -> (str, List[str]):
    if c is None:
        return "", []
    return (c.statement or "").strip(), c.sources or []


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_unc_requirements(evaluator: Evaluator, parent_node, data: EligibilityExtraction) -> None:
    # Parent node for UNC NCAA Division I requirements (critical)
    unc_node = evaluator.add_parallel(
        id="ncaa_division_i_requirements_unc",
        desc="Accurate NCAA Division I academic eligibility requirements that apply at UNC (initial + continuing)",
        parent=parent_node,
        critical=True
    )

    # Initial Eligibility (critical)
    initial_node = evaluator.add_parallel(
        id="initial_eligibility_unc",
        desc="Initial NCAA Division I eligibility requirements for incoming student-athletes at UNC",
        parent=unc_node,
        critical=True
    )

    # core courses requirement
    leaf_core = evaluator.add_leaf(
        id="core_courses_requirement",
        desc="Complete 16 NCAA-approved core courses with the specified distribution (English/Math/Science/Social Science as given)",
        parent=initial_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(
        data.initial_eligibility_unc.core_courses_requirement if data.initial_eligibility_unc else None
    )
    await evaluator.verify(
        claim=stmt,
        node=leaf_core,
        sources=srcs,
        additional_instruction=(
            "Verify that the cited source(s) explicitly state NCAA Division I initial-eligibility requires 16 NCAA-approved core courses "
            "and includes the subject-area distribution (e.g., English, math [Algebra I or higher], natural/physical science including lab, "
            "social science, and additional core). Minor wording variations are acceptable if the substantive distribution and total of 16 are present."
        )
    )

    # minimum core-course GPA 2.3
    leaf_gpa = evaluator.add_leaf(
        id="minimum_core_course_gpa",
        desc="Earn a minimum NCAA core-course GPA of 2.3 for Division I initial eligibility",
        parent=initial_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(
        data.initial_eligibility_unc.minimum_core_course_gpa if data.initial_eligibility_unc else None
    )
    await evaluator.verify(
        claim=stmt,
        node=leaf_gpa,
        sources=srcs,
        additional_instruction=(
            "Confirm the source(s) state the NCAA Division I minimum core-course GPA is 2.3 (on a 4.0 scale). "
            "Allow formatting like 2.300 or 'at least 2.3'."
        )
    )

    # high school graduation
    leaf_grad = evaluator.add_leaf(
        id="high_school_graduation",
        desc="Graduate from high school",
        parent=initial_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(
        data.initial_eligibility_unc.high_school_graduation if data.initial_eligibility_unc else None
    )
    await evaluator.verify(
        claim=stmt,
        node=leaf_grad,
        sources=srcs,
        additional_instruction=(
            "Verify that the cited source(s) state high school graduation is required for NCAA Division I initial eligibility."
        )
    )

    # Continuing Eligibility (critical)
    cont_node = evaluator.add_parallel(
        id="continuing_eligibility_unc",
        desc="Continuing eligibility requirements once enrolled at UNC",
        parent=unc_node,
        critical=True
    )

    # minimum cumulative GPA at UNC (e.g., 2.0)
    leaf_cont_gpa = evaluator.add_leaf(
        id="minimum_cumulative_gpa_unc",
        desc="Maintain a minimum cumulative GPA of 2.0 at UNC for continuing eligibility",
        parent=cont_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(
        data.continuing_eligibility_unc.minimum_cumulative_gpa_unc if data.continuing_eligibility_unc else None
    )
    await evaluator.verify(
        claim=stmt,
        node=leaf_cont_gpa,
        sources=srcs,
        additional_instruction=(
            "Check whether UNC or NCAA sources indicate the minimum cumulative GPA to remain eligible (often 'good academic standing'). "
            "If UNC's policy defines good standing as 2.0 and the claim equates that to the athletic eligibility GPA, consider it consistent."
        )
    )

    # semester 6 degree-applicable hours
    leaf_6hrs = evaluator.add_leaf(
        id="semester_credit_hours_unc",
        desc="Pass at least 6 degree-applicable hours each semester at UNC",
        parent=cont_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(
        data.continuing_eligibility_unc.semester_credit_hours_unc if data.continuing_eligibility_unc else None
    )
    await evaluator.verify(
        claim=stmt,
        node=leaf_6hrs,
        sources=srcs,
        additional_instruction=(
            "Verify that the source(s) state the per-term (semester) 6-hour progress-toward-degree requirement for NCAA Division I student-athletes."
        )
    )

    # annual 18 degree-applicable hours
    leaf_18hrs = evaluator.add_leaf(
        id="annual_credit_hours_unc",
        desc="Pass at least 18 degree-applicable hours each academic year (fall and spring) at UNC",
        parent=cont_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(
        data.continuing_eligibility_unc.annual_credit_hours_unc if data.continuing_eligibility_unc else None
    )
    await evaluator.verify(
        claim=stmt,
        node=leaf_18hrs,
        sources=srcs,
        additional_instruction=(
            "Verify that the source(s) state the annual progress-toward-degree credit hour requirement (e.g., 18 in fall+spring) for NCAA Division I student-athletes. "
            "Accept equivalent phrasing if it clearly asserts the same annual threshold."
        )
    )


async def build_ivy_requirements(evaluator: Evaluator, parent_node, data: EligibilityExtraction) -> None:
    ivy_node = evaluator.add_parallel(
        id="ivy_league_requirements_harvard_yale",
        desc="Accurate Ivy League academic eligibility requirements for Harvard/Yale, including additions beyond NCAA Division I standards",
        parent=parent_node,
        critical=True
    )

    # Ivy uses NCAA DI initial baseline
    leaf_baseline = evaluator.add_leaf(
        id="ivy_ncaa_baseline_alignment",
        desc="States that Ivy League eligibility uses NCAA Division I initial-eligibility baseline requirements (i.e., meeting NCAA baseline as the foundation)",
        parent=ivy_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(
        (data.ivy_league_requirements_harvard_yale.ivy_ncaa_baseline_alignment
         if data.ivy_league_requirements_harvard_yale else None)
    )
    await evaluator.verify(
        claim=stmt,
        node=leaf_baseline,
        sources=srcs,
        additional_instruction=(
            "Verify that the Ivy League policy or university sources state that NCAA Division I initial-eligibility standards are the baseline for Ivy League student-athletes."
        )
    )

    # Ivy continuing eligibility / credit-hour (progress-toward-degree)
    leaf_ptd = evaluator.add_leaf(
        id="ivy_continuing_eligibility_credit_hours",
        desc="Describes the continuing eligibility / credit-hour (progress-toward-degree) requirements that apply once enrolled at Harvard/Yale as NCAA Division I student-athletes (i.e., provides the applicable credit-hour/academic progress requirements rather than only admissions/recruiting standards)",
        parent=ivy_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(
        (data.ivy_league_requirements_harvard_yale.ivy_continuing_eligibility_credit_hours
         if data.ivy_league_requirements_harvard_yale else None)
    )
    await evaluator.verify(
        claim=stmt,
        node=leaf_ptd,
        sources=srcs,
        additional_instruction=(
            "Verify that the cited source(s) describe the progress-toward-degree or credit-hour requirements that apply to Ivy League student-athletes while enrolled (e.g., per-term and annual hour requirements)."
        )
    )

    # Academic Index system (critical parallel sub-node)
    ai_node = evaluator.add_parallel(
        id="academic_index_system",
        desc="Describes the Ivy League Academic Index (AI) system and required rules/thresholds",
        parent=ivy_node,
        critical=True
    )

    ai_data: Optional[AcademicIndex] = (
        data.ivy_league_requirements_harvard_yale.academic_index_system
        if data.ivy_league_requirements_harvard_yale else None
    )

    # AI inputs
    leaf_ai_inputs = evaluator.add_leaf(
        id="ai_inputs",
        desc="AI is calculated using GPA, class rank, and standardized test scores",
        parent=ai_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(ai_data.ai_inputs if ai_data else None)
    await evaluator.verify(
        claim=stmt,
        node=leaf_ai_inputs,
        sources=srcs,
        additional_instruction=(
            "Verify that the source(s) explicitly describe the Academic Index components including GPA, class rank, and standardized test scores (SAT/ACT or similar). "
            "Minor synonyms/phrasing are acceptable if the same components are clearly conveyed."
        )
    )

    # AI range
    leaf_ai_range = evaluator.add_leaf(
        id="ai_range",
        desc="AI ranges from 60 to 240 points",
        parent=ai_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(ai_data.ai_range if ai_data else None)
    await evaluator.verify(
        claim=stmt,
        node=leaf_ai_range,
        sources=srcs,
        additional_instruction=(
            "Verify that the source(s) state (or clearly imply) that the Ivy Academic Index is on a 60–240 scale. "
            "If a source provides an equivalent historical scale, consider it consistent when it clearly maps to 60–240."
        )
    )

    # AI minimum score (e.g., 176)
    leaf_ai_min = evaluator.add_leaf(
        id="ai_minimum_score",
        desc="Minimum AI score of 176 required for Ivy League admission (per constraints)",
        parent=ai_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(ai_data.ai_minimum_score if ai_data else None)
    await evaluator.verify(
        claim=stmt,
        node=leaf_ai_min,
        sources=srcs,
        additional_instruction=(
            "Verify that the cited source(s) explicitly reference a minimum AI threshold around 176 (or the exact number claimed) "
            "for Ivy League recruited athletes or admissions consideration. If the source uses approximate or policy-specific wording, "
            "accept if it clearly supports the stated threshold."
        )
    )

    # Team average AI rule
    leaf_ai_team = evaluator.add_leaf(
        id="team_average_ai_rule",
        desc="Each team’s average AI must be within one standard deviation of the school’s overall admitted class",
        parent=ai_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(ai_data.team_average_ai_rule if ai_data else None)
    await evaluator.verify(
        claim=stmt,
        node=leaf_ai_team,
        sources=srcs,
        additional_instruction=(
            "Verify that the source(s) state the Academic Index rule that team average AI must be within one standard deviation "
            "of the school's overall admitted class AI (or substantively equivalent formulation)."
        )
    )


async def build_financial_aid(evaluator: Evaluator, parent_node, data: EligibilityExtraction) -> None:
    fin_node = evaluator.add_parallel(
        id="financial_aid_and_scholarships",
        desc="Key difference in financial aid and scholarship policies between NCAA Division I (UNC-like) and Ivy League schools",
        parent=parent_node,
        critical=True
    )

    # DI athletic scholarships
    leaf_d1_sch = evaluator.add_leaf(
        id="ncaa_d1_athletic_scholarships",
        desc="NCAA Division I institutions like UNC can offer athletic scholarships",
        parent=fin_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(
        data.financial_aid_and_scholarships.ncaa_d1_athletic_scholarships if data.financial_aid_and_scholarships else None
    )
    await evaluator.verify(
        claim=stmt,
        node=leaf_d1_sch,
        sources=srcs,
        additional_instruction=(
            "Verify that NCAA Division I schools (such as UNC) may award athletic scholarships (including football). "
            "Institutional or NCAA sources indicating scholarship availability suffice."
        )
    )

    # Ivy need-based only policy
    leaf_ivy_need = evaluator.add_leaf(
        id="ivy_need_based_only_policy",
        desc="Ivy League schools provide only need-based financial aid (i.e., no athletic scholarships and no merit-based scholarships)",
        parent=fin_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(
        data.financial_aid_and_scholarships.ivy_need_based_only_policy if data.financial_aid_and_scholarships else None
    )
    await evaluator.verify(
        claim=stmt,
        node=leaf_ivy_need,
        sources=srcs,
        additional_instruction=(
            "Verify that Ivy League institutions do not award athletic or merit scholarships and provide need-based aid only."
        )
    )

    # Ivy 1954 Agreement basis
    leaf_ivy_1954 = evaluator.add_leaf(
        id="ivy_1954_agreement_policy_basis",
        desc="Ivy League no-scholarship policy is mandated by the 1954 Ivy League Agreement",
        parent=fin_node,
        critical=True
    )
    stmt, srcs = _claim_and_sources(
        data.financial_aid_and_scholarships.ivy_1954_agreement_policy_basis if data.financial_aid_and_scholarships else None
    )
    await evaluator.verify(
        claim=stmt,
        node=leaf_ivy_1954,
        sources=srcs,
        additional_instruction=(
            "Verify that the source(s) tie the Ivy League no-athletic-scholarship policy to the 1954 Ivy Group Agreement (or substantively equivalent founding agreement/policy)."
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
    """
    Evaluate an answer for NCAA Division I (UNC) vs Ivy League (Harvard/Yale) academic eligibility and financial aid policies.
    """
    # 1) Initialize evaluator and root
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

    # Create a critical top-level container to respect the rubric's critical root semantics
    overall_node = evaluator.add_parallel(
        id="overall_comparison_requirements",
        desc="Detailed comparison of (1) NCAA Division I academic eligibility at UNC (initial + continuing), (2) Ivy League (Harvard/Yale) academic eligibility including additions beyond NCAA, and (3) financial aid/scholarship policy differences between NCAA D1 and Ivy League",
        parent=root,
        critical=True
    )

    # 2) Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=EligibilityExtraction,
        extraction_name="extracted_eligibility_and_policies"
    )

    # 3) Build subtrees and perform verifications
    await build_unc_requirements(evaluator, overall_node, extracted)
    await build_ivy_requirements(evaluator, overall_node, extracted)
    await build_financial_aid(evaluator, overall_node, extracted)

    # 4) Return structured summary with verification tree
    return evaluator.get_summary()