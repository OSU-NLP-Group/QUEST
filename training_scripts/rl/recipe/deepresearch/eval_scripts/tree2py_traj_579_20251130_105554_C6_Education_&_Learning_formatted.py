import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "teacher_cert_pathways"
TASK_DESCRIPTION = (
    "You hold a bachelor's degree in mathematics from a regionally accredited university with a cumulative GPA of 2.8. "
    "You currently reside in California and want to pursue a post-baccalaureate pathway to become a certified secondary "
    "mathematics teacher. Your constraints are: (1) maximum budget of $15,000 for the entire program, (2) must complete "
    "the program within 18 months, and (3) you plan to eventually relocate to Pennsylvania to teach due to family reasons. "
    "Compare the teacher certification/credential pathways at California State University Northridge (CSUN), Penn State University, "
    "and Purdue University. For each institution: (1) Determine whether you meet the minimum GPA requirements for admission to their "
    "post-baccalaureate secondary mathematics teaching program or credential program. (2) For the programs where you ARE eligible, "
    "calculate the total estimated cost including tuition and required application fees, and determine whether the program fits within "
    "your $15,000 budget. (3) For the eligible programs that fit your budget, identify whether the program can be completed within your "
    "18-month timeline. (4) For any programs that meet all three criteria (eligibility, budget, timeline), explain how a credential or "
    "teaching license obtained from that program can be used to obtain a teaching certificate in Pennsylvania, referencing the specific "
    "reciprocity mechanisms or interstate agreements that apply. Provide specific numerical values for costs, identify the GPA requirements "
    "for each institution, state the typical program completion time, and include reference URLs that support your findings for each institution's "
    "requirements, costs, and timeline."
)

BUDGET_LIMIT = 15000.0
TIMELINE_LIMIT_MONTHS = 18
APPLICANT_GPA = 2.8

EXPECTED_MIN_GPA = {
    "CSUN": "2.5",
    "Penn State": "3.0",
    "Purdue": "2.5",
}


class InstitutionPathway(BaseModel):
    institution: Optional[str] = None
    program_name: Optional[str] = None
    program_urls: List[str] = Field(default_factory=list)

    min_gpa: Optional[str] = None
    min_gpa_urls: List[str] = Field(default_factory=list)

    gpa_conclusion: Optional[str] = None  # expected values: "eligible" or "ineligible"

    total_cost: Optional[str] = None  # numeric string (e.g., "14250", "$14,250")
    cost_notes: Optional[str] = None
    cost_urls: List[str] = Field(default_factory=list)
    budget_fit: Optional[str] = None  # "within_budget" or "over_budget"

    timeline_months: Optional[str] = None  # numeric or textual (e.g., "12", "3 semesters (~12 months)")
    timeline_urls: List[str] = Field(default_factory=list)
    timeline_fit: Optional[str] = None  # "within_18" or "over_18"

    reciprocity_explanation: Optional[str] = None
    reciprocity_urls: List[str] = Field(default_factory=list)


class PathwaysExtraction(BaseModel):
    csun: Optional[InstitutionPathway] = None
    penn_state: Optional[InstitutionPathway] = None
    purdue: Optional[InstitutionPathway] = None


def prompt_extract_pathways() -> str:
    return """
    Extract the structured comparison details for three institutions: CSUN, Penn State, and Purdue, about post-baccalaureate secondary mathematics teacher certification/credential pathways.

    For each institution (CSUN, Penn State, Purdue), extract the following fields exactly as named:

    1) institution: The institution label ("CSUN", "Penn State", or "Purdue").
    2) program_name: The exact name of the program/pathway being evaluated (e.g., "Single Subject Credential - Mathematics", "Post-baccalaureate teacher certification – Mathematics", etc.).
    3) program_urls: An array of explicit URLs provided in the answer that point to the program page(s) or official requirements pages.
    4) min_gpa: The minimum GPA requirement stated by the answer for admission to the program. Return as a simple string (e.g., "2.5", "3.0"). If multiple are mentioned, include the primary overall requirement.
    5) min_gpa_urls: An array of URLs in the answer that specifically support the GPA requirement.
    6) gpa_conclusion: The answer’s stated conclusion for whether a 2.8 GPA meets the program’s minimum requirement ("eligible" or "ineligible"). If not stated, return null.
    7) total_cost: The answer’s numeric total estimated cost for the entire program including tuition and required application fees. Return as a string (e.g., "14250", "$14,250"). If not directly provided, return null.
    8) cost_notes: Any textual notes summarizing components (tuition, units, fees) the answer used to compute total cost. If none, return null.
    9) cost_urls: An array of URLs that support the cost components (tuition rates, application fees).
    10) budget_fit: The answer’s conclusion whether the total cost is within the $15,000 budget ("within_budget" or "over_budget"). If not concluded, return null.
    11) timeline_months: The answer’s stated typical completion time in months (or a textual equivalence like semesters with approximate months). Return as a string.
    12) timeline_urls: An array of URLs supporting the program’s typical completion timeline.
    13) timeline_fit: The answer’s conclusion whether the program can be completed within 18 months ("within_18" or "over_18"). If not concluded, return null.
    14) reciprocity_explanation: For programs that meet eligibility + budget + timeline, the answer’s explanation of how the resulting credential/license can be used to obtain Pennsylvania certification (e.g., via NASDTEC or PA PDE out-of-state pathway). If the program did not meet all three criteria, return null or the answer’s explicit statement.
    15) reciprocity_urls: An array of URLs provided in the answer that support the reciprocity explanation (e.g., Pennsylvania Department of Education guidance, NASDTEC resources, etc.).

    Return a JSON object with top-level keys: "csun", "penn_state", "purdue", each containing the above fields. If any field is missing from the answer, set it to null or an empty array as appropriate. Only include URLs that are explicitly present in the answer (plain or markdown link form).
    """


def _sources_or_fallback(primary: List[str], fallback: List[str]) -> List[str]:
    return primary if primary else fallback


def _extract_first_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", s.replace(",", ""))
        if m:
            return float(m.group(0))
    except Exception:
        return None
    return None


def _normalize_flag(val: Optional[str], positives: List[str]) -> bool:
    if not val:
        return False
    v = val.strip().lower()
    return any(p in v for p in positives)


async def verify_institution(
    evaluator: Evaluator,
    parent_node,
    key: str,
    label: str,
    pathway: InstitutionPathway,
    expected_min_gpa: Optional[str] = None,
) -> None:
    inst_node = evaluator.add_sequential(
        id=f"{key}_Pathway",
        desc=f"{label} pathway evaluation (GPA eligibility → cost/budget if eligible → timeline if eligible+within budget → PA reciprocity if all criteria met).",
        parent=parent_node,
        critical=False,
    )

    # Program identification existence
    program_exists = bool(pathway and pathway.program_name and pathway.program_urls)
    evaluator.add_custom_node(
        result=program_exists,
        id=f"{key}_Program_Identified_Exists",
        desc=f"{label}: Program name and at least one supporting URL are provided.",
        parent=inst_node,
        critical=True,
    )

    # Program identification verification with URLs
    prog_id_node = evaluator.add_leaf(
        id=f"{key}_Program_Identified",
        desc=f"Names the specific {label} credential/pathway for secondary mathematics and provides a supporting URL.",
        parent=inst_node,
        critical=True,
    )
    program_claim = (
        f"The provided URL(s) correspond to the official {label} page(s) for the post-baccalaureate secondary mathematics "
        f"teacher certification/credential program described as '{pathway.program_name}'."
    )
    await evaluator.verify(
        claim=program_claim,
        node=prog_id_node,
        sources=pathway.program_urls,
        additional_instruction="Confirm the page(s) are about this institution's program for secondary/single subject mathematics teacher certification/credential.",
    )

    # Minimum GPA requirement presence
    min_gpa_present = bool(pathway and pathway.min_gpa)
    evaluator.add_custom_node(
        result=min_gpa_present,
        id=f"{key}_Min_GPA_Present",
        desc=f"{label}: The minimum GPA requirement value is stated in the answer.",
        parent=inst_node,
        critical=True,
    )

    # Minimum GPA requirement verification (with expected number context if provided)
    min_gpa_verify_node = evaluator.add_leaf(
        id=f"{key}_Min_GPA_Requirement",
        desc=f"States {label} minimum GPA requirement for admission and provides a supporting URL.",
        parent=inst_node,
        critical=True,
    )
    min_gpa_sources = _sources_or_fallback(pathway.min_gpa_urls, pathway.program_urls)
    if expected_min_gpa:
        min_gpa_claim = (
            f"The minimum GPA requirement for admission to this {label} program is {expected_min_gpa}."
        )
        add_ins = (
            f"Verify the minimum GPA value on the provided official page(s). "
            f"If the page shows a different threshold than {expected_min_gpa}, mark as not supported."
        )
    else:
        min_gpa_claim = (
            f"The minimum GPA requirement for admission to this {label} program is {pathway.min_gpa}."
        )
        add_ins = "Verify the stated GPA threshold using the provided URL(s)."
    await evaluator.verify(
        claim=min_gpa_claim,
        node=min_gpa_verify_node,
        sources=min_gpa_sources,
        additional_instruction=add_ins,
    )

    # GPA eligibility conclusion verification (logical check)
    gpa_elig_node = evaluator.add_leaf(
        id=f"{key}_GPA_Eligibility_Conclusion",
        desc=f"Correctly concludes whether applicant GPA 2.8 meets {label} minimum GPA requirement and labels eligible/ineligible.",
        parent=inst_node,
        critical=True,
    )
    min_gpa_number = _extract_first_float(pathway.min_gpa if pathway and pathway.min_gpa else expected_min_gpa)
    if min_gpa_number is not None:
        logical_meets = APPLICANT_GPA >= min_gpa_number
        stated = (pathway.gpa_conclusion or "").strip().lower()
        expected_text = "eligible" if logical_meets else "ineligible"
        gpa_claim = (
            f"Given a minimum GPA requirement of {min_gpa_number}, an applicant with a 2.8 GPA should be '{expected_text}'. "
            f"The answer's conclusion is '{stated}'. This conclusion is correct."
        )
        add_ins = "Judge the correctness of the conclusion strictly based on comparing 2.8 against the minimum GPA threshold."
    else:
        gpa_claim = (
            f"The answer concludes the applicant is '{pathway.gpa_conclusion}' for {label}. "
            f"Evaluate whether this conclusion is correct based on the minimum GPA requirement described on the program page(s)."
        )
        add_ins = "Use basic logic: compare 2.8 to the minimum GPA requirement stated on the page(s); judge correctness."
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_elig_node,
        additional_instruction=add_ins,
    )

    # Gating: proceed to cost only if eligible (non-critical gating)
    eligible_gate = evaluator.add_custom_node(
        result=_normalize_flag(pathway.gpa_conclusion, ["eligible", "meets", "qualifies", "yes"]),
        id=f"{key}_Eligible_Gate",
        desc=f"{label}: Non-critical gate indicating the applicant is eligible (controls whether cost step is needed).",
        parent=inst_node,
        critical=False,
    )

    # Cost and budget check
    cost_node = evaluator.add_leaf(
        id=f"{key}_Cost_and_Budget_Check",
        desc=f"If eligible: provides numeric total estimated cost (tuition + required application fees) and concludes whether within $15,000; includes supporting URL(s).",
        parent=inst_node,
        critical=True,
    )
    cost_claim = (
        f"The total estimated cost for the entire {label} program, including tuition and required application fees, is stated as '{pathway.total_cost}'. "
        f"Based on this total, the program is '{pathway.budget_fit}' relative to the $15,000 budget."
    )
    await evaluator.verify(
        claim=cost_claim,
        node=cost_node,
        sources=pathway.cost_urls,
        additional_instruction=(
            "Verify the total includes tuition and required application fees using the provided cost URLs. "
            "Then judge whether the total is within or over the $15,000 budget as claimed."
        ),
        extra_prerequisites=[eligible_gate, gpa_elig_node],
    )

    # Gating: proceed to timeline only if eligible and within budget (non-critical gating)
    budget_gate = evaluator.add_custom_node(
        result=_normalize_flag(pathway.budget_fit, ["within_budget", "under 15000", "within $15000", "yes"]),
        id=f"{key}_Budget_Gate",
        desc=f"{label}: Non-critical gate indicating total cost fits within $15,000 budget (controls timeline step).",
        parent=inst_node,
        critical=False,
    )

    # Timeline check
    timeline_node = evaluator.add_leaf(
        id=f"{key}_Timeline_Check",
        desc=f"If eligible AND within budget: states typical completion time and whether within 18 months; includes supporting URL.",
        parent=inst_node,
        critical=True,
    )
    timeline_claim = (
        f"The typical completion time for the {label} program is stated as '{pathway.timeline_months}', "
        f"and the program is '{pathway.timeline_fit}' relative to the 18-month timeline."
    )
    await evaluator.verify(
        claim=timeline_claim,
        node=timeline_node,
        sources=pathway.timeline_urls,
        additional_instruction=(
            "Verify the typical time to complete the program using the provided URL(s). "
            "Then judge whether this timeline fits within 18 months as claimed."
        ),
        extra_prerequisites=[eligible_gate, cost_node, budget_gate],
    )

    # Gating: proceed to PA reciprocity only if eligibility + budget + timeline all satisfied (non-critical gating)
    all_three_gate = evaluator.add_custom_node(
        result=(
            _normalize_flag(pathway.gpa_conclusion, ["eligible", "meets", "qualifies", "yes"])
            and _normalize_flag(pathway.budget_fit, ["within_budget", "under 15000", "within $15000", "yes"])
            and _normalize_flag(pathway.timeline_fit, ["within_18", "within 18", "<=18", "yes"])
        ),
        id=f"{key}_AllThree_Gate",
        desc=f"{label}: Non-critical gate indicating all three criteria (eligibility, budget, timeline) are met (controls PA reciprocity step).",
        parent=inst_node,
        critical=False,
    )

    # PA reciprocity check
    reciprocity_node = evaluator.add_leaf(
        id=f"{key}_PA_Reciprocity",
        desc=f"If all three criteria are met: explains how credential/license can be used to obtain PA certification, referencing reciprocity/agreements and providing supporting URL(s).",
        parent=inst_node,
        critical=True,
    )
    reciprocity_claim = (
        f"The explanation describes how a credential/license from {label} can be used to obtain a Pennsylvania teaching certificate "
        f"(e.g., via reciprocity or PA PDE out-of-state pathways): '{pathway.reciprocity_explanation}'."
    )
    await evaluator.verify(
        claim=reciprocity_claim,
        node=reciprocity_node,
        sources=pathway.reciprocity_urls,
        additional_instruction=(
            "Check the provided URLs (e.g., Pennsylvania Department of Education guidance, NASDTEC resources) to verify the described reciprocity or out-of-state certification pathway is valid."
        ),
        extra_prerequisites=[eligible_gate, cost_node, budget_gate, timeline_node, all_three_gate],
    )


async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
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

    analysis_node = evaluator.add_parallel(
        id="Teacher_Certification_Pathway_Analysis",
        desc="Compare CSUN, Penn State, and Purdue pathways for GPA eligibility, total cost vs $15,000, completion time vs 18 months, and PA reciprocity (with numeric values and supporting URLs).",
        parent=root,
        critical=False,
    )

    extraction = await evaluator.extract(
        prompt=prompt_extract_pathways(),
        template_class=PathwaysExtraction,
        extraction_name="pathways_extraction",
    )

    evaluator.add_ground_truth({
        "constraints": {
            "budget_limit": BUDGET_LIMIT,
            "timeline_limit_months": TIMELINE_LIMIT_MONTHS,
            "applicant_gpa": APPLICANT_GPA,
        },
        "expected_min_gpa": EXPECTED_MIN_GPA,
    }, gt_type="expected_parameters")

    csun_path = extraction.csun or InstitutionPathway(institution="CSUN")
    penn_path = extraction.penn_state or InstitutionPathway(institution="Penn State")
    purdue_path = extraction.purdue or InstitutionPathway(institution="Purdue")

    await verify_institution(
        evaluator=evaluator,
        parent_node=analysis_node,
        key="CSUN",
        label="California State University Northridge (CSUN)",
        pathway=csun_path,
        expected_min_gpa=EXPECTED_MIN_GPA.get("CSUN"),
    )

    await verify_institution(
        evaluator=evaluator,
        parent_node=analysis_node,
        key="Penn_State",
        label="Penn State University",
        pathway=penn_path,
        expected_min_gpa=EXPECTED_MIN_GPA.get("Penn State"),
    )

    await verify_institution(
        evaluator=evaluator,
        parent_node=analysis_node,
        key="Purdue",
        label="Purdue University",
        pathway=purdue_path,
        expected_min_gpa=EXPECTED_MIN_GPA.get("Purdue"),
    )

    return evaluator.get_summary()