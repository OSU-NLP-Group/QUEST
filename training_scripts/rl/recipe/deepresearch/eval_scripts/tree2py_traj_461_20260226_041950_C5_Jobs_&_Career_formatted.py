import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "levar_woods_career_progression"
TASK_DESCRIPTION = """A career counselor is preparing a case study on successful transitions to assistant head coach positions in college football. Using LeVar Woods' career progression from the University of Iowa to Michigan State University as an example, provide the following information:

1. How many total years did LeVar Woods serve on the Iowa coaching staff before joining Michigan State in January 2026?

2. Of those years at Iowa, how many were spent in on-field coaching positions (excluding administrative assistant roles)?

3. How many years did he serve as Iowa's special teams coordinator?

4. What was his annual base salary at Iowa for the 2024 season?

5. What is his annual salary at Michigan State for the 2026 season?

6. Based on the salaries identified above, what is the dollar amount of his salary increase from Iowa to Michigan State?

7. Assistant head coach positions at major college football programs typically require a minimum of 5 years of coaching experience at the collegiate level. Did LeVar Woods meet this typical minimum requirement based on his on-field coaching experience at Iowa?

For each answer, provide supporting reference URL(s) from credible sources that verify the information.
"""


# ----------------------------- Data Models ---------------------------------- #
class TenureAtIowa(BaseModel):
    total_years: Optional[str] = None
    start_year: Optional[str] = None  # e.g., "2008"
    end_year: Optional[str] = None    # e.g., "2025"
    sources: List[str] = Field(default_factory=list)


class OnFieldInfo(BaseModel):
    on_field_years: Optional[str] = None
    start_year: Optional[str] = None  # e.g., "2012"
    end_year: Optional[str] = None    # e.g., "2025"
    admin_exclusion_years: Optional[str] = None  # e.g., "2008-2011"
    sources: List[str] = Field(default_factory=list)


class CoordinatorInfo(BaseModel):
    coordinator_years: Optional[str] = None       # e.g., "8"
    start_year: Optional[str] = None              # e.g., "2017"
    end_year: Optional[str] = None                # e.g., "2025"
    duties_added_year: Optional[str] = None       # e.g., "2017"
    sources: List[str] = Field(default_factory=list)


class SalaryInfo(BaseModel):
    iowa_salary_2024: Optional[str] = None        # e.g., "$775,000"
    iowa_salary_sources: List[str] = Field(default_factory=list)
    msu_salary_2026: Optional[str] = None         # e.g., "$1,100,000"
    msu_salary_sources: List[str] = Field(default_factory=list)


class IncreaseInfo(BaseModel):
    salary_increase: Optional[str] = None         # e.g., "$325,000"
    calculation_text: Optional[str] = None        # e.g., "$1,100,000 - $775,000 = $325,000"


class RequirementInfo(BaseModel):
    typical_minimum_years: Optional[str] = None   # e.g., "5"
    met_minimum: Optional[str] = None             # e.g., "Yes" or "No"
    requirement_sources: List[str] = Field(default_factory=list)


class CareerExtraction(BaseModel):
    tenure: Optional[TenureAtIowa] = None
    on_field: Optional[OnFieldInfo] = None
    coordinator: Optional[CoordinatorInfo] = None
    salaries: Optional[SalaryInfo] = None
    increase: Optional[IncreaseInfo] = None
    requirement: Optional[RequirementInfo] = None


# --------------------------- Extraction Prompt ------------------------------ #
def prompt_extract_levar_woods() -> str:
    return """
    Extract structured information from the answer text about LeVar Woods' career progression. Return a JSON object with the following fields. Use strings for all numeric values. For each item, also extract the supporting URL(s) that the answer provides.

    {
      "tenure": {
        "total_years": string or null,                    // e.g., "18"
        "start_year": string or null,                     // e.g., "2008"
        "end_year": string or null,                       // e.g., "2025"
        "sources": [urls...]                               // URLs supporting Iowa tenure and/or years
      },
      "on_field": {
        "on_field_years": string or null,                 // e.g., "14"
        "start_year": string or null,                     // e.g., "2012"
        "end_year": string or null,                       // e.g., "2025"
        "admin_exclusion_years": string or null,          // e.g., "2008-2011"
        "sources": [urls...]                               // URLs supporting on-field timeline and admin exclusion
      },
      "coordinator": {
        "coordinator_years": string or null,              // e.g., "8"
        "start_year": string or null,                     // e.g., "2017"
        "end_year": string or null,                       // e.g., "2025"
        "duties_added_year": string or null,              // e.g., "2017"
        "sources": [urls...]                               // URLs supporting coordinator timeline/years
      },
      "salaries": {
        "iowa_salary_2024": string or null,               // e.g., "$775,000"
        "iowa_salary_sources": [urls...],                  // URLs supporting Iowa 2024 salary
        "msu_salary_2026": string or null,                // e.g., "$1,100,000"
        "msu_salary_sources": [urls...]                    // URLs supporting MSU 2026 salary
      },
      "increase": {
        "salary_increase": string or null,                // e.g., "$325,000"
        "calculation_text": string or null                // e.g., "$1,100,000 - $775,000 = $325,000"
      },
      "requirement": {
        "typical_minimum_years": string or null,          // e.g., "5"
        "met_minimum": string or null,                    // "Yes" or "No"
        "requirement_sources": [urls...]                   // URLs supporting the typical minimum requirement
      }
    }

    IMPORTANT:
    - Extract only URLs explicitly present in the answer text. If no URLs are provided for an item, return an empty array for that item's "sources".
    - Keep all numbers as strings (e.g., "18", "14", "$775,000").
    - If any field is not mentioned, set it to null or an empty array for sources.
    """


# ----------------------------- Helper Utils --------------------------------- #
def safe_list(x: Optional[List[str]]) -> List[str]:
    return x if x else []


def safe_str(x: Optional[str]) -> str:
    return x or ""


# -------------------------- Verification Builders --------------------------- #
async def build_total_years_nodes(evaluator: Evaluator, root, ex: CareerExtraction) -> None:
    parent = evaluator.add_parallel(
        id="total_years_iowa",
        desc="Correctly identify total years LeVar Woods served on Iowa coaching staff (2008-2025)",
        parent=root,
        critical=True
    )

    tenure = ex.tenure or TenureAtIowa()

    # Value leaf: arithmetic/consistency check
    value_node = evaluator.add_leaf(
        id="total_years_value",
        desc="State that LeVar Woods served 18 years at Iowa",
        parent=parent,
        critical=True
    )
    claim_val = (
        f"Counting inclusive years from {safe_str(tenure.start_year)} to {safe_str(tenure.end_year)} "
        f"equals {safe_str(tenure.total_years)} years."
        if tenure.start_year and tenure.end_year and tenure.total_years
        else f"The total years LeVar Woods served on Iowa's coaching staff is {safe_str(tenure.total_years)}."
    )
    await evaluator.verify(
        claim=claim_val,
        node=value_node,
        additional_instruction="Use inclusive counting (e.g., 2008–2009 is 2 years). If any year is missing, judge based on the stated total years."
    )

    # Reference leaf: must be supported by URLs; fail immediately if none
    sources = safe_list(tenure.sources)
    if sources:
        ref_node = evaluator.add_leaf(
            id="total_years_reference",
            desc="Provide valid URL reference supporting the 18-year tenure at Iowa",
            parent=parent,
            critical=True
        )
        claim_ref = (
            f"LeVar Woods served on Iowa's coaching staff from {safe_str(tenure.start_year)} to {safe_str(tenure.end_year)}, "
            f"totaling {safe_str(tenure.total_years)} years."
            if tenure.start_year and tenure.end_year and tenure.total_years
            else "LeVar Woods' total tenure on Iowa's coaching staff was 18 years."
        )
        await evaluator.verify(
            claim=claim_ref,
            node=ref_node,
            sources=sources,
            additional_instruction="Confirm that the cited page(s) explicitly support the tenure years for LeVar Woods at Iowa."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="total_years_reference",
            desc="Provide valid URL reference supporting the 18-year tenure at Iowa",
            parent=parent,
            critical=True
        )


async def build_on_field_nodes(evaluator: Evaluator, root, ex: CareerExtraction) -> None:
    parent = evaluator.add_parallel(
        id="on_field_coaching_years",
        desc="Correctly identify years in on-field coaching positions (excluding administrative roles)",
        parent=root,
        critical=True
    )

    onf = ex.on_field or OnFieldInfo()

    # Value leaf: arithmetic/consistency check
    value_node = evaluator.add_leaf(
        id="on_field_years_value",
        desc="State that LeVar Woods had 14 years of on-field coaching (2012-2025)",
        parent=parent,
        critical=True
    )
    claim_val = (
        f"Counting inclusive years from {safe_str(onf.start_year)} to {safe_str(onf.end_year)} equals {safe_str(onf.on_field_years)} years."
        if onf.start_year and onf.end_year and onf.on_field_years
        else f"The total on-field coaching years at Iowa is {safe_str(onf.on_field_years)}."
    )
    await evaluator.verify(
        claim=claim_val,
        node=value_node,
        additional_instruction="Use inclusive counting. Focus only on on-field coaching years; administrative assistant roles should be excluded."
    )

    # Administrative exclusion leaf
    admin_sources = safe_list(onf.sources)
    if admin_sources:
        admin_node = evaluator.add_leaf(
            id="administrative_exclusion",
            desc="Correctly exclude administrative assistant years (2008-2011) from on-field coaching count",
            parent=parent,
            critical=True
        )
        claim_admin = (
            f"From {safe_str(onf.admin_exclusion_years)}, LeVar Woods served in administrative assistant roles at Iowa, "
            f"which are not counted as on-field coaching."
            if onf.admin_exclusion_years
            else "LeVar Woods' administrative assistant years (2008–2011) were correctly excluded from on-field coaching."
        )
        await evaluator.verify(
            claim=claim_admin,
            node=admin_node,
            sources=admin_sources,
            additional_instruction="Verify the role descriptions indicate administrative assistant positions (non on-field) for 2008–2011."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="administrative_exclusion",
            desc="Correctly exclude administrative assistant years (2008-2011) from on-field coaching count",
            parent=parent,
            critical=True
        )

    # Reference leaf
    ref_sources = safe_list(onf.sources)
    if ref_sources:
        ref_node = evaluator.add_leaf(
            id="on_field_years_reference",
            desc="Provide valid URL reference supporting the on-field coaching timeline",
            parent=parent,
            critical=True
        )
        claim_ref = (
            f"LeVar Woods served in on-field coaching roles at Iowa from {safe_str(onf.start_year)} to {safe_str(onf.end_year)}."
            if onf.start_year and onf.end_year
            else "LeVar Woods' on-field coaching timeline at Iowa totals 14 years."
        )
        await evaluator.verify(
            claim=claim_ref,
            node=ref_node,
            sources=ref_sources,
            additional_instruction="Confirm the pages show the transition from administrative roles to on-field coaching and the specified timeline."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="on_field_years_reference",
            desc="Provide valid URL reference supporting the on-field coaching timeline",
            parent=parent,
            critical=True
        )


async def build_coordinator_nodes(evaluator: Evaluator, root, ex: CareerExtraction) -> None:
    parent = evaluator.add_parallel(
        id="coordinator_years",
        desc="Correctly identify years as special teams coordinator",
        parent=root,
        critical=False  # non-critical to allow mixed strictness with timeline note
    )

    coord = ex.coordinator or CoordinatorInfo()

    # Value leaf
    value_node = evaluator.add_leaf(
        id="coordinator_years_value",
        desc="State that LeVar Woods served as special teams coordinator for 8 years (2017-2025)",
        parent=parent,
        critical=True
    )
    claim_val = f"LeVar Woods served as Iowa's special teams coordinator for {safe_str(coord.coordinator_years)} years."
    await evaluator.verify(
        claim=claim_val,
        node=value_node,
        additional_instruction="Judge based on the answer statement about the number of coordinator years."
    )

    # Timeline note leaf (non-critical)
    timeline_sources = safe_list(coord.sources)
    if timeline_sources:
        timeline_node = evaluator.add_leaf(
            id="coordinator_timeline",
            desc="Note that coordinator duties were added in 2017",
            parent=parent,
            critical=False
        )
        claim_tl = (
            f"LeVar Woods added special teams coordinator duties in {safe_str(coord.duties_added_year)}."
            if coord.duties_added_year
            else "LeVar Woods added special teams coordinator duties in 2017."
        )
        await evaluator.verify(
            claim=claim_tl,
            node=timeline_node,
            sources=timeline_sources,
            additional_instruction="Verify the page(s) explicitly state that he became special teams coordinator in 2017."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="coordinator_timeline",
            desc="Note that coordinator duties were added in 2017",
            parent=parent,
            critical=False
        )

    # Reference leaf (critical)
    ref_sources = safe_list(coord.sources)
    if ref_sources:
        ref_node = evaluator.add_leaf(
            id="coordinator_years_reference",
            desc="Provide valid URL reference supporting coordinator timeline",
            parent=parent,
            critical=True
        )
        claim_ref = (
            f"LeVar Woods served as Iowa's special teams coordinator from {safe_str(coord.start_year)} to {safe_str(coord.end_year)}, "
            f"for {safe_str(coord.coordinator_years)} years."
            if coord.start_year and coord.end_year and coord.coordinator_years
            else "LeVar Woods' special teams coordinator tenure at Iowa is accurately supported by sources."
        )
        await evaluator.verify(
            claim=claim_ref,
            node=ref_node,
            sources=ref_sources,
            additional_instruction="Confirm the pages support both the start year (2017) and tenure length for the special teams coordinator role."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="coordinator_years_reference",
            desc="Provide valid URL reference supporting coordinator timeline",
            parent=parent,
            critical=True
        )


async def build_iowa_salary_nodes(evaluator: Evaluator, root, ex: CareerExtraction) -> None:
    parent = evaluator.add_parallel(
        id="iowa_salary",
        desc="Correctly identify LeVar Woods' annual salary at Iowa in 2024",
        parent=root,
        critical=True
    )

    sal = ex.salaries or SalaryInfo()

    # Value leaf (critical)
    if safe_list(sal.iowa_salary_sources):
        val_node = evaluator.add_leaf(
            id="iowa_salary_value",
            desc="State that his Iowa salary was $775,000",
            parent=parent,
            critical=True
        )
        claim_val = f"LeVar Woods' annual base salary at Iowa for the 2024 season was {safe_str(sal.iowa_salary_2024)}."
        await evaluator.verify(
            claim=claim_val,
            node=val_node,
            sources=sal.iowa_salary_sources,
            additional_instruction="Confirm the 2024 Iowa season base salary (exclude bonuses)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="iowa_salary_value",
            desc="State that his Iowa salary was $775,000",
            parent=parent,
            critical=True
        )

    # Reference leaf (critical)
    if safe_list(sal.iowa_salary_sources):
        ref_node = evaluator.add_leaf(
            id="iowa_salary_reference",
            desc="Provide valid URL reference supporting the $775,000 Iowa salary",
            parent=parent,
            critical=True
        )
        claim_ref = f"The cited source(s) show LeVar Woods' 2024 Iowa salary as {safe_str(sal.iowa_salary_2024)}."
        await evaluator.verify(
            claim=claim_ref,
            node=ref_node,
            sources=sal.iowa_salary_sources,
            additional_instruction="Ensure the page explicitly mentions the 2024 base salary amount."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="iowa_salary_reference",
            desc="Provide valid URL reference supporting the $775,000 Iowa salary",
            parent=parent,
            critical=True
        )


async def build_msu_salary_nodes(evaluator: Evaluator, root, ex: CareerExtraction) -> None:
    parent = evaluator.add_parallel(
        id="msu_salary",
        desc="Correctly identify LeVar Woods' annual salary at Michigan State",
        parent=root,
        critical=True
    )

    sal = ex.salaries or SalaryInfo()

    # Value leaf (critical)
    if safe_list(sal.msu_salary_sources):
        val_node = evaluator.add_leaf(
            id="msu_salary_value",
            desc="State that his Michigan State salary is $1,100,000",
            parent=parent,
            critical=True
        )
        claim_val = f"LeVar Woods' annual salary at Michigan State for the 2026 season is {safe_str(sal.msu_salary_2026)}."
        await evaluator.verify(
            claim=claim_val,
            node=val_node,
            sources=sal.msu_salary_sources,
            additional_instruction="Confirm the annual salary figure for the 2026 season at Michigan State."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="msu_salary_value",
            desc="State that his Michigan State salary is $1,100,000",
            parent=parent,
            critical=True
        )

    # Reference leaf (critical)
    if safe_list(sal.msu_salary_sources):
        ref_node = evaluator.add_leaf(
            id="msu_salary_reference",
            desc="Provide valid URL reference supporting the $1,100,000 Michigan State salary",
            parent=parent,
            critical=True
        )
        claim_ref = f"The cited source(s) show LeVar Woods' 2026 Michigan State salary as {safe_str(sal.msu_salary_2026)}."
        await evaluator.verify(
            claim=claim_ref,
            node=ref_node,
            sources=sal.msu_salary_sources,
            additional_instruction="Ensure the page explicitly mentions the 2026 annual salary amount."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="msu_salary_reference",
            desc="Provide valid URL reference supporting the $1,100,000 Michigan State salary",
            parent=parent,
            critical=True
        )


async def build_salary_increase_nodes(evaluator: Evaluator, root, ex: CareerExtraction) -> None:
    parent = evaluator.add_parallel(
        id="salary_increase",
        desc="Correctly calculate salary increase from Iowa to Michigan State",
        parent=root,
        critical=False  # allow partial credit with calculation step
    )

    inc = ex.increase or IncreaseInfo()
    sal = ex.salaries or SalaryInfo()

    # Value leaf (critical under this parent)
    val_node = evaluator.add_leaf(
        id="salary_increase_value",
        desc="State that the salary increase was $325,000",
        parent=parent,
        critical=True
    )
    claim_val = (
        f"Based on the salaries {safe_str(sal.iowa_salary_2024)} at Iowa (2024) and {safe_str(sal.msu_salary_2026)} at Michigan State (2026), "
        f"the salary increase is {safe_str(inc.salary_increase)}."
        if sal.iowa_salary_2024 and sal.msu_salary_2026 and inc.salary_increase
        else f"The salary increase amount is {safe_str(inc.salary_increase)}."
    )
    await evaluator.verify(
        claim=claim_val,
        node=val_node,
        additional_instruction="Perform exact arithmetic in US dollars; ignore bonuses. If both source salaries are provided, verify the difference matches the stated increase."
    )

    # Calculation leaf (non-critical)
    calc_node = evaluator.add_leaf(
        id="salary_increase_calculation",
        desc="Show calculation: $1,100,000 - $775,000 = $325,000",
        parent=parent,
        critical=False
    )
    claim_calc = safe_str(inc.calculation_text) or "$1,100,000 - $775,000 = $325,000"
    await evaluator.verify(
        claim=claim_calc,
        node=calc_node,
        additional_instruction="Check the arithmetic correctness of the subtraction shown."
    )


async def build_min_requirement_nodes(evaluator: Evaluator, root, ex: CareerExtraction) -> None:
    parent = evaluator.add_parallel(
        id="minimum_qualification_met",
        desc="Correctly determine whether LeVar Woods met typical 5-year minimum for assistant head coach positions",
        parent=root,
        critical=True
    )

    req = ex.requirement or RequirementInfo()
    onf = ex.on_field or OnFieldInfo()

    # Determination leaf (critical)
    det_node = evaluator.add_leaf(
        id="qualification_determination",
        desc="State that LeVar Woods exceeded the typical 5-year minimum with 14 years of on-field coaching",
        parent=parent,
        critical=True
    )
    claim_det = (
        f"With {safe_str(onf.on_field_years)} years of on-field coaching at Iowa, LeVar Woods met or exceeded the typical minimum requirement of "
        f"{safe_str(req.typical_minimum_years)} years for assistant head coach positions."
        if onf.on_field_years and req.typical_minimum_years
        else "LeVar Woods exceeded a typical 5-year minimum with 14 years of on-field coaching."
    )
    await evaluator.verify(
        claim=claim_det,
        node=det_node,
        additional_instruction="Judge based on the on-field years stated in the answer versus the typical minimum requirement."
    )

    # Requirement reference leaf (critical)
    req_sources = safe_list(req.requirement_sources)
    if req_sources:
        ref_node = evaluator.add_leaf(
            id="minimum_requirement_reference",
            desc="Provide valid URL reference supporting the typical 5-year minimum requirement for assistant head coach positions",
            parent=parent,
            critical=True
        )
        claim_ref = (
            f"Assistant head coach positions at major college football programs typically require at least "
            f"{safe_str(req.typical_minimum_years)} years of collegiate coaching experience."
            if req.typical_minimum_years
            else "Assistant head coach positions at major college football programs typically require a minimum of 5 years of collegiate coaching experience."
        )
        await evaluator.verify(
            claim=claim_ref,
            node=ref_node,
            sources=req_sources,
            additional_instruction="Verify that the cited source(s) explicitly state a typical minimum (around 5 years) for assistant head coach roles at major programs."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="minimum_requirement_reference",
            desc="Provide valid URL reference supporting the typical 5-year minimum requirement for assistant head coach positions",
            parent=parent,
            critical=True
        )


# --------------------------- Main Evaluation -------------------------------- #
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
    """
    Evaluate the agent's answer for LeVar Woods' career progression and return a structured summary.
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

    # Extract structured info from the answer
    extracted: CareerExtraction = await evaluator.extract(
        prompt=prompt_extract_levar_woods(),
        template_class=CareerExtraction,
        extraction_name="career_progression_extraction"
    )

    # Build verification subtrees
    await build_total_years_nodes(evaluator, root, extracted)
    await build_on_field_nodes(evaluator, root, extracted)
    await build_coordinator_nodes(evaluator, root, extracted)
    await build_iowa_salary_nodes(evaluator, root, extracted)
    await build_msu_salary_nodes(evaluator, root, extracted)
    await build_salary_increase_nodes(evaluator, root, extracted)
    await build_min_requirement_nodes(evaluator, root, extracted)

    # Return final structured evaluation summary
    return evaluator.get_summary()