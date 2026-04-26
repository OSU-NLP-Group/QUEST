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
TASK_ID = "cmu_ri_info"
TASK_DESCRIPTION = """
The Robotics Institute at Carnegie Mellon University is one of the world's leading centers for robotics research and education. A prospective postdoctoral researcher is gathering information about the institute before applying. Provide the following information: (1) The exact month and year the Robotics Institute was founded, (2) The name of the professor who served as the institute's director from January 2022 through April 2025, (3) The institute's approximate annual research funding budget as of 2024-2025, (4) The minimum annual salary requirement for postdoctoral positions based on NIH guidelines effective in fiscal year 2024-2025, (5) The minimum degree requirement typically required for postdoctoral research positions at university research institutes, (6) Whether doctoral students at research universities are typically required to hold regular thesis committee meetings (and if so, the minimum frequency), and (7) At least two major federal funding agencies that commonly support robotics research at U.S. universities.
"""

# Expected values (used for claims and ground truth context)
EXPECTED_FOUNDING_MONTH_YEAR = "October 1979"
EXPECTED_DIRECTOR_NAME = "Matthew Johnson-Roberson"
EXPECTED_BUDGET_APPROX = "approximately $73 million"
EXPECTED_POSTDOC_MIN_SALARY = "$61,008 per year"
EXPECTED_COMMITTEE_MIN_FREQ = "at least once per year"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RIExtraction(BaseModel):
    institute_name: Optional[str] = None
    institute_url: Optional[str] = None  # Prefer official RI/CMU URL

    founding_month_year: Optional[str] = None
    founding_sources: List[str] = Field(default_factory=list)

    director_name: Optional[str] = None
    director_term_text: Optional[str] = None  # e.g., "January 2022 – April 2025"
    director_sources: List[str] = Field(default_factory=list)

    annual_research_budget: Optional[str] = None  # e.g., "$73 million", "approx. $73M"
    budget_sources: List[str] = Field(default_factory=list)

    funding_agencies: List[str] = Field(default_factory=list)  # e.g., ["NSF", "DARPA", ...]
    funding_agencies_sources: List[str] = Field(default_factory=list)

    minimum_postdoc_salary: Optional[str] = None  # e.g., "$61,008 per year"
    salary_sources: List[str] = Field(default_factory=list)

    degree_requirement: Optional[str] = None  # e.g., "Doctoral degree (PhD, MD, or equivalent)"
    degree_sources: List[str] = Field(default_factory=list)

    committee_meeting_requirement: Optional[str] = None  # e.g., "Yes, required"
    committee_meeting_frequency: Optional[str] = None  # e.g., "at least once per year"
    committee_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ri_info() -> str:
    return """
Extract the following information exactly as stated in the answer text. Do NOT invent or infer anything; if an item is not explicitly mentioned, set it to null (for strings) or an empty list (for arrays). For URLs, extract only valid URLs explicitly present in the answer (plain or markdown), and include the protocol. If an item mentions multiple sources, include all of them.

Return a single JSON object with these keys:
- institute_name: The institute’s name as written in the answer (e.g., "Robotics Institute, Carnegie Mellon University" or "CMU Robotics Institute").
- institute_url: A single official institute page URL if provided (prefer ri.cmu.edu or cmu.edu URLs); if multiple are present, pick the most central official page; else null.

- founding_month_year: The founding month and year (e.g., "October 1979"); else null.
- founding_sources: List of URLs cited that support the founding month/year; empty if none.

- director_name: The director's name as stated for the period in question; else null.
- director_term_text: The director’s service period text if stated (e.g., "January 2022 through April 2025"); else null.
- director_sources: List of URLs cited that support the director and/or term; empty if none.

- annual_research_budget: The approximate annual research funding budget as stated (e.g., "approximately $73 million"); else null.
- budget_sources: List of URLs cited that support the budget figure; empty if none.

- funding_agencies: List of major U.S. federal funding agencies listed (e.g., "NSF", "DARPA", "ONR", "DOE", "NIH"); empty if none.
- funding_agencies_sources: List of URLs cited that support that the listed agencies fund robotics research (generally or at CMU); empty if none.

- minimum_postdoc_salary: The NIH-based entry-level minimum annual salary for postdocs for FY 2024–2025 as stated (e.g., "$61,008 per year"); else null.
- salary_sources: List of URLs cited that support the salary figure (prefer NIH official stipend guidance or an official university policy tying to NIH); empty if none.

- degree_requirement: The minimum degree requirement for postdoctoral positions as stated (e.g., "Doctoral degree (PhD, MD, or equivalent)"); else null.
- degree_sources: List of URLs cited that support the degree requirement (official university policy/postdoctoral affairs page or reputable institutional source); empty if none.

- committee_meeting_requirement: Whether doctoral students are typically required to hold thesis committee meetings (e.g., "Yes, typically required"); else null.
- committee_meeting_frequency: The minimum frequency stated (e.g., "at least once per year"); else null.
- committee_sources: List of URLs cited that support this committee meeting requirement/frequency from graduate school policy/department handbook or similar; empty if none.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# ----------------------- Institute Identification ------------------------- #
async def verify_institute_identification(evaluator: Evaluator, parent_node, info: RIExtraction) -> None:
    node = evaluator.add_parallel(
        id="Institute_Identification",
        desc="Correctly identify the subject institute.",
        parent=parent_node,
        critical=True
    )

    # Existence check (name and a supporting official URL)
    exists = evaluator.add_custom_node(
        result=_nonempty(info.institute_name) and _nonempty(info.institute_url),
        id="Institute_Identification_Provided",
        desc="Institute name and official URL provided",
        parent=node,
        critical=True
    )

    # Leaf: Institute_Name (simple verification against the answer)
    name_leaf = evaluator.add_leaf(
        id="Institute_Name",
        desc="The institute is identified as the Carnegie Mellon University Robotics Institute (RI).",
        parent=node,
        critical=True
    )
    claim_name = "The institute identified in the answer is the Robotics Institute at Carnegie Mellon University (CMU)."
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        additional_instruction="Allow minor naming variations such as 'CMU Robotics Institute' or 'Robotics Institute (CMU)'. Focus on confirming CMU's Robotics Institute."
    )

    # Leaf: Institute_Name_Reference (verification by official URL)
    ref_leaf = evaluator.add_leaf(
        id="Institute_Name_Reference",
        desc="Provides a supporting URL from an official CMU/RI webpage confirming the institute identity.",
        parent=node,
        critical=True
    )
    claim_ref = "This webpage confirms the institute identity as the Robotics Institute at Carnegie Mellon University."
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=info.institute_url,
        additional_instruction="Prefer ri.cmu.edu or cmu.edu pages. The page should clearly indicate 'Robotics Institute' associated with Carnegie Mellon University."
    )


# --------------------------- Founding Information ------------------------- #
async def verify_founding_information(evaluator: Evaluator, parent_node, info: RIExtraction) -> None:
    node = evaluator.add_parallel(
        id="Founding_Information",
        desc="Provide the institute founding month and year.",
        parent=parent_node,
        critical=True
    )

    # Existence check for founding info and sources
    exists = evaluator.add_custom_node(
        result=_nonempty(info.founding_month_year) and bool(info.founding_sources),
        id="Founding_Info_Provided",
        desc="Founding month/year and at least one supporting source provided",
        parent=node,
        critical=True
    )

    # Leaf: Founding_Month_Year (simple verification against the answer)
    month_year_leaf = evaluator.add_leaf(
        id="Founding_Month_Year",
        desc=f"States the founding month and year as {EXPECTED_FOUNDING_MONTH_YEAR}.",
        parent=node,
        critical=True
    )
    claim_month_year = f"The Robotics Institute was founded in {EXPECTED_FOUNDING_MONTH_YEAR}."
    await evaluator.verify(
        claim=claim_month_year,
        node=month_year_leaf,
        additional_instruction="Confirm the answer text explicitly states the founding month and year."
    )

    # Leaf: Founding_Date_Reference (verification by URLs)
    ref_leaf = evaluator.add_leaf(
        id="Founding_Date_Reference",
        desc=f"Provides a supporting URL from an official CMU/RI source confirming the {EXPECTED_FOUNDING_MONTH_YEAR} founding date.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_month_year,
        node=ref_leaf,
        sources=info.founding_sources,
        additional_instruction="Look for explicit language on official CMU/RI pages or reputable institutional histories confirming the month and year."
    )


# --------------------------- Director Information ------------------------- #
async def verify_director_information(evaluator: Evaluator, parent_node, info: RIExtraction) -> None:
    node = evaluator.add_parallel(
        id="Director_Information",
        desc="Identify the director and verify the required term dates.",
        parent=parent_node,
        critical=True
    )

    # Existence check for director info and sources
    exists = evaluator.add_custom_node(
        result=_nonempty(info.director_name) and bool(info.director_sources),
        id="Director_Info_Provided",
        desc="Director name and at least one supporting source provided",
        parent=node,
        critical=True
    )

    # Leaf: Director_Name (simple verification against the answer)
    name_leaf = evaluator.add_leaf(
        id="Director_Name",
        desc=f"Identifies the director as {EXPECTED_DIRECTOR_NAME}.",
        parent=node,
        critical=True
    )
    claim_name = f"The director is {EXPECTED_DIRECTOR_NAME}."
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        additional_instruction="Allow minor variations in punctuation or hyphenation of the name."
    )

    # Leaf: Director_Service_Period (simple verification against the answer)
    period_leaf = evaluator.add_leaf(
        id="Director_Service_Period",
        desc="Explicitly states that the directorship term covers January 2022 through April 2025.",
        parent=node,
        critical=True
    )
    claim_period = "The directorship term covers January 2022 through April 2025."
    await evaluator.verify(
        claim=claim_period,
        node=period_leaf,
        additional_instruction="Confirm that the answer text clearly indicates coverage from Jan 2022 through Apr 2025."
    )

    # Leaf: Director_Reference (verification by URLs)
    ref_leaf = evaluator.add_leaf(
        id="Director_Reference",
        desc="Provides a supporting URL from an official CMU/RI webpage (or comparably reputable institutional record) confirming the director and/or the specified term.",
        parent=node,
        critical=True
    )
    claim_ref = f"{EXPECTED_DIRECTOR_NAME} served as director of the CMU Robotics Institute from January 2022 through April 2025."
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=info.director_sources,
        additional_instruction="Prefer official RI/CMU pages, announcements, or institutional profiles with explicit dates."
    )


# --------------------------- Funding Information -------------------------- #
async def verify_funding_information(evaluator: Evaluator, parent_node, info: RIExtraction) -> None:
    node = evaluator.add_parallel(
        id="Funding_Information",
        desc="Provide the institute’s approximate annual research funding budget and identify common federal funders.",
        parent=parent_node,
        critical=True
    )

    # Existence for budget + sources
    budget_exists = evaluator.add_custom_node(
        result=_nonempty(info.annual_research_budget) and bool(info.budget_sources),
        id="Budget_Info_Provided",
        desc="Annual research budget figure and at least one supporting source provided",
        parent=node,
        critical=True
    )

    # Leaf: Annual_Research_Budget (simple verification against the answer)
    budget_leaf = evaluator.add_leaf(
        id="Annual_Research_Budget",
        desc=f"States the institute’s approximate annual research funding budget as {EXPECTED_BUDGET_APPROX} (as of 2024–2025 context).",
        parent=node,
        critical=True
    )
    claim_budget = f"The institute’s annual research funding budget is {EXPECTED_BUDGET_APPROX} (as of 2024–2025)."
    await evaluator.verify(
        claim=claim_budget,
        node=budget_leaf,
        additional_instruction="Accept reasonable phrasing like '~$73M', 'approximately $73M', or similar."
    )

    # Leaf: Budget_Reference (verification by URLs)
    budget_ref_leaf = evaluator.add_leaf(
        id="Budget_Reference",
        desc=f"Provides a supporting URL from an official CMU/RI source or reputable academic/institutional source confirming the {EXPECTED_BUDGET_APPROX} figure.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_budget,
        node=budget_ref_leaf,
        sources=info.budget_sources,
        additional_instruction="Look for official institute reports, fact sheets, or reputable institutional sources with the budget figure."
    )

    # Existence check for agencies listed and at least one supporting URL
    agencies_exist = evaluator.add_custom_node(
        result=(len(info.funding_agencies) >= 2) and bool(info.funding_agencies_sources),
        id="Funding_Agencies_Provided",
        desc="At least two federal funding agencies listed and at least one supporting source provided",
        parent=node,
        critical=True
    )

    # Leaf: Federal_Funding_Agencies (satisfy listing requirement)
    agencies_leaf = evaluator.add_custom_node(
        result=(len(info.funding_agencies) >= 2),
        id="Federal_Funding_Agencies",
        desc="Lists at least two major U.S. federal funding agencies that commonly support robotics research at U.S. universities (e.g., NSF, DARPA, ONR, DOE, NIH).",
        parent=node,
        critical=True
    )

    # Leaf: Funding_Agencies_Reference (verification by URLs)
    agencies_ref_leaf = evaluator.add_leaf(
        id="Funding_Agencies_Reference",
        desc="Provides at least one supporting URL from a government agency site or reputable academic/institutional source evidencing that the listed agencies fund robotics research (generally or at CMU).",
        parent=node,
        critical=True
    )
    agencies_list_str = ", ".join(info.funding_agencies) if info.funding_agencies else "the listed agencies"
    claim_agencies_ref = f"The agencies ({agencies_list_str}) are major U.S. federal funders of robotics research at universities."
    await evaluator.verify(
        claim=claim_agencies_ref,
        node=agencies_ref_leaf,
        sources=info.funding_agencies_sources,
        additional_instruction="Government agency pages or reputable academic sources should show programs/funding supporting robotics research."
    )


# ---------------------- Postdoctoral Position Requirements ---------------- #
async def verify_postdoc_requirements(evaluator: Evaluator, parent_node, info: RIExtraction) -> None:
    node = evaluator.add_parallel(
        id="Postdoctoral_Position_Requirements",
        desc="Provide required postdoctoral salary minimum and degree requirement with appropriate citations.",
        parent=parent_node,
        critical=True
    )

    # Salary existence check
    salary_exists = evaluator.add_custom_node(
        result=_nonempty(info.minimum_postdoc_salary) and bool(info.salary_sources),
        id="Postdoc_Salary_Provided",
        desc="Minimum postdoc salary and at least one supporting source provided",
        parent=node,
        critical=True
    )

    # Leaf: Minimum_Postdoc_Salary (simple verification against the answer)
    salary_leaf = evaluator.add_leaf(
        id="Minimum_Postdoc_Salary",
        desc=f"States the NIH FY2024–2025 entry-level postdoctoral minimum as {EXPECTED_POSTDOC_MIN_SALARY} (as the minimum requirement per the constraints).",
        parent=node,
        critical=True
    )
    claim_salary = f"The NIH FY2024–2025 entry-level postdoctoral minimum is {EXPECTED_POSTDOC_MIN_SALARY}."
    await evaluator.verify(
        claim=claim_salary,
        node=salary_leaf,
        additional_instruction="Accept reasonable numeric formatting variations (e.g., commas, currency symbols)."
    )

    # Leaf: Salary_Standard_Reference (verification by URLs)
    salary_ref_leaf = evaluator.add_leaf(
        id="Salary_Standard_Reference",
        desc="Provides a supporting URL from official NIH stipend/salary guidance or an official university postdoc policy explicitly tying to NIH FY2024–2025 standards.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_salary,
        node=salary_ref_leaf,
        sources=info.salary_sources,
        additional_instruction="Prefer nih.gov pages for stipend guidance or official university policy pages referencing NIH FY2024."
    )

    # Degree existence check
    degree_exists = evaluator.add_custom_node(
        result=_nonempty(info.degree_requirement) and bool(info.degree_sources),
        id="Postdoc_Degree_Provided",
        desc="Degree requirement and at least one supporting source provided",
        parent=node,
        critical=True
    )

    # Leaf: Degree_Requirement (simple verification against the answer)
    degree_leaf = evaluator.add_leaf(
        id="Degree_Requirement",
        desc="States that postdoctoral positions require a completed doctoral degree (PhD, MD, or equivalent terminal degree).",
        parent=node,
        critical=True
    )
    claim_degree = "Postdoctoral positions require a completed doctoral degree (PhD, MD, or equivalent terminal degree)."
    await evaluator.verify(
        claim=claim_degree,
        node=degree_leaf,
        additional_instruction="Allow reasonable phrasing variants indicating a terminal doctoral degree requirement."
    )

    # Leaf: Degree_Requirement_Reference (verification by URLs)
    degree_ref_leaf = evaluator.add_leaf(
        id="Degree_Requirement_Reference",
        desc="Provides a supporting URL from an official university policy/postdoctoral affairs page or reputable academic/institutional source confirming the doctoral-degree requirement.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_degree,
        node=degree_ref_leaf,
        sources=info.degree_sources,
        additional_instruction="Prefer official university policy pages or reputable academic sources stating the doctoral degree requirement."
    )


# -------------------- Committee Meeting Requirements ---------------------- #
async def verify_committee_requirements(evaluator: Evaluator, parent_node, info: RIExtraction) -> None:
    node = evaluator.add_parallel(
        id="Committee_Meeting_Requirements",
        desc="Address typical doctoral thesis committee meeting requirements and frequency.",
        parent=parent_node,
        critical=True
    )

    # Existence check for committee requirement/frequency and sources
    committee_exists = evaluator.add_custom_node(
        result=_nonempty(info.committee_meeting_requirement) and _nonempty(info.committee_meeting_frequency) and bool(info.committee_sources),
        id="Committee_Requirement_Provided",
        desc="Committee meeting requirement/frequency and at least one supporting source provided",
        parent=node,
        critical=True
    )

    # Leaf: Meeting_Requirement_And_Frequency (simple verification against the answer)
    meeting_leaf = evaluator.add_leaf(
        id="Meeting_Requirement_And_Frequency",
        desc="States that doctoral students typically are required to hold thesis committee meetings, with a minimum frequency of at least once per year.",
        parent=node,
        critical=True
    )
    claim_meeting = "Doctoral students are typically required to hold thesis committee meetings at least once per year."
    await evaluator.verify(
        claim=claim_meeting,
        node=meeting_leaf,
        additional_instruction="Confirm the answer text specifies an annual (or at least once per year) minimum frequency requirement."
    )

    # Leaf: Meeting_Requirement_Reference (verification by URLs)
    meeting_ref_leaf = evaluator.add_leaf(
        id="Meeting_Requirement_Reference",
        desc="Provides a supporting URL from a graduate school policy, department handbook, or comparable official academic policy source confirming the annual (or at least once-per-year) committee meeting requirement.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_meeting,
        node=meeting_ref_leaf,
        sources=info.committee_sources,
        additional_instruction="Prefer graduate school or department policy/handbook pages explicitly stating annual committee meetings."
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
    Evaluate the provided answer for CMU Robotics Institute information task.
    Builds a hierarchical verification tree and returns the evaluation summary.
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

    # Add ground truth expectations for context in summary
    evaluator.add_ground_truth({
        "expected_founding_month_year": EXPECTED_FOUNDING_MONTH_YEAR,
        "expected_director_name": EXPECTED_DIRECTOR_NAME,
        "expected_annual_research_budget": EXPECTED_BUDGET_APPROX,
        "expected_postdoc_min_salary": EXPECTED_POSTDOC_MIN_SALARY,
        "expected_committee_min_frequency": EXPECTED_COMMITTEE_MIN_FREQ
    }, gt_type="expected_values")

    # Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_ri_info(),
        template_class=RIExtraction,
        extraction_name="ri_extraction"
    )

    # Create the main critical root node under evaluator.root to comply with critical consistency
    main_root = evaluator.add_parallel(
        id="Root_Research_Institute_Information",
        desc="Provide and verify the required information about the Carnegie Mellon University Robotics Institute per the question and constraints.",
        parent=root,
        critical=True
    )

    # Build subtrees per rubric
    await verify_institute_identification(evaluator, main_root, extracted_info)
    await verify_founding_information(evaluator, main_root, extracted_info)
    await verify_director_information(evaluator, main_root, extracted_info)
    await verify_funding_information(evaluator, main_root, extracted_info)
    await verify_postdoc_requirements(evaluator, main_root, extracted_info)
    await verify_committee_requirements(evaluator, main_root, extracted_info)

    # Return structured evaluation summary
    return evaluator.get_summary()