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
TASK_ID = "texas_dual_credit_district"
TASK_DESCRIPTION = (
    "Identify a school district in Texas that meets all of the following criteria:\n\n"
    "1. The district's total student enrollment exceeded 70,000 students as of the 2024-25 school year\n"
    "2. The district ranks among the top 10 largest school districts in Texas by enrollment\n"
    "3. The district operates an active dual credit partnership program that complies with Texas Higher Education Coordinating Board regulations\n"
    "4. During the 2023-24 academic year, students from this district enrolled in at least 6,000 dual credit courses\n"
    "5. During the 2023-24 academic year, students from this district earned a combined total of at least 20,000 college credit hours through dual credit courses\n\n"
    "Provide the name of the school district and reference URLs documenting: (1) the district's enrollment figures, "
    "(2) the dual credit partnership program details, and (3) the dual credit program performance statistics for 2023-24."
)

# --------------------------------------------------------------------------- #
# Extraction models                                                           
# --------------------------------------------------------------------------- #
class DistrictExtraction(BaseModel):
    district_name: Optional[str] = None

    # Required reference URLs
    enrollment_urls: List[str] = Field(default_factory=list)
    program_urls: List[str] = Field(default_factory=list)
    performance_urls: List[str] = Field(default_factory=list)

    # Optional reference URLs for ranking (if provided)
    ranking_urls: List[str] = Field(default_factory=list)

    # Optional textual figures mentioned in the answer
    enrollment_figure_2024_25: Optional[str] = None
    ranking_position_text: Optional[str] = None
    courses_enrolled_2023_24: Optional[str] = None
    credit_hours_2023_24: Optional[str] = None

    # Optional partner institution names and compliance indicator text
    partner_institutions: List[str] = Field(default_factory=list)
    compliance_evidence_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           
# --------------------------------------------------------------------------- #
def prompt_extract_district_info() -> str:
    return (
        "From the answer, extract structured information for the single Texas school district identified. "
        "Return the following fields:\n"
        "1) district_name: The exact name of the chosen school district.\n"
        "2) enrollment_urls: A list of URLs that document the district's enrollment figures (prefer official sources or statewide reports).\n"
        "3) program_urls: A list of URLs that document the district's dual credit partnership program details (prefer official district pages or partner college pages).\n"
        "4) performance_urls: A list of URLs that document the dual credit program performance statistics for the 2023-24 academic year.\n"
        "5) ranking_urls: If the answer cites a source showing Texas enrollment ranking (e.g., top 10 largest districts), include those URLs; otherwise return an empty list.\n"
        "6) enrollment_figure_2024_25: If a specific 2024-25 enrollment figure (or phrasing like 'over 70,000') is mentioned in the answer, extract it as text; else null.\n"
        "7) ranking_position_text: If a ranking position or phrasing like 'top 10' is mentioned, extract it as text; else null.\n"
        "8) courses_enrolled_2023_24: If a number or textual figure for dual credit course enrollments in 2023-24 is mentioned, extract it as text; else null.\n"
        "9) credit_hours_2023_24: If a number or textual figure for total dual credit college credit hours earned in 2023-24 is mentioned, extract it as text; else null.\n"
        "10) partner_institutions: Extract the names of any colleges/universities the district lists as dual credit partners.\n"
        "11) compliance_evidence_text: If the answer states compliance with THECB regulations (e.g., mentions 'Texas Higher Education Coordinating Board', 'THECB', or '19 TAC § 4.84'), extract that text; else null.\n\n"
        "Rules:\n"
        "- If multiple districts are mentioned, select the first one and extract information for it only.\n"
        "- Extract only URLs explicitly present in the answer. Use full URLs; if protocol is missing, prepend http://.\n"
        "- If a field is missing in the answer, return null (or an empty list for list fields).\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            
# --------------------------------------------------------------------------- #
def _unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result = []
    for lst in url_lists:
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification logic                                                          
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extraction: DistrictExtraction) -> None:
    # Create the critical top-level node under root
    top = evaluator.add_parallel(
        id="Texas_School_District_Identification",
        desc="Identify a Texas school district that meets all specified dual credit program criteria for the 2023-24 and 2024-25 academic years",
        parent=evaluator.root,
        critical=True
    )

    # District name must be provided (critical)
    name_provided = evaluator.add_custom_node(
        result=bool(extraction.district_name and extraction.district_name.strip()),
        id="District_Name_Provided",
        desc="The answer provides the name of the school district",
        parent=top,
        critical=True
    )

    district_name = extraction.district_name or ""

    # Geographic location check (critical)
    geo_node = evaluator.add_leaf(
        id="Geographic_Location",
        desc="The identified school district is located in the state of Texas",
        parent=top,
        critical=True
    )
    geo_sources = _unique_urls(extraction.enrollment_urls, extraction.program_urls, extraction.performance_urls, extraction.ranking_urls)
    geo_claim = f"The school district named '{district_name}' is located in Texas."
    await evaluator.verify(
        claim=geo_claim,
        node=geo_node,
        sources=geo_sources,
        additional_instruction="Confirm that the district is a Texas (TX) school district. Accept explicit mentions of Texas or pages that clearly pertain to Texas school districts."
    )

    # Dual Credit Partnership Verification (critical group)
    dcp_group = evaluator.add_parallel(
        id="Dual_Credit_Partnership_Verification",
        desc="The school district has an active dual credit partnership program complying with Texas regulations",
        parent=top,
        critical=True
    )

    # Partnership reference existence (critical)
    partnership_ref = evaluator.add_custom_node(
        result=len(extraction.program_urls) > 0,
        id="Partnership_Reference",
        desc="Provide a reference URL documenting the dual credit partnership or program details",
        parent=dcp_group,
        critical=True
    )

    # Partnership existence (critical)
    partners_text = ", ".join(extraction.partner_institutions) if extraction.partner_institutions else "at least one partner institution"
    partnership_exists_leaf = evaluator.add_leaf(
        id="Partnership_Existence",
        desc="The school district has an active dual credit partnership with at least one college or university",
        parent=dcp_group,
        critical=True
    )
    partnership_claim = (
        f"The school district '{district_name}' has an active dual credit partnership with at least one college or university. "
        f"Partner institutions include: {partners_text}."
    )
    await evaluator.verify(
        claim=partnership_claim,
        node=partnership_exists_leaf,
        sources=extraction.program_urls,
        additional_instruction="Look for explicit descriptions of dual credit partnerships, MOUs, or active program pages listing college partners."
    )

    # Texas regulatory compliance (critical)
    compliance_leaf = evaluator.add_leaf(
        id="Texas_Regulatory_Compliance",
        desc="The dual credit program complies with Texas Higher Education Coordinating Board regulations (19 TAC § 4.84)",
        parent=dcp_group,
        critical=True
    )
    compliance_claim = (
        f"The dual credit program of '{district_name}' complies with Texas Higher Education Coordinating Board regulations (19 TAC § 4.84)."
    )
    await evaluator.verify(
        claim=compliance_claim,
        node=compliance_leaf,
        sources=extraction.program_urls,
        additional_instruction="Search for mentions of 'Texas Higher Education Coordinating Board', 'THECB', 'Texas Administrative Code', or '19 TAC § 4.84' indicating compliance."
    )

    # District Enrollment Metrics (critical group)
    enroll_group = evaluator.add_parallel(
        id="District_Enrollment_Metrics",
        desc="The school district meets specified enrollment size criteria for the 2024-25 school year",
        parent=top,
        critical=True
    )

    # Enrollment reference existence (critical)
    enrollment_ref = evaluator.add_custom_node(
        result=len(extraction.enrollment_urls) > 0,
        id="Enrollment_Reference",
        desc="Provide a reference URL documenting the district's enrollment figures",
        parent=enroll_group,
        critical=True
    )

    # Total enrollment threshold (critical)
    enroll_leaf = evaluator.add_leaf(
        id="Total_Enrollment_Threshold",
        desc="The school district's total student enrollment exceeds 70,000 students as of the 2024-25 school year",
        parent=enroll_group,
        critical=True
    )
    fig_text = extraction.enrollment_figure_2024_25 or "a value above 70,000"
    enroll_claim = (
        f"As of the 2024-25 school year, {district_name}'s total student enrollment exceeded 70,000 students. "
        f"The page indicates {fig_text}."
    )
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_leaf,
        sources=extraction.enrollment_urls,
        additional_instruction="Verify the 2024-25 enrollment figure. Accept phrasing like 'over 70,000', 'more than 70,000', or an explicit number above 70,000."
    )

    # State ranking position (critical)
    ranking_leaf = evaluator.add_leaf(
        id="State_Ranking_Position",
        desc="The school district ranks among the top 10 largest school districts in Texas by enrollment",
        parent=enroll_group,
        critical=True
    )
    ranking_sources = extraction.ranking_urls if extraction.ranking_urls else extraction.enrollment_urls
    rank_text = extraction.ranking_position_text or "top 10"
    ranking_claim = (
        f"By enrollment, {district_name} ranks among the top 10 largest Texas school districts. The answer indicates '{rank_text}'."
    )
    await evaluator.verify(
        claim=ranking_claim,
        node=ranking_leaf,
        sources=ranking_sources,
        additional_instruction="Confirm that the district is within the top 10 in Texas by enrollment; check statewide ranking pages or explicit statements."
    )

    # Dual Credit Program Performance (critical group)
    perf_group = evaluator.add_parallel(
        id="Dual_Credit_Program_Performance",
        desc="The school district's dual credit program meets specified performance metrics for the 2023-24 academic year",
        parent=top,
        critical=True
    )

    # Performance reference existence (critical)
    perf_ref = evaluator.add_custom_node(
        result=len(extraction.performance_urls) > 0,
        id="Program_Performance_Reference",
        desc="Provide a reference URL documenting the dual credit program enrollment and credit hours data",
        parent=perf_group,
        critical=True
    )

    # Course enrollment volume (critical)
    courses_leaf = evaluator.add_leaf(
        id="Course_Enrollment_Volume",
        desc="Students from this district enrolled in at least 6,000 dual credit courses during the 2023-24 academic year",
        parent=perf_group,
        critical=True
    )
    courses_text = extraction.courses_enrolled_2023_24 or ">= 6,000 enrollments"
    courses_claim = (
        f"During the 2023-24 academic year, students from {district_name} enrolled in at least 6,000 dual credit courses. "
        f"The page indicates {courses_text}."
    )
    await evaluator.verify(
        claim=courses_claim,
        node=courses_leaf,
        sources=extraction.performance_urls,
        additional_instruction="Look for 'dual credit course enrollments', 'enrollments', or 'sections' counts for 2023-24 that are 6,000 or higher."
    )

    # Credit hours achievement (critical)
    hours_leaf = evaluator.add_leaf(
        id="Credit_Hours_Achievement",
        desc="Students from this district earned at least 20,000 combined college credit hours through dual credit courses in 2023-24",
        parent=perf_group,
        critical=True
    )
    hours_text = extraction.credit_hours_2023_24 or ">= 20,000 credit hours"
    hours_claim = (
        f"During the 2023-24 academic year, students from {district_name} earned a combined total of at least 20,000 college credit hours through dual credit courses. "
        f"The page indicates {hours_text}."
    )
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=extraction.performance_urls,
        additional_instruction="Verify that the total dual credit college credit hours earned in 2023-24 are at least 20,000."
    )


# --------------------------------------------------------------------------- #
# Main entry point                                                            
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

    # Extract district info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_district_info(),
        template_class=DistrictExtraction,
        extraction_name="district_extraction"
    )

    # Ground truth constraints summary (for context in output)
    evaluator.add_ground_truth({
        "constraints": {
            "location": "Texas",
            "enrollment_threshold_2024_25": "> 70,000 students",
            "ranking": "Top 10 largest Texas districts by enrollment",
            "dual_credit_courses_2023_24": ">= 6,000",
            "dual_credit_credit_hours_2023_24": ">= 20,000",
            "compliance": "THECB 19 TAC § 4.84"
        }
    }, gt_type="constraints_summary")

    # Build and verify the rubric tree
    await build_and_verify_tree(evaluator, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()