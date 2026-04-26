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
TASK_ID = "itmc_public_university_verification"
TASK_DESCRIPTION = """
Identify a public university that meets ALL of the following criteria:

1. The university is located in a U.S. state that is a member of the Interstate Teacher Mobility Compact (ITMC).

2. The state's public university system is governed by a state-level Board of Governors consisting of exactly 17 members.

3. Each university in the state system has its own Board of Trustees consisting of exactly 13 members, with 6 members appointed by the state Governor and 5 members appointed by the state Board of Governors.

4. The university is accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC).

5. The university competes in NCAA Division I athletics.

For your answer, provide:
- The name of the state
- The name of one public university that meets all criteria
- URL references for:
  - The state's ITMC membership status
  - The state's university system governance structure (Board of Governors composition)
  - The university's Board of Trustees composition
  - The university's SACSCOC accreditation status
  - The university's NCAA Division I athletic status

Additionally, document the NCAA Division I academic eligibility standards that student-athletes at this university must meet to compete, including:
- The total number of core courses required
- The minimum core-course GPA
- The progressive completion requirement (10/7 rule)
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SubmissionExtraction(BaseModel):
    # Required names
    state: Optional[str] = None
    university: Optional[str] = None

    # Evidence URL groups (lists to allow multiple references)
    itmc_urls: List[str] = Field(default_factory=list)
    bog_governance_urls: List[str] = Field(default_factory=list)
    systemwide_bot_rule_urls: List[str] = Field(default_factory=list)
    university_bot_urls: List[str] = Field(default_factory=list)
    sacscoc_urls: List[str] = Field(default_factory=list)
    ncaa_division1_urls: List[str] = Field(default_factory=list)

    # NCAA DI eligibility statements mentioned in the answer (textual)
    di_core_course_total: Optional[str] = None
    di_distribution_text: Optional[str] = None
    di_min_core_gpa: Optional[str] = None
    di_progressive_rule_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_submission() -> str:
    return """
    Extract the following information exactly as presented in the answer.

    1) Names:
       - state: The U.S. state explicitly identified.
       - university: The single public university explicitly identified.

    2) Evidence URLs:
       Extract all URLs provided for each category. Only include valid, explicit URLs from the answer text.
       - itmc_urls: URLs that substantiate the state's membership in the Interstate Teacher Mobility Compact (ITMC).
       - bog_governance_urls: URLs that explain the state-level Board of Governors governing the state public university system and/or its composition.
       - systemwide_bot_rule_urls: URLs that describe the system-wide rule for each university's Board of Trustees (e.g., membership count and appointment sources).
       - university_bot_urls: URLs that describe the selected university's specific Board of Trustees composition/appointments.
       - sacscoc_urls: URLs that substantiate the university's SACSCOC accreditation status (ideally SACSCOC or institutional accreditation page).
       - ncaa_division1_urls: URLs that substantiate the university's NCAA Division I athletic status.

    3) NCAA Division I eligibility statements (as stated in the answer; do not infer):
       - di_core_course_total: The total number of core courses required (e.g., "16").
       - di_distribution_text: The detailed breakdown of the 16 core courses (e.g., "4 English; 3 math (Algebra 1 or higher); 2 science (including 1 lab if offered); 1 additional year English/math/science; 2 social science; and 4 additional approved courses"), if provided.
       - di_min_core_gpa: The minimum core-course GPA required (e.g., "2.3"), if provided.
       - di_progressive_rule_text: The description of the 10/7 progressive completion rule as stated (e.g., "10 of 16 core courses completed, including 7 in English/math/science, before the start of the 7th semester of high school"), if provided.

    Rules:
    - Return null for any missing scalar field.
    - Return an empty list for any URL category with no URLs.
    - For URLs, include full URLs, and extract all that were given for each category.
    - Do not invent or normalize numeric values; copy what the answer states.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def union_urls(*url_lists: List[str]) -> List[str]:
    """Order-preserving de-duplication across multiple URL lists."""
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for url in lst or []:
            u = (url or "").strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Tree construction and verification                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, root, ext: SubmissionExtraction) -> None:
    """
    Build the verification tree exactly per the rubric and schedule all verifications.
    """
    # Create the top-level critical evaluation node (root child)
    top = evaluator.add_parallel(
        id="Complete_Task_Evaluation",
        desc="Evaluate whether the response identifies a qualifying public university and provides all required evidence and NCAA eligibility documentation per the question and constraints.",
        parent=root,
        critical=True
    )

    # 1) Answer provides required names
    names_node = evaluator.add_parallel(
        id="Answer_Provides_Required_Names",
        desc="Response provides the requested names (state and one public university).",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(ext.state),
        id="State_Name_Provided",
        desc="Provides the name of the state.",
        parent=names_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(ext.university),
        id="University_Name_Provided",
        desc="Provides the name of one public university.",
        parent=names_node,
        critical=True
    )

    # Prepare combined sources used in several checks
    combined_publicness_sources = union_urls(
        ext.bog_governance_urls,
        ext.systemwide_bot_rule_urls,
        ext.university_bot_urls
    )
    combined_location_sources = union_urls(
        ext.bog_governance_urls,
        ext.systemwide_bot_rule_urls,
        ext.university_bot_urls,
        ext.sacscoc_urls,
        ext.ncaa_division1_urls
    )

    # Collect all verifications to run in parallel
    batch: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # 2) State ITMC membership
    itmc_node = evaluator.add_parallel(
        id="State_ITMC_Membership",
        desc="State meets the ITMC membership criterion and includes required URL evidence.",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ext.itmc_urls) > 0,
        id="ITMC_Membership_URL",
        desc="Provides a URL reference supporting the state's ITMC membership status.",
        parent=itmc_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="ITMC_Membership_Verification",
        desc="The identified state is a member of the Interstate Teacher Mobility Compact (ITMC).",
        parent=itmc_node,
        critical=True
    )
    claim = f"The U.S. state of {ext.state} is a member of the Interstate Teacher Mobility Compact (ITMC)."
    batch.append((claim, ext.itmc_urls, leaf, "Use the provided official or authoritative source to confirm current membership (not merely proposed or pending)."))

    # 3) State Board of Governors governance
    bog_node = evaluator.add_parallel(
        id="State_Board_of_Governors_Governance",
        desc="State public university system governance matches the Board of Governors requirement and includes required URL evidence.",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ext.bog_governance_urls) > 0,
        id="BOG_Governance_URL",
        desc="Provides a URL reference supporting the Board of Governors governance structure and composition.",
        parent=bog_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="BOG_Governs_State_System",
        desc="The state's public university system is governed by a state-level Board of Governors.",
        parent=bog_node,
        critical=True
    )
    claim = f"The public university system of {ext.state} is governed by a state-level Board of Governors."
    batch.append((claim, ext.bog_governance_urls, leaf, "Accept synonymous names like 'State University System' governed by a 'Board of Governors'."))

    leaf = evaluator.add_leaf(
        id="BOG_Has_Exactly_17_Members",
        desc="The state-level Board of Governors consists of exactly 17 members.",
        parent=bog_node,
        critical=True
    )
    claim = "The state-level Board of Governors has exactly 17 members."
    batch.append((claim, ext.bog_governance_urls, leaf, "Count appointed and ex officio seats if listed; confirm the total equals 17."))

    # 4) University meets system and public requirements
    univ_req_node = evaluator.add_parallel(
        id="University_Meets_System_And_Public_Requirements",
        desc="Chosen university matches the required institutional constraints implied by the state's system and the question.",
        parent=top,
        critical=True
    )
    # University is public
    leaf = evaluator.add_leaf(
        id="University_Is_Public",
        desc="University is a public (state-funded) institution.",
        parent=univ_req_node,
        critical=True
    )
    claim = f"{ext.university} is a public (state-funded) university."
    batch.append((claim, combined_publicness_sources, leaf, "Membership in the state's public university system and state-appointed trustees strongly indicate public status."))

    # University in chosen state
    leaf = evaluator.add_leaf(
        id="University_In_Chosen_State",
        desc="University is located in the identified qualifying state.",
        parent=univ_req_node,
        critical=True
    )
    claim = f"{ext.university} is located in the U.S. state of {ext.state}."
    batch.append((claim, combined_location_sources, leaf, "Accept evidence that shows the campus city with the state's name or abbreviation."))

    # University in state system
    leaf = evaluator.add_leaf(
        id="University_In_State_System",
        desc="University is part of the state's public university system governed as specified.",
        parent=univ_req_node,
        critical=True
    )
    claim = f"{ext.university} is part of {ext.state}'s public university system governed by the state Board of Governors."
    batch.append((claim, combined_publicness_sources, leaf, "Look for the institution being listed as part of the state university system or governed by the Board of Governors."))

    # 5) System-wide Board of Trustees requirement
    sys_bot_node = evaluator.add_parallel(
        id="State_Systemwide_Board_of_Trustees_Requirement",
        desc="The state system satisfies the system-wide trustees criterion (applies to each university in the system) and includes required URL evidence.",
        parent=top,
        critical=True
    )
    # Allow either a dedicated systemwide rule URL or a BOG regulation page to fulfill evidence
    systemwide_sources = union_urls(ext.systemwide_bot_rule_urls, ext.bog_governance_urls)
    evaluator.add_custom_node(
        result=len(systemwide_sources) > 0,
        id="Systemwide_BOT_Rule_URL",
        desc="Provides a URL reference supporting that the trustees structure/composition requirement applies system-wide (i.e., to each university in the system).",
        parent=sys_bot_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="Systemwide_BOT_Rule_Applies_To_Each_University",
        desc="The state system has a rule/structure such that each university in the system has its own Board of Trustees meeting the specified composition requirements.",
        parent=sys_bot_node,
        critical=True
    )
    claim = (
        f"In {ext.state}'s public university system, each university has its own Board of Trustees with 13 members, "
        f"including exactly 6 appointed by the Governor and 5 appointed by the state Board of Governors."
    )
    batch.append((claim, systemwide_sources, leaf, "Confirm this rule applies system-wide (not just one university)."))

    # 6) University Board of Trustees composition
    univ_bot_node = evaluator.add_parallel(
        id="University_Board_of_Trustees_Composition",
        desc="Selected university's Board of Trustees structure matches the required counts/appointment sources and includes required URL evidence.",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ext.university_bot_urls) > 0,
        id="BOT_Composition_URL",
        desc="Provides a URL reference supporting the university Board of Trustees composition/appointment structure.",
        parent=univ_bot_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="BOT_Has_Exactly_13_Members",
        desc="University has a Board of Trustees consisting of exactly 13 members.",
        parent=univ_bot_node,
        critical=True
    )
    claim = f"The Board of Trustees of {ext.university} has exactly 13 members."
    batch.append((claim, ext.university_bot_urls, leaf, "Confirm the total number of trustees is 13."))

    leaf = evaluator.add_leaf(
        id="BOT_Exactly_6_Appointed_By_Governor",
        desc="Exactly 6 trustees are appointed by the state Governor.",
        parent=univ_bot_node,
        critical=True
    )
    claim = f"Exactly 6 members of {ext.university}'s Board of Trustees are appointed by the Governor of {ext.state}."
    batch.append((claim, ext.university_bot_urls, leaf, "Verify the appointment source and exact count '6 by the Governor'."))

    leaf = evaluator.add_leaf(
        id="BOT_Exactly_5_Appointed_By_BOG",
        desc="Exactly 5 trustees are appointed by the state Board of Governors.",
        parent=univ_bot_node,
        critical=True
    )
    claim = f"Exactly 5 members of {ext.university}'s Board of Trustees are appointed by the state Board of Governors."
    batch.append((claim, ext.university_bot_urls, leaf, "Verify the appointment source and exact count '5 by the Board of Governors'."))

    # 7) SACSCOC accreditation
    sacs_node = evaluator.add_parallel(
        id="SACSCOC_Accreditation",
        desc="University meets SACSCOC accreditation requirement and includes required URL evidence.",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ext.sacscoc_urls) > 0,
        id="SACSCOC_URL",
        desc="Provides a URL reference supporting SACSCOC accreditation status.",
        parent=sacs_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="Accredited_By_SACSCOC",
        desc="University is accredited by SACSCOC.",
        parent=sacs_node,
        critical=True
    )
    claim = f"{ext.university} is accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC)."
    batch.append((claim, ext.sacscoc_urls, leaf, "Confirm institutional accreditation by SACSCOC (not programmatic)."))

    # 8) NCAA Division I status
    ncaa_node = evaluator.add_parallel(
        id="NCAA_Division_I_Status",
        desc="University meets NCAA Division I requirement and includes required URL evidence.",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ext.ncaa_division1_urls) > 0,
        id="NCAA_DI_URL",
        desc="Provides a URL reference supporting NCAA Division I athletic status.",
        parent=ncaa_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="Competes_In_NCAA_Division_I",
        desc="University competes in NCAA Division I athletics.",
        parent=ncaa_node,
        critical=True
    )
    claim = f"{ext.university} competes in NCAA Division I athletics."
    batch.append((claim, ext.ncaa_division1_urls, leaf, "Use authoritative sources (NCAA, conference, or institutional athletics page)."))

    # 9) NCAA DI academic eligibility standards documented (verify statements appear in the answer)
    di_doc_node = evaluator.add_parallel(
        id="NCAA_DI_Academic_Eligibility_Standards_Documented",
        desc="Response documents the NCAA Division I academic eligibility standards with the specific values required by the constraints.",
        parent=top,
        critical=True
    )
    # 9.1 Total core courses = 16
    leaf = evaluator.add_leaf(
        id="Core_Courses_Total_Is_16",
        desc="States that NCAA Division I eligibility requires 16 NCAA-approved core courses (total).",
        parent=di_doc_node,
        critical=True
    )
    claim = "The answer explicitly states that NCAA Division I eligibility requires a total of 16 core courses."
    batch.append((claim, None, leaf, "Check the answer content itself; allow minor phrasing differences as long as '16 core courses' is clearly stated."))

    # 9.2 Distribution matches constraint
    leaf = evaluator.add_leaf(
        id="Core_Course_Distribution_Matches_Constraint",
        desc="Includes the specified 16-core-course distribution: 4 English; 3 math (Algebra 1 or higher); 2 science (including 1 lab if offered); 1 additional year English/math/science; 2 social science; and 4 additional approved courses.",
        parent=di_doc_node,
        critical=True
    )
    claim = (
        "The answer includes the NCAA Division I 16-core-course distribution exactly or equivalently: "
        "4 English; 3 math (Algebra I or higher); 2 science (including 1 lab if offered); "
        "1 additional year of English, math, or science; 2 social science; and 4 additional approved courses."
    )
    batch.append((claim, None, leaf, "Allow paraphrases that keep the same numbers and categories; minor wording differences are acceptable."))

    # 9.3 Minimum core-course GPA = 2.3
    leaf = evaluator.add_leaf(
        id="Minimum_Core_Course_GPA_Is_2_3",
        desc="States that the minimum NCAA Division I core-course GPA for eligibility is 2.3.",
        parent=di_doc_node,
        critical=True
    )
    claim = "The answer states that the minimum NCAA Division I core-course GPA requirement is 2.3."
    batch.append((claim, None, leaf, "Accept forms like 'minimum 2.3 core-course GPA' or '2.3 GPA in core courses'."))

    # 9.4 Progressive completion 10/7 rule
    leaf = evaluator.add_leaf(
        id="Progressive_Completion_10_7_Correct",
        desc="Correctly explains the 10/7 progressive completion rule: 10 of 16 core courses completed (including 7 in English/math/science) before the start of the 7th semester of high school.",
        parent=di_doc_node,
        critical=True
    )
    claim = (
        "The answer correctly explains the NCAA Division I 10/7 progressive completion rule: "
        "by the start of the 7th semester of high school, 10 of the 16 core courses must be completed, "
        "including 7 in English, math, or science."
    )
    batch.append((claim, None, leaf, "Allow slight paraphrasing; must include both '10 of 16' and '7 in English/math/science' before the 7th semester."))

    # Execute all verifications in parallel
    await evaluator.batch_verify(batch)


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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point for evaluating an answer for the ITMC public university task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall evaluation combines parallel criteria
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_submission(),
        template_class=SubmissionExtraction,
        extraction_name="submission_extraction",
    )

    # Optional: record simple custom info on URL counts
    evaluator.add_custom_info(
        info={
            "state": extracted.state,
            "university": extracted.university,
            "url_counts": {
                "itmc": len(extracted.itmc_urls),
                "bog_governance": len(extracted.bog_governance_urls),
                "systemwide_bot_rule": len(extracted.systemwide_bot_rule_urls),
                "university_bot": len(extracted.university_bot_urls),
                "sacscoc": len(extracted.sacscoc_urls),
                "ncaa_division1": len(extracted.ncaa_division1_urls),
            },
        },
        info_type="extraction_summary",
        info_name="extraction_summary",
    )

    # Build tree and verify
    await build_and_verify(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()