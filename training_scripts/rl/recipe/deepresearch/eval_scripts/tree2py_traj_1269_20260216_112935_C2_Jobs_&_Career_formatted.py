import asyncio
import logging
from typing import Optional, Dict, Any

from pydantic import BaseModel

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "neu_job_feb2026"
TASK_DESCRIPTION = (
    "Identify a current job posting at Northeastern University in Boston, Massachusetts that meets ALL of the following "
    "criteria: (1) Requires a bachelor's degree as the minimum educational qualification, (2) Requires exactly 5 years "
    "of professional work experience, (3) Is a full-time staff or administrative position (not a faculty position), "
    "(4) Was posted in February 2026. Provide the job title and a reference URL to the job posting."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SingleJobExtraction(BaseModel):
    job_title: Optional[str] = None
    job_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_single_job() -> str:
    return """
    Extract the first job posting referenced in the answer.

    Return a JSON object with:
    - job_title: the job title as written in the answer
    - job_url: the URL to the specific job posting page (not a general job listing page if a specific posting URL is present)

    Rules:
    - If multiple jobs or URLs are given, choose the first that appears to be a direct job posting page.
    - Accept URLs that appear in plain text or markdown link format; extract the actual URL.
    - Do not invent any information; only return data explicitly present in the answer.
    - If a field is missing, set it to null.
    - If a URL lacks a protocol, prepend http:// to make it a valid URL.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_job_posting(
    evaluator: Evaluator,
    root_node,
    job: SingleJobExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and verify each criterion
    using the provided job posting URL as evidence.
    """
    # Top-level critical node
    correct_node = evaluator.add_parallel(
        id="Correct_Position_Identified",
        desc="The candidate has correctly identified a current Northeastern University job posting that meets all specified criteria and provides the required outputs",
        parent=root_node,
        critical=True,
    )

    # Required outputs (critical)
    outputs_node = evaluator.add_parallel(
        id="Required_Output_Provided",
        desc="The response includes the required identifying information for the posting",
        parent=correct_node,
        critical=True,
    )

    # Check job title provided
    title_ok = bool(job.job_title and str(job.job_title).strip())
    evaluator.add_custom_node(
        result=title_ok,
        id="Job_Title_Provided",
        desc="The response provides the job title",
        parent=outputs_node,
        critical=True,
    )

    # Check reference URL provided (basic plausibility)
    url_ok = bool(job.job_url and isinstance(job.job_url, str) and job.job_url.strip())
    evaluator.add_custom_node(
        result=url_ok,
        id="Reference_URL_Provided",
        desc="The response provides a reference URL to the job posting",
        parent=outputs_node,
        critical=True,
    )

    # Posting meets all criteria (critical)
    meets_all_node = evaluator.add_parallel(
        id="Posting_Meets_All_Criteria",
        desc="The identified posting satisfies all stated constraints (institution/location, qualifications, position type, posting date, and is current)",
        parent=correct_node,
        critical=True,
    )

    # Institution and location (critical leaf)
    inst_loc_leaf = evaluator.add_leaf(
        id="Institution_and_Location",
        desc="The position is posted at Northeastern University in Boston, Massachusetts",
        parent=meets_all_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This job posting is for Northeastern University and the primary work location is Boston, Massachusetts (Boston, MA).",
        node=inst_loc_leaf,
        sources=job.job_url if url_ok else None,
        additional_instruction=(
            "Verify BOTH the institution and location on the job posting page. "
            "Accept variants such as 'Boston, MA' or 'Boston campus'. "
            "If multiple locations are listed, there must be an explicit indication that the role is in, or primarily based in, Boston, MA. "
            "Reject if the role is clearly for a different campus/city with no clear Boston, MA association."
        ),
    )

    # Posting is current (critical leaf)
    current_leaf = evaluator.add_leaf(
        id="Posting_Is_Current",
        desc="The posting is current at the time of the response (e.g., the listing is accessible and not indicated as closed/archived/filled)",
        parent=meets_all_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This job posting is currently open/active (not closed, archived, expired, or filled).",
        node=current_leaf,
        sources=job.job_url if url_ok else None,
        additional_instruction=(
            "Use page evidence such as the presence of an 'Apply' button or language indicating the posting is open. "
            "If the page states 'no longer available', 'closed', 'archived', or similar, then it is NOT current."
        ),
    )

    # Qualification requirements (critical group)
    qual_node = evaluator.add_parallel(
        id="Qualification_Requirements",
        desc="The position meets the educational and experience requirements",
        parent=meets_all_node,
        critical=True,
    )

    # Educational requirement (critical leaf)
    edu_leaf = evaluator.add_leaf(
        id="Educational_Requirement",
        desc="The position requires a bachelor's degree as the minimum educational qualification",
        parent=qual_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The minimum educational qualification required for this position is a bachelor's degree.",
        node=edu_leaf,
        sources=job.job_url if url_ok else None,
        additional_instruction=(
            "Look for explicit statements such as 'Bachelor's degree required'. "
            "It's acceptable if the posting says bachelor's required and higher degrees are preferred. "
            "Reject if the minimum requirement is a master's, PhD, or higher, or if no bachelor's minimum is indicated."
        ),
    )

    # Experience requirement (critical leaf)
    exp_leaf = evaluator.add_leaf(
        id="Experience_Requirement",
        desc="The position requires exactly 5 years of professional work experience",
        parent=qual_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The position requires exactly 5 years of professional work experience.",
        node=exp_leaf,
        sources=job.job_url if url_ok else None,
        additional_instruction=(
            "Accept only if the page explicitly indicates 5 years (e.g., 'requires 5 years of experience'). "
            "Reject if it states '5+ years', 'at least 5 years', 'minimum of 5 years', ranges including 5 (e.g., '4–6 years' or '5–7 years'), "
            "or any phrasing that does not indicate exactly 5 years."
        ),
    )

    # Position details (critical group)
    pos_node = evaluator.add_parallel(
        id="Position_Details",
        desc="The position meets the employment status, category, and timing requirements",
        parent=meets_all_node,
        critical=True,
    )

    # Employment status = full-time (critical leaf)
    empl_leaf = evaluator.add_leaf(
        id="Employment_Status",
        desc="The position is a full-time role (not part-time, temporary, or substitute)",
        parent=pos_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This position is full-time.",
        node=empl_leaf,
        sources=job.job_url if url_ok else None,
        additional_instruction=(
            "Look for 'Full-time' or equivalent on the posting. "
            "Reject if part-time, temporary, seasonal, or substitute."
        ),
    )

    # Position category = staff/administrative, not faculty (critical leaf)
    cat_leaf = evaluator.add_leaf(
        id="Position_Category",
        desc="The position is a staff or administrative role (not a faculty position)",
        parent=pos_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This position is a staff or administrative role (non-faculty).",
        node=cat_leaf,
        sources=job.job_url if url_ok else None,
        additional_instruction=(
            "Accept categories such as 'Staff', 'Administrative', 'Professional Staff'. "
            "Reject if the posting indicates 'Faculty', 'Professor', 'Teaching Faculty', or similar academic ranks."
        ),
    )

    # Posting timeliness = posted in February 2026 (critical leaf)
    time_leaf = evaluator.add_leaf(
        id="Posting_Timeliness",
        desc="The job posting was published in February 2026",
        parent=pos_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This job was posted in February 2026.",
        node=time_leaf,
        sources=job.job_url if url_ok else None,
        additional_instruction=(
            "Verify the posting date on the page. Accept if the 'Posted' or 'Date posted' is in February 2026. "
            "Do not rely on 'Updated' or 'Last updated' alone unless the posting date is clearly indicated as February 2026."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate a single answer for the Northeastern University job posting task.
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

    # Extract the first job posting (title + URL) from the answer
    extracted_job: SingleJobExtraction = await evaluator.extract(
        prompt=prompt_extract_single_job(),
        template_class=SingleJobExtraction,
        extraction_name="job_extraction",
    )

    # Build tree and verify
    await verify_job_posting(evaluator, root, extracted_job)

    # Return the evaluation summary
    return evaluator.get_summary()