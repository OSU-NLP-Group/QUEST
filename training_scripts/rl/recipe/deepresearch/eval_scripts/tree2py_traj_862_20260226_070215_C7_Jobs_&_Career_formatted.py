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
TASK_ID = "teacher_job_fair_2026_tx"
TASK_DESCRIPTION = (
    "A prospective teacher is planning to attend a teacher job fair in 2026 and needs to find one that meets specific "
    "requirements. Identify the name of the school district hosting the teacher job fair that satisfies ALL of the following "
    "criteria: The fair must be held in Texas. The fair must be held in March 2026. The fair must be held on a Saturday. "
    "The fair must accept teachers as eligible participants. The fair must require online pre-registration. The fair must "
    "require bringing certification documentation (such as teaching certificate, exam scores, or Statement of Eligibility). "
    "The fair must require completing a district employment application before attending the fair. The fair must provide a "
    "specific street address for the venue location. The fair must have clearly defined start and end times. The fair must be "
    "specifically for professional/certified educator positions (not support staff or auxiliary positions). The fair must be "
    "organized directly by a school district (not a third-party organization). The fair must require bringing a confirmation "
    "badge or name tag from registration. The fair must require bringing copies of a professional resume. Provide the name of "
    "the school district and the specific date of the job fair."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class JobFairExtraction(BaseModel):
    # Core identifiers and details extracted from the agent's answer
    district_name: Optional[str] = None                 # Name of the school district hosting the fair
    event_name: Optional[str] = None                    # Event/fair name as written
    date_text: Optional[str] = None                     # Specific date string the answer provides (e.g., "Saturday, March 7, 2026")
    address: Optional[str] = None                       # Street address line if provided (e.g., "123 Main St.")
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    start_time: Optional[str] = None                    # Start time string if given
    end_time: Optional[str] = None                      # End time string if given

    primary_url: Optional[str] = None                   # The main/primary URL cited for the fair (prefer the district page)
    source_urls: List[str] = Field(default_factory=list)  # All URLs cited that support this fair (can include primary)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_job_fair() -> str:
    return """
    You will extract exactly one teacher job fair candidate that the answer proposes as satisfying the task. 
    If the answer mentions multiple fairs, pick the first one that appears to meet the requirements. 
    Extract the fields exactly as they appear in the answer text. Do not invent anything.

    Required fields:
    - district_name: The name of the school district that is hosting/organizing the fair (not a third-party). 
                     If the answer only names the fair but not the district, return null.
    - event_name: The name of the fair/event if mentioned, otherwise null.
    - date_text: The specific calendar date string mentioned in the answer for the fair (include month/day/year and day-of-week if present).
    - address: The specific street address string for the venue (if provided), otherwise null.
    - city: The city portion of the address if provided, otherwise null.
    - state: The state (e.g., 'TX' or 'Texas') if provided in the answer address, otherwise null.
    - zip: The ZIP/postal code if provided, otherwise null.
    - start_time: The start time if provided, otherwise null.
    - end_time: The end time if provided, otherwise null.

    URL sources:
    - primary_url: The main URL for the event details page (prefer the official district website). If none is cited, set to null.
    - source_urls: An array of ALL URLs cited in the answer that support this fair (include the primary URL as the first element if present).
                   If the answer provides no URLs, return an empty array.

    Return all fields as a single JSON object. For any missing field, return null or empty array as specified.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(extraction: JobFairExtraction) -> List[str]:
    urls: List[str] = []
    if extraction.primary_url:
        u = extraction.primary_url.strip()
        if u:
            urls.append(u)
    for u in extraction.source_urls or []:
        if u and isinstance(u, str):
            u2 = u.strip()
            if u2 and u2 not in urls:
                urls.append(u2)
    return urls


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def verify_teacher_job_fair(
    evaluator: Evaluator,
    parent_node,
    extracted: JobFairExtraction
) -> None:
    """
    Build leaf checks for all criteria under the parent node and verify them against cited sources.
    We add one gating custom node 'Sources_Provided' (critical) to enforce source-grounding.
    """
    # 0) Gate with sources existence (critical) so other leaf verifications will be skipped if missing.
    sources = _combine_sources(extracted)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="Sources_Provided",
        desc="At least one URL source is provided for the job fair",
        parent=parent_node,
        critical=True
    )

    # Prepare all leaf nodes (critical vs non-critical as per rubric)
    # Note: We keep all leaf nodes as single checks. Most are critical; dress code is non-critical.
    nodes_and_claims: List[tuple] = []

    # 1. Located in Texas (critical)
    node_loc_tx = evaluator.add_leaf(
        id="Located_in_Texas",
        desc="The job fair must be held in Texas",
        parent=parent_node,
        critical=True
    )
    claim_loc_tx = "The event page shows that the job fair venue is located in Texas (TX)."
    ins_loc_tx = "Look for city/state in the address or references like 'TX' or 'Texas' associated with the venue."

    nodes_and_claims.append((claim_loc_tx, sources, node_loc_tx, ins_loc_tx))

    # 2. Held in March 2026 (critical)
    node_march_2026 = evaluator.add_leaf(
        id="Held_in_March_2026",
        desc="The job fair must be held in March 2026",
        parent=parent_node,
        critical=True
    )
    if extracted.date_text:
        claim_march_2026 = f"The job fair is scheduled for {extracted.date_text}, which is in March 2026."
    else:
        claim_march_2026 = "The job fair occurs in March 2026."
    ins_march_2026 = "Verify the event date on the page is within March 2026."
    nodes_and_claims.append((claim_march_2026, sources, node_march_2026, ins_march_2026))

    # 3. Held on Saturday (critical)
    node_sat = evaluator.add_leaf(
        id="Held_on_Saturday",
        desc="The job fair must be held on a Saturday",
        parent=parent_node,
        critical=True
    )
    if extracted.date_text:
        claim_sat = f"The job fair date shown ({extracted.date_text}) is on a Saturday."
    else:
        claim_sat = "The job fair is held on a Saturday."
    ins_sat = "Accept if the page explicitly indicates Saturday for the event date. If only the full date is shown, check if it includes the word 'Saturday'."
    nodes_and_claims.append((claim_sat, sources, node_sat, ins_sat))

    # 4. Accepts teachers (critical)
    node_teachers = evaluator.add_leaf(
        id="Accepts_Teachers",
        desc="The job fair must accept teachers as eligible participants",
        parent=parent_node,
        critical=True
    )
    claim_teachers = "Teachers (certified educators) are eligible participants for this job fair."
    ins_teachers = "Look for eligibility statements indicating teachers/certified educators are invited or targeted."
    nodes_and_claims.append((claim_teachers, sources, node_teachers, ins_teachers))

    # 5. Requires online pre-registration (critical)
    node_prereg = evaluator.add_leaf(
        id="Requires_Online_Preregistration",
        desc="The job fair must require online pre-registration",
        parent=parent_node,
        critical=True
    )
    claim_prereg = "Online pre-registration is required to attend the job fair."
    ins_prereg = "Look for explicit 'pre-registration required', 'register online', or 'must register in advance online' requirements."
    nodes_and_claims.append((claim_prereg, sources, node_prereg, ins_prereg))

    # 6. Requires certification documentation (critical)
    node_cert_docs = evaluator.add_leaf(
        id="Requires_Certification_Documentation",
        desc="The job fair must require bringing certification documentation (such as teaching certificate, exam scores, or Statement of Eligibility)",
        parent=parent_node,
        critical=True
    )
    claim_cert_docs = "Attendees must bring certification documentation (e.g., teaching certificate, exam scores, Statement of Eligibility)."
    ins_cert_docs = "Accept if the page lists any educator credential documentation required at check-in."
    nodes_and_claims.append((claim_cert_docs, sources, node_cert_docs, ins_cert_docs))

    # 7. Requires district employment application beforehand (critical)
    node_district_app = evaluator.add_leaf(
        id="Requires_District_Application",
        desc="The job fair must require completing a district employment application before attending",
        parent=parent_node,
        critical=True
    )
    claim_district_app = "A completed district employment application is required prior to attending the job fair."
    ins_district_app = "Verify that the page requires applicants to complete the district's job application before the event."
    nodes_and_claims.append((claim_district_app, sources, node_district_app, ins_district_app))

    # 8. Provides specific street address (critical)
    node_address = evaluator.add_leaf(
        id="Provides_Specific_Address",
        desc="The job fair must provide a specific street address for the venue",
        parent=parent_node,
        critical=True
    )
    if extracted.address:
        claim_address = f"The event page provides a specific street address for the venue (e.g., '{extracted.address}')."
    else:
        claim_address = "The event page provides a specific street address for the venue location (including street number and street name)."
    ins_address = "Look for a full street address line (e.g., number + street name). A vague building name alone is insufficient."
    nodes_and_claims.append((claim_address, sources, node_address, ins_address))

    # 9. Has clearly defined start and end times (critical)
    node_time_window = evaluator.add_leaf(
        id="Has_Defined_Time_Window",
        desc="The job fair must have clearly defined start and end times",
        parent=parent_node,
        critical=True
    )
    if extracted.start_time and extracted.end_time:
        claim_time_window = f"The event page states a specific start and end time (e.g., {extracted.start_time} to {extracted.end_time})."
    else:
        claim_time_window = "The event page states a specific time range with both a start and end time for the job fair."
    ins_time_window = "Verify that both a start time and an end time are present (e.g., '9:00 AM – 12:00 PM')."
    nodes_and_claims.append((claim_time_window, sources, node_time_window, ins_time_window))

    # 10. For professional/certified educator positions only (critical)
    node_prof_positions = evaluator.add_leaf(
        id="For_Professional_Positions",
        desc="The job fair must be specifically for professional/certified educator positions (not support staff or auxiliary positions)",
        parent=parent_node,
        critical=True
    )
    claim_prof_positions = "The job fair targets professional/certified educator positions (not support/auxiliary roles)."
    ins_prof_positions = "Look for language like 'certified', 'teacher', 'professional educator', and exclusions of 'support' or 'auxiliary' positions."
    nodes_and_claims.append((claim_prof_positions, sources, node_prof_positions, ins_prof_positions))

    # 11. Organized directly by a school district (critical)
    node_district_host = evaluator.add_leaf(
        id="Organized_by_School_District",
        desc="The job fair must be organized directly by a school district (not a third-party organization or consortium)",
        parent=parent_node,
        critical=True
    )
    if extracted.district_name:
        claim_district_host = f"The job fair is organized directly by {extracted.district_name}, a school district (not a third-party)."
    else:
        claim_district_host = "The job fair is organized directly by a school district (not a third-party organization)."
    ins_district_host = "Prefer evidence from an official district website/page stating the district is hosting or organizing the fair."
    nodes_and_claims.append((claim_district_host, sources, node_district_host, ins_district_host))

    # 12. Requires bringing confirmation badge or name tag (critical)
    node_badge = evaluator.add_leaf(
        id="Requires_Confirmation_Badge",
        desc="The job fair must require bringing a confirmation badge, name tag, or identification from registration",
        parent=parent_node,
        critical=True
    )
    claim_badge = "Attendees are required to bring a confirmation badge, name tag, or registration confirmation for check-in."
    ins_badge = "Accept phrasing like 'print your badge', 'bring your confirmation name tag', or similar."
    nodes_and_claims.append((claim_badge, sources, node_badge, ins_badge))

    # 13. Requires bringing copies of a resume (critical)
    node_resume = evaluator.add_leaf(
        id="Requires_Resume_Copies",
        desc="The job fair must require bringing copies of a professional resume",
        parent=parent_node,
        critical=True
    )
    claim_resume = "Attendees must bring copies of a professional resume."
    ins_resume = "Look for explicit mentions of bringing multiple printed copies of resumes."
    nodes_and_claims.append((claim_resume, sources, node_resume, ins_resume))

    # 14. Professional dress code (non-critical)
    node_dress = evaluator.add_leaf(
        id="Professional_Dress_Code",
        desc="The job fair information states a professional dress code expectation",
        parent=parent_node,
        critical=False
    )
    claim_dress = "The event page states a professional or business attire dress code is expected."
    ins_dress = "Accept 'professional attire', 'business professional', or 'business casual' if clearly framed as a dress code expectation."
    nodes_and_claims.append((claim_dress, sources, node_dress, ins_dress))

    # Run all verifications in parallel; gating from 'Sources_Provided' (critical custom node) will auto-skip if it failed.
    await evaluator.batch_verify(nodes_and_claims)


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
    Evaluate an answer for the teacher job fair 2026 in Texas task.
    Returns a structured summary with the verification tree and final score.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
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

    # Extract candidate fair info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_job_fair(),
        template_class=JobFairExtraction,
        extraction_name="job_fair_extraction"
    )

    # Build top-level node for rubric checks (keep non-critical to allow optional sub-checks)
    fair_node = evaluator.add_parallel(
        id="Teacher_Job_Fair_Identification",
        desc="Identify the teacher job fair in 2026 that meets all specified criteria",
        parent=root,
        critical=False
    )

    # Add a small piece of custom info to help debugging
    evaluator.add_custom_info(
        info={
            "district_name": extracted.district_name,
            "event_name": extracted.event_name,
            "date_text": extracted.date_text,
            "address": extracted.address,
            "city": extracted.city,
            "state": extracted.state,
            "zip": extracted.zip,
            "start_time": extracted.start_time,
            "end_time": extracted.end_time,
            "primary_url": extracted.primary_url,
            "source_urls": extracted.source_urls,
        },
        info_type="extraction_overview",
        info_name="extracted_job_fair_overview"
    )

    # Verify against rubric
    await verify_teacher_job_fair(evaluator, fair_node, extracted)

    # Return evaluation summary
    return evaluator.get_summary()