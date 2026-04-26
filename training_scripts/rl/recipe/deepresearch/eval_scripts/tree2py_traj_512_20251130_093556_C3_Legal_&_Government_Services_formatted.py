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
TASK_ID = "pa_clemency_life_process"
TASK_DESCRIPTION = (
    "I am researching the clemency process in Pennsylvania for individuals serving life sentences. "
    "Please provide comprehensive information about the following aspects of Pennsylvania's clemency system: "
    "(1) The name and composition (number of members) of the state board or entity responsible for reviewing clemency applications in Pennsylvania, "
    "(2) The specific voting threshold required at the merit review stage for a life sentence applicant to be granted a public hearing (specify both the number of votes required and the total number of board members), "
    "(3) The specific voting requirement at the public hearing stage for a life sentence case to be recommended for clemency (specify whether it requires a simple majority, supermajority, unanimous vote, and the exact number of votes required out of the total board members), "
    "(4) The complete list of document types that applicants must submit with their clemency application in Pennsylvania, and "
    "(5) The specific office or position that holds the final decision-making authority for granting or denying clemency in Pennsylvania after receiving the board's recommendation. "
    "For each piece of information above, please provide at least one supporting URL from an official Pennsylvania government source (such as pa.gov domains or official state agency websites) that verifies the information."
)

# Expected normative references used in additional instructions
EXPECTED_DOC_LIST = [
    "criminal complaint",
    "affidavit of probable cause",
    "criminal information or indictment",
    "final plea or verdict",
    "sentencing order",
    "documentation of financial obligation status"
]
EXPECTED_PROCESS_SEQUENCE = [
    "application submission",
    "Board review",
    "DOC investigation",
    "merit review",
    "public hearing",
    "Governor's decision"
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BoardInfo(BaseModel):
    name: Optional[str] = None
    member_count: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MeritReviewLife(BaseModel):
    votes_required: Optional[str] = None
    total_members: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MeritReviewGeneral(BaseModel):
    votes_required: Optional[str] = None
    total_members: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HearingPrerequisite(BaseModel):
    statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PublicHearingLifeDeath(BaseModel):
    vote_type: Optional[str] = None  # e.g., "unanimous"
    votes_required: Optional[str] = None  # e.g., "5"
    total_members: Optional[str] = None  # e.g., "5"
    urls: List[str] = Field(default_factory=list)


class PublicHearingGeneral(BaseModel):
    vote_type: Optional[str] = None  # e.g., "majority"
    votes_required: Optional[str] = None  # e.g., "3"
    total_members: Optional[str] = None  # e.g., "5"
    urls: List[str] = Field(default_factory=list)


class DocumentsRequirement(BaseModel):
    required_documents: List[str] = Field(default_factory=list)
    incarcerated_exception: Optional[str] = None  # e.g., "currently incarcerated applicants do not need to submit court documents"
    urls: List[str] = Field(default_factory=list)


class FinalAuthority(BaseModel):
    office_title: Optional[str] = None  # e.g., "Governor of Pennsylvania"
    statement: Optional[str] = None  # e.g., "may approve or disapprove Board recommendations"
    urls: List[str] = Field(default_factory=list)


class ProcessDuration(BaseModel):
    duration_phrase: Optional[str] = None  # e.g., "a few years"
    urls: List[str] = Field(default_factory=list)


class ProcessSteps(BaseModel):
    steps_ordered: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class PAClemencyExtraction(BaseModel):
    board: Optional[BoardInfo] = None
    merit_life: Optional[MeritReviewLife] = None
    merit_general: Optional[MeritReviewGeneral] = None
    hearing_prereq: Optional[HearingPrerequisite] = None
    hearing_life_death: Optional[PublicHearingLifeDeath] = None
    hearing_general: Optional[PublicHearingGeneral] = None
    documents: Optional[DocumentsRequirement] = None
    final_authority: Optional[FinalAuthority] = None
    duration: Optional[ProcessDuration] = None
    process_steps: Optional[ProcessSteps] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pa_clemency() -> str:
    return """
Extract structured information from the answer about Pennsylvania’s clemency process, focusing on life-sentence specifics. Return exactly the fields requested below. For every item that requests URLs, extract only URLs that are explicitly mentioned in the answer text. Prefer official Pennsylvania government sources (e.g., *.pa.gov domains or official PA state agency sites like boardofpardons.pa.gov, governor.pa.gov, padoc.pa.gov). If no URL is provided in the answer for a specific item, return an empty list for that item's URLs.

Fields to extract:
1) board:
   - name: The official name of the reviewing entity (e.g., “Board of Pardons” / “Pennsylvania Board of Pardons”)
   - member_count: The number of members on the board/entity (as stated in the answer)
   - urls: URLs the answer provides that support the board name and/or composition

2) merit_life:
   - votes_required: The number of votes required at merit review for a life sentence applicant to receive a public hearing (numerator)
   - total_members: The total number of Board members considered at merit review (denominator)
   - urls: Official URLs the answer provides that support the life-sentence merit review threshold

3) merit_general:
   - votes_required: The number of votes required at merit review for general (non-life) applicants to receive a public hearing (numerator)
   - total_members: The total number of Board members considered at merit review (denominator)
   - urls: Official URLs the answer provides that support the general-applicant merit review threshold

4) hearing_prereq:
   - statement: The answer’s statement indicating whether a public hearing is constitutionally required before Board recommendation to the Governor
   - urls: Official URLs the answer provides that support this constitutional prerequisite

5) hearing_life_death:
   - vote_type: The vote type at the public hearing for life/death sentences (e.g., “unanimous”)
   - votes_required: The exact number of votes required at the public hearing for life/death sentences (numerator)
   - total_members: The total number of Board members (denominator)
   - urls: Official URLs the answer provides that support this public-hearing requirement

6) hearing_general:
   - vote_type: The vote type at the public hearing for general cases (e.g., “majority”)
   - votes_required: The exact number of votes required (numerator)
   - total_members: The total number of Board members (denominator)
   - urls: Official URLs the answer provides that support this general-case hearing requirement

7) documents:
   - required_documents: A list of document types the answer claims are required (the complete list; include each as a separate item)
   - incarcerated_exception: The answer’s statement about whether currently incarcerated applicants must submit court documents
   - urls: Official URLs the answer provides that support the documents list and/or the incarcerated-applicant exception

8) final_authority:
   - office_title: The office holding the final authority (e.g., “Governor of Pennsylvania”)
   - statement: The answer’s statement about that authority (e.g., “may approve or disapprove Board recommendations”)
   - urls: Official URLs the answer provides that support the final decision authority

9) duration:
   - duration_phrase: The answer’s phrasing of the typical total duration (e.g., “a few years from application to decision”)
   - urls: Official URLs the answer provides that support the typical duration

10) process_steps:
   - steps_ordered: The process steps as listed in the answer (an ordered list, one step per item; keep the answer’s wording)
   - urls: Official URLs the answer provides that describe the process steps

General rules:
- Extract exactly what the answer states. Do not invent information.
- For URLs, include full valid URLs (http/https).
- If any field is missing in the answer, set it to null (for a single value) or [] (for lists).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def safe_str(x: Optional[str]) -> str:
    return x if (x is not None) else ""


def join_list(items: List[str]) -> str:
    return "; ".join(items) if items else ""


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def verify_board_entity(evaluator: Evaluator, parent_node, board: Optional[BoardInfo]) -> None:
    node = evaluator.add_parallel(
        id="Board_Entity_Responsible",
        desc="Identify the entity responsible for reviewing clemency applications in Pennsylvania.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Board_Name (answer states official name)
    leaf_name = evaluator.add_leaf(
        id="Board_Name",
        desc="States the official name of the reviewing entity (Board of Pardons).",
        parent=node,
        critical=True
    )
    claim_name = "The answer states that the entity responsible for reviewing clemency applications is the Pennsylvania Board of Pardons (aka Board of Pardons)."
    await evaluator.verify(
        claim=claim_name,
        node=leaf_name,
        additional_instruction="Use the answer text to confirm the entity name. Accept synonyms like 'Board of Pardons' or 'Pennsylvania Board of Pardons'. If the answer does not clearly state this, mark as incorrect."
    )

    # Leaf: Board_Member_Count (answer states 5 members)
    leaf_count = evaluator.add_leaf(
        id="Board_Member_Count",
        desc="States the board/entity composition as 5 members.",
        parent=node,
        critical=True
    )
    claim_count = "The answer states that the Pennsylvania Board of Pardons consists of 5 members."
    await evaluator.verify(
        claim=claim_count,
        node=leaf_count,
        additional_instruction="Verify in the answer text that the membership count is explicitly stated as '5'. If not explicit, mark as incorrect."
    )

    # Leaf: Board_Official_Source_URL (official PA source verifies name/composition)
    leaf_src = evaluator.add_leaf(
        id="Board_Official_Source_URL",
        desc="Provides at least one supporting URL from an official Pennsylvania government source (e.g., pa.gov or official state agency site) for the board name/composition.",
        parent=node,
        critical=True
    )
    claim_src = "This webpage is an official Pennsylvania government source (pa.gov domain or official PA state agency) and it supports the Board of Pardons identity and 5-member composition."
    await evaluator.verify(
        claim=claim_src,
        node=leaf_src,
        sources=(board.urls if board else []),
        additional_instruction="Treat URLs as official only if they are on a .pa.gov or official Pennsylvania state agency domain (e.g., boardofpardons.pa.gov, governor.pa.gov, padoc.pa.gov). Fail if none of the provided URLs are official or support the stated facts."
    )


async def verify_merit_review_life(evaluator: Evaluator, parent_node, merit_life: Optional[MeritReviewLife]) -> None:
    node = evaluator.add_parallel(
        id="Merit_Review_Life_Sentence_Threshold",
        desc="Merit review voting threshold for life sentence applicants to receive a public hearing.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Life_Merit_Review_Vote_Threshold (answer states 3 of 5)
    leaf_thresh = evaluator.add_leaf(
        id="Life_Merit_Review_Vote_Threshold",
        desc="States that life sentence applicants require approval from 3 of the 5 Board members at merit review to receive a public hearing (i.e., makes clear both numerator and denominator).",
        parent=node,
        critical=True
    )
    claim_thresh = "The answer states that at merit review, life sentence applicants require approval from 3 of the 5 Board members to receive a public hearing."
    await evaluator.verify(
        claim=claim_thresh,
        node=leaf_thresh,
        additional_instruction="Confirm this exact threshold is stated in the answer (3 of 5). If the answer gives a different threshold or omits it, mark as incorrect."
    )

    # Leaf: Life_Merit_Review_Official_Source_URL (official PA source supports threshold)
    leaf_src = evaluator.add_leaf(
        id="Life_Merit_Review_Official_Source_URL",
        desc="Provides at least one official Pennsylvania government URL supporting the life-sentence merit review threshold.",
        parent=node,
        critical=True
    )
    claim_src = "This official Pennsylvania source explicitly supports that life sentence applicants need 3 of 5 votes at the merit review stage to receive a public hearing."
    await evaluator.verify(
        claim=claim_src,
        node=leaf_src,
        sources=(merit_life.urls if merit_life else []),
        additional_instruction="Only accept URLs from official PA government sources. The page must explicitly state the 3-of-5 threshold for life-sentence merit review."
    )


async def verify_merit_review_general(evaluator: Evaluator, parent_node, merit_general: Optional[MeritReviewGeneral]) -> None:
    node = evaluator.add_parallel(
        id="Merit_Review_General_Threshold",
        desc="Merit review voting threshold for general (non-life) applicants to receive a public hearing (constraint requirement).",
        parent=parent_node,
        critical=True
    )

    # Leaf: General_Merit_Review_Vote_Threshold (answer states 2 of 5)
    leaf_thresh = evaluator.add_leaf(
        id="General_Merit_Review_Vote_Threshold",
        desc="States that general applicants require approval from 2 of the 5 Board members at merit review to receive a public hearing.",
        parent=node,
        critical=True
    )
    claim_thresh = "The answer states that at merit review, general (non-life) applicants require approval from 2 of the 5 Board members to receive a public hearing."
    await evaluator.verify(
        claim=claim_thresh,
        node=leaf_thresh,
        additional_instruction="Confirm this exact threshold is stated in the answer (2 of 5). If the answer gives a different threshold or omits it, mark as incorrect."
    )

    # Leaf: General_Merit_Review_Official_Source_URL (official PA source supports threshold)
    leaf_src = evaluator.add_leaf(
        id="General_Merit_Review_Official_Source_URL",
        desc="Provides at least one official Pennsylvania government URL supporting the general-applicant merit review threshold.",
        parent=node,
        critical=True
    )
    claim_src = "This official Pennsylvania source explicitly supports that general (non-life) applicants need 2 of 5 votes at the merit review stage to receive a public hearing."
    await evaluator.verify(
        claim=claim_src,
        node=leaf_src,
        sources=(merit_general.urls if merit_general else []),
        additional_instruction="Only accept URLs from official PA government sources. The page must explicitly state the 2-of-5 threshold for general merit review."
    )


async def verify_hearing_prerequisite(evaluator: Evaluator, parent_node, hearing_prereq: Optional[HearingPrerequisite]) -> None:
    node = evaluator.add_parallel(
        id="Public_Hearing_Prerequisite",
        desc="Public hearing prerequisite before recommendation to the Governor (constraint requirement).",
        parent=parent_node,
        critical=True
    )

    # Leaf: Hearing_Constitutional_Requirement (answer states prerequisite)
    leaf_req = evaluator.add_leaf(
        id="Hearing_Constitutional_Requirement",
        desc="States that a public hearing is constitutionally required before the Board can recommend clemency to the Governor.",
        parent=node,
        critical=True
    )
    claim_req = "The answer states that a public hearing is constitutionally required before the Board can recommend clemency to the Governor."
    await evaluator.verify(
        claim=claim_req,
        node=leaf_req,
        additional_instruction="Confirm that the answer explicitly states the constitutional/public-hearing prerequisite. If not stated, mark as incorrect."
    )

    # Leaf: Hearing_Prerequisite_Official_Source_URL (official PA source supports prerequisite)
    leaf_src = evaluator.add_leaf(
        id="Hearing_Prerequisite_Official_Source_URL",
        desc="Provides at least one official Pennsylvania government URL supporting the constitutional/public-hearing prerequisite claim.",
        parent=node,
        critical=True
    )
    claim_src = "This official Pennsylvania source explicitly supports that a public hearing is required before the Board can recommend clemency to the Governor."
    await evaluator.verify(
        claim=claim_src,
        node=leaf_src,
        sources=(hearing_prereq.urls if hearing_prereq else []),
        additional_instruction="Accept only official PA government URLs; the page must make the prerequisite clear."
    )


async def verify_hearing_life_death(evaluator: Evaluator, parent_node, life_death: Optional[PublicHearingLifeDeath]) -> None:
    node = evaluator.add_parallel(
        id="Public_Hearing_Life_Or_Death_Vote",
        desc="Voting requirement at the public hearing stage for life/death sentence cases (constraint requirement).",
        parent=parent_node,
        critical=True
    )

    # Leaf: Life_Death_Public_Hearing_Unanimity (answer states unanimous 5 of 5)
    leaf_unanimity = evaluator.add_leaf(
        id="Life_Death_Public_Hearing_Unanimity",
        desc="States that life sentence (and death sentence) cases require a unanimous vote, i.e., 5 of 5 Board members, at the public hearing to be recommended to the Governor (includes vote type and exact count).",
        parent=node,
        critical=True
    )
    claim_unanimity = "The answer states that life and death sentence cases require a unanimous vote of 5 out of 5 Board members at the public hearing to be recommended to the Governor."
    await evaluator.verify(
        claim=claim_unanimity,
        node=leaf_unanimity,
        additional_instruction="Confirm the answer explicitly states 'unanimous' and '5 of 5'. If not, mark as incorrect."
    )

    # Leaf: Life_Death_Public_Hearing_Official_Source_URL (official PA source supports unanimity)
    leaf_src = evaluator.add_leaf(
        id="Life_Death_Public_Hearing_Official_Source_URL",
        desc="Provides at least one official Pennsylvania government URL supporting the life/death public hearing voting requirement.",
        parent=node,
        critical=True
    )
    claim_src = "This official Pennsylvania source explicitly supports that life/death sentence cases require a unanimous 5-of-5 vote at the public hearing to be recommended to the Governor."
    await evaluator.verify(
        claim=claim_src,
        node=leaf_src,
        sources=(life_death.urls if life_death else []),
        additional_instruction="Accept only official PA government URLs; the page must explicitly state the unanimous 5-of-5 requirement."
    )


async def verify_hearing_general(evaluator: Evaluator, parent_node, general: Optional[PublicHearingGeneral]) -> None:
    node = evaluator.add_parallel(
        id="Public_Hearing_General_Vote",
        desc="Voting requirement at the public hearing stage for general cases (constraint requirement).",
        parent=parent_node,
        critical=True
    )

    # Leaf: General_Public_Hearing_Majority (answer states majority, at least 3 of 5)
    leaf_majority = evaluator.add_leaf(
        id="General_Public_Hearing_Majority",
        desc="States that general cases require a majority vote (at least 3 of 5 Board members) at the public hearing for recommendation to the Governor (includes vote type and exact count/threshold).",
        parent=node,
        critical=True
    )
    claim_majority = "The answer states that general cases require a majority vote—at least 3 of 5 Board members—at the public hearing to be recommended to the Governor."
    await evaluator.verify(
        claim=claim_majority,
        node=leaf_majority,
        additional_instruction="Confirm the answer explicitly states 'majority' and 'at least 3 of 5'. If not, mark as incorrect."
    )

    # Leaf: General_Public_Hearing_Official_Source_URL (official PA source supports majority)
    leaf_src = evaluator.add_leaf(
        id="General_Public_Hearing_Official_Source_URL",
        desc="Provides at least one official Pennsylvania government URL supporting the general-case public hearing voting requirement.",
        parent=node,
        critical=True
    )
    claim_src = "This official Pennsylvania source explicitly supports that general cases require a majority vote (at least 3 of 5) at the public hearing for recommendation to the Governor."
    await evaluator.verify(
        claim=claim_src,
        node=leaf_src,
        sources=(general.urls if general else []),
        additional_instruction="Accept only official PA government URLs; the page must state majority (≥3 of 5) for general cases."
    )


async def verify_documents_requirements(evaluator: Evaluator, parent_node, docs: Optional[DocumentsRequirement]) -> None:
    node = evaluator.add_parallel(
        id="Application_Document_Requirements",
        desc="Required clemency application document types and exception for currently incarcerated applicants (constraint requirements).",
        parent=parent_node,
        critical=True
    )

    # Leaf: Required_Documents_Complete_List (answer provides complete list)
    leaf_list = evaluator.add_leaf(
        id="Required_Documents_Complete_List",
        desc="Provides the complete required application document types: criminal complaint; affidavit of probable cause; criminal information/indictment; final plea or verdict; sentencing order; documentation of financial obligation status.",
        parent=node,
        critical=True
    )
    provided_list = join_list(docs.required_documents if docs else [])
    claim_list = (
        "The answer provides the complete required application document list, including: "
        "criminal complaint; affidavit of probable cause; criminal information or indictment; "
        "final plea or verdict; sentencing order; documentation of financial obligation status."
    )
    await evaluator.verify(
        claim=claim_list,
        node=leaf_list,
        additional_instruction=(
            "Check the answer text: all six items must be present (synonyms acceptable: e.g., 'criminal information/indictment'). "
            "If any item is missing, mark as incorrect. The extracted list from the answer is: "
            f"{provided_list if provided_list else '[no list extracted]'}"
        )
    )

    # Leaf: Incarcerated_Exception (answer states exception)
    leaf_exc = evaluator.add_leaf(
        id="Incarcerated_Exception",
        desc="States that currently incarcerated applicants do not need to submit court documents.",
        parent=node,
        critical=True
    )
    claim_exc = "The answer states that currently incarcerated applicants do not need to submit court documents."
    await evaluator.verify(
        claim=claim_exc,
        node=leaf_exc,
        additional_instruction="Confirm the answer explicitly states this exception. If it does not, mark as incorrect."
    )

    # Leaf: Documents_Official_Source_URL (official PA source supports documents and exception)
    leaf_src = evaluator.add_leaf(
        id="Documents_Official_Source_URL",
        desc="Provides at least one official Pennsylvania government URL supporting the required documents list (and the incarcerated-applicant exception if asserted).",
        parent=node,
        critical=True
    )
    claim_src = (
        "This official Pennsylvania source supports the complete required documents list and, if applicable, the rule that currently incarcerated applicants do not need to submit court documents."
    )
    await evaluator.verify(
        claim=claim_src,
        node=leaf_src,
        sources=(docs.urls if docs else []),
        additional_instruction="Accept only official PA government URLs; the page must clearly list the required documents and note the incarcerated-applicant exception if applicable."
    )


async def verify_final_authority(evaluator: Evaluator, parent_node, final_auth: Optional[FinalAuthority]) -> None:
    node = evaluator.add_parallel(
        id="Final_Clemency_Authority",
        desc="Final decision-making authority after the Board's recommendation.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Governor_Final_Authority (answer identifies Governor as final decision-maker)
    leaf_gov = evaluator.add_leaf(
        id="Governor_Final_Authority",
        desc="Identifies the Governor of Pennsylvania as holding final clemency decision authority and states the Governor may approve or disapprove Board recommendations.",
        parent=node,
        critical=True
    )
    claim_gov = "The answer identifies the Governor of Pennsylvania as the final clemency decision authority and states that the Governor may approve or disapprove Board recommendations."
    await evaluator.verify(
        claim=claim_gov,
        node=leaf_gov,
        additional_instruction="Confirm the answer explicitly names the Governor and the approve/disapprove power. If omitted, mark as incorrect."
    )

    # Leaf: Final_Authority_Official_Source_URL (official PA source supports Governor's authority)
    leaf_src = evaluator.add_leaf(
        id="Final_Authority_Official_Source_URL",
        desc="Provides at least one official Pennsylvania government URL supporting the Governor's final authority over clemency decisions.",
        parent=node,
        critical=True
    )
    claim_src = "This official Pennsylvania source supports that the Governor has final clemency authority and can approve or disapprove Board recommendations."
    await evaluator.verify(
        claim=claim_src,
        node=leaf_src,
        sources=(final_auth.urls if final_auth else []),
        additional_instruction="Accept only official PA government URLs; the page must state the Governor’s final approval/disapproval role."
    )


async def verify_duration(evaluator: Evaluator, parent_node, duration: Optional[ProcessDuration]) -> None:
    node = evaluator.add_parallel(
        id="Typical_Process_Duration",
        desc="Typical duration of the clemency process (constraint requirement).",
        parent=parent_node,
        critical=True
    )

    # Leaf: Duration_Few_Years (answer states typical duration is a few years)
    leaf_dur = evaluator.add_leaf(
        id="Duration_Few_Years",
        desc="States that the complete clemency process typically takes a few years from application to final decision.",
        parent=node,
        critical=True
    )
    claim_dur = "The answer states that the complete Pennsylvania clemency process typically takes a few years from application to final decision."
    await evaluator.verify(
        claim=claim_dur,
        node=leaf_dur,
        additional_instruction="Confirm the answer explicitly uses wording equivalent to 'a few years'. If absent or contradictory, mark as incorrect."
    )

    # Leaf: Duration_Official_Source_URL (official PA source supports typical duration)
    leaf_src = evaluator.add_leaf(
        id="Duration_Official_Source_URL",
        desc="Provides at least one official Pennsylvania government URL supporting the typical duration claim.",
        parent=node,
        critical=True
    )
    claim_src = "This official Pennsylvania source indicates the clemency process typically takes a few years from application to final decision."
    await evaluator.verify(
        claim=claim_src,
        node=leaf_src,
        sources=(duration.urls if duration else []),
        additional_instruction="Accept only official PA government URLs; the page must clearly indicate multi-year duration or equivalent phrasing."
    )


async def verify_process_steps(evaluator: Evaluator, parent_node, steps: Optional[ProcessSteps]) -> None:
    node = evaluator.add_parallel(
        id="Sequential_Process_Steps",
        desc="States the sequential steps of the process (constraint requirement).",
        parent=parent_node,
        critical=True
    )

    # Leaf: Process_Steps_Listed_In_Order (answer lists required steps in order)
    leaf_steps = evaluator.add_leaf(
        id="Process_Steps_Listed_In_Order",
        desc="States the process steps in sequence: application submission → Board review → DOC investigation → merit review → public hearing → Governor's decision.",
        parent=node,
        critical=True
    )
    provided_steps = steps.steps_ordered if steps else []
    provided_str = " → ".join(provided_steps) if provided_steps else "[no steps extracted]"
    expected_str = " → ".join(EXPECTED_PROCESS_SEQUENCE)
    claim_steps = (
        f"The answer lists the process steps in the required sequence: {expected_str}. "
        f"The answer's listed sequence is: {provided_str}."
    )
    await evaluator.verify(
        claim=claim_steps,
        node=leaf_steps,
        additional_instruction=(
            "Check the answer text for these steps in order. Allow reasonable synonyms "
            "(e.g., 'Department of Corrections (DOC)' for 'DOC', 'Governor decision' for 'Governor's decision'). "
            "Pass only if the meaning and order match the required sequence."
        )
    )

    # Leaf: Process_Steps_Official_Source_URL (official PA source supports sequence)
    leaf_src = evaluator.add_leaf(
        id="Process_Steps_Official_Source_URL",
        desc="Provides at least one official Pennsylvania government URL supporting the stated process steps.",
        parent=node,
        critical=True
    )
    claim_src = (
        "This official Pennsylvania source describes the clemency process in the sequence: "
        "application submission → Board review → DOC investigation → merit review → public hearing → Governor's decision."
    )
    await evaluator.verify(
        claim=claim_src,
        node=leaf_src,
        sources=(steps.urls if steps else []),
        additional_instruction="Accept only official PA government URLs; allow step-name synonyms; the page must support the stated sequence."
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
    Evaluate an answer for the Pennsylvania life-sentence clemency process task.
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_pa_clemency(),
        template_class=PAClemencyExtraction,
        extraction_name="pa_clemency_extraction"
    )

    # Build verification tree following rubric
    # Each subgroup is critical and evaluated in parallel under root
    await verify_board_entity(evaluator, root, extracted.board)
    await verify_merit_review_life(evaluator, root, extracted.merit_life)
    await verify_merit_review_general(evaluator, root, extracted.merit_general)
    await verify_hearing_prerequisite(evaluator, root, extracted.hearing_prereq)
    await verify_hearing_life_death(evaluator, root, extracted.hearing_life_death)
    await verify_hearing_general(evaluator, root, extracted.hearing_general)
    await verify_documents_requirements(evaluator, root, extracted.documents)
    await verify_final_authority(evaluator, root, extracted.final_authority)
    await verify_duration(evaluator, root, extracted.duration)
    await verify_process_steps(evaluator, root, extracted.process_steps)

    # Return structured evaluation summary
    return evaluator.get_summary()