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
TASK_ID = "federalist_journalist_and_appropriations_2026"
TASK_DESCRIPTION = """
As of February 13, 2026, identify the full name of the journalist who meets all of the following criteria: 
(1) Currently serves as an elections correspondent at The Federalist, 
(2) Graduated from Fordham University with a degree in International Political Economy, 
(3) Was named a 2025 Publius Fellow at the Claremont Institute, and 
(4) Previously worked at The Daily Caller. 

Additionally, identify: 
(a) Which of the 12 FY 2026 appropriations bills remains without full-year funding as of February 13, 2026 (relying instead on a continuing resolution), 
(b) The deadline date for this continuing resolution, and 
(c) The specific event (including the victim's name, the involved federal agency, and the date) that triggered the political standoff preventing full-year funding for this department.
"""

AS_OF_DATE = "February 13, 2026"
EXPECTED_DEPARTMENT = "Department of Homeland Security"
EXPECTED_EVENT_VICTIM = "Alex Pretti"
EXPECTED_EVENT_AGENCY = "U.S. Customs and Border Protection"
EXPECTED_EVENT_DATE = "January 24, 2026"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class JournalistCriteria(BaseModel):
    """Extraction for the journalist identification criteria and sources."""
    name: Optional[str] = None

    current_title: Optional[str] = None
    current_employer: Optional[str] = None
    sources_current: List[str] = Field(default_factory=list)

    education_school: Optional[str] = None
    education_degree: Optional[str] = None
    sources_education: List[str] = Field(default_factory=list)

    fellowship_institution: Optional[str] = None
    fellowship_program: Optional[str] = None
    fellowship_year: Optional[str] = None
    sources_fellowship: List[str] = Field(default_factory=list)

    previous_employer: Optional[str] = None
    sources_previous: List[str] = Field(default_factory=list)


class FundingContext(BaseModel):
    """Extraction for appropriations bill context and sources."""
    unfunded_bill: Optional[str] = None  # e.g., "Department of Homeland Security" or "Homeland Security appropriations"
    cr_deadline: Optional[str] = None    # date string, e.g., "February 13, 2026"

    trigger_victim: Optional[str] = None
    trigger_agency: Optional[str] = None
    trigger_date: Optional[str] = None

    sources_bill: List[str] = Field(default_factory=list)
    sources_cr: List[str] = Field(default_factory=list)
    sources_event: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_journalist() -> str:
    return """
    Identify the single journalist the answer is referring to and extract the following fields exactly as stated in the answer text. 
    Also extract explicit URL sources cited in the answer for each criterion.

    Fields to extract:
    - name: The full name of the journalist identified in the answer.
    - current_title: The journalist's current role/title (e.g., "elections correspondent").
    - current_employer: The current employer organization (e.g., "The Federalist").
    - sources_current: All URLs that support the current role/employer information.

    - education_school: The university they graduated from (should be "Fordham University").
    - education_degree: The degree or major (should be "International Political Economy").
    - sources_education: All URLs that support the education information.

    - fellowship_institution: The fellowship's institution (should be "Claremont Institute").
    - fellowship_program: The specific fellowship program (should be "Publius Fellow" or "Publius Fellowship").
    - fellowship_year: The fellowship year (should be "2025").
    - sources_fellowship: All URLs that support the fellowship information.

    - previous_employer: The previous employer organization (should be "The Daily Caller").
    - sources_previous: All URLs that support the previous employment information.

    Rules:
    - Extract only what appears in the answer. If a field is not present, set it to null (or empty list for sources).
    - For source fields, extract explicit URLs only (plain or in markdown). Do not infer or invent URLs.
    - If the same URL is listed multiple times, include it once.
    """


def prompt_extract_funding() -> str:
    return f"""
    Extract the appropriations context as described in the answer, focusing on the situation as of {AS_OF_DATE}. 
    Also extract explicit URL sources cited in the answer for each sub-part.

    Fields to extract:
    - unfunded_bill: Exactly which FY 2026 regular appropriations bill (department) remains without full-year funding (e.g., "Department of Homeland Security").
    - cr_deadline: The continuing resolution's deadline date (e.g., "February 13, 2026").
    - trigger_victim: The name of the victim involved in the triggering event (e.g., "Alex Pretti").
    - trigger_agency: The federal agency involved (e.g., "U.S. Customs and Border Protection" or "CBP").
    - trigger_date: The date of the event (e.g., "January 24, 2026").

    Source fields:
    - sources_bill: URLs that support which appropriations bill lacks full-year funding and relies on a CR.
    - sources_cr: URLs that specifically state the CR deadline date.
    - sources_event: URLs that support the details of the triggering event (victim name, agency, date).

    Rules:
    - Extract only URLs explicitly present in the answer.
    - If any information is missing in the answer, set the corresponding field to null (or empty list for sources).
    - Normalize agency naming but do not invent; if the answer uses "CBP", record "CBP". If it uses the full name, record that.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _merge_sources(*lists: List[str]) -> List[str]:
    """Merge lists of URLs and deduplicate while preserving order."""
    seen = set()
    merged = []
    for lst in lists:
        for url in lst:
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_journalist(
    evaluator: Evaluator,
    parent_node,
    jc: JournalistCriteria
) -> None:
    """
    Build the Journalist_Identification subtree and perform verifications.
    """
    jnode = evaluator.add_parallel(
        id="Journalist_Identification",
        desc="Correctly identify the journalist who meets all specified biographical and professional criteria",
        parent=parent_node,
        critical=True
    )

    # Critical gating: name must be provided
    evaluator.add_custom_node(
        result=_non_empty(jc.name),
        id="Journalist_Name_Provided",
        desc="The journalist's full name is provided in the answer",
        parent=jnode,
        critical=True
    )

    # Critical gating: sources for each criterion must be provided
    evaluator.add_custom_node(
        result=len(jc.sources_current) > 0,
        id="Current_Sources_Provided",
        desc="Sources are provided for current role/employer",
        parent=jnode,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(jc.sources_education) > 0,
        id="Education_Sources_Provided",
        desc="Sources are provided for education",
        parent=jnode,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(jc.sources_fellowship) > 0,
        id="Fellowship_Sources_Provided",
        desc="Sources are provided for fellowship",
        parent=jnode,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(jc.sources_previous) > 0,
        id="Previous_Sources_Provided",
        desc="Sources are provided for previous employment",
        parent=jnode,
        critical=True
    )

    # 1) Current employer/title at The Federalist
    current_leaf = evaluator.add_leaf(
        id="Current_Employer",
        desc="The journalist must be an elections correspondent at The Federalist as of February 2026",
        parent=jnode,
        critical=True
    )
    current_claim = f"As of February 2026, {jc.name or ''} serves as an elections correspondent at The Federalist."
    await evaluator.verify(
        claim=current_claim,
        node=current_leaf,
        sources=jc.sources_current,
        additional_instruction=(
            "Confirm the page shows the person is currently with The Federalist in an elections-focused correspondent role. "
            "Allow reasonable variants: 'elections correspondent', 'election(s) correspondent', 'elections reporter'. "
            "If the page indicates a different outlet or role, mark as not supported."
        ),
    )

    # 2) Education at Fordham University, degree in International Political Economy
    edu_leaf = evaluator.add_leaf(
        id="Educational_Background",
        desc="The journalist must have graduated from Fordham University with a degree in International Political Economy",
        parent=jnode,
        critical=True
    )
    edu_claim = f"{jc.name or ''} graduated from Fordham University with a degree in International Political Economy."
    await evaluator.verify(
        claim=edu_claim,
        node=edu_leaf,
        sources=jc.sources_education,
        additional_instruction=(
            "Verify the page states graduation from Fordham University and the degree/major is International Political Economy. "
            "Accept reasonable formats (e.g., BA in International Political Economy, major in International Political Economy)."
        ),
    )

    # 3) 2025 Publius Fellow at the Claremont Institute
    fellow_leaf = evaluator.add_leaf(
        id="Fellowship",
        desc="The journalist must have been named a 2025 Publius Fellow at the Claremont Institute",
        parent=jnode,
        critical=True
    )
    fellow_claim = f"{jc.name or ''} was named a 2025 Publius Fellow at the Claremont Institute."
    await evaluator.verify(
        claim=fellow_claim,
        node=fellow_leaf,
        sources=jc.sources_fellowship,
        additional_instruction=(
            "Confirm the page clearly indicates Publius Fellowship with year 2025 at the Claremont Institute. "
            "Accept variants like 'Publius Fellow (2025)'."
        ),
    )

    # 4) Previously worked at The Daily Caller
    prev_leaf = evaluator.add_leaf(
        id="Previous_Employment",
        desc="The journalist must have previously worked at The Daily Caller",
        parent=jnode,
        critical=True
    )
    prev_claim = f"{jc.name or ''} previously worked at The Daily Caller."
    await evaluator.verify(
        claim=prev_claim,
        node=prev_leaf,
        sources=jc.sources_previous,
        additional_instruction=(
            "Verify prior employment history indicates The Daily Caller as a former workplace. "
            "It may be a prior role or internship; if no association is shown, mark as not supported."
        ),
    )


async def verify_appropriations_and_context(
    evaluator: Evaluator,
    parent_node,
    fc: FundingContext
) -> None:
    """
    Build the Unfunded_Appropriations_Bill leaf and Contextual_Information subtree and perform verifications.
    """
    # Sibling gating node: ensure bill/CR sources exist (critical)
    evaluator.add_custom_node(
        result=(len(fc.sources_bill) > 0 or len(fc.sources_cr) > 0),
        id="Bill_Context_Sources_Provided",
        desc="Sources are provided for which bill remains unfunded / CR status",
        parent=parent_node,
        critical=True
    )

    # Unfunded appropriations bill (critical leaf under Complete_Task)
    bill_leaf = evaluator.add_leaf(
        id="Unfunded_Appropriations_Bill",
        desc="Correctly identify which of the 12 FY 2026 appropriations bills remains without full-year funding as of February 13, 2026",
        parent=parent_node,
        critical=True
    )
    bill_claim = (
        f"As of {AS_OF_DATE}, the FY 2026 Homeland Security appropriations bill does not have full-year funding and "
        f"relies on a continuing resolution."
    )
    await evaluator.verify(
        claim=bill_claim,
        node=bill_leaf,
        sources=_merge_sources(fc.sources_bill, fc.sources_cr),
        additional_instruction=(
            "Confirm the evidence explicitly indicates the Department of Homeland Security (DHS) appropriations bill "
            "was under a continuing resolution and lacked full-year funding as of February 13, 2026. "
            "Accept naming variants like 'Homeland Security bill' or 'DHS appropriations'."
        ),
    )

    # Contextual information subtree
    ctx_node = evaluator.add_parallel(
        id="Contextual_Information",
        desc="Provide accurate details about the deadline and the triggering event for the funding standoff",
        parent=parent_node,
        critical=True
    )

    # Critical gating: ensure CR deadline sources are provided
    evaluator.add_custom_node(
        result=len(fc.sources_cr) > 0,
        id="CR_Sources_Provided",
        desc="Sources are provided for the continuing resolution deadline",
        parent=ctx_node,
        critical=True
    )

    # Critical gating: ensure event sources are provided
    evaluator.add_custom_node(
        result=len(fc.sources_event) > 0,
        id="Event_Sources_Provided",
        desc="Sources are provided for the triggering event details",
        parent=ctx_node,
        critical=True
    )

    # CR deadline leaf
    cr_leaf = evaluator.add_leaf(
        id="CR_Deadline",
        desc="The continuing resolution deadline for the unfunded department must be February 13, 2026",
        parent=ctx_node,
        critical=True
    )
    cr_claim = f"The continuing resolution deadline for the Department of Homeland Security funding is {AS_OF_DATE}."
    await evaluator.verify(
        claim=cr_claim,
        node=cr_leaf,
        sources=fc.sources_cr,
        additional_instruction=(
            "Confirm the evidence shows the CR deadline date is February 13, 2026. "
            "Accept reasonable date formatting variants (e.g., 'Feb. 13, 2026', '2/13/2026')."
        ),
    )

    # Triggering event leaf
    event_leaf = evaluator.add_leaf(
        id="Triggering_Event",
        desc="The event that triggered the political standoff must be the killing of Alex Pretti by CBP agents on January 24, 2026",
        parent=ctx_node,
        critical=True
    )
    event_claim = (
        f"The funding standoff preventing full-year DHS funding was triggered by the killing of {EXPECTED_EVENT_VICTIM} "
        f"by {EXPECTED_EVENT_AGENCY} agents on {EXPECTED_EVENT_DATE}."
    )
    await evaluator.verify(
        claim=event_claim,
        node=event_leaf,
        sources=fc.sources_event,
        additional_instruction=(
            "Verify that the page explicitly supports all three details: victim name (Alex Pretti), agency (CBP/U.S. Customs and Border Protection), "
            "and date (January 24, 2026). If any of these elements is missing or contradicted, mark as not supported."
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
    Evaluate the given answer for the Federalist journalist and appropriations 2026 task.
    """
    # Initialize evaluator and root
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

    # Extract structured information in parallel
    journalist_task = evaluator.extract(
        prompt=prompt_extract_journalist(),
        template_class=JournalistCriteria,
        extraction_name="journalist_criteria",
    )
    funding_task = evaluator.extract(
        prompt=prompt_extract_funding(),
        template_class=FundingContext,
        extraction_name="funding_context",
    )
    jc, fc = await asyncio.gather(journalist_task, funding_task)

    # Create the "Complete_Task" critical node
    complete_node = evaluator.add_parallel(
        id="Complete_Task",
        desc="Correctly identify the journalist, the unfunded appropriations bill, and the relevant contextual details",
        parent=root,
        critical=True
    )

    # Add Ground Truth expectations (for transparency; not used for verification)
    evaluator.add_ground_truth({
        "as_of_date": AS_OF_DATE,
        "expected_unfunded_bill": EXPECTED_DEPARTMENT,
        "expected_cr_deadline": AS_OF_DATE,
        "expected_trigger_event": {
            "victim": EXPECTED_EVENT_VICTIM,
            "agency": EXPECTED_EVENT_AGENCY,
            "date": EXPECTED_EVENT_DATE
        }
    }, gt_type="expected_requirements")

    # Journalist subtree
    await verify_journalist(evaluator, complete_node, jc)

    # Appropriations and contextual information
    await verify_appropriations_and_context(evaluator, complete_node, fc)

    # Return evaluation summary
    return evaluator.get_summary()