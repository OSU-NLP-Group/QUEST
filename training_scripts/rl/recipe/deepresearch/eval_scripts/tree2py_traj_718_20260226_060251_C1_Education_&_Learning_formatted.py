import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "university_president_effective_date_2024_2026"
TASK_DESCRIPTION = (
    "Identify one U.S. university president whose appointment became effective between June 1, 2024, "
    "and July 1, 2026 (inclusive). Provide the president's name, the university, the effective start date "
    "of their presidency, and a verifiable source URL documenting this appointment."
)

DATE_RANGE_START = "2024-06-01"
DATE_RANGE_END = "2026-07-01"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PresidentItem(BaseModel):
    name: Optional[str] = None
    university: Optional[str] = None
    effective_start_date: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class PresidentsExtraction(BaseModel):
    presidents: List[PresidentItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_presidents() -> str:
    return """
    Extract from the answer all entries that describe a person becoming the president of a university and provide:
    - name: the person's full name
    - university: the institution's name
    - effective_start_date: the effective start date when the presidency begins (not the announcement date). If the answer mentions both an announcement date and an effective date, extract the effective date (e.g., phrased as “effective July 1, 2025”, “begins July 1, 2025”, “starts on July 1, 2025”, or “takes office on July 1, 2025”). If no effective date is provided, return null.
    - source_urls: all explicit URLs in the answer that are cited as sources for this entry. Include all URLs associated with this entry. Do not invent URLs.

    Output a JSON object with a single field:
    - presidents: an array of objects, each with the fields above.

    Important:
    - Only extract entries where the person is stated to be (or to become) the president of a university or a university system campus. Ignore roles like provost, dean, chancellor (unless clearly equivalent to 'president' in that institution's terminology), or interim notes unless explicitly called “President”.
    - Keep the text of the effective_start_date exactly as presented in the answer (free-form string).
    - Include every URL mentioned in the answer for that entry under source_urls (markdown links should be resolved to the actual URL).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def select_first_candidate(extraction: PresidentsExtraction) -> PresidentItem:
    """
    Return the first provided president item; if none exist, return an empty placeholder.
    """
    if extraction and extraction.presidents:
        return extraction.presidents[0]
    return PresidentItem()


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root,
    item: PresidentItem,
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Create the main critical node (parallel aggregation)
    top_node = evaluator.add_parallel(
        id="University_President_Identification",
        desc="Identifies a U.S. university president whose appointment became effective between June 1, 2024, and July 1, 2026",
        parent=root,
        critical=True,
    )

    # Create Verifiable Documentation as a critical parallel sub-node,
    # further split into concrete leaf checks to avoid bundling multiple verifications.
    verif_doc_node = evaluator.add_parallel(
        id="Verifiable_Documentation",
        desc="Provides official source URL documenting the appointment and effective date",
        parent=top_node,
        critical=True,
    )

    # 1) Source URL Provided (existence check)
    source_provided = evaluator.add_custom_node(
        result=bool(item.source_urls),
        id="Source_URL_Provided",
        desc="At least one source URL is provided in the answer",
        parent=verif_doc_node,
        critical=True,
    )

    # 2) Source is Official (university/system .edu or .gov; or clearly official announcement page)
    source_official_leaf = evaluator.add_leaf(
        id="Source_Is_Official",
        desc="At least one provided source is an official university or state system/government page",
        parent=verif_doc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This webpage is an official page of a university or university system (e.g., on a .edu domain) or a government/state system domain (e.g., .gov), suitable as an official source.",
        node=source_official_leaf,
        sources=item.source_urls,
        additional_instruction=(
            "You will be given one URL at a time. Consider it 'official' if it is clearly on the institution's own domain "
            "(commonly .edu) or an official state/university system or government site (e.g., .gov or a recognized state "
            "system domain). Do not treat generic news media (.com) as official unless it is the institution's own site."
        ),
    )

    # 3) Source explicitly documents both the appointment as president AND the effective start date
    source_supports_leaf = evaluator.add_leaf(
        id="Source_States_Appointment_And_Effective_Date",
        desc="A provided source explicitly states the appointment as president and the effective start date",
        parent=verif_doc_node,
        critical=True,
    )
    person_disp = item.name or "(missing name)"
    univ_disp = item.university or "(missing university)"
    date_disp = item.effective_start_date or "(missing date)"
    await evaluator.verify(
        claim=(
            f"This webpage explicitly documents that {person_disp} was appointed or named as president of {univ_disp} "
            f"and it provides the effective start date as {date_disp} (or equivalent wording like 'effective', 'begins', "
            f"'starts', 'takes office on')."
        ),
        node=source_supports_leaf,
        sources=item.source_urls,
        additional_instruction=(
            "Support requires BOTH: (1) explicit mention of the person being the university president (or becoming president), "
            "(2) an explicit effective start date. If the page lacks the effective date or only mentions selection/announcement "
            "without the effective date, then this is NOT supported."
        ),
    )

    # Institution Type: The institution is a university (higher education institution)
    inst_type_leaf = evaluator.add_leaf(
        id="Institution_Type",
        desc="The identified institution is a university (higher education institution)",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{univ_disp} is a university-level higher education institution.",
        node=inst_type_leaf,
        sources=item.source_urls,
        additional_instruction=(
            "Focus on whether the institution is a university (or a constituent campus of a university). "
            "Evidence can include the page referring to it as a 'University' or a campus of a university system. "
            "Do not accept K-12 schools or non-academic organizations."
        ),
    )

    # U.S. Location: The university is located in the United States
    us_loc_leaf = evaluator.add_leaf(
        id="US_Location",
        desc="The university is located in the United States",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{univ_disp} is located in the United States.",
        node=us_loc_leaf,
        sources=item.source_urls,
        additional_instruction=(
            "Use information from the page (including headers/footers) to determine U.S. location. "
            "University press/announcement pages often indicate the state/city or clearly imply the U.S. context."
        ),
    )

    # Start date within range: between 2024-06-01 and 2026-07-01 inclusive
    date_range_leaf = evaluator.add_leaf(
        id="Start_Date_Within_Range",
        desc="The president's effective start date is between June 1, 2024, and July 1, 2026 (inclusive)",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"According to the provided source(s), the effective start date for {person_disp} as president of {univ_disp} "
            f"is {date_disp}, and this date falls between {DATE_RANGE_START} and {DATE_RANGE_END} inclusive."
        ),
        node=date_range_leaf,
        sources=item.source_urls,
        additional_instruction=(
            "Consider only the effective start date (not announcement/selection dates). "
            "You must check that the effective date lies on or after 2024-06-01 and on or before 2026-07-01. "
            "Allow reasonable date format variations (e.g., 'July 1, 2025')."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the U.S. university president effective date task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level is parallel; children are critical
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

    # Record date range info for transparency
    evaluator.add_custom_info(
        info={"inclusive_range_start": DATE_RANGE_START, "inclusive_range_end": DATE_RANGE_END},
        info_type="constraints",
        info_name="date_range_constraints",
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_presidents(),
        template_class=PresidentsExtraction,
        extraction_name="president_candidates",
    )

    # Select the first candidate (the rubric requires one)
    candidate = select_first_candidate(extracted)

    # Optionally record the selected candidate for debugging transparency
    evaluator.add_custom_info(
        info={
            "selected_name": candidate.name,
            "selected_university": candidate.university,
            "selected_effective_start_date": candidate.effective_start_date,
            "selected_source_urls": candidate.source_urls,
        },
        info_type="selection",
        info_name="selected_candidate",
    )

    # Build tree and run verifications
    await build_and_verify_tree(evaluator, root, candidate)

    # Return summary result
    return evaluator.get_summary()