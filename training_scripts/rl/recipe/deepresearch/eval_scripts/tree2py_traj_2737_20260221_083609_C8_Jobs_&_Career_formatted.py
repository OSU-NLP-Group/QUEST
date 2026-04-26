import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "three_us_university_presidents_2022_2026"
TASK_DESCRIPTION = (
    "Identify three individuals who were appointed as university presidents in the United States between July 2022 and July 2026 (inclusive). "
    "Each individual must meet one of the following three distinct criteria:\n\n"
    "First President: An individual who became the first African American president of a university in Michigan that was founded in the 19th century (1801-1900).\n\n"
    "Second President: An individual who served as a business school dean for at least 10 years before being appointed as a university president.\n\n"
    "Third President: An individual who served as a university president or chancellor at one institution before being appointed as president at a different university.\n\n"
    "For each of the three individuals, provide:\n"
    "1. Their full name\n"
    "2. Their previous position and institution\n"
    "3. Their new position and institution\n"
    "4. The appointment date or start date of their new presidency\n"
    "5. A URL reference that verifies this information"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BillPinkInfo(BaseModel):
    name: Optional[str] = None
    new_title: Optional[str] = None  # Expected to include "19th President" or equivalent
    new_institution: Optional[str] = None  # Ferris State University
    start_date: Optional[str] = None  # Expected: July 11, 2022 (or July 2022)
    previous_title: Optional[str] = None  # President
    previous_institution: Optional[str] = None  # Grand Rapids Community College
    ferris_founded_year: Optional[str] = None  # 1884
    ferris_location: Optional[str] = None  # Big Rapids, Michigan
    urls: List[str] = Field(default_factory=list)


class ScottBeardsleyInfo(BaseModel):
    name: Optional[str] = None  # Scott C. Beardsley
    darden_dean_title: Optional[str] = None  # 9th Dean of the Darden School of Business
    darden_start_date: Optional[str] = None  # August 1, 2015
    uv_president_ordinal: Optional[str] = None  # 10th
    uv_appoint_date: Optional[str] = None  # December 19, 2025
    uv_term_start_date: Optional[str] = None  # January 1, 2026
    urls: List[str] = Field(default_factory=list)


class KentSyverudInfo(BaseModel):
    name: Optional[str] = None  # Kent Syverud
    syracuse_chancellor_title: Optional[str] = None  # 12th Chancellor and President (Syracuse University)
    syracuse_start_date: Optional[str] = None  # January 2014
    umich_elected_date: Optional[str] = None  # January 13, 2026
    umich_term_begin_by: Optional[str] = None  # by July 1, 2026
    urls: List[str] = Field(default_factory=list)


class AppointmentsExtraction(BaseModel):
    bill_pink: Optional[BillPinkInfo] = None
    scott_beardsley: Optional[ScottBeardsleyInfo] = None
    kent_syverud: Optional[KentSyverudInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract structured details for three specific individuals if they appear in the answer. If any field is not explicitly present in the answer, set it to null and keep URL arrays empty if no URLs are given.

Target 1 — Bill Pink (Ferris State University):
- name
- new_title (e.g., "19th President")
- new_institution (expected: "Ferris State University")
- start_date (e.g., "July 11, 2022")
- previous_title (e.g., "President")
- previous_institution (e.g., "Grand Rapids Community College")
- ferris_founded_year (expected: "1884" if stated)
- ferris_location (e.g., "Big Rapids, Michigan")
- urls: all URLs cited that support any of the Bill Pink/Ferris facts

Target 2 — Scott C. Beardsley (University of Virginia):
- name
- darden_dean_title (e.g., "9th Dean of the Darden School of Business")
- darden_start_date (e.g., "August 1, 2015")
- uv_president_ordinal (e.g., "10th")
- uv_appoint_date (e.g., "December 19, 2025")
- uv_term_start_date (e.g., "January 1, 2026")
- urls: all URLs cited that support any of the Beardsley/UVA/Darden facts

Target 3 — Kent Syverud (University of Michigan):
- name
- syracuse_chancellor_title (e.g., "12th Chancellor and President")
- syracuse_start_date (e.g., "January 2014")
- umich_elected_date (e.g., "January 13, 2026")
- umich_term_begin_by (e.g., "by July 1, 2026")
- urls: all URLs cited that support any of the Syverud/Syracuse/UMich facts

Return a JSON object with keys: bill_pink, scott_beardsley, kent_syverud. Each is an object with the fields listed. If the answer uses different individuals, still return the above keys with nulls for fields and empty url arrays.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls or []


# --------------------------------------------------------------------------- #
# Verification functions for each constrained individual                      #
# --------------------------------------------------------------------------- #
async def verify_bill_pink(evaluator: Evaluator, parent) -> None:
    """
    Verify constrained First President: Bill Pink at Ferris State University, with specified facts and evidence.
    """
    # Extracted block
    extraction: AppointmentsExtraction = evaluator._extraction_results[-1]["result"]  # type: ignore
    # Reconstruct model back to object if needed; but we recorded dict. Instead, re-extract from evaluator context:
    # Better: ask evaluator to find last extraction object? Not available. We'll instead pass the object from caller.
    # To keep function self-contained, we will expect caller to pass the actual object instead.
    pass


# We'll implement the verification functions with a proper signature that accepts the extracted objects directly.

async def verify_bill_pink_with_info(evaluator: Evaluator, parent, info: Optional[BillPinkInfo]) -> None:
    group = evaluator.add_parallel(
        id="Bill_Pink_Ferris_State",
        desc="Constrained First President: Bill Pink at Ferris State University with specified facts and evidence.",
        parent=parent,
        critical=True,
    )

    urls = _safe_urls(info.urls if info else None)

    # URL evidence existence (critical precondition)
    evaluator.add_custom_node(
        result=bool(urls),
        id="Bill_Pink_URL_Evidence_Provided",
        desc="Provides at least one URL that supports the above Bill Pink/Ferris State facts.",
        parent=group,
        critical=True,
    )

    # Name check
    node_name = evaluator.add_leaf(
        id="Name_Is_Bill_Pink",
        desc="The individual’s full name is Bill Pink.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="The individual's full name is Bill Pink.",
        node=node_name,
        sources=urls,
        additional_instruction="Use the provided sources to confirm the appointed president is named Bill Pink.",
    )

    # New position title + institution
    node_position = evaluator.add_leaf(
        id="New_Position_Ferris_19th_President",
        desc="States the new position/institution as 19th President of Ferris State University.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Bill Pink was appointed the 19th President of Ferris State University.",
        node=node_position,
        sources=urls,
        additional_instruction="Accept minor phrasing variants (e.g., 'named' or 'selected'), but the ordinal '19th' and institution 'Ferris State University' must be clearly supported.",
    )

    # Start date (July 11, 2022)
    node_start = evaluator.add_leaf(
        id="Start_Date_July_11_2022",
        desc="States the presidency start date as July 11, 2022 (July 2022).",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Bill Pink's presidency at Ferris State University began on July 11, 2022.",
        node=node_start,
        sources=urls,
        additional_instruction="If the source mentions a start date in July 2022 that is clearly July 11, count it as supported. Slight date format differences are ok.",
    )

    # Previous position (GRCC President)
    node_prev = evaluator.add_leaf(
        id="Previous_Position_GRCC_President",
        desc="States the previous position/institution as President of Grand Rapids Community College.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Before Ferris State, Bill Pink served as President of Grand Rapids Community College.",
        node=node_prev,
        sources=urls,
        additional_instruction="Look for 'Grand Rapids Community College' or 'GRCC' and confirm his role as its president.",
    )

    # First African American president at FSU
    node_first_aa = evaluator.add_leaf(
        id="First_African_American_President_FSU",
        desc="States that Bill Pink is the first African American president in Ferris State University’s history.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Bill Pink is the first African American president in Ferris State University’s history.",
        node=node_first_aa,
        sources=urls,
        additional_instruction="The claim must be explicitly stated or unmistakably clear on the provided source(s).",
    )

    # Ferris founded 1884
    node_founded = evaluator.add_leaf(
        id="Ferris_Founded_1884",
        desc="States that Ferris State University was founded in 1884.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Ferris State University was founded in 1884.",
        node=node_founded,
        sources=urls,
        additional_instruction="If an official or credible source among the provided URLs states the founding year as 1884, mark as supported.",
    )

    # Ferris location
    node_location = evaluator.add_leaf(
        id="Ferris_Located_Big_Rapids_Michigan",
        desc="States that Ferris State University is located in Big Rapids, Michigan.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Ferris State University is located in Big Rapids, Michigan.",
        node=node_location,
        sources=urls,
        additional_instruction="The source should clearly indicate the campus or institution is in Big Rapids, Michigan.",
    )


async def verify_scott_beardsley_with_info(evaluator: Evaluator, parent, info: Optional[ScottBeardsleyInfo]) -> None:
    group = evaluator.add_parallel(
        id="Scott_C_Beardsley_UVA",
        desc="Constrained Second President: Scott C. Beardsley at University of Virginia with specified facts and evidence.",
        parent=parent,
        critical=True,
    )

    urls = _safe_urls(info.urls if info else None)

    # URL evidence existence (critical precondition)
    evaluator.add_custom_node(
        result=bool(urls),
        id="Scott_Beardsley_URL_Evidence_Provided",
        desc="Provides at least one URL that supports the above Scott C. Beardsley/UVA/Darden facts.",
        parent=group,
        critical=True,
    )

    # Name check
    node_name = evaluator.add_leaf(
        id="Name_Is_Scott_C_Beardsley",
        desc="The individual’s full name is Scott C. Beardsley.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="The individual's full name is Scott C. Beardsley.",
        node=node_name,
        sources=urls,
        additional_instruction="Use the provided sources (UVA/Darden announcements, etc.) to confirm the full name 'Scott C. Beardsley'.",
    )

    # Darden dean with start date
    node_darden = evaluator.add_leaf(
        id="Previously_9th_Dean_Darden_Aug_1_2015",
        desc="States prior role as the 9th Dean of the Darden School of Business, with start date August 1, 2015.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Scott C. Beardsley became the 9th dean of the University of Virginia Darden School of Business on August 1, 2015.",
        node=node_darden,
        sources=urls,
        additional_instruction="The source should show both the ordinal (9th dean) and the start date (Aug 1, 2015) or equivalent phrasing.",
    )

    # Dean for at least 10 years before presidency
    node_ten_years = evaluator.add_leaf(
        id="Dean_Service_At_Least_10_Years_Before_Presidency",
        desc="States that he served as Darden dean for approximately/at least 10 years before his UVA presidency.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Scott C. Beardsley served as Darden School of Business dean for at least 10 years before being appointed as the University of Virginia president.",
        node=node_ten_years,
        sources=urls,
        additional_instruction=(
            "If the sources provide Darden dean start date in 2015 and his UVA presidential appointment in December 2025 or term start in January 2026, "
            "you may deduce that he served ~10 years. Explicit mention of '10 years' is not required if dates clearly imply it."
        ),
    )

    # Appointed 10th President, Dec 19, 2025
    node_appoint = evaluator.add_leaf(
        id="Appointed_10th_President_UVA_Dec_19_2025",
        desc="States he was appointed as the 10th President of the University of Virginia on December 19, 2025.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Scott C. Beardsley was appointed as the 10th President of the University of Virginia on December 19, 2025.",
        node=node_appoint,
        sources=urls,
        additional_instruction="The source should make clear the ordinal ('10th President') and appointment date (Dec 19, 2025).",
    )

    # Term began Jan 1, 2026
    node_term = evaluator.add_leaf(
        id="Term_Began_Jan_1_2026",
        desc="States his term as UVA President began on January 1, 2026.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Scott C. Beardsley's term as University of Virginia president began on January 1, 2026.",
        node=node_term,
        sources=urls,
        additional_instruction="Verify that the source states his term start date as January 1, 2026 (or an equivalent phrasing indicating that start date).",
    )


async def verify_kent_syverud_with_info(evaluator: Evaluator, parent, info: Optional[KentSyverudInfo]) -> None:
    group = evaluator.add_parallel(
        id="Kent_Syverud_University_of_Michigan",
        desc="Constrained Third President: Kent Syverud elected University of Michigan president, with specified facts and evidence.",
        parent=parent,
        critical=True,
    )

    urls = _safe_urls(info.urls if info else None)

    # URL evidence existence (critical precondition)
    evaluator.add_custom_node(
        result=bool(urls),
        id="Kent_Syverud_URL_Evidence_Provided",
        desc="Provides at least one URL that supports the above Kent Syverud/Syracuse/University of Michigan facts.",
        parent=group,
        critical=True,
    )

    # Name check
    node_name = evaluator.add_leaf(
        id="Name_Is_Kent_Syverud",
        desc="The individual’s full name is Kent Syverud.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="The individual's full name is Kent Syverud.",
        node=node_name,
        sources=urls,
        additional_instruction="Use the provided sources (UMich, Syracuse, press releases) to confirm the full name 'Kent Syverud'.",
    )

    # Syracuse Chancellor & President since Jan 2014
    node_syr = evaluator.add_leaf(
        id="Current_Syracuse_Chancellor_And_President_Since_Jan_2014",
        desc="States he currently serves as the 12th Chancellor and President of Syracuse University since January 2014.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Kent Syverud has served as the 12th Chancellor and President of Syracuse University since January 2014.",
        node=node_syr,
        sources=urls,
        additional_instruction="Accept equivalent phrasing indicating he has been Chancellor/President of Syracuse since January 2014.",
    )

    # Elected 16th UMich President on Jan 13, 2026
    node_elected = evaluator.add_leaf(
        id="Elected_16th_President_UMich_Jan_13_2026",
        desc="States he was elected as the 16th President of the University of Michigan on January 13, 2026.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Kent Syverud was elected as the 16th President of the University of Michigan on January 13, 2026.",
        node=node_elected,
        sources=urls,
        additional_instruction="Confirm both the ordinal ('16th President') and the election date (Jan 13, 2026).",
    )

    # Term begins by July 1, 2026
    node_term = evaluator.add_leaf(
        id="Term_Begins_By_July_1_2026",
        desc="States his University of Michigan presidency term will begin by July 1, 2026.",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Kent Syverud's term as University of Michigan president will begin by July 1, 2026.",
        node=node_term,
        sources=urls,
        additional_instruction="Allow 'by July 1, 2026', 'effective July 1, 2026', or equivalent formulations meaning no later than July 1, 2026.",
    )

    # Moved from one presidency/chancellorship to another at different institution
    node_move = evaluator.add_leaf(
        id="Moved_From_One_Presidency_To_Another_Different_Institution",
        desc="States that he moved from being president/chancellor at Syracuse University to being president at a different university (University of Michigan).",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim="Kent Syverud moved from serving as Chancellor/President at Syracuse University to becoming President at the University of Michigan.",
        node=node_move,
        sources=urls,
        additional_instruction="Treat 'Chancellor and President' at Syracuse as equivalent to serving as a university's chief executive. The claim should be clear from the sources.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the three constrained U.S. university president appointments (July 2022–July 2026).
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

    # Extract structured information
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AppointmentsExtraction,
        extraction_name="appointments_extraction",
    )

    # Add a critical parent node for the overall task
    main = evaluator.add_parallel(
        id="Three_University_President_Appointments",
        desc="Provide the three required US university president appointments (as constrained) and required supporting details/URLs.",
        parent=root,
        critical=True,
    )

    # Add ground truth (expected targets and key facts) for transparency
    evaluator.add_ground_truth({
        "expected_first_president": {
            "name": "Bill Pink",
            "institution": "Ferris State University",
            "start_date": "July 11, 2022",
            "previous_institution": "Grand Rapids Community College",
            "first_african_american": True,
            "founded_year": "1884",
            "location": "Big Rapids, Michigan",
        },
        "expected_second_president": {
            "name": "Scott C. Beardsley",
            "darden_start_date": "August 1, 2015",
            "uva_appointment_date": "December 19, 2025",
            "uva_term_start_date": "January 1, 2026",
            "uva_ordinal": "10th",
            "criterion": "Served as a business school dean for ~10 years before presidency"
        },
        "expected_third_president": {
            "name": "Kent Syverud",
            "syracuse_start_date": "January 2014",
            "umich_elected_date": "January 13, 2026",
            "umich_term_begin_by": "July 1, 2026",
            "criterion": "Previously president/chancellor at one institution, then president at a different university"
        }
    })

    # Verify each constrained individual under the critical parent
    await verify_bill_pink_with_info(evaluator, main, extraction.bill_pink if extraction else None)
    await verify_scott_beardsley_with_info(evaluator, main, extraction.scott_beardsley if extraction else None)
    await verify_kent_syverud_with_info(evaluator, main, extraction.kent_syverud if extraction else None)

    # Return evaluation summary
    return evaluator.get_summary()