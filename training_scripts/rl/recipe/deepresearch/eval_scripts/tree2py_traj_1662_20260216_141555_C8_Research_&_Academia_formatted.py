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
TASK_ID = "univ_rankings_report_2026"
TASK_DESCRIPTION = (
    "A research administrator is preparing a comprehensive report comparing university ranking methodologies and needs "
    "to gather specific information for an upcoming academic conference presentation. The report requires the following details:\n\n"
    "1. Times Higher Education (THE) World University Rankings 2026 Methodology: Provide the exact percentage weight for each of the five pillars "
    "(Teaching, Research Environment, Research Quality, International outlook, and Industry), the total number of performance metrics used in the rankings, "
    "and the date when the methodology was published.\n\n"
    "2. QS World University Rankings 2026 Methodology: Provide the exact percentage weight for each of the five lenses "
    "(Research & Discovery, Employability & Outcomes, Learning Experience, Global Engagement, and Sustainability), the specific weight of the Academic Reputation indicator, "
    "the specific weight of the Citations per Faculty indicator, the minimum number of joint papers required over a five-year period for a partnership to be considered "
    "\"sustained\" in the International Research Network indicator, and the date when the rankings were published.\n\n"
    "3. AERA 2026 Annual Meeting Details: Provide the city and state where the conference will be held, the start date and end date of the conference, and the name of the venue.\n\n"
    "4. NIH Grant Application Requirement: Provide the earliest application due date (on or after which date) when all key personnel must have completed Research Security Training for NIH grant submissions.\n\n"
    "Please provide all requested information with supporting reference URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class THEMethodology(BaseModel):
    methodology_urls: List[str] = Field(default_factory=list)
    teaching_weight: Optional[str] = None
    research_environment_weight: Optional[str] = None
    research_quality_weight: Optional[str] = None
    international_outlook_weight: Optional[str] = None
    industry_weight: Optional[str] = None
    total_metrics: Optional[str] = None
    publication_date: Optional[str] = None


class QSMethodology(BaseModel):
    methodology_urls: List[str] = Field(default_factory=list)
    research_discovery_weight: Optional[str] = None
    employability_outcomes_weight: Optional[str] = None
    learning_experience_weight: Optional[str] = None
    global_engagement_weight: Optional[str] = None
    sustainability_weight: Optional[str] = None
    academic_reputation_weight: Optional[str] = None
    citations_per_faculty_weight: Optional[str] = None
    irn_sustained_threshold: Optional[str] = None
    publication_date: Optional[str] = None


class AERAInfo(BaseModel):
    info_urls: List[str] = Field(default_factory=list)
    city: Optional[str] = None
    state: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    venue: Optional[str] = None


class NIHRequirement(BaseModel):
    info_urls: List[str] = Field(default_factory=list)
    earliest_due_date: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_the() -> str:
    return """
    Extract the Times Higher Education (THE) World University Rankings 2026 methodology details exactly as stated in the answer.

    Return a JSON with:
    - methodology_urls: an array of all URLs in the answer that are cited for THE WUR 2026 methodology.
    - teaching_weight: the Teaching pillar weight as written (include the % sign if present, e.g., "29.5%").
    - research_environment_weight: the Research Environment pillar weight as written (e.g., "29.0%").
    - research_quality_weight: the Research Quality pillar weight as written (e.g., "30.0%").
    - international_outlook_weight: the International outlook pillar weight as written (e.g., "7.5%").
    - industry_weight: the Industry pillar weight as written (e.g., "4.0%").
    - total_metrics: the total number of performance metrics used as written (e.g., "18").
    - publication_date: the methodology publication date as written in the answer (e.g., "September 22, 2025").

    Rules:
    - Only extract values explicitly appearing in the answer.
    - If a field is missing, set it to null (or empty array for URLs).
    - For URLs, include every explicit URL mentioned that supports THE methodology; extract full URLs.
    """


def prompt_extract_qs() -> str:
    return """
    Extract the QS World University Rankings 2026 methodology details exactly as stated in the answer.

    Return a JSON with:
    - methodology_urls: an array of all URLs in the answer that are cited for QS WUR 2026 methodology or rankings page.
    - research_discovery_weight: the Research & Discovery lens weight as written (e.g., "50%").
    - employability_outcomes_weight: the Employability & Outcomes lens weight as written (e.g., "20%").
    - learning_experience_weight: the Learning Experience lens weight as written (e.g., "10%").
    - global_engagement_weight: the Global Engagement lens weight as written (e.g., "15%").
    - sustainability_weight: the Sustainability lens weight as written (e.g., "5%").
    - academic_reputation_weight: the Academic Reputation indicator weight as written (e.g., "30%").
    - citations_per_faculty_weight: the Citations per Faculty indicator weight as written (e.g., "20%").
    - irn_sustained_threshold: the sustained-partnership threshold for the IRN indicator as written (e.g., "3 or more joint papers over five years").
    - publication_date: the date when the QS 2026 rankings were published as written in the answer (e.g., "June 19, 2025").

    Rules:
    - Only extract values explicitly appearing in the answer.
    - If a field is missing, set it to null (or empty array for URLs).
    - For URLs, include every explicit URL mentioned that supports the QS methodology or rankings publication; extract full URLs.
    """


def prompt_extract_aera() -> str:
    return """
    Extract the AERA 2026 Annual Meeting details exactly as stated in the answer.

    Return a JSON with:
    - info_urls: an array of all URLs cited for AERA 2026 Annual Meeting information.
    - city: the city where the meeting will be held (e.g., "Los Angeles").
    - state: the U.S. state where the meeting will be held (e.g., "California" or "CA").
    - start_date: the start date as written (e.g., "April 8, 2026").
    - end_date: the end date as written (e.g., "April 12, 2026").
    - venue: the venue name as written (e.g., "Los Angeles Convention Center").

    Rules:
    - Only extract values explicitly appearing in the answer.
    - If a field is missing, set it to null (or empty array for URLs).
    - For URLs, include every explicit URL mentioned that supports AERA 2026 details; extract full URLs.
    """


def prompt_extract_nih() -> str:
    return """
    Extract the NIH grant application Research Security Training requirement details exactly as stated in the answer.

    Return a JSON with:
    - info_urls: an array of all URLs cited for NIH research security training requirement.
    - earliest_due_date: the earliest application due date on or after which all key personnel must have completed Research Security Training, as written (e.g., "May 25, 2026").

    Rules:
    - Only extract values explicitly appearing in the answer.
    - If a field is missing, set it to null (or empty array for URLs).
    - For URLs, include every explicit URL mentioned that supports the NIH requirement; extract full URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ensure_list(urls: Optional[List[str]]) -> List[str]:
    return urls if isinstance(urls, list) else []


def _city_state_str(city: Optional[str], state: Optional[str]) -> str:
    city_part = city or ""
    state_part = state or ""
    if city_part and state_part:
        return f"{city_part}, {state_part}"
    return (city_part or state_part or "").strip()


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_the_section(evaluator: Evaluator, parent, the: THEMethodology) -> None:
    the_node = evaluator.add_parallel(
        id="THE_Rankings_Information",
        desc="Complete information about THE World University Rankings 2026 methodology",
        parent=parent,
        critical=False
    )

    # 1) Methodology URL validity
    the_url_leaf = evaluator.add_leaf(
        id="THE_Methodology_URL",
        desc="Provide valid reference URL for THE World University Rankings 2026 methodology",
        parent=the_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page explains the Times Higher Education (THE) World University Rankings 2026 methodology.",
        node=the_url_leaf,
        sources=_ensure_list(the.methodology_urls),
        additional_instruction="Verify that the page is by THE (or clearly cites THE) and is the methodology for the 2026 World University Rankings."
    )

    # 2) Teaching weight
    the_teach_leaf = evaluator.add_leaf(
        id="THE_Teaching_Weight",
        desc="Provide the exact percentage weight (29.5%) for the Teaching pillar in THE World University Rankings 2026",
        parent=the_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the THE World University Rankings 2026 methodology, the Teaching pillar weight is {the.teaching_weight}.",
        node=the_teach_leaf,
        sources=_ensure_list(the.methodology_urls),
        additional_instruction="Check that the page states the Teaching pillar weight for 2026 exactly. Allow minor formatting (e.g., 'percent' vs '%')."
    )

    # 3) Research Environment weight
    the_re_env_leaf = evaluator.add_leaf(
        id="THE_Research_Environment_Weight",
        desc="Provide the exact percentage weight (29.0%) for the Research Environment pillar in THE World University Rankings 2026",
        parent=the_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the THE World University Rankings 2026 methodology, the Research Environment pillar weight is {the.research_environment_weight}.",
        node=the_re_env_leaf,
        sources=_ensure_list(the.methodology_urls),
        additional_instruction="Verify the Research Environment pillar weight for 2026 as stated on the methodology page."
    )

    # 4) Research Quality weight
    the_re_qual_leaf = evaluator.add_leaf(
        id="THE_Research_Quality_Weight",
        desc="Provide the exact percentage weight (30.0%) for the Research Quality pillar in THE World University Rankings 2026",
        parent=the_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the THE World University Rankings 2026 methodology, the Research Quality pillar weight is {the.research_quality_weight}.",
        node=the_re_qual_leaf,
        sources=_ensure_list(the.methodology_urls),
        additional_instruction="Verify the Research Quality pillar weight for 2026 as stated on the methodology page."
    )

    # 5) International outlook weight
    the_intl_leaf = evaluator.add_leaf(
        id="THE_International_Outlook_Weight",
        desc="Provide the exact percentage weight (7.5%) for the International outlook pillar in THE World University Rankings 2026",
        parent=the_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the THE World University Rankings 2026 methodology, the International outlook pillar weight is {the.international_outlook_weight}.",
        node=the_intl_leaf,
        sources=_ensure_list(the.methodology_urls),
        additional_instruction="Verify the International outlook pillar weight for 2026."
    )

    # 6) Industry weight
    the_industry_leaf = evaluator.add_leaf(
        id="THE_Industry_Weight",
        desc="Provide the exact percentage weight (4.0%) for the Industry pillar in THE World University Rankings 2026",
        parent=the_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the THE World University Rankings 2026 methodology, the Industry pillar weight is {the.industry_weight}.",
        node=the_industry_leaf,
        sources=_ensure_list(the.methodology_urls),
        additional_instruction="Verify the Industry pillar weight for 2026."
    )

    # 7) Total performance metrics
    the_metrics_leaf = evaluator.add_leaf(
        id="THE_Total_Metrics",
        desc="Provide the total number of performance metrics (18) used in THE World University Rankings 2026",
        parent=the_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The THE World University Rankings 2026 uses {the.total_metrics} performance metrics.",
        node=the_metrics_leaf,
        sources=_ensure_list(the.methodology_urls),
        additional_instruction="Confirm the exact count of performance metrics/indicators for the 2026 methodology."
    )

    # 8) Publication date
    the_pub_date_leaf = evaluator.add_leaf(
        id="THE_Publication_Date",
        desc="Provide the publication date (September 22, 2025) of THE World University Rankings 2026 methodology",
        parent=the_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The THE World University Rankings 2026 methodology was published on {the.publication_date}.",
        node=the_pub_date_leaf,
        sources=_ensure_list(the.methodology_urls),
        additional_instruction="Verify the stated publication date of the 2026 methodology from the referenced page."
    )


async def build_qs_section(evaluator: Evaluator, parent, qs: QSMethodology) -> None:
    qs_node = evaluator.add_parallel(
        id="QS_Rankings_Information",
        desc="Complete information about QS World University Rankings 2026 methodology",
        parent=parent,
        critical=False
    )

    # 1) Methodology URL validity
    qs_url_leaf = evaluator.add_leaf(
        id="QS_Methodology_URL",
        desc="Provide valid reference URL for QS World University Rankings 2026 methodology",
        parent=qs_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page explains the QS World University Rankings 2026 methodology or is the official rankings page for the 2026 release.",
        node=qs_url_leaf,
        sources=_ensure_list(qs.methodology_urls),
        additional_instruction="Verify that the page is by QS and covers the 2026 WUR methodology or official 2026 rankings information."
    )

    # 2) Lens weights
    qs_rd_leaf = evaluator.add_leaf(
        id="QS_Research_Discovery_Weight",
        desc="Provide the exact percentage weight (50%) for the Research & Discovery lens in QS World University Rankings 2026",
        parent=qs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the QS World University Rankings 2026, the Research & Discovery lens weight is {qs.research_discovery_weight}.",
        node=qs_rd_leaf,
        sources=_ensure_list(qs.methodology_urls),
        additional_instruction="Confirm the Research & Discovery lens weight for 2026."
    )

    qs_emp_leaf = evaluator.add_leaf(
        id="QS_Employability_Outcomes_Weight",
        desc="Provide the exact percentage weight (20%) for the Employability & Outcomes lens in QS World University Rankings 2026",
        parent=qs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the QS World University Rankings 2026, the Employability & Outcomes lens weight is {qs.employability_outcomes_weight}.",
        node=qs_emp_leaf,
        sources=_ensure_list(qs.methodology_urls),
        additional_instruction="Confirm the Employability & Outcomes lens weight for 2026."
    )

    qs_learn_leaf = evaluator.add_leaf(
        id="QS_Learning_Experience_Weight",
        desc="Provide the exact percentage weight (10%) for the Learning Experience lens in QS World University Rankings 2026",
        parent=qs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the QS World University Rankings 2026, the Learning Experience lens weight is {qs.learning_experience_weight}.",
        node=qs_learn_leaf,
        sources=_ensure_list(qs.methodology_urls),
        additional_instruction="Confirm the Learning Experience lens weight for 2026."
    )

    qs_global_leaf = evaluator.add_leaf(
        id="QS_Global_Engagement_Weight",
        desc="Provide the exact percentage weight (15%) for the Global Engagement lens in QS World University Rankings 2026",
        parent=qs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the QS World University Rankings 2026, the Global Engagement lens weight is {qs.global_engagement_weight}.",
        node=qs_global_leaf,
        sources=_ensure_list(qs.methodology_urls),
        additional_instruction="Confirm the Global Engagement lens weight for 2026."
    )

    qs_sust_leaf = evaluator.add_leaf(
        id="QS_Sustainability_Weight",
        desc="Provide the exact percentage weight (5%) for the Sustainability lens in QS World University Rankings 2026",
        parent=qs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the QS World University Rankings 2026, the Sustainability lens weight is {qs.sustainability_weight}.",
        node=qs_sust_leaf,
        sources=_ensure_list(qs.methodology_urls),
        additional_instruction="Confirm the Sustainability lens weight for 2026."
    )

    # 3) Indicator weights
    qs_acad_rep_leaf = evaluator.add_leaf(
        id="QS_Academic_Reputation_Weight",
        desc="Provide the exact percentage weight (30%) for the Academic Reputation indicator in QS World University Rankings 2026",
        parent=qs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the QS World University Rankings 2026, the Academic Reputation indicator weight is {qs.academic_reputation_weight}.",
        node=qs_acad_rep_leaf,
        sources=_ensure_list(qs.methodology_urls),
        additional_instruction="Confirm the Academic Reputation indicator weight for 2026."
    )

    qs_cpf_leaf = evaluator.add_leaf(
        id="QS_Citations_per_Faculty_Weight",
        desc="Provide the exact percentage weight (20%) for the Citations per Faculty indicator in QS World University Rankings 2026",
        parent=qs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the QS World University Rankings 2026, the Citations per Faculty indicator weight is {qs.citations_per_faculty_weight}.",
        node=qs_cpf_leaf,
        sources=_ensure_list(qs.methodology_urls),
        additional_instruction="Confirm the Citations per Faculty indicator weight for 2026."
    )

    # 4) IRN threshold
    qs_irn_leaf = evaluator.add_leaf(
        id="QS_IRN_Papers_Threshold",
        desc='Provide the minimum number of joint papers (3 or more) required over a five-year period for a partnership to be considered sustained in the International Research Network indicator',
        parent=qs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the QS 2026 methodology, the International Research Network (IRN) indicator defines a sustained partnership as {qs.irn_sustained_threshold}.",
        node=qs_irn_leaf,
        sources=_ensure_list(qs.methodology_urls),
        additional_instruction="Confirm the exact sustained-partnership threshold phrasing, e.g., '3 or more joint papers over five years'."
    )

    # 5) Publication date of rankings
    qs_pub_leaf = evaluator.add_leaf(
        id="QS_Publication_Date",
        desc="Provide the publication date (June 19, 2025) of QS World University Rankings 2026",
        parent=qs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The QS World University Rankings 2026 were published on {qs.publication_date}.",
        node=qs_pub_leaf,
        sources=_ensure_list(qs.methodology_urls),
        additional_instruction="Confirm the 2026 rankings release/publication date from QS's official materials."
    )


async def build_aera_section(evaluator: Evaluator, parent, aera: AERAInfo) -> None:
    aera_node = evaluator.add_parallel(
        id="AERA_Conference_Information",
        desc="Complete information about the 2026 AERA Annual Meeting",
        parent=parent,
        critical=False
    )

    # 1) Information URL validity
    aera_url_leaf = evaluator.add_leaf(
        id="AERA_Information_URL",
        desc="Provide valid reference URL for AERA 2026 Annual Meeting information",
        parent=aera_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page provides official information about the AERA 2026 Annual Meeting.",
        node=aera_url_leaf,
        sources=_ensure_list(aera.info_urls),
        additional_instruction="Verify the page is by AERA (or an official venue partner page) and specifically concerns the 2026 Annual Meeting."
    )

    # 2) Location
    aera_loc_leaf = evaluator.add_leaf(
        id="AERA_Location",
        desc="Provide the city and state (Los Angeles, California) where the 2026 AERA Annual Meeting will be held",
        parent=aera_node,
        critical=True
    )
    loc_str = _city_state_str(aera.city, aera.state)
    await evaluator.verify(
        claim=f"The 2026 AERA Annual Meeting will be held in {loc_str}.",
        node=aera_loc_leaf,
        sources=_ensure_list(aera.info_urls),
        additional_instruction="Confirm both the city and state for the 2026 meeting location (e.g., Los Angeles, California)."
    )

    # 3) Start date
    aera_start_leaf = evaluator.add_leaf(
        id="AERA_Start_Date",
        desc="Provide the start date (April 8, 2026) of the 2026 AERA Annual Meeting",
        parent=aera_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2026 AERA Annual Meeting starts on {aera.start_date}.",
        node=aera_start_leaf,
        sources=_ensure_list(aera.info_urls),
        additional_instruction="Confirm the officially stated start date for the 2026 meeting."
    )

    # 4) End date
    aera_end_leaf = evaluator.add_leaf(
        id="AERA_End_Date",
        desc="Provide the end date (April 12, 2026) of the 2026 AERA Annual Meeting",
        parent=aera_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2026 AERA Annual Meeting ends on {aera.end_date}.",
        node=aera_end_leaf,
        sources=_ensure_list(aera.info_urls),
        additional_instruction="Confirm the officially stated end date for the 2026 meeting."
    )

    # 5) Venue
    aera_venue_leaf = evaluator.add_leaf(
        id="AERA_Venue",
        desc="Provide the name of the venue (Los Angeles Convention Center) where the 2026 AERA Annual Meeting will be held",
        parent=aera_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2026 AERA Annual Meeting will be held at {aera.venue}.",
        node=aera_venue_leaf,
        sources=_ensure_list(aera.info_urls),
        additional_instruction="Confirm the official venue name for the 2026 meeting (e.g., Los Angeles Convention Center)."
    )


async def build_nih_section(evaluator: Evaluator, parent, nih: NIHRequirement) -> None:
    nih_node = evaluator.add_parallel(
        id="NIH_Requirements_Information",
        desc="Information about NIH grant application requirements for research security training",
        parent=parent,
        critical=False
    )

    # 1) Requirements URL validity
    nih_url_leaf = evaluator.add_leaf(
        id="NIH_Requirements_URL",
        desc="Provide valid reference URL for NIH grant application requirements regarding research security training",
        parent=nih_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page describes NIH requirements for Research Security Training for grant applications.",
        node=nih_url_leaf,
        sources=_ensure_list(nih.info_urls),
        additional_instruction="Verify the page is an official NIH/OD/Grants Policy or equivalent authoritative page describing Research Security Training requirements."
    )

    # 2) Training effective date
    nih_date_leaf = evaluator.add_leaf(
        id="NIH_Training_Effective_Date",
        desc='Provide the earliest application due date (May 25, 2026) on or after which all key personnel must have completed Research Security Training for NIH grant submissions',
        parent=nih_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For NIH applications with due dates on or after {nih.earliest_due_date}, all key personnel must have completed Research Security Training.",
        node=nih_date_leaf,
        sources=_ensure_list(nih.info_urls),
        additional_instruction="Confirm the earliest application due date from NIH policy or notice."
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
) -> Dict[str, Any]:
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

    # Extract four sections in parallel
    the_task = evaluator.extract(
        prompt=prompt_extract_the(),
        template_class=THEMethodology,
        extraction_name="the_methodology"
    )
    qs_task = evaluator.extract(
        prompt=prompt_extract_qs(),
        template_class=QSMethodology,
        extraction_name="qs_methodology"
    )
    aera_task = evaluator.extract(
        prompt=prompt_extract_aera(),
        template_class=AERAInfo,
        extraction_name="aera_info"
    )
    nih_task = evaluator.extract(
        prompt=prompt_extract_nih(),
        template_class=NIHRequirement,
        extraction_name="nih_requirement"
    )

    the_data, qs_data, aera_data, nih_data = await asyncio.gather(the_task, qs_task, aera_task, nih_task)

    # Build and verify each section under root
    await build_the_section(evaluator, root, the_data)
    await build_qs_section(evaluator, root, qs_data)
    await build_aera_section(evaluator, root, aera_data)
    await build_nih_section(evaluator, root, nih_data)

    return evaluator.get_summary()