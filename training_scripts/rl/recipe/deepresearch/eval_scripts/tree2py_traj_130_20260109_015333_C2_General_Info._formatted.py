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
TASK_ID = "2024_ceo_with_prior_restaurant_ceo_growth"
TASK_DESCRIPTION = (
    "Identify the name of a person who was appointed as CEO (or Chairman and CEO) of a major U.S. company in 2024, "
    "holds an MBA degree, attended university (either at the undergraduate or graduate level) in the state of Ohio or Illinois, "
    "previously served as CEO of a publicly traded company in the restaurant or food service industry for at least 5 years, "
    "and during their tenure led that company to achieve significant measurable growth (such as doubling revenue or substantially increasing stock price)."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AppointmentInfo(BaseModel):
    person_name: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    appointment_date: Optional[str] = None
    appointment_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MBAInfo(BaseModel):
    has_mba: Optional[str] = None  # e.g., "yes", "MBA", or a phrase indicating MBA
    institution: Optional[str] = None
    degree_details: Optional[str] = None  # e.g., "MBA", "MBA, Finance"
    sources: List[str] = Field(default_factory=list)


class StateEduInfo(BaseModel):
    attended: Optional[str] = None  # "yes"/"no"/description
    school_name: Optional[str] = None
    state: Optional[str] = None  # Expect "Ohio" or "Illinois" (or abbreviations OH/IL)
    level: Optional[str] = None  # e.g., "undergraduate", "graduate", "MBA", etc.
    sources: List[str] = Field(default_factory=list)


class PriorCEOInfo(BaseModel):
    company: Optional[str] = None
    publicly_traded: Optional[str] = None  # "yes"/"no"/description
    industry: Optional[str] = None  # should indicate "restaurant" or "food service"
    tenure_start: Optional[str] = None  # free text date/year
    tenure_end: Optional[str] = None  # free text date/year
    tenure_years_stated: Optional[str] = None  # e.g., "6 years"
    growth_description: Optional[str] = None  # free text describing growth
    growth_metric: Optional[str] = None  # e.g., "revenue doubled", "stock price up 150%"
    sources: List[str] = Field(default_factory=list)


class CandidateExtraction(BaseModel):
    appointment: Optional[AppointmentInfo] = None
    education_mba: Optional[MBAInfo] = None
    education_state: Optional[StateEduInfo] = None
    prior_ceo: Optional[PriorCEOInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_candidate() -> str:
    return """
    From the provided answer text, extract the primary candidate (person) presented as meeting ALL of the following:
    – Appointed as CEO (or Chairman and CEO) of a major U.S. company in 2024.
    – Holds an MBA degree (from an accredited university).
    – Attended university (undergraduate or graduate) in either Ohio or Illinois.
    – Previously served as CEO of a publicly traded restaurant/food service company for at least 5 years.
    – During that prior tenure, led the company to significant measurable growth (e.g., doubling revenue or substantial stock price increase).

    If multiple candidates are discussed, select the most emphasized one that is asserted to meet all criteria. Do NOT invent details. Only extract information explicitly present in the answer.

    Return a JSON object shaped as CandidateExtraction with the following structure:

    appointment:
      - person_name: The person's full name as given in the answer.
      - company: The major U.S. company where the person was appointed in 2024.
      - role: The appointed role title (e.g., "CEO", "Chairman and CEO").
      - appointment_date: The appointment date string if available (free text).
      - appointment_year: The 4-digit year of the appointment if given; else null.
      - sources: A list of URLs the answer cites specifically for the appointment (press releases, reputable news pages, etc.).
    
    education_mba:
      - has_mba: A short indicator text confirming MBA (e.g., "yes", "MBA", "MBA in Marketing"); null if not present.
      - institution: The MBA-granting institution name if present; else null.
      - degree_details: Any additional MBA detail (e.g., concentration); else null.
      - sources: URLs supporting the MBA claim (bio pages, school profiles, news).
    
    education_state:
      - attended: A short indicator (e.g., "yes", "undergraduate", "graduate") that indicates attendance in OH or IL; null if not present.
      - school_name: University or college name in Ohio or Illinois attended at undergrad/grad level; else null.
      - state: The state text (ideally "Ohio" or "Illinois", or abbreviations "OH"/"IL"); else null.
      - level: If the answer indicates undergrad or graduate explicitly, put here; else null.
      - sources: URLs supporting the OH/IL attendance/location.
    
    prior_ceo:
      - company: The name of the prior publicly traded restaurant/food service company where the person served as CEO.
      - publicly_traded: A short indicator text if the company is publicly traded (e.g., "publicly traded", "NYSE: TICKER"); else null.
      - industry: Industry description; should indicate restaurant or food service.
      - tenure_start: Start date/year string for CEO role; else null.
      - tenure_end: End date/year string for CEO role; else null.
      - tenure_years_stated: If the answer states a total, put that here (e.g., "6 years"); else null.
      - growth_description: Text describing measurable growth under their tenure; else null.
      - growth_metric: The metric if available (e.g., "revenue doubled", "stock up 120%"); else null.
      - sources: URLs supporting prior CEO role, public listing, industry, tenure, and growth.

    SPECIAL RULES FOR URL SOURCES:
    - Only extract URLs that are explicitly present in the answer text (including markdown links).
    - If a URL is missing a protocol, add http://
    - Do NOT fabricate URLs. If no URL is provided for a specific claim area, return an empty list for that field's sources.

    If any field is missing in the answer, return null for that field and an empty list for sources as appropriate. Do not infer beyond what is stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nz(x: Optional[str], default: str = "Unknown") -> str:
    return x if x and str(x).strip() else default


def _merge_sources(*args: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in args:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            url = u.strip()
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_appointment(
    evaluator: Evaluator,
    parent_node,
    appt: Optional[AppointmentInfo],
) -> None:
    # Create the "2024_CEO_Appointment" parallel critical node
    appointment_node = evaluator.add_parallel(
        id="2024_CEO_Appointment",
        desc="Verify the person was appointed as CEO (or Chairman and CEO) of a major U.S. company in 2024.",
        parent=parent_node,
        critical=True,
    )

    # Prepare data
    name = _nz(appt.person_name if appt else None, "the person")
    company = _nz(appt.company if appt else None, "the company")
    role = _nz(appt.role if appt else None, "the role")
    appt_year = _nz(appt.appointment_year if appt else None, "2024")  # fallback doesn't matter; sources decide
    appt_sources = (appt.sources if appt else []) or []

    # Create leaves (all critical)
    leaf_year = evaluator.add_leaf(
        id="Appointment_Year_2024",
        desc="The person was appointed in 2024.",
        parent=appointment_node,
        critical=True,
    )
    leaf_role = evaluator.add_leaf(
        id="Role_Is_CEO_or_Chairman_and_CEO",
        desc="The appointed role is CEO or Chairman and CEO.",
        parent=appointment_node,
        critical=True,
    )
    leaf_major = evaluator.add_leaf(
        id="Company_Is_Major_US_Company",
        desc="The appointment is at a major U.S. company (as required by the prompt).",
        parent=appointment_node,
        critical=True,
    )

    # Build claims
    claim_year = (
        f"{name} was appointed to the role of {role} at {company} in 2024."
    )
    claim_role = (
        f"The appointed role for {name} at {company} is CEO or 'Chairman and CEO' (i.e., includes the Chief Executive Officer role)."
    )
    claim_major = (
        f"{company} is a major U.S. company."
    )

    claims = [
        (claim_year, appt_sources, leaf_year,
         "Confirm that the appointment news/article or press release explicitly indicates the event happened in calendar year 2024."),
        (claim_role, appt_sources, leaf_role,
         "Confirm the appointed title explicitly includes 'CEO' or 'Chairman and CEO'. Treat 'Chief Executive Officer' as CEO."),
        (claim_major, appt_sources, leaf_major,
         "Accept indicators such as Fortune 1000/500 membership, S&P 500 inclusion, nationwide presence, or comparable stature to judge 'major U.S. company'. The evidence must be present on the provided page(s)."),
    ]

    await evaluator.batch_verify(claims)


async def build_and_verify_education(
    evaluator: Evaluator,
    parent_node,
    mba: Optional[MBAInfo],
    state_edu: Optional[StateEduInfo],
    appt: Optional[AppointmentInfo],
) -> None:
    # Create the "Educational_Requirements" parallel critical node
    edu_node = evaluator.add_parallel(
        id="Educational_Requirements",
        desc="Verify the person meets the education requirements.",
        parent=parent_node,
        critical=True,
    )

    # Prepare data
    name = _nz(appt.person_name if appt else None, "the person")

    mba_institution = _nz(mba.institution if mba else None, "the MBA institution")
    mba_sources = (mba.sources if mba else []) or []

    school_name = _nz(state_edu.school_name if state_edu else None, "the university")
    state_text = _nz(state_edu.state if state_edu else None, "the state")
    state_sources = (state_edu.sources if state_edu else []) or []

    # It is reasonable to include appointment sources if the answer cites the bio there
    combined_mba_sources = _merge_sources(mba_sources, appt.sources if appt else [])
    combined_state_sources = _merge_sources(state_sources, mba_sources, appt.sources if appt else [])

    # Leaves (all critical under critical parent)
    leaf_mba = evaluator.add_leaf(
        id="MBA_From_Accredited_University",
        desc="The person holds an MBA degree from an accredited university.",
        parent=edu_node,
        critical=True,
    )
    leaf_state = evaluator.add_leaf(
        id="Attended_University_in_OH_or_IL",
        desc="The person attended a university (undergraduate or graduate) in Ohio or Illinois.",
        parent=edu_node,
        critical=True,
    )

    # Claims
    claim_mba = (
        f"{name} holds an MBA degree from {mba_institution}, which is an accredited university."
    )
    claim_state = (
        f"{name} attended {school_name} located in {state_text}, and this is in the state of Ohio or Illinois, at the undergraduate or graduate level."
    )

    claims = [
        (claim_mba, combined_mba_sources, leaf_mba,
         "Verify that the pages explicitly indicate the person holds an MBA and that the MBA institution is an accredited university. "
         "Look for phrases like 'accredited', 'AACSB-accredited', or other accreditation statements on the provided URLs."),
        (claim_state, combined_state_sources, leaf_state,
         "Verify both attendance AND location. The university must be in Ohio or Illinois. "
         "If the page confirms the person attended the school (undergraduate or graduate) and the school's location is in OH or IL, consider it satisfied."),
    ]

    await evaluator.batch_verify(claims)


async def build_and_verify_prior_ceo(
    evaluator: Evaluator,
    parent_node,
    prior: Optional[PriorCEOInfo],
    appt: Optional[AppointmentInfo],
) -> None:
    # Create the "Prior_CEO_Experience" parallel critical node
    prior_node = evaluator.add_parallel(
        id="Prior_CEO_Experience",
        desc="Verify the person’s prior CEO experience meets all constraints.",
        parent=parent_node,
        critical=True,
    )

    # Prepare data
    name = _nz(appt.person_name if appt else None, "the person")
    p_company = _nz(prior.company if prior else None, "the prior company")
    p_industry = _nz(prior.industry if prior else None, "the industry")
    start = _nz(prior.tenure_start if prior else None, "the start date")
    end = _nz(prior.tenure_end if prior else None, "the end date")
    years_stated = _nz(prior.tenure_years_stated if prior else None, "")
    growth_desc = _nz(prior.growth_description if prior else None, "significant measurable growth")
    growth_metric = _nz(prior.growth_metric if prior else None, "a substantial measurable increase")

    prior_sources = (prior.sources if prior else []) or []
    # It's also reasonable to include appointment sources if the bio covering prior career is there
    combined_prior_sources = _merge_sources(prior_sources, appt.sources if appt else [])

    # Create leaves (all critical under a critical parent)
    leaf_public = evaluator.add_leaf(
        id="Previously_CEO_of_Publicly_Traded_Company",
        desc="The person previously served as CEO of a publicly traded company.",
        parent=prior_node,
        critical=True,
    )
    leaf_industry = evaluator.add_leaf(
        id="Prior_Company_In_Restaurant_or_Food_Service",
        desc="That publicly traded company is in the restaurant or food service industry.",
        parent=prior_node,
        critical=True,
    )
    leaf_tenure = evaluator.add_leaf(
        id="Prior_CEO_Tenure_At_Least_5_Years",
        desc="The prior CEO tenure was at least 5 years.",
        parent=prior_node,
        critical=True,
    )
    leaf_growth = evaluator.add_leaf(
        id="Significant_Measurable_Growth_During_Tenure",
        desc="During that tenure, the company achieved significant measurable growth (e.g., revenue doubling or substantial stock price increase).",
        parent=prior_node,
        critical=True,
    )

    # Build claims
    claim_public = (
        f"{name} previously served as CEO of {p_company}, and {p_company} is publicly traded (e.g., listed on a stock exchange)."
    )
    claim_industry = (
        f"{p_company} operates in the restaurant or food service industry."
    )
    if years_stated and years_stated.lower() != "unknown":
        claim_tenure = (
            f"{name}'s tenure as CEO of {p_company} lasted at least 5 years (from {start} to {end}, stated as {years_stated})."
        )
    else:
        claim_tenure = (
            f"{name}'s tenure as CEO of {p_company} lasted at least 5 years (from {start} to {end})."
        )
    claim_growth = (
        f"During {name}'s CEO tenure at {p_company}, the company achieved significant measurable growth: "
        f"{growth_desc} (e.g., {growth_metric})."
    )

    claims = [
        (claim_public, combined_prior_sources, leaf_public,
         "Confirm both aspects: (1) the person was CEO of the company; and (2) the company is publicly traded (NYSE/Nasdaq, a ticker symbol, or similar)."),
        (claim_industry, combined_prior_sources, leaf_industry,
         "Verify that the company's business is clearly restaurant or food service (quick-service, casual dining, hospitality with food services, etc.)."),
        (claim_tenure, combined_prior_sources, leaf_tenure,
         "Check the start and end dates (or stated duration) to ensure the tenure is at least 5 years. "
         "If an explicit total number of years is stated and ≥ 5, accept that. Otherwise, compute based on dates."),
        (claim_growth, combined_prior_sources, leaf_growth,
         "Confirm that the sources explicitly support significant measurable growth during the tenure, "
         "such as doubling revenue, large multi-year revenue growth, or substantial stock price appreciation. "
         "General statements without measurable evidence should not be accepted."),
    ]

    await evaluator.batch_verify(claims)


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 2024 CEO selection with prior restaurant CEO growth constraints.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # The top-level aggregator can be parallel
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

    # Extract structured candidate information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_candidate(),
        template_class=CandidateExtraction,
        extraction_name="candidate_extraction",
    )

    # Build a top-level critical node representing the rubric root (under framework root)
    rubric_root = evaluator.add_parallel(
        id="CEO_Meeting_All_Criteria",
        desc="Identify a person appointed as CEO (or Chairman and CEO) of a major U.S. company in 2024 who meets all education and prior-CEO-experience requirements.",
        parent=root,
        critical=True,
    )

    # Build and verify subtrees according to rubric
    await build_and_verify_appointment(evaluator, rubric_root, extracted.appointment)
    await build_and_verify_education(evaluator, rubric_root, extracted.education_mba, extracted.education_state, extracted.appointment)
    await build_and_verify_prior_ceo(evaluator, rubric_root, extracted.prior_ceo, extracted.appointment)

    # Return structured summary
    return evaluator.get_summary()