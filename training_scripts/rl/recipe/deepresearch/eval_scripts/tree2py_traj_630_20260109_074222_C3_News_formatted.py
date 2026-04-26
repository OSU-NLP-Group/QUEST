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
TASK_ID = "ceo_company_city_census_2024"
TASK_DESCRIPTION = (
    "In 2024, a major athletic footwear and apparel company announced the return of a former executive who had "
    "previously retired from the company in 2020 after holding the position of President of Consumer and Marketplace. "
    "This executive started their career with the company as an intern in 1988 and was named the company's new CEO, "
    "with the appointment effective in October 2024. Identify this CEO and the company they lead. Then, determine "
    "where this CEO earned their undergraduate degree and in what year they graduated. Next, identify the city where "
    "that university is located. According to U.S. Census Bureau data released in 2025 for July 1, 2024 population "
    "estimates, what population milestone did this city achieve in 2024, and what was the city's ranking among all U.S. "
    "cities by population?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CEOInfo(BaseModel):
    name: Optional[str] = None
    company: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class EducationInfo(BaseModel):
    university: Optional[str] = None
    grad_year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class UniversityCityInfo(BaseModel):
    city: Optional[str] = None
    country: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CensusInfo(BaseModel):
    population_2024: Optional[str] = None
    milestone: Optional[str] = None
    ranking_2024: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AnswerExtraction(BaseModel):
    ceo: Optional[CEOInfo] = None
    education: Optional[EducationInfo] = None
    university_city: Optional[UniversityCityInfo] = None
    census: Optional[CensusInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the following structured information from the answer. Only extract what is explicitly stated in the answer text, and only extract URLs explicitly present in the answer.

    Structure:
    {
      "ceo": {
        "name": string or null,                       // CEO full name as stated (e.g., "Elliott Hill")
        "company": string or null,                    // Company name (e.g., "Nike")
        "urls": [string]                              // URLs cited for the CEO appointment/career info (press release, credible news). Only URLs explicitly provided in the answer.
      },
      "education": {
        "university": string or null,                 // Undergraduate university name
        "grad_year": string or null,                  // 4-digit graduation year for undergraduate degree (e.g., "1990")
        "urls": [string]                              // URLs cited for education details
      },
      "university_city": {
        "city": string or null,                       // U.S. city where the university is located (e.g., "Lubbock")
        "country": string or null,                    // Country for the university (e.g., "United States")
        "urls": [string]                              // URLs cited for the university location
      },
      "census": {
        "population_2024": string or null,            // City's July 1, 2024 population estimate figure as written (e.g., "301,123")
        "milestone": string or null,                  // Population milestone achieved in 2024 (e.g., "surpassed 300,000")
        "ranking_2024": string or null,               // National ranking among U.S. cities by population for 2024 (e.g., "62" or "62nd")
        "urls": [string]                              // URLs cited for U.S. Census Bureau (Vintage 2024 release in 2025) or credible sources citing it
      }
    }

    Rules:
    - Extract only what appears in the answer text.
    - For URLs, extract only valid URLs, including those embedded in markdown links. If a URL is missing protocol, prepend http://
    - If something is missing in the answer, return null for single fields or [] for url lists.
    - Prefer exact strings as shown in the answer (do not normalize numbers).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s) and bool(s.strip())


def _dedup_sources(*lists: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not _nonempty(u):
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_ceo_and_company(
    evaluator: Evaluator,
    parent_node,
    ceo: Optional[CEOInfo],
) -> None:
    node = evaluator.add_parallel(
        id="CEO_and_Company",
        desc="Provide the CEO identity and the company they lead.",
        parent=parent_node,
        critical=True,
    )

    # CEO name existence (as a binary check)
    ceo_name_ok = _nonempty(ceo.name if ceo else None) and (" " in (ceo.name or "").strip())
    evaluator.add_custom_node(
        result=ceo_name_ok,
        id="CEO_Name",
        desc="States the CEO's first and last name.",
        parent=node,
        critical=True,
    )

    # Company name existence (as a binary check)
    company_ok = _nonempty(ceo.company if ceo else None)
    evaluator.add_custom_node(
        result=company_ok,
        id="Company_Name",
        desc="States the company's name.",
        parent=node,
        critical=True,
    )


async def build_ceo_constraint_checks(
    evaluator: Evaluator,
    parent_node,
    ceo: Optional[CEOInfo],
) -> None:
    node = evaluator.add_parallel(
        id="CEO_Constraint_Checks",
        desc="Verify the CEO/company pair satisfies the appointment and career-history constraints from the prompt.",
        parent=parent_node,
        critical=True,
    )

    ceo_name = (ceo.name or "").strip() if ceo else ""
    company = (ceo.company or "").strip() if ceo else ""
    sources = ceo.urls if ceo else []

    # Create leaves
    n1 = evaluator.add_leaf(
        id="Appointment_Announced_2024",
        desc="The CEO appointment was announced in 2024.",
        parent=node,
        critical=True,
    )
    n2 = evaluator.add_leaf(
        id="Appointment_Effective_October_2024",
        desc="The CEO appointment was effective in October 2024.",
        parent=node,
        critical=True,
    )
    n3 = evaluator.add_leaf(
        id="Retired_2020_From_Same_Company",
        desc="The CEO previously retired from the same company in 2020.",
        parent=node,
        critical=True,
    )
    n4 = evaluator.add_leaf(
        id="Prior_Role_President_Consumer_and_Marketplace",
        desc="Before retiring, the CEO held the position of President of Consumer and Marketplace.",
        parent=node,
        critical=True,
    )
    n5 = evaluator.add_leaf(
        id="Started_As_Intern_1988",
        desc="The CEO started at the company as an intern in 1988.",
        parent=node,
        critical=True,
    )

    claims_and_sources = [
        (
            f"The appointment of {ceo_name} as CEO of {company} was announced in calendar year 2024.",
            sources,
            n1,
            "Rely on the press release or credible news. The claim must be explicitly stated. If no source URL supports this, mark as not supported."
        ),
        (
            f"The appointment of {ceo_name} as CEO of {company} was effective in October 2024.",
            sources,
            n2,
            "Accept formats like 'effective October 1, 2024' or 'effective Oct. 2024'. The effective date must clearly be in October 2024."
        ),
        (
            f"{ceo_name} previously retired from {company} in 2020.",
            sources,
            n3,
            "The source should explicitly say the person retired in 2020 from the same company (a return in 2024)."
        ),
        (
            f"Before retiring, {ceo_name} held the position of 'President of Consumer and Marketplace' at {company}.",
            sources,
            n4,
            "Minor punctuation/casing variations are acceptable, but the title must match substantively."
        ),
        (
            f"{ceo_name} started at {company} as an intern in 1988.",
            sources,
            n5,
            "The source must state or strongly indicate the person began as an intern in 1988."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def build_undergraduate_education(
    evaluator: Evaluator,
    parent_node,
    ceo: Optional[CEOInfo],
    edu: Optional[EducationInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Undergraduate_Education",
        desc="Provide the CEO's undergraduate institution and graduation year, and ensure it is a U.S. university with an identifiable graduation year.",
        parent=parent_node,
        critical=True,
    )

    ceo_name = (ceo.name or "").strip() if ceo else ""
    university = (edu.university or "").strip() if edu else ""
    grad_year = (edu.grad_year or "").strip() if edu else ""
    sources = _dedup_sources(edu.urls if edu else [], ceo.urls if ceo else [])

    n_uni = evaluator.add_leaf(
        id="Undergrad_University_Name",
        desc="Names the university where the CEO earned their undergraduate degree.",
        parent=node,
        critical=True,
    )
    n_year = evaluator.add_leaf(
        id="Graduation_Year",
        desc="Provides the year the CEO graduated from their undergraduate program.",
        parent=node,
        critical=True,
    )
    n_year_pub = evaluator.add_leaf(
        id="Graduation_Year_Publicly_Identifiable",
        desc="The stated graduation year is identifiable from publicly available information (not left unknown/unstated).",
        parent=node,
        critical=True,
    )
    n_us = evaluator.add_leaf(
        id="University_Is_In_US",
        desc="The undergraduate university is located in the United States.",
        parent=node,
        critical=True,
    )

    claims_and_sources = [
        (
            f"{ceo_name} earned an undergraduate degree from {university}.",
            sources,
            n_uni,
            "Verify that the person earned their undergraduate degree from this university. Minor naming variants acceptable."
        ),
        (
            f"{ceo_name} graduated (undergraduate) in {grad_year}.",
            sources,
            n_year,
            "Check that the undergraduate graduation year is exactly identifiable (4 digits) in the sources."
        ),
        (
            f"The undergraduate graduation year for {ceo_name} is explicitly stated as {grad_year} in publicly available sources.",
            sources,
            n_year_pub,
            "Ensure the year is explicitly present in the provided URLs (not inferred)."
        ),
        (
            f"{university} is located in the United States.",
            sources,
            n_us,
            "If needed, use the university's official site or Wikipedia among the provided URLs; confirm the country is U.S."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def build_university_city(
    evaluator: Evaluator,
    parent_node,
    edu: Optional[EducationInfo],
    city_info: Optional[UniversityCityInfo],
) -> None:
    node = evaluator.add_parallel(
        id="University_City",
        desc="Identify the U.S. city where the university is located.",
        parent=parent_node,
        critical=True,
    )

    university = (edu.university or "").strip() if edu else ""
    city = (city_info.city or "").strip() if city_info else ""
    sources = _dedup_sources(city_info.urls if city_info else [], edu.urls if edu else [])

    n_name = evaluator.add_leaf(
        id="City_Name",
        desc="Provides the city name where the university is located.",
        parent=node,
        critical=True,
    )
    n_us = evaluator.add_leaf(
        id="City_Is_In_United_States",
        desc="Confirms the city is in the United States.",
        parent=node,
        critical=True,
    )

    claims_and_sources = [
        (
            f"The university {university} is located in {city}.",
            sources,
            n_name,
            "Verify that this is the primary campus city associated with the undergraduate degree."
        ),
        (
            f"The city of {city} is in the United States.",
            sources,
            n_us,
            "Confirm that this city is a U.S. city."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def build_city_population_and_ranking(
    evaluator: Evaluator,
    parent_node,
    city_info: Optional[UniversityCityInfo],
    census: Optional[CensusInfo],
) -> None:
    node = evaluator.add_parallel(
        id="City_Population_Milestone_and_Ranking_2024",
        desc="Using U.S. Census Bureau July 1, 2024 population estimates (released in 2025), provide the city's population figure, the milestone it achieved in 2024, and its national population ranking among U.S. cities.",
        parent=parent_node,
        critical=True,
    )

    city = (city_info.city or "").strip() if city_info else ""
    population = (census.population_2024 or "").strip() if census else ""
    milestone = (census.milestone or "").strip() if census else ""
    ranking = (census.ranking_2024 or "").strip() if census else ""
    sources = census.urls if census else []

    n_pop = evaluator.add_leaf(
        id="Census_July1_2024_Population_Figure",
        desc="Provides the city's population estimate for July 1, 2024 from U.S. Census Bureau estimates (Vintage 2024 release).",
        parent=node,
        critical=True,
    )
    n_milestone = evaluator.add_leaf(
        id="Population_Milestone_Description",
        desc="States the population milestone the city achieved in 2024 (as a clearly defined milestone/threshold) and ensures it is consistent with the provided Census population figure.",
        parent=node,
        critical=True,
    )
    n_rank = evaluator.add_leaf(
        id="National_Population_Ranking_Among_US_Cities",
        desc="Provides the city's ranking among all U.S. cities by population according to the 2024 (July 1, 2024 estimate) data.",
        parent=node,
        critical=True,
    )

    claims_and_sources = [
        (
            f"According to the U.S. Census Bureau (Vintage 2024 estimates released in 2025), the July 1, 2024 population estimate for {city} is {population}.",
            sources,
            n_pop,
            "Use official Census sources or credible pages citing the Vintage 2024 release. Allow minor formatting differences (commas)."
        ),
        (
            f"In 2024, {city} achieved this population milestone: {milestone}.",
            sources,
            n_milestone,
            "Verify the milestone is explicitly supported or is an immediate consequence of the stated 2024 population figure (e.g., 'surpassed 300,000'). Ensure no contradiction."
        ),
        (
            f"Based on the July 1, 2024 estimates, {city} ranked {ranking} among all U.S. cities by population.",
            sources,
            n_rank,
            "Allow ordinal/numeric variants (e.g., '12' vs '12th'). The ranking should be from the 2024 estimates."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the CEO/company/education/city/census task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # The overall flow is sequential as per rubric
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

    # 1) Extract structured information from the answer
    extracted: AnswerExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AnswerExtraction,
        extraction_name="structured_answer",
    )

    # 2) Build the rubric tree
    task_node = evaluator.add_sequential(
        id="Task_Completion",
        desc=(
            "Identify the CEO and company matching the 2024 return/appointment description, then provide the CEO's "
            "undergraduate school and graduation year, the university's U.S. city, and the city's July 1, 2024 Census-"
            "estimate population milestone and national ranking."
        ),
        parent=root,
        critical=True,
    )

    # Subtree 1: CEO and Company
    await build_ceo_and_company(evaluator, task_node, extracted.ceo)

    # Subtree 2: CEO Constraint Checks
    await build_ceo_constraint_checks(evaluator, task_node, extracted.ceo)

    # Subtree 3: Undergraduate Education
    await build_undergraduate_education(evaluator, task_node, extracted.ceo, extracted.education)

    # Subtree 4: University City
    await build_university_city(evaluator, task_node, extracted.education, extracted.university_city)

    # Subtree 5: City Population Milestone and Ranking (2024)
    await build_city_population_and_ranking(evaluator, task_node, extracted.university_city, extracted.census)

    # 3) Return structured summary
    return evaluator.get_summary()