import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "filmmaker_identification_1979_1990_oscars_company_film_2018_2019"
TASK_DESCRIPTION = """Who is the filmmaker that meets all of the following criteria:

1. Studied filmmaking at a university-affiliated film school between 1979 and 1990
2. Made history by becoming the first person to win the Academy Award for Best Cinematography for a film they also directed
3. Founded or co-founded a production company with headquarters located in California
4. Directed and served as cinematographer on a film released between 2018-2019 that won at least three Academy Awards, including Best Director, Best Cinematography, and Best Foreign Language Film (or Best International Feature Film)

Provide the filmmaker's first and last name, along with supporting reference URLs for each of the four criteria.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EducationInfo(BaseModel):
    film_school: Optional[str] = None
    university: Optional[str] = None
    study_years_text: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HistoricOscarInfo(BaseModel):
    film_title: Optional[str] = None
    award_year: Optional[str] = None
    is_first_in_history: Optional[bool] = None
    urls: List[str] = Field(default_factory=list)


class CompanyInfo(BaseModel):
    company_name: Optional[str] = None
    role: Optional[str] = None  # e.g., "founder", "co-founder"
    headquarters: Optional[str] = None  # free-form location text
    urls: List[str] = Field(default_factory=list)


class Film2018Info(BaseModel):
    film_title: Optional[str] = None
    release_year: Optional[str] = None
    served_as_director: Optional[bool] = None
    served_as_cinematographer: Optional[bool] = None
    awards_won: List[str] = Field(default_factory=list)  # explicit categories named in the answer, if any
    urls: List[str] = Field(default_factory=list)


class FilmmakerExtraction(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    education: Optional[EducationInfo] = None
    historic_oscar: Optional[HistoricOscarInfo] = None
    company: Optional[CompanyInfo] = None
    film_2018_2019: Optional[Film2018Info] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_filmmaker() -> str:
    return """
    Extract the filmmaker's identity and the details for each of the four criteria from the provided answer. Only extract what is explicitly stated in the answer and only include URLs that appear in the answer text. Do not invent or infer missing information.

    Return a JSON object with the following fields:

    - first_name: The filmmaker's first name (string or null if not provided).
    - last_name: The filmmaker's last name (string or null if not provided).

    - education: Object (or null) containing:
        - film_school: Name of the film school where the person studied filmmaking (string or null).
        - university: Name of the university that the film school is part of or affiliated with (string or null).
        - study_years_text: Any years or time span text stated in the answer for when they studied (string or null, e.g., "early 1980s", "1979–1981").
        - start_year: A specific start year if explicitly stated (string or null).
        - end_year: A specific end year if explicitly stated (string or null).
        - urls: Array of URLs that the answer cites specifically to support the education criterion. Include only valid URLs that appear in the answer.

    - historic_oscar: Object (or null) containing:
        - film_title: The specific film title tied to the Best Cinematography win (string or null).
        - award_year: The ceremony year or relevant year if mentioned (string or null).
        - is_first_in_history: true/false if the answer explicitly claims they were the first person in Academy Awards history to win Best Cinematography for a film they also directed; null if not stated.
        - urls: Array of URLs that the answer cites to support this historic Oscar achievement.

    - company: Object (or null) containing:
        - company_name: Name of the production company founded or co-founded by the filmmaker (string or null).
        - role: "founder" or "co-founder" if explicitly stated; otherwise null.
        - headquarters: Stated HQ location text if given (string or null).
        - urls: Array of URLs that the answer cites to support the founding/co-founding and HQ in California facts.

    - film_2018_2019: Object (or null) containing:
        - film_title: The title of a feature film released between 2018 and 2019 (inclusive) that the filmmaker both directed and served as cinematographer on (string or null).
        - release_year: The year of release (string or null).
        - served_as_director: true/false if the answer explicitly states they directed this film; null if not stated.
        - served_as_cinematographer: true/false if the answer explicitly states they were the cinematographer/DP on this film; null if not stated.
        - awards_won: Array of Oscar categories explicitly stated as won by this film (e.g., "Best Director", "Best Cinematography", "Best Foreign Language Film" or "Best International Feature Film").
        - urls: Array of URLs the answer cites for this film and its Oscar wins.

    Special rules for URL extraction:
    - Extract only URLs that actually appear in the answer (including markdown links). Do not infer or add any URLs not present in the answer.
    - Include full URLs. If a URL is missing protocol, prepend http://.

    If any field is not explicitly mentioned in the answer, set it to null (or an empty array for 'urls' or 'awards_won').
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_full_name(first_name: Optional[str], last_name: Optional[str]) -> str:
    first = first_name.strip() if first_name else ""
    last = last_name.strip() if last_name else ""
    return (first + " " + last).strip()


def has_nonempty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls or [])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_answer_outputs_checks(
    evaluator: Evaluator,
    parent,
    extracted: FilmmakerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="answer_outputs",
        desc="Check that the response provides the required identity fields and references.",
        parent=parent,
        critical=True,
    )

    # Provide_First_And_Last_Name
    name_ok = (extracted.first_name is not None and str(extracted.first_name).strip() != "") and \
              (extracted.last_name is not None and str(extracted.last_name).strip() != "")
    evaluator.add_custom_node(
        result=name_ok,
        id="provide_first_and_last_name",
        desc="Response includes the filmmaker's first and last name.",
        parent=node,
        critical=True
    )

    # Provide_Reference_URLs_For_Each_Criterion
    edu_ok = has_nonempty_urls(getattr(extracted.education or EducationInfo(), "urls", []))
    hist_ok = has_nonempty_urls(getattr(extracted.historic_oscar or HistoricOscarInfo(), "urls", []))
    comp_ok = has_nonempty_urls(getattr(extracted.company or CompanyInfo(), "urls", []))
    film_ok = has_nonempty_urls(getattr(extracted.film_2018_2019 or Film2018Info(), "urls", []))

    evaluator.add_custom_node(
        result=edu_ok and hist_ok and comp_ok and film_ok,
        id="provide_reference_urls_for_each_criterion",
        desc="Response includes publicly accessible reference URLs supporting each of the four main criteria categories (education, historic Oscar achievement, production company, 2018–2019 film/awards).",
        parent=node,
        critical=True
    )


async def build_education_checks(
    evaluator: Evaluator,
    parent,
    extracted: FilmmakerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="educational_background",
        desc="Verify filmmaking study at a university-affiliated film school during 1979–1990 (inclusive), with references.",
        parent=parent,
        critical=True
    )

    edu = extracted.education or EducationInfo()
    full_name = safe_full_name(extracted.first_name, extracted.last_name)
    edu_urls = edu.urls

    # Attended_University_Affiliated_Film_School
    leaf_affiliation = evaluator.add_leaf(
        id="attended_university_affiliated_film_school",
        desc="Filmmaker studied filmmaking at a film school affiliated with or part of a university.",
        parent=node,
        critical=True
    )
    school_phrase = edu.film_school or "a film school"
    if edu.university:
        claim_aff = f"{full_name} studied filmmaking at {school_phrase}, which is part of or affiliated with {edu.university} (a university)."
    else:
        claim_aff = f"{full_name} studied filmmaking at {school_phrase}, which is part of or affiliated with a university (i.e., a university-affiliated film school)."
    await evaluator.verify(
        claim=claim_aff,
        node=leaf_affiliation,
        sources=edu_urls,
        additional_instruction="Support should explicitly indicate the film school is part of or affiliated with a university (e.g., 'at [University]' or 'the film school of [University]'). Allow reasonable equivalents and official university pages."
    )

    # Study_Period_1979_1990
    leaf_period = evaluator.add_leaf(
        id="study_period_1979_1990",
        desc="Study/enrollment occurred between 1979 and 1990 (inclusive).",
        parent=node,
        critical=True
    )
    if edu.start_year or edu.end_year or edu.study_years_text:
        period_hint = f" The answer mentions: {edu.study_years_text or ''} {edu.start_year or ''} {edu.end_year or ''}.".strip()
    else:
        period_hint = ""
    claim_period = f"{full_name} studied filmmaking during a timeframe that falls between 1979 and 1990 inclusive.{(' ' + period_hint).strip()}"
    await evaluator.verify(
        claim=claim_period,
        node=leaf_period,
        sources=edu_urls,
        additional_instruction="Consider explicit years or clear phrasing such as 'early 1980s'. Accept if the documented timeframe lies wholly within 1979–1990 (inclusive)."
    )


async def build_historic_oscar_checks(
    evaluator: Evaluator,
    parent,
    extracted: FilmmakerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="historic_oscar_achievement",
        desc="Verify the filmmaker was the first person in Academy Awards history to win Best Cinematography for a film they also directed, with references.",
        parent=parent,
        critical=True
    )

    hist = extracted.historic_oscar or HistoricOscarInfo()
    full_name = safe_full_name(extracted.first_name, extracted.last_name)
    hist_urls = hist.urls
    film_title = hist.film_title or "the film"

    # Won_Best_Cinematography_For_Self_Directed_Film
    leaf_bc_selfdir = evaluator.add_leaf(
        id="won_best_cinematography_for_self_directed_film",
        desc="Filmmaker won the Academy Award for Best Cinematography for a film that the filmmaker also directed.",
        parent=node,
        critical=True
    )
    claim_bc_selfdir = f"{full_name} won the Academy Award for Best Cinematography for {film_title}, and {full_name} also directed that same film."
    await evaluator.verify(
        claim=claim_bc_selfdir,
        node=leaf_bc_selfdir,
        sources=hist_urls,
        additional_instruction="Confirm both facts: (1) the person won Best Cinematography; (2) for the same film, they are credited as the director. Prefer authoritative sources (Academy site, major trades, or film databases)."
    )

    # First_In_Oscar_History
    leaf_first_history = evaluator.add_leaf(
        id="first_in_oscar_history",
        desc="Filmmaker was the first person in Academy Awards history to achieve Best Cinematography for a film they directed.",
        parent=node,
        critical=True
    )
    claim_first_history = f"{full_name} was the first person in Academy Awards history to win Best Cinematography for a film they also directed."
    await evaluator.verify(
        claim=claim_first_history,
        node=leaf_first_history,
        sources=hist_urls,
        additional_instruction="Look for explicit phrasing confirming 'first in history' for this specific combination (winning Best Cinematography on a film they directed)."
    )


async def build_production_company_checks(
    evaluator: Evaluator,
    parent,
    extracted: FilmmakerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="production_company",
        desc="Verify the filmmaker founded/co-founded a production company whose headquarters are in California and whose name is publicly documented, with references.",
        parent=parent,
        critical=True
    )

    comp = extracted.company or CompanyInfo()
    full_name = safe_full_name(extracted.first_name, extracted.last_name)
    comp_urls = comp.urls
    company_name = comp.company_name or "the production company"

    # Company_Founded_Or_Cofounded
    leaf_founded = evaluator.add_leaf(
        id="company_founded_or_cofounded",
        desc="Filmmaker founded or co-founded a film/television production company.",
        parent=node,
        critical=True
    )
    role_phrase = comp.role if comp.role else "founded or co-founded"
    claim_founded = f"{full_name} {role_phrase} {company_name}."
    await evaluator.verify(
        claim=claim_founded,
        node=leaf_founded,
        sources=comp_urls,
        additional_instruction="Confirm that the person is a founder or co-founder of the stated production company."
    )

    # Company_Name_Publicly_Documented
    leaf_name_doc = evaluator.add_leaf(
        id="company_name_publicly_documented",
        desc="The production company name is publicly documented.",
        parent=node,
        critical=True
    )
    claim_name_doc = f"The production company associated with {full_name} is named '{company_name}'."
    await evaluator.verify(
        claim=claim_name_doc,
        node=leaf_name_doc,
        sources=comp_urls,
        additional_instruction="Confirm that the company's official/publicly known name matches the stated name."
    )

    # California_Headquarters
    leaf_ca_hq = evaluator.add_leaf(
        id="california_headquarters",
        desc="The production company's headquarters are located in the state of California.",
        parent=node,
        critical=True
    )
    claim_ca_hq = f"The headquarters (or primary office) of {company_name} are located in California, United States."
    await evaluator.verify(
        claim=claim_ca_hq,
        node=leaf_ca_hq,
        sources=comp_urls,
        additional_instruction="Accept if the documented headquarters or primary office location is within California (e.g., Los Angeles, Santa Monica, San Francisco, etc.)."
    )


async def build_film_2018_2019_awards_checks(
    evaluator: Evaluator,
    parent,
    extracted: FilmmakerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="film_2018_2019_awards",
        desc="Verify the filmmaker directed and served as cinematographer on a feature film released 2018–2019 that won the specified Academy Awards, with references.",
        parent=parent,
        critical=True
    )

    film = extracted.film_2018_2019 or Film2018Info()
    full_name = safe_full_name(extracted.first_name, extracted.last_name)
    film_urls = film.urls
    film_title = film.film_title or "the film"

    # Create leaf nodes
    leaf_release = evaluator.add_leaf(
        id="feature_film_released_2018_2019",
        desc="A feature film exists that was released between 2018 and 2019 (inclusive).",
        parent=node,
        critical=True
    )
    leaf_directed = evaluator.add_leaf(
        id="filmmaker_directed_film",
        desc="Filmmaker directed that 2018–2019 feature film.",
        parent=node,
        critical=True
    )
    leaf_cinematographer = evaluator.add_leaf(
        id="filmmaker_served_as_cinematographer",
        desc="Filmmaker served as cinematographer (DP) on that 2018–2019 feature film.",
        parent=node,
        critical=True
    )
    leaf_won_bd = evaluator.add_leaf(
        id="film_won_best_director",
        desc="That film won the Academy Award for Best Director.",
        parent=node,
        critical=True
    )
    leaf_won_bc = evaluator.add_leaf(
        id="film_won_best_cinematography",
        desc="That film won the Academy Award for Best Cinematography.",
        parent=node,
        critical=True
    )
    leaf_won_bflf = evaluator.add_leaf(
        id="film_won_best_foreign_language_or_international_feature",
        desc="That film won the Academy Award for Best Foreign Language Film or Best International Feature Film.",
        parent=node,
        critical=True
    )

    # Build claims and verify in parallel
    claims_and_sources: List[tuple] = []

    claim_release = f"{film_title} was released in 2018 or 2019."
    claims_and_sources.append((
        claim_release,
        film_urls,
        leaf_release,
        "Accept if the film's widely recognized release year is 2018 or 2019 (festival premiere or general release acceptable)."
    ))

    claim_directed = f"{full_name} directed {film_title}."
    claims_and_sources.append((
        claim_directed,
        film_urls,
        leaf_directed,
        "Confirm the person is credited as the film's director."
    ))

    claim_cinematographer = f"{full_name} served as the cinematographer (director of photography) on {film_title}."
    claims_and_sources.append((
        claim_cinematographer,
        film_urls,
        leaf_cinematographer,
        "Confirm the person is credited as cinematographer/DP."
    ))

    claim_bd = f"{film_title} won the Academy Award for Best Director."
    claims_and_sources.append((
        claim_bd,
        film_urls,
        leaf_won_bd,
        "Confirm that the Best Director Oscar was awarded for this film (i.e., the director of this film won Best Director for it)."
    ))

    claim_bc = f"{film_title} won the Academy Award for Best Cinematography."
    claims_and_sources.append((
        claim_bc,
        film_urls,
        leaf_won_bc,
        "Confirm that this film won Best Cinematography at the Academy Awards."
    ))

    claim_bflf = f"{film_title} won the Academy Award for Best Foreign Language Film or Best International Feature Film."
    claims_and_sources.append((
        claim_bflf,
        film_urls,
        leaf_won_bflf,
        "Confirm that this film won either 'Best Foreign Language Film' or 'Best International Feature Film' at the Academy Awards."
    ))

    await evaluator.batch_verify(claims_and_sources)


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the filmmaker identification task and return the structured result.
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
        default_model=model
    )

    # Extraction
    extracted: FilmmakerExtraction = await evaluator.extract(
        prompt=prompt_extract_filmmaker(),
        template_class=FilmmakerExtraction,
        extraction_name="filmmaker_extraction"
    )

    # Build main critical aggregation node
    fi_root = evaluator.add_parallel(
        id="filmmaker_identification",
        desc="Identify a filmmaker who meets all specified education, historic Oscar achievement, production company, and 2018–2019 film/awards criteria, and provide supporting reference URLs.",
        parent=root,
        critical=True
    )

    # Answer Outputs checks
    await build_answer_outputs_checks(evaluator, fi_root, extracted)

    # Educational Background checks
    await build_education_checks(evaluator, fi_root, extracted)

    # Historic Oscar Achievement checks
    await build_historic_oscar_checks(evaluator, fi_root, extracted)

    # Production Company checks
    await build_production_company_checks(evaluator, fi_root, extracted)

    # Film 2018–2019 & Awards checks
    await build_film_2018_2019_awards_checks(evaluator, fi_root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()