import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "french_director_identification_2023_awards_and_career"
TASK_DESCRIPTION = (
    "Identify the full name of the French film director who satisfies all of the following criteria: "
    "(1) Was born in 1978, "
    "(2) Graduated from Beaux-Arts de Paris, "
    "(3) Won the Palme d'Or at the 76th Cannes Film Festival in 2023, "
    "(4) Collaborated with cinematographer Simon Beaufils on at least three feature films, including films released in 2016, 2019, and 2023, "
    "(5) For the 2023 film: it was shot in the French Alps (Savoie department, including Villarembert) over 44 shooting days from late February to mid-May 2022, "
    "(6) Won the Academy Award for Best Original Screenplay for the 2023 film at the 96th Academy Awards in 2024, "
    "(7) Became the first female French filmmaker to be nominated for the Academy Award for Best Director, "
    "(8) Made their feature film debut at the 2013 Cannes Film Festival as part of the ACID programme. "
    "Provide the director's full name and include reference URLs supporting each category of information."
)

class DirectorExtraction(BaseModel):
    director_full_name: Optional[str] = None
    director_nationality: Optional[str] = None
    birth_year: Optional[str] = None
    education_institution: Optional[str] = None
    film_2016_title: Optional[str] = None
    film_2019_title: Optional[str] = None
    film_2023_title: Optional[str] = None
    film_2013_debut_title: Optional[str] = None
    background_sources: List[str] = Field(default_factory=list)
    awards_sources: List[str] = Field(default_factory=list)
    collaboration_sources_2016: List[str] = Field(default_factory=list)
    collaboration_sources_2019: List[str] = Field(default_factory=list)
    collaboration_sources_2023: List[str] = Field(default_factory=list)
    production_sources: List[str] = Field(default_factory=list)
    career_sources: List[str] = Field(default_factory=list)

def prompt_extract_director_info() -> str:
    return (
        "From the provided answer, extract the following structured information about the identified director and cite URLs grouped by category. "
        "Return null for any missing field. "
        "Fields to extract: "
        "1) director_full_name: The director's full name (e.g., 'First Last'); "
        "2) director_nationality: The nationality explicitly stated (e.g., 'French'); "
        "3) birth_year: The birth year (as a string, e.g., '1978'); "
        "4) education_institution: The institution stated for graduation (e.g., 'Beaux-Arts de Paris', 'ENSBA', 'École nationale supérieure des Beaux-Arts'); "
        "5) film_2016_title: The 2016 feature film title the director collaborated on with cinematographer Simon Beaufils; "
        "6) film_2019_title: The 2019 feature film title the director collaborated on with Simon Beaufils; "
        "7) film_2023_title: The 2023 film title associated with Cannes Palme d'Or and the detailed production notes; "
        "8) film_2013_debut_title: The director's first feature film title presented at Cannes 2013 ACID; "
        "9) background_sources: URL list that supports nationality, birth year, and education; "
        "10) awards_sources: URL list that supports 2023 Palme d'Or, 2024 Best Original Screenplay Oscar at the 96th Academy Awards, and the 'first female French filmmaker nominated for Best Director' claim; "
        "11) collaboration_sources_2016: URL list that supports the 2016 Simon Beaufils collaboration; "
        "12) collaboration_sources_2019: URL list that supports the 2019 Simon Beaufils collaboration; "
        "13) collaboration_sources_2023: URL list that supports the 2023 Simon Beaufils collaboration; "
        "14) production_sources: URL list that supports Savoie/French Alps, Villarembert, 44 shooting days, and late-February to mid-May 2022 for the 2023 film; "
        "15) career_sources: URL list that supports the 2013 Cannes ACID debut. "
        "Extraction rules for URLs: extract only valid URLs explicitly present in the answer text (including markdown link targets); do not invent URLs; if a URL is missing protocol, prepend 'http://'. "
    )

def _safe_name(extracted: DirectorExtraction) -> str:
    return extracted.director_full_name or "the director"

def _safe_title(title: Optional[str], fallback: str) -> str:
    return title.strip() if title else fallback

def _has_full_name(name: Optional[str]) -> bool:
    if not name:
        return False
    parts = [p for p in name.strip().split() if p]
    return len(parts) >= 2

async def add_answer_format_section(evaluator: Evaluator, parent, extracted: DirectorExtraction) -> None:
    node = evaluator.add_parallel(
        id="Answer_Format",
        desc="Verify the response provides the requested identification output",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_full_name(extracted.director_full_name),
        id="Director_Full_Name_Provided",
        desc="Response provides the director's full name (not only a surname or partial name)",
        parent=node,
        critical=True
    )

async def add_personal_background_section(evaluator: Evaluator, parent, extracted: DirectorExtraction) -> None:
    node = evaluator.add_parallel(
        id="Personal_Background",
        desc="Verify the director's nationality, birth year, and educational background",
        parent=parent,
        critical=True
    )
    name = _safe_name(extracted)
    bg_urls = extracted.background_sources

    nat_node = evaluator.add_leaf(
        id="French_Nationality",
        desc="The director must be French",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is French.",
        node=nat_node,
        sources=bg_urls,
        additional_instruction="Accept phrasing like 'French film director' or 'French filmmaker' as confirming nationality."
    )

    by_node = evaluator.add_leaf(
        id="Birth_Year_1978",
        desc="The director was born in 1978",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} was born in 1978.",
        node=by_node,
        sources=bg_urls,
        additional_instruction="If the page shows a full birthdate in 1978, that counts as support."
    )

    edu_node = evaluator.add_leaf(
        id="Beaux_Arts_Education",
        desc="The director graduated from Beaux-Arts de Paris",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} graduated from Beaux-Arts de Paris.",
        node=edu_node,
        sources=bg_urls,
        additional_instruction="Allow equivalent names such as ENSBA or École nationale supérieure des Beaux-Arts (École des Beaux-Arts in Paris)."
    )

    evaluator.add_custom_node(
        result=(len(bg_urls) > 0),
        id="Background_References",
        desc="Provide reference URL(s) supporting the personal background claims",
        parent=node,
        critical=True
    )

async def add_awards_section(evaluator: Evaluator, parent, extracted: DirectorExtraction) -> None:
    node = evaluator.add_parallel(
        id="2023_Film_Awards",
        desc="Verify the director's achievements with their 2023 film",
        parent=parent,
        critical=True
    )
    name = _safe_name(extracted)
    film_2023 = _safe_title(extracted.film_2023_title, "the director's 2023 film")
    awards_urls = extracted.awards_sources

    palme_node = evaluator.add_leaf(
        id="Palme_dOr_2023",
        desc="The director won the Palme d'Or at the 76th Cannes Film Festival (2023)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} won the Palme d'Or at the 76th Cannes Film Festival in 2023 for {film_2023}.",
        node=palme_node,
        sources=awards_urls,
        additional_instruction="Verify win status in 2023; accept exact phrasing like 'Palme d'Or winner'."
    )

    oscar_node = evaluator.add_leaf(
        id="Oscar_Best_Original_Screenplay",
        desc="The director won the Academy Award for Best Original Screenplay for the 2023 film at the 96th Academy Awards (2024)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In 2024 at the 96th Academy Awards, {name} won Best Original Screenplay for {film_2023}.",
        node=oscar_node,
        sources=awards_urls,
        additional_instruction="Confirm the award category 'Best Original Screenplay' and the ceremony number (96th) in 2024."
    )

    first_node = evaluator.add_leaf(
        id="First_French_Female_Director_Nominee",
        desc="The director became the first female French filmmaker to be nominated for the Academy Award for Best Director",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} became the first female French filmmaker ever nominated for the Academy Award for Best Director.",
        node=first_node,
        sources=awards_urls,
        additional_instruction="Allow equivalent phrasing like 'first French woman nominated for Best Director'."
    )

    evaluator.add_custom_node(
        result=(len(awards_urls) > 0),
        id="Awards_References",
        desc="Provide reference URL(s) supporting the awards/recognition claims",
        parent=node,
        critical=True
    )

async def add_collaboration_section(evaluator: Evaluator, parent, extracted: DirectorExtraction) -> None:
    node = evaluator.add_parallel(
        id="Cinematographer_Collaboration",
        desc="Verify collaboration with cinematographer Simon Beaufils on the specified films/years (implying at least three feature-film collaborations)",
        parent=parent,
        critical=True
    )
    name = _safe_name(extracted)

    coll16_node = evaluator.add_leaf(
        id="2016_Feature_Film_Collaboration",
        desc="The director collaborated with cinematographer Simon Beaufils on a feature film released in 2016",
        parent=node,
        critical=True
    )
    film_2016 = _safe_title(extracted.film_2016_title, "a feature film released in 2016")
    await evaluator.verify(
        claim=f"{name} collaborated with cinematographer Simon Beaufils on {film_2016} (released in 2016).",
        node=coll16_node,
        sources=extracted.collaboration_sources_2016,
        additional_instruction="Look for credit lines such as 'Director of Photography' or 'Cinematographer: Simon Beaufils' tied to the 2016 feature."
    )

    coll19_node = evaluator.add_leaf(
        id="2019_Feature_Film_Collaboration",
        desc="The director collaborated with cinematographer Simon Beaufils on a feature film released in 2019",
        parent=node,
        critical=True
    )
    film_2019 = _safe_title(extracted.film_2019_title, "a feature film released in 2019")
    await evaluator.verify(
        claim=f"{name} collaborated with cinematographer Simon Beaufils on {film_2019} (released in 2019).",
        node=coll19_node,
        sources=extracted.collaboration_sources_2019,
        additional_instruction="Confirm Simon Beaufils is credited as cinematographer/DP for the 2019 feature."
    )

    coll23_node = evaluator.add_leaf(
        id="2023_Feature_Film_Collaboration",
        desc="The director collaborated with cinematographer Simon Beaufils on the 2023 Palme d'Or-winning film",
        parent=node,
        critical=True
    )
    film_2023 = _safe_title(extracted.film_2023_title, "the 2023 film")
    await evaluator.verify(
        claim=f"{name} collaborated with cinematographer Simon Beaufils on {film_2023} (released in 2023).",
        node=coll23_node,
        sources=extracted.collaboration_sources_2023,
        additional_instruction="Confirm Simon Beaufils is the cinematographer for the 2023 feature."
    )

    evaluator.add_custom_node(
        result=(len(extracted.collaboration_sources_2016) + len(extracted.collaboration_sources_2019) + len(extracted.collaboration_sources_2023) > 0),
        id="Collaboration_References",
        desc="Provide reference URL(s) supporting the Simon Beaufils collaboration claims",
        parent=node,
        critical=True
    )

async def add_production_details_section(evaluator: Evaluator, parent, extracted: DirectorExtraction) -> None:
    node = evaluator.add_parallel(
        id="2023_Film_Production_Details",
        desc="Verify specified production details of the 2023 film",
        parent=parent,
        critical=True
    )
    film_2023 = _safe_title(extracted.film_2023_title, "the 2023 film")
    prod_urls = extracted.production_sources

    alps_node = evaluator.add_leaf(
        id="Shot_In_French_Alps_Savoie",
        desc="The 2023 film was shot in the French Alps, specifically in the Savoie department",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{film_2023} was shot in the French Alps, specifically in the Savoie department.",
        node=alps_node,
        sources=prod_urls,
        additional_instruction="Look for filming location descriptions naming 'Savoie', 'French Alps', or regional towns in Savoie."
    )

    villa_node = evaluator.add_leaf(
        id="Filming_In_Villarembert",
        desc="Filming of the 2023 film took place in Villarembert",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Filming for {film_2023} took place in Villarembert (Savoie).",
        node=villa_node,
        sources=prod_urls,
        additional_instruction="Confirm mentions of Villarembert as a filming location for the 2023 film."
    )

    days_node = evaluator.add_leaf(
        id="Shooting_Duration_44_Days",
        desc="The 2023 film had 44 shooting days",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{film_2023} had 44 shooting days.",
        node=days_node,
        sources=prod_urls,
        additional_instruction="Confirm total shooting days equals 44; accept phrasing like 'shot over 44 days'."
    )

    period_node = evaluator.add_leaf(
        id="Shooting_Period_Late_Feb_To_Mid_May_2022",
        desc="Shooting of the 2023 film took place from late February to mid-May 2022",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The shoot for {film_2023} ran from late February to mid-May 2022.",
        node=period_node,
        sources=prod_urls,
        additional_instruction="Confirm a start date in late February 2022 and an end date in mid-May 2022; reasonable calendar phrasing is acceptable."
    )

    evaluator.add_custom_node(
        result=(len(prod_urls) > 0),
        id="Production_References",
        desc="Provide reference URL(s) supporting the 2023 film production-detail claims",
        parent=node,
        critical=True
    )

async def add_career_milestones_section(evaluator: Evaluator, parent, extracted: DirectorExtraction) -> None:
    node = evaluator.add_parallel(
        id="Career_Milestones",
        desc="Verify the director's feature debut and Cannes/ACID presentation details",
        parent=parent,
        critical=True
    )
    name = _safe_name(extracted)
    debut_title = _safe_title(extracted.film_2013_debut_title, "the debut feature")
    career_urls = extracted.career_sources

    cannes_node = evaluator.add_leaf(
        id="Feature_Debut_2013_Cannes",
        desc="The director's first feature film was presented at the 2013 Cannes Film Festival",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name}'s first feature, {debut_title}, was presented at the 2013 Cannes Film Festival.",
        node=cannes_node,
        sources=career_urls,
        additional_instruction="Confirm presence at Cannes 2013; film title may be translated or have alternate titles."
    )

    acid_node = evaluator.add_leaf(
        id="ACID_Programme_2013",
        desc="The 2013 feature debut was presented as part of the ACID programme at Cannes",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{debut_title} was presented in the ACID programme at Cannes in 2013.",
        node=acid_node,
        sources=career_urls,
        additional_instruction="Look for ACID mentions (Association du Cinéma Indépendant pour sa Diffusion) tied to Cannes 2013."
    )

    evaluator.add_custom_node(
        result=(len(career_urls) > 0),
        id="Career_References",
        desc="Provide reference URL(s) supporting the career-milestone claims",
        parent=node,
        critical=True
    )

async def build_verification_tree(evaluator: Evaluator, extracted: DirectorExtraction) -> None:
    root = evaluator.find_node("root")
    dir_node = evaluator.add_parallel(
        id="Director_Identification",
        desc="Verify the response identifies a director meeting all specified criteria and provides required citations",
        parent=root,
        critical=True
    )
    await add_answer_format_section(evaluator, dir_node, extracted)
    await add_personal_background_section(evaluator, dir_node, extracted)
    await add_awards_section(evaluator, dir_node, extracted)
    await add_collaboration_section(evaluator, dir_node, extracted)
    await add_production_details_section(evaluator, dir_node, extracted)
    await add_career_milestones_section(evaluator, dir_node, extracted)

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
    evaluator = Evaluator()
    evaluator.initialize(
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_director_info(),
        template_class=DirectorExtraction,
        extraction_name="director_info"
    )

    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()