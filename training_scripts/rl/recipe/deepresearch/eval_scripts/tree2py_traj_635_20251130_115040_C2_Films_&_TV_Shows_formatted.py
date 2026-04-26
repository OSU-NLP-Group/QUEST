import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_adaptation_2024_film"
TASK_DESCRIPTION = """
Identify the 2024 theatrical film that adapts a Broadway musical, was directed by Jon M. Chu, and was released in the United States on November 22, 2024. For this film, provide the following information:

1. The film's title
2. The theatrical runtime in minutes
3. The on-screen title as it appears in the film (which may differ from the general release title)
4. The name of the director
5. The names of the two lead actresses and the character names they each portray (specifically, identify who plays Elphaba Thropp and who plays Galinda/Glinda Upland)

For each piece of information provided, include reference URLs that support your answer.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmIdentificationExtract(BaseModel):
    """Basic film identification and sources supporting core criteria."""
    title: Optional[str] = None
    identification_urls: List[str] = Field(default_factory=list)


class ProductionDetailsExtract(BaseModel):
    """Production-related details and supporting URLs."""
    runtime_minutes: Optional[str] = None
    runtime_urls: List[str] = Field(default_factory=list)
    on_screen_title: Optional[str] = None
    on_screen_title_urls: List[str] = Field(default_factory=list)


class PersonnelDetailsExtract(BaseModel):
    """Key personnel and lead actresses with role mapping and supporting URLs."""
    director_name: Optional[str] = None
    director_urls: List[str] = Field(default_factory=list)

    elphaba_actress_name: Optional[str] = None
    elphaba_character_name: Optional[str] = None
    elphaba_urls: List[str] = Field(default_factory=list)

    glinda_actress_name: Optional[str] = None
    glinda_character_name: Optional[str] = None
    glinda_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_film_identification() -> str:
    return """
    Extract the single film identified in the answer that meets ALL of the following criteria:
    - It is a 2024 theatrical release.
    - It adapts a Broadway musical.
    - It had a U.S. release date of November 22, 2024.
    - It was directed by Jon M. Chu.

    Return:
    - title: The film's title as stated in the answer.
    - identification_urls: A list of all URL(s) cited in the answer that support any of the identification criteria above
      (e.g., official site, studio page, trade press coverage, Wikipedia/IMDb/Box Office Mojo pages, etc.).
      Extract only actual URLs explicitly present in the answer (plain links or markdown links). Do not invent URLs.

    If multiple films are mentioned, choose the one that meets the criteria above. If the answer does not provide such a film,
    return null for the title and an empty list for identification_urls.
    """


def prompt_extract_production_details() -> str:
    return """
    For the identified film in the answer, extract the following production details exactly as stated in the answer:

    - runtime_minutes: The theatrical runtime in minutes (return as a string, e.g., "166").
    - runtime_urls: URL(s) cited for the runtime. Extract only actual URLs present in the answer.
    - on_screen_title: The on-screen title as it appears in the film (may differ from general release title). Return
      exactly what the answer states for the on-screen title, if provided.
    - on_screen_title_urls: URL(s) cited for the on-screen title. Extract only actual URLs present in the answer.

    If a field is not mentioned, set it to null (for text fields) or an empty list (for URLs).
    """


def prompt_extract_personnel_details() -> str:
    return """
    For the identified film in the answer, extract the key personnel and lead actresses with their roles:

    - director_name: The name of the director (return as a string exactly as written in the answer).
    - director_urls: URL(s) supporting the director attribution. Extract only actual URLs present in the answer.
    - elphaba_actress_name: The actress who plays Elphaba Thropp (return as a string).
    - elphaba_character_name: The exact character name given in the answer for that actress (e.g., "Elphaba Thropp").
    - elphaba_urls: URL(s) supporting this casting attribution. Extract only actual URLs present in the answer.
    - glinda_actress_name: The actress who plays Galinda/Glinda Upland (return as a string).
    - glinda_character_name: The exact character name given in the answer for that actress (e.g., "Galinda Upland" or "Glinda Upland").
    - glinda_urls: URL(s) supporting this casting attribution. Extract only actual URLs present in the answer.

    If any field is not mentioned, set it to null (for text fields) or an empty list (for URLs).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_film_identification(
    evaluator: Evaluator,
    parent_node,
    film: FilmIdentificationExtract,
) -> None:
    """
    Build and verify the FilmIdentification subtree:
    - Ensure film title and sources exist
    - Verify Broadway musical adaptation
    - Verify US release date of Nov 22, 2024
    - Verify it is a 2024 theatrical release
    """
    node = evaluator.add_sequential(
        id="FilmIdentification",
        desc="Provide the film title and verify (with supporting reference URL(s)) that it is a 2024 theatrical release that adapts a Broadway musical and had a U.S. release date of November 22, 2024",
        parent=parent_node,
        critical=True,
    )

    # Existence gates (critical)
    evaluator.add_custom_node(
        result=bool(film.title and film.title.strip()),
        id="FilmTitleProvided",
        desc="Film title is provided",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(film.identification_urls),
        id="FilmIdentificationSourcesProvided",
        desc="Identification sources (URLs) are provided",
        parent=node,
        critical=True,
    )

    # 1) Broadway adaptation check
    adapts_node = evaluator.add_leaf(
        id="FilmBroadwayAdaptation",
        desc="The identified film adapts a Broadway musical",
        parent=node,
        critical=True,
    )
    adapts_claim = f"The film '{film.title}' adapts a Broadway musical."
    await evaluator.verify(
        claim=adapts_claim,
        node=adapts_node,
        sources=film.identification_urls,
        additional_instruction="Confirm that the film is based on or adapts a Broadway musical. Accept explicit mentions like 'based on the Broadway musical Wicked'.",
    )

    # 2) US release date check
    release_date_node = evaluator.add_leaf(
        id="FilmUSReleaseDate_2024_11_22",
        desc="The identified film had a U.S. release date of November 22, 2024",
        parent=node,
        critical=True,
    )
    release_claim = f"The film '{film.title}' was released in the United States on November 22, 2024."
    await evaluator.verify(
        claim=release_claim,
        node=release_date_node,
        sources=film.identification_urls,
        additional_instruction="Verify the U.S. theatrical release date on authoritative sources (e.g., official studio info, trade press, Wikipedia/Box Office Mojo).",
    )

    # 3) 2024 theatrical release check
    year_node = evaluator.add_leaf(
        id="FilmTheatricalReleaseYear_2024",
        desc="The identified film is a 2024 theatrical release",
        parent=node,
        critical=True,
    )
    year_claim = f"The film '{film.title}' was theatrically released in 2024."
    await evaluator.verify(
        claim=year_claim,
        node=year_node,
        sources=film.identification_urls,
        additional_instruction="Confirm the film's theatrical release year is 2024 (opening date or general release date within 2024).",
    )


async def verify_production_details(
    evaluator: Evaluator,
    parent_node,
    film_title: Optional[str],
    prod: ProductionDetailsExtract,
) -> None:
    """
    Build and verify ProductionDetails subtree:
    - Runtime (provided + accurate)
    - On-screen title (provided + supported)
    """
    prod_node = evaluator.add_parallel(
        id="ProductionDetails",
        desc="Report required production specifications for the identified film, each with supporting reference URL(s)",
        parent=parent_node,
        critical=False,
    )

    # Runtime (critical under ProductionDetails)
    runtime_node = evaluator.add_sequential(
        id="Runtime",
        desc="Report the theatrical runtime in minutes with supporting reference URL(s)",
        parent=prod_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(prod.runtime_minutes and prod.runtime_minutes.strip()),
        id="RuntimeProvided",
        desc="Runtime value is provided",
        parent=runtime_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(prod.runtime_urls),
        id="RuntimeSourcesProvided",
        desc="Runtime sources (URLs) are provided",
        parent=runtime_node,
        critical=True,
    )
    runtime_leaf = evaluator.add_leaf(
        id="RuntimeAccurate",
        desc="The runtime is accurately cited",
        parent=runtime_node,
        critical=True,
    )
    runtime_claim = f"The theatrical runtime of '{film_title or 'the film'}' is {prod.runtime_minutes} minutes."
    await evaluator.verify(
        claim=runtime_claim,
        node=runtime_leaf,
        sources=prod.runtime_urls,
        additional_instruction="Verify the runtime as stated. Allow minor discrepancies due to rounding but prefer the widely cited official figure.",
    )

    # On-screen title (critical under ProductionDetails)
    onscreen_node = evaluator.add_sequential(
        id="OnScreenTitle",
        desc="Report the on-screen title as it appears in the film (which may differ from the general release title) with supporting reference URL(s)",
        parent=prod_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(prod.on_screen_title and prod.on_screen_title.strip()),
        id="OnScreenTitleProvided",
        desc="On-screen title is provided",
        parent=onscreen_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(prod.on_screen_title_urls),
        id="OnScreenTitleSourcesProvided",
        desc="On-screen title sources (URLs) are provided",
        parent=onscreen_node,
        critical=True,
    )
    onscreen_leaf = evaluator.add_leaf(
        id="OnScreenTitleSupported",
        desc="The on-screen title is accurately cited",
        parent=onscreen_node,
        critical=True,
    )
    onscreen_claim = f"The on-screen title as it appears in the film is '{prod.on_screen_title}'."
    await evaluator.verify(
        claim=onscreen_claim,
        node=onscreen_leaf,
        sources=prod.on_screen_title_urls,
        additional_instruction="Confirm the film's on-screen title (e.g., as shown in opening or title card). Accept credible sources (official posts, trade press, reviews) that explicitly mention the on-screen title.",
    )


async def verify_key_personnel(
    evaluator: Evaluator,
    parent_node,
    film_title: Optional[str],
    ppl: PersonnelDetailsExtract,
) -> None:
    """
    Build and verify KeyPersonnel subtree:
    - Director (provided + accurate)
    - LeadActresses (Elphaba and Glinda) provided + supported
    """
    key_node = evaluator.add_parallel(
        id="KeyPersonnel",
        desc="Report required personnel/cast details for the identified film, each with supporting reference URL(s)",
        parent=parent_node,
        critical=False,
    )

    # Director (critical)
    director_node = evaluator.add_sequential(
        id="Director",
        desc="Name the director of the film with supporting reference URL(s)",
        parent=key_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(ppl.director_name and ppl.director_name.strip()),
        id="DirectorProvided",
        desc="Director name is provided",
        parent=director_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(ppl.director_urls),
        id="DirectorSourcesProvided",
        desc="Director sources (URLs) are provided",
        parent=director_node,
        critical=True,
    )
    director_leaf = evaluator.add_leaf(
        id="DirectorAccurate",
        desc="Director is accurately cited",
        parent=director_node,
        critical=True,
    )
    director_claim = f"The film '{film_title or 'the film'}' was directed by {ppl.director_name}."
    await evaluator.verify(
        claim=director_claim,
        node=director_leaf,
        sources=ppl.director_urls,
        additional_instruction="Verify the director attribution from authoritative sources. Allow minor name variations (e.g., middle initial) to be considered equivalent.",
    )

    # LeadActresses (non-critical main)
    leads_node = evaluator.add_parallel(
        id="LeadActresses",
        desc="Identify the two lead actresses and the character names they portray (Elphaba Thropp; Galinda/Glinda Upland), with supporting reference URL(s)",
        parent=key_node,
        critical=False,
    )

    # Elphaba (critical under leads)
    elphaba_node = evaluator.add_sequential(
        id="ElphabaActress",
        desc="Name the actress who plays Elphaba Thropp and provide the character name with supporting reference URL(s)",
        parent=leads_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(ppl.elphaba_actress_name and ppl.elphaba_actress_name.strip() and ppl.elphaba_character_name and ppl.elphaba_character_name.strip()),
        id="ElphabaProvided",
        desc="Elphaba actress and character are provided",
        parent=elphaba_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(ppl.elphaba_urls),
        id="ElphabaSourcesProvided",
        desc="Elphaba sources (URLs) are provided",
        parent=elphaba_node,
        critical=True,
    )
    elphaba_leaf = evaluator.add_leaf(
        id="ElphabaAccurate",
        desc="Elphaba casting attribution is accurately cited",
        parent=elphaba_node,
        critical=True,
    )
    elphaba_claim = (
        f"{ppl.elphaba_actress_name} portrays {ppl.elphaba_character_name} in the film '{film_title or 'the film'}'."
    )
    await evaluator.verify(
        claim=elphaba_claim,
        node=elphaba_leaf,
        sources=ppl.elphaba_urls,
        additional_instruction="Confirm the casting attribution. Accept reasonable character name variants referencing Elphaba Thropp.",
    )

    # Glinda (critical under leads)
    glinda_node = evaluator.add_sequential(
        id="GlindaActress",
        desc="Name the actress who plays Galinda/Glinda Upland and provide the character name with supporting reference URL(s)",
        parent=leads_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(ppl.glinda_actress_name and ppl.glinda_actress_name.strip() and ppl.glinda_character_name and ppl.glinda_character_name.strip()),
        id="GlindaProvided",
        desc="Glinda actress and character are provided",
        parent=glinda_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(ppl.glinda_urls),
        id="GlindaSourcesProvided",
        desc="Glinda sources (URLs) are provided",
        parent=glinda_node,
        critical=True,
    )
    glinda_leaf = evaluator.add_leaf(
        id="GlindaAccurate",
        desc="Glinda casting attribution is accurately cited",
        parent=glinda_node,
        critical=True,
    )
    glinda_claim = (
        f"{ppl.glinda_actress_name} portrays {ppl.glinda_character_name} in the film '{film_title or 'the film'}'."
    )
    await evaluator.verify(
        claim=glinda_claim,
        node=glinda_leaf,
        sources=ppl.glinda_urls,
        additional_instruction="Confirm the casting attribution. Accept 'Galinda' and 'Glinda' variants, and recognize 'Glinda Upland' naming.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Broadway musical film adaptation (2024) task.
    """
    # Initialize evaluator with a sequential root to gate later checks on initial identification
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured information (in parallel)
    film_extract_task = evaluator.extract(
        prompt=prompt_extract_film_identification(),
        template_class=FilmIdentificationExtract,
        extraction_name="film_identification",
    )
    prod_extract_task = evaluator.extract(
        prompt=prompt_extract_production_details(),
        template_class=ProductionDetailsExtract,
        extraction_name="production_details",
    )
    ppl_extract_task = evaluator.extract(
        prompt=prompt_extract_personnel_details(),
        template_class=PersonnelDetailsExtract,
        extraction_name="personnel_details",
    )

    film_info, prod_info, ppl_info = await asyncio.gather(
        film_extract_task, prod_extract_task, ppl_extract_task
    )

    # Build and run verification subtrees in order
    await verify_film_identification(evaluator, root, film_info)
    await verify_production_details(evaluator, root, film_info.title, prod_info)
    await verify_key_personnel(evaluator, root, film_info.title, ppl_info)

    # Return evaluation summary
    return evaluator.get_summary()