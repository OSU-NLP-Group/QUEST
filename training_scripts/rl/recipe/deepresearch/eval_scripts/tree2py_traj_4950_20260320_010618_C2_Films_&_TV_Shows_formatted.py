import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "actress_2010_shyamalan_character"
TASK_DESCRIPTION = (
    "An actress made her film debut in 2006 at the age of 11 in a Christmas comedy film that starred Danny DeVito and "
    "Matthew Broderick. She later had a major recurring role in a psychological horror drama television series that "
    "premiered on March 18, 2013, on the A&E network. This series was based on characters from Robert Bloch's novel "
    "'Psycho.' The actress was born on January 9, 1995. What character did this actress play in the 2010 film directed "
    "by M. Night Shyamalan?"
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class FilmDebutInfo(BaseModel):
    film_title: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TVSeriesInfo(BaseModel):
    title: Optional[str] = None
    premiere_date: Optional[str] = None
    network: Optional[str] = None
    based_on: Optional[str] = None
    role_description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BirthInfo(BaseModel):
    birth_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Film2010Info(BaseModel):
    title: Optional[str] = None
    director: Optional[str] = None
    year: Optional[str] = None
    character_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    character_sources: List[str] = Field(default_factory=list)


class ActressExtraction(BaseModel):
    actress_name: Optional[str] = None
    film_debut: Optional[FilmDebutInfo] = None
    tv_series: Optional[TVSeriesInfo] = None
    birth: Optional[BirthInfo] = None
    film_2010: Optional[Film2010Info] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_actress_info() -> str:
    return """
Extract the following information from the answer:

1) actress_name: The full name of the actress identified in the answer.
2) film_debut:
   - film_title: The title of the film in which she made her film debut.
   - year: The year of that film debut.
   - sources: A list of URLs explicitly cited in the answer that support her film debut details (use only URLs present in the answer).
3) tv_series:
   - title: The title of the psychological horror drama TV series (premiered March 18, 2013) in which she had a major recurring role.
   - premiere_date: The exact premiere date mentioned for the series (if present).
   - network: The TV network mentioned (if present).
   - based_on: The 'based on' statement (if present), e.g., characters from 'Psycho'.
   - role_description: Any description indicating 'major recurring role' or similar.
   - sources: A list of URLs explicitly cited in the answer that support the TV series details (use only URLs present in the answer).
4) birth:
   - birth_date: Her birth date, as stated in the answer (e.g., 'January 9, 1995').
   - sources: A list of URLs explicitly cited in the answer that support the birth date (use only URLs present in the answer).
5) film_2010:
   - title: The title of the 2010 film directed by M. Night Shyamalan (as stated in the answer).
   - director: The director named for that film (as stated in the answer).
   - year: The release year of that film (as stated in the answer).
   - character_name: The character name the actress played in that 2010 film.
   - sources: A list of URLs explicitly cited in the answer that support the 2010 film details (use only URLs present in the answer).
   - character_sources: A list of URLs explicitly cited in the answer that support the character role in that 2010 film (use only URLs present in the answer). If not provided separately, leave empty.

Rules:
- Extract only what is explicitly present in the answer text. Do not invent any information. If an item is not mentioned, return null (for strings) or [] (for lists).
- For sources, return only valid URLs explicitly mentioned in the answer (including markdown links).
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_sources(*source_lists: Optional[List[str]]) -> List[str]:
    """Combine multiple source lists, preserving order, removing empties, and keeping uniques."""
    seen = set()
    combined: List[str] = []
    for sl in source_lists:
        if not sl:
            continue
        for u in sl:
            if not u:
                continue
            if u not in seen:
                combined.append(u)
                seen.add(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_actress_identification_checks(evaluator: Evaluator, parent_node, data: ActressExtraction) -> None:
    """
    Build the 'Actress_Identification' subtree:
      - Film_Debut_2006 (with URL existence + claim supported by URLs)
      - TV_Series_2013 (with URL existence + claim supported by URLs)
      - Birth_Date_1995 (with URL existence + claim supported by URLs)
      - Actress name presence check
    All nodes here are critical under a critical parent as required by the rubric.
    """
    act_name = data.actress_name or ""

    # Actress_Identification (critical, parallel)
    ident_node = evaluator.add_parallel(
        id="Actress_Identification",
        desc="Correctly identify the actress who meets all specified career and biographical constraints",
        parent=parent_node,
        critical=True,
    )

    # Simple existence check: actress name present
    evaluator.add_custom_node(
        result=bool(act_name.strip()),
        id="Actress_Name_Present",
        desc="The actress's name is explicitly provided in the answer",
        parent=ident_node,
        critical=True,
    )

    # Film_Debut_2006 (critical, parallel)
    fd = data.film_debut or FilmDebutInfo()
    film_debut_node = evaluator.add_parallel(
        id="Film_Debut_2006",
        desc="The actress made her film debut in 2006 in 'Deck the Halls'",
        parent=ident_node,
        critical=True,
    )

    # Film_Debut_URL existence (critical)
    evaluator.add_custom_node(
        result=bool(fd.sources),
        id="Film_Debut_URL",
        desc="URL reference confirming the actress's film debut details",
        parent=film_debut_node,
        critical=True,
    )

    # Film debut supported by cited sources (critical)
    film_debut_claim_leaf = evaluator.add_leaf(
        id="Film_Debut_Supported",
        desc="The actress made her film debut in 2006 in the Christmas comedy 'Deck the Halls'",
        parent=film_debut_node,
        critical=True,
    )
    film_debut_claim = (
        f"{act_name} made her film debut in 2006 in the Christmas comedy film 'Deck the Halls'. "
        f"If the sources mention her role (e.g., 'Mackenzie'), that still counts as her debut."
    )
    await evaluator.verify(
        claim=film_debut_claim,
        node=film_debut_claim_leaf,
        sources=fd.sources,
        additional_instruction="Verify that the sources explicitly indicate this was her film debut and that the film is Deck the Halls (2006). Allow minor wording variations.",
    )

    # TV_Series_2013 (critical, parallel)
    tv = data.tv_series or TVSeriesInfo()
    tv_node = evaluator.add_parallel(
        id="TV_Series_2013",
        desc="The actress had a major recurring role in a psychological horror drama series that premiered on March 18, 2013 on A&E, based on 'Psycho'",
        parent=ident_node,
        critical=True,
    )

    # TV_Series_URL existence (critical)
    evaluator.add_custom_node(
        result=bool(tv.sources),
        id="TV_Series_URL",
        desc="URL reference confirming the actress's TV series role and premiere details",
        parent=tv_node,
        critical=True,
    )

    # TV series supported by cited sources (critical)
    tv_claim_leaf = evaluator.add_leaf(
        id="TV_Series_Supported",
        desc="The actress had a major recurring role in the specified series (premiered Mar 18, 2013 on A&E; based on Psycho characters)",
        parent=tv_node,
        critical=True,
    )
    tv_title = tv.title or "the series"
    tv_claim = (
        f"{act_name} had a major recurring role in the psychological horror drama TV series '{tv_title}', "
        f"which premiered on March 18, 2013 on the A&E network and is based on characters from Robert Bloch's novel 'Psycho'."
    )
    await evaluator.verify(
        claim=tv_claim,
        node=tv_claim_leaf,
        sources=tv.sources,
        additional_instruction="Confirm: (1) her recurring role, (2) series premiere date 2013-03-18 on A&E, and (3) based on Psycho characters.",
    )

    # Birth_Date_1995 (critical, parallel)
    bd = data.birth or BirthInfo()
    birth_node = evaluator.add_parallel(
        id="Birth_Date_1995",
        desc="The actress was born on January 9, 1995",
        parent=ident_node,
        critical=True,
    )

    # Birth_Date_URL existence (critical)
    evaluator.add_custom_node(
        result=bool(bd.sources),
        id="Birth_Date_URL",
        desc="URL reference confirming the actress's birth date",
        parent=birth_node,
        critical=True,
    )

    # Birth date supported by cited sources (critical)
    birth_claim_leaf = evaluator.add_leaf(
        id="Birth_Date_Supported",
        desc="The actress was born on January 9, 1995",
        parent=birth_node,
        critical=True,
    )
    birth_claim = f"{act_name} was born on January 9, 1995."
    await evaluator.verify(
        claim=birth_claim,
        node=birth_claim_leaf,
        sources=bd.sources,
        additional_instruction="Verify the exact birth date matches January 9, 1995.",
    )


async def build_character_2010_checks(evaluator: Evaluator, parent_node, data: ActressExtraction) -> None:
    """
    Build the 'Character_Name_2010_Film' subtree:
      - Film_Verification_2010 (URL existence + film/year/director + actress appears)
      - Character_Name_Accuracy (URL existence + character name accuracy)
    """
    act_name = data.actress_name or ""
    f2010 = data.film_2010 or Film2010Info()

    # Character_Name_2010_Film (critical, parallel)
    film2010_node = evaluator.add_parallel(
        id="Character_Name_2010_Film",
        desc="Provide the correct character name the actress played in the 2010 M. Night Shyamalan film",
        parent=parent_node,
        critical=True,
    )

    # Film_Verification_2010 (critical, parallel)
    film_verif_node = evaluator.add_parallel(
        id="Film_Verification_2010",
        desc="The actress appeared in a film released in 2010 that was directed by M. Night Shyamalan",
        parent=film2010_node,
        critical=True,
    )

    # Film_2010_URL existence (critical)
    evaluator.add_custom_node(
        result=bool(f2010.sources),
        id="Film_2010_URL",
        desc="URL reference confirming the 2010 film and its director",
        parent=film_verif_node,
        critical=True,
    )

    # Film director+year+appearance supported (critical)
    film_supported_leaf = evaluator.add_leaf(
        id="Film_2010_Supported",
        desc="The 2010 film details (director M. Night Shyamalan) and actress appearance are supported by sources",
        parent=film_verif_node,
        critical=True,
    )
    if f2010.title:
        film_2010_claim = (
            f"{act_name} appeared in the 2010 film '{f2010.title}', which was directed by M. Night Shyamalan."
        )
    else:
        film_2010_claim = (
            f"{act_name} appeared in a film released in 2010 that was directed by M. Night Shyamalan."
        )
    await evaluator.verify(
        claim=film_2010_claim,
        node=film_supported_leaf,
        sources=f2010.sources,
        additional_instruction="Confirm both the director attribution (M. Night Shyamalan) and the 2010 release year, and that the actress appears in the film.",
    )

    # Character_Name_Accuracy (critical, parallel)
    char_node = evaluator.add_parallel(
        id="Character_Name_Accuracy",
        desc="The correct character name from the 2010 film is provided",
        parent=film2010_node,
        critical=True,
    )

    # Character_Name_URL existence: require both a character name and at least one relevant source (character_sources or film_2010 sources)
    char_sources_combined = _safe_sources(f2010.character_sources, f2010.sources)
    evaluator.add_custom_node(
        result=bool((f2010.character_name or "").strip()) and bool(char_sources_combined),
        id="Character_Name_URL",
        desc="URL reference confirming the character name in the 2010 film (or the film page itself is provided)",
        parent=char_node,
        critical=True,
    )

    # Character name supported by sources (critical)
    char_supported_leaf = evaluator.add_leaf(
        id="Character_Name_Supported",
        desc="The provided character name for the 2010 film is correct",
        parent=char_node,
        critical=True,
    )
    film_title_for_char = f2010.title or "the 2010 film"
    char_claim = (
        f"In {film_title_for_char} (2010), {act_name} played the character '{f2010.character_name or ''}'."
    )
    await evaluator.verify(
        claim=char_claim,
        node=char_supported_leaf,
        sources=char_sources_combined,
        additional_instruction="Verify the mapping from actress to character for the specified 2010 film.",
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
    Evaluate an answer for the 'actress_2010_shyamalan_character' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # We'll add a critical sequential child as per rubric
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_actress_info(),
        template_class=ActressExtraction,
        extraction_name="actress_info",
    )

    # Build root-level 'Complete_Task' as a critical sequential node
    complete_task = evaluator.add_sequential(
        id="Complete_Task",
        desc="Successfully identify the actress based on career constraints and provide the character name from the 2010 film",
        parent=root,
        critical=True,
    )

    # Subtree 1: Actress Identification
    await build_actress_identification_checks(evaluator, complete_task, extracted)

    # Subtree 2: Character Name for 2010 Shyamalan Film
    await build_character_2010_checks(evaluator, complete_task, extracted)

    # Return summary
    return evaluator.get_summary()