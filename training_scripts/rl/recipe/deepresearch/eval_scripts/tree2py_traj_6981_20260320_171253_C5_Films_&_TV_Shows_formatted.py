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
TASK_ID = "film_identification_asc_2023"
TASK_DESCRIPTION = """
Identify the 2023 theatrical film that meets all of the following criteria:

1. The film's cinematographer won the Outstanding Achievement in Cinematography in Theatrical Releases award at the 38th American Society of Cinematographers Awards ceremony, which was held on March 3, 2024.

2. The film grossed over $900 million at the worldwide box office in 2023, according to Box Office Mojo.

3. The film was released by Universal Pictures.

4. The film's director had directed at least 10 feature-length films prior to directing this film.

5. This film represents at least the third collaboration between the director and the cinematographer.

Please provide the film's title along with supporting reference URLs for each criterion.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class FilmAnswerExtraction(BaseModel):
    film_title: Optional[str] = None
    director_name: Optional[str] = None
    cinematographer_name: Optional[str] = None

    asc_award_urls: List[str] = Field(default_factory=list)
    box_office_urls: List[str] = Field(default_factory=list)
    distribution_urls: List[str] = Field(default_factory=list)
    release_urls: List[str] = Field(default_factory=list)
    director_filmography_urls: List[str] = Field(default_factory=list)
    collaboration_urls: List[str] = Field(default_factory=list)
    technical_format_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_film_info() -> str:
    return """
    Extract the single 2023 theatrical film the answer proposes as meeting all specified criteria.
    Return the following fields:

    - film_title: exact film title mentioned (string)
    - director_name: the director's name (string)
    - cinematographer_name: the cinematographer's name (string)

    For each criterion, extract the list of reference URLs explicitly provided in the answer (only actual URLs; ignore citations without URLs):
    - asc_award_urls: URLs supporting that the cinematographer won the ASC Outstanding Achievement in Cinematography (Theatrical Releases/Theatrical Feature Film) at the 38th ASC Awards (2024).
    - box_office_urls: URLs (preferably Box Office Mojo) supporting that the film grossed over $900 million worldwide in 2023.
    - distribution_urls: URLs supporting that the film was released/distributed by Universal Pictures.
    - release_urls: URLs supporting that the film received a theatrical release in 2023.
    - director_filmography_urls: URLs supporting that the director had at least 10 prior feature-length films before this film.
    - collaboration_urls: URLs supporting that this film is at least the third collaboration between the director and the cinematographer.
    - technical_format_urls: URLs supporting that the film was shot on 65mm large-format film (optional/bonus).

    If any field is missing, set it to null or an empty list as appropriate. Do not fabricate or infer data not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


def _has_bom_url(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    lowered = [u.lower() for u in urls if isinstance(u, str)]
    return any("boxofficemojo.com" in u for u in lowered)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def add_basic_info_checks(evaluator: Evaluator, parent_node, info: FilmAnswerExtraction) -> None:
    basic_node = evaluator.add_sequential(
        id="Basic_Info",
        desc="Basic identification info is present (film title, director, cinematographer)",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty_str(info.film_title),
        id="film_title_provided",
        desc="Film title is provided",
        parent=basic_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty_str(info.director_name),
        id="director_name_provided",
        desc="Director name is provided",
        parent=basic_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty_str(info.cinematographer_name),
        id="cinematographer_name_provided",
        desc="Cinematographer name is provided",
        parent=basic_node,
        critical=True
    )


async def add_asc_award_check(evaluator: Evaluator, parent_node, info: FilmAnswerExtraction) -> None:
    node = evaluator.add_sequential(
        id="ASC_Award_Recognition",
        desc="Cinematographer won ASC Outstanding Achievement in Cinematography (Theatrical Releases) at the 38th ASC Awards (held March 3, 2024)",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(info.asc_award_urls),
        id="asc_sources_provided",
        desc="At least one ASC award reference URL is provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="asc_award_supported",
        desc="ASC award win is supported by cited sources",
        parent=node,
        critical=True
    )

    film = info.film_title or ""
    dp = info.cinematographer_name or ""
    claim = (
        f"{dp} won the American Society of Cinematographers (ASC) Outstanding Achievement in Cinematography "
        f"in Theatrical Feature Film (aka Theatrical Releases) for '{film}' at the 38th ASC Awards "
        f"(ceremony held March 3, 2024)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.asc_award_urls,
        additional_instruction=(
            "Accept category name variants such as 'Theatrical Feature Film' or 'Theatrical Releases'. "
            "The evidence should clearly indicate that the cinematographer won (not just nominated) "
            "at the 38th ASC Awards in 2024 for the specified film. If the source is from the ASC website, "
            "press release, or reputable coverage listing winners, that is sufficient. "
            "Do not penalize if the page does not explicitly print the ceremony date, as long as it unambiguously "
            "indicates the 38th ASC Awards winners in 2024."
        )
    )


async def add_box_office_check(evaluator: Evaluator, parent_node, info: FilmAnswerExtraction) -> None:
    node = evaluator.add_sequential(
        id="Box_Office_Achievement",
        desc="Film grossed over $900M worldwide in 2023 according to Box Office Mojo",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(info.box_office_urls),
        id="box_office_sources_provided",
        desc="At least one box office reference URL is provided",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_bom_url(info.box_office_urls),
        id="box_office_has_bom_source",
        desc="At least one Box Office Mojo URL is provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="box_office_over_900m_2023",
        desc="Box Office Mojo source(s) support > $900M worldwide in 2023",
        parent=node,
        critical=True
    )

    film = info.film_title or ""
    claim = (
        f"According to Box Office Mojo, '{film}' earned over $900,000,000 (over $900 million) in worldwide box office in 2023."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.box_office_urls,
        additional_instruction=(
            "Prefer evidence from Box Office Mojo. Accept the '2023 Worldwide Grosses' page or a film-specific "
            "page that clearly indicates the 2023 worldwide gross total exceeding $900M. If the film page only shows "
            "lifetime grosses without a 2023 breakdown, look for BOM's 2023 yearly ranking/list. The claim should be "
            "considered supported if the BOM page(s) explicitly show > $900M for 2023."
        )
    )


async def add_distribution_check(evaluator: Evaluator, parent_node, info: FilmAnswerExtraction) -> None:
    node = evaluator.add_sequential(
        id="Distribution_Studio",
        desc="Film was released/distributed by Universal Pictures",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(info.distribution_urls),
        id="distribution_sources_provided",
        desc="At least one distribution/studio reference URL is provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="released_by_universal",
        desc="Sources support that Universal Pictures released/distributed the film",
        parent=node,
        critical=True
    )

    film = info.film_title or ""
    claim = f"'{film}' was released (i.e., distributed) by Universal Pictures."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.distribution_urls,
        additional_instruction=(
            "Confirm that Universal Pictures is listed as the distributor/releasing studio. "
            "Accept phrasing like 'distributed by Universal Pictures' or 'released by Universal'."
        )
    )


async def add_release_period_check(evaluator: Evaluator, parent_node, info: FilmAnswerExtraction) -> None:
    node = evaluator.add_sequential(
        id="Release_Period",
        desc="Film received a theatrical release in 2023",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(info.release_urls),
        id="release_sources_provided",
        desc="At least one release-date reference URL is provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="theatrical_release_in_2023",
        desc="Sources support that the film had a theatrical release in 2023",
        parent=node,
        critical=True
    )

    film = info.film_title or ""
    claim = f"'{film}' had a theatrical release in 2023 (in at least one country/territory)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.release_urls,
        additional_instruction=(
            "Verify that the release date(s) include a theatrical release in calendar year 2023. "
            "A release in any country in 2023 counts. Accept phrases like 'theatrically released in 2023'."
        )
    )


async def add_director_prior_work_check(evaluator: Evaluator, parent_node, info: FilmAnswerExtraction) -> None:
    node = evaluator.add_sequential(
        id="Director_Prior_Work",
        desc="Director had directed at least 10 feature-length films prior to this film",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(info.director_filmography_urls),
        id="director_filmography_sources_provided",
        desc="At least one director filmography reference URL is provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="director_prior_10_features",
        desc="Sources support that director had >= 10 prior feature-length films before this film",
        parent=node,
        critical=True
    )

    film = info.film_title or ""
    director = info.director_name or ""
    claim = (
        f"Before directing '{film}', {director} had already directed at least 10 feature-length films "
        f"(count only feature-length films; exclude shorts, TV episodes, commercials)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.director_filmography_urls,
        additional_instruction=(
            "Use reputable filmography sources (e.g., official sites, major databases, reputable publications). "
            "Count only feature-length directing credits prior to the given film; do not count shorts, TV episodes, "
            "music videos, commercials, or producer/writer credits."
        )
    )


async def add_collaboration_history_check(evaluator: Evaluator, parent_node, info: FilmAnswerExtraction) -> None:
    node = evaluator.add_sequential(
        id="Collaboration_History",
        desc="This film is at least the third collaboration between the director and the cinematographer",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(info.collaboration_urls),
        id="collab_sources_provided",
        desc="At least one collaboration-history reference URL is provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="at_least_third_collaboration",
        desc="Sources support that the director and cinematographer have collaborated on at least three films including this one",
        parent=node,
        critical=True
    )

    film = info.film_title or ""
    director = info.director_name or ""
    dp = info.cinematographer_name or ""
    claim = (
        f"The collaboration between director {director} and cinematographer {dp} includes at least three feature films, "
        f"with '{film}' being at least their third collaboration."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.collaboration_urls,
        additional_instruction=(
            "Look for lists or filmography pages that enumerate the director–cinematographer collaborations. "
            "If at least two earlier feature collaborations are listed prior to the target film (making the target "
            "film the third or later), then this claim is supported."
        )
    )


async def add_technical_format_check(evaluator: Evaluator, parent_node, info: FilmAnswerExtraction) -> None:
    node = evaluator.add_sequential(
        id="Technical_Format",
        desc="The film was shot on 65mm large-format film stock (bonus, non-critical)",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_has_urls(info.technical_format_urls),
        id="tech_format_sources_provided",
        desc="At least one technical-format reference URL is provided",
        parent=node,
        critical=True  # Critical within this optional subtree; the subtree itself is non-critical overall
    )

    leaf = evaluator.add_leaf(
        id="shot_on_65mm",
        desc="Sources support that the film was shot on 65mm large-format (e.g., 5-perf 65mm, IMAX 65mm/15-perf)",
        parent=node,
        critical=True
    )

    film = info.film_title or ""
    claim = (
        f"'{film}' was shot on 65mm large-format film stock (e.g., 5-perf 65mm and/or 15-perf IMAX 65mm)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.technical_format_urls,
        additional_instruction=(
            "Accept evidence that mentions 65mm negative formats, 5-perf 65mm (65mm Panavision), or IMAX 65mm (15-perf). "
            "References to 'large format 65mm' are sufficient."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the 2023 film identification task using the Mind2Web2 framework.
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

    # 1) Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_film_info(),
        template_class=FilmAnswerExtraction,
        extraction_name="film_answer_extraction"
    )

    # 2) Build verification tree
    film_node = evaluator.add_parallel(
        id="Film_Identification",
        desc="Correctly identify the 2023 theatrical film that meets all specified criteria and provide supporting reference URLs",
        parent=root,
        critical=False  # Allow inclusion of a non-critical bonus check
    )

    # Core prerequisites
    await add_basic_info_checks(evaluator, film_node, extracted)

    # Critical criteria
    await add_asc_award_check(evaluator, film_node, extracted)
    await add_box_office_check(evaluator, film_node, extracted)
    await add_distribution_check(evaluator, film_node, extracted)
    await add_release_period_check(evaluator, film_node, extracted)
    await add_director_prior_work_check(evaluator, film_node, extracted)
    await add_collaboration_history_check(evaluator, film_node, extracted)

    # Bonus (non-critical)
    await add_technical_format_check(evaluator, film_node, extracted)

    # 3) Return structured evaluation summary
    return evaluator.get_summary()