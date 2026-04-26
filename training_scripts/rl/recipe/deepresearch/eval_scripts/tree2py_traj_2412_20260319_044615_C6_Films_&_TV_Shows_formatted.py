import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hbo_2025_2027_slate"
TASK_DESCRIPTION = """Identify four HBO original series from the 2025-2027 production slate that meet the following specific criteria:

1. A medical drama series that premiered in January 2025, and provide its title, exact premiere date, and city setting.

2. A supernatural horror prequel series scheduled to premiere in October 2025, and provide its title, exact premiere date, and the franchise it belongs to.

3. A crime drama series premiering in September 2025 that consists of exactly 7 episodes in its first season, and provide its title and exact premiere date.

4. The Harry Potter television adaptation, and provide:
   - The scheduled premiere year
   - The production studio name and location where filming takes place
   - The names of the three actors cast as Harry Potter, Hermione Granger, and Ron Weasley
   - The name of the actor cast as Albus Dumbledore

For each series, provide a reference URL supporting the information."""


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class Series1Extraction(BaseModel):
    title: Optional[str] = None
    premiere_date: Optional[str] = None  # Expect exact date string (e.g., "January 12, 2025")
    city_setting: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Series2Extraction(BaseModel):
    title: Optional[str] = None
    premiere_date: Optional[str] = None  # Expect exact date string in October 2025
    franchise: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Series3Extraction(BaseModel):
    title: Optional[str] = None
    premiere_date: Optional[str] = None  # Expect exact date string in September 2025
    season1_episodes: Optional[str] = None  # Keep as string to be robust ("7", "seven", etc.)
    sources: List[str] = Field(default_factory=list)


class Series4Cast(BaseModel):
    harry: Optional[str] = None
    hermione: Optional[str] = None
    ron: Optional[str] = None
    dumbledore: Optional[str] = None


class Series4Extraction(BaseModel):
    scheduled_premiere_year: Optional[str] = None
    studio_name: Optional[str] = None
    studio_location: Optional[str] = None
    filming_began: Optional[str] = None  # Expect phrasing like "June 2025" or "Summer 2025"
    cast: Series4Cast = Field(default_factory=Series4Cast)
    first_cast_announced_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_series_1() -> str:
    return """
    From the answer, extract information for the HBO medical drama that premiered in January 2025.
    Return a JSON object with the following fields:
    - title: the series title
    - premiere_date: the exact premiere date string as given in the answer (e.g., "January 12, 2025")
    - city_setting: the specific city where the series is set (e.g., "Chicago")
    - sources: an array of URLs the answer cites for this series (only valid URLs explicitly present in the answer)
    If a field is not mentioned, set it to null (or [] for sources).
    """


def prompt_extract_series_2() -> str:
    return """
    From the answer, extract information for the HBO supernatural horror prequel scheduled to premiere in October 2025.
    Return a JSON object with:
    - title: the series title
    - premiere_date: the exact scheduled premiere date string as given in the answer (e.g., "October 17, 2025")
    - franchise: the established franchise or IP this prequel belongs to (e.g., "The Conjuring", "True Detective", etc.)
    - sources: an array of URLs the answer cites for this series
    If a field is not mentioned, set it to null (or [] for sources).
    """


def prompt_extract_series_3() -> str:
    return """
    From the answer, extract information for the HBO crime drama premiering in September 2025 with exactly 7 episodes in its first season.
    Return a JSON object with:
    - title: the series title
    - premiere_date: the exact premiere date string (e.g., "September 21, 2025")
    - season1_episodes: the number of episodes in season 1 as written in the answer (e.g., "7" or "seven")
    - sources: an array of URLs the answer cites for this series
    If a field is not mentioned, set it to null (or [] for sources).
    """


def prompt_extract_series_4() -> str:
    return """
    From the answer, extract information for the HBO Harry Potter television adaptation.
    Return a JSON object with:
    - scheduled_premiere_year: the scheduled premiere year (e.g., "2027")
    - studio_name: the production studio where filming takes place (e.g., "Warner Bros. Studios Leavesden")
    - studio_location: the geographic location of the studio (e.g., "Hertfordshire, UK")
    - filming_began: the time when filming began (e.g., "Summer 2025", "June 2025")
    - cast: an object with the following fields (strings or null):
        - harry
        - hermione
        - ron
        - dumbledore
    - first_cast_announced_name: the name of the first announced cast member (if provided in the answer)
    - sources: an array of URLs the answer cites for this series
    If a field is not mentioned, set it to null (or [] for sources).
    """


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def _has_at_least_one_url(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str) and u.strip():
            return True
    return False


def _safe_series_name(name: Optional[str]) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else "the series"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_series_1(evaluator: Evaluator, parent_node, s1: Series1Extraction) -> None:
    group = evaluator.add_parallel(
        id="series_1",
        desc="Series 1: medical drama that premiered in January 2025; provide title, exact premiere date, and city setting, with a supporting URL.",
        parent=parent_node,
        critical=False
    )

    # Critical existence checks (custom nodes)
    evaluator.add_custom_node(
        result=bool(s1.title and s1.title.strip()),
        id="s1_title_provided",
        desc="Provide the title for Series 1.",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(s1.city_setting and s1.city_setting.strip()),
        id="s1_city_setting_provided",
        desc="Provide the specific city setting for Series 1.",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_at_least_one_url(s1.sources),
        id="s1_reference_url",
        desc="Provide at least one verifiable reference URL supporting the Series 1 details.",
        parent=group,
        critical=True
    )

    # Leaves to verify with sources
    nodes_and_claims: List[tuple[str, List[str], Any, str]] = []

    n1 = evaluator.add_leaf(
        id="s1_is_hbo_original",
        desc="Series 1 is an HBO original series.",
        parent=group,
        critical=True
    )
    claim1 = f"The series {_safe_series_name(s1.title)} is an HBO original series (HBO or HBO/Max branded)."
    add1 = "Accept 'HBO Original' or equivalent branding (e.g., 'HBO Original', 'Max Original' under HBO). The page should clearly attribute the series to HBO/HBO-branded originals."
    nodes_and_claims.append((claim1, s1.sources, n1, add1))

    n2 = evaluator.add_leaf(
        id="s1_is_medical_drama",
        desc="Series 1 is a medical drama.",
        parent=group,
        critical=True
    )
    claim2 = f"The series {_safe_series_name(s1.title)} is a medical drama (hospital/medical setting, genre tag 'medical drama')."
    add2 = "Confirm the genre is medical drama (allow synonyms like 'hospital drama' or 'medical series')."
    nodes_and_claims.append((claim2, s1.sources, n2, add2))

    n3 = evaluator.add_leaf(
        id="s1_exact_premiere_date_in_jan_2025",
        desc="Provide the exact premiere date for Series 1, and the date is in January 2025.",
        parent=group,
        critical=True
    )
    if s1.premiere_date and s1.premiere_date.strip():
        claim3 = f"The series {_safe_series_name(s1.title)} premiered on {s1.premiere_date}, which is in January 2025."
    else:
        claim3 = f"The series {_safe_series_name(s1.title)} premiered on an exact day in January 2025."
    add3 = "Verify the precise day-of-month and ensure it falls in January 2025 (any time zone acceptable). The page must explicitly state an exact date in January 2025."
    nodes_and_claims.append((claim3, s1.sources, n3, add3))

    await evaluator.batch_verify(nodes_and_claims)


async def verify_series_2(evaluator: Evaluator, parent_node, s2: Series2Extraction) -> None:
    group = evaluator.add_parallel(
        id="series_2",
        desc="Series 2: supernatural horror prequel scheduled to premiere in October 2025; provide title, exact premiere date, and franchise/IP, with a supporting URL.",
        parent=parent_node,
        critical=False
    )

    # Critical existence checks (custom nodes)
    evaluator.add_custom_node(
        result=bool(s2.title and s2.title.strip()),
        id="s2_title_provided",
        desc="Provide the title for Series 2.",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(s2.franchise and s2.franchise.strip()),
        id="s2_franchise_ip_provided",
        desc="Identify the franchise/IP Series 2 belongs to.",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_at_least_one_url(s2.sources),
        id="s2_reference_url",
        desc="Provide at least one verifiable reference URL supporting the Series 2 details.",
        parent=group,
        critical=True
    )

    # Leaves to verify with sources
    nodes_and_claims: List[tuple[str, List[str], Any, str]] = []

    n1 = evaluator.add_leaf(
        id="s2_is_hbo_original",
        desc="Series 2 is an HBO original series.",
        parent=group,
        critical=True
    )
    claim1 = f"The series {_safe_series_name(s2.title)} is an HBO original series (HBO or HBO/Max branded)."
    add1 = "Accept 'HBO Original' or 'Max Original' under HBO. The page should explicitly attribute it as such."
    nodes_and_claims.append((claim1, s2.sources, n1, add1))

    n2 = evaluator.add_leaf(
        id="s2_is_supernatural_horror",
        desc="Series 2 is a supernatural horror series.",
        parent=group,
        critical=True
    )
    claim2 = f"The series {_safe_series_name(s2.title)} is a supernatural horror series."
    add2 = "Confirm supernatural and horror elements (allow 'supernatural thriller/horror')."
    nodes_and_claims.append((claim2, s2.sources, n2, add2))

    n3 = evaluator.add_leaf(
        id="s2_is_prequel_based_on_franchise",
        desc="Series 2 is a prequel series based on an established franchise/IP.",
        parent=group,
        critical=True
    )
    if s2.franchise and s2.franchise.strip():
        claim3 = f"The series {_safe_series_name(s2.title)} is a prequel within the '{s2.franchise}' franchise."
    else:
        claim3 = f"The series {_safe_series_name(s2.title)} is a prequel within an established franchise."
    add3 = "Confirm the page states it is a prequel and clearly ties it to a known franchise/IP."
    nodes_and_claims.append((claim3, s2.sources, n3, add3))

    n4 = evaluator.add_leaf(
        id="s2_exact_premiere_date_in_oct_2025",
        desc="Provide the exact premiere date for Series 2, and the date is in October 2025.",
        parent=group,
        critical=True
    )
    if s2.premiere_date and s2.premiere_date.strip():
        claim4 = f"The series {_safe_series_name(s2.title)} is scheduled to premiere on {s2.premiere_date}, which falls in October 2025."
    else:
        claim4 = f"The series {_safe_series_name(s2.title)} is scheduled to premiere on an exact day in October 2025."
    add4 = "Verify that an exact day-of-month in October 2025 is provided by the page (announced/scheduled is acceptable)."
    nodes_and_claims.append((claim4, s2.sources, n4, add4))

    await evaluator.batch_verify(nodes_and_claims)


async def verify_series_3(evaluator: Evaluator, parent_node, s3: Series3Extraction) -> None:
    group = evaluator.add_parallel(
        id="series_3",
        desc="Series 3: crime drama premiering in September 2025 with exactly 7 episodes in its first season; provide title and exact premiere date, with a supporting URL.",
        parent=parent_node,
        critical=False
    )

    # Critical existence checks (custom nodes)
    evaluator.add_custom_node(
        result=bool(s3.title and s3.title.strip()),
        id="s3_title_provided",
        desc="Provide the title for Series 3.",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_at_least_one_url(s3.sources),
        id="s3_reference_url",
        desc="Provide at least one verifiable reference URL supporting the Series 3 details.",
        parent=group,
        critical=True
    )

    # Leaves to verify with sources
    nodes_and_claims: List[tuple[str, List[str], Any, str]] = []

    n1 = evaluator.add_leaf(
        id="s3_is_hbo_original",
        desc="Series 3 is an HBO original series.",
        parent=group,
        critical=True
    )
    claim1 = f"The series {_safe_series_name(s3.title)} is an HBO original series (HBO or HBO/Max branded)."
    add1 = "Accept 'HBO Original' or 'Max Original' under HBO branding."
    nodes_and_claims.append((claim1, s3.sources, n1, add1))

    n2 = evaluator.add_leaf(
        id="s3_is_crime_drama",
        desc="Series 3 is a crime drama series.",
        parent=group,
        critical=True
    )
    claim2 = f"The series {_safe_series_name(s3.title)} is a crime drama."
    add2 = "Confirm 'crime drama' (allow variants like 'crime series', 'criminal investigation drama')."
    nodes_and_claims.append((claim2, s3.sources, n2, add2))

    n3 = evaluator.add_leaf(
        id="s3_exact_premiere_date_in_sep_2025",
        desc="Provide the exact premiere date for Series 3, and the date is in September 2025.",
        parent=group,
        critical=True
    )
    if s3.premiere_date and s3.premiere_date.strip():
        claim3 = f"The series {_safe_series_name(s3.title)} premiered on {s3.premiere_date}, which is in September 2025."
    else:
        claim3 = f"The series {_safe_series_name(s3.title)} premiered on an exact day in September 2025."
    add3 = "Verify the page provides an exact day-of-month in September 2025."
    nodes_and_claims.append((claim3, s3.sources, n3, add3))

    n4 = evaluator.add_leaf(
        id="s3_first_season_has_exactly_7_episodes",
        desc="Series 3 has exactly 7 episodes in its first season.",
        parent=group,
        critical=True
    )
    claim4 = f"Season 1 of {_safe_series_name(s3.title)} consists of exactly 7 episodes."
    add4 = "Confirm the first season episode count is exactly 7 (accept 'miniseries of 7 episodes' or equivalent)."
    nodes_and_claims.append((claim4, s3.sources, n4, add4))

    await evaluator.batch_verify(nodes_and_claims)


async def verify_series_4(evaluator: Evaluator, parent_node, s4: Series4Extraction) -> None:
    group = evaluator.add_parallel(
        id="series_4",
        desc="Series 4: the Harry Potter television adaptation; provide scheduled premiere year, production studio name and filming location, production timing, key cast, and a supporting URL.",
        parent=parent_node,
        critical=False
    )

    # Critical existence checks (custom nodes)
    evaluator.add_custom_node(
        result=_has_at_least_one_url(s4.sources),
        id="s4_reference_url",
        desc="Provide at least one verifiable reference URL supporting the Series 4 details.",
        parent=group,
        critical=True
    )

    nodes_and_claims: List[tuple[str, List[str], Any, str]] = []

    # HBO Original
    n1 = evaluator.add_leaf(
        id="s4_is_hbo_original",
        desc="Series 4 is an HBO original series.",
        parent=group,
        critical=True
    )
    claim1 = "The Harry Potter television adaptation is an HBO (or HBO/Max) original series."
    add1 = "Accept 'HBO Original' or 'Max Original' under HBO. The page must clearly indicate HBO/HBO-branded original."
    nodes_and_claims.append((claim1, s4.sources, n1, add1))

    # Is Harry Potter TV adaptation
    n2 = evaluator.add_leaf(
        id="s4_is_harry_potter_adaptation",
        desc="Series 4 is the Harry Potter television adaptation.",
        parent=group,
        critical=True
    )
    claim2 = "This series is the official television adaptation of the Harry Potter books/franchise."
    add2 = "The page should clearly describe the show as a TV series adaptation of Harry Potter."
    nodes_and_claims.append((claim2, s4.sources, n2, add2))

    # Scheduled premiere year 2027
    n3 = evaluator.add_leaf(
        id="s4_scheduled_premiere_year_is_2027",
        desc="Series 4 is scheduled to premiere in 2027 (and the answer provides this year).",
        parent=group,
        critical=True
    )
    claim3 = "The Harry Potter TV series is scheduled to premiere in 2027."
    add3 = "Verify the article or official info states year 2027 as the scheduled premiere window."
    nodes_and_claims.append((claim3, s4.sources, n3, add3))

    # Filming at WB Leavesden, Hertfordshire, UK
    n4 = evaluator.add_leaf(
        id="s4_filming_at_wb_leavesden_hertfordshire_uk",
        desc="Series 4 filming takes place at Warner Bros Studios Leavesden, located in Hertfordshire, UK (and the answer provides the studio name and location).",
        parent=group,
        critical=True
    )
    claim4 = "Filming takes place at Warner Bros. Studios Leavesden in Hertfordshire, UK."
    add4 = "The page should mention the studio 'Warner Bros. Studios Leavesden' and its location 'Hertfordshire, UK'."
    nodes_and_claims.append((claim4, s4.sources, n4, add4))

    # Filming began summer 2025
    n5 = evaluator.add_leaf(
        id="s4_filming_began_summer_2025",
        desc="Series 4 filming began in summer 2025.",
        parent=group,
        critical=True
    )
    claim5 = "Filming for the Harry Potter TV series began in summer 2025."
    add5 = "Treat June–August 2025 (Northern Hemisphere) as summer. Accept explicit statements like 'filming began in June/July/August 2025' or 'summer 2025'."
    nodes_and_claims.append((claim5, s4.sources, n5, add5))

    # Cast: Harry = Dominic McLaughlin
    n6 = evaluator.add_leaf(
        id="s4_cast_harry_is_dominic_mclaughlin",
        desc="Dominic McLaughlin is cast as Harry Potter.",
        parent=group,
        critical=True
    )
    claim6 = "Dominic McLaughlin is cast as Harry Potter in the HBO Harry Potter television series."
    add6 = "Allow minor name variants or middle names; confirm the role 'Harry Potter' is attributed to Dominic McLaughlin."
    nodes_and_claims.append((claim6, s4.sources, n6, add6))

    # Cast: Hermione = Arabella Stanton
    n7 = evaluator.add_leaf(
        id="s4_cast_hermione_is_arabella_stanton",
        desc="Arabella Stanton is cast as Hermione Granger.",
        parent=group,
        critical=True
    )
    claim7 = "Arabella Stanton is cast as Hermione Granger in the HBO Harry Potter television series."
    add7 = "Allow minor name variants; confirm the role 'Hermione Granger' is attributed to Arabella Stanton."
    nodes_and_claims.append((claim7, s4.sources, n7, add7))

    # Cast: Ron = Alastair Stout
    n8 = evaluator.add_leaf(
        id="s4_cast_ron_is_alastair_stout",
        desc="Alastair Stout is cast as Ron Weasley.",
        parent=group,
        critical=True
    )
    claim8 = "Alastair Stout is cast as Ron Weasley in the HBO Harry Potter television series."
    add8 = "Allow minor name variants; confirm the role 'Ron Weasley' is attributed to Alastair Stout."
    nodes_and_claims.append((claim8, s4.sources, n8, add8))

    # Cast: Dumbledore = John Lithgow
    n9 = evaluator.add_leaf(
        id="s4_cast_dumbledore_is_john_lithgow",
        desc="John Lithgow is cast as Albus Dumbledore.",
        parent=group,
        critical=True
    )
    claim9 = "John Lithgow is cast as Albus Dumbledore in the HBO Harry Potter television series."
    add9 = "Allow minor name variants; confirm the role 'Albus Dumbledore' is attributed to John Lithgow."
    nodes_and_claims.append((claim9, s4.sources, n9, add9))

    # Lithgow first cast member announced
    n10 = evaluator.add_leaf(
        id="s4_lithgow_first_cast_member_announced",
        desc="John Lithgow was the first cast member announced for Series 4.",
        parent=group,
        critical=True
    )
    claim10 = "John Lithgow was the first cast member announced for the HBO Harry Potter television series."
    add10 = "The page should explicitly or implicitly indicate he was the earliest/first announced cast member."
    nodes_and_claims.append((claim10, s4.sources, n10, add10))

    await evaluator.batch_verify(nodes_and_claims)


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
    Evaluate an answer for the HBO 2025–2027 slate task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: each series evaluated independently
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

    # Extract information for each of the four series in parallel
    s1_task = evaluator.extract(
        prompt=prompt_extract_series_1(),
        template_class=Series1Extraction,
        extraction_name="series_1_extraction",
    )
    s2_task = evaluator.extract(
        prompt=prompt_extract_series_2(),
        template_class=Series2Extraction,
        extraction_name="series_2_extraction",
    )
    s3_task = evaluator.extract(
        prompt=prompt_extract_series_3(),
        template_class=Series3Extraction,
        extraction_name="series_3_extraction",
    )
    s4_task = evaluator.extract(
        prompt=prompt_extract_series_4(),
        template_class=Series4Extraction,
        extraction_name="series_4_extraction",
    )

    s1, s2, s3, s4 = await asyncio.gather(s1_task, s2_task, s3_task, s4_task)

    # Build verification subtrees per series
    await verify_series_1(evaluator, root, s1)
    await verify_series_2(evaluator, root, s2)
    await verify_series_3(evaluator, root, s3)
    await verify_series_4(evaluator, root, s4)

    # Final structured summary
    return evaluator.get_summary()