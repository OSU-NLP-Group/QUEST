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
TASK_ID = "actress_2026_oscar_all_criteria"
TASK_DESCRIPTION = """
Identify the name of the actress who meets all of the following criteria:

Personal Background:
- Born in Killarney, County Kerry, Ireland
- Born in December 1989

2026 Academy Award Achievement:
- Won the Academy Award for Best Actress at the 98th Academy Awards
- The 98th Academy Awards ceremony took place on March 15, 2026
- The ceremony was held at the Dolby Theatre in Los Angeles, California

Film Production Details:
- The film for which she won the Oscar won the 2025 TIFF (Toronto International Film Festival) People's Choice Award
- The film was directed by Chloé Zhao, who previously won the Academy Award for Best Director for "Nomadland" in 2021
- The film was produced by Amblin Partners, a production company co-founded by Steven Spielberg
- The film was based on a novel that won the Women's Prize for Fiction in 2020

Awards Season Sweep:
- Won the Golden Globe Award for Best Actress (Drama) in January 2026
- Won the BAFTA Award for Best Actress in 2026
- Won the Critics' Choice Award for Best Actress in 2026
- Won the Actor Award for Best Actress in 2026

Historical Significance:
- She is the first Irish woman to win the Academy Award for Best Actress

Provide the actress's full name and include reference URLs that verify each major category of criteria (personal background, Oscar achievement, film production details, awards sweep, and historical significance).
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PersonalBackground(BaseModel):
    birthplace: Optional[str] = None
    birth_month_year: Optional[str] = None  # e.g., "December 1989"
    birth_date_full: Optional[str] = None   # optional full date if provided
    sources: List[str] = Field(default_factory=list)


class OscarAchievement(BaseModel):
    oscar_film_title: Optional[str] = None  # film she won for (as claimed in the answer)
    ceremony_date: Optional[str] = None     # e.g., "March 15, 2026"
    ceremony_venue: Optional[str] = None    # e.g., "Dolby Theatre, Los Angeles, California"
    sources: List[str] = Field(default_factory=list)


class FilmDetails(BaseModel):
    film_title: Optional[str] = None  # the Oscar-winning film title (redundant but helpful)
    lead_role_note: Optional[str] = None
    tiff_award_note: Optional[str] = None  # e.g., "2025 TIFF People's Choice Award"
    director: Optional[str] = None         # should be "Chloé Zhao"
    production_company: Optional[str] = None  # should be "Amblin Partners"
    based_on_novel: Optional[str] = None   # e.g., "yes"/"true"/"based on ..."
    source_novel_title: Optional[str] = None
    director_credential_note: Optional[str] = None  # Zhao Best Director 2021 for Nomadland
    company_cofounder_note: Optional[str] = None     # Spielberg co-founder note
    sources: List[str] = Field(default_factory=list)                # general film refs
    director_cred_sources: List[str] = Field(default_factory=list)  # Zhao 2021 ref(s)
    company_sources: List[str] = Field(default_factory=list)        # Amblin + Spielberg ref(s)
    novel_sources: List[str] = Field(default_factory=list)          # Novel + Women's Prize ref(s)


class AwardsSweep(BaseModel):
    golden_globe_note: Optional[str] = None
    bafta_note: Optional[str] = None
    critics_choice_note: Optional[str] = None
    actor_award_note: Optional[str] = None
    gg_sources: List[str] = Field(default_factory=list)
    bafta_sources: List[str] = Field(default_factory=list)
    critics_choice_sources: List[str] = Field(default_factory=list)
    actor_award_sources: List[str] = Field(default_factory=list)


class HistoricalSignificance(BaseModel):
    significance_note: Optional[str] = None  # "first Irish woman to win Best Actress"
    sources: List[str] = Field(default_factory=list)


class ActressExtraction(BaseModel):
    actress_name: Optional[str] = None
    personal: PersonalBackground = Field(default_factory=PersonalBackground)
    oscars: OscarAchievement = Field(default_factory=OscarAchievement)
    film: FilmDetails = Field(default_factory=FilmDetails)
    awards: AwardsSweep = Field(default_factory=AwardsSweep)
    history: HistoricalSignificance = Field(default_factory=HistoricalSignificance)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_actress_profile() -> str:
    return """
    Extract the structured data about the identified actress and all required verification sources from the answer.

    Required fields:
    - actress_name: The actress's full name, exactly as given in the answer.

    personal:
      - birthplace: The actress's birthplace as stated (e.g., "Killarney, County Kerry, Ireland").
      - birth_month_year: The month and year of birth as stated (e.g., "December 1989").
      - birth_date_full: Full birth date if present (e.g., "28 December 1989"); else null.
      - sources: URL list explicitly provided in the answer that verify birthplace and birth date/month-year.

    oscars:
      - oscar_film_title: Film title she won the Best Actress Oscar for (as stated).
      - ceremony_date: The date of the 98th Academy Awards ceremony as stated (e.g., "March 15, 2026"), or null if not stated.
      - ceremony_venue: The venue of the ceremony as stated (e.g., "Dolby Theatre, Los Angeles, California"), or null if not stated.
      - sources: URL list explicitly provided in the answer that verify she won Best Actress at the 98th Academy Awards and verify the ceremony date and venue.

    film:
      - film_title: The Oscar-winning film title (if repeated separately; otherwise null).
      - lead_role_note: The statement indicating she had the lead role (or equivalent) in that film, if present.
      - tiff_award_note: The statement indicating the film won the 2025 TIFF People's Choice Award, if present.
      - director: Director name of that film (should be "Chloé Zhao" if stated).
      - production_company: Production company credited for that film (should include "Amblin Partners" if stated).
      - based_on_novel: Whether the film is based on a novel (string; "yes"/"true"/"based on", or null if not explicitly stated).
      - source_novel_title: The novel's title, if stated.
      - director_credential_note: Statement that Chloé Zhao previously won the Academy Award for Best Director for "Nomadland" in 2021, if present.
      - company_cofounder_note: Statement that Amblin Partners is co-founded by Steven Spielberg, if present.
      - sources: URL list explicitly provided for the film details (director, production company, TIFF award, adaptation).
      - director_cred_sources: URL list (if any) explicitly provided to support Zhao's 2021 Best Director win.
      - company_sources: URL list (if any) explicitly provided to support Amblin Partners details (including Spielberg co-founder).
      - novel_sources: URL list (if any) explicitly provided to support the source novel and its Women's Prize 2020.

    awards:
      - golden_globe_note: Statement that she won Golden Globe Best Actress (Drama) in January 2026, if present.
      - bafta_note: Statement that she won the BAFTA Best Actress in 2026, if present.
      - critics_choice_note: Statement that she won the Critics' Choice Best Actress in 2026, if present.
      - actor_award_note: Statement that she won the Actor (SAG) Best Actress in 2026, if present.
      - gg_sources: URL list for Golden Globes verification.
      - bafta_sources: URL list for BAFTA verification.
      - critics_choice_sources: URL list for Critics' Choice verification.
      - actor_award_sources: URL list for Actor/SAG verification.

    history:
      - significance_note: Statement that she is the first Irish woman to win the Academy Award for Best Actress, if present.
      - sources: URL list verifying the historical significance.

    Notes:
    - Only extract URLs explicitly present in the answer text. If none are provided for a field, return an empty list.
    - Preserve strings as given in the answer when possible; do not normalize the content beyond basic trimming.
    - If a field is missing in the answer, set it to null (for strings) or [] (for lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _nonempty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and any(isinstance(u, str) and u.strip() for u in urls))


def _merge_urls(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    out.append(uu)
    return out


def _actor_display_name(extracted: ActressExtraction) -> str:
    return extracted.actress_name if _nonempty_str(extracted.actress_name) else "the identified actress"


def _film_display_title(extracted: ActressExtraction) -> str:
    # Prefer explicit film.film_title, else oscars.oscar_film_title, else generic
    if _nonempty_str(extracted.film.film_title):
        return extracted.film.film_title  # type: ignore
    if _nonempty_str(extracted.oscars.oscar_film_title):
        return extracted.oscars.oscar_film_title  # type: ignore
    return "the film for which she won the Oscar"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_actress_name_provided(evaluator: Evaluator, parent_node, extracted: ActressExtraction) -> None:
    evaluator.add_custom_node(
        result=_nonempty_str(extracted.actress_name),
        id="Actress_Name_Provided",
        desc="The solution provides the actress's full name.",
        parent=parent_node,
        critical=True
    )


async def verify_personal_background(evaluator: Evaluator, parent_node, extracted: ActressExtraction) -> None:
    node = evaluator.add_parallel(
        id="Personal_Background",
        desc="Verify the actress's birthplace and birth month/year, with references.",
        parent=parent_node,
        critical=True
    )

    # References existence
    evaluator.add_custom_node(
        result=_nonempty_urls(extracted.personal.sources),
        id="Personal_Background_References",
        desc="Provide URL reference(s) that verify the personal background criteria (birthplace and birth date/month-year).",
        parent=node,
        critical=True
    )

    # Born in Killarney
    born_killarney = evaluator.add_leaf(
        id="Born_in_Killarney",
        desc="The actress was born in Killarney, County Kerry, Ireland.",
        parent=node,
        critical=True
    )
    claim_killarney = f"{_actor_display_name(extracted)} was born in Killarney, County Kerry, Ireland."
    await evaluator.verify(
        claim=claim_killarney,
        node=born_killarney,
        sources=extracted.personal.sources,
        additional_instruction="Verify birthplace explicitly states Killarney in County Kerry, Ireland. Allow minor formatting variations."
    )

    # Born in December 1989
    born_dec_1989 = evaluator.add_leaf(
        id="Born_in_Dec_1989",
        desc="The actress was born in December 1989.",
        parent=node,
        critical=True
    )
    claim_dec = f"{_actor_display_name(extracted)} was born in December 1989."
    await evaluator.verify(
        claim=claim_dec,
        node=born_dec_1989,
        sources=extracted.personal.sources,
        additional_instruction="Verify the month and year (December 1989). Accept if a full date clearly falls in December 1989."
    )


async def verify_oscar_achievement(evaluator: Evaluator, parent_node, extracted: ActressExtraction) -> None:
    node = evaluator.add_parallel(
        id="Oscar_Achievement_2026",
        desc="Verify the actress won Best Actress at the 98th Academy Awards and that the ceremony details match, with references.",
        parent=parent_node,
        critical=True
    )

    # References existence
    evaluator.add_custom_node(
        result=_nonempty_urls(extracted.oscars.sources),
        id="Oscar_Achievement_References",
        desc="Provide URL reference(s) verifying the Oscar win and the ceremony date and venue.",
        parent=node,
        critical=True
    )

    # Won Best Actress at the 98th Academy Awards
    won_best_actress = evaluator.add_leaf(
        id="Won_Best_Actress_98th_Oscars",
        desc="The actress won the Academy Award for Best Actress at the 98th Academy Awards.",
        parent=node,
        critical=True
    )
    film_title = _film_display_title(extracted)
    claim_win = f"{_actor_display_name(extracted)} won the Academy Award for Best Actress at the 98th Academy Awards, for '{film_title}'."
    await evaluator.verify(
        claim=claim_win,
        node=won_best_actress,
        sources=extracted.oscars.sources,
        additional_instruction="Confirm the 98th Academy Awards (2026) Best Actress winner is the identified actress. Allow category naming variants like 'Best Actress in a Leading Role'."
    )

    # Ceremony date
    ceremony_date = evaluator.add_leaf(
        id="Ceremony_Date",
        desc="The 98th Academy Awards ceremony took place on March 15, 2026.",
        parent=node,
        critical=True
    )
    claim_date = "The 98th Academy Awards ceremony took place on March 15, 2026."
    await evaluator.verify(
        claim=claim_date,
        node=ceremony_date,
        sources=extracted.oscars.sources,
        additional_instruction="Check that the date is March 15, 2026. The page may include schedule or summary with the exact date."
    )

    # Ceremony venue
    ceremony_venue = evaluator.add_leaf(
        id="Ceremony_Venue",
        desc="The ceremony was held at the Dolby Theatre in Los Angeles, California.",
        parent=node,
        critical=True
    )
    claim_venue = "The 98th Academy Awards ceremony was held at the Dolby Theatre in Los Angeles, California."
    await evaluator.verify(
        claim=claim_venue,
        node=ceremony_venue,
        sources=extracted.oscars.sources,
        additional_instruction="Confirm the ceremony venue is the Dolby Theatre in Los Angeles, California. Minor formatting differences are acceptable."
    )


async def verify_film_production_details(evaluator: Evaluator, parent_node, extracted: ActressExtraction) -> None:
    node = evaluator.add_parallel(
        id="Film_Production_Details",
        desc="Verify the production/source/festival-award facts about the film for which she won the Best Actress Oscar (i.e., the Oscar-winning film must be the one satisfying all these film constraints), with references.",
        parent=parent_node,
        critical=True
    )

    # Gather sources
    film_sources = extracted.film.sources
    dir_cred_sources = extracted.film.director_cred_sources
    company_sources = extracted.film.company_sources
    novel_sources = extracted.film.novel_sources
    all_film_refs = _merge_urls(film_sources, dir_cred_sources, company_sources, novel_sources)

    # References existence
    evaluator.add_custom_node(
        result=_nonempty_urls(all_film_refs),
        id="Film_Production_References",
        desc="Provide URL reference(s) verifying the film production details (lead role, TIFF award, director, director credential, production company, company co-founder fact, adaptation source, and source novel award).",
        parent=node,
        critical=True
    )

    film_title = _film_display_title(extracted)
    actor_name = _actor_display_name(extracted)

    # Lead role in Oscar-winning film
    lead_role = evaluator.add_leaf(
        id="Lead_Role_In_Oscar_Winning_Film",
        desc="The film for which she won the Oscar starred her in the lead role.",
        parent=node,
        critical=True
    )
    claim_lead = f"{actor_name} starred in the lead role in '{film_title}'."
    await evaluator.verify(
        claim=claim_lead,
        node=lead_role,
        sources=film_sources,
        additional_instruction="Confirm she is the lead/leading role or star of the film. Accept common phrasings like 'starring' or 'lead performance'."
    )

    # TIFF 2025 People's Choice Award
    tiff_award = evaluator.add_leaf(
        id="TIFF_Award_2025",
        desc="That Oscar-winning film won the 2025 TIFF (Toronto International Film Festival) People's Choice Award.",
        parent=node,
        critical=True
    )
    claim_tiff = f"'{film_title}' won the People's Choice Award at the 2025 Toronto International Film Festival."
    await evaluator.verify(
        claim=claim_tiff,
        node=tiff_award,
        sources=film_sources,
        additional_instruction="Confirm TIFF 2025 People's Choice Award winner. Allow naming variants like 'TIFF People's Choice Award'."
    )

    # Directed by Chloé Zhao
    directed_by = evaluator.add_leaf(
        id="Film_Directed_by_Chloe_Zhao",
        desc="That Oscar-winning film was directed by Chloé Zhao.",
        parent=node,
        critical=True
    )
    claim_dir = f"'{film_title}' was directed by Chloé Zhao."
    await evaluator.verify(
        claim=claim_dir,
        node=directed_by,
        sources=film_sources,
        additional_instruction="Verify the director is Chloé Zhao. Allow diacritic-insensitive match for 'Chloé' vs 'Chloe'."
    )

    # Zhao Best Director for Nomadland in 2021
    zhao_bd_2021 = evaluator.add_leaf(
        id="Chloe_Zhao_Best_Director_Oscar_2021_Nomadland",
        desc="Chloé Zhao previously won the Academy Award for Best Director for 'Nomadland' in 2021.",
        parent=node,
        critical=True
    )
    claim_zhao_bd = "Chloé Zhao won the Academy Award for Best Director for 'Nomadland' in 2021."
    await evaluator.verify(
        claim=claim_zhao_bd,
        node=zhao_bd_2021,
        sources=_merge_urls(dir_cred_sources, film_sources),
        additional_instruction="Confirm Zhao's 2021 Best Director Oscar for Nomadland. Allow minor title formatting differences."
    )

    # Produced by Amblin Partners
    produced_by_amblin = evaluator.add_leaf(
        id="Film_Produced_by_Amblin_Partners",
        desc="That Oscar-winning film was produced by Amblin Partners.",
        parent=node,
        critical=True
    )
    claim_prod = f"'{film_title}' was produced by Amblin Partners."
    await evaluator.verify(
        claim=claim_prod,
        node=produced_by_amblin,
        sources=_merge_urls(film_sources, company_sources),
        additional_instruction="Confirm Amblin Partners is among the producing companies for the film."
    )

    # Amblin co-founded by Steven Spielberg
    amblin_cofounder = evaluator.add_leaf(
        id="Amblin_Partners_CoFounded_by_Steven_Spielberg",
        desc="Amblin Partners is a production company co-founded by Steven Spielberg.",
        parent=node,
        critical=True
    )
    claim_cofounder = "Amblin Partners was co-founded by Steven Spielberg."
    await evaluator.verify(
        claim=claim_cofounder,
        node=amblin_cofounder,
        sources=_merge_urls(company_sources, film_sources),
        additional_instruction="Verify company founding info that lists Steven Spielberg as a co-founder."
    )

    # Film based on a novel
    based_on_novel = evaluator.add_leaf(
        id="Film_Based_on_a_Novel",
        desc="That Oscar-winning film was based on a novel (i.e., adapted from a novel).",
        parent=node,
        critical=True
    )
    claim_adapt = f"'{film_title}' is based on a novel."
    await evaluator.verify(
        claim=claim_adapt,
        node=based_on_novel,
        sources=_merge_urls(film_sources, novel_sources),
        additional_instruction="Confirm the film is an adaptation of a novel. Pages like production notes, Wikipedia, or official sites are acceptable."
    )

    # Source novel won Women's Prize for Fiction in 2020
    womens_prize = evaluator.add_leaf(
        id="Source_Novel_Won_Womens_Prize_2020",
        desc="The novel on which the Oscar-winning film was based won the Women's Prize for Fiction in 2020.",
        parent=node,
        critical=True
    )
    novel_title = extracted.film.source_novel_title if _nonempty_str(extracted.film.source_novel_title) else "the source novel"
    claim_wp = f"{novel_title} won the Women's Prize for Fiction in 2020."
    await evaluator.verify(
        claim=claim_wp,
        node=womens_prize,
        sources=_merge_urls(novel_sources, film_sources),
        additional_instruction="Verify the novel (the source material for the film) won the Women's Prize for Fiction in 2020. Allow alternate naming 'Women's Prize'."
    )


async def verify_awards_sweep(evaluator: Evaluator, parent_node, extracted: ActressExtraction) -> None:
    node = evaluator.add_parallel(
        id="Awards_Season_Sweep",
        desc="Verify the actress won the specified non-Oscar major Best Actress awards in the stated timeframe, with references (together with the Oscar win, this constitutes the sweep described in the constraints).",
        parent=parent_node,
        critical=True
    )

    # References existence (require at least one URL for each of the four awards)
    evaluator.add_custom_node(
        result=_nonempty_urls(extracted.awards.gg_sources)
               and _nonempty_urls(extracted.awards.bafta_sources)
               and _nonempty_urls(extracted.awards.critics_choice_sources)
               and _nonempty_urls(extracted.awards.actor_award_sources),
        id="Awards_Sweep_References",
        desc="Provide URL reference(s) verifying the Golden Globe, BAFTA, Critics' Choice, and Actor Award wins.",
        parent=node,
        critical=True
    )

    actor_name = _actor_display_name(extracted)

    # Golden Globe
    gg = evaluator.add_leaf(
        id="Golden_Globe_Win",
        desc="The actress won the Golden Globe Award for Best Actress (Drama) in January 2026.",
        parent=node,
        critical=True
    )
    claim_gg = f"{actor_name} won the Golden Globe Award for Best Actress in a Motion Picture – Drama in January 2026."
    await evaluator.verify(
        claim=claim_gg,
        node=gg,
        sources=extracted.awards.gg_sources,
        additional_instruction="Verify a Golden Globes 2026 win in the Drama lead actress category; confirm the month is January 2026. Allow category phrasing variants."
    )

    # BAFTA
    bafta = evaluator.add_leaf(
        id="BAFTA_Win",
        desc="The actress won the BAFTA Award for Best Actress in 2026.",
        parent=node,
        critical=True
    )
    claim_bafta = f"{actor_name} won the BAFTA Award for Best Actress (Leading Actress) in 2026."
    await evaluator.verify(
        claim=claim_bafta,
        node=bafta,
        sources=extracted.awards.bafta_sources,
        additional_instruction="Verify a 2026 BAFTA win in the Leading Actress category. Accept phrasing like 'Best Leading Actress'."
    )

    # Critics' Choice
    cca = evaluator.add_leaf(
        id="Critics_Choice_Win",
        desc="The actress won the Critics' Choice Award for Best Actress in 2026.",
        parent=node,
        critical=True
    )
    claim_cca = f"{actor_name} won the Critics' Choice Award for Best Actress in 2026."
    await evaluator.verify(
        claim=claim_cca,
        node=cca,
        sources=extracted.awards.critics_choice_sources,
        additional_instruction="Verify the Critics' Choice (CCA) 2026 Best Actress win. Allow naming variants like 'Critics Choice'."
    )

    # Actor Award (SAG)
    sag = evaluator.add_leaf(
        id="Actor_Award_Win",
        desc="The actress won the Actor Award for Best Actress in 2026.",
        parent=node,
        critical=True
    )
    claim_sag = f"{actor_name} won the Actor (SAG) award for a leading actress performance in 2026."
    await evaluator.verify(
        claim=claim_sag,
        node=sag,
        sources=extracted.awards.actor_award_sources,
        additional_instruction="Treat 'Actor Award' as the Screen Actors Guild (SAG) acting awards; accept category phrasing such as 'Outstanding Performance by a Female Actor in a Leading Role' or equivalent rebranded naming."
    )


async def verify_historical_significance(evaluator: Evaluator, parent_node, extracted: ActressExtraction) -> None:
    node = evaluator.add_parallel(
        id="Historical_Significance",
        desc="Verify the historical significance claim, with references.",
        parent=parent_node,
        critical=True
    )

    # References existence
    evaluator.add_custom_node(
        result=_nonempty_urls(extracted.history.sources),
        id="Historical_Significance_References",
        desc="Provide URL reference(s) confirming she is the first Irish woman to win Best Actress Oscar.",
        parent=node,
        critical=True
    )

    # First Irish woman to win Best Actress
    first_irish = evaluator.add_leaf(
        id="First_Irish_Woman_Best_Actress",
        desc="The actress is the first Irish woman to win the Academy Award for Best Actress.",
        parent=node,
        critical=True
    )
    claim_first = f"{_actor_display_name(extracted)} is the first Irish woman to win the Academy Award for Best Actress."
    await evaluator.verify(
        claim=claim_first,
        node=first_irish,
        sources=extracted.history.sources,
        additional_instruction="Confirm 'first Irish woman' to win Best Actress (lead actress) at the Academy Awards. Allow phrasing variants; ensure nationality criterion is Irish (citizenship or widely recognized Irish identity)."
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
) -> Dict:
    """
    Evaluate an answer for the 2026 Best Actress identification task against the rubric.
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_actress_profile(),
        template_class=ActressExtraction,
        extraction_name="actress_profile"
    )

    # Build a critical task root (since Evaluator root is non-critical by design)
    task_root = evaluator.add_parallel(
        id="Root",
        desc="Identify an actress who meets all specified criteria and provide reference URLs verifying each major category (personal background, Oscar achievement, film production details, awards sweep, historical significance).",
        parent=root,
        critical=True
    )

    # Add sub-verifications
    await verify_actress_name_provided(evaluator, task_root, extracted)
    await verify_personal_background(evaluator, task_root, extracted)
    await verify_oscar_achievement(evaluator, task_root, extracted)
    await verify_film_production_details(evaluator, task_root, extracted)
    await verify_awards_sweep(evaluator, task_root, extracted)
    await verify_historical_significance(evaluator, task_root, extracted)

    # Return final structured summary
    return evaluator.get_summary()