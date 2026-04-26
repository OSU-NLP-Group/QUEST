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
TASK_ID = "emmy_2025_premiere_winners"
TASK_DESCRIPTION = (
    "Identify television series that premiered in 2025 and won one of the three major Emmy awards "
    "(Outstanding Drama Series, Outstanding Comedy Series, or Outstanding Limited or Anthology Series) "
    "at the 77th Primetime Emmy Awards ceremony. For each identified series, provide the following information: "
    "(1) the specific major Emmy award category the series won, (2) the exact premiere date (month, day, and year), "
    "(3) the streaming platform or network where it premiered, (4) the creator(s) of the series, "
    "(5) if applicable, the name of any lead actor from the series who won an Emmy acting award (Outstanding Lead Actor "
    "or Outstanding Lead Actress in a Drama Series, Comedy Series, or Limited or Anthology Series) at the 77th Emmy Awards "
    "for their role in that specific series, and (6) the total number of episodes in Season 1."
)

ALLOWED_MAJOR_EMMY_CATEGORIES = [
    "Outstanding Drama Series",
    "Outstanding Comedy Series",
    "Outstanding Limited or Anthology Series",
]

ALLOWED_LEAD_ACTING_CATEGORIES = [
    "Outstanding Lead Actor in a Drama Series",
    "Outstanding Lead Actress in a Drama Series",
    "Outstanding Lead Actor in a Comedy Series",
    "Outstanding Lead Actress in a Comedy Series",
    "Outstanding Lead Actor in a Limited or Anthology Series",
    "Outstanding Lead Actress in a Limited or Anthology Series",
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SeriesItem(BaseModel):
    name: Optional[str] = None
    major_emmy_category: Optional[str] = None
    award_urls: List[str] = Field(default_factory=list)

    premiere_date: Optional[str] = None
    premiere_platform: Optional[str] = None
    premiere_urls: List[str] = Field(default_factory=list)

    creators: List[str] = Field(default_factory=list)
    creator_urls: List[str] = Field(default_factory=list)

    lead_actor_name: Optional[str] = None
    lead_actor_award_category: Optional[str] = None
    lead_actor_urls: List[str] = Field(default_factory=list)

    season1_episode_count: Optional[str] = None
    episode_count_urls: List[str] = Field(default_factory=list)


class SeriesExtraction(BaseModel):
    series: List[SeriesItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series() -> str:
    return """
    Extract up to three television series from the answer that the author claims:
    • Premiered in 2025, and
    • Won one of the three major Emmy awards at the 77th Primetime Emmy Awards:
      - Outstanding Drama Series
      - Outstanding Comedy Series
      - Outstanding Limited or Anthology Series

    For each series, extract the following fields exactly as they appear in the answer:
    - name: The series title.
    - major_emmy_category: The exact major Emmy category claimed for the win (must be one of the three above).
    - award_urls: A list of URL(s) explicitly cited that confirm the Emmy win (official Emmy site, major outlets, etc.).
    - premiere_date: The exact premiere date (month, day, year) as stated.
    - premiere_platform: The streaming platform or network where it premiered.
    - premiere_urls: A list of URL(s) explicitly cited that confirm the premiere date and platform.
    - creators: A list of the creator(s) of the series.
    - creator_urls: A list of URL(s) explicitly cited that confirm the creator(s).
    - lead_actor_name: If applicable, the name of a lead actor/actress from the series who won an Emmy acting award
      (Outstanding Lead Actor or Outstanding Lead Actress in Drama/Comedy/Limited or Anthology Series) at the 77th
      Emmys specifically for this series. If not mentioned or not applicable, set to null.
    - lead_actor_award_category: If a lead actor is named, the exact lead acting category they won (e.g., "Outstanding Lead Actress in a Drama Series").
      If not applicable, set to null.
    - lead_actor_urls: URL(s) that specifically confirm the actor's Emmy win for this series; empty list if not applicable.
    - season1_episode_count: The total number of episodes in Season 1, as stated (string). If not provided, set to null.
    - episode_count_urls: A list of URL(s) that confirm the Season 1 episode count; empty list if not provided.

    IMPORTANT:
    - Return a JSON object with a top-level key "series" that is an array of at most 3 SeriesItem objects.
    - If the answer mentions more than 3 qualifying series, include the first 3 in the order they appear.
    - If the answer mentions fewer than 3, include whatever is present.
    - For any field not provided in the answer, set it to null (or empty list for URL arrays).
    - For URL fields, extract only explicit URLs from the answer (plain URLs or markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper utils                                                                #
# --------------------------------------------------------------------------- #
def safe_series_name(name: Optional[str]) -> str:
    return name if (name and name.strip()) else "this series"

def category_is_allowed(category: Optional[str]) -> bool:
    if not category:
        return False
    c = category.strip()
    return any(c.lower() == allowed.lower() for allowed in ALLOWED_MAJOR_EMMY_CATEGORIES)

def acting_category_is_allowed(category: Optional[str]) -> bool:
    if not category:
        return False
    c = category.strip()
    return any(c.lower() == allowed.lower() for allowed in ALLOWED_LEAD_ACTING_CATEGORIES)


# --------------------------------------------------------------------------- #
# Per-series verification construction                                        #
# --------------------------------------------------------------------------- #
async def build_series_verification(
    evaluator: Evaluator,
    parent,
    idx_zero_based: int,
    item: SeriesItem
) -> None:
    """
    Build the verification subtree for one series (series_1/series_2/series_3).
    """
    series_idx = idx_zero_based + 1
    series_node = evaluator.add_parallel(
        id=f"series_{series_idx}",
        desc=f"Evaluation of the {'first' if series_idx==1 else 'second' if series_idx==2 else 'third'} identified series meeting all criteria",
        parent=parent,
        critical=False  # series-level non-critical to allow partial credit across series
    )

    # --------------------- PREMIERE INFO GROUP (Create first for gating) --------------------- #
    premiere_group = evaluator.add_parallel(
        id=f"series_{series_idx}_premiere_info",
        desc="Premiere date and platform information is accurate",
        parent=series_node,
        critical=True
    )

    # Existence of premiere references (critical)
    premiere_ref_exists_node = evaluator.add_custom_node(
        result=bool(item.premiere_urls),
        id=f"series_{series_idx}_premiere_reference",
        desc="A reference URL is provided that confirms the premiere date and platform",
        parent=premiere_group,
        critical=True
    )

    # Verify premiere date accuracy (critical)
    premiere_date_leaf = evaluator.add_leaf(
        id=f"series_{series_idx}_premiere_date",
        desc="The exact premiere date (month, day, and year) is provided and accurate",
        parent=premiere_group,
        critical=True
    )
    premiere_date_claim = (
        f"The series {safe_series_name(item.name)} premiered on {item.premiere_date}."
        if item.premiere_date else
        "The series premiered on the stated date."
    )
    await evaluator.verify(
        claim=premiere_date_claim,
        node=premiere_date_leaf,
        sources=item.premiere_urls if item.premiere_urls else None,
        additional_instruction="Verify the premiere date exactly or with minor formatting variations (e.g., abbreviations). Consider the first public release on the stated platform or network.",
        extra_prerequisites=[premiere_ref_exists_node]
    )

    # Verify premiere platform accuracy (critical)
    premiere_platform_leaf = evaluator.add_leaf(
        id=f"series_{series_idx}_premiere_platform",
        desc="The streaming platform or network is correctly identified",
        parent=premiere_group,
        critical=True
    )
    premiere_platform_claim = (
        f"The series {safe_series_name(item.name)} premiered on the platform or network '{item.premiere_platform}'."
        if item.premiere_platform else
        "The series premiered on the stated platform or network."
    )
    await evaluator.verify(
        claim=premiere_platform_claim,
        node=premiere_platform_leaf,
        sources=item.premiere_urls if item.premiere_urls else None,
        additional_instruction="Verify the platform or network where the series first premiered, as stated in the answer.",
        extra_prerequisites=[premiere_ref_exists_node]
    )

    # --------------------- CREATOR INFO GROUP --------------------- #
    creator_group = evaluator.add_parallel(
        id=f"series_{series_idx}_creator_info",
        desc="Creator information is accurate",
        parent=series_node,
        critical=True
    )

    # Existence of creator references (critical)
    creator_ref_exists_node = evaluator.add_custom_node(
        result=bool(item.creator_urls),
        id=f"series_{series_idx}_creator_reference",
        desc="A reference URL is provided that confirms the creator(s)",
        parent=creator_group,
        critical=True
    )

    # Verify creators correctness (critical)
    creator_identified_leaf = evaluator.add_leaf(
        id=f"series_{series_idx}_creator_identified",
        desc="The creator(s) of the series is correctly identified",
        parent=creator_group,
        critical=True
    )
    creators_str = ", ".join(item.creators) if item.creators else ""
    creator_claim = (
        f"The creator(s) of {safe_series_name(item.name)} is/are {creators_str}."
        if creators_str else
        "The creator(s) of the series are correctly identified in the answer."
    )
    await evaluator.verify(
        claim=creator_claim,
        node=creator_identified_leaf,
        sources=item.creator_urls if item.creator_urls else None,
        additional_instruction="Accept co-creators and reasonable variants (e.g., 'created by' vs 'developed by' when used interchangeably by official sources).",
        extra_prerequisites=[creator_ref_exists_node]
    )

    # --------------------- EPISODE COUNT GROUP --------------------- #
    episode_group = evaluator.add_parallel(
        id=f"series_{series_idx}_episode_count",
        desc="Season 1 episode count information is accurate",
        parent=series_node,
        critical=True
    )

    # Existence of episode count references (critical)
    episode_ref_exists_node = evaluator.add_custom_node(
        result=bool(item.episode_count_urls),
        id=f"series_{series_idx}_count_reference",
        desc="A reference URL is provided that confirms the episode count",
        parent=episode_group,
        critical=True
    )

    # Verify episode count correctness (critical)
    count_leaf = evaluator.add_leaf(
        id=f"series_{series_idx}_count_correct",
        desc="The total number of episodes in Season 1 is provided and accurate",
        parent=episode_group,
        critical=True
    )
    count_claim = (
        f"Season 1 of {safe_series_name(item.name)} has {item.season1_episode_count} episodes."
        if item.season1_episode_count else
        "Season 1 has the number of episodes as stated in the answer."
    )
    await evaluator.verify(
        claim=count_claim,
        node=count_leaf,
        sources=item.episode_count_urls if item.episode_count_urls else None,
        additional_instruction="Verify the Season 1 episode count (allowing for minor variances like special episodes if the source clearly explains).",
        extra_prerequisites=[episode_ref_exists_node]
    )

    # --------------------- LEAD ACTOR EMMY (OPTIONAL) --------------------- #
    actor_group = evaluator.add_parallel(
        id=f"series_{series_idx}_lead_actor_emmy",
        desc="If a lead actor from the series won an Emmy acting award at the 77th Emmy Awards for this series, the information is provided and accurate",
        parent=series_node,
        critical=False
    )

    if item.lead_actor_name:
        # Existence of actor win references (critical within optional group)
        actor_ref_exists_node = evaluator.add_custom_node(
            result=bool(item.lead_actor_urls),
            id=f"series_{series_idx}_actor_reference",
            desc="If an actor is named, a reference URL is provided that confirms the actor's Emmy win for this series",
            parent=actor_group,
            critical=True
        )

        # Verify actor & win (non-critical leaf under optional group)
        actor_leaf = evaluator.add_leaf(
            id=f"series_{series_idx}_actor_name",
            desc="If applicable, the name of the lead actor who won an Emmy acting award (Outstanding Lead Actor/Actress in Drama, Comedy, or Limited Series) at the 77th Emmy Awards for their role in this series is provided and accurate",
            parent=actor_group,
            critical=False
        )
        actor_cat = item.lead_actor_award_category if item.lead_actor_award_category else "a lead acting category"
        actor_claim = (
            f"{item.lead_actor_name} won {actor_cat} at the 77th Primetime Emmy Awards for their role in {safe_series_name(item.name)}."
        )
        add_ins = "Verify the actor's Emmy win is specifically for this series at the 77th Primetime Emmy Awards. Allow minor name formatting differences."
        if item.lead_actor_award_category and not acting_category_is_allowed(item.lead_actor_award_category):
            add_ins += " Note: The stated acting category may be mis-specified; verify the actual lead acting category per the source."

        await evaluator.verify(
            claim=actor_claim,
            node=actor_leaf,
            sources=item.lead_actor_urls if item.lead_actor_urls else None,
            additional_instruction=add_ins,
            extra_prerequisites=[actor_ref_exists_node]
        )
    else:
        # Not applicable: explicitly pass a custom note so the optional group doesn't penalize
        evaluator.add_custom_node(
            result=True,
            id=f"series_{series_idx}_actor_not_applicable",
            desc="No lead actor Emmy win claimed for this series (not applicable)",
            parent=actor_group,
            critical=False
        )

    # --------------------- EMMY QUALIFICATION GROUP --------------------- #
    emmy_group = evaluator.add_parallel(
        id=f"series_{series_idx}_emmy_qualification",
        desc="The series won one of the three major Emmy awards at the 77th Emmy Awards and premiered in 2025",
        parent=series_node,
        critical=True
    )

    # Verify the stated award category is one of the allowed (critical)
    award_category_leaf = evaluator.add_leaf(
        id=f"series_{series_idx}_award_category_stated",
        desc="The specific major Emmy award category won is correctly identified",
        parent=emmy_group,
        critical=True
    )
    stated_cat = item.major_emmy_category if item.major_emmy_category else ""
    award_cat_claim = (
        f"The stated award category '{stated_cat}' is one of the allowed major categories: "
        f"{', '.join(ALLOWED_MAJOR_EMMY_CATEGORIES)}."
        if stated_cat else
        "The stated award category belongs to the allowed major categories."
    )
    await evaluator.verify(
        claim=award_cat_claim,
        node=award_category_leaf,
        additional_instruction="Allow minor formatting differences but require semantic equivalence to one of the three specified categories."
    )

    # Existence of award references (critical)
    emmy_ref_exists_node = evaluator.add_custom_node(
        result=bool(item.award_urls),
        id=f"series_{series_idx}_emmy_reference",
        desc="A reference URL is provided that confirms the Emmy award win",
        parent=emmy_group,
        critical=True
    )

    # Verify the Emmy win: won one of the three major categories at 77th (critical)
    award_won_leaf = evaluator.add_leaf(
        id=f"series_{series_idx}_award_won",
        desc="The series won Outstanding Drama Series, Outstanding Comedy Series, or Outstanding Limited or Anthology Series at the 77th Primetime Emmy Awards",
        parent=emmy_group,
        critical=True
    )
    series_name_for_claim = safe_series_name(item.name)
    if item.major_emmy_category and category_is_allowed(item.major_emmy_category):
        award_won_claim = (
            f"{series_name_for_claim} won the {item.major_emmy_category} at the 77th Primetime Emmy Awards."
        )
    else:
        award_won_claim = (
            f"{series_name_for_claim} won one of the three major Emmy awards (Drama, Comedy, or Limited/Anthology Series) at the 77th Primetime Emmy Awards."
        )
    await evaluator.verify(
        claim=award_won_claim,
        node=award_won_leaf,
        sources=item.award_urls if item.award_urls else None,
        additional_instruction="Confirm this is a WIN (not a nomination) and that it is at the 77th Primetime Emmy Awards.",
        extra_prerequisites=[emmy_ref_exists_node]
    )

    # Verify the premiere year is 2025 (critical)
    premiered_2025_leaf = evaluator.add_leaf(
        id=f"series_{series_idx}_premiered_2025",
        desc="The series premiered in 2025",
        parent=emmy_group,
        critical=True
    )
    premiered_2025_claim = "The series premiered in 2025."
    await evaluator.verify(
        claim=premiered_2025_claim,
        node=premiered_2025_leaf,
        sources=item.premiere_urls if item.premiere_urls else None,
        additional_instruction="Use the premiere date evidence; accept regional/platform-specific first release dates that fall in 2025.",
        extra_prerequisites=[premiere_ref_exists_node]
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
    Evaluate an answer for the 2025 premiere + 77th Emmy winners task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Evaluate series independently
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

    # Extract structured series information
    extraction = await evaluator.extract(
        prompt=prompt_extract_series(),
        template_class=SeriesExtraction,
        extraction_name="series_extraction"
    )

    # Record allowed categories as custom info for transparency
    evaluator.add_custom_info(
        info={
            "allowed_major_emmy_categories": ALLOWED_MAJOR_EMMY_CATEGORIES,
            "allowed_lead_acting_categories": ALLOWED_LEAD_ACTING_CATEGORIES
        },
        info_type="policy",
        info_name="allowed_categories"
    )

    # Normalize to first 3 series, pad with empty if fewer
    series_list: List[SeriesItem] = list(extraction.series[:3])
    while len(series_list) < 3:
        series_list.append(SeriesItem())

    # Build verification subtrees for up to 3 series
    for i, item in enumerate(series_list):
        await build_series_verification(evaluator, root, i, item)

    # Return the evaluation summary
    return evaluator.get_summary()