import asyncio
import logging
import re
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


TASK_ID = "goty2024_studios"
TASK_DESCRIPTION = (
    "Identify at least two game development studios whose games won 'Game of the Year' awards at major gaming "
    "industry award ceremonies held in 2024 (specifically The Game Awards or TIGA Games Industry Awards). For each studio, "
    "provide: (1) The studio's complete name, (2) The studio's headquarters location (city and country), (3) The name of "
    "the winning game, (4) The specific award ceremony where it won Game of the Year, (5) The exact date when that award "
    "ceremony was held, (6) The game's publisher, (7) The studio's founding year, and (8) The approximate number of "
    "employees at the studio."
)


# ----------------------------- Data Models --------------------------------- #
class StudioItem(BaseModel):
    studio_name: Optional[str] = None
    hq_city: Optional[str] = None
    hq_country: Optional[str] = None
    winning_game: Optional[str] = None
    award_ceremony: Optional[str] = None
    award_date: Optional[str] = None
    publisher: Optional[str] = None
    founding_year: Optional[str] = None
    employee_count: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class StudiosExtraction(BaseModel):
    studios: List[StudioItem] = Field(default_factory=list)


# --------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_studios() -> str:
    return """
Extract up to five distinct studio items mentioned in the answer. For each studio item, return the following fields:

- studio_name: the complete name of the game development studio as written in the answer.
- hq_city: the city of the studio's headquarters (if specified in the answer; else null).
- hq_country: the country of the studio's headquarters (if specified in the answer; else null).
- winning_game: the game title that is claimed to have won "Game of the Year" in 2024 (if provided; else null).
- award_ceremony: the specific award ceremony name as written in the answer (e.g., "The Game Awards 2024", "TGA 2024", "TIGA Games Industry Awards 2024", etc.; if provided; else null).
- award_date: the exact calendar date of that ceremony as written in the answer (e.g., "12 December 2024", "Nov 28, 2024"; if provided; else null).
- publisher: the game's publisher as written in the answer (if provided; else null).
- founding_year: the studio's founding year as written in the answer (if provided; else null).
- employee_count: the approximate number of employees at the studio as written in the answer (if provided; else null).
- source_urls: an array of ALL URLs explicitly cited in the answer for this studio item (e.g., official site, Wikipedia, awards pages, news, press releases). Include only URLs that appear in the answer; do not invent.

Rules:
- Do not infer or add new information; copy values exactly as presented in the answer.
- If a field is missing, set it to null (for strings) or [] for source_urls.
- Return a JSON object with a top-level key "studios" containing an array of these studio item objects, in the order they appear in the answer.
"""


# ---------------------------- Helper Functions ----------------------------- #
def _normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s or None


def _no_source_instruction(sources: List[str]) -> str:
    if not sources:
        return "No URLs were provided by the answer for this studio item. Treat the claim as NOT SUPPORTED due to missing evidence."
    return ""


# --------------------------- Verification Logic ---------------------------- #
async def verify_studio_item(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    item: StudioItem,
    studio_index: int,
) -> Dict[str, Any]:
    """
    Build verification subtree for one studio item and run verifications.
    Returns a dict with keys: 'studio_name', 'qual_node' (VerificationNode).
    """
    studio_node = evaluator.add_parallel(
        id=f"Studio_{studio_index+1}",
        desc=f"Evaluation of studio item #{studio_index+1} (if provided).",
        parent=parent_node,
        critical=False
    )

    qual_node = evaluator.add_parallel(
        id=f"Studio_{studio_index+1}_Qualifying_Package",
        desc=f"Studio #{studio_index+1} meets award criteria and includes all required attributes (accurate as of end of 2024).",
        parent=studio_node,
        critical=True
    )

    sources = item.source_urls or []

    # 1) Studio_Complete_Name (existence check)
    evaluator.add_custom_node(
        result=bool(item.studio_name and item.studio_name.strip()),
        id=f"Studio_{studio_index+1}_Studio_Complete_Name",
        desc="Provides the studio's complete name.",
        parent=qual_node,
        critical=True
    )

    # 2) Studio_Is_Game_Dev_Studio (verify via sources)
    leaf = evaluator.add_leaf(
        id=f"Studio_{studio_index+1}_Studio_Is_Game_Dev_Studio",
        desc="The named entity is a game development studio (not merely a publisher/label/award organizer).",
        parent=qual_node,
        critical=True
    )
    claim = f"{item.studio_name} is a game development studio (developer), not just a publisher/label or an award organizer."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            _no_source_instruction(sources)
            + " Verify that the entity is primarily a game developer/studio. Accept if sources explicitly refer to it as a developer or development studio."
        )
    )

    # 3) Headquarters_City_And_Country (verify via sources, require both)
    leaf = evaluator.add_leaf(
        id=f"Studio_{studio_index+1}_Headquarters_City_And_Country",
        desc="Provides the studio headquarters location including both city and country.",
        parent=qual_node,
        critical=True
    )
    claim = f"The headquarters of {item.studio_name} is located in {item.hq_city}, {item.hq_country}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            _no_source_instruction(sources)
            + " This claim should include BOTH a city and a country. If either part is missing or sources do not support both, mark as NOT SUPPORTED. "
              "Allow equivalent city/country naming variants if clearly the same place."
        )
    )

    # 4) Winning_Game_Name (existence check)
    evaluator.add_custom_node(
        result=bool(item.winning_game and item.winning_game.strip()),
        id=f"Studio_{studio_index+1}_Winning_Game_Name",
        desc="Provides the name of the game that won Game of the Year.",
        parent=qual_node,
        critical=True
    )

    # 5) Award_Ceremony_Name_Allowed (logic check, allow variants)
    leaf = evaluator.add_leaf(
        id=f"Studio_{studio_index+1}_Award_Ceremony_Name_Allowed",
        desc="Identifies the award ceremony and it is either The Game Awards 2024 or the TIGA Games Industry Awards 2024.",
        parent=qual_node,
        critical=True
    )
    # We phrase as a logic/normalization check with tolerance to common variants/abbreviations.
    ceremony_str = item.award_ceremony or ""
    claim = (
        f"The specified award ceremony string '{ceremony_str}' refers to either "
        f"'The Game Awards 2024' (aka 'TGA 2024') or 'TIGA Games Industry Awards 2024' (aka 'TIGA Awards 2024')."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction="Be tolerant to common naming variants/abbreviations (e.g., 'TGA 2024', 'The Game Awards (2024)', 'TIGA Awards 2024'). "
                              "If the string clearly maps to one of these two 2024 ceremonies, mark as Correct; otherwise Incorrect."
    )

    # 6) GOTY_Winner_Verification (verify via sources)
    leaf = evaluator.add_leaf(
        id=f"Studio_{studio_index+1}_GOTY_Winner_Verification",
        desc="The provided game is a Game of the Year winner at the specified ceremony (not merely nominated/shortlisted).",
        parent=qual_node,
        critical=True
    )
    claim = (
        f"The game '{item.winning_game}' developed by {item.studio_name} won the 'Game of the Year' award at {item.award_ceremony}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            _no_source_instruction(sources)
            + " Verify that the game actually WON the top 'Game of the Year' award (not just nominated/shortlisted). "
              "Accept official award pages, reputable news, or press releases as support."
        )
    )

    # 7) Award_Ceremony_Exact_Date (verify via sources)
    leaf = evaluator.add_leaf(
        id=f"Studio_{studio_index+1}_Award_Ceremony_Exact_Date",
        desc="Provides the exact calendar date (day-month-year) when the specified ceremony was held in 2024, and the date is correct.",
        parent=qual_node,
        critical=True
    )
    claim = f"The {item.award_ceremony} ceremony took place on {item.award_date}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            _no_source_instruction(sources)
            + " Cross-check the exact event date from authoritative sources. "
              "If the ceremony is the TIGA Games Industry Awards 2024, the correct date should be November 28, 2024 (allow formatting variants like '28 November 2024' or 'Nov 28, 2024'). "
              "If the ceremony is The Game Awards 2024, verify the exact December 2024 date shown by credible sources and confirm it matches the provided date exactly."
        )
    )

    # 8) Game_Publisher (verify via sources)
    leaf = evaluator.add_leaf(
        id=f"Studio_{studio_index+1}_Game_Publisher",
        desc="Provides the game's publisher.",
        parent=qual_node,
        critical=True
    )
    claim = f"The publisher of '{item.winning_game}' is '{item.publisher}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            _no_source_instruction(sources)
            + " Verify the game's publisher. If multiple co-publishers exist, consider the claim Correct if the provided publisher is among the listed publishers on credible sources."
        )
    )

    # 9) Studio_Founding_Year (verify via sources)
    leaf = evaluator.add_leaf(
        id=f"Studio_{studio_index+1}_Studio_Founding_Year",
        desc="Provides the studio's founding year.",
        parent=qual_node,
        critical=True
    )
    claim = f"{item.studio_name} was founded in {item.founding_year}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            _no_source_instruction(sources)
            + " Verify the founding year from reputable sources (official site, Wikipedia with citations, reputable profiles). "
              "Allow minor presentation variants (e.g., full date vs year only) as long as the year matches."
        )
    )

    # 10) Studio_Approx_Employee_Count (verify via sources)
    leaf = evaluator.add_leaf(
        id=f"Studio_{studio_index+1}_Studio_Approx_Employee_Count",
        desc="Provides an approximate number of employees at the studio.",
        parent=qual_node,
        critical=True
    )
    claim = f"{item.studio_name} has approximately {item.employee_count} employees."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            _no_source_instruction(sources)
            + " Verify that credible sources indicate a similar order-of-magnitude employee count. "
              "Tolerate approximate phrasing (e.g., 'about 200', '200+', '200–250'). If the claim lacks support, mark as NOT SUPPORTED."
        )
    )

    return {
        "studio_name": item.studio_name or "",
        "qual_node": qual_node
    }


# ----------------------------- Main Evaluator ------------------------------ #
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Identify at least two distinct game development studios whose games won a 'Game of the Year' award at either "
            "The Game Awards 2024 or the TIGA Games Industry Awards 2024, and provide all required attributes for each qualifying studio/game "
            "(accurate as of end of 2024)."
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract studio items from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_studios(),
        template_class=StudiosExtraction,
        extraction_name="studios_extraction"
    )

    # Build "Evaluate_Provided_Studio_Items"
    eval_items_node = evaluator.add_parallel(
        id="Evaluate_Provided_Studio_Items",
        desc="Evaluate each provided studio item (up to 5) against the qualification criteria and required attributes.",
        parent=root,
        critical=False
    )

    # Prepare up to 5 studio items (pad with empty items if fewer than 2 are present to keep structure stable)
    studios_list: List[StudioItem] = list(extracted.studios[:5])
    while len(studios_list) < 2:
        studios_list.append(StudioItem())
    while len(studios_list) < 5:
        studios_list.append(StudioItem())

    # Verify each studio item
    qual_nodes: List[VerificationNode] = []
    studio_names: List[str] = []
    for idx, studio_item in enumerate(studios_list[:5]):
        result = await verify_studio_item(evaluator, eval_items_node, studio_item, idx)
        qual_nodes.append(result["qual_node"])
        studio_names.append(result["studio_name"])

    # Compute which Qualifying_Package nodes passed and ensure distinct studios
    passed_names: List[str] = []
    for name, qn in zip(studio_names, qual_nodes):
        # Trigger score computation for each Qualifying_Package
        score = qn.aggregated_score
        if score == 1.0 and name:
            passed_names.append(name)

    # Distinctness by simple normalization
    normalized = [_normalize_name(n) for n in passed_names if _normalize_name(n)]
    distinct_count = len(set(normalized))

    # Add final critical gate: At_Least_Two_Distinct_Studios_Qualify
    evaluator.add_custom_node(
        result=distinct_count >= 2,
        id="At_Least_Two_Distinct_Studios_Qualify",
        desc="At least two of the Studio_# nodes evaluated above pass their Qualifying_Package checks, and the passing items refer to different (non-identical) studios.",
        parent=root,
        critical=True
    )

    # Add custom info summary
    evaluator.add_custom_info(
        info={
            "passed_studios": passed_names,
            "distinct_passed_count": distinct_count,
            "total_items_evaluated": len(studios_list)
        },
        info_type="summary",
        info_name="evaluation_summary"
    )

    return evaluator.get_summary()