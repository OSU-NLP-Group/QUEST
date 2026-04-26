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
TASK_ID = "ewc_2026_identification"
TASK_DESCRIPTION = (
    "Identify the name of the esports tournament that meets ALL of the following criteria:\n\n"
    "1. The tournament has a total prize pool of exactly $75 million USD\n"
    "2. The tournament takes place in the year 2026\n"
    "3. The tournament is held in Riyadh, Saudi Arabia\n"
    "4. The tournament runs from July 6 through August 23, 2026, lasting seven weeks\n"
    "5. The tournament features exactly 24 different competitive game titles\n"
    "6. The tournament consists of 25 separate tournaments\n"
    "7. More than 2,000 players participate in the tournament\n"
    "8. Approximately 200 Clubs participate in the tournament\n"
    "9. Players represent over 100 countries\n"
    "10. The tournament details were officially announced on January 20, 2026, via PRNewswire\n"
    "11. The EWC Club Championship within this tournament awards $30 million to the top 24 Clubs\n"
    "12. Individual Game Championships within this tournament have combined prize allocations exceeding $39 million\n"
    "13. Fortnite and Trackmania are specifically mentioned as new additions to the game lineup\n"
    "14. Tickets for the tournament become available starting January 22, 2026\n"
    "15. The 2026 tournament is described as returning to Riyadh (not the first edition)\n\n"
    "What is the official name of this esports tournament?"
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class TournamentExtraction(BaseModel):
    tournament_name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_tournament() -> str:
    return (
        "From the answer, extract the following fields for the single esports tournament that the answer identifies:\n"
        "- tournament_name: The official name of the tournament as written in the answer (do not add extra descriptors).\n"
        "- source_urls: A list of every URL mentioned in the answer text. Include full URLs that appear either as plain links or within markdown links. "
        "Include all unique URLs that could support the claims (e.g., PRNewswire press release, official tournament site, news releases). "
        "Do not invent URLs. If no URLs are present, return an empty list.\n"
        "Return a JSON object with fields: tournament_name, source_urls."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for u in urls or []:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def _name_or_generic(name: Optional[str]) -> str:
    if name and name.strip():
        return name.strip()
    return "the tournament"


def _build_ins(base_specific: str, tour_name: Optional[str]) -> str:
    tname = _name_or_generic(tour_name)
    return (
        f"Verify the claim strictly against the provided URL sources. "
        f"Treat '{tname}' as equivalent to 'Esports World Cup' and 'EWC' if the context clearly matches the same event. "
        f"Focus on the 2026 edition in Riyadh, Saudi Arabia. "
        f"Minor wording variations are acceptable but the fact itself must be explicitly supported by the page. "
        f"If multiple URLs are provided, it is sufficient that any one clearly supports the claim. "
        f"{base_specific}"
    )


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def add_and_verify_criteria(
    evaluator: Evaluator,
    parent_node,
    tournament_name: Optional[str],
    source_urls: List[str],
) -> None:
    """
    Create all rubric leaf nodes under parent_node and verify them using the provided sources.
    """
    urls = _normalize_urls(source_urls)
    tname = _name_or_generic(tournament_name)

    claims_and_nodes: List[tuple[str, List[str] | None, Any, Optional[str]]] = []

    # 1. Total_Prize_Pool_Exactly_75M_USD
    node_75m = evaluator.add_leaf(
        id="Total_Prize_Pool_Exactly_75M_USD",
        desc="Tournament has a total prize pool of exactly $75 million USD",
        parent=parent_node,
        critical=True,
    )
    claim_75m = f"The total prize pool for the 2026 {tname} is exactly $75 million USD."
    ins_75m = _build_ins(
        "Confirm that the total prize pool is stated as precisely $75,000,000 (not 'over', 'approximately', or any other amount).",
        tournament_name,
    )
    claims_and_nodes.append((claim_75m, urls if urls else None, node_75m, ins_75m))

    # 2. Takes_Place_In_2026
    node_2026 = evaluator.add_leaf(
        id="Takes_Place_In_2026",
        desc="Tournament takes place in the year 2026",
        parent=parent_node,
        critical=True,
    )
    claim_2026 = f"The {tname} takes place in 2026."
    ins_2026 = _build_ins("Confirm that the edition referenced is for the year 2026.", tournament_name)
    claims_and_nodes.append((claim_2026, urls if urls else None, node_2026, ins_2026))

    # 3. Held_In_Riyadh_Saudi_Arabia
    node_riyadh = evaluator.add_leaf(
        id="Held_In_Riyadh_Saudi_Arabia",
        desc="Tournament is held in Riyadh, Saudi Arabia",
        parent=parent_node,
        critical=True,
    )
    claim_riyadh = f"The {tname} is held in Riyadh, Saudi Arabia."
    ins_riyadh = _build_ins("Look for 'Riyadh' in the location fields or body text. References to KSA are acceptable if it explicitly indicates Riyadh.", tournament_name)
    claims_and_nodes.append((claim_riyadh, urls if urls else None, node_riyadh, ins_riyadh))

    # 4. Runs_July_6_To_Aug_23_2026
    node_dates = evaluator.add_leaf(
        id="Runs_July_6_To_Aug_23_2026",
        desc="Tournament runs from July 6 through August 23, 2026",
        parent=parent_node,
        critical=True,
    )
    claim_dates = f"The {tname} runs from July 6 through August 23, 2026."
    ins_dates = _build_ins("Confirm the full start and end dates match exactly July 6 to August 23, 2026.", tournament_name)
    claims_and_nodes.append((claim_dates, urls if urls else None, node_dates, ins_dates))

    # 5. Lasts_Seven_Weeks
    node_7w = evaluator.add_leaf(
        id="Lasts_Seven_Weeks",
        desc="Tournament lasts seven weeks",
        parent=parent_node,
        critical=True,
    )
    claim_7w = f"The {tname} lasts seven weeks."
    ins_7w = _build_ins("Confirm that the duration is described as seven weeks explicitly or equivalent phrasing.", tournament_name)
    claims_and_nodes.append((claim_7w, urls if urls else None, node_7w, ins_7w))

    # 6. Exactly_24_Game_Titles
    node_24titles = evaluator.add_leaf(
        id="Exactly_24_Game_Titles",
        desc="Tournament features exactly 24 different competitive game titles",
        parent=parent_node,
        critical=True,
    )
    claim_24titles = f"The {tname} features exactly 24 different competitive game titles."
    ins_24titles = _build_ins("The number must be exactly 24.", tournament_name)
    claims_and_nodes.append((claim_24titles, urls if urls else None, node_24titles, ins_24titles))

    # 7. Consists_Of_25_Separate_Tournaments
    node_25tournaments = evaluator.add_leaf(
        id="Consists_Of_25_Separate_Tournaments",
        desc="Tournament consists of 25 separate tournaments",
        parent=parent_node,
        critical=True,
    )
    claim_25tournaments = f"The {tname} consists of 25 separate tournaments."
    ins_25tournaments = _build_ins("Confirm that the structure includes 25 distinct tournaments.", tournament_name)
    claims_and_nodes.append((claim_25tournaments, urls if urls else None, node_25tournaments, ins_25tournaments))

    # 8. More_Than_2000_Players
    node_2000p = evaluator.add_leaf(
        id="More_Than_2000_Players",
        desc="More than 2,000 players participate",
        parent=parent_node,
        critical=True,
    )
    claim_2000p = f"More than 2,000 players participate in the {tname}."
    ins_2000p = _build_ins("Look for phrasing like 'over 2,000 players' or a number clearly greater than 2,000.", tournament_name)
    claims_and_nodes.append((claim_2000p, urls if urls else None, node_2000p, ins_2000p))

    # 9. Approximately_200_Clubs
    node_200clubs = evaluator.add_leaf(
        id="Approximately_200_Clubs",
        desc="Approximately 200 Clubs participate",
        parent=parent_node,
        critical=True,
    )
    claim_200clubs = f"Approximately 200 Clubs participate in the {tname}."
    ins_200clubs = _build_ins("Accept language such as 'around 200 clubs' or 'approximately 200 clubs'.", tournament_name)
    claims_and_nodes.append((claim_200clubs, urls if urls else None, node_200clubs, ins_200clubs))

    # 10. Over_100_Countries_Represented
    node_100countries = evaluator.add_leaf(
        id="Over_100_Countries_Represented",
        desc="Players represent over 100 countries",
        parent=parent_node,
        critical=True,
    )
    claim_100countries = f"Players in the {tname} represent over 100 countries."
    ins_100countries = _build_ins("Look for 'over 100 countries' or 'more than 100 countries'.", tournament_name)
    claims_and_nodes.append((claim_100countries, urls if urls else None, node_100countries, ins_100countries))

    # 11. Announced_Jan_20_2026
    node_ann_date = evaluator.add_leaf(
        id="Announced_Jan_20_2026",
        desc="Tournament details were officially announced on January 20, 2026",
        parent=parent_node,
        critical=True,
    )
    claim_ann_date = f"The {tname} details were officially announced on January 20, 2026."
    ins_ann_date = _build_ins("Prefer an official announcement source; confirm the announcement date is January 20, 2026.", tournament_name)
    claims_and_nodes.append((claim_ann_date, urls if urls else None, node_ann_date, ins_ann_date))

    # 12. Announcement_Via_PRNewswire
    node_via_prn = evaluator.add_leaf(
        id="Announcement_Via_PRNewswire",
        desc="The announcement was made via PRNewswire",
        parent=parent_node,
        critical=True,
    )
    claim_via_prn = f"The announcement of the {tname} was made via PRNewswire."
    ins_via_prn = _build_ins("At least one provided URL should be a PRNewswire press release or an official page clearly indicating PRNewswire as the announcement channel.", tournament_name)
    claims_and_nodes.append((claim_via_prn, urls if urls else None, node_via_prn, ins_via_prn))

    # 13. EWC_Club_Championship_30M_To_Top_24_Clubs
    node_club_30m = evaluator.add_leaf(
        id="EWC_Club_Championship_30M_To_Top_24_Clubs",
        desc="EWC Club Championship awards $30 million to the top 24 Clubs",
        parent=parent_node,
        critical=True,
    )
    claim_club_30m = f"The EWC Club Championship within the {tname} awards $30 million to the top 24 Clubs."
    ins_club_30m = _build_ins("Confirm the allocation specifically mentions '$30 million' to 'top 24 Clubs' for the Club Championship.", tournament_name)
    claims_and_nodes.append((claim_club_30m, urls if urls else None, node_club_30m, ins_club_30m))

    # 14. Individual_Game_Championships_Combined_Prize_Exceeds_39M
    node_indiv_39m = evaluator.add_leaf(
        id="Individual_Game_Championships_Combined_Prize_Exceeds_39M",
        desc="Individual Game Championships have combined prize allocations exceeding $39 million",
        parent=parent_node,
        critical=True,
    )
    claim_indiv_39m = f"The Individual Game Championships within the {tname} have combined prize allocations exceeding $39 million."
    ins_indiv_39m = _build_ins("The sum should be clearly indicated as greater than $39 million.", tournament_name)
    claims_and_nodes.append((claim_indiv_39m, urls if urls else None, node_indiv_39m, ins_indiv_39m))

    # 15. Fortnite_Listed_As_New_Addition
    node_fortnite = evaluator.add_leaf(
        id="Fortnite_Listed_As_New_Addition",
        desc="Fortnite is mentioned as a new addition to the game lineup",
        parent=parent_node,
        critical=True,
    )
    claim_fortnite = f"Fortnite is specifically mentioned as a new addition to the 2026 {tname} game lineup."
    ins_fortnite = _build_ins("Look for explicit phrasing like 'new addition', 'added', or 'joining the lineup' referring to Fortnite.", tournament_name)
    claims_and_nodes.append((claim_fortnite, urls if urls else None, node_fortnite, ins_fortnite))

    # 16. Trackmania_Listed_As_New_Addition
    node_trackmania = evaluator.add_leaf(
        id="Trackmania_Listed_As_New_Addition",
        desc="Trackmania is mentioned as a new addition to the game lineup",
        parent=parent_node,
        critical=True,
    )
    claim_trackmania = f"Trackmania is specifically mentioned as a new addition to the 2026 {tname} game lineup."
    ins_trackmania = _build_ins("Look for explicit phrasing like 'new addition', 'added', or 'joining the lineup' referring to Trackmania.", tournament_name)
    claims_and_nodes.append((claim_trackmania, urls if urls else None, node_trackmania, ins_trackmania))

    # 17. Tickets_Available_From_Jan_22_2026
    node_tickets = evaluator.add_leaf(
        id="Tickets_Available_From_Jan_22_2026",
        desc="Tickets become available starting January 22, 2026",
        parent=parent_node,
        critical=True,
    )
    claim_tickets = f"Tickets for the {tname} become available starting January 22, 2026."
    ins_tickets = _build_ins("Confirm that ticket availability begins on January 22, 2026.", tournament_name)
    claims_and_nodes.append((claim_tickets, urls if urls else None, node_tickets, ins_tickets))

    # 18. Returning_To_Riyadh_Not_First_Edition
    node_returning = evaluator.add_leaf(
        id="Returning_To_Riyadh_Not_First_Edition",
        desc="2026 tournament is described as returning to Riyadh (not the first edition)",
        parent=parent_node,
        critical=True,
    )
    claim_returning = f"The 2026 {tname} is described as returning to Riyadh (i.e., not the first edition)."
    ins_returning = _build_ins("Look for language such as 'returns to Riyadh' or 'returning to Riyadh' clearly indicating a prior edition in Riyadh.", tournament_name)
    claims_and_nodes.append((claim_returning, urls if urls else None, node_returning, ins_returning))

    # Execute verifications in parallel
    await evaluator.batch_verify(claims_and_nodes)


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
    Evaluate an answer for the esports tournament identification task.
    """
    # Initialize evaluator with a parallel root
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
        default_model=model,
    )

    # Extract tournament name and sources from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_tournament(),
        template_class=TournamentExtraction,
        extraction_name="tournament_extraction",
    )

    # Build the rubric tree: top-level aggregation node (critical, parallel)
    top = evaluator.add_parallel(
        id="Esports_Tournament_Identification",
        desc="Verify the provided official tournament name satisfies all stated criteria",
        parent=root,
        critical=True,
    )

    # Leaf: Provides_Official_Tournament_Name (existence check)
    name_exists = extraction.tournament_name is not None and extraction.tournament_name.strip() != ""
    evaluator.add_custom_node(
        result=name_exists,
        id="Provides_Official_Tournament_Name",
        desc="Response provides the official name of the esports tournament",
        parent=top,
        critical=True,
    )

    # All remaining criteria leaves verified against provided sources
    await add_and_verify_criteria(
        evaluator=evaluator,
        parent_node=top,
        tournament_name=extraction.tournament_name,
        source_urls=extraction.source_urls or [],
    )

    # Return structured evaluation summary
    return evaluator.get_summary()