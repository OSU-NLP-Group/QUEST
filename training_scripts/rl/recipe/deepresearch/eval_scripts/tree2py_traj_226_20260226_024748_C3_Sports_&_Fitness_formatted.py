import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nfl_largest_stadium_standard_capacity"
TASK_DESCRIPTION = (
    "What is the largest NFL stadium by standard seating capacity that meets the Super Bowl hosting requirement of at least 70,000 seats? "
    "Provide the stadium's official name, exact standard seating capacity, location (city and state), home team(s), and reference URL(s) from "
    "official or reliable sources supporting the capacity information."
)


# --------------------------------------------------------------------------- #
# Data Models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StadiumCandidate(BaseModel):
    official_name: Optional[str] = None
    standard_seating_capacity: Optional[str] = None  # Keep as string; allow commas/phrases
    city: Optional[str] = None
    state: Optional[str] = None
    home_teams: List[str] = Field(default_factory=list)

    # URL fields - explicitly extracted from the answer text
    capacity_reference_urls: List[str] = Field(
        default_factory=list,
        description="URLs that explicitly support the stated standard seating capacity."
    )
    largest_claim_reference_urls: List[str] = Field(
        default_factory=list,
        description="URLs that support the claim that this stadium is the largest by standard seating capacity among NFL stadiums."
    )
    general_reference_urls: List[str] = Field(
        default_factory=list,
        description="Other URLs mentioned for the stadium (official team/stadium pages, Wikipedia, etc.)."
    )


class StadiumAnswerExtraction(BaseModel):
    candidates: List[StadiumCandidate] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadium_answer() -> str:
    return """
    Extract all stadium candidates that the answer identifies as the final or main answer to the query:
    "largest NFL stadium by standard seating capacity that meets the ≥70,000 Super Bowl hosting requirement."
    
    For each candidate, extract the following fields exactly as written in the answer:
    - official_name: The official/current stadium name (string).
    - standard_seating_capacity: The exact standard seating capacity number/phrase as reported (string; keep commas or qualifiers like 'approx.' if present).
    - city: City where the stadium is located (string).
    - state: State where the stadium is located (string).
    - home_teams: A list of NFL home team names associated with this stadium (list of strings).
    - capacity_reference_urls: A list of URLs explicitly cited to support the standard seating capacity figure (list of strings).
    - largest_claim_reference_urls: A list of URLs explicitly cited to support that this stadium is the largest by standard seating capacity among NFL stadiums (list of strings).
    - general_reference_urls: Any other URLs mentioned in the answer for this stadium (list of strings). Include official team/stadium pages or authoritative pages that might support location/home-team/in-use facts. Do not repeat URLs already listed in capacity_reference_urls or largest_claim_reference_urls.
    
    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer text (plain URLs or markdown links).
    - Do not invent URLs; if not provided, leave the list empty.
    - The 'candidates' array should include every stadium that the answer presents as the final/main answer. If the answer names multiple final candidates, include all of them as separate entries in the 'candidates' array. If there is only one, include just that one.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def uniq_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str):
                u_clean = u.strip()
                if u_clean and u_clean not in seen:
                    seen.add(u_clean)
                    merged.append(u_clean)
    return merged


def has_digits(s: Optional[str]) -> bool:
    return bool(s and re.search(r"\d", s))


def pick_first_candidate(extracted: StadiumAnswerExtraction) -> StadiumCandidate:
    if extracted and extracted.candidates:
        return extracted.candidates[0]
    return StadiumCandidate()


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    root_desc: str,
    extraction: StadiumAnswerExtraction,
) -> None:
    # Root node (critical parallel per rubric)
    root = evaluator.add_parallel(
        id="Stadium_Selection_Task",
        desc=root_desc,
        critical=True,
    )

    # Determine candidate to verify (first one, if multiple)
    cand = pick_first_candidate(extraction)

    # Precompute URL bundles
    capacity_urls = cand.capacity_reference_urls or []
    largest_urls = cand.largest_claim_reference_urls or []
    general_urls = cand.general_reference_urls or []
    all_urls = uniq_urls(capacity_urls, largest_urls, general_urls)

    # 1) Single_Stadium_Answered (leaf via custom check)
    evaluator.add_custom_node(
        result=(len(extraction.candidates) == 1),
        id="Single_Stadium_Answered",
        desc="Response identifies exactly one stadium as the final answer candidate (not multiple candidates).",
        parent=root,
        critical=True,
    )

    # 2) Largest_Stadium_Verification (leaf with URL-based verification)
    largest_node = evaluator.add_leaf(
        id="Largest_Stadium_Verification",
        desc="Selected stadium is the largest by standard seating capacity among NFL home stadiums with standard capacity ≥70,000.",
        parent=root,
        critical=True,
    )
    largest_sources = largest_urls if largest_urls else all_urls
    if largest_sources:
        largest_claim = (
            f"{cand.official_name or 'The selected stadium'} is the largest NFL stadium by standard seating capacity "
            f"among current NFL home stadiums that have a standard seating capacity of at least 70,000 seats."
        )
        await evaluator.verify(
            claim=largest_claim,
            node=largest_node,
            sources=largest_sources,
            additional_instruction=(
                "Focus strictly on standard (regular) seated capacity, not expandable, standing-room, or special-event capacity. "
                "A supporting page may explicitly state 'largest' or provide a reliable list/ranking by standard seating capacity that makes this clear. "
                "If evidence is ambiguous or refers only to expandable/record attendance, do not support."
            ),
        )
    else:
        largest_node.score = 0.0
        largest_node.status = "failed"

    # 3) Required_Output_Fields_Provided (parallel critical)
    required_fields = evaluator.add_parallel(
        id="Required_Output_Fields_Provided",
        desc="Provides all required fields requested in the question.",
        parent=root,
        critical=True,
    )

    # 3.a) Official_Stadium_Name_Provided
    name_node = evaluator.add_custom_node(
        result=bool(cand.official_name and cand.official_name.strip()),
        id="Official_Stadium_Name_Provided",
        desc="Provides the stadium's official name.",
        parent=required_fields,
        critical=True,
    )

    # 3.b) Exact_Standard_Seating_Capacity_Provided
    # Require that there's some digits in the field to count as a number provided.
    cap_present_node = evaluator.add_custom_node(
        result=has_digits(cand.standard_seating_capacity),
        id="Exact_Standard_Seating_Capacity_Provided",
        desc="Provides the exact standard seating capacity number.",
        parent=required_fields,
        critical=True,
    )

    # 3.c) Location_City_Provided
    city_node = evaluator.add_custom_node(
        result=bool(cand.city and cand.city.strip()),
        id="Location_City_Provided",
        desc="Provides the stadium's city.",
        parent=required_fields,
        critical=True,
    )

    # 3.d) Location_State_Provided
    state_node = evaluator.add_custom_node(
        result=bool(cand.state and cand.state.strip()),
        id="Location_State_Provided",
        desc="Provides the stadium's state.",
        parent=required_fields,
        critical=True,
    )

    # 3.e) Home_Team_Names_Provided
    home_teams_node = evaluator.add_custom_node(
        result=bool(cand.home_teams and len(cand.home_teams) > 0),
        id="Home_Team_Names_Provided",
        desc="Provides the home team(s) that use the stadium.",
        parent=required_fields,
        critical=True,
    )

    # 3.f) Capacity_Supporting_Reference_URLs_Provided
    capacity_support_node = evaluator.add_leaf(
        id="Capacity_Supporting_Reference_URLs_Provided",
        desc="Provides accessible reference URL(s) from official or reliable sources that explicitly support the stated standard seating capacity.",
        parent=required_fields,
        critical=True,
    )
    if capacity_urls:
        cap_value = cand.standard_seating_capacity or "the stated value"
        cap_claim = (
            f"The provided URL(s) explicitly support that the standard seating capacity of "
            f"{cand.official_name or 'the stadium'} is {cap_value}."
        )
        await evaluator.verify(
            claim=cap_claim,
            node=capacity_support_node,
            sources=capacity_urls,
            additional_instruction=(
                "Accept sources that explicitly list the stadium's standard/regular seating capacity matching the stated value. "
                "Do not accept sources that only mention expandable/maximum/record attendance figures without the standard capacity. "
                "Official team/stadium sites, the NFL site, and well-maintained, up-to-date encyclopedic pages (e.g., Wikipedia) "
                "are acceptable if they clearly state standard seating capacity."
            ),
        )
    else:
        capacity_support_node.score = 0.0
        capacity_support_node.status = "failed"

    # 4) Eligibility_Constraints_Satisfied (parallel critical)
    eligibility = evaluator.add_parallel(
        id="Eligibility_Constraints_Satisfied",
        desc="Selected stadium satisfies the stated eligibility constraints (NFL home stadium, US, in use 2025–26, standard capacity used, standard capacity ≥70,000).",
        parent=root,
        critical=True,
    )

    # 4.a) Is_NFL_Home_Stadium
    nfl_home_node = evaluator.add_leaf(
        id="Is_NFL_Home_Stadium",
        desc="Stadium is home to at least one NFL team.",
        parent=eligibility,
        critical=True,
    )
    nfl_sources = all_urls if all_urls else capacity_urls
    if nfl_sources and cand.home_teams:
        team_list = ", ".join(cand.home_teams)
        nfl_claim = (
            f"{cand.official_name or 'The stadium'} serves as the home stadium for the NFL team(s): {team_list}."
        )
        await evaluator.verify(
            claim=nfl_claim,
            node=nfl_home_node,
            sources=nfl_sources,
            additional_instruction=(
                "Confirm that the named teams are NFL teams and that the stadium is indeed their home venue. "
                "Do not accept college-only or non-NFL usage as sufficient."
            ),
            extra_prerequisites=[home_teams_node],  # Skip if home teams missing
        )
    else:
        nfl_home_node.score = 0.0
        nfl_home_node.status = "failed"

    # 4.b) Located_In_United_States
    loc_us_node = evaluator.add_leaf(
        id="Located_In_United_States",
        desc="Stadium is located in the United States.",
        parent=eligibility,
        critical=True,
    )
    loc_sources = all_urls if all_urls else capacity_urls
    if loc_sources and cand.city and cand.state:
        loc_claim = (
            f"{cand.official_name or 'The stadium'} is located in {cand.city}, {cand.state}, United States."
        )
        await evaluator.verify(
            claim=loc_claim,
            node=loc_us_node,
            sources=loc_sources,
            additional_instruction="Verify that the stadium's location is in the U.S., matching the stated city and state.",
            extra_prerequisites=[city_node, state_node],
        )
    else:
        loc_us_node.score = 0.0
        loc_us_node.status = "failed"

    # 4.c) In_Use_For_2025_26_Season
    in_use_node = evaluator.add_leaf(
        id="In_Use_For_2025_26_Season",
        desc="Stadium is currently in use for the 2025–26 NFL season.",
        parent=eligibility,
        critical=True,
    )
    in_use_sources = all_urls if all_urls else capacity_urls
    if in_use_sources and cand.home_teams:
        in_use_claim = (
            f"{cand.official_name or 'The stadium'} is in active use as an NFL home stadium for the 2025–26 season."
        )
        await evaluator.verify(
            claim=in_use_claim,
            node=in_use_node,
            sources=in_use_sources,
            additional_instruction=(
                "Treat current official team/stadium listings or authoritative sources that indicate the stadium is the active home "
                "venue during the 2025–26 NFL season as sufficient. If evidence shows it has been replaced or is not in use for that season, do not support."
            ),
            extra_prerequisites=[home_teams_node],
        )
    else:
        in_use_node.score = 0.0
        in_use_node.status = "failed"

    # 4.d) Uses_Standard_Seating_Capacity_Figure
    std_cap_node = evaluator.add_leaf(
        id="Uses_Standard_Seating_Capacity_Figure",
        desc="Capacity figure used is the standard listed seating capacity (not peak/expandable/event-specific capacity).",
        parent=eligibility,
        critical=True,
    )
    if capacity_urls and cand.standard_seating_capacity:
        std_cap_claim = (
            f"The capacity figure {cand.standard_seating_capacity} for {cand.official_name or 'the stadium'} refers to the standard seating capacity, "
            f"not an expandable or special-event capacity."
        )
        await evaluator.verify(
            claim=std_cap_claim,
            node=std_cap_node,
            sources=capacity_urls,
            additional_instruction=(
                "Confirm that the cited number is labeled as 'capacity' or 'seating capacity' in the standard sense. "
                "If a page only mentions 'expandable to' or special configurations, it should not count as standard."
            ),
            extra_prerequisites=[cap_present_node, name_node],
        )
    else:
        std_cap_node.score = 0.0
        std_cap_node.status = "failed"

    # 4.e) Meets_70000_Minimum_Capacity
    ge70_node = evaluator.add_leaf(
        id="Meets_70000_Minimum_Capacity",
        desc="Standard seating capacity is at least 70,000 seats.",
        parent=eligibility,
        critical=True,
    )
    if capacity_urls:
        ge70_claim = (
            f"The standard seating capacity of {cand.official_name or 'the stadium'} is at least 70,000 seats."
        )
        await evaluator.verify(
            claim=ge70_claim,
            node=ge70_node,
            sources=capacity_urls,
            additional_instruction=(
                "Use the same understanding of 'standard seating capacity'. If the supporting pages provide a precise number, "
                "judge whether it is ≥ 70,000. Do not use expandable capacities."
            ),
            extra_prerequisites=[std_cap_node],
        )
    else:
        ge70_node.score = 0.0
        ge70_node.status = "failed"


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
    Evaluate an answer for the NFL largest stadium (standard capacity) task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel (critical gating handled by critical flags)
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

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_stadium_answer(),
        template_class=StadiumAnswerExtraction,
        extraction_name="stadium_answer_extraction",
    )

    # Build verification tree and run verifications
    await build_and_verify(
        evaluator=evaluator,
        root_desc="Identify the largest NFL stadium by standard seating capacity that meets the ≥70,000 Super Bowl hosting requirement, and provide the required details with reliable references.",
        extraction=extraction,
    )

    # Optionally record some custom info (for debugging/report)
    try:
        cand = pick_first_candidate(extraction)
        evaluator.add_custom_info(
            info={
                "official_name": cand.official_name,
                "standard_seating_capacity": cand.standard_seating_capacity,
                "city": cand.city,
                "state": cand.state,
                "home_teams": cand.home_teams,
                "capacity_reference_urls": cand.capacity_reference_urls,
                "largest_claim_reference_urls": cand.largest_claim_reference_urls,
                "general_reference_urls": cand.general_reference_urls,
                "candidate_count": len(extraction.candidates),
            },
            info_type="parsed_candidate_summary",
        )
    except Exception:
        pass

    # Return summary
    return evaluator.get_summary()