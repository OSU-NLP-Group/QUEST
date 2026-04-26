import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task metadata
# -----------------------------------------------------------------------------
TASK_ID = "msu_oc_2026"
TASK_DESCRIPTION = (
    "Identify the offensive coordinator who was hired by Michigan State University's football program and whose "
    "appointment was officially announced on January 2, 2026. Provide the following information: the coordinator's "
    "full name, the exact date of the official announcement, their previous position and team immediately before "
    "joining Michigan State, the number of seasons they spent at their previous team, the university where they played "
    "college football as a player, the length of the contract they signed with Michigan State, and at least two URL "
    "references from official university sources or reliable sports news outlets confirming this information."
)

EXPECTED_CONSTRAINTS = {
    "official_announcement_date": "January 2, 2026",
    "head_coach": "Pat Fitzgerald",
    "previous_team": "Alabama (Alabama Crimson Tide)",
    "previous_position": "co-offensive coordinator and quarterbacks coach",
    "seasons_at_previous_team": "two seasons (2024–2025)",
    "college_played_for": "University of Michigan",
    "contract_length": "three years"
}


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class CoordinatorHireExtraction(BaseModel):
    full_name: Optional[str] = None
    announcement_date: Optional[str] = None
    head_coach: Optional[str] = None
    previous_team: Optional[str] = None
    previous_position: Optional[str] = None
    seasons_at_previous_team: Optional[str] = None
    college_played_for: Optional[str] = None
    contract_length: Optional[str] = None
    cfp_context_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_hire_info() -> str:
    return """
    From the provided answer, extract the following fields exactly as they appear in the answer text. Do not infer.
    If an item is missing, set it to null. For URLs, extract only actual URLs explicitly present in the answer.

    Fields to extract:
    - full_name: The full name of the person identified as Michigan State's offensive coordinator in the answer.
    - announcement_date: The official announcement date as written in the answer (e.g., "January 2, 2026", "Jan. 2, 2026").
    - head_coach: The name of the head coach who made or announced the hire (if stated).
    - previous_team: The immediate previous team/employer before joining Michigan State (as stated).
    - previous_position: The immediate previous position/title before joining Michigan State (as stated).
    - seasons_at_previous_team: How long they spent at the previous team (as stated; e.g., "two seasons (2024–2025)").
    - college_played_for: The university where the person played college football.
    - contract_length: The contract length signed with Michigan State (e.g., "three years", "3-year deal").
    - cfp_context_statement: If the answer notes that the appointment followed Alabama's College Football Playoff (CFP) run,
      copy that clause/statement. Otherwise null.
    - sources: A list of all URLs provided in the answer (including press releases, official sites, or reliable sports news).
      Extract only actual URLs (plain or markdown link URLs). Deduplicate if repeated.

    Return a JSON object with those exact keys.
    """


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    if url.strip().lower().startswith("mailto:"):
        return False
    pattern = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)
    return bool(pattern.match(url.strip()))


def normalize_and_dedupe_urls(urls: List[str], max_urls: int = 10) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u2 = u.strip()
        # Ensure protocol if missing is not enforced here; extraction tool may prepend; keep as-is but filter invalids
        if is_valid_url(u2) and u2 not in seen:
            cleaned.append(u2)
            seen.add(u2)
        if len(cleaned) >= max_urls:
            break
    return cleaned


# -----------------------------------------------------------------------------
# Verification subroutines
# -----------------------------------------------------------------------------
async def verify_identity(evaluator: Evaluator, parent_node, info: CoordinatorHireExtraction) -> None:
    node = evaluator.add_parallel(
        id="identity",
        desc="Correctly identifies the offensive coordinator hired by Michigan State in the relevant announcement.",
        parent=parent_node,
        critical=True
    )

    # Leaf: coordinator_full_name (existence)
    name_exists = bool(info.full_name and info.full_name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id="coordinator_full_name",
        desc="Provides the offensive coordinator's full name.",
        parent=node,
        critical=True
    )

    # Leaf: official_announcement_date = January 2, 2026 (check the answer states this date)
    date_leaf = evaluator.add_leaf(
        id="official_announcement_date",
        desc="States the official announcement date as January 2, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the official announcement date is January 2, 2026 (allowing formats like 'Jan. 2, 2026').",
        node=date_leaf,
        additional_instruction="Judge only based on whether the answer text asserts this specific date, allowing minor formatting variations like 'Jan. 2, 2026' vs 'January 2, 2026'."
    )

    # Leaf: hired_by_head_coach = Pat Fitzgerald (check the answer states this)
    hired_by_leaf = evaluator.add_leaf(
        id="hired_by_head_coach",
        desc="States that the hiring was made by head coach Pat Fitzgerald.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the hire was made or announced by head coach Pat Fitzgerald.",
        node=hired_by_leaf,
        additional_instruction="Look for explicit mention in the answer text that head coach Pat Fitzgerald made or announced the hire."
    )


async def verify_prior_role_and_tenure(evaluator: Evaluator, parent_node, info: CoordinatorHireExtraction) -> None:
    node = evaluator.add_parallel(
        id="prior_role_and_tenure_constraints",
        desc="Correctly states the coordinator's immediately previous employment details and tenure constraints before joining Michigan State.",
        parent=parent_node,
        critical=True
    )

    # previous_team_immediately_before_msu -> Alabama (Alabama Crimson Tide)
    prev_team_leaf = evaluator.add_leaf(
        id="previous_team_immediately_before_msu",
        desc="Identifies the previous team/employer immediately before Michigan State as Alabama (Alabama Crimson Tide).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the immediately previous team before joining Michigan State was Alabama (the Alabama Crimson Tide).",
        node=prev_team_leaf,
        additional_instruction="Allow reasonable wording like 'Alabama' or 'Alabama Crimson Tide'. Focus on whether the answer conveys this."
    )

    # previous_position_immediately_before_msu -> co-offensive coordinator and quarterbacks coach
    prev_pos_leaf = evaluator.add_leaf(
        id="previous_position_immediately_before_msu",
        desc="Identifies the previous position immediately before Michigan State as co-offensive coordinator and quarterbacks coach.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the immediately previous position was co-offensive coordinator and quarterbacks coach.",
        node=prev_pos_leaf,
        additional_instruction="Minor phrasing variations are acceptable, e.g., 'co-OC and QBs coach', 'co-offensive coordinator/quarterbacks coach'."
    )

    # seasons_at_previous_team -> two seasons (2024–2025)
    seasons_leaf = evaluator.add_leaf(
        id="seasons_at_previous_team",
        desc="States the coordinator spent two seasons at the previous team (2024–2025).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the coordinator spent two seasons (2024–2025) at the previous team.",
        node=seasons_leaf,
        additional_instruction="Accept expressions like 'two years', '2024 and 2025', or an en dash '2024–2025'."
    )

    # context_followed_cfp_run -> appointment followed Alabama's College Football Playoff run
    cfp_leaf = evaluator.add_leaf(
        id="context_followed_cfp_run",
        desc="Notes that the appointment followed Alabama's College Football Playoff run.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer notes that the appointment followed Alabama's College Football Playoff (CFP) run.",
        node=cfp_leaf,
        additional_instruction="Phrases like 'following Alabama's CFP run', 'after the College Football Playoff', or similar should count."
    )


async def verify_background_and_contract(evaluator: Evaluator, parent_node, info: CoordinatorHireExtraction) -> None:
    node = evaluator.add_parallel(
        id="background_and_contract_constraints",
        desc="Correctly states the coordinator's playing background and Michigan State contract constraint.",
        parent=parent_node,
        critical=True
    )

    # college_played_for -> University of Michigan
    college_leaf = evaluator.add_leaf(
        id="college_played_for",
        desc="Identifies the university where the coordinator played college football as the University of Michigan.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the coordinator played college football at the University of Michigan.",
        node=college_leaf,
        additional_instruction="Allow 'Michigan'/'Michigan Wolverines' as equivalent to 'University of Michigan'."
    )

    # contract_length -> three years
    contract_leaf = evaluator.add_leaf(
        id="contract_length",
        desc="States the contract length signed with Michigan State as three years.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the contract length signed with Michigan State is three years.",
        node=contract_leaf,
        additional_instruction="Allow '3-year deal', 'three-year contract', or equivalent phrasing."
    )


async def verify_sources_and_corroboration(evaluator: Evaluator, parent_node, info: CoordinatorHireExtraction) -> None:
    node = evaluator.add_parallel(
        id="sources_and_corroboration",
        desc="Provides at least two acceptable URLs and they corroborate the required facts (collectively, not necessarily each fact per single source).",
        parent=parent_node,
        critical=True
    )

    valid_sources = normalize_and_dedupe_urls(info.sources or [], max_urls=10)

    # url_reference_1
    evaluator.add_custom_node(
        result=len(valid_sources) >= 1,
        id="url_reference_1",
        desc="Provides one valid URL from an official university source or a reliable sports news outlet.",
        parent=node,
        critical=True
    )

    # url_reference_2
    evaluator.add_custom_node(
        result=len(valid_sources) >= 2,
        id="url_reference_2",
        desc="Provides a second valid URL from an official university source or a reliable sports news outlet.",
        parent=node,
        critical=True
    )

    corr = evaluator.add_parallel(
        id="corroboration_by_attribute",
        desc="Across the provided sources, each required fact is corroborated by at least one of the URLs.",
        parent=node,
        critical=True
    )

    # If no sources, add failed custom leaves for all corroboration items to avoid fallback to simple_verify.
    if len(valid_sources) == 0:
        evaluator.add_custom_node(
            result=False,
            id="supports_hire_and_role",
            desc="At least one provided source corroborates that the named person was hired/appointed as Michigan State offensive coordinator.",
            parent=corr,
            critical=True
        )
        evaluator.add_custom_node(False, "supports_announcement_date",
                                  "At least one provided source corroborates the official announcement date (January 2, 2026).",
                                  parent=corr, critical=True)
        evaluator.add_custom_node(False, "supports_hired_by_fitzgerald",
                                  "At least one provided source corroborates that head coach Pat Fitzgerald made/announced the hire.",
                                  parent=corr, critical=True)
        evaluator.add_custom_node(False, "supports_previous_team_alabama",
                                  "At least one provided source corroborates that the previous team/employer immediately before Michigan State was Alabama (Alabama Crimson Tide).",
                                  parent=corr, critical=True)
        evaluator.add_custom_node(False, "supports_previous_position_cooc_qb",
                                  "At least one provided source corroborates that the previous position was co-offensive coordinator and quarterbacks coach.",
                                  parent=corr, critical=True)
        evaluator.add_custom_node(False, "supports_two_seasons_2024_2025",
                                  "At least one provided source corroborates the two-season tenure at Alabama (2024–2025).",
                                  parent=corr, critical=True)
        evaluator.add_custom_node(False, "supports_played_at_michigan",
                                  "At least one provided source corroborates that the coordinator played college football at the University of Michigan.",
                                  parent=corr, critical=True)
        evaluator.add_custom_node(False, "supports_three_year_contract",
                                  "At least one provided source corroborates the three-year Michigan State contract term.",
                                  parent=corr, critical=True)
        evaluator.add_custom_node(False, "supports_cfp_run_context",
                                  "At least one provided source corroborates the statement that the appointment followed Alabama's College Football Playoff run.",
                                  parent=corr, critical=True)
        return

    # Create leaves
    hire_role_leaf = evaluator.add_leaf(
        id="supports_hire_and_role",
        desc="At least one provided source corroborates that the named person was hired/appointed as Michigan State offensive coordinator.",
        parent=corr,
        critical=True
    )
    ann_date_leaf = evaluator.add_leaf(
        id="supports_announcement_date",
        desc="At least one provided source corroborates the official announcement date (January 2, 2026).",
        parent=corr,
        critical=True
    )
    hired_by_fitz_leaf = evaluator.add_leaf(
        id="supports_hired_by_fitzgerald",
        desc="At least one provided source corroborates that head coach Pat Fitzgerald made/announced the hire.",
        parent=corr,
        critical=True
    )
    prev_team_leaf = evaluator.add_leaf(
        id="supports_previous_team_alabama",
        desc="At least one provided source corroborates that the previous team/employer immediately before Michigan State was Alabama (Alabama Crimson Tide).",
        parent=corr,
        critical=True
    )
    prev_pos_leaf = evaluator.add_leaf(
        id="supports_previous_position_cooc_qb",
        desc="At least one provided source corroborates that the previous position was co-offensive coordinator and quarterbacks coach.",
        parent=corr,
        critical=True
    )
    seasons_leaf = evaluator.add_leaf(
        id="supports_two_seasons_2024_2025",
        desc="At least one provided source corroborates the two-season tenure at Alabama (2024–2025).",
        parent=corr,
        critical=True
    )
    played_mich_leaf = evaluator.add_leaf(
        id="supports_played_at_michigan",
        desc="At least one provided source corroborates that the coordinator played college football at the University of Michigan.",
        parent=corr,
        critical=True
    )
    contract_leaf = evaluator.add_leaf(
        id="supports_three_year_contract",
        desc="At least one provided source corroborates the three-year Michigan State contract term.",
        parent=corr,
        critical=True
    )
    cfp_ctx_leaf = evaluator.add_leaf(
        id="supports_cfp_run_context",
        desc="At least one provided source corroborates the statement that the appointment followed Alabama's College Football Playoff run.",
        parent=corr,
        critical=True
    )

    name_for_claim = info.full_name if (info.full_name and info.full_name.strip()) else "the named person"

    # Batch verification for URL-supported claims
    claims_and_sources = [
        (
            f"The page explicitly states that {name_for_claim} was hired, appointed, or named as Michigan State's offensive coordinator.",
            valid_sources,
            hire_role_leaf,
            "Headlines or body text are acceptable. Keywords like 'hired', 'named', 'appointed', 'offensive coordinator' for Michigan State should appear."
        ),
        (
            "The page indicates the official announcement date as January 2, 2026 (e.g., page publish date or explicit mention).",
            valid_sources,
            ann_date_leaf,
            "Accept explicit publish date of the press release as evidence. Allow minor format variations like 'Jan. 2, 2026'."
        ),
        (
            "The page states that head coach Pat Fitzgerald made or announced the hire.",
            valid_sources,
            hired_by_fitz_leaf,
            "Look for phrases like 'head coach Pat Fitzgerald announced', 'Pat Fitzgerald named X as MSU OC'."
        ),
        (
            f"The page states that before joining Michigan State, {name_for_claim} was with Alabama (Alabama Crimson Tide).",
            valid_sources,
            prev_team_leaf,
            "Accept 'Alabama' or 'Alabama Crimson Tide'."
        ),
        (
            f"The page states that {name_for_claim} previously served as co-offensive coordinator and quarterbacks coach.",
            valid_sources,
            prev_pos_leaf,
            "Allow hyphenation and shorthand variants like 'co-OC/QBs', 'co-offensive coordinator/quarterbacks coach'."
        ),
        (
            f"The page indicates that {name_for_claim} spent two seasons (2024–2025) at Alabama.",
            valid_sources,
            seasons_leaf,
            "Accept 'two years', '2024 and 2025', or an en dash '2024–2025'."
        ),
        (
            f"The page states that {name_for_claim} played college football at the University of Michigan.",
            valid_sources,
            played_mich_leaf,
            "Allow references to 'Michigan Wolverines'."
        ),
        (
            f"The page states that {name_for_claim} signed a three-year contract with Michigan State.",
            valid_sources,
            contract_leaf,
            "Accept 'three-year deal', '3-year contract', or equivalent."
        ),
        (
            "The page notes that the appointment followed Alabama's College Football Playoff (CFP) run.",
            valid_sources,
            cfp_ctx_leaf,
            "Look for phrasing like 'following Alabama's CFP run' or 'after the College Football Playoff'."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is a neutral aggregator; we'll add a critical child wrapper
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

    # Create a critical overall node to reflect rubric root criticality
    overall = evaluator.add_parallel(
        id="overall_evaluation",
        desc="Evaluate whether the answer identifies the Michigan State football offensive coordinator announced on Jan 2, 2026 and provides all required attributes/constraints with adequate sourcing.",
        parent=root,
        critical=True
    )

    # Extraction
    extracted: CoordinatorHireExtraction = await evaluator.extract(
        prompt=prompt_extract_hire_info(),
        template_class=CoordinatorHireExtraction,
        extraction_name="hire_info"
    )

    # Prepare and record sources
    valid_sources = normalize_and_dedupe_urls(extracted.sources or [], max_urls=10)
    evaluator.add_custom_info(
        info={"extracted_sources": extracted.sources, "valid_sources_used": valid_sources},
        info_type="extraction_aux",
        info_name="source_urls_info"
    )

    # Add ground truth/constraints for transparency
    evaluator.add_ground_truth(
        {
            "expected_constraints": EXPECTED_CONSTRAINTS,
            "note": "These are the constraints that the answer must satisfy and that sources should corroborate."
        }
    )

    # Build verification tree
    await verify_identity(evaluator, overall, extracted)
    await verify_prior_role_and_tenure(evaluator, overall, extracted)
    await verify_background_and_contract(evaluator, overall, extracted)
    await verify_sources_and_corroboration(evaluator, overall, extracted)

    return evaluator.get_summary()