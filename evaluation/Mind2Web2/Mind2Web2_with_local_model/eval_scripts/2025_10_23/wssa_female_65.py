import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator, AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wssa_female_65"
TASK_DESCRIPTION = """
Please find the official homepages of three US female athletes from the World Sport Stacking Association who have each participated in 65+ tournaments throughout their careers.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AthleteInfo(BaseModel):
    """Information about a single athlete with all related URLs."""
    name: Optional[str] = None
    related_urls: List[str] = Field(default_factory=list)


class AthletesResponse(BaseModel):
    """The complete list of athletes extracted from the answer."""
    athletes: List[AthleteInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_athletes() -> str:
    return """
    Extract information about each athlete mentioned in the answer. For each athlete, extract:
    1. Their name
    2. ALL related URLs mentioned in connection with this athlete (including homepage, profile pages, any WSSA-related links, etc.)

    If any information is not provided, set the corresponding field to null or empty list.
    Include ALL URLs that could potentially contain information about the athlete, even if they appear to be the same domain.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_athlete_comprehensive(
        evaluator: Evaluator,
        parent_node,
        athlete: AthleteInfo,
        athlete_index: int,
) -> None:
    """
    Verify all criteria for a single athlete with a comprehensive check.
    """
    athlete_name = athlete.name if athlete.name else f"Unknown Athlete {athlete_index}"

    # Create parallel parent node for this athlete's verification
    athlete_parent = evaluator.add_parallel(
        id=f"athlete_{athlete_index}",
        desc=f"Athlete {athlete_index} ({athlete_name}) verification",
        parent=parent_node,
        critical=False  # Allow partial scoring across athletes
    )

    # Critical existence check
    has_data = bool(athlete.name) and bool(athlete.related_urls)
    existence_check = evaluator.add_custom_node(
        result=has_data,
        id=f"athlete_{athlete_index}_exists",
        desc=f"Athlete {athlete_index} has name and URLs provided",
        parent=athlete_parent,
        critical=True
    )

    # Comprehensive verification node
    verification_node = evaluator.add_leaf(
        id=f"athlete_{athlete_index}_comprehensive",
        desc=f"Athlete {athlete_index} ({athlete_name}) meets all required criteria: is female, from the US, has participated in 65+ tournaments, and has a valid WSSA homepage",
        parent=athlete_parent,
        critical=True
    )

    # Always call verify - the existence check will gate this
    comprehensive_claim = f"This webpage is the official WSSA homepage for {athlete.name}, and {athlete.name} is a female athlete from the United States who has participated in 65 or more tournaments throughout her career"

    await evaluator.verify(
        claim=comprehensive_claim,
        node=verification_node,
        sources=athlete.related_urls,
        additional_instruction=f"""
        Verify ALL of the following criteria for {athlete.name}:
        1. The webpage is an official homepage/profile on the World Sport Stacking Association website (e.g., if it appears to be a WSSA page, or the url contains "thewssa.com" or similar)
        2. {athlete.name} is identified as female (If it is the WSSA homepage, this should be clear in the Gender part of the profile)
        3. {athlete.name} is from the United States (If it is the WSSA homepage, this should be clear in the profile)
        4. The page shows evidence that {athlete.name} has participated in 65 or more tournaments (If it is the WSSA homepage, this should be clear in the Tournaments section)

        ALL four criteria must be satisfied for this verification to pass.
        """
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
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Initialize evaluator ----------------------------------- #
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

    # -------- 2. Extract structured info from the answer ---------------- #
    athletes_info = await evaluator.extract(
        prompt=prompt_extract_athletes(),
        template_class=AthletesResponse,
        extraction_name="athletes"
    )

    # -------- 3. Ensure we have exactly 3 athletes for verification ----- #
    # Pad missing athletes with empty AthleteInfo objects
    athletes_to_verify = list(athletes_info.athletes)
    while len(athletes_to_verify) < 3:
        athletes_to_verify.append(AthleteInfo(name=None, related_urls=[]))

    # Only take first 3 if more than 3 provided
    athletes_to_verify = athletes_to_verify[:3]

    # -------- 4. Verify each athlete ------------------------------------ #
    for i, athlete in enumerate(athletes_to_verify):
        await verify_athlete_comprehensive(
            evaluator=evaluator,
            parent_node=root,
            athlete=athlete,
            athlete_index=i + 1,
        )

    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()