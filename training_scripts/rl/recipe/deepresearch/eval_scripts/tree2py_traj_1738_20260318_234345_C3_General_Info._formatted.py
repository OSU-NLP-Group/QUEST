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
TASK_ID = "hadestown_orpheus_2026"
TASK_DESCRIPTION = """
Identify the Broadway actor who is currently playing Orpheus in Hadestown as of March 2026. For this actor, provide the following information in sequence: (1) their exact birth date and birthplace, (2) the full name of their twin sibling, (3) the specific role they played in the Broadway production of Les Misérables during 2014-2015, and (4) details about their performance as the youngest Jean Valjean in the 2015 Broadway Easter Bonnet Competition. Include reference URLs for each piece of information.
"""

# Canonical facts for verification
CANONICAL_ACTOR = "Joshua Colley"
CANONICAL_ROLE_HADESTOWN = "Orpheus"
CANONICAL_START_DATE = "March 3, 2026"

CANONICAL_BIRTH_DATE = "January 20, 2002"
ALLOWED_BIRTHPLACES = ["New Port Richey, Florida", "Trinity, Florida"]

CANONICAL_TWIN_NAME = "Cameron Colley"

LES_MIS_ROLE = "Gavroche"
LES_MIS_PERIOD = "2014-2015"

EASTER_BONNET_YEAR = 2015

# --------------------------------------------------------------------------- #
# Pydantic models for structured extraction                                   #
# --------------------------------------------------------------------------- #
class ActorIdentification(BaseModel):
    actor_name: Optional[str] = None
    role_name: Optional[str] = None
    start_date: Optional[str] = None
    actor_urls: List[str] = Field(default_factory=list)


class BirthInfo(BaseModel):
    birth_date: Optional[str] = None
    birthplace: Optional[str] = None
    birth_urls: List[str] = Field(default_factory=list)


class TwinInfo(BaseModel):
    twin_full_name: Optional[str] = None
    twin_urls: List[str] = Field(default_factory=list)


class LesMisInfo(BaseModel):
    role_name: Optional[str] = None
    time_period: Optional[str] = None
    les_mis_urls: List[str] = Field(default_factory=list)


class EasterBonnetInfo(BaseModel):
    performance_details: Optional[str] = None
    easter_bonnet_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_actor_identification() -> str:
    return """
    From the answer, extract information about who (if anyone) is identified as playing Orpheus in Hadestown as of March 2026.
    Return JSON fields:
    - actor_name: the actor's full name as stated in the answer text
    - role_name: the role name as written in the answer (should be 'Orpheus' if provided)
    - start_date: the stated start date for this actor's first performance (e.g., 'March 3, 2026'); if a range or approximate is given, extract exactly what appears
    - actor_urls: an array of all URLs explicitly cited to support the identification and/or start date
    If any field is not present, return null for that field and an empty array for URLs when none are given.
    """


def prompt_extract_birth_info() -> str:
    return """
    Extract the birth information for the identified actor from the answer text.
    Return JSON fields:
    - birth_date: the exact date of birth as written (e.g., 'January 20, 2002')
    - birthplace: the birthplace as written (e.g., 'New Port Richey, Florida' or 'Trinity, Florida')
    - birth_urls: an array of URLs explicitly cited to support the birth date and/or birthplace
    If any field is not mentioned, return null, and return an empty array for URLs if none are present.
    """


def prompt_extract_twin_info() -> str:
    return """
    Extract information about the twin sibling of the identified actor from the answer text.
    Return JSON fields:
    - twin_full_name: the full name of the twin sibling as written (e.g., 'Cameron Colley')
    - twin_urls: an array of URLs explicitly cited to support the twin sibling information
    If the information is not provided, return null and an empty array of URLs.
    """


def prompt_extract_les_mis_info() -> str:
    return """
    Extract information about the actor's role in the Broadway production of Les Misérables during 2014-2015, as provided in the answer text.
    Return JSON fields:
    - role_name: the specific role name as written (e.g., 'Gavroche')
    - time_period: the time period string as written (e.g., '2014-2015', '2014 to 2015', or '2014–2015')
    - les_mis_urls: an array of URLs explicitly cited to support this Les Misérables role and period
    If any field is missing, return null and provide an empty URLs array if none are present.
    """


def prompt_extract_easter_bonnet_info() -> str:
    return """
    Extract details about the 2015 Broadway Easter Bonnet Competition performance as described in the answer.
    Return JSON fields:
    - performance_details: the exact phrasing provided (e.g., 'performed as the youngest Jean Valjean in the 2015 Easter Bonnet Competition')
    - easter_bonnet_urls: an array of URLs explicitly cited to support this performance detail
    If not provided, return null and an empty array of URLs.
    """


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_actor_identification_checks(
    evaluator: Evaluator,
    parent_node,
    actor_info: ActorIdentification,
) -> None:
    node = evaluator.add_parallel(
        id="actor_identification",
        desc="Correctly identify Joshua Colley as the actor playing Orpheus in Hadestown starting March 3, 2026, with reference URL",
        parent=parent_node,
        critical=True
    )

    # Actor identity (simple name equivalence)
    leaf_identity = evaluator.add_leaf(
        id="actor_identity",
        desc="Identify the actor as Joshua Colley playing Orpheus in Hadestown",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The identified actor name '{actor_info.actor_name or ''}' refers to '{CANONICAL_ACTOR}'.",
        node=leaf_identity,
        additional_instruction="This is a pure name-equivalence check; allow case-insensitive matching and minor formatting differences."
    )

    # Start date (string denotes March 3, 2026)
    leaf_start = evaluator.add_leaf(
        id="start_date",
        desc="Verify the start date as March 3, 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided start date string '{actor_info.start_date or ''}' denotes {CANONICAL_START_DATE}.",
        node=leaf_start,
        additional_instruction="Accept common variants like 'Mar 3, 2026', '03/03/2026', or 'March 3rd, 2026'. This is a date-normalization check."
    )

    # Actor reference URL(s) support identity and start date
    leaf_ref = evaluator.add_leaf(
        id="actor_reference_url",
        desc="Provide reference URL supporting the actor identification and role",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms that {CANONICAL_ACTOR} plays {CANONICAL_ROLE_HADESTOWN} in the Broadway musical Hadestown, with first performance on {CANONICAL_START_DATE}.",
        node=leaf_ref,
        sources=actor_info.actor_urls,
        additional_instruction="The page must explicitly mention Joshua Colley and the Orpheus role; ideally it also specifies the March 3, 2026 start date."
    )


async def build_biographical_checks(
    evaluator: Evaluator,
    parent_node,
    birth_info: BirthInfo,
    twin_info: TwinInfo,
) -> None:
    bio_node = evaluator.add_parallel(
        id="biographical_information",
        desc="Provide core biographical information including birth details and twin sibling",
        parent=parent_node,
        critical=False
    )

    # Birth details
    birth_node = evaluator.add_parallel(
        id="birth_details",
        desc="Provide exact birth date (January 20, 2002) and birthplace (New Port Richey, Florida) with reference URL",
        parent=bio_node,
        critical=False
    )

    # Birth date (string denotes Jan 20, 2002)
    leaf_birth_date = evaluator.add_leaf(
        id="birth_date",
        desc="Verify birth date as January 20, 2002",
        parent=birth_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided birth date string '{birth_info.birth_date or ''}' denotes {CANONICAL_BIRTH_DATE}.",
        node=leaf_birth_date,
        additional_instruction="This is a date-normalization check; accept common formatting variants."
    )

    # Birthplace (string denotes allowed birthplace)
    leaf_birthplace = evaluator.add_leaf(
        id="birthplace",
        desc="Verify birthplace as New Port Richey, Florida (or Trinity, Florida)",
        parent=birth_node,
        critical=True
    )
    allowed_places_text = " or ".join(ALLOWED_BIRTHPLACES)
    await evaluator.verify(
        claim=f"The provided birthplace string '{birth_info.birthplace or ''}' denotes a location equivalent to {allowed_places_text}.",
        node=leaf_birthplace,
        additional_instruction="Allow reasonable variants including inclusion of county (Pasco County) or minor punctuation/casing differences."
    )

    # Birth reference URL(s) support birth date and birthplace
    leaf_birth_ref = evaluator.add_leaf(
        id="birth_reference_url",
        desc="Provide reference URL supporting birth information",
        parent=birth_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms that {CANONICAL_ACTOR} was born on {CANONICAL_BIRTH_DATE} in either New Port Richey, Florida or Trinity, Florida.",
        node=leaf_birth_ref,
        sources=birth_info.birth_urls,
        additional_instruction="The evidence page should explicitly state the date of birth and at least one of the two acceptable birthplaces."
    )

    # Twin sibling
    twin_node = evaluator.add_parallel(
        id="twin_sibling",
        desc="Provide the full name of twin sibling (Cameron Colley) with reference URL",
        parent=bio_node,
        critical=False
    )

    leaf_twin_name = evaluator.add_leaf(
        id="twin_name",
        desc="Identify twin brother as Cameron Colley",
        parent=twin_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided twin sibling full name '{twin_info.twin_full_name or ''}' equals '{CANONICAL_TWIN_NAME}'.",
        node=leaf_twin_name,
        additional_instruction="Full name equivalence; allow case-insensitive match."
    )

    leaf_twin_ref = evaluator.add_leaf(
        id="twin_reference_url",
        desc="Provide reference URL supporting twin sibling information",
        parent=twin_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms that {CANONICAL_ACTOR} has a twin sibling named {CANONICAL_TWIN_NAME}.",
        node=leaf_twin_ref,
        sources=twin_info.twin_urls,
        additional_instruction="The page should explicitly mention that Joshua Colley has a twin and that the twin's name is Cameron Colley."
    )


async def build_career_checks(
    evaluator: Evaluator,
    parent_node,
    les_mis_info: LesMisInfo,
    easter_bonnet_info: EasterBonnetInfo,
) -> None:
    career_node = evaluator.add_parallel(
        id="career_history",
        desc="Provide specific Broadway career information from Les Misérables and Easter Bonnet Competition",
        parent=parent_node,
        critical=False
    )

    # Les Misérables role
    lesmis_node = evaluator.add_parallel(
        id="les_miserables_role",
        desc="Specify the role of Gavroche in Les Misérables during 2014-2015 with reference URL",
        parent=career_node,
        critical=False
    )

    leaf_role_name = evaluator.add_leaf(
        id="role_name",
        desc="Identify the role as Gavroche",
        parent=lesmis_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided role name '{les_mis_info.role_name or ''}' equals '{LES_MIS_ROLE}'.",
        node=leaf_role_name,
        additional_instruction="Simple name equivalence; allow case-insensitive match."
    )

    leaf_time_period = evaluator.add_leaf(
        id="time_period",
        desc="Verify the time period as 2014-2015",
        parent=lesmis_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided time period string '{les_mis_info.time_period or ''}' denotes the years 2014 to 2015.",
        node=leaf_time_period,
        additional_instruction="Accept variants like '2014–2015' (en dash), '2014 to 2015', or '2014/2015'."
    )

    leaf_lesmis_ref = evaluator.add_leaf(
        id="les_mis_reference_url",
        desc="Provide reference URL supporting Les Misérables role information",
        parent=lesmis_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms that {CANONICAL_ACTOR} played {LES_MIS_ROLE} in the Broadway production of Les Misérables during 2014–2015.",
        node=leaf_lesmis_ref,
        sources=les_mis_info.les_mis_urls,
        additional_instruction="The page should explicitly mention both the role (Gavroche) and that it was the Broadway production in the 2014–2015 period."
    )

    # Easter Bonnet performance
    eb_node = evaluator.add_parallel(
        id="easter_bonnet_performance",
        desc="Provide details about performance as youngest Jean Valjean in 2015 Easter Bonnet Competition with reference URL",
        parent=career_node,
        critical=False
    )

    leaf_eb_details = evaluator.add_leaf(
        id="performance_details",
        desc="Verify performance as youngest Jean Valjean in the 2015 Easter Bonnet Competition",
        parent=eb_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms that in {EASTER_BONNET_YEAR}, during the Broadway Easter Bonnet Competition, {CANONICAL_ACTOR} performed as the youngest Jean Valjean.",
        node=leaf_eb_details,
        sources=easter_bonnet_info.easter_bonnet_urls,
        additional_instruction="Look for explicit mention of 'youngest Jean Valjean' and the event being the Broadway Cares/Equity Fights AIDS Easter Bonnet Competition (2015)."
    )

    leaf_eb_ref = evaluator.add_leaf(
        id="easter_bonnet_reference_url",
        desc="Provide reference URL supporting Easter Bonnet performance information",
        parent=eb_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page serves as a valid reference confirming {CANONICAL_ACTOR}'s 2015 Easter Bonnet Competition appearance as the youngest Jean Valjean.",
        node=leaf_eb_ref,
        sources=easter_bonnet_info.easter_bonnet_urls,
        additional_instruction="A good source is acceptable if it clearly documents the claim; duplicates of the same source are fine."
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Enforce the ordered workflow
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

    # Parallelize extractions
    actor_task = evaluator.extract(
        prompt=prompt_extract_actor_identification(),
        template_class=ActorIdentification,
        extraction_name="actor_identification",
    )
    birth_task = evaluator.extract(
        prompt=prompt_extract_birth_info(),
        template_class=BirthInfo,
        extraction_name="birth_info",
    )
    twin_task = evaluator.extract(
        prompt=prompt_extract_twin_info(),
        template_class=TwinInfo,
        extraction_name="twin_info",
    )
    lesmis_task = evaluator.extract(
        prompt=prompt_extract_les_mis_info(),
        template_class=LesMisInfo,
        extraction_name="les_mis_info",
    )
    eb_task = evaluator.extract(
        prompt=prompt_extract_easter_bonnet_info(),
        template_class=EasterBonnetInfo,
        extraction_name="easter_bonnet_info",
    )

    actor_info, birth_info, twin_info, les_mis_info, easter_bonnet_info = await asyncio.gather(
        actor_task, birth_task, twin_task, lesmis_task, eb_task
    )

    # Optional: record ground truth for context
    evaluator.add_ground_truth(
        {
            "actor": CANONICAL_ACTOR,
            "role": CANONICAL_ROLE_HADESTOWN,
            "start_date": CANONICAL_START_DATE,
            "birth_date": CANONICAL_BIRTH_DATE,
            "allowed_birthplaces": ALLOWED_BIRTHPLACES,
            "twin_full_name": CANONICAL_TWIN_NAME,
            "les_mis_role": LES_MIS_ROLE,
            "les_mis_period": LES_MIS_PERIOD,
            "easter_bonnet_year": EASTER_BONNET_YEAR,
        },
        gt_type="canonical_facts",
    )

    # Build verification tree according to rubric (sequential at the top-level)
    # 1) Actor identification (critical gate)
    await build_actor_identification_checks(evaluator, root, actor_info)

    # 2) Biographical information (non-critical, parallel)
    await build_biographical_checks(evaluator, root, birth_info, twin_info)

    # 3) Career history (non-critical, parallel)
    await build_career_checks(evaluator, root, les_mis_info, easter_bonnet_info)

    return evaluator.get_summary()