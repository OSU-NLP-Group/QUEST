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
TASK_ID = "snowfall_actor_identification"
TASK_DESCRIPTION = """
Who is the British actor born on September 2, 1991, in Peckham, South East London, who starred as Franklin Saint in the FX series Snowfall that ran for six seasons from 2017 to 2023?
"""

# Optional ground truth reference (not enforced directly, but helpful for context)
GROUND_TRUTH = {
    "name": "Damson Idris",
    "birth_date": "September 2, 1991",
    "birthplace": "Peckham, South East London, England",
    "nationality": "British",
    "descent": "Nigerian",
    "occupation": "Actor",
    "role": "Franklin Saint",
    "series": "Snowfall",
    "series_run": "Six seasons from 2017 to 2023"
}

# --------------------------------------------------------------------------- #
# Data models for extracting information from the answer                      #
# --------------------------------------------------------------------------- #
class PersonInfo(BaseModel):
    """Structured extraction of the identified person and cited sources."""
    name: Optional[str] = None
    birthplace: Optional[str] = None
    birth_date: Optional[str] = None
    nationality: Optional[str] = None
    descent: Optional[str] = None
    occupation: Optional[str] = None
    role: Optional[str] = None
    series_run: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_person_info() -> str:
    return """
    Extract the identified person's information as presented in the answer. Return a JSON object with the following fields:
    - name: The full name of the identified person.
    - birthplace: The birthplace string as stated (e.g., "Peckham, South East London, England").
    - birth_date: The birth date string as stated (e.g., "September 2, 1991" or "2 September 1991").
    - nationality: The nationality as stated (e.g., "British").
    - descent: The descent/heritage as stated (e.g., "Nigerian descent").
    - occupation: The occupation as stated (e.g., "actor").
    - role: The role in Snowfall as stated (e.g., "Franklin Saint").
    - series_run: The series run description as stated (e.g., "six seasons from 2017 to 2023").
    - source_urls: An array of all URLs explicitly present in the answer that are relevant sources supporting the above facts.
    
    Special rules:
    - Only extract URLs actually present in the answer. Do not invent any URLs.
    - If a field is not mentioned, set it to null. If there are no URLs, return an empty array for source_urls.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_identified_person_verification(
        evaluator: Evaluator,
        root_node,
        person: PersonInfo
) -> None:
    """
    Build the verification subtree for the identified person and run verifications.
    """
    # Add the main critical node for the identified person
    identified_node = evaluator.add_parallel(
        id="Identified_Person",
        desc="The correct person is identified and satisfies all stated constraints in the question/constraints.",
        parent=root_node,
        critical=True
    )

    # Prepare sources (if none or empty, pass None to use simple verification)
    sources = person.source_urls if person.source_urls else None
    person_name = person.name or "the identified person"

    # 1) Birthplace Verification
    birthplace_leaf = evaluator.add_leaf(
        id="Birthplace_Verification",
        desc="The identified person was born in Peckham, South East London, UK.",
        parent=identified_node,
        critical=True
    )
    birthplace_claim = f"{person_name} was born in Peckham, South East London, England."
    await evaluator.verify(
        claim=birthplace_claim,
        node=birthplace_leaf,
        sources=sources,
        additional_instruction=(
            "Allow reasonable geographic variants such as 'Peckham, London, England' or 'Peckham, South London'. "
            "The key is that the birthplace clearly indicates Peckham in London, England."
        )
    )

    # 2) Birth Date Verification
    birthdate_leaf = evaluator.add_leaf(
        id="Birth_Date_Verification",
        desc="The identified person was born on September 2, 1991.",
        parent=identified_node,
        critical=True
    )
    birthdate_claim = f"{person_name} was born on September 2, 1991."
    await evaluator.verify(
        claim=birthdate_claim,
        node=birthdate_leaf,
        sources=sources,
        additional_instruction=(
            "Treat 'September 2, 1991' and '2 September 1991' as equivalent. "
            "Minor formatting differences are acceptable as long as the date matches."
        )
    )

    # 3) Nationality and Descent Verification
    nationality_leaf = evaluator.add_leaf(
        id="Nationality_Descent_Verification",
        desc="The identified person is British of Nigerian descent.",
        parent=identified_node,
        critical=True
    )
    nationality_claim = f"{person_name} is British and of Nigerian descent."
    await evaluator.verify(
        claim=nationality_claim,
        node=nationality_leaf,
        sources=sources,
        additional_instruction=(
            "This can be supported by statements like 'British actor of Nigerian descent', "
            "'British-Nigerian', or references to Nigerian heritage/parents."
        )
    )

    # 4) Occupation Verification
    occupation_leaf = evaluator.add_leaf(
        id="Occupation_Verification",
        desc="The identified person is an actor.",
        parent=identified_node,
        critical=True
    )
    occupation_claim = f"{person_name} is an actor."
    await evaluator.verify(
        claim=occupation_claim,
        node=occupation_leaf,
        sources=sources,
        additional_instruction=(
            "Accept equivalents like 'film and television actor' or 'actor and producer' as long as 'actor' is accurate."
        )
    )

    # 5) Role Verification
    role_leaf = evaluator.add_leaf(
        id="Role_Verification",
        desc="The identified person starred as Franklin Saint in the FX series Snowfall.",
        parent=identified_node,
        critical=True
    )
    role_claim = f"{person_name} starred as Franklin Saint in the FX series Snowfall."
    await evaluator.verify(
        claim=role_claim,
        node=role_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the person was the lead actor portraying Franklin Saint in 'Snowfall'. "
            "Cast listings, official bios, or reputable media sources are acceptable."
        )
    )

    # 6) Series Run Verification
    series_run_leaf = evaluator.add_leaf(
        id="Series_Run_Verification",
        desc="Snowfall ran for six seasons from 2017 to 2023.",
        parent=identified_node,
        critical=True
    )
    series_run_claim = "The FX series Snowfall ran for six seasons from 2017 to 2023."
    await evaluator.verify(
        claim=series_run_claim,
        node=series_run_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that Snowfall premiered in 2017 and concluded in 2023 after six seasons. "
            "Accept language like 'six seasons, ending in 2023' or 'premiered in 2017' with a final season in 2023."
        )
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
    Evaluate an answer to the Snowfall actor identification task.
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
        default_model=model,
    )

    # Extract structured person info from the answer
    person_info = await evaluator.extract(
        prompt=prompt_extract_person_info(),
        template_class=PersonInfo,
        extraction_name="identified_person_info",
    )

    # Add ground truth information for transparency (not used as a hard check)
    evaluator.add_ground_truth({
        "expected": GROUND_TRUTH
    }, gt_type="ground_truth")

    # Build verification tree for the identified person and run checks
    await build_identified_person_verification(evaluator, root, person_info)

    # Return evaluation summary
    return evaluator.get_summary()