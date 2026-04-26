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
TASK_ID = "mega_shark_discovery"
TASK_DESCRIPTION = (
    "Identify the recent mega-shark fossil discovery near Darwin, Australia, that was coordinated by the Swedish Museum "
    "of Natural History and published in the journal Communications Biology. Provide the type of fossils found, their "
    "age, and the estimated size of the shark."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DiscoveryInfo(BaseModel):
    """
    Structured information extracted from the agent's answer about the mega-shark fossil discovery.
    """
    study_title_or_name: Optional[str] = None
    location_statement: Optional[str] = None
    journal_name: Optional[str] = None
    coordinator_org: Optional[str] = None
    fossil_type: Optional[str] = None
    fossil_age: Optional[str] = None
    shark_size_estimate: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_discovery_info() -> str:
    return """
    Extract the key details about the specific mega-shark fossil discovery described in the answer. 
    Return a JSON object with the following fields (use null for missing fields):

    - study_title_or_name: The title or name of the discovery/study if mentioned explicitly.
    - location_statement: The location phrase describing where the fossils were found (e.g., "near Darwin in northern Australia", "near Darwin, NT, Australia").
    - journal_name: The journal stated as the place of publication (e.g., "Communications Biology").
    - coordinator_org: The coordinating institution/organization (e.g., "Swedish Museum of Natural History").
    - fossil_type: The type of fossils found (e.g., "teeth", "vertebrae", "cartilage", "coprolites", etc., as written in the answer).
    - fossil_age: The age of the fossils or age range/period (e.g., "about 11 million years", "Miocene (11–12 Ma)", etc., verbatim from the answer).
    - shark_size_estimate: The estimated size/length of the shark (e.g., "about 7 meters", "up to ~10 m"), exactly as presented in the answer.
    - source_urls: An array of all URLs explicitly included in the answer (including hyperlinks shown in markdown). 
      Only include valid URLs; if none are provided, return an empty array.

    Important:
    - Do not invent information. Extract exactly what is written in the answer.
    - For URLs, extract the actual URL targets; include full protocol (http/https). If protocol is missing, prepend http://.
    - If multiple values are mentioned for a field, choose the most specific or definitive one presented.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_discovery_identification_checks(
    evaluator: Evaluator,
    parent_node,
    info: DiscoveryInfo
) -> None:
    """
    Build and run verification checks for identifying the specific discovery/study:
    - Near Darwin in northern Australia
    - Published in Communications Biology
    - Coordinated by the Swedish Museum of Natural History
    """

    discovery_node = evaluator.add_parallel(
        id="Discovery_Identification",
        desc="Identifies the specific discovery/study that matches the given constraints.",
        parent=parent_node,
        critical=True
    )

    urls = info.source_urls or []

    # 1) Location near Darwin (Northern Australia)
    loc_leaf = evaluator.add_leaf(
        id="Located_Near_Darwin_Northern_Australia",
        desc="States or clearly indicates the fossil discovery location is near Darwin in northern Australia.",
        parent=discovery_node,
        critical=True
    )
    loc_claim = (
        "The fossil discovery took place near Darwin in northern Australia (i.e., in the Darwin region of the Northern Territory)."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=urls,
        additional_instruction=(
            "Support the claim if any provided webpage explicitly mentions the fossils were found near Darwin, "
            "in or around the Darwin region, Darwin Harbour, or in the Northern Territory near Darwin. "
            "Equivalent phrasings like 'near Darwin (NT)' or 'in the Darwin area' count as near Darwin."
        ),
    )

    # 2) Published in Communications Biology
    journal_leaf = evaluator.add_leaf(
        id="Published_in_Communications_Biology",
        desc="States or clearly indicates the research findings were published in the journal Communications Biology.",
        parent=discovery_node,
        critical=True
    )
    journal_claim = "The research findings were published in the journal Communications Biology."
    await evaluator.verify(
        claim=journal_claim,
        node=journal_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm that the page(s) explicitly reference 'Communications Biology' (a Nature Portfolio journal). "
            "Minor variations such as capitalization or abbreviated forms are acceptable if unambiguously referring to the same journal. "
            "Do not accept 'Nature Communications' or other journals as equivalent."
        ),
    )

    # 3) Coordinated by Swedish Museum of Natural History
    coord_leaf = evaluator.add_leaf(
        id="Coordinated_by_Swedish_Museum_of_Natural_History",
        desc="States or clearly indicates the study was coordinated by the Swedish Museum of Natural History.",
        parent=discovery_node,
        critical=True
    )
    coord_claim = "The study was coordinated by the Swedish Museum of Natural History."
    await evaluator.verify(
        claim=coord_claim,
        node=coord_leaf,
        sources=urls,
        additional_instruction=(
            "Accept equivalent phrasings that clearly indicate coordination or leadership by the Swedish Museum of Natural History "
            "(also known as Naturhistoriska riksmuseet, located in Stockholm, Sweden)."
        ),
    )


async def build_required_information_checks(
    evaluator: Evaluator,
    parent_node,
    info: DiscoveryInfo
) -> None:
    """
    Build presence checks for the required information:
    - Fossil type provided
    - Fossil age provided
    - Shark size estimate provided

    These are critical presence checks per the rubric.
    """
    required_node = evaluator.add_parallel(
        id="Required_Information",
        desc="Provides the required information about the discovery.",
        parent=parent_node,
        critical=True
    )

    # Fossil type presence
    fossil_type_present = bool(info.fossil_type and info.fossil_type.strip())
    evaluator.add_custom_node(
        result=fossil_type_present,
        id="Fossil_Type_Provided",
        desc="Specifies the type of fossils found (e.g., what anatomical elements/material).",
        parent=required_node,
        critical=True
    )

    # Fossil age presence
    fossil_age_present = bool(info.fossil_age and info.fossil_age.strip())
    evaluator.add_custom_node(
        result=fossil_age_present,
        id="Fossil_Age_Provided",
        desc="Provides the age (or age range/period) of the fossils.",
        parent=required_node,
        critical=True
    )

    # Shark size estimate presence
    shark_size_present = bool(info.shark_size_estimate and info.shark_size_estimate.strip())
    evaluator.add_custom_node(
        result=shark_size_present,
        id="Shark_Size_Estimate_Provided",
        desc="Provides the estimated size of the shark (e.g., length/size estimate).",
        parent=required_node,
        critical=True
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
    Evaluate an answer for the Mega Shark Discovery task using the Mind2Web2 evaluation framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root container; actual critical logic in child node
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
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_discovery_info(),
        template_class=DiscoveryInfo,
        extraction_name="discovery_info"
    )

    # Build the rubric tree under a critical sequential node matching the rubric root
    mega_task_node = evaluator.add_sequential(
        id="Mega_Shark_Discovery_Task",
        desc="Correctly identifies the specified mega-shark fossil discovery and provides the required details.",
        parent=root,
        critical=True
    )

    # Part 1: Discovery Identification (critical, parallel)
    await build_discovery_identification_checks(evaluator, mega_task_node, extracted_info)

    # Part 2: Required Information (critical, parallel)
    await build_required_information_checks(evaluator, mega_task_node, extracted_info)

    # Return the evaluation summary
    return evaluator.get_summary()