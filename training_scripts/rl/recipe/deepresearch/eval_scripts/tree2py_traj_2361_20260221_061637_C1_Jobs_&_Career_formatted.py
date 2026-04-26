import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cwru_career_info_2024_fds"
TASK_DESCRIPTION = (
    "What knowledge rate did Case Western Reserve University's Center for Career Success achieve in the Class of 2024 First Destination Survey, "
    "and what is the contact phone number for the Center for Career Success?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CWRUCareerInfo(BaseModel):
    knowledge_rate: Optional[str] = None
    knowledge_rate_sources: List[str] = Field(default_factory=list)
    contact_phone: Optional[str] = None
    contact_phone_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cwru_info() -> str:
    return """
    Extract exactly the following fields from the provided answer text. Do not infer or invent anything.

    Fields to extract:
    1) knowledge_rate: The knowledge rate for Case Western Reserve University's Class of 2024 First Destination Survey (FDS), as written in the answer (e.g., "94%", "94.0%", "94 percent"). Extract it as a string exactly as presented.
    2) knowledge_rate_sources: An array of explicit URLs cited in the answer that directly support the knowledge rate for the Class of 2024 FDS. Include only URLs that are actually shown in the answer text (plain URLs or markdown links). If none are present, return an empty array.
    3) contact_phone: The contact phone number for Case Western Reserve University's Center for Career Success, as written in the answer. Keep the original formatting (e.g., "(216) 368-xxxx" or "216-368-xxxx").
    4) contact_phone_sources: An array of explicit URLs cited in the answer that directly support the Center for Career Success contact phone number. Include only URLs that are actually shown in the answer text. If none are present, return an empty array.

    Important rules:
    - Only extract values explicitly present in the answer text.
    - For URL fields, include only valid URLs that appear in the answer. If a URL lacks http/https, prepend http://.
    - If a field is not present in the answer, set it to null (for single fields) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: CWRUCareerInfo) -> None:
    """
    Build and execute the verification tree for the CWRU Career information task.
    """
    # Create a top-level grouping node to mirror the rubric. The rubric marks this as critical.
    cwru_info_node = evaluator.add_parallel(
        id="CWRU_Career_Information",
        desc="Provide the knowledge rate from CWRU's Class of 2024 First Destination Survey and the contact phone number for the Center for Career Success",
        parent=evaluator.root,
        critical=True  # As specified by rubric; this makes all children also need to be critical.
    )

    # ---------------------------- Knowledge Rate ---------------------------- #
    kr_group = evaluator.add_parallel(
        id="Knowledge_Rate",
        desc="The knowledge rate percentage achieved by the Class of 2024 First Destination Survey",
        parent=cwru_info_node,
        critical=True  # Child of a critical node must be critical
    )

    # Existence check for knowledge_rate
    kr_exists = evaluator.add_custom_node(
        result=bool(extracted.knowledge_rate and str(extracted.knowledge_rate).strip()),
        id="knowledge_rate_exists",
        desc="Knowledge rate is provided in the answer",
        parent=kr_group,
        critical=True
    )

    # Sources presence for knowledge_rate
    kr_sources_present = evaluator.add_custom_node(
        result=bool(extracted.knowledge_rate_sources),
        id="knowledge_rate_sources_present",
        desc="Knowledge rate has supporting URL sources provided in the answer",
        parent=kr_group,
        critical=True
    )

    # Verify knowledge_rate value is supported by the cited sources
    kr_supported = evaluator.add_leaf(
        id="knowledge_rate_supported_by_sources",
        desc="The reported Class of 2024 FDS knowledge rate is supported by the cited sources",
        parent=kr_group,
        critical=True
    )
    kr_value = extracted.knowledge_rate or ""
    kr_claim = (
        f"The knowledge rate reported for Case Western Reserve University's Class of 2024 First Destination Survey "
        f"is {kr_value}."
    )
    await evaluator.verify(
        claim=kr_claim,
        node=kr_supported,
        sources=extracted.knowledge_rate_sources,
        additional_instruction=(
            "Check the provided webpage(s) for explicit mentions of the 'knowledge rate' for the Class of 2024 "
            "First Destination Survey at Case Western Reserve University (CWRU). Treat minor formatting differences "
            "as equivalent (e.g., '94%' vs '94.0%'). Prioritize text that explicitly uses terms like 'knowledge rate', "
            "'knowledge/response rate', or 'First Destination Survey (FDS)'."
        ),
    )

    # ---------------------------- Contact Phone ----------------------------- #
    phone_group = evaluator.add_parallel(
        id="Contact_Phone",
        desc="The contact phone number for Case Western Reserve University's Center for Career Success",
        parent=cwru_info_node,
        critical=True  # Child of a critical node must be critical
    )

    # Existence check for contact_phone
    phone_exists = evaluator.add_custom_node(
        result=bool(extracted.contact_phone and str(extracted.contact_phone).strip()),
        id="contact_phone_exists",
        desc="Contact phone number is provided in the answer",
        parent=phone_group,
        critical=True
    )

    # Sources presence for contact_phone
    phone_sources_present = evaluator.add_custom_node(
        result=bool(extracted.contact_phone_sources),
        id="contact_phone_sources_present",
        desc="Contact phone number has supporting URL sources provided in the answer",
        parent=phone_group,
        critical=True
    )

    # Verify contact_phone value is supported by the cited sources
    phone_supported = evaluator.add_leaf(
        id="contact_phone_supported_by_sources",
        desc="The Center for Career Success contact phone number is supported by the cited sources",
        parent=phone_group,
        critical=True
    )
    phone_value = extracted.contact_phone or ""
    phone_claim = (
        f"The contact phone number for Case Western Reserve University's Center for Career Success is '{phone_value}'."
    )
    await evaluator.verify(
        claim=phone_claim,
        node=phone_supported,
        sources=extracted.contact_phone_sources,
        additional_instruction=(
            "Verify that the cited page shows the phone number for the Center for Career Success at Case Western "
            "Reserve University. Accept reasonable formatting variations (e.g., '(216) 368-xxxx' vs '216-368-xxxx' or "
            "with/without spaces). Ensure the number belongs to the Center for Career Success specifically."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the CWRU Career Success knowledge rate and contact phone question.
    """
    # Initialize evaluator with a parallel root (two independent items)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_cwru_info(),
        template_class=CWRUCareerInfo,
        extraction_name="cwru_career_info",
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, extracted_info)

    # Return evaluation summary
    return evaluator.get_summary()