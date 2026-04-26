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
TASK_ID = "stanford_hai_director_2024"
TASK_DESCRIPTION = (
    "Who is a current co-director of Stanford HAI (Stanford Institute for Human-Centered Artificial Intelligence) "
    "as of 2024? Provide the person's name and an official Stanford University or Stanford HAI website URL that verifies their position."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LeadershipExtraction(BaseModel):
    """
    Structured extraction for the Stanford HAI leadership identification task.
    """
    person_name: Optional[str] = None
    institute_name: Optional[str] = None
    role_title: Optional[str] = None
    evidence_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_leadership() -> str:
    return (
        "Extract the following from the provided answer text:\n"
        "1. person_name: The full name of the individual identified as a Stanford HAI Director or Co-Director.\n"
        "2. institute_name: The institute named in the answer (e.g., 'Stanford HAI' or 'Stanford Institute for Human-Centered Artificial Intelligence').\n"
        "3. role_title: The leadership role stated for the person (e.g., 'Director', 'Co-Director', 'Interim Director').\n"
        "4. evidence_urls: All webpage URLs that the answer cites to verify the leadership role. Include every URL explicitly present in the answer text, "
        "   such as plain URLs or markdown links. Do not invent or infer any URLs.\n"
        "If any field is not explicitly present, return null for that field (or an empty list for evidence_urls)."
    )


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    extracted: LeadershipExtraction,
) -> None:
    """
    Build and execute the verification tree for the Stanford HAI leadership identification.
    The top-level node is critical (must pass), and all its children are critical checks.
    """
    top_node = evaluator.add_parallel(
        id="Stanford_HAI_Leadership_Identification",
        desc="Identify a current Stanford HAI Director or Co-Director as of 2024 and provide official Stanford/Stanford HAI web evidence.",
        parent=root_node,
        critical=True,
    )

    # 1) Person name provided (existence check)
    name_present = bool(extracted.person_name and extracted.person_name.strip())
    evaluator.add_custom_node(
        result=name_present,
        id="Person_Name_Provided",
        desc="The answer provides the person's name (a specific individual).",
        parent=top_node,
        critical=True,
    )

    # 2) Correct institute: Stanford HAI
    institute_node = evaluator.add_leaf(
        id="Correct_Institute_Stanford_HAI",
        desc="The answer identifies the relevant institute as Stanford HAI (Stanford Institute for Human-Centered Artificial Intelligence) at Stanford University.",
        parent=top_node,
        critical=True,
    )
    institute_claim = (
        "The answer text clearly identifies the institute as Stanford HAI "
        "(Stanford Institute for Human-Centered Artificial Intelligence) at Stanford University."
    )
    await evaluator.verify(
        claim=institute_claim,
        node=institute_node,
        sources=None,
        additional_instruction=(
            "Judge based solely on the answer text. Accept if the answer mentions 'Stanford HAI' or "
            "'Stanford Institute for Human-Centered Artificial Intelligence' (allow minor wording variations)."
        ),
    )

    # 3) Official URL evidence provided and valid (at least one URL is official and indicates the role)
    official_url_node = evaluator.add_leaf(
        id="Official_URL_Evidence_Provided_and_Valid",
        desc="At least one official Stanford University or Stanford HAI webpage URL is included and explicitly indicates the person's Director/Co-Director role at Stanford HAI.",
        parent=top_node,
        critical=True,
    )
    official_url_claim = (
        f"At least one of the provided URLs is an official Stanford University or Stanford HAI webpage "
        f"that explicitly indicates that {extracted.person_name or 'the named person'} is a Director or Co-Director of Stanford HAI."
    )
    await evaluator.verify(
        claim=official_url_claim,
        node=official_url_node,
        sources=extracted.evidence_urls if extracted.evidence_urls else None,
        additional_instruction=(
            "Only accept URLs hosted on the official Stanford domain (*.stanford.edu), which includes hai.stanford.edu. "
            "Reject third‑party domains, news outlets, or social media. The page must explicitly indicate the person's "
            "Director or Co‑Director role at Stanford HAI. If the answer provides no URLs, judge this as incorrect."
        ),
    )

    # 4) Correct leadership role (Director/Co-Director at Stanford HAI)
    role_node = evaluator.add_leaf(
        id="Correct_Leadership_Role",
        desc="The identified person holds an official Director or Co-Director position at Stanford HAI.",
        parent=top_node,
        critical=True,
    )
    role_claim = (
        f"The provided official webpage(s) indicate that {extracted.person_name or 'the named person'} "
        f"holds an official Director or Co-Director position at Stanford HAI."
    )
    await evaluator.verify(
        claim=role_claim,
        node=role_node,
        sources=extracted.evidence_urls if extracted.evidence_urls else None,
        additional_instruction=(
            "Verify on the official Stanford/HAI page(s) that the person is listed with the title "
            "'Director' or 'Co‑Director' (allow minor variants like 'Co‑Directors' list entries). "
            "Do not accept unrelated roles (e.g., advisory board or affiliate) and do not accept "
            "third‑party sources. If no official URL is available, judge as unsupported."
        ),
    )

    # 5) Current as of 2024 (not solely historical)
    current_node = evaluator.add_leaf(
        id="Current_as_of_2024",
        desc="The leadership position is described as current/active as of 2024 (not solely historical).",
        parent=top_node,
        critical=True,
    )
    current_claim = (
        f"As of 2024, {extracted.person_name or 'the named person'} is currently serving as a Director or Co‑Director of Stanford HAI "
        f"(i.e., the role is active/current, not solely a past role)."
    )
    await evaluator.verify(
        claim=current_claim,
        node=current_node,
        sources=extracted.evidence_urls if extracted.evidence_urls else None,
        additional_instruction=(
            "Prefer official leadership/people pages showing current roster. If the page appears to be historical news with no indication "
            "of the current status as of 2024, judge as unsupported. Allow reasonable inference from an official current leadership page "
            "even if the page itself lacks an explicit 2024 timestamp."
        ),
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
    """
    Evaluate an answer for identifying a current Stanford HAI co-director (as of 2024) with official evidence.
    """
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_leadership(),
        template_class=LeadershipExtraction,
        extraction_name="leadership_extraction",
    )

    # Build and execute verification tree
    await build_verification_tree(evaluator, root, extracted)

    # Return summary with verification tree and scoring
    return evaluator.get_summary()