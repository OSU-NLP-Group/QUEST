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
TASK_ID = "nas_rl_iclr_cvpr_followup"
TASK_DESCRIPTION = (
    "Identify the first author of the paper 'Neural Architecture Search with Reinforcement Learning' presented at ICLR 2017. "
    "Then, find the co-author on this paper who is affiliated with Google Brain. Verify whether these two researchers subsequently "
    "co-authored a follow-up paper on transferable architectures for image recognition at CVPR 2018. If such a paper exists, "
    "provide its complete title and the full list of co-authors."
)

# Ground truth metadata (for reference in summary)
GROUND_TRUTH = {
    "iclr2017_expected_first_author": "Barret Zoph",
    "iclr2017_expected_google_brain_coauthor": "Quoc V. Le",
    "cvpr2018_followup_expected_title": "Learning Transferable Architectures for Scalable Image Recognition",
    "cvpr2018_followup_expected_authors": [
        "Barret Zoph",
        "Vijay Vasudevan",
        "Jonathon Shlens",
        "Quoc V. Le",
    ],
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ICLR2017Info(BaseModel):
    first_author: Optional[str] = None
    google_brain_coauthor: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CVPR2018FollowupInfo(BaseModel):
    followup_exists: Optional[bool] = None
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_iclr2017() -> str:
    return (
        "From the provided answer, locate the information specifically about the ICLR 2017 paper "
        "'Neural Architecture Search with Reinforcement Learning'. Extract the following fields:\n"
        "1) first_author: the first author named for this ICLR 2017 paper in the answer (return null if missing).\n"
        "2) google_brain_coauthor: the co-author identified in the answer as affiliated with Google Brain (return null if missing). "
        "If multiple co-authors are said to be Google Brain-affiliated, choose the one explicitly tied to the target ICLR 2017 paper.\n"
        "3) sources: an array of all URLs the answer cites that correspond to this ICLR 2017 paper (e.g., OpenReview, arXiv, Google Scholar, "
        "conference pages). Extract only URLs explicitly present in the answer. If none are provided, return an empty array."
    )


def prompt_extract_cvpr2018_followup() -> str:
    return (
        "From the provided answer, determine whether it claims that a follow-up CVPR 2018 paper (on transferable architectures for image recognition) exists. "
        "Then extract the details the answer provides for that follow-up paper.\n"
        "Return the following fields:\n"
        "1) followup_exists: a boolean indicating whether the answer explicitly states that such a follow-up paper exists (true/false; null if unclear).\n"
        "2) title: the complete title of the follow-up CVPR 2018 paper as given by the answer (null if missing).\n"
        "3) authors: an array of the full author list for the follow-up paper as given by the answer, in the order presented (empty array if missing).\n"
        "4) sources: an array of all URLs the answer cites for this follow-up CVPR 2018 paper (official CVPR proceedings, arXiv, IEEE, etc.). "
        "Extract only URLs explicitly present in the answer. If none are provided, return an empty array."
    )


# --------------------------------------------------------------------------- #
# Verification helper functions                                               #
# --------------------------------------------------------------------------- #
async def verify_iclr2017_first_author(
    evaluator: Evaluator,
    parent_node,
    iclr: ICLR2017Info,
) -> None:
    """
    Subtree for ICLR 2017 first author verification.
    """
    iclr_first_author_node = evaluator.add_parallel(
        id="ICLR2017_First_Author",
        desc="States the first author of 'Neural Architecture Search with Reinforcement Learning' (ICLR 2017) as specified in the constraints.",
        parent=parent_node,
        critical=True,
    )

    leaf_first_author = evaluator.add_leaf(
        id="First_Author_Name_Correct",
        desc="First author is Barret Zoph.",
        parent=iclr_first_author_node,
        critical=True,
    )

    provided_name = iclr.first_author or ""
    claim = f"The name '{provided_name}' and 'Barret Zoph' refer to the same person."
    await evaluator.verify(
        claim=claim,
        node=leaf_first_author,
        additional_instruction=(
            "Judge only whether the answer's extracted first author matches the expected 'Barret Zoph'. "
            "Allow minor or reasonable variants (e.g., casing, punctuation, or middle initials). "
            "Do not require web evidence for this equality check."
        ),
    )


async def verify_iclr2017_google_brain_coauthor(
    evaluator: Evaluator,
    parent_node,
    iclr: ICLR2017Info,
) -> None:
    """
    Subtree for ICLR 2017 Google Brain-affiliated coauthor verification.
    """
    iclr_gb_node = evaluator.add_parallel(
        id="ICLR2017_GoogleBrain_Affiliated_Coauthor",
        desc="Identifies the co-author of the same ICLR 2017 paper who is affiliated with Google Brain, as specified in the constraints.",
        parent=parent_node,
        critical=True,
    )

    leaf_gb_coauthor = evaluator.add_leaf(
        id="GoogleBrain_Coauthor_Correct",
        desc="Google Brain-affiliated co-author is Quoc V. Le.",
        parent=iclr_gb_node,
        critical=True,
    )

    provided_name = iclr.google_brain_coauthor or ""
    claim = f"The name '{provided_name}' and 'Quoc V. Le' refer to the same person."
    await evaluator.verify(
        claim=claim,
        node=leaf_gb_coauthor,
        additional_instruction=(
            "Judge only whether the answer's identified Google Brain-affiliated coauthor matches the expected 'Quoc V. Le'. "
            "Allow minor or reasonable variants (e.g., casing, punctuation, or middle initials). "
            "Do not require web evidence for this equality check."
        ),
    )


async def verify_cvpr2018_followup(
    evaluator: Evaluator,
    parent_node,
    followup: CVPR2018FollowupInfo,
) -> None:
    """
    Subtree for CVPR 2018 follow-up existence, title, and author list checks.
    """
    followup_node = evaluator.add_sequential(
        id="CVPR2018_Followup_Paper_Check_And_Report",
        desc="Correctly determines whether the specified follow-up CVPR 2018 paper exists, and if it exists provides its complete title and full author list as specified in the constraints.",
        parent=parent_node,
        critical=True,
    )

    # 1) Existence verdict (critical leaf)
    existence_leaf = evaluator.add_leaf(
        id="Followup_Existence_Verdict_Correct",
        desc="Correctly indicates that the follow-up paper exists (per constraints).",
        parent=followup_node,
        critical=True,
    )
    # We prefer evidence-based verification when URLs are provided.
    existence_claim = (
        "There exists a CVPR 2018 paper on transferable architectures for image recognition co-authored by Barret Zoph and Quoc V. Le."
    )
    await evaluator.verify(
        claim=existence_claim,
        node=existence_leaf,
        sources=followup.sources if followup and followup.sources else None,
        additional_instruction=(
            "Mark as supported only if at least one provided URL is an official source (e.g., CVPR proceedings, arXiv, IEEE) "
            "that clearly shows the paper exists and is co-authored by Barret Zoph and Quoc V. Le. "
            "If no URLs are provided, do not rely on your own knowledge; treat the claim as not supported."
        ),
    )

    # 2) Title correctness (critical leaf)
    title_leaf = evaluator.add_leaf(
        id="Followup_Title_Correct",
        desc="Provides the complete follow-up paper title: 'Learning Transferable Architectures for Scalable Image Recognition'.",
        parent=followup_node,
        critical=True,
    )
    provided_title = (followup.title or "").strip()
    expected_title = GROUND_TRUTH["cvpr2018_followup_expected_title"]
    title_claim = (
        f"The title provided in the answer ('{provided_title}') matches exactly the expected title: '{expected_title}'."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=followup.sources if followup and followup.sources else None,
        additional_instruction=(
            "Be strict about the title equality (case-insensitive and minor punctuation differences are acceptable), "
            "but do not accept missing words or different titles. Confirm against the provided URLs if available."
        ),
    )

    # 3) Full author list correctness (critical leaf)
    author_list_leaf = evaluator.add_leaf(
        id="Followup_Full_Author_List_Correct",
        desc="Provides the full author list exactly as specified: Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le.",
        parent=followup_node,
        critical=True,
    )
    provided_authors_str = ", ".join(followup.authors or [])
    expected_authors_str = ", ".join(GROUND_TRUTH["cvpr2018_followup_expected_authors"])
    author_claim = (
        f"The full author list provided in the answer ('{provided_authors_str}') matches exactly: '{expected_authors_str}'."
    )
    await evaluator.verify(
        claim=author_claim,
        node=author_list_leaf,
        sources=followup.sources if followup and followup.sources else None,
        additional_instruction=(
            "Be strict about matching the complete set and order of authors: Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le. "
            "Allow minor formatting or casing differences and middle initials, but do not allow missing or extra authors or reordering. "
            "Confirm against the provided URLs if available."
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
) -> Dict[str, Any]:
    """
    Evaluate an agent's answer for the NAS RL ICLR 2017 and CVPR 2018 follow-up task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root combines its single critical child
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

    # Record ground truth reference
    evaluator.add_ground_truth(
        {
            "expected_iclr2017_first_author": GROUND_TRUTH["iclr2017_expected_first_author"],
            "expected_iclr2017_google_brain_coauthor": GROUND_TRUTH["iclr2017_expected_google_brain_coauthor"],
            "expected_cvpr2018_title": GROUND_TRUTH["cvpr2018_followup_expected_title"],
            "expected_cvpr2018_authors": GROUND_TRUTH["cvpr2018_followup_expected_authors"],
        },
        gt_type="ground_truth",
    )

    # Extract required structured information from the answer (can be done concurrently)
    iclr_task = evaluator.extract(
        prompt=prompt_extract_iclr2017(),
        template_class=ICLR2017Info,
        extraction_name="iclr2017_info",
    )
    followup_task = evaluator.extract(
        prompt=prompt_extract_cvpr2018_followup(),
        template_class=CVPR2018FollowupInfo,
        extraction_name="cvpr2018_followup_info",
    )
    iclr_info, followup_info = await asyncio.gather(iclr_task, followup_task)

    # Build the critical, sequential task completion node
    task_node = evaluator.add_sequential(
        id="Task_Completion",
        desc=(
            "Provide the first author of the specified ICLR 2017 paper, identify the co-author affiliated with Google Brain, "
            "and verify/report the CVPR 2018 follow-up paper (title and full author list) if it exists."
        ),
        parent=root,
        critical=True,
    )

    # Subtrees according to rubric
    await verify_iclr2017_first_author(evaluator, task_node, iclr_info)
    await verify_iclr2017_google_brain_coauthor(evaluator, task_node, iclr_info)
    await verify_cvpr2018_followup(evaluator, task_node, followup_info)

    # Return final structured summary
    return evaluator.get_summary()