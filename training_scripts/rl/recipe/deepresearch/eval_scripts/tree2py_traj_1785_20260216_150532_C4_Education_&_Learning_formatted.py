import asyncio
import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_foundation_hsp_grad_reqs"
TASK_DESCRIPTION = "What are the minimum credit, subject-specific credit, examination, and grade requirements that students must meet to graduate from a Texas public high school under the Foundation High School Program?"

# Ground truth (for reporting context only; verification relies on cited sources)
EXPECTED_REQUIREMENTS = {
    "total_credits": "Minimum of 22 high school credits",
    "english_credits": "4 English credits, including English I and English II",
    "math_credits": "Minimum of 3 mathematics credits",
    "science_credits": "Minimum of 3 science credits",
    "social_studies_credits": "Minimum of 3 social studies credits",
    "eoc_exams": "Pass EOC exams in Algebra I, Biology, English I, English II, and U.S. History",
    "passing_grade": "Receive a grade of 70% or above to earn credit for each course"
}


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class AnswerSources(BaseModel):
    """
    Extract all source URLs that the answer cites. Only include valid URLs.
    """
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_sources() -> str:
    return """
    Extract all URLs explicitly mentioned in the answer text. Include:
    - Plain URLs (http:// or https://)
    - URLs embedded in markdown links
    Rules:
    - Return each unique URL once.
    - Only include valid and complete URLs (with http:// or https://).
    - If no URLs are present, return an empty array.
    
    Output fields:
    - source_urls: array of strings, each a full URL.
    """


# --------------------------------------------------------------------------- #
# Helper Functions for Verification                                           #
# --------------------------------------------------------------------------- #
async def add_requirement_checks(
    evaluator: Evaluator,
    parent_node,
    base_id: str,
    stated_claim: str,
    supported_claim: str,
    sources: List[str],
    stated_instruction: Optional[str] = None,
    supported_instruction: Optional[str] = None
) -> None:
    """
    Build a sequential node that checks:
      1) The answer explicitly states the requirement (simple verification against the answer text).
      2) The requirement is supported by the cited sources (URL-based verification).
    Both sub-checks are critical. If the first fails, the second will be skipped automatically.
    """
    req_node = evaluator.add_sequential(
        id=base_id,
        desc=stated_claim.replace("The answer explicitly states that ", "").strip(),
        parent=parent_node,
        critical=True
    )

    # Leaf A: Answer explicitly states the requirement (critical)
    stated_leaf = evaluator.add_leaf(
        id=f"{base_id}_stated",
        desc=f"Answer states requirement: {stated_claim}",
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim=stated_claim,
        node=stated_leaf,
        additional_instruction=stated_instruction or "Look for explicit or equivalent phrasing in the answer; minor wording differences are acceptable."
    )

    # Leaf B: Requirement is supported by cited sources (critical)
    if sources and len(sources) > 0:
        supported_leaf = evaluator.add_leaf(
            id=f"{base_id}_supported",
            desc=f"Requirement supported by cited sources: {supported_claim}",
            parent=req_node,
            critical=True
        )
        await evaluator.verify(
            claim=supported_claim,
            node=supported_leaf,
            sources=sources,
            additional_instruction=supported_instruction or "Verify that at least one cited source explicitly supports this requirement."
        )
    else:
        # No sources were provided; treat as unsupported (fail this critical leaf)
        evaluator.add_custom_node(
            result=False,
            id=f"{base_id}_supported_no_sources",
            desc="Requirement support check failed: no sources were provided in the answer to verify this requirement.",
            parent=req_node,
            critical=True
        )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for Texas Foundation HSP graduation requirements.
    The evaluation checks that the answer both states each minimum requirement and that those requirements
    are supported by the answer's cited sources.
    """
    # Initialize evaluator and root
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

    # Extract all source URLs from the answer
    sources_extraction = await evaluator.extract(
        prompt=prompt_extract_sources(),
        template_class=AnswerSources,
        extraction_name="answer_sources"
    )
    all_sources = list(dict.fromkeys(sources_extraction.source_urls)) if sources_extraction and sources_extraction.source_urls else []

    # Record ground truth context for transparency
    evaluator.add_ground_truth(
        {
            "expected_requirements": EXPECTED_REQUIREMENTS,
            "note": "These expected requirements are provided for context in the evaluation summary. Actual verification checks rely on the answer text and its cited sources."
        }
    )

    # Create a main parallel node mirroring the rubric's top-level requirement set (critical)
    main_node = evaluator.add_parallel(
        id="Texas_Foundation_HSP_Graduation_Requirements",
        desc="States all minimum graduation requirements under the Texas Foundation High School Program as specified in the constraints.",
        parent=root,
        critical=True
    )

    # 1) Total Credits Requirement
    await add_requirement_checks(
        evaluator=evaluator,
        parent_node=main_node,
        base_id="Total_Credits_Requirement",
        stated_claim="The answer explicitly states that students must earn a minimum of 22 high school credits to graduate under the Texas Foundation High School Program.",
        supported_claim="Under the Texas Foundation High School Program, students must earn a minimum of 22 high school credits to graduate.",
        sources=all_sources,
        stated_instruction="Accept phrasing such as 'minimum of 22 credits', 'at least 22 credits', or '22 total credits'. If the answer only mentions another number (e.g., 26) without stating the 22-credit minimum, this should be considered not stated.",
        supported_instruction="Prefer official TEA or Texas Administrative Code sources; if multiple URLs are provided, any that clearly state the 22-credit minimum is acceptable."
    )

    # 2) English Credits Requirement
    await add_requirement_checks(
        evaluator=evaluator,
        parent_node=main_node,
        base_id="English_Credits_Requirement",
        stated_claim="The answer explicitly states that students must earn 4 English credits and specifically includes English I and English II.",
        supported_claim="Under the Texas Foundation High School Program, students must complete 4 English Language Arts credits, which include English I and English II.",
        sources=all_sources,
        stated_instruction="Allow 'English 1' and 'English 2' as equivalent to 'English I' and 'English II'. The answer must mention both English I and English II by name.",
        supported_instruction="Accept pages that list 'English I, English II, and two additional English courses' or equivalent language."
    )

    # 3) Mathematics Credits Requirement
    await add_requirement_checks(
        evaluator=evaluator,
        parent_node=main_node,
        base_id="Mathematics_Credits_Requirement",
        stated_claim="The answer explicitly states that students must earn a minimum of 3 mathematics credits.",
        supported_claim="Under the Texas Foundation High School Program, students must complete at least 3 mathematics credits.",
        sources=all_sources,
        stated_instruction="Accept synonyms such as 'math credits' for 'mathematics credits'.",
        supported_instruction="Accept pages that enumerate Algebra I, Geometry, and an additional math credit (or equivalent phrasing) as satisfying the 3-credit minimum."
    )

    # 4) Science Credits Requirement
    await add_requirement_checks(
        evaluator=evaluator,
        parent_node=main_node,
        base_id="Science_Credits_Requirement",
        stated_claim="The answer explicitly states that students must earn a minimum of 3 science credits.",
        supported_claim="Under the Texas Foundation High School Program, students must complete at least 3 science credits.",
        sources=all_sources,
        stated_instruction="Minor wording differences are acceptable as long as the 3-credit minimum for science is clearly stated.",
        supported_instruction="Accept pages that include Biology and additional science credits, as long as the 3-credit minimum is explicit."
    )

    # 5) Social Studies Credits Requirement
    await add_requirement_checks(
        evaluator=evaluator,
        parent_node=main_node,
        base_id="Social_Studies_Credits_Requirement",
        stated_claim="The answer explicitly states that students must earn a minimum of 3 social studies credits.",
        supported_claim="Under the Texas Foundation High School Program, students must complete at least 3 social studies credits.",
        sources=all_sources,
        stated_instruction="Accept 'social studies' phrased in full; minor variations in wording are acceptable if the 3-credit minimum is clear.",
        supported_instruction="Accept pages that list U.S. History, U.S. Government (0.5), Economics (0.5), and World Geography/History or similar, as long as the 3-credit minimum is explicit."
    )

    # 6) EOC Exams Requirement
    await add_requirement_checks(
        evaluator=evaluator,
        parent_node=main_node,
        base_id="EOC_Exam_Requirement",
        stated_claim="The answer explicitly states that students must pass the EOC exams in Algebra I, Biology, English I, English II, and U.S. History.",
        supported_claim="Under the Texas Foundation High School Program, students must pass the EOC exams in Algebra I, Biology, English I, English II, and U.S. History to graduate.",
        sources=all_sources,
        stated_instruction="Accept 'End-of-Course' spelled out or abbreviated as EOC; allow minor formatting variations in subject names (e.g., 'U.S. History' vs 'US History').",
        supported_instruction="Prefer official TEA assessment pages or Texas Administrative Code references that list these EOCs as graduation requirements."
    )

    # 7) Passing Grade Requirement
    await add_requirement_checks(
        evaluator=evaluator,
        parent_node=main_node,
        base_id="Passing_Grade_Requirement",
        stated_claim="The answer explicitly states that students must receive a grade of 70% or above to earn credit for each course.",
        supported_claim="To earn credit for a high school course in Texas, a student must receive a grade of 70% or above.",
        sources=all_sources,
        stated_instruction="Accept phrasing like '70 or higher', '70 out of 100', or 'a minimum grade of 70' as equivalent.",
        supported_instruction="Accept Texas Education Code or TEA/district policy pages stating a 70 or above is required to earn course credit."
    )

    # Add custom info about sources for transparency
    evaluator.add_custom_info(
        info={"total_extracted_sources": len(all_sources), "sources": all_sources},
        info_type="sources_overview"
    )

    # Return summary
    return evaluator.get_summary()