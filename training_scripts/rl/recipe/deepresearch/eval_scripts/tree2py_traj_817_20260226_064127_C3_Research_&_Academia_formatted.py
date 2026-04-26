import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nature_index_2025_us_top10"
TASK_DESCRIPTION = (
    "Identify the highest-ranked U.S. research institution that appears in the global top 10 of the Nature Index 2025 "
    "Research Leaders ranking. Provide the following information: (1) the institution's name, (2) its exact global "
    "ranking position in the Nature Index 2025 Research Leaders, (3) the number of highly cited researchers from this "
    "institution according to the Clarivate Highly Cited Researchers 2025 list (released November 12, 2025), and (4) "
    "URL references from credible sources supporting your answer for both the Nature Index ranking and the highly cited "
    "researchers information."
)

EXPECTED_INSTITUTION = "Harvard University"
EXPECTED_RANK = "2"  # Accept variants like "#2", "2nd", "rank 2"
EXPECTED_HCR_COUNT = "170"
EXPECTED_HCR_RELEASE_DATE = "November 12, 2025"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class InstitutionExtraction(BaseModel):
    institution_name: Optional[str] = None
    ranking_position: Optional[str] = None
    hcr_count: Optional[str] = None
    ranking_source_urls: List[str] = Field(default_factory=list)
    hcr_source_urls: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_institution_info() -> str:
    return (
        "From the provided answer, extract the following fields as they explicitly appear:\n"
        "1) institution_name: The name of the highest-ranked U.S. institution mentioned.\n"
        "2) ranking_position: The exact global ranking position the answer states for this institution in the Nature Index 2025 Research Leaders (e.g., '#2', '2', '2nd').\n"
        "3) hcr_count: The number of Highly Cited Researchers (HCR) attributed to this institution according to Clarivate's Highly Cited Researchers 2025 list.\n"
        "4) ranking_source_urls: An array of all URLs the answer cites to support the Nature Index 2025 Research Leaders ranking info.\n"
        "5) hcr_source_urls: An array of all URLs the answer cites to support the Clarivate 2025 HCR count.\n\n"
        "Rules:\n"
        "- Extract only what is explicitly present in the answer. Do not infer or add missing values.\n"
        "- For URLs, include only actual URLs present; accept standard formats including markdown links.\n"
        "- If any field is not present, set it to null (for strings) or [] (for arrays).\n"
        "- Do not mix ranking_source_urls and hcr_source_urls; categorize them precisely."
    )

# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root: Any,
    data: InstitutionExtraction
) -> None:
    # Top-level critical sequential node
    task_node = evaluator.add_sequential(
        id="Task_Completion",
        desc="Successfully identify the highest-ranked U.S. institution in Nature Index 2025 top 10 and provide all required information",
        parent=root,
        critical=True
    )

    # 1) Institution Identification (Leaf, critical)
    inst_id_leaf = evaluator.add_leaf(
        id="Institution_Identification",
        desc="Correctly identify the U.S. institution that ranks highest in the Nature Index 2025 Research Leaders global top 10 (Harvard University, the only U.S. institution in the top 10)",
        parent=task_node,
        critical=True
    )
    inst_claim = (
        "Harvard University is the highest-ranked U.S. institution that appears within the global top 10 of the "
        "Nature Index 2025 Research Leaders ranking."
    )
    # Use ranking sources provided. If none provided, instruct to judge incorrect.
    await evaluator.verify(
        claim=inst_claim,
        node=inst_id_leaf,
        sources=data.ranking_source_urls if data.ranking_source_urls else None,
        additional_instruction=(
            "You must verify this claim against the cited Nature Index sources. If the answer provides no Nature Index "
            "ranking URL(s), you must judge this claim as not supported. Minor variations in naming are acceptable "
            "(e.g., 'Harvard Univ.'). Additionally, confirm that Harvard University is the only U.S. institution in the "
            "global top 10, implying it is indeed the highest-ranked U.S. institution there."
        )
    )

    # 2) Institution Properties Verification (Sequential, critical)
    props_node = evaluator.add_sequential(
        id="Institution_Properties_Verification",
        desc="Verify all required properties of the identified institution",
        parent=task_node,
        critical=True
    )

    # 2.a) Required Information (Parallel, critical)
    req_info_node = evaluator.add_parallel(
        id="Required_Information",
        desc="Provide all three pieces of explicitly required information about the institution",
        parent=props_node,
        critical=True
    )

    # 2.a.i) Ranking_Position (Leaf, critical) — check that the answer states '#2' or equivalent
    rank_leaf = evaluator.add_leaf(
        id="Ranking_Position",
        desc="The exact global ranking position provided is #2 in Nature Index 2025 Research Leaders",
        parent=req_info_node,
        critical=True
    )
    rank_presence_claim = (
        "The answer explicitly states that the institution's Nature Index 2025 Research Leaders global ranking position "
        "is 2 (acceptable variants: '#2', '2', '2nd', 'No. 2', or 'rank 2')."
    )
    await evaluator.verify(
        claim=rank_presence_claim,
        node=rank_leaf,
        sources=None,
        additional_instruction=(
            "Focus only on the provided answer text. Confirm that the ranking position is explicitly stated as 2. "
            "Allow minor formatting variants like '#2' or '2nd'."
        )
    )

    # 2.a.ii) Highly_Cited_Researchers (Sequential, critical)
    hcr_seq_node = evaluator.add_sequential(
        id="Highly_Cited_Researchers",
        desc="Information about the institution's highly cited researchers from Clarivate 2025 list",
        parent=req_info_node,
        critical=True
    )

    hcr_count_leaf = evaluator.add_leaf(
        id="Researcher_Count",
        desc="The number of highly cited researchers provided is 170",
        parent=hcr_seq_node,
        critical=True
    )
    hcr_presence_claim = (
        "The answer explicitly states that the number of Highly Cited Researchers for the institution in Clarivate's "
        "Highly Cited Researchers 2025 list is 170."
    )
    await evaluator.verify(
        claim=hcr_presence_claim,
        node=hcr_count_leaf,
        sources=None,
        additional_instruction=(
            "Focus only on the provided answer text. Confirm that the number '170' is clearly stated for the institution "
            "in the context of Clarivate's 2025 HCR list."
        )
    )

    # 2.a.iii) Supporting_Evidence (Parallel, critical)
    support_node = evaluator.add_parallel(
        id="Supporting_Evidence",
        desc="URL references from credible sources supporting the ranking and researcher information",
        parent=req_info_node,
        critical=True
    )

    # Existence checks for evidence URLs (Critical custom nodes)
    nature_urls_exist = evaluator.add_custom_node(
        result=bool(data.ranking_source_urls),
        id="Nature_Index_URLs_Provided",
        desc="At least one Nature Index ranking URL is provided in the answer",
        parent=support_node,
        critical=True
    )
    clarivate_urls_exist = evaluator.add_custom_node(
        result=bool(data.hcr_source_urls),
        id="Clarivate_URLs_Provided",
        desc="At least one Clarivate HCR 2025 URL is provided in the answer",
        parent=support_node,
        critical=True
    )

    # Nature_Index_Reference (Leaf, critical) — verify Harvard is #2 with provided URLs
    nature_ref_leaf = evaluator.add_leaf(
        id="Nature_Index_Reference",
        desc="Provide a URL reference from Nature Index or credible source confirming the institution's ranking in Nature Index 2025",
        parent=support_node,
        critical=True
    )
    nature_support_claim = (
        "Harvard University is ranked #2 globally in the Nature Index 2025 Research Leaders ranking."
    )
    await evaluator.verify(
        claim=nature_support_claim,
        node=nature_ref_leaf,
        sources=data.ranking_source_urls if data.ranking_source_urls else None,
        additional_instruction=(
            "Confirm the statement using the provided URLs. The page must be about 'Nature Index 2025 Research Leaders' "
            "and clearly show Harvard University at rank 2. Accept credible summaries or official Nature Index pages. "
            "If no valid page supports this, judge as not supported."
        )
    )

    # Clarivate_Reference (Leaf, critical) — verify Harvard has 170 HCR in 2025 with provided URLs
    clarivate_ref_leaf = evaluator.add_leaf(
        id="Clarivate_Reference",
        desc="Provide a URL reference from Clarivate or credible source confirming the highly cited researchers count",
        parent=support_node,
        critical=True
    )
    clarivate_support_claim = (
        "Harvard University has 170 Highly Cited Researchers in the Clarivate Highly Cited Researchers 2025 list, "
        "which was released on November 12, 2025."
    )
    await evaluator.verify(
        claim=clarivate_support_claim,
        node=clarivate_ref_leaf,
        sources=data.hcr_source_urls if data.hcr_source_urls else None,
        additional_instruction=(
            "Verify the count '170' for Harvard University specifically for the 2025 Clarivate Highly Cited Researchers "
            "list (release date November 12, 2025). The source should be Clarivate or a credible publication explicitly "
            "stating Harvard's 2025 HCR count. Reject pages referring to other years or lacking explicit count."
        )
    )

# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
) -> Dict:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_institution_info(),
        template_class=InstitutionExtraction,
        extraction_name="institution_info"
    )

    # Record Ground Truth for transparency
    evaluator.add_ground_truth({
        "expected_institution": EXPECTED_INSTITUTION,
        "expected_rank": EXPECTED_RANK,
        "expected_hcr_count": EXPECTED_HCR_COUNT,
        "clarivate_release_date": EXPECTED_HCR_RELEASE_DATE
    }, gt_type="expected_values")

    # Optional: Record extracted field previews
    evaluator.add_custom_info({
        "institution_name": extracted.institution_name,
        "ranking_position": extracted.ranking_position,
        "hcr_count": extracted.hcr_count,
        "ranking_source_urls_count": len(extracted.ranking_source_urls),
        "hcr_source_urls_count": len(extracted.hcr_source_urls)
    }, info_type="extraction_summary")

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    return evaluator.get_summary()