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
TASK_ID = "wv_veto_override_2025"
TASK_DESCRIPTION = """
In West Virginia, according to the state constitution, what is the minimum number of votes required in the House of Delegates to override the governor's veto of a supplementary appropriation bill in 2025? Provide the specific constitutional provision that establishes this requirement.
"""

EXPECTED_FACTS = {
    "constitutional_provision": "Article VI, Section 51",
    "threshold_basis": "two-thirds of the members elected",
    "house_membership_2025": "100 members",
    "minimum_votes": "67"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WVVetoOverrideExtraction(BaseModel):
    # What provision was cited
    provision: Optional[str] = None
    provision_quote: Optional[str] = None

    # How the bill is classified (supplemental appropriation under budget/appropriation framework)
    classification_statement: Optional[str] = None

    # Threshold description in the answer
    threshold_statement: Optional[str] = None

    # House size in the answer (text) and as a number if present
    house_membership_statement: Optional[str] = None
    house_membership_number: Optional[str] = None

    # Minimum votes computed in the answer (text) and numeric if present
    min_votes_statement: Optional[str] = None
    min_votes_number: Optional[str] = None
    calculation_explanation: Optional[str] = None

    # Source URLs provided in the answer
    constitution_urls: List[str] = Field(default_factory=list)
    house_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_wv_facts() -> str:
    return """
    Extract from the answer the key items below. Return null for any field not explicitly present.
    Fields to extract:
    - provision: The exact constitutional provision citation used (e.g., "Article VI, Section 51", or a legally equivalent short form such as "Art. VI § 51").
    - provision_quote: Any quotation or paraphrase from the answer that describes what that provision requires for veto overrides of budget/appropriation/supplementary appropriation bills.
    - classification_statement: The sentence(s) where the answer classifies a supplementary appropriation bill as falling under the budget/appropriations framework governed by the cited provision.
    - threshold_statement: The answer's statement of the veto-override threshold (e.g., "two-thirds of the members elected" vs "members present").
    - house_membership_statement: The sentence(s) where the answer states the size of the West Virginia House of Delegates as of 2025.
    - house_membership_number: The numeric size of the House if digits are present in the answer text (e.g., "100"). If only spelled-out words are present (e.g., "one hundred") and no digits appear, return null.
    - min_votes_statement: The sentence(s) where the answer states the minimum number of votes required in the House of Delegates to override the veto in 2025.
    - min_votes_number: The numeric minimum vote count if digits are present in the answer text (e.g., "67"). If only spelled-out words are present (e.g., "sixty-seven") and no digits appear, return null.
    - calculation_explanation: Any explanation of how the minimum votes were computed (e.g., "ceiling(2/3 × 100) = 67").
    - constitution_urls: A list of all URLs in the answer that are intended to support the constitutional provision and its threshold (extract only actual URLs that appear).
    - house_urls: A list of all URLs in the answer that are intended to support the West Virginia House of Delegates membership count (extract only actual URLs that appear).

    Notes:
    - Extract only what appears in the answer text verbatim. Do not infer or invent.
    - For URLs, include only valid URLs explicitly present in the answer (plain or markdown).
    """


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_complete_answer(
    evaluator: Evaluator,
    root,
    extracted: WVVetoOverrideExtraction,
) -> None:
    """
    Build the verification tree based on the rubric and run verifications.
    """

    # Top-level "complete_answer" node (critical, parallel aggregation)
    complete_node = evaluator.add_parallel(
        id="complete_answer",
        desc="State the minimum House of Delegates votes needed to override a gubernatorial veto of a supplementary appropriation bill in 2025 and cite the specific West Virginia constitutional provision establishing the rule.",
        parent=root,
        critical=True
    )

    # 1) constitutional_provision_identification (leaf)
    node_provision_id = evaluator.add_leaf(
        id="constitutional_provision_identification",
        desc="Identifies Article VI, Section 51 of the West Virginia Constitution as the governing provision for veto overrides of supplementary appropriation bills.",
        parent=complete_node,
        critical=True
    )
    claim_provision_id = (
        "The answer identifies Article VI, Section 51 of the West Virginia Constitution "
        "as the provision governing veto overrides for supplementary appropriation (budget/appropriation) bills "
        "(allow equivalent citations like 'Art. VI § 51' or 'W. Va. Const. art. VI, § 51')."
    )
    await evaluator.verify(
        claim=claim_provision_id,
        node=node_provision_id,
        additional_instruction="Judge using only the answer content; allow minor citation format variations that are legally equivalent."
    )

    # 2) bill_type_classification (leaf)
    node_classification = evaluator.add_leaf(
        id="bill_type_classification",
        desc="Classifies a supplementary appropriation bill as falling under the budget/appropriations bill category governed by Article VI, Section 51.",
        parent=complete_node,
        critical=True
    )
    claim_classification = (
        "The answer explicitly classifies a supplementary appropriation bill as part of the budget/appropriations framework "
        "governed by Article VI, Section 51."
    )
    await evaluator.verify(
        claim=claim_classification,
        node=node_classification,
        additional_instruction="Look for language tying 'supplementary appropriation bill' to the Article VI, Section 51 budgeting/appropriations regime."
    )

    # 3) override_threshold_extraction (leaf)
    node_threshold = evaluator.add_leaf(
        id="override_threshold_extraction",
        desc="States that the constitutional threshold for overriding the veto is two-thirds of the members elected to the House of Delegates (i.e., not based on members present).",
        parent=complete_node,
        critical=True
    )
    claim_threshold = (
        "The answer states that the constitutional threshold for overriding the veto is two-thirds of the members elected "
        "to the House of Delegates, not just two-thirds of members present."
    )
    await evaluator.verify(
        claim=claim_threshold,
        node=node_threshold,
        additional_instruction="Accept equivalent phrasings that clearly indicate 'two-thirds of the members elected' rather than 'members present'."
    )

    # 4) chamber_composition_identification (leaf)
    node_house_size = evaluator.add_leaf(
        id="chamber_composition_identification",
        desc="States that the West Virginia House of Delegates consists of 100 members as of 2025.",
        parent=complete_node,
        critical=True
    )
    claim_house_size = (
        "The answer states that the West Virginia House of Delegates consists of 100 members (for the 2025 context)."
    )
    await evaluator.verify(
        claim=claim_house_size,
        node=node_house_size,
        additional_instruction="It's acceptable if the answer simply states '100 members' without explicitly repeating 'as of 2025' so long as the count used is 100."
    )

    # 5) vote_calculation (leaf)
    node_vote_calc = evaluator.add_leaf(
        id="vote_calculation",
        desc="Computes the minimum votes as ceiling(2/3 × 100) = 67 votes.",
        parent=complete_node,
        critical=True
    )
    claim_vote_calc = (
        "The answer computes the minimum required House votes as 67 (ceiling of two-thirds of 100)."
    )
    await evaluator.verify(
        claim=claim_vote_calc,
        node=node_vote_calc,
        additional_instruction="Focus on whether the answer explicitly concludes 67 votes as the minimum."
    )

    # 6) source_urls_provided (parallel container with two critical checks)
    sources_node = evaluator.add_parallel(
        id="source_urls_provided",
        desc="Provides verifiable reference URL(s) supporting the constitutional provision text (Article VI, Section 51) and the House membership count used for the calculation.",
        parent=complete_node,
        critical=True
    )

    # 6a) Constitution URLs existence (critical custom)
    constitution_urls_exist = evaluator.add_custom_node(
        result=bool(extracted.constitution_urls),
        id="constitution_urls_exist",
        desc="At least one URL is provided to support the constitutional provision and its threshold.",
        parent=sources_node,
        critical=True
    )

    # 6b) Constitution URLs support claim (critical leaf, verified by URLs)
    constitution_urls_support = evaluator.add_leaf(
        id="constitution_urls_support_provision",
        desc="Cited constitution URL(s) support Article VI, Section 51 and the 'two-thirds of the members elected' override threshold for budget/appropriation bills.",
        parent=sources_node,
        critical=True
    )
    claim_constitution_support = (
        "This source contains or clearly reflects Article VI, Section 51 of the West Virginia Constitution, "
        "and shows that overriding the governor's veto of budget/appropriation (including supplementary appropriation) bills "
        "requires approval by two-thirds of the members elected to each house."
    )
    await evaluator.verify(
        claim=claim_constitution_support,
        node=constitution_urls_support,
        sources=extracted.constitution_urls,
        additional_instruction="Allow exact constitution text pages or authoritative summaries that quote the same rule; minor wording differences are acceptable if they unambiguously convey the same requirement."
    )

    # 6c) House URLs existence (critical custom)
    house_urls_exist = evaluator.add_custom_node(
        result=bool(extracted.house_urls),
        id="house_urls_exist",
        desc="At least one URL is provided to support the 100-member House figure.",
        parent=sources_node,
        critical=True
    )

    # 6d) House URLs support membership count (critical leaf, verified by URLs)
    house_urls_support = evaluator.add_leaf(
        id="house_urls_support_membership",
        desc="Cited URL(s) support that the West Virginia House of Delegates has 100 members (for the 2025 context).",
        parent=sources_node,
        critical=True
    )
    claim_house_support = "This source states that the West Virginia House of Delegates has 100 members."
    await evaluator.verify(
        claim=claim_house_support,
        node=house_urls_support,
        sources=extracted.house_urls,
        additional_instruction="Accept official legislative pages or reliable references (e.g., legislature site, encyclopedia) that state 100 members."
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
    Evaluate an answer for the West Virginia veto override minimum votes task.
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_wv_facts(),
        template_class=WVVetoOverrideExtraction,
        extraction_name="extracted_answer_facts"
    )

    # Add ground truth for reference (not used for judgment directly)
    evaluator.add_ground_truth(
        {
            "expected_constitutional_provision": EXPECTED_FACTS["constitutional_provision"],
            "expected_threshold_basis": EXPECTED_FACTS["threshold_basis"],
            "expected_house_membership_2025": EXPECTED_FACTS["house_membership_2025"],
            "expected_minimum_votes": EXPECTED_FACTS["minimum_votes"],
        },
        gt_type="expected_facts"
    )

    # Build tree and run verifications
    await build_and_verify_complete_answer(evaluator, root, extracted)

    # Return standardized summary
    return evaluator.get_summary()