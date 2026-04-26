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
TASK_ID = "cruelty_free_cert_rigor"
TASK_DESCRIPTION = """
Which cruelty-free certification for cosmetics and personal care products is considered more rigorous: Leaping Bunny or PETA? Provide supporting evidence for your answer.
"""

# Optional ground truth context (for info only; not used directly in verification)
GROUND_TRUTH_INFO = {
    "expected_more_rigorous": "Leaping Bunny",
    "typical_reasons": [
        "Requires full supply-chain verification via a Supplier Monitoring System (SMS) covering raw materials to finished products.",
        "Requires independent audits/third-party verification to ensure compliance.",
        "Generally regarded by credible ethical consumer guides and industry sources as more comprehensive and trusted than PETA's program."
    ],
    "peta_notes": [
        "PETA's 'Beauty Without Bunnies' program typically relies on company self-attestation and does not universally require third-party audits or full supply-chain monitoring."
    ]
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AnswerEvidenceExtraction(BaseModel):
    """
    Structured extraction of the agent's answer:
    - chosen_certification: The certification identified as more rigorous (e.g., 'Leaping Bunny' or 'PETA').
    - evidence_points: Reasons/examples cited to justify rigor (each as a short string, as listed in the answer).
    - sources: All URLs cited in the answer as supporting evidence (e.g., certification pages, credible articles).
    """
    chosen_certification: Optional[str] = None
    evidence_points: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_answer_evidence() -> str:
    return """
    Extract structured information from the answer about which certification is claimed to be more rigorous and the supporting evidence.

    Return a JSON object with the following fields:
    1) chosen_certification: The certification explicitly identified in the answer as the more rigorous one between 'Leaping Bunny' and 'PETA'. If the answer implies rigor without naming one, return the phrase used; otherwise null.
    2) evidence_points: A list of short strings, each capturing a distinct reason or piece of evidence mentioned in the answer. Examples include: 'independent audits', 'third-party verification', 'full supply-chain verification', 'Supplier Monitoring System', 'recognized as most comprehensive/trusted by credible sources'.
       - If none are mentioned, return an empty list.
    3) sources: All URLs that the answer cites as supporting evidence (e.g., official certification pages, credible articles, ethical consumer guides). Extract only the actual URLs present in the answer (plain or markdown). If none are present, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extraction: AnswerEvidenceExtraction,
    root_node_desc: str
) -> None:
    """
    Build the verification tree and run checks according to the rubric:
    - Answer_Evaluation (critical, parallel)
      - Certification_Identification (critical leaf)
      - Supporting_Evidence (critical leaf)
    """
    # Create the top-level evaluation node as critical (parallel)
    answer_eval_node = evaluator.add_parallel(
        id="Answer_Evaluation",
        desc=root_node_desc,
        parent=evaluator.root,
        critical=True
    )

    # Leaf 1: Certification Identification
    # Verify that the answer explicitly identifies Leaping Bunny as the more rigorous certification compared to PETA.
    cert_id_leaf = evaluator.add_leaf(
        id="Certification_Identification",
        desc="The answer correctly identifies Leaping Bunny as the more rigorous cruelty-free certification compared to PETA",
        parent=answer_eval_node,
        critical=True
    )

    cert_claim = (
        "The answer explicitly identifies 'Leaping Bunny' as the more rigorous cruelty-free certification "
        "compared to PETA."
    )
    await evaluator.verify(
        claim=cert_claim,
        node=cert_id_leaf,
        additional_instruction=(
            "Judge based on the provided answer content. Accept phrasing variations such as "
            "'Leaping Bunny is stricter/more stringent/more rigorous than PETA', "
            "or references to 'Cruelty Free International's Leaping Bunny program'. "
            "If the answer names PETA as more rigorous or does not clearly pick Leaping Bunny over PETA, mark incorrect."
        ),
    )

    # Leaf 2: Supporting Evidence
    # Verify that the answer provides supporting evidence for why Leaping Bunny is more rigorous.
    evidence_leaf = evaluator.add_leaf(
        id="Supporting_Evidence",
        desc=(
            "The answer provides supporting evidence for why the identified certification is more rigorous "
            "(e.g., independent audit requirements, supply-chain verification/Supplier Monitoring System, and/or "
            "recognition by credible industry sources/ethical consumer guides as the most comprehensive or trusted standard)"
        ),
        parent=answer_eval_node,
        critical=True
    )

    # Build a flexible claim that can be checked either against answer content or the cited sources.
    # Use disjunctive phrasing to allow any valid evidence category to satisfy the requirement.
    # When sources are present, verify against the URLs; otherwise, verify based on the answer text alone.
    sources_to_check = extraction.sources if extraction and extraction.sources else None

    # Provide a concise summary of the evidence points (if any) to guide the verifier.
    evidence_summary = ""
    if extraction and extraction.evidence_points:
        joined_points = "; ".join(extraction.evidence_points[:6])  # cap to keep prompt concise
        evidence_summary = f"Evidence points extracted from the answer: {joined_points}."

    evidence_claim = (
        "At least one of the following statements is supported: "
        "(1) Leaping Bunny certification requires independent audits or third-party verification; "
        "(2) Leaping Bunny certification requires full supply-chain verification via a Supplier Monitoring System (SMS); "
        "(3) Credible industry sources or ethical consumer guides recognize Leaping Bunny as the most comprehensive or trusted cruelty-free standard."
    )

    await evaluator.verify(
        claim=evidence_claim,
        node=evidence_leaf,
        sources=sources_to_check,
        additional_instruction=(
            "Evaluate whether the answer provides substantive supporting evidence. "
            "If URLs are provided, check if any of them substantively support at least one of the statements. "
            "If no URLs are provided, judge based on the answer text alone. "
            "Allow reasonable phrasing variations such as 'supplier monitoring system', 'supply chain oversight', "
            "'independent verification', 'third-party audits', or mentions of recognition by credible guides/sources. "
            + (evidence_summary if evidence_summary else "")
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
    model: str = "o4-mini"
) -> Dict:
    """
    Entry point for evaluating the agent's answer to the cruelty-free certification rigor question.
    """
    # Initialize evaluator (root is non-critical by framework default; we add a critical child node)
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

    # Extract structured evidence from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_answer_evidence(),
        template_class=AnswerEvidenceExtraction,
        extraction_name="answer_evidence"
    )

    # Record optional ground truth context
    evaluator.add_ground_truth(GROUND_TRUTH_INFO, gt_type="ground_truth_context")

    # Build verification tree and run checks
    await build_and_verify_tree(
        evaluator=evaluator,
        extraction=extraction,
        root_node_desc=(
            "Evaluate whether the answer correctly identifies the more rigorous cruelty-free certification "
            "and provides supporting evidence"
        )
    )

    # Return unified summary with verification tree and score
    return evaluator.get_summary()