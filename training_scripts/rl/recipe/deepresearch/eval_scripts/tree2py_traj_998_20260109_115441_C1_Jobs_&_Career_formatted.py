import asyncio
import logging
from typing import Any, List, Optional, Dict
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "comptia_security_plus_requirements"
TASK_DESCRIPTION = (
    "I am planning to obtain the CompTIA Security+ certification and want to understand the ongoing requirements. "
    "Please provide the following information about the CompTIA Security+ certification: "
    "(1) How many Continuing Education Units (CEUs) are required to renew the Security+ certification? "
    "(2) What is the passing score for the Security+ exam (on the 100-900 scale)? "
    "Please ensure all information is obtained from CompTIA's official website and include the reference URL."
)

# Ground truth info (for reference logging only; verification still uses answer + sources)
GROUND_TRUTH = {
    "ceu_required": "50 CEUs",
    "renewal_window": "3 years",
    "passing_score": "750 on the 100–900 scale",
    "official_domain": "comptia.org"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SecurityPlusExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer.
    """
    ceus_required: Optional[str] = None          # e.g., "50 CEUs", "50"
    renewal_window: Optional[str] = None         # e.g., "3 years", "three-year certification period"
    passing_score: Optional[str] = None          # e.g., "750", "750/900", "750 on the 100–900 scale"
    reference_urls: List[str] = Field(default_factory=list)  # all URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_security_plus() -> str:
    return """
    Extract the following information exactly as stated in the answer about CompTIA Security+:
    1) ceus_required: The number of Continuing Education Units (CEUs) the answer states are required to renew Security+. Return it as a short string exactly as written (e.g., "50 CEUs", "50").
    2) renewal_window: The renewal time window stated (e.g., "3 years", "three-year certification period from the date of certification"). Return a concise phrase exactly as written.
    3) passing_score: The passing score for Security+ on the 100–900 scale as stated (e.g., "750", "750 on the 100–900 scale").
    4) reference_urls: All reference URLs explicitly provided in the answer text (including plain URLs and markdown links). Extract only valid URLs. If protocol is missing, prepend http://. Do not invent URLs.

    If any of the above items are not mentioned in the answer, set them to null (for strings) or an empty list (for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_official_comptia_url(url: str) -> bool:
    """
    Check if the URL belongs to CompTIA's official domain (comptia.org).
    Accept subdomains like www.comptia.org, certification.comptia.org, etc.
    """
    if not url:
        return False
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        return host.endswith("comptia.org")
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_security_plus_tree(
    evaluator: Evaluator,
    root_node,
    extracted: SecurityPlusExtraction,
) -> None:
    """
    Build the verification tree and run checks according to the rubric.
    """
    # Create the main critical parallel node as per rubric
    main = evaluator.add_parallel(
        id="security_plus_requirements",
        desc="Verify the answer matches the specified Security+ renewal and exam requirements and includes official CompTIA sourcing with reference URL(s).",
        parent=root_node,
        critical=True,  # Parent is critical; all children must be critical
    )

    # --- Reference URL(s) Provided (Critical) ---
    has_urls = bool(extracted.reference_urls)
    ref_urls_node = evaluator.add_custom_node(
        result=has_urls,
        id="reference_urls_provided",
        desc="Provides at least one reference URL for verification.",
        parent=main,
        critical=True
    )

    # --- Official CompTIA Source Used (Critical) ---
    # Require all provided URLs to be from comptia.org
    all_official = has_urls and all(is_official_comptia_url(u) for u in extracted.reference_urls)
    official_src_node = evaluator.add_custom_node(
        result=all_official,
        id="official_comptia_source_used",
        desc="The supporting citation(s) used to justify the claims are from CompTIA’s official website (e.g., comptia.org).",
        parent=main,
        critical=True
    )

    # --- CEU Renewal Requirement (50 CEUs) (Critical) ---
    # Verify the claim exactly as stated in the answer against the provided sources.
    ceu_leaf = evaluator.add_leaf(
        id="ceu_renewal_requirement",
        desc="States that renewing CompTIA Security+ requires 50 CEUs.",
        parent=main,
        critical=True
    )

    ceu_value = extracted.ceus_required or "UNKNOWN"
    ceu_claim = f"Renewing CompTIA Security+ requires {ceu_value} CEUs."
    await evaluator.verify(
        claim=ceu_claim,
        node=ceu_leaf,
        sources=extracted.reference_urls if extracted.reference_urls else None,
        extra_prerequisites=[ref_urls_node, official_src_node],
        additional_instruction=(
            "Verify exclusively using the cited CompTIA official page(s). "
            "The claim must match Security+ CE program requirements. "
            "Allow minor formatting variants (e.g., 'CEUs' vs 'Continuing Education Units'). "
            "If the page states '50 CEUs' for Security+ renewal and the answer claim matches that value, pass; otherwise fail."
        ),
    )

    # --- Renewal Time Window (3 years) (Critical) ---
    renewal_leaf = evaluator.add_leaf(
        id="renewal_time_window",
        desc="States that renewal must occur within a three-year certification period from the date of certification.",
        parent=main,
        critical=True
    )

    renewal_value = extracted.renewal_window or "UNKNOWN"
    renewal_claim = (
        f"CompTIA Security+ renewal must occur within a {renewal_value} certification period from the date of certification."
    )
    await evaluator.verify(
        claim=renewal_claim,
        node=renewal_leaf,
        sources=extracted.reference_urls if extracted.reference_urls else None,
        extra_prerequisites=[ref_urls_node, official_src_node],
        additional_instruction=(
            "Verify exclusively on the cited CompTIA official page(s). "
            "Accept equivalent phrasing such as '3 years', 'three years', 'three-year certification period'. "
            "Ensure the statement pertains specifically to Security+ renewal policy."
        ),
    )

    # --- Exam Passing Score (750 on 100–900 scale) (Critical) ---
    score_leaf = evaluator.add_leaf(
        id="exam_passing_score",
        desc="States that the passing score is 750 on the 100–900 scale.",
        parent=main,
        critical=True
    )

    score_value = extracted.passing_score or "UNKNOWN"
    score_claim = f"The passing score for CompTIA Security+ on the 100–900 scale is {score_value}."
    await evaluator.verify(
        claim=score_claim,
        node=score_leaf,
        sources=extracted.reference_urls if extracted.reference_urls else None,
        extra_prerequisites=[ref_urls_node, official_src_node],
        additional_instruction=(
            "Verify on the CompTIA official page(s) that the passing score for Security+ is 750 on the 100–900 scale. "
            "Allow minor textual variations (e.g., '750/900', '750 on a scale of 100 to 900')."
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
    Evaluate an answer for the CompTIA Security+ requirements task and return a structured result.
    """
    # Initialize evaluator (root node is non-critical by default)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent checks at the top-level
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
        prompt=prompt_extract_security_plus(),
        template_class=SecurityPlusExtraction,
        extraction_name="security_plus_extraction",
    )

    # Record ground truth as informational context
    evaluator.add_ground_truth({
        "expected": GROUND_TRUTH,
        "note": "Ground truth recorded for reference; verification uses the answer's stated values checked against CompTIA official sources."
    })

    # Build verification tree and run checks
    await build_security_plus_tree(evaluator, root, extracted)

    # Return structured summary with final score and verification tree
    return evaluator.get_summary()