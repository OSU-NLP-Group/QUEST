import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "rutgers_grad_requirements"
TASK_DESCRIPTION = (
    "A student is planning to transfer to Rutgers University to complete a bachelor's degree. "
    "According to Rutgers' official graduation requirements: "
    "(1) What is the total number of credit hours required for graduation? "
    "(2) What is the residency requirement that specifies the minimum number of credits that must be earned at Rutgers?"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TotalCreditsInfo(BaseModel):
    """Information about total credits requirement as stated in the answer."""
    stated_credits: Optional[str] = None  # e.g., "120 credits", "120–132 credits", "at least 120 credits"
    urls: List[str] = Field(default_factory=list)  # URLs cited for total credit requirement


class ResidencyInfo(BaseModel):
    """Information about residency requirement as stated in the answer."""
    stated_residency: Optional[str] = None  # e.g., "30 of the last 42 credits at Rutgers"
    urls: List[str] = Field(default_factory=list)  # URLs cited for residency requirement


class RutgersRequirementsExtraction(BaseModel):
    """Combined extraction for both requirements."""
    total_credits: Optional[TotalCreditsInfo] = None
    residency: Optional[ResidencyInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract from the answer exactly what it states for:
    1) The total number of credit hours required to graduate with a bachelor's degree at Rutgers University.
    2) The residency requirement that specifies how many credits must be earned at Rutgers.

    For each part, extract:
    - stated_credits (for total credits) or stated_residency (for residency): a concise phrase capturing exactly what the answer states (e.g., "120–132 credits", "minimum 120 credits", "30 of the last 42 credits at Rutgers").
    - urls: all reference URLs explicitly cited in the answer for that part. Only include actual URLs found in the answer text (including markdown links). If none are provided, return an empty list.

    JSON structure to return:
    {
      "total_credits": {
        "stated_credits": string or null,
        "urls": [list of urls]
      },
      "residency": {
        "stated_residency": string or null,
        "urls": [list of urls]
      }
    }

    Rules:
    - Do not invent text or URLs. Extract exactly what's present in the answer.
    - Keep URLs complete (with protocol). If a URL lacks protocol, prepend http://
    - De-duplicate URLs while preserving order.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    """Return sanitized list of non-empty URLs."""
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if isinstance(u, str) and u.strip():
            cleaned.append(u.strip())
    return cleaned


def build_total_credits_claim(total_info: Optional[TotalCreditsInfo]) -> str:
    """
    Build a verification claim for total credits based on what the answer stated.
    If the answer explicitly mentions a 120–132 range, use that; otherwise fall back to a general minimum claim.
    """
    stated = (total_info.stated_credits or "").lower().strip() if total_info else ""
    has_120 = "120" in stated
    has_132 = "132" in stated or ("120–132" in stated) or ("120-132" in stated)

    if has_120 and has_132:
        return "Rutgers University requires between 120 and 132 credits to graduate with a bachelor's degree, depending on school or major."
    elif has_120:
        return "Rutgers University requires a minimum of 120 credits to graduate with a bachelor's degree; some programs may require more depending on school or major."
    elif stated:
        return f"Rutgers University's bachelor's graduation credit requirement is: {total_info.stated_credits}."
    else:
        # Generic, still acceptable to check support if URLs exist
        return "Rutgers University requires a minimum number of undergraduate degree credits (typically at least 120) to graduate."


def build_residency_claim(res_info: Optional[ResidencyInfo]) -> str:
    """
    Build a verification claim for residency requirement.
    Prefer the canonical statement 'At least 30 of the final 42 credits must be earned at Rutgers'.
    If the answer stated something else, reflect it; otherwise use canonical form.
    """
    stated = (res_info.stated_residency or "").strip() if res_info else ""
    if stated:
        # Try to normalize common variants while preserving the answer's content
        return f"Rutgers University's residency requirement as stated is: {stated}."
    # Canonical statement
    return "Rutgers University's residency requirement: students must earn at least 30 of their final 42 credits at Rutgers."


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_total_credit_requirement(
    evaluator: Evaluator,
    parent_node,
    extracted: RutgersRequirementsExtraction,
) -> None:
    """
    Build and evaluate the 'total_credit_requirement' subtree.
    Critical node with two critical leaves:
      - credit_range_accuracy: checks the answer states '120 to 132 credits depending on the major'
      - reference_url_total_credits: verifies this requirement with Rutgers official sources (URLs from the answer)
    """
    # Create parent node: critical Parallel
    total_node = evaluator.add_parallel(
        id="total_credit_requirement",
        desc="Answer correctly identifies the minimum total credit hours required for a bachelor's degree at Rutgers University",
        parent=parent_node,
        critical=True
    )

    # Leaf: credit_range_accuracy (simple check against the answer content)
    range_leaf = evaluator.add_leaf(
        id="credit_range_accuracy",
        desc="The answer states that Rutgers requires 120 to 132 credits for graduation depending on the major",
        parent=total_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that Rutgers requires 120 to 132 credits for graduation depending on the major.",
        node=range_leaf,
        additional_instruction=(
            "Judge based on the provided answer text only. Accept reasonable wording variations such as "
            "'between 120 and 132', '120–132', 'ranges from 120 to 132', or 'varies by major/school'. "
            "Focus on whether the answer explicitly conveys this 120–132 range."
        )
    )

    # Leaf: reference_url_total_credits (must use official Rutgers sources and support the claim)
    ref_urls = _non_empty_urls(extracted.total_credits.urls if extracted and extracted.total_credits else [])
    if not ref_urls:
        # Fail immediately if no URLs provided
        evaluator.add_leaf(
            id="reference_url_total_credits",
            desc="Provides a reference URL from Rutgers official sources supporting the total credit requirement",
            parent=total_node,
            critical=True,
            score=0.0,
            status="failed"
        )
    else:
        ref_leaf = evaluator.add_leaf(
            id="reference_url_total_credits",
            desc="Provides a reference URL from Rutgers official sources supporting the total credit requirement",
            parent=total_node,
            critical=True
        )
        claim = build_total_credits_claim(extracted.total_credits if extracted else None)
        await evaluator.verify(
            claim=claim,
            node=ref_leaf,
            sources=ref_urls,
            additional_instruction=(
                "ONLY PASS if at least one provided URL is an official Rutgers domain (rutgers.edu or its subdomains) "
                "and its content supports the claim. Treat pages from Rutgers schools/registrars/advising as official. "
                "Look for explicit language about bachelor's graduation credit requirements (e.g., minimum 120 credits; "
                "some programs requiring more)."
            )
        )


async def verify_residency_requirement(
    evaluator: Evaluator,
    parent_node,
    extracted: RutgersRequirementsExtraction,
) -> None:
    """
    Build and evaluate the 'residency_requirement' subtree.
    Critical node with two critical leaves:
      - residency_credit_accuracy: checks the answer states '30 of the last 42 credits at Rutgers'
      - reference_url_residency: verifies this requirement with Rutgers official sources (URLs from the answer)
    """
    # Create parent node: critical Parallel
    resid_node = evaluator.add_parallel(
        id="residency_requirement",
        desc="Answer correctly identifies the residency requirement specifying how many credits must be earned at Rutgers",
        parent=parent_node,
        critical=True
    )

    # Leaf: residency_credit_accuracy (simple check against the answer content)
    acc_leaf = evaluator.add_leaf(
        id="residency_credit_accuracy",
        desc="The answer states that students must earn a minimum of 30 of the last 42 credits at Rutgers",
        parent=resid_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that students must earn a minimum of 30 of the last 42 credits at Rutgers.",
        node=acc_leaf,
        additional_instruction=(
            "Judge based on the provided answer text only. Accept reasonable wording variations such as "
            "'at least 30 of the final 42 credits must be earned at Rutgers', or equivalent phrases conveying the same policy."
        )
    )

    # Leaf: reference_url_residency (must use official Rutgers sources and support the claim)
    ref_urls = _non_empty_urls(extracted.residency.urls if extracted and extracted.residency else [])
    if not ref_urls:
        evaluator.add_leaf(
            id="reference_url_residency",
            desc="Provides a reference URL from Rutgers official sources supporting the residency requirement",
            parent=resid_node,
            critical=True,
            score=0.0,
            status="failed"
        )
    else:
        ref_leaf = evaluator.add_leaf(
            id="reference_url_residency",
            desc="Provides a reference URL from Rutgers official sources supporting the residency requirement",
            parent=resid_node,
            critical=True
        )
        claim = build_residency_claim(extracted.residency if extracted else None)
        await evaluator.verify(
            claim=claim,
            node=ref_leaf,
            sources=ref_urls,
            additional_instruction=(
                "ONLY PASS if at least one provided URL is an official Rutgers domain (rutgers.edu or its subdomains) "
                "and it clearly supports the residency policy that at least 30 of the final 42 credits must be earned at Rutgers. "
                "Minor wording variations are acceptable as long as the policy is equivalent."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for Rutgers graduation requirements (total credits and residency).
    Returns a standardized summary including the verification tree and final score.
    """
    # Initialize evaluator with root parallel strategy
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
    # Mark root as critical (as required by rubric)
    evaluator.root.critical = True

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RutgersRequirementsExtraction,
        extraction_name="rutgers_requirements_extraction"
    )

    # Optional: add ground truth expectations (as guidance context)
    evaluator.add_ground_truth({
        "expected_total_credits_statement": "120 to 132 credits depending on the major",
        "expected_residency_statement": "At least 30 of the last 42 credits must be earned at Rutgers",
        "notes": "Evidence must come from Rutgers official domains (rutgers.edu or subdomains)."
    })

    # Build and run verification subtrees
    await verify_total_credit_requirement(evaluator, root, extracted)
    await verify_residency_requirement(evaluator, root, extracted)

    # Return the structured evaluation summary
    return evaluator.get_summary()