import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fcc_nors_initial_notification"
TASK_DESCRIPTION = (
    "According to FCC regulations, within how many minutes must wireless service providers submit an initial "
    "notification to the Commission after discovering a reportable network outage under the Network Outage "
    "Reporting System (NORS)? Provide your answer as the number of minutes and include a reference URL from an "
    "official source."
)

# Ground truth for initial notification timeline (minutes)
EXPECTED_MINUTES = 120

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NORSInitialNotificationExtraction(BaseModel):
    """
    Extracted information from the agent's answer related to the FCC NORS initial notification requirement.
    """
    minutes: Optional[str] = None  # Prefer string to handle variants like "120" or "two hours"
    urls: List[str] = Field(default_factory=list)  # Reference URLs provided in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nors_initial_notification() -> str:
    return """
    Extract the information the answer provides about the FCC NORS initial notification timing for wireless service providers:
    1) minutes: The number of minutes stated in the answer for the initial notification requirement under NORS. 
       - If the answer uses hours instead of minutes (e.g., "within two hours" or "2 hours"), convert it to minutes and return the numeric minutes as a string (e.g., "120").
       - If multiple times are mentioned, select the one clearly associated with the initial notification to the Commission after discovering a reportable network outage (NORS).
       - If the answer does not specify the minutes or timing, return null.
    2) urls: Extract all reference URLs explicitly provided in the answer. Include full URLs presented in plain text or markdown links.
       - Do not invent URLs. Only include those explicitly present in the answer text.
       - If no URLs are provided, return an empty array.
    Return the result as a JSON object with fields: minutes (string or null) and urls (array of strings).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def filter_authoritative_urls(urls: List[str]) -> List[str]:
    """
    Retain only URLs from authoritative sources:
    - FCC website domains (fcc.gov, docs.fcc.gov and subdomains)
    - eCFR (ecfr.gov)
    - Federal Register (federalregister.gov)
    """
    allowed_domains = {"fcc.gov", "docs.fcc.gov", "ecfr.gov", "federalregister.gov"}
    filtered = []
    for u in urls:
        try:
            netloc = urlparse(u).netloc.lower()
            # Allow subdomains for the allowed domains
            if any(netloc == d or netloc.endswith(f".{d}") for d in allowed_domains):
                filtered.append(u)
        except Exception:
            # Ignore malformed URLs
            continue
    return filtered


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root,
    extraction: NORSInitialNotificationExtraction
) -> None:
    """
    Build the verification tree according to the rubric and execute the checks.
    """
    # Create a critical parallel parent node under the (non-critical) evaluator root
    main_node = evaluator.add_parallel(
        id="fcc_nors_main",
        desc="Verify the correct initial notification timeline for wireless service providers under FCC NORS reporting requirements",
        parent=root,
        critical=True
    )

    # Child 1: Check the answer correctly states "within 120 minutes"
    node_timeline = evaluator.add_leaf(
        id="notification_timeline",
        desc="The answer correctly identifies that wireless service providers must submit a notification within 120 minutes of discovering a reportable outage",
        parent=main_node,
        critical=True
    )

    # Claim focusing on the presence and correctness of "120 minutes" in the answer
    claim_timeline = (
        "The answer explicitly states that wireless service providers must submit an initial notification to the "
        f"FCC within {EXPECTED_MINUTES} minutes (i.e., within two hours) of discovering a reportable Network Outage under NORS."
    )
    await evaluator.verify(
        claim=claim_timeline,
        node=node_timeline,
        additional_instruction=(
            "Judge only whether the answer text itself asserts the correct timeline. "
            "Treat phrasing such as 'within 120 minutes', 'within two hours', or 'within 2 hours' as equivalent. "
            "Minor wording variations are acceptable as long as the requirement is clearly the initial notification "
            "to the Commission after discovering a reportable outage under NORS (for wireless providers)."
        ),
    )

    # Child 2: Verify the provided reference URL(s) are authoritative and support the 120-minute timeline
    node_reference = evaluator.add_leaf(
        id="reference_url",
        desc="The answer provides a valid reference URL from an authoritative source (FCC website or official federal regulation) that supports the 120-minute notification timeline",
        parent=main_node,
        critical=True
    )

    # Filter URLs to authoritative ones; pass all authoritative URLs to multi-URL verifier
    authoritative_urls = filter_authoritative_urls(extraction.urls)

    claim_reference = (
        "Wireless service providers must submit the initial NORS notification within 120 minutes of discovering a "
        "reportable outage, and this requirement is confirmed by an authoritative FCC or official federal regulation source."
    )
    await evaluator.verify(
        claim=claim_reference,
        node=node_reference,
        sources=authoritative_urls,
        additional_instruction=(
            "Only consider the claim supported if the page is authoritative: "
            "FCC domains (including docs.fcc.gov) or ecfr.gov (eCFR Title 47 Part 4) or federalregister.gov. "
            "Explicitly look for language that the initial NORS notification must be submitted within 120 minutes "
            "(two hours) of discovering a reportable outage by wireless service providers. "
            "If the URL is non-authoritative or the content does not clearly support the 120-minute timeline, "
            "judge it as not supported."
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
    Evaluate an answer for the FCC NORS initial notification timeline task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root wrapper; actual critical aggregation is under main node
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_nors_initial_notification(),
        template_class=NORSInitialNotificationExtraction,
        extraction_name="nors_initial_notification"
    )

    # Add ground truth information
    evaluator.add_ground_truth({
        "expected_minutes": EXPECTED_MINUTES,
        "requirement": "Initial notification must be submitted within 120 minutes after discovering a reportable outage under NORS",
        "authoritative_sources_examples": [
            "FCC Part 4 outage reporting (fcc.gov, docs.fcc.gov)",
            "eCFR Title 47 Part 4 (ecfr.gov)"
        ]
    })

    # Build and run verification
    await build_and_verify_tree(evaluator, root, extraction)

    # Return structured result
    return evaluator.get_summary()