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
TASK_ID = "gaming_subscription_service_eval"
TASK_DESCRIPTION = (
    "Identify a gaming subscription service available in the United States that meets all of the following "
    "requirements: the subscription must cost $15 per month or less, include cloud gaming functionality, support "
    "streaming to mobile devices such as phones or tablets, provide online multiplayer features, offer a game library "
    "containing at least 200 games, support both console and PC platforms, and enable streaming to web browsers."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GamingServiceExtraction(BaseModel):
    service_name: Optional[str] = None
    monthly_price: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_service() -> str:
    return """
    Extract the single gaming subscription service proposed in the answer and any cited sources.
    Return the following fields:
    - service_name: the exact name of the gaming subscription service the answer recommends or settles on. 
      If multiple are mentioned, pick the one the answer ultimately recommends; otherwise pick the first one.
    - monthly_price: the stated monthly price for the recommended plan (as a string, keep currency symbols/units if present). If absent, return null.
    - source_urls: a list of all URLs explicitly cited in the answer that support information about this service 
      (accept plain URLs or markdown links; include all relevant links such as official pages, support docs, or news pages).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_service_name(name: Optional[str]) -> str:
    return name or "the service"


async def build_and_verify_service_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: GamingServiceExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run verifications using cited sources.
    """
    svc_name = _safe_service_name(extracted.service_name)
    sources = extracted.source_urls

    # Parent node (critical as per rubric)
    service_node = evaluator.add_parallel(
        id="Gaming_Subscription_Service",
        desc="Identify a gaming subscription service that meets all specified requirements",
        parent=parent_node,
        critical=True,
    )

    # Gating existence/sourcing check (critical)
    evaluator.add_custom_node(
        result=bool(extracted.service_name) and bool(sources),
        id="Service_Identified_And_Sources",
        desc="A specific gaming subscription service is identified and at least one supporting source URL is provided",
        parent=service_node,
        critical=True,
    )

    # Create leaf nodes for each rubric criterion (all critical)
    price_node = evaluator.add_leaf(
        id="Price_Requirement",
        desc="The subscription service costs $15 per month or less",
        parent=service_node,
        critical=True,
    )
    cloud_node = evaluator.add_leaf(
        id="Cloud_Gaming_Support",
        desc="The service includes cloud gaming functionality",
        parent=service_node,
        critical=True,
    )
    mobile_node = evaluator.add_leaf(
        id="Mobile_Streaming",
        desc="The service supports streaming to mobile devices (phones or tablets)",
        parent=service_node,
        critical=True,
    )
    online_node = evaluator.add_leaf(
        id="Online_Multiplayer",
        desc="The service includes online multiplayer features",
        parent=service_node,
        critical=True,
    )
    library_node = evaluator.add_leaf(
        id="Game_Library_Size",
        desc="The service offers a game library of at least 200 games",
        parent=service_node,
        critical=True,
    )
    platform_node = evaluator.add_leaf(
        id="Platform_Support",
        desc="The service supports both console and PC platforms",
        parent=service_node,
        critical=True,
    )
    browser_node = evaluator.add_leaf(
        id="Browser_Streaming",
        desc="The service supports streaming to web browsers",
        parent=service_node,
        critical=True,
    )
    us_node = evaluator.add_leaf(
        id="US_Availability",
        desc="The service is available in the United States",
        parent=service_node,
        critical=True,
    )

    # Prepare claims and run verifications in parallel
    claims_and_sources: List[tuple[str, List[str], Any, Optional[str]]] = []

    # Price requirement: threshold check (<= $15/month). If the tier for cloud gaming differs, verify the tier that includes cloud gaming.
    claims_and_sources.append((
        f"The monthly price for the gaming subscription service '{svc_name}' is 15 US dollars per month or less.",
        sources,
        price_node,
        "Verify the regular monthly price (not limited-time promotional discounts). If cloud gaming is available only on a specific tier, verify that tier's monthly price."
    ))

    # Cloud gaming functionality
    claims_and_sources.append((
        f"The gaming subscription service '{svc_name}' includes cloud gaming functionality (remote game streaming).",
        sources,
        cloud_node,
        "Confirm that games can be streamed from remote servers to the user's device; do not confuse this with cloud saves or remote downloads."
    ))

    # Mobile streaming
    claims_and_sources.append((
        f"The gaming subscription service '{svc_name}' supports streaming gameplay to mobile devices such as phones or tablets.",
        sources,
        mobile_node,
        "Look for official support for mobile gameplay via native apps or mobile web browsers."
    ))

    # Online multiplayer
    claims_and_sources.append((
        f"The gaming subscription service '{svc_name}' includes online multiplayer features for playing with or against others over the internet.",
        sources,
        online_node,
        "Accept evidence that online multiplayer is enabled/allowed as part of the subscription or the service ecosystem."
    ))

    # Library size >= 200 games
    claims_and_sources.append((
        f"The gaming subscription service '{svc_name}' offers a game library of at least 200 games.",
        sources,
        library_node,
        "Accept explicit statements like '200+', 'over 200', or a catalog size >= 200 available to play/stream."
    ))

    # Platform support: both console and PC
    claims_and_sources.append((
        f"The gaming subscription service '{svc_name}' supports both console and PC platforms.",
        sources,
        platform_node,
        "Confirm that the service can be used to play on a console platform (e.g., Xbox, PlayStation, Switch) AND on PC."
    ))

    # Browser streaming
    claims_and_sources.append((
        f"The gaming subscription service '{svc_name}' supports streaming gameplay in web browsers.",
        sources,
        browser_node,
        "Look for mentions of supported browsers (e.g., Chrome, Edge, Safari, Firefox) for gameplay streaming."
    ))

    # US availability
    claims_and_sources.append((
        f"The gaming subscription service '{svc_name}' is available in the United States.",
        sources,
        us_node,
        "Evidence can include country availability lists, region support pages, or official announcements for US availability."
    ))

    # Execute all verifications concurrently; each leaf will auto-skip if the gating node failed
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the gaming subscription service task and return a structured result dictionary.
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_service(),
        template_class=GamingServiceExtraction,
        extraction_name="service_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_service_tree(evaluator, root, extraction)

    # Return the final structured summary
    return evaluator.get_summary()