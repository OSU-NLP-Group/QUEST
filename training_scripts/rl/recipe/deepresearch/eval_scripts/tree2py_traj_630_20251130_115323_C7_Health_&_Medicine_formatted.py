import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pharmacy_flu_vaccination_services_comparison"
TASK_DESCRIPTION = (
    "Compare flu vaccination services across Walgreens, CVS, Walmart, and Kroger, reporting minimum age, "
    "walk-in acceptance, and online scheduling, with reference URLs supporting each required data point, "
    "based on current 2025–2026 flu season policies."
)

# Optional rubric expectations (used for guidance in verification instructions; not hard ground truth)
EXPECTED_BY_RUBRIC: Dict[str, Dict[str, Optional[str | bool]]] = {
    "Walgreens": {
        "min_age_note": "3 years and older",
        "walk_in_expected": True,
        "online_sched_expected": True,
    },
    "CVS": {
        "min_age_note": "3 years or older (state variation may apply)",
        "walk_in_expected": True,
        "online_sched_expected": True,
    },
    "Walmart": {
        "min_age_note": "3 years and older (in most states)",
        "walk_in_expected": True,
        "online_sched_expected": True,
    },
    "Kroger": {
        "min_age_note": "3 years and older",
        "walk_in_expected": None,  # Not pre-fixed to an outcome per rubric (must provide clear yes/no)
        "online_sched_expected": True,
    },
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ChainPolicyItem(BaseModel):
    chain_name: Optional[str] = None

    # Season context evidence: URLs that indicate the content is current for 2025–2026 season
    season_context_urls: List[str] = Field(default_factory=list)

    # Minimum age requirement and supporting URLs
    min_age_text: Optional[str] = None
    min_age_urls: List[str] = Field(default_factory=list)

    # Walk-in acceptance statement and supporting URLs
    walk_in_text: Optional[str] = None
    walk_in_urls: List[str] = Field(default_factory=list)

    # Online scheduling availability statement and supporting URLs
    online_sched_text: Optional[str] = None
    online_sched_urls: List[str] = Field(default_factory=list)


class FluPoliciesExtraction(BaseModel):
    walgreens: Optional[ChainPolicyItem] = None
    cvs: Optional[ChainPolicyItem] = None
    walmart: Optional[ChainPolicyItem] = None
    kroger: Optional[ChainPolicyItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_policies() -> str:
    return (
        "Extract flu-shot service information for Walgreens, CVS, Walmart, and Kroger exactly as presented in the answer. "
        "For each chain, return a JSON object with these fields:\n"
        "- chain_name: The pharmacy chain name (Walgreens, CVS, Walmart, Kroger).\n"
        "- season_context_urls: An array of URLs that explicitly indicate the information is current for the 2025–2026 flu season, "
        "or otherwise explicitly labeled as current for that season. If no such URLs are provided, return an empty array.\n"
        "- min_age_text: The stated minimum age requirement text for flu vaccinations at pharmacy locations (string). If missing, return null.\n"
        "- min_age_urls: An array of supporting URLs for the minimum age requirement. Extract only URLs explicitly provided in the answer; if none, return an empty array.\n"
        "- walk_in_text: The stated walk-in acceptance policy text (e.g., 'walk-ins accepted', 'no appointment needed', or 'appointment required'). If missing, return null.\n"
        "- walk_in_urls: An array of supporting URLs for the walk-in policy. If none, return an empty array.\n"
        "- online_sched_text: The stated online appointment scheduling availability text (e.g., 'online scheduling available'). If missing, return null.\n"
        "- online_sched_urls: An array of supporting URLs for the online scheduling statement. If none, return an empty array.\n\n"
        "Return the result as a JSON object with keys: walgreens, cvs, walmart, kroger. For any chain not mentioned in the answer, set its value to null. "
        "Extract URLs exactly as they appear (plain URL or markdown link). Do not invent or infer URLs."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _slug(chain_name: str) -> str:
    return chain_name.lower().replace(" ", "_")


def _has_valid_url(urls: List[str]) -> bool:
    for u in urls:
        if isinstance(u, str) and ("http://" in u or "https://" in u):
            return True
    return False


def _collect_fallback_sources(item: ChainPolicyItem) -> List[str]:
    # Combine all available URLs for use as fallback sources if a specific bucket is empty
    urls = []
    urls.extend(item.season_context_urls or [])
    urls.extend(item.min_age_urls or [])
    urls.extend(item.walk_in_urls or [])
    urls.extend(item.online_sched_urls or [])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def _season_instruction(chain: str) -> str:
    return (
        f"Confirm that the cited page(s) for {chain} explicitly indicate the 2025–2026 flu season. "
        "Accept synonyms like '2025-2026', '2025/2026', or clearly labeled content indicating the page is current for the 2025–2026 season. "
        "If the page is clearly general but explicitly states it is updated or current for the 2025–2026 season, that is acceptable. "
        "If none of the provided pages indicate season context, mark as not supported."
    )


def _min_age_instruction(chain: str) -> str:
    expect_note = EXPECTED_BY_RUBRIC.get(chain, {}).get("min_age_note")
    base = (
        "Verify the statement about the minimum age for receiving a flu vaccination at in-store pharmacy locations using the provided URLs. "
        "Focus on flu shots and pharmacy-administered vaccines (not urgent care or third-party clinics). "
        "Allow minor wording variations (e.g., 'age 3+' vs. '3 years and older'). "
    )
    if isinstance(expect_note, str):
        base += (
            f"For this evaluation, the rubric expects the policy to align with '{expect_note}'. "
            "If the cited page clearly indicates a different minimum age for the current 2025–2026 season, judge based on the page content. "
            "If the page is ambiguous or silent, consider the claim unsupported."
        )
    return base


def _walk_in_instruction(chain: str) -> str:
    expected = EXPECTED_BY_RUBRIC.get(chain, {}).get("walk_in_expected")
    base = (
        "Verify whether the chain accepts walk-ins for flu shots (i.e., no prior scheduling required). "
        "Accept equivalent phrases such as 'walk-ins welcome', 'no appointment needed', 'stop by today', etc. "
        "If the cited page states the opposite (appointments required, walk-ins not accepted), mark accordingly."
    )
    if expected is True:
        base += " The rubric expects this chain to accept walk-ins; if the page states otherwise, mark as not supported."
    elif expected is False:
        base += " The rubric expects this chain to require appointments; if the page states walk-ins are accepted, mark as not supported."
    else:
        base += " The rubric does not pre-fix the outcome for this chain; judge strictly based on the cited page(s)."
    return base


def _online_sched_instruction(chain: str) -> str:
    expected = EXPECTED_BY_RUBRIC.get(chain, {}).get("online_sched_expected")
    base = (
        "Verify whether online appointment scheduling for flu shots is available on the chain's website. "
        "Accept equivalent phrasing such as 'schedule online', 'book online', 'online appointments'."
    )
    if expected is True:
        base += " The rubric expects online scheduling to be offered; if the page indicates otherwise, mark as not supported."
    elif expected is False:
        base += " The rubric expects online scheduling to NOT be offered; if the page indicates availability, mark as not supported."
    else:
        base += " Judge strictly based on the cited page(s)."
    return base


def _build_walk_in_claim(chain: str, walk_in_text: Optional[str]) -> str:
    # Build a clear yes/no claim from the provided text (fallback to positive phrasing if unclear)
    text = (walk_in_text or "").lower()
    negatives = ["appointment required", "appointments required", "appointment-only", "appointment only", "no walk-ins", "walk-ins not accepted"]
    positives = ["walk-ins", "walk in", "walk-ins accepted", "walk-ins welcome", "no appointment needed", "no appointment required"]
    if any(n in text for n in negatives) and not any(p in text for p in positives):
        return f"{chain} does not accept walk-ins for flu shots; a prior appointment is required."
    if any(p in text for p in positives):
        return f"{chain} accepts walk-ins for flu shots; no prior scheduling is required."
    # If ambiguous or missing, phrase neutral but evaluable claim leaning positive to match common practice (rubric expects yes for most chains)
    return f"{chain} accepts walk-ins for flu shots."
    

def _build_online_sched_claim(chain: str, online_text: Optional[str]) -> str:
    text = (online_text or "").lower()
    positives = ["online scheduling", "book online", "schedule online", "online appointment", "appointments online"]
    negatives = ["no online scheduling", "cannot schedule online"]
    if any(n in text for n in negatives) and not any(p in text for p in positives):
        return f"{chain} does not offer online appointment scheduling for flu shots."
    return f"{chain} offers online appointment scheduling for flu shots."


# --------------------------------------------------------------------------- #
# Verification for one pharmacy                                               #
# --------------------------------------------------------------------------- #
async def verify_pharmacy(
    evaluator: Evaluator,
    parent_node,
    chain_name: str,
    item: Optional[ChainPolicyItem],
) -> None:
    """
    Build verification nodes and execute checks for a single pharmacy chain.
    """
    chain_desc = f"{chain_name} flu-shot service info and supporting URLs"
    chain_node = evaluator.add_parallel(
        id=_slug(chain_name),
        desc=chain_desc,
        parent=parent_node,
        critical=False,  # allow partial-credit across chains
    )

    # Normalize item
    safe_item = item or ChainPolicyItem(chain_name=chain_name)

    # Season policy context (critical)
    season_node = evaluator.add_leaf(
        id=f"{_slug(chain_name)}_season_policy_context",
        desc=f"Evidence/wording indicates the provided {chain_name} flu-shot policies are current for the 2025–2026 flu season",
        parent=chain_node,
        critical=True,
    )
    season_sources = safe_item.season_context_urls or _collect_fallback_sources(safe_item) or None
    season_claim = (
        f"For {chain_name}, the cited sources explicitly indicate the policies are current for the 2025–2026 flu season "
        f"(e.g., '2025–2026', '2025-2026', '2025/2026', or explicitly labeled current for that season)."
    )
    await evaluator.verify(
        claim=season_claim,
        node=season_node,
        sources=season_sources,
        additional_instruction=_season_instruction(chain_name),
    )

    # Minimum age value check (critical)
    min_age_node = evaluator.add_leaf(
        id=f"{_slug(chain_name)}_minimum_age",
        desc=f"States {chain_name} minimum age requirement for pharmacy flu shots",
        parent=chain_node,
        critical=True,
    )
    # Build claim preferentially from rubric expectation but verify strictly via URLs
    expected_note = EXPECTED_BY_RUBRIC.get(chain_name, {}).get("min_age_note")
    if isinstance(expected_note, str):
        min_age_claim = f"At {chain_name}, the minimum age for receiving a flu vaccination at pharmacy locations is {expected_note}."
    else:
        # fallback to the extracted text, if provided
        if safe_item.min_age_text:
            min_age_claim = f"At {chain_name}, the minimum age for receiving a flu vaccination at pharmacy locations is: {safe_item.min_age_text}."
        else:
            min_age_claim = f"{chain_name} has a stated minimum age requirement for pharmacy flu shots."
    min_age_sources = safe_item.min_age_urls or _collect_fallback_sources(safe_item) or None
    await evaluator.verify(
        claim=min_age_claim,
        node=min_age_node,
        sources=min_age_sources,
        additional_instruction=_min_age_instruction(chain_name),
    )

    # Minimum age URL existence (critical)
    min_age_url_exists = evaluator.add_custom_node(
        result=_has_valid_url(safe_item.min_age_urls),
        id=f"{_slug(chain_name)}_minimum_age_url",
        desc=f"Provides a reference URL supporting {chain_name} minimum age requirement",
        parent=chain_node,
        critical=True,
    )

    # Walk-in acceptance policy (critical)
    walk_in_node = evaluator.add_leaf(
        id=f"{_slug(chain_name)}_walk_in_policy",
        desc=f"States whether {chain_name} accepts walk-ins (no prior scheduling required)",
        parent=chain_node,
        critical=True,
    )
    walk_in_claim = _build_walk_in_claim(chain_name, safe_item.walk_in_text)
    walk_in_sources = safe_item.walk_in_urls or _collect_fallback_sources(safe_item) or None
    await evaluator.verify(
        claim=walk_in_claim,
        node=walk_in_node,
        sources=walk_in_sources,
        additional_instruction=_walk_in_instruction(chain_name),
    )

    # Walk-in URL existence (critical)
    walk_in_url_exists = evaluator.add_custom_node(
        result=_has_valid_url(safe_item.walk_in_urls),
        id=f"{_slug(chain_name)}_walk_in_policy_url",
        desc=f"Provides a reference URL supporting {chain_name} walk-in policy",
        parent=chain_node,
        critical=True,
    )

    # Online scheduling availability (critical)
    online_sched_node = evaluator.add_leaf(
        id=f"{_slug(chain_name)}_online_scheduling",
        desc=f"States that {chain_name} offers online appointment scheduling for flu shots",
        parent=chain_node,
        critical=True,
    )
    online_sched_claim = _build_online_sched_claim(chain_name, safe_item.online_sched_text)
    online_sched_sources = safe_item.online_sched_urls or _collect_fallback_sources(safe_item) or None
    await evaluator.verify(
        claim=online_sched_claim,
        node=online_sched_node,
        sources=online_sched_sources,
        additional_instruction=_online_sched_instruction(chain_name),
    )

    # Online scheduling URL existence (critical)
    online_sched_url_exists = evaluator.add_custom_node(
        result=_has_valid_url(safe_item.online_sched_urls),
        id=f"{_slug(chain_name)}_online_scheduling_url",
        desc=f"Provides a reference URL supporting {chain_name} online scheduling availability",
        parent=chain_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the pharmacy flu-shot services comparison task.
    """
    # Initialize evaluator (root node is non-critical by framework design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # pharmacies evaluated independently
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

    # Record rubric expectation for transparency (optional, not used for scoring directly)
    evaluator.add_ground_truth({"expected_by_rubric": EXPECTED_BY_RUBRIC}, gt_type="rubric_expectations")

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_policies(),
        template_class=FluPoliciesExtraction,
        extraction_name="flu_policies_extraction",
    )

    # Build four pharmacy subtrees under the root
    await verify_pharmacy(evaluator, root, "Walgreens", extraction.walgreens)
    await verify_pharmacy(evaluator, root, "CVS", extraction.cvs)
    await verify_pharmacy(evaluator, root, "Walmart", extraction.walmart)
    await verify_pharmacy(evaluator, root, "Kroger", extraction.kroger)

    # Return standardized summary
    return evaluator.get_summary()