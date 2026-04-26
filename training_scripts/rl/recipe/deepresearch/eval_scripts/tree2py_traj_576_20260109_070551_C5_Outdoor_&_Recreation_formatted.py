import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wa_backpacking_wilderness_2026"
TASK_DESCRIPTION = (
    "I'm planning a backpacking trip in Washington State for July 2026 with a group of 9 people. "
    "Identify 4 wilderness areas where we can go backpacking that meet these requirements: "
    "(1) Do NOT require entering a permit lottery or advance reservation system, "
    "(2) Have free self-issued wilderness permits (no permit fees), and "
    "(3) Allow groups of 9 or more people. "
    "For each wilderness area, provide its official name and one official reference URL from a government agency website "
    "(such as U.S. Forest Service or Washington State Parks)."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WildernessArea(BaseModel):
    official_name: Optional[str] = None
    gov_url: Optional[str] = None


class AreasExtraction(BaseModel):
    areas: List[WildernessArea] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_wilderness_areas() -> str:
    return (
        "Extract up to four wilderness areas listed in the answer (in the order they appear). "
        "For each area, extract:\n"
        "1) official_name: the official wilderness area name exactly as written in the answer.\n"
        "2) gov_url: one official reference URL from a government agency website (e.g., fs.usda.gov, nps.gov, blm.gov, parks.wa.gov, dnr.wa.gov). "
        "   If multiple URLs are mentioned for that area, prefer a government domain. "
        "   If the answer does not provide a URL for the area, return null for gov_url.\n"
        "Return a JSON object with an 'areas' array containing up to four objects with these fields. "
        "Do not invent or infer any names or URLs not present in the answer. "
        "If more than four areas are mentioned, include only the first four. "
        "If fewer than four are mentioned, include only those provided."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _is_government_domain(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    allowed_suffixes = (
        ".gov",
        ".mil",
        "fs.usda.gov",  # USFS
        "usda.gov",     # USDA parent
    )
    # Common acceptable WA state government domains
    wa_gov_domains = (
        "parks.wa.gov",
        "dnr.wa.gov",
        "ecology.wa.gov",
        "wdfw.wa.gov",
        "wa.gov",  # catch-all WA government
    )
    if netloc.endswith(allowed_suffixes):
        return True
    if netloc in wa_gov_domains or netloc.endswith(".wa.gov"):
        return True
    # Other federal agency domains that may host official info
    federal_whitelist = ("nps.gov", "blm.gov", "usgs.gov", "doi.gov", "recreation.gov")
    if any(netloc.endswith(d) for d in federal_whitelist):
        return True
    return False


# --------------------------------------------------------------------------- #
# Verification for one area                                                   #
# --------------------------------------------------------------------------- #
async def verify_area(
    evaluator: Evaluator,
    parent_node,
    area: WildernessArea,
    area_index: int,
) -> None:
    """
    Build verification subtree for one wilderness area.
    """
    idx = area_index + 1
    name = area.official_name or ""
    url = area.gov_url or ""

    area_node = evaluator.add_parallel(
        id=f"Area_{idx}",
        desc=f"Evaluate the {idx}th wilderness area",
        parent=parent_node,
        critical=False  # allow partial credit per area
    )

    # Official Name provided (existence)
    name_exists = bool(name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id=f"Area_{idx}_Official_Name",
        desc="Provides the wilderness area's official name",
        parent=area_node,
        critical=True
    )

    # Government URL provided (existence) - auxiliary gating to avoid null URL verification
    url_exists_node = evaluator.add_custom_node(
        result=bool(url.strip()),
        id=f"Area_{idx}_Gov_URL_Provided",
        desc="Government reference URL is provided",
        parent=area_node,
        critical=True
    )

    # Located in Washington State
    located_node = evaluator.add_leaf(
        id=f"Area_{idx}_Located_In_WA",
        desc="Wilderness area is located in Washington State",
        parent=area_node,
        critical=True
    )
    claim_loc = f"The wilderness area named '{name}' is located in Washington State."
    await evaluator.verify(
        claim=claim_loc,
        node=located_node,
        sources=url if url_exists_node.score == 1.0 else None,
        additional_instruction=(
            "Verify using the official page whether the wilderness is in Washington State. "
            "If the page indicates the wilderness lies wholly or partially in Washington, it should be considered correct. "
            "If the page indicates it is in a different state, mark incorrect."
        ),
    )

    # Does NOT require entering a permit lottery
    no_lottery_node = evaluator.add_leaf(
        id=f"Area_{idx}_No_Permit_Lottery",
        desc="Does NOT require entering a permit lottery",
        parent=area_node,
        critical=True
    )
    claim_lottery = (
        f"The wilderness area '{name}' does not require entering any advance permit lottery for general "
        "overnight/backpacking entry."
    )
    await evaluator.verify(
        claim=claim_lottery,
        node=no_lottery_node,
        sources=url if url_exists_node.score == 1.0 else None,
        additional_instruction=(
            "Judge based on the official page. Consider the general wilderness entry policy: "
            "if the page does not require lotteries and indicates permits are simply self-issued or no reservation, "
            "this supports the claim. "
            "Note: Some specific subzones may have quotas/reservations; such exceptions do not negate the general wilderness "
            "policy unless the entire wilderness requires a lottery."
        ),
    )

    # Does NOT require an advance reservation system
    no_res_node = evaluator.add_leaf(
        id=f"Area_{idx}_No_Advance_Reservations",
        desc="Does NOT require an advance reservation system",
        parent=area_node,
        critical=True
    )
    claim_res = (
        f"The wilderness area '{name}' does not require an advance reservation system for general "
        "overnight/backpacking entry."
    )
    await evaluator.verify(
        claim=claim_res,
        node=no_res_node,
        sources=url if url_exists_node.score == 1.0 else None,
        additional_instruction=(
            "Use the official page to check whether advance reservations are required for general wilderness entry. "
            "If permits are self-issued at trailheads or ranger stations or otherwise not reservable in advance, "
            "mark as supported. Specific subzones with reservations do not negate the general wilderness policy."
        ),
    )

    # Has free self-issued wilderness permits (no permit fees)
    free_self_node = evaluator.add_leaf(
        id=f"Area_{idx}_Free_Self_Issued_No_Fee",
        desc="Has free self-issued wilderness permits (no permit fees)",
        parent=area_node,
        critical=True
    )
    claim_free = (
        f"Wilderness permits for '{name}' are free (no fee) and self-issued (e.g., at trailheads or ranger stations)."
    )
    await evaluator.verify(
        claim=claim_free,
        node=free_self_node,
        sources=url if url_exists_node.score == 1.0 else None,
        additional_instruction=(
            "Confirm that the official page describes wilderness permits as 'free' and 'self-issued' or similar language. "
            "If any fee is required or permits are not self-issued, mark incorrect."
        ),
    )

    # Allows groups of 9 or more
    group_node = evaluator.add_leaf(
        id=f"Area_{idx}_Group_Size_At_Least_9",
        desc="Allows groups of 9 or more people",
        parent=area_node,
        critical=True
    )
    claim_group = (
        f"The posted maximum group size for the wilderness area '{name}' is at least 9 (e.g., 9, 10, 12), "
        "meaning a group of nine is permitted."
    )
    await evaluator.verify(
        claim=claim_group,
        node=group_node,
        sources=url if url_exists_node.score == 1.0 else None,
        additional_instruction=(
            "Check the official page for the maximum group size or party size. "
            "If the maximum is ≥9 (e.g., 10 or 12), the claim is supported. "
            "If the maximum is <9 (e.g., 8), mark incorrect."
        ),
    )

    # Provides one official reference URL from a government agency website for this area
    gov_ref_node = evaluator.add_leaf(
        id=f"Area_{idx}_Gov_Reference_URL",
        desc="Provides one official reference URL from a government agency website for this area",
        parent=area_node,
        critical=True
    )
    # If URL is missing or not government domain, we want this to fail.
    # We'll use the page to help judge government agency provenance.
    claim_gov = (
        f"The provided URL is an official government agency page that provides authoritative information about the "
        f"'{name}' Wilderness (e.g., US Forest Service, National Park Service, BLM, or Washington State government)."
    )
    # If url missing, pass None; we also add instruction to treat missing as incorrect
    await evaluator.verify(
        claim=claim_gov,
        node=gov_ref_node,
        sources=url if url_exists_node.score == 1.0 else None,
        additional_instruction=(
            "Determine if the page is operated by a government agency (domains like fs.usda.gov, nps.gov, blm.gov, parks.wa.gov, dnr.wa.gov, *.wa.gov). "
            "If the URL is missing, or the site is clearly non-government (e.g., .com, .org not official), mark incorrect."
        ),
    )

    # Additional: July 2026 context (non-critical; simple verify against the answer context)
    july_node = evaluator.add_leaf(
        id=f"Area_{idx}_July_2026_Context",
        desc="Information is presented as applicable for July travel (or notes any seasonal/dated caveats relevant to July 2026)",
        parent=area_node,
        critical=False
    )
    claim_july = (
        "The answer frames the information for July travel (or mentions July 2026) or includes any seasonal caveats relevant to July."
    )
    await evaluator.verify(
        claim=claim_july,
        node=july_node,
        sources=None,
        additional_instruction=(
            "Look for references to 'July', 'summer conditions', 'seasonal access', or 'July 2026' within the answer context. "
            "This check focuses on the answer framing, not the external page."
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
    Evaluate the agent's answer for the Washington State wilderness backpacking task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level criteria evaluated independently
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

    # Extract up to four wilderness areas from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_wilderness_areas(),
        template_class=AreasExtraction,
        extraction_name="wilderness_areas",
    )

    # Keep only first four
    areas = (extracted.areas or [])[:4]

    # Root-level critical check: exactly 4 distinct wilderness areas
    names_norm = [_normalize_name(a.official_name) for a in areas]
    distinct_names = len(set(n for n in names_norm if n)) == len(names_norm) and len(areas) == 4
    evaluator.add_custom_node(
        result=distinct_names,
        id="Provide_4_Distinct_Areas",
        desc="Response provides exactly 4 distinct wilderness areas (not duplicates)",
        parent=root,
        critical=True
    )

    # Optional custom info: URL domain government-ness per area
    gov_domain_stats: List[Dict[str, Any]] = []
    for i, a in enumerate(areas):
        gov_domain_stats.append({
            "index": i + 1,
            "name": a.official_name,
            "url": a.gov_url,
            "is_government_domain": _is_government_domain(a.gov_url)
        })
    evaluator.add_custom_info({"gov_domain_stats": gov_domain_stats}, info_type="auxiliary")

    # Build and verify each area subtree
    for i in range(len(areas)):
        await verify_area(evaluator, root, areas[i], i)

    # If fewer than 4 areas provided, still create placeholder nodes to reflect missing entries
    # (keeps tree shape predictable)
    for i in range(len(areas), 4):
        placeholder = WildernessArea(official_name=None, gov_url=None)
        await verify_area(evaluator, root, placeholder, i)

    # Return evaluation summary
    return evaluator.get_summary()