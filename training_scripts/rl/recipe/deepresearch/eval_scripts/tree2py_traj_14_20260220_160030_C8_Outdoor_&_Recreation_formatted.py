import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task Constants
# -----------------------------------------------------------------------------
TASK_ID = "az_state_parks_accessible_rv_full_hookups"
TASK_DESCRIPTION = (
    "Identify 4 Arizona State Parks that offer campgrounds with both full hookups "
    "(water, electric, and sewer connections at the campsite) and accessible facilities "
    "suitable for RV campers using wheelchairs, where the electrical service includes "
    "50 amp connections and accessible restroom/shower facilities are available in the campground."
)


# -----------------------------------------------------------------------------
# Data Models for Extraction
# -----------------------------------------------------------------------------
class ParkItem(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ParksExtraction(BaseModel):
    parks: List[ParkItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_parks() -> str:
    return """
Extract up to all parks the answer claims meet the following criteria:
- The park is an Arizona State Park in the state of Arizona.
- The park's campground has ADA-compliant accessible campsites suitable for RV campers using wheelchairs.
- The park offers campsites with full hookups (water, electric, and sewer connections at the campsite).
- The park provides 50-amp electrical service at some campsites.
- The park has accessible restroom and shower facilities within the campground.

Return a JSON object with:
- parks: an array of objects. Each object must include:
  - name: the park name exactly as it appears in the answer (or null if not provided).
  - sources: all URLs explicitly cited in the answer that support the claims for this park.
    Include official Arizona State Parks pages (e.g., azstateparks.com) and any other URLs cited for this park.
    Return an empty list if none are explicitly provided.

Rules:
- Only extract URLs that are explicitly present in the answer (including within markdown links).
- Do not fabricate or infer URLs.
- If more than 4 parks are mentioned, include them all; the evaluator will only use the first 4.
- If fewer than 4 are mentioned, include whatever is present.
    """


# -----------------------------------------------------------------------------
# Verification Helpers
# -----------------------------------------------------------------------------
def _normalize_sources_list(lst: Optional[List[str]]) -> List[str]:
    if not lst:
        return []
    # Remove duplicates and strip whitespace
    seen = set()
    out = []
    for url in lst:
        if not isinstance(url, str):
            continue
        u = url.strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


async def verify_one_park(
    evaluator: Evaluator,
    parent_node,
    park_index: int,
    park: ParkItem,
) -> None:
    """
    Build the subtree for a single park and launch verifications.

    Structure:
      Park_{i} (parallel, critical children)
        ├─ Park_{i}_Required_Info (custom critical)  -> name present and at least 1 URL
        ├─ Park_{i}_Is_Arizona_State_Park (leaf, critical) -> verify via URLs
        ├─ Park_{i}_Has_Accessible_Campsites (leaf, critical) -> verify via URLs
        ├─ Park_{i}_Has_Full_Hookups (leaf, critical) -> verify via URLs
        ├─ Park_{i}_Has_50_Amp_Electric (leaf, critical) -> verify via URLs
        └─ Park_{i}_Has_Accessible_Restroom_Shower (leaf, critical) -> verify via URLs

    Notes:
    - We add a 'required info' critical node to enforce source-grounded verification.
    - Other critical leaves will auto-skip if the required info node fails (due to the framework's auto preconditions on critical siblings).
    """
    # Create the park container node
    park_node = evaluator.add_parallel(
        id=f"Park_{park_index}",
        desc=[
            "First Arizona State Park meeting all criteria",
            "Second Arizona State Park meeting all criteria",
            "Third Arizona State Park meeting all criteria",
            "Fourth Arizona State Park meeting all criteria",
        ][park_index - 1] if 1 <= park_index <= 4 else f"Arizona State Park #{park_index} meeting all criteria",
        parent=parent_node,
        critical=True  # Each park must meet all criteria if we want overall pass
    )

    name = park.name or ""
    sources = _normalize_sources_list(park.sources)

    # Critical precondition: name and at least one source URL present
    evaluator.add_custom_node(
        result=bool(name.strip()) and len(sources) > 0,
        id=f"Park_{park_index}_Required_Info",
        desc=f"Park #{park_index} has required info (park name and at least one supporting URL)",
        parent=park_node,
        critical=True
    )

    # Leaf 1: Is Arizona State Park
    node_state_park = evaluator.add_leaf(
        id=f"Park_{park_index}_Is_Arizona_State_Park",
        desc="The park is an Arizona State Park (publicly operated recreation facility in Arizona)",
        parent=park_node,
        critical=True
    )
    claim_state_park = (
        f"'{name}' is an official Arizona State Park within the state of Arizona "
        f"(i.e., part of the Arizona State Parks & Trails system)."
    )
    await evaluator.verify(
        claim=claim_state_park,
        node=node_state_park,
        sources=sources,
        additional_instruction=(
            "Look for evidence on azstateparks.com or equivalent authoritative sources "
            "explicitly indicating this is an Arizona State Park."
        )
    )

    # Leaf 2: Accessible campsites (ADA)
    node_accessible_sites = evaluator.add_leaf(
        id=f"Park_{park_index}_Has_Accessible_Campsites",
        desc="The park has ADA-compliant accessible campsites available",
        parent=park_node,
        critical=True
    )
    claim_accessible_sites = (
        f"The campground at '{name}' includes ADA-compliant accessible campsites suitable for RV campers using wheelchairs."
    )
    await evaluator.verify(
        claim=claim_accessible_sites,
        node=node_accessible_sites,
        sources=sources,
        additional_instruction=(
            "Accept phrasing such as 'accessible sites', 'ADA sites', 'wheelchair accessible campsites', "
            "or equivalent wording indicating ADA compliance or wheelchair accessibility."
        )
    )

    # Leaf 3: Full hookups (W/E/S at campsite)
    node_full_hookups = evaluator.add_leaf(
        id=f"Park_{park_index}_Has_Full_Hookups",
        desc="The park offers campsites with full hookups including water, electric, and sewer connections at the campsite",
        parent=park_node,
        critical=True
    )
    claim_full_hookups = (
        f"'{name}' offers campsites with full hookups at the site, including water, electric, and sewer connections."
    )
    await evaluator.verify(
        claim=claim_full_hookups,
        node=node_full_hookups,
        sources=sources,
        additional_instruction=(
            "Full hookups should explicitly include water, electric, and sewer at the individual campsite "
            "(not just a dump station). Accept common wording such as 'full hookups', 'W/E/S', or 'water/electric/sewer'."
        )
    )

    # Leaf 4: 50-amp electric
    node_50_amp = evaluator.add_leaf(
        id=f"Park_{park_index}_Has_50_Amp_Electric",
        desc="The park provides 50 amp electrical service at campsites",
        parent=park_node,
        critical=True
    )
    claim_50_amp = f"'{name}' has campsites that provide 50-amp electrical service (e.g., 50/30A pedestals)."
    await evaluator.verify(
        claim=claim_50_amp,
        node=node_50_amp,
        sources=sources,
        additional_instruction=(
            "Look for '50 amp', '50A', or combined '50/30 amp' service references at campsites. "
            "It's acceptable if only some sites offer 50A, as long as it is available."
        )
    )

    # Leaf 5: Accessible restroom/shower facilities in the campground
    node_restroom_shower = evaluator.add_leaf(
        id=f"Park_{park_index}_Has_Accessible_Restroom_Shower",
        desc="The park has accessible restroom and shower facilities available in the campground",
        parent=park_node,
        critical=True
    )
    claim_restroom_shower = (
        f"The campground at '{name}' has accessible restroom and shower facilities (ADA accessible)."
    )
    await evaluator.verify(
        claim=claim_restroom_shower,
        node=node_restroom_shower,
        sources=sources,
        additional_instruction=(
            "Accept explicit or clear wording that restroom and shower buildings in the campground have "
            "accessible features suitable for wheelchair users (e.g., ADA restrooms/showers)."
        )
    )


# -----------------------------------------------------------------------------
# Main Evaluation Entry Point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Arizona State Parks accessible RV + full hookups task.
    """
    # Initialize evaluator (root is always created as non-critical by the framework)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parks are independent items
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

    # Extract park items
    extracted = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction"
    )

    # Normalize and keep only first 4 parks (pad with empty if fewer)
    parks: List[ParkItem] = extracted.parks[:4] if extracted and extracted.parks else []
    while len(parks) < 4:
        parks.append(ParkItem())  # Placeholder to trigger failure on required info

    # Build per-park verification subtrees (Park_1 ... Park_4)
    # To adhere to rubric naming, we will use indices 1..4 for IDs and descriptions.
    for i in range(1, 5):
        await verify_one_park(
            evaluator=evaluator,
            parent_node=root,
            park_index=i,
            park=parks[i - 1]
        )

    # Add an overall gating node to simulate "Root critical" requirement:
    # Pass only if ALL 4 parks fully met all criteria (each park subtree aggregated score == 1.0).
    # This node is critical; if it fails, the root's aggregated score becomes 0.
    # We locate the four park container nodes by their IDs.
    park_node_results = {}
    for i in range(1, 5):
        node = evaluator.find_node(f"Park_{i}")
        # If a node is missing, treat as fail
        passed = bool(node and node.aggregated_score == 1.0)
        park_node_results[f"Park_{i}"] = "passed" if passed else "failed"

    evaluator.add_custom_node(
        result=all(status == "passed" for status in park_node_results.values()),
        id="All_Four_Parks_Valid",
        desc="All four parks fully meet the specified criteria (overall gating)",
        parent=root,
        critical=True
    )

    # Record a small custom info summary about which parks passed
    evaluator.add_custom_info(
        info={"per_park_pass_status": park_node_results},
        info_type="park_pass_status"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()