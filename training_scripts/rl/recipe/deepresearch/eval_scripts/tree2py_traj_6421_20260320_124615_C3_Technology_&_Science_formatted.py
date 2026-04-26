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
TASK_ID = "chips_phoenix_fab"
TASK_DESCRIPTION = (
    "Which semiconductor fabrication facility located within Phoenix, Arizona city limits is a recipient of at least "
    "$6 billion in direct funding under the CHIPS and Science Act, produces leading-edge semiconductors at process "
    "nodes of 14 nanometers or below on 300mm wafers, consumes between 4 and 5 million gallons of water per day in "
    "its first operational phase, and achieved mass production of 4-nanometer process technology by the first half of 2025?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityExtraction(BaseModel):
    """
    Structured extraction of the facility identity and the per-criterion URL sources
    explicitly cited in the answer text.
    """
    facility_name: Optional[str] = None
    facility_aliases: List[str] = Field(default_factory=list)

    # Per-criterion URL buckets (only URLs explicitly present in the answer)
    location_urls: List[str] = Field(default_factory=list)
    funding_urls: List[str] = Field(default_factory=list)  # CHIPS Act direct funding
    leading_edge_node_urls: List[str] = Field(default_factory=list)  # <= 14 nm processes
    wafer_size_urls: List[str] = Field(default_factory=list)  # 300mm/12-inch wafer manufacturing
    water_use_urls: List[str] = Field(default_factory=list)  # 4–5M GPD in first operational phase
    mass_prod_4nm_urls: List[str] = Field(default_factory=list)  # 4 nm mass production by H1 2025
    campus_size_urls: List[str] = Field(default_factory=list)  # >= 1,000 acres campus
    cleanroom_size_urls: List[str] = Field(default_factory=list)  # > 1.5M sq ft (~140,000 m2)
    investment_urls: List[str] = Field(default_factory=list)  # total planned investment > $60B

    # If the answer provides a shared/bulk sources section not mapped to any single criterion
    shared_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_sources() -> str:
    return """
    Extract the semiconductor fabrication facility mentioned in the answer and collect the specific URLs (explicitly
    present in the answer) that support each of the following verification criteria.

    Required JSON fields:
    - facility_name: string or null
    - facility_aliases: array of strings (other names, project/site identifiers, abbreviations) or empty array

    For each criterion below, return an array of URLs explicitly cited in the answer text that could support it.
    Only include URLs that are actually present in the answer (plain URLs or markdown links). Do not invent or infer.

    - location_urls: URLs supporting that the facility is located within Phoenix, Arizona city limits
    - funding_urls: URLs supporting that the facility received at least $6 billion in direct funding under the CHIPS and Science Act
    - leading_edge_node_urls: URLs supporting that the facility produces leading-edge semiconductors at <= 14nm nodes
    - wafer_size_urls: URLs supporting that the facility manufactures on 300mm (12-inch) wafers
    - water_use_urls: URLs supporting that the first operational phase uses between 4–5 million gallons of water per day
    - mass_prod_4nm_urls: URLs supporting that the facility achieved mass production of 4nm process technology by H1 2025
    - campus_size_urls: URLs supporting that the campus spans at least 1,000 acres
    - cleanroom_size_urls: URLs supporting that the cleanroom area exceeds 1.5 million sq ft (~140,000 m²)
    - investment_urls: URLs supporting that total planned investment across all phases exceeds $60 billion

    - shared_urls: If the answer gives a general sources section (not mapped per-criterion), put those URLs here.

    Rules:
    - Deduplicate URLs within each array.
    - Only return URLs explicitly present in the answer. If nothing is available for a criterion, return an empty array.
    - Do not include non-URL citations (e.g., "per news reports") unless a concrete URL is provided.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: Optional[List[str]]) -> List[str]:
    """Combine multiple URL lists, preserve order, and deduplicate."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if u not in seen:
                combined.append(u)
                seen.add(u)
    return combined


async def _add_verified_leaf(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    parent=None,
    additional_instruction: str = "None",
) -> bool:
    """
    Create a critical leaf node and perform verification. If no URLs are provided,
    enforce source grounding by auto-failing the node without LLM verification.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=True,
    )

    if not urls:
        # Enforce source-grounding: no URLs → auto-fail this factual check
        leaf.score = 0.0
        leaf.status = "failed"
        evaluator.add_custom_info(
            info={
                "criterion_id": node_id,
                "reason": "no_sources_provided",
                "message": "No URLs were provided in the answer for this verification; auto-failing to enforce source grounding."
            },
            info_type="missing_sources",
        )
        return False

    # Delegate to LLM-as-a-judge with URL evidence
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=additional_instruction
    )
    return leaf.status == "passed"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_facility_constraints(
    evaluator: Evaluator,
    parent,
    extracted: FacilityExtraction,
) -> None:
    """
    Build and execute all rubric-specified verification leaves directly under the root (parallel, all critical).
    """
    facility_name = (extracted.facility_name or "").strip()

    # 1) Facility identification (existence check)
    evaluator.add_custom_node(
        result=bool(facility_name),
        id="facility_identification",
        desc="Provide the name/identifier of the semiconductor fabrication facility.",
        parent=parent,
        critical=True
    )

    # Helper alias for shared sources
    shared = extracted.shared_urls

    # 2) Location within Phoenix city limits
    await _add_verified_leaf(
        evaluator,
        node_id="location_verification",
        desc="Verify the facility is located within Phoenix, Arizona city limits.",
        claim=f"The facility '{facility_name}' is located within the municipal city limits of Phoenix, Arizona (not merely the metro area).",
        urls=_combine_sources(extracted.location_urls, shared),
        parent=parent,
        additional_instruction=(
            "Confirm that the site is inside Phoenix city limits (e.g., described as being in Phoenix/North Phoenix), "
            "and not in neighboring municipalities (e.g., Chandler, Tempe, Mesa, Glendale). Evidence should clearly "
            "place the facility within Phoenix city boundaries."
        )
    )

    # 3) CHIPS Act direct funding >= $6B
    await _add_verified_leaf(
        evaluator,
        node_id="chips_act_funding_verification",
        desc="Verify the facility is a recipient of direct CHIPS and Science Act funding and that the direct award is at least $6 billion.",
        claim=f"The facility '{facility_name}' received at least $6 billion in direct funding under the CHIPS and Science Act.",
        urls=_combine_sources(extracted.funding_urls, shared),
        parent=parent,
        additional_instruction=(
            "Verify the 'direct funding' award amount from CHIPS for America (U.S. Department of Commerce) is ≥ $6B. "
            "Do not count tax credits, loans, or state/local incentives as part of the direct award. Phrases like 'up to $6.6B' qualify."
        )
    )

    # 4) Leading-edge production at <= 14nm
    await _add_verified_leaf(
        evaluator,
        node_id="leading_edge_node_verification",
        desc="Verify the facility produces leading-edge semiconductors at process nodes of 14 nanometers (nm) or below.",
        claim=f"The facility '{facility_name}' produces leading-edge semiconductors at process nodes of 14 nm or smaller (e.g., 14nm, 7nm, 5nm, 4nm).",
        urls=_combine_sources(extracted.leading_edge_node_urls, shared),
        parent=parent,
        additional_instruction=(
            "Accept statements that the fab manufactures nodes at 14nm or below (e.g., N7, N5, N4, etc.). "
            "The source should explicitly tie the leading-edge node capability to this specific facility/campus."
        )
    )

    # 5) Wafer size 300mm (12-inch)
    await _add_verified_leaf(
        evaluator,
        node_id="wafer_size_verification",
        desc="Verify the facility manufactures semiconductors on 300mm (12-inch) wafers.",
        claim=f"The facility '{facility_name}' manufactures semiconductors on 300mm (12-inch) wafers.",
        urls=_combine_sources(extracted.wafer_size_urls, shared),
        parent=parent,
        additional_instruction=(
            "The evidence should indicate 300mm (12-inch) wafer production at this facility. "
            "Minor wording variations are acceptable if they clearly indicate 300mm wafer manufacturing."
        )
    )

    # 6) Water consumption 4–5M GPD in first operational phase
    await _add_verified_leaf(
        evaluator,
        node_id="water_consumption_verification",
        desc="Verify the facility's daily water usage for its first operational phase is between 4 million and 5 million gallons per day.",
        claim=f"In its first operational phase, the facility '{facility_name}' uses between 4 and 5 million gallons of water per day.",
        urls=_combine_sources(extracted.water_use_urls, shared),
        parent=parent,
        additional_instruction=(
            "Confirm the first phase (initial fab/phase 1) water demand is within 4–5 million gallons per day. "
            "Values like ~4.9M GPD match. If the value is clearly outside this range, it should fail."
        )
    )

    # 7) Mass production of 4nm by H1 2025
    await _add_verified_leaf(
        evaluator,
        node_id="mass_production_4nm_by_h1_2025_verification",
        desc="Verify the facility achieved mass production of 4-nanometer process technology by the first half of 2025.",
        claim=f"The facility '{facility_name}' achieved mass production of 4nm process technology by the first half of 2025 (no later than June 30, 2025).",
        urls=_combine_sources(extracted.mass_prod_4nm_urls, shared),
        parent=parent,
        additional_instruction=(
            "Look for phrasing like 'mass production', 'volume production', or equivalent for 4nm at this facility, "
            "with a timing of H1 2025 (by end of June 2025). Announcements, official statements, or credible reporting qualify."
        )
    )

    # 8) Campus size >= 1,000 acres
    await _add_verified_leaf(
        evaluator,
        node_id="campus_size_verification",
        desc="Verify the facility campus spans at least 1,000 acres.",
        claim=f"The campus of the facility '{facility_name}' spans at least 1,000 acres.",
        urls=_combine_sources(extracted.campus_size_urls, shared),
        parent=parent,
        additional_instruction=(
            "Accept statements such as 'over 1,000 acres', 'approximately 1,100 acres', etc., that clearly place the campus ≥ 1,000 acres."
        )
    )

    # 9) Cleanroom size > 1.5M sq ft (~140,000 m²)
    await _add_verified_leaf(
        evaluator,
        node_id="cleanroom_size_verification",
        desc="Verify the facility contains a cleanroom exceeding 1.5 million square feet (approximately 140,000 square meters).",
        claim=f"The facility '{facility_name}' contains a cleanroom larger than 1.5 million square feet (≈140,000 m²).",
        urls=_combine_sources(extracted.cleanroom_size_urls, shared),
        parent=parent,
        additional_instruction=(
            "Allow sources that specify either square feet or square meters. "
            "1.5 million sq ft ≈ 139,354 m²; claims around/above ~140,000 m² qualify."
        )
    )

    # 10) Total planned investment > $60B
    await _add_verified_leaf(
        evaluator,
        node_id="total_planned_investment_verification",
        desc="Verify the total planned investment across all phases exceeds $60 billion.",
        claim=f"The total planned investment across all phases for the facility '{facility_name}' exceeds $60 billion.",
        urls=_combine_sources(extracted.investment_urls, shared),
        parent=parent,
        additional_instruction=(
            "Accept figures clearly above $60B (e.g., $65B). Ensure the investment figure pertains to all phases of this Phoenix campus."
        )
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
    Evaluate an answer for identifying the Phoenix (AZ) semiconductor fab that meets all CHIPS/technology/scale constraints.

    Returns a standardized summary dictionary produced by the Evaluator.
    """
    # Initialize evaluator with a parallel root to reflect independent critical checks
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Identify the semiconductor fabrication facility within Phoenix, Arizona city limits that satisfies all "
            "stated constraints (funding, technology, wafer size, water use, production milestone, campus size, "
            "cleanroom size, and total investment)."
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facility_sources(),
        template_class=FacilityExtraction,
        extraction_name="facility_extraction"
    )

    # Record minimal GT/context info for transparency (no hard GT values provided for this task)
    evaluator.add_custom_info(
        info={
            "evaluation_focus": [
                "Phoenix city limits location",
                "CHIPS Act direct funding ≥ $6B",
                "Leading-edge nodes ≤ 14nm",
                "300mm wafers",
                "Phase 1 water use 4–5M GPD",
                "4nm mass production by H1 2025",
                "Campus ≥ 1,000 acres",
                "Cleanroom > 1.5M sq ft (~140,000 m²)",
                "Total investment > $60B"
            ]
        },
        info_type="rubric_overview",
        info_name="rubric_overview"
    )

    # Build and run all rubric verifications
    await verify_facility_constraints(evaluator, root, extracted)

    # Return standardized evaluation summary
    return evaluator.get_summary()