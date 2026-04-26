import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "modern_classA_dc_requirements_2026"
TASK_DESCRIPTION = """
What are the minimum facility specifications that a 100,000 square foot e-commerce distribution center must meet to be considered a modern Class A facility in the United States as of 2026, and what additional regulatory requirements apply specifically in California for logistics facilities of this size?
"""


# --------------------------------------------------------------------------- #
# Data models for information extraction                                      #
# --------------------------------------------------------------------------- #
class ClearHeightSpec(BaseModel):
    min_clear_height_ft: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LoadingDockSpec(BaseModel):
    min_dock_quantity: Optional[str] = None
    standard_dock_door_height_ft: Optional[str] = None
    standard_dock_platform_height_in: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ColumnSpacingSpec(BaseModel):
    column_spacing_ft: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PowerSupplySpec(BaseModel):
    min_amperage: Optional[str] = None
    phase_type: Optional[str] = None  # e.g., "three-phase"
    sources: List[str] = Field(default_factory=list)


class CASetbackSpec(BaseModel):
    industrial_zone_setback_ft: Optional[str] = None
    non_industrial_setback_ft: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FacilitySpecsExtraction(BaseModel):
    clear_height: Optional[ClearHeightSpec] = None
    loading_dock: Optional[LoadingDockSpec] = None
    column_spacing: Optional[ColumnSpacingSpec] = None
    power_supply: Optional[PowerSupplySpec] = None
    ca_setbacks: Optional[CASetbackSpec] = None


# --------------------------------------------------------------------------- #
# Extraction prompt builder                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_specs() -> str:
    return """
    Extract the claimed specifications and supporting source URLs for a modern Class A e-commerce distribution center (100,000 sq ft, U.S., as of 2026), plus California-specific regulatory setbacks if provided.

    Return JSON in this exact nested structure (use null for any missing value; keep numbers as strings with units as presented, e.g., "36 ft", "48 inches"):

    {
      "clear_height": {
        "min_clear_height_ft": string|null,
        "sources": string[]  // URLs that directly support the clear height spec
      },
      "loading_dock": {
        "min_dock_quantity": string|null,                // minimum number of loading docks (for ~100,000 sq ft)
        "standard_dock_door_height_ft": string|null,     // standard dock door height in feet (e.g., "9 ft", "10 ft")
        "standard_dock_platform_height_in": string|null, // standard dock platform height in inches above grade (e.g., "48 inches")
        "sources": string[]                               // URLs that discuss loading dock standards/quantities/dimensions
      },
      "column_spacing": {
        "column_spacing_ft": string|null, // e.g., "50x50 ft", "52x56 ft", "50 ft by 50 ft"
        "sources": string[]               // URLs that discuss modern warehouse column spacing
      },
      "power_supply": {
        "min_amperage": string|null,  // e.g., "2000 A", "1600 amps"
        "phase_type": string|null,    // e.g., "three-phase"
        "sources": string[]           // URLs that discuss electrical service levels required for large warehouses
      },
      "ca_setbacks": {
        "industrial_zone_setback_ft": string|null,     // loading bay to property line in industrial zones under AB 98 / SB 415 (if provided)
        "non_industrial_setback_ft": string|null,      // loading bay to sensitive receptors in non-industrial areas under AB 98 / SB 415 (if provided)
        "sources": string[]                            // URLs citing AB 98 / SB 415 (California warehouse/logistics siting standards)
      }
    }

    Important:
    - Only extract values explicitly present in the answer.
    - For each "sources" field, include only valid URLs explicitly cited in the answer text (plain links or markdown links).
    - Do not invent numbers or URLs. If not present, use null for the value and an empty array for sources.
    - When multiple reasonable values are listed (e.g., ranges), extract the specific value the answer presents as the "minimum" or "standard". If the answer lists a range or multiple, extract the range/string exactly as written (e.g., "32–36 ft").
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return s is not None and str(s).strip() != ""


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any((u or "").strip() for u in urls)


async def _verify_with_sources_or_mark_failed(
    evaluator: Evaluator,
    node_parent,
    *,
    leaf_id: str,
    leaf_desc: str,
    claim: str,
    sources: Optional[List[str]],
    additional_instruction: str,
    extra_prerequisites: Optional[List] = None,
) -> None:
    """
    Add a verification leaf; if sources are missing, mark the leaf failed immediately.
    Otherwise, run URL-grounded verification.
    """
    leaf = evaluator.add_leaf(
        id=leaf_id,
        desc=leaf_desc,
        parent=node_parent,
        critical=True,
    )

    if not _has_urls(sources):
        # No sources: this must fail per source-grounding requirement
        leaf.score = 0.0
        leaf.status = "failed"
        evaluator.add_custom_info(
            {"reason": "missing_sources", "node": leaf_id, "desc": leaf_desc},
            info_type="diagnostic",
            info_name=f"missing_sources_{leaf_id}",
        )
        return

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources or [],
        additional_instruction=additional_instruction,
        extra_prerequisites=extra_prerequisites or [],
    )


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_clear_height_subtree(evaluator: Evaluator, parent_node, specs: FacilitySpecsExtraction):
    node = evaluator.add_parallel(
        id="clear_height_specifications",
        desc="Specify the minimum clear height requirement for modern Class A distribution centers",
        parent=parent_node,
        critical=False,
    )

    ch = specs.clear_height or ClearHeightSpec()
    # Leaf: clear_height_value (existence)
    evaluator.add_custom_node(
        result=_nonempty_str(ch.min_clear_height_ft),
        id="clear_height_value",
        desc="Provide the minimum clear height measurement in feet (industry standards for modern Class A distribution)",
        parent=node,
        critical=True,
    )

    # Leaf: clear_height_reference (support by URLs)
    await _verify_with_sources_or_mark_failed(
        evaluator,
        node,
        leaf_id="clear_height_reference",
        leaf_desc="Provide URL reference supporting the clear height specification from industry sources",
        claim=(
            f"Industry sources indicate that modern Class A distribution or e-commerce warehouse facilities in the U.S. "
            f"commonly require a minimum clear height around {ch.min_clear_height_ft}. "
            f"Treat values as supported if the page states this number, a very close figure, or a range that includes or exceeds it (as of 2026)."
        ),
        sources=ch.sources,
        additional_instruction=(
            "Focus on 'clear height' (distance from floor to lowest obstruction). "
            "Accept if the source states an equivalent minimum or a typical range that includes the stated figure. "
            "Minor unit formatting or rounding differences are acceptable."
        ),
        extra_prerequisites=[
            evaluator.find_node("clear_height_value")
        ],
    )


async def build_loading_dock_subtree(evaluator: Evaluator, parent_node, specs: FacilitySpecsExtraction):
    node = evaluator.add_parallel(
        id="loading_dock_specifications",
        desc="Specify loading dock requirements for a 100,000 sq ft distribution facility",
        parent=parent_node,
        critical=False,
    )

    ld = specs.loading_dock or LoadingDockSpec()

    # Existence checks for each value (critical)
    qty_node = evaluator.add_custom_node(
        result=_nonempty_str(ld.min_dock_quantity),
        id="dock_quantity",
        desc="Specify minimum number of loading docks appropriate for a 100,000 sq ft facility based on industry standards",
        parent=node,
        critical=True,
    )
    door_h_node = evaluator.add_custom_node(
        result=_nonempty_str(ld.standard_dock_door_height_ft),
        id="dock_height_standard",
        desc="Specify standard dock door height dimensions in feet",
        parent=node,
        critical=True,
    )
    plat_h_node = evaluator.add_custom_node(
        result=_nonempty_str(ld.standard_dock_platform_height_in),
        id="dock_platform_height",
        desc="Specify standard dock platform height in inches above ground level",
        parent=node,
        critical=True,
    )

    # Single reference leaf (critical), grounded by provided URLs
    # Note: This leaf validates that at least one cited source is an industry source
    # that discusses appropriate loading dock standards (quantity and/or dimensions).
    # We gate on the above three nodes so if any are missing, this becomes skipped/failed at aggregation.
    composed_text = []
    if _nonempty_str(ld.min_dock_quantity):
        composed_text.append(f"a suitable minimum dock-door count around {ld.min_dock_quantity} for ~100,000 sq ft facilities")
    if _nonempty_str(ld.standard_dock_door_height_ft):
        composed_text.append(f"standard dock door height around {ld.standard_dock_door_height_ft}")
    if _nonempty_str(ld.standard_dock_platform_height_in):
        composed_text.append(f"dock platform height around {ld.standard_dock_platform_height_in} above grade")
    composed_claim_detail = "; ".join(composed_text) if composed_text else "loading dock standards (quantity and/or dimensions)"

    await _verify_with_sources_or_mark_failed(
        evaluator,
        node,
        leaf_id="loading_dock_reference",
        leaf_desc="Provide URL reference supporting loading dock specifications from industry sources",
        claim=(
            f"At least one of the provided industry sources discusses {composed_claim_detail} for distribution/warehouse facilities. "
            f"Treat the specifications as supported if the page states the same figures, very close figures, or typical ranges that include them."
        ),
        sources=ld.sources,
        additional_instruction=(
            "Evaluate whether the page is an industry/authority source (e.g., broker reports, developers, logistics design guides) "
            "and it discusses dock counts for facilities of this scale and/or typical dock door and platform heights (e.g., ~9–10 ft doors, ~48 inches platforms). "
            "It is acceptable if separate sources cover different elements."
        ),
        extra_prerequisites=[qty_node, door_h_node, plat_h_node],
    )


async def build_column_spacing_subtree(evaluator: Evaluator, parent_node, specs: FacilitySpecsExtraction):
    node = evaluator.add_parallel(
        id="column_spacing_specifications",
        desc="Specify column spacing requirements for modern warehouses",
        parent=parent_node,
        critical=False,
    )

    cs = specs.column_spacing or ColumnSpacingSpec()

    evaluator.add_custom_node(
        result=_nonempty_str(cs.column_spacing_ft),
        id="column_spacing_value",
        desc="Provide column spacing dimensions in feet (modern warehouse standards)",
        parent=node,
        critical=True,
    )

    await _verify_with_sources_or_mark_failed(
        evaluator,
        node,
        leaf_id="column_spacing_reference",
        leaf_desc="Provide URL reference supporting column spacing specification from industry sources",
        claim=(
            f"Industry sources discuss modern warehouse column spacing consistent with {cs.column_spacing_ft} "
            f"(for example common grids like 50'x50', 52'x56', etc.). Consider supported if the page states the same number(s) "
            f"or an equivalent/typical grid including the claimed dimension."
        ),
        sources=cs.sources,
        additional_instruction=(
            "Look for mentions of column spacing/grid (e.g., 50x50, 52x56). Minor formatting differences are acceptable."
        ),
        extra_prerequisites=[evaluator.find_node("column_spacing_value")],
    )


async def build_power_supply_subtree(evaluator: Evaluator, parent_node, specs: FacilitySpecsExtraction):
    node = evaluator.add_parallel(
        id="power_supply_specifications",
        desc="Specify power supply requirements for large warehouse operations",
        parent=parent_node,
        critical=False,
    )

    pwr = specs.power_supply or PowerSupplySpec()

    amp_node = evaluator.add_custom_node(
        result=_nonempty_str(pwr.min_amperage),
        id="power_amperage",
        desc="Specify minimum amperage requirements for large-scale operations",
        parent=node,
        critical=True,
    )

    phase_node = evaluator.add_custom_node(
        result=_nonempty_str(pwr.phase_type),
        id="power_phase_type",
        desc="Specify whether single-phase or three-phase electrical service is required for large operations",
        parent=node,
        critical=True,
    )

    await _verify_with_sources_or_mark_failed(
        evaluator,
        node,
        leaf_id="power_reference",
        leaf_desc="Provide URL reference supporting power supply specifications from industry sources",
        claim=(
            f"At least one provided industry source indicates that large warehouse/distribution facilities commonly require "
            f"{pwr.phase_type} electrical service and discuss minimum service capacity around {pwr.min_amperage} (or comparable kVA). "
            f"Treat the claim as supported if the page states these or very close/equivalent specifications."
        ),
        sources=pwr.sources,
        additional_instruction=(
            "Check that the page discusses electrical service for industrial/warehouse uses (e.g., three-phase availability, minimum amperage or service sizes). "
            "Allow typical equivalents (kVA vs amps) and minor rounding."
        ),
        extra_prerequisites=[amp_node, phase_node],
    )


async def build_ca_setback_subtree(evaluator: Evaluator, parent_node, specs: FacilitySpecsExtraction):
    node = evaluator.add_parallel(
        id="california_setback_requirements",
        desc="Specify California-specific setback requirements for logistics facilities under AB 98/SB 415",
        parent=parent_node,
        critical=False,
    )

    ca = specs.ca_setbacks or CASetbackSpec()

    ind_node = evaluator.add_custom_node(
        result=_nonempty_str(ca.industrial_zone_setback_ft),
        id="industrial_zone_setback",
        desc="Specify the setback distance in feet required from loading bays to property lines in industrial zones as mandated by AB 98",
        parent=node,
        critical=True,
    )

    nonind_node = evaluator.add_custom_node(
        result=_nonempty_str(ca.non_industrial_setback_ft),
        id="non_industrial_zone_setback",
        desc="Specify the setback distance in feet required from loading bays to sensitive receptors in non-industrial areas as mandated by AB 98",
        parent=node,
        critical=True,
    )

    await _verify_with_sources_or_mark_failed(
        evaluator,
        node,
        leaf_id="california_regulation_reference",
        leaf_desc="Provide URL reference to AB 98 or SB 415 California warehouse regulatory information",
        claim=(
            f"As of 2026, California AB 98 and/or SB 415 (or implementing state guidance) establish warehouse/logistics "
            f"setback standards for loading bays, including setbacks of approximately {ca.industrial_zone_setback_ft} from "
            f"property lines in industrial zones and approximately {ca.non_industrial_setback_ft} from sensitive receptors "
            f"in non-industrial contexts (wording/ranges acceptable if clearly consistent)."
        ),
        sources=ca.sources,
        additional_instruction=(
            "Verify the page actually references California AB 98 and/or SB 415 (or official state/agency guidance implementing them) "
            "and discusses warehouse/logistics facility setbacks/buffers. Accept if distances match exactly or fall within clearly "
            "equivalent/range-based requirements described on the page. Prioritize primary/official sources when present."
        ),
        extra_prerequisites=[ind_node, nonind_node],
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
    Evaluate an answer for modern Class A (2026) distribution center specifications and California-specific regulations.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent spec groups, allow partial credit
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

    # Top-level rubric node (set to non-critical to avoid hard fail-all; children contain critical leaves)
    top = evaluator.add_parallel(
        id="modern_distribution_center_requirements",
        desc="Provide complete facility specifications for a 100,000 sq ft modern Class A e-commerce distribution center in the United States (2026), plus California-specific regulations",
        parent=root,
        critical=False,
    )

    # Extraction
    extracted_specs = await evaluator.extract(
        prompt=prompt_extract_facility_specs(),
        template_class=FacilitySpecsExtraction,
        extraction_name="facility_specs_extraction",
    )

    # Build subtrees (can run sequentially; each subtree adds its own leaves)
    await build_clear_height_subtree(evaluator, top, extracted_specs)
    await build_loading_dock_subtree(evaluator, top, extracted_specs)
    await build_column_spacing_subtree(evaluator, top, extracted_specs)
    await build_power_supply_subtree(evaluator, top, extracted_specs)
    await build_ca_setback_subtree(evaluator, top, extracted_specs)

    return evaluator.get_summary()