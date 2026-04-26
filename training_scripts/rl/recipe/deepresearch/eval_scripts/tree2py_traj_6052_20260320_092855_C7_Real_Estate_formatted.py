import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "warehouse_specs_midatl_midwest_conversion"
TASK_DESCRIPTION = (
    "A commercial real estate developer in the Mid-Atlantic and Midwest regions is evaluating properties for "
    "conversion to warehouse and distribution centers following recent retail closures. Based on current industry "
    "standards for logistics operations, identify the key specifications that a commercial property in Maryland, "
    "Ohio, Pennsylvania, or New Jersey must meet to be suitable for warehouse/distribution center use. The "
    "specifications must cover building dimensions, loading infrastructure, parking requirements, safety systems, "
    "accessibility compliance, zoning, location factors, structural elements, utilities, and property availability."
)

# Ground-truth style requirements (used for threshold checks and reporting)
GROUND_TRUTH_REQUIREMENTS = {
    "Minimum_Building_Size": "Minimum building size is at least 10,000 square feet of usable space.",
    "Clear_Height_Specification": "Minimum clear height is at least 28 feet for standard distribution operations.",
    "Loading_Dock_Ratio": "Provide at least 1 loading dock per 10,000 square feet of building area.",
    "Loading_Dock_Height_Standard": "Standard dock height is typically 48–52 inches above grade.",
    "Dock_Door_Dimensions": "Dock doors are at least 8 feet wide by 9 feet tall.",
    "Parking_Space_Ratio": "Provide 1–2 car parking spaces per 1,000 square feet of building area.",
    "Fire_Safety_System": "Provide automatic sprinkler system compliant with NFPA 13 (e.g., ESFR where applicable).",
    "ADA_Compliance": "Meet Americans with Disabilities Act (ADA) accessibility requirements.",
    "Zoning_Classification": "Zoning must allow warehouse/distribution operations (industrial or equivalent).",
    "Highway_Accessibility": "Reasonable access to major highways or transportation routes.",
    "Column_Spacing": "Typical column spacing supports racking, commonly around 40–50 feet grids.",
    "Power_Infrastructure": "Adequate electrical infrastructure for warehouse operations (e.g., 3‑phase, sufficient amperage).",
    "Property_Availability": "Property is available for lease or purchase."
}

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class SpecItem(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class WarehouseSpecs(BaseModel):
    # Geographic scope
    Geographic_Location: Optional[SpecItem] = None

    # Building dimensions, docks, parking, safety, accessibility, zoning, location, structural, utilities, availability
    Minimum_Building_Size: Optional[SpecItem] = None
    Clear_Height_Specification: Optional[SpecItem] = None
    Loading_Dock_Ratio: Optional[SpecItem] = None
    Loading_Dock_Height_Standard: Optional[SpecItem] = None
    Dock_Door_Dimensions: Optional[SpecItem] = None
    Parking_Space_Ratio: Optional[SpecItem] = None
    Fire_Safety_System: Optional[SpecItem] = None
    ADA_Compliance: Optional[SpecItem] = None
    Zoning_Classification: Optional[SpecItem] = None
    Highway_Accessibility: Optional[SpecItem] = None
    Column_Spacing: Optional[SpecItem] = None
    Power_Infrastructure: Optional[SpecItem] = None
    Property_Availability: Optional[SpecItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_warehouse_specs() -> str:
    return """
    Extract, from the provided answer text only, the specific specification statements and the exact URL sources cited for each of the following specification categories for warehouse/distribution suitability. For each category, return:
    - value: the exact wording of the specification as stated in the answer (keep units and qualifiers, do not normalize).
    - sources: an array of URL strings explicitly cited in the answer that support this specification. If no URLs are present in the answer for that category, return an empty list.

    Categories (use these exact keys in the JSON):
    - Geographic_Location
    - Minimum_Building_Size
    - Clear_Height_Specification
    - Loading_Dock_Ratio
    - Loading_Dock_Height_Standard
    - Dock_Door_Dimensions
    - Parking_Space_Ratio
    - Fire_Safety_System
    - ADA_Compliance
    - Zoning_Classification
    - Highway_Accessibility
    - Column_Spacing
    - Power_Infrastructure
    - Property_Availability

    Important extraction rules:
    1) Extract only what the answer explicitly states. Do not infer or add content not present in the answer.
    2) For sources, include only actual URLs that appear in the answer (plain links or links embedded in markdown). If the answer cites a source without a URL, do not fabricate one; return an empty list for sources in that category.
    3) If a category is not mentioned in the answer, set its value to null and sources to [].
    4) Do not normalize values; keep the author's units and phrasing intact (e.g., '28 ft clear' or '48–52 inches').

    Return a single JSON object with the fields listed above. Each field value is an object of the shape:
    {
      "value": string | null,
      "sources": string[]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_item(item: Optional[SpecItem]) -> SpecItem:
    return item or SpecItem(value=None, sources=[])


def _has_text(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_geographic_location(evaluator: Evaluator, parent, specs: WarehouseSpecs) -> None:
    """
    Geographic_Location (CRITICAL)
    - Check the answer explicitly states the target states scope (MD, OH, PA, NJ) or equivalent phrasing.
    - This is a property of the answer text, so we use simple verification without URL evidence.
    """
    node = evaluator.add_sequential(
        id="Geographic_Location",
        desc="Property is located in Maryland, Ohio, Pennsylvania, or New Jersey",
        parent=parent,
        critical=True
    )

    geo_item = _safe_item(specs.Geographic_Location)

    # Existence: answer mentions something about the geographic scope
    evaluator.add_custom_node(
        result=_has_text(geo_item.value),
        id="Geographic_Location_exists",
        desc="Geographic scope is explicitly stated in the answer",
        parent=node,
        critical=True
    )

    # Simple verify using the answer text context
    v = evaluator.add_leaf(
        id="Geographic_Location_verify",
        desc="Answer explicitly references Maryland, Ohio, Pennsylvania, or New Jersey (or their abbreviations)",
        parent=node,
        critical=True
    )
    claim = (
        "The answer explicitly states that the specifications apply to properties in Maryland, Ohio, Pennsylvania, "
        "or New Jersey. Accept common abbreviations (MD, OH, PA, NJ) and phrasing that clearly references these "
        "states or explicitly includes them within 'Mid-Atlantic and Midwest' scope."
    )
    await evaluator.verify(
        claim=claim,
        node=v,
        additional_instruction="Judge using the answer text only. Do not require web evidence for this scope check."
    )


async def verify_spec_with_sources_and_threshold(
    evaluator: Evaluator,
    parent,
    spec_id: str,
    spec_desc: str,
    item: SpecItem,
    support_instruction: str,
    threshold_requirement: Optional[str] = None,
    threshold_instruction: Optional[str] = None,
    require_sources: bool = True,
    id_suffix: str = ""
) -> None:
    """
    Generic verifier for a spec that:
    1) Requires the spec to be present (value text) and, usually, to have sources.
    2) Verifies that the cited sources support the stated value (verify_by_urls).
    3) Optionally verifies via simple logic that the value meets or exceeds the rubric threshold.

    All children are critical as per rubric.
    """
    node = evaluator.add_sequential(
        id=spec_id if not id_suffix else f"{spec_id}_{id_suffix}",
        desc=spec_desc,
        parent=parent,
        critical=True
    )

    # Existence check
    exists = _has_text(item.value) and (len(item.sources) > 0 if require_sources else True)
    existence_reason = (
        "Specification value provided and at least one source URL is cited in the answer"
        if require_sources
        else "Specification value provided"
    )
    evaluator.add_custom_node(
        result=exists,
        id=f"{spec_id}_exists",
        desc=existence_reason,
        parent=node,
        critical=True
    )

    # 1) Sources support the stated value (grounded verification)
    support_leaf = evaluator.add_leaf(
        id=f"{spec_id}_sources_support_value",
        desc="Cited source(s) support the stated specification value",
        parent=node,
        critical=True
    )
    support_claim = f"The following warehouse/distribution specification is correct as stated: \"{item.value or ''}\"."
    await evaluator.verify(
        claim=support_claim,
        node=support_leaf,
        sources=item.sources,
        additional_instruction=support_instruction
    )

    # 2) Optional threshold/simple check using answer text only
    if threshold_requirement and threshold_instruction:
        thr_leaf = evaluator.add_leaf(
            id=f"{spec_id}_meets_requirement",
            desc="Stated value meets or exceeds the required threshold",
            parent=node,
            critical=True
        )
        thr_claim = (
            f"Given the provided specification text: \"{item.value or ''}\", it meets or exceeds the required "
            f"threshold: {threshold_requirement}"
        )
        await evaluator.verify(
            claim=thr_claim,
            node=thr_leaf,
            additional_instruction=threshold_instruction
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for warehouse/distribution suitability specifications in MD/OH/PA/NJ.
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
        default_model=model
    )

    # Add a top-level node mirroring the rubric root (parallel, non-critical)
    main_node = evaluator.add_parallel(
        id="Suitable_Commercial_Property_Identification",
        desc="Identifies commercial property specifications meeting warehouse conversion requirements",
        parent=root,
        critical=False
    )

    # Extract structured info from the answer
    extracted: WarehouseSpecs = await evaluator.extract(
        prompt=prompt_extract_warehouse_specs(),
        template_class=WarehouseSpecs,
        extraction_name="warehouse_spec_extraction"
    )

    # Record ground-truth style requirements for transparency
    evaluator.add_ground_truth(
        gt_info=GROUND_TRUTH_REQUIREMENTS,
        gt_type="required_spec_thresholds"
    )

    # Geographic location (answer-text verification only)
    await verify_geographic_location(evaluator, main_node, extracted)

    # Common instructions for source-supported verifications
    common_support_ins = (
        "Use the cited URL(s) to confirm the specification as written. Accept clear synonyms, equivalent units, "
        "and close variants that are standard in U.S. warehouse/distribution practice (e.g., 48–52 inches is "
        "consistent with 48 inches). If a source shows a stricter standard (e.g., higher clear height), that "
        "still supports a minimum threshold. Sources may be national-level industry publications (e.g., NAIOP, "
        "CBRE, Prologis, code references) or state/municipal guidance for MD/OH/PA/NJ."
    )

    # Quantitative threshold check instruction (simple verify on answer text)
    numeric_threshold_ins = (
        "Judge using the provided answer text only. Interpret numbers and units with common sense (ft/feet, in/inches). "
        "Treat phrases like 'minimum', 'at least', '≥', 'or higher' as meeting/exceeding the threshold. For ranges, "
        "consider whether the stated number or range satisfies the minimum or falls within the required interval."
    )

    # Retrieve safe items
    items = {
        "Minimum_Building_Size": _safe_item(extracted.Minimum_Building_Size),
        "Clear_Height_Specification": _safe_item(extracted.Clear_Height_Specification),
        "Loading_Dock_Ratio": _safe_item(extracted.Loading_Dock_Ratio),
        "Loading_Dock_Height_Standard": _safe_item(extracted.Loading_Dock_Height_Standard),
        "Dock_Door_Dimensions": _safe_item(extracted.Dock_Door_Dimensions),
        "Parking_Space_Ratio": _safe_item(extracted.Parking_Space_Ratio),
        "Fire_Safety_System": _safe_item(extracted.Fire_Safety_System),
        "ADA_Compliance": _safe_item(extracted.ADA_Compliance),
        "Zoning_Classification": _safe_item(extracted.Zoning_Classification),
        "Highway_Accessibility": _safe_item(extracted.Highway_Accessibility),
        "Column_Spacing": _safe_item(extracted.Column_Spacing),
        "Power_Infrastructure": _safe_item(extracted.Power_Infrastructure),
        "Property_Availability": _safe_item(extracted.Property_Availability),
    }

    # Spec-by-spec verification according to rubric
    # 1) Minimum_Building_Size (with threshold)
    await verify_spec_with_sources_and_threshold(
        evaluator, main_node,
        spec_id="Minimum_Building_Size",
        spec_desc="Property has minimum 10,000 square feet of usable space",
        item=items["Minimum_Building_Size"],
        support_instruction=common_support_ins,
        threshold_requirement=GROUND_TRUTH_REQUIREMENTS["Minimum_Building_Size"],
        threshold_instruction=numeric_threshold_ins,
        require_sources=True
    )

    # 2) Clear_Height_Specification (with threshold)
    await verify_spec_with_sources_and_threshold(
        evaluator, main_node,
        spec_id="Clear_Height_Specification",
        spec_desc="Property has or can achieve minimum 28 feet clear height for standard distribution operations",
        item=items["Clear_Height_Specification"],
        support_instruction=common_support_ins,
        threshold_requirement=GROUND_TRUTH_REQUIREMENTS["Clear_Height_Specification"],
        threshold_instruction=numeric_threshold_ins,
        require_sources=True
    )

    # 3) Loading_Dock_Ratio (with threshold)
    await verify_spec_with_sources_and_threshold(
        evaluator, main_node,
        spec_id="Loading_Dock_Ratio",
        spec_desc="Property has or can accommodate minimum 1 loading dock per 10,000 square feet",
        item=items["Loading_Dock_Ratio"],
        support_instruction=common_support_ins,
        threshold_requirement=GROUND_TRUTH_REQUIREMENTS["Loading_Dock_Ratio"],
        threshold_instruction=numeric_threshold_ins,
        require_sources=True
    )

    # 4) Loading_Dock_Height_Standard (with threshold range 48–52 in)
    await verify_spec_with_sources_and_threshold(
        evaluator, main_node,
        spec_id="Loading_Dock_Height_Standard",
        spec_desc="Loading docks are or can be constructed at 48-52 inches above grade",
        item=items["Loading_Dock_Height_Standard"],
        support_instruction=common_support_ins,
        threshold_requirement=GROUND_TRUTH_REQUIREMENTS["Loading_Dock_Height_Standard"],
        threshold_instruction=(
            "Judge using the answer text. The stated dock height should fall in the standard range 48–52 inches "
            "(allowing minor formatting variants, e.g., 48-52 in, 4 ft ± a few inches)."
        ),
        require_sources=True
    )

    # 5) Dock_Door_Dimensions (with threshold)
    await verify_spec_with_sources_and_threshold(
        evaluator, main_node,
        spec_id="Dock_Door_Dimensions",
        spec_desc="Dock doors are or can be sized at minimum 8 feet wide by 9 feet tall",
        item=items["Dock_Door_Dimensions"],
        support_instruction=common_support_ins,
        threshold_requirement=GROUND_TRUTH_REQUIREMENTS["Dock_Door_Dimensions"],
        threshold_instruction=numeric_threshold_ins,
        require_sources=True
    )

    # 6) Parking_Space_Ratio (with threshold 1–2 / 1,000 sf)
    await verify_spec_with_sources_and_threshold(
        evaluator, main_node,
        spec_id="Parking_Space_Ratio",
        spec_desc="Property provides or can accommodate 1-2 parking spaces per 1,000 square feet",
        item=items["Parking_Space_Ratio"],
        support_instruction=common_support_ins,
        threshold_requirement=GROUND_TRUTH_REQUIREMENTS["Parking_Space_Ratio"],
        threshold_instruction=(
            "Judge using the answer text. Consider '1–2 per 1,000 sq ft' satisfied by values within this range "
            "or clearly equivalent phrasing (e.g., 1 per 1,000 up to 2 per 1,000)."
        ),
        require_sources=True
    )

    # 7) Fire_Safety_System (require NFPA 13 notion; include simple check that mentions NFPA 13)
    await verify_spec_with_sources_and_threshold(
        evaluator, main_node,
        spec_id="Fire_Safety_System",
        spec_desc="Property has or can accommodate automatic sprinkler systems compliant with NFPA 13",
        item=items["Fire_Safety_System"],
        support_instruction=common_support_ins,
        threshold_requirement=GROUND_TRUTH_REQUIREMENTS["Fire_Safety_System"],
        threshold_instruction=(
            "Judge using the answer text. The stated requirement should indicate NFPA 13 compliance explicitly "
            "or by clear implication (e.g., ESFR sprinklers under NFPA 13)."
        ),
        require_sources=True
    )

    # 8) ADA_Compliance (qualitative; require sources + simple check mentions ADA)
    await verify_spec_with_sources_and_threshold(
        evaluator, main_node,
        spec_id="ADA_Compliance",
        spec_desc="Property meets or can be modified to meet ADA accessibility requirements",
        item=items["ADA_Compliance"],
        support_instruction=common_support_ins,
        threshold_requirement=GROUND_TRUTH_REQUIREMENTS["ADA_Compliance"],
        threshold_instruction="Judge using the answer text. The stated requirement should mention ADA or ADA compliance.",
        require_sources=True
    )

    # 9) Zoning_Classification (qualitative; require sources + simple check mentions industrial/permits warehousing)
    await verify_spec_with_sources_and_threshold(
        evaluator, main_node,
        spec_id="Zoning_Classification",
        spec_desc="Property is zoned for industrial or commercial/industrial use permitting warehouse operations",
        item=items["Zoning_Classification"],
        support_instruction=(
            common_support_ins
            + " Accept local zoning nomenclature (e.g., I-1, IL, IG, LI, GI, IP, 'industrial/commercial'). "
              "Source should indicate warehousing/distribution is a permitted or conditional use in the cited zoning."
        ),
        threshold_requirement=GROUND_TRUTH_REQUIREMENTS["Zoning_Classification"],
        threshold_instruction="Judge using the answer text. The statement should indicate industrial or equivalent zoning that permits warehousing.",
        require_sources=True
    )

    # 10) Highway_Accessibility (qualitative; require sources)
    await verify_spec_with_sources_and_threshold(
        evaluator, main_node,
        spec_id="Highway_Accessibility",
        spec_desc="Property has reasonable access to major highways or transportation routes",
        item=items["Highway_Accessibility"],
        support_instruction=common_support_ins,
        threshold_requirement=None,
        threshold_instruction=None,
        require_sources=True
    )

    # 11) Column_Spacing (with typical 40–50 ft range threshold)
    await verify_spec_with_sources_and_threshold(
        evaluator, main_node,
        spec_id="Column_Spacing",
        spec_desc="Property has adequate column spacing (typically 40-50 feet) to support warehouse racking systems",
        item=items["Column_Spacing"],
        support_instruction=common_support_ins,
        threshold_requirement=GROUND_TRUTH_REQUIREMENTS["Column_Spacing"],
        threshold_instruction=(
            "Judge using the answer text. The stated spacing should be consistent with typical racking-compatible "
            "grids (around 40–50 feet). Allow common formats like 40' x 50', 50' bays, etc."
        ),
        require_sources=True
    )

    # 12) Power_Infrastructure (qualitative; require sources)
    await verify_spec_with_sources_and_threshold(
        evaluator, main_node,
        spec_id="Power_Infrastructure",
        spec_desc="Property has adequate electrical infrastructure for warehouse operations",
        item=items["Power_Infrastructure"],
        support_instruction=(
            common_support_ins
            + " Accept mentions of typical warehouse electrical characteristics (e.g., 3‑phase power, 277/480V, "
              "sufficient amperage for material handling and automation)."
        ),
        threshold_requirement=None,
        threshold_instruction=None,
        require_sources=True
    )

    # 13) Property_Availability (answer-text check is acceptable; sources optional in many answers)
    # For practicality, do not require sources; ensure the answer states availability.
    await verify_spec_with_sources_and_threshold(
        evaluator, main_node,
        spec_id="Property_Availability",
        spec_desc="Property is available for lease or purchase",
        item=items["Property_Availability"],
        support_instruction="If sources are provided, they should indicate availability; otherwise, judge using the answer text.",
        threshold_requirement=GROUND_TRUTH_REQUIREMENTS["Property_Availability"],
        threshold_instruction="Judge using the answer text only; availability should be explicitly stated.",
        require_sources=False
    )

    # Final structured summary
    return evaluator.get_summary()