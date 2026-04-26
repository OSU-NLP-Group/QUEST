import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fl_pharmacist_ce_renewal"
TASK_DESCRIPTION = (
    "A Florida licensed pharmacist holds both Test and Treat certification and Collaborative Practice Agreement (CPA) "
    "certification. They are preparing for their standard biennial license renewal (not their first renewal). Determine "
    "the total minimum number of continuing education hours required for their license renewal, and provide a complete breakdown "
    "specifying: the base continuing education hours required for standard pharmacist license renewal in Florida; the minimum number "
    "of those base hours that must be live continuing education; the required hours for the mandatory Medication Errors course (must be Florida Board of Pharmacy-approved); "
    "the required hours for the mandatory Controlled Substances course (must be Florida Board of Pharmacy-approved); the additional continuing education hours required for maintaining "
    "Test and Treat certification (beyond the base requirement); and the additional continuing education hours required for maintaining Collaborative Practice Agreement certification "
    "(beyond the base requirement). For each requirement, provide the official source reference (Florida Board of Pharmacy website, Florida Statutes, or Florida Administrative Code) that "
    "establishes that specific hour requirement."
)

# Ground truth expectations (used for consistency checks and recording)
GROUND_TRUTH = {
    "base_total_hours": 30,
    "base_live_min_hours": 10,
    "medication_errors_hours": 2,
    "controlled_substances_hours": 2,
    "test_and_treat_additional_hours": 3,
    "cpa_additional_hours": 8,
    "expected_total_min_hours": 30 + 3 + 8  # Medication Errors & Controlled Substances are part of the base 30
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementItem(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RenewalContext(BaseModel):
    mentions_biennial: Optional[bool] = None
    mentions_non_first_renewal: Optional[bool] = None
    context_excerpt: Optional[str] = None


class FLCEExtraction(BaseModel):
    context: Optional[RenewalContext] = None

    base_total: Optional[RequirementItem] = None
    base_live_min: Optional[RequirementItem] = None
    medication_errors: Optional[RequirementItem] = None
    controlled_substances: Optional[RequirementItem] = None

    test_and_treat_additional: Optional[RequirementItem] = None
    cpa_additional: Optional[RequirementItem] = None

    cumulative_additivity_statement_present: Optional[bool] = None
    total_min_hours: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_fl_ce() -> str:
    return """
    You must extract a structured breakdown of Florida pharmacist continuing education (CE) requirements for a non-first, standard biennial renewal (2-year/24-month period), including certification add-ons. Extract ONLY what is explicitly stated in the provided answer. Do NOT infer or invent anything.

    Extract the following fields:

    1) context:
       - mentions_biennial: true/false — Whether the answer explicitly states the requirements apply to a biennial (two-year/24-month) renewal period.
       - mentions_non_first_renewal: true/false — Whether the answer explicitly indicates this is NOT the first renewal.
       - context_excerpt: brief quote or phrase from the answer showing this context (if available), else null.

    2) base_total:
       - value: the base total CE hours stated for standard biennial pharmacist renewal (e.g., "30", "30 hours"). If not stated, null.
       - sources: an array of official source URLs explicitly cited in the answer for the base total hours requirement. ONLY include official sources:
         • Florida Board of Pharmacy website pages
         • Florida Statutes pages
         • Florida Administrative Code pages
         If none are cited, return an empty array.

    3) base_live_min:
       - value: the minimum live CE hours within the base requirement (e.g., "10", "at least 10"). If not stated, null.
       - sources: official source URLs cited in the answer establishing the live-hour minimum. If none are cited, return an empty array.

    4) medication_errors:
       - value: the required hours for a Medication Errors course (should be 2 hours if correctly stated). If not stated, null.
       - sources: official source URLs cited in the answer that establish this requirement. If none are cited, return an empty array.

    5) controlled_substances:
       - value: the required hours for a Controlled Substances course (should be 2 hours if correctly stated). If not stated, null.
       - sources: official source URLs cited in the answer that establish this requirement. If none are cited, return an empty array.

    6) test_and_treat_additional:
       - value: the additional CE hours required to maintain Test and Treat certification beyond the base (should be 3 hours if correctly stated). If not stated, null.
       - sources: official source URLs cited in the answer establishing this requirement. If none are cited, return an empty array.

    7) cpa_additional:
       - value: the additional CE hours required to maintain a Collaborative Practice Agreement (CPA) beyond the base (should be 8 hours if correctly stated). If not stated, null.
       - sources: official source URLs cited in the answer establishing this requirement. If none are cited, return an empty array.

    8) cumulative_additivity_statement_present:
       - true/false — Whether the answer explicitly states that if BOTH Test and Treat and CPA are held, the additional CE hours are cumulative (i.e., they both add on top of the base requirement).

    9) total_min_hours:
       - value: the total minimum CE hours for the scenario explicitly stated in the answer text. If not stated, null.

    Rules:
    - Return the exact hours as strings if present (e.g., "30", "10", "2"). If a phrase like "at least 10" appears, return the phrase string in 'value'.
    - For sources, include ONLY official sources explicitly cited in the answer. Do not infer URLs; do not include third-party or non-official summary sites.
    - If a field is not mentioned, set it to null or empty array accordingly.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_int(value: Optional[str]) -> Optional[int]:
    """Extract the first integer found in a string; return None if not found."""
    if not value or not isinstance(value, str):
        return None
    m = re.search(r"\d+", value)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification helper for requirements                                       #
# --------------------------------------------------------------------------- #
async def add_requirement_checks(
    evaluator: Evaluator,
    parent_node,
    *,
    group_id: str,
    value: Optional[str],
    expected_int: int,
    sources: List[str],
    value_desc: str,
    source_claim: str,
    source_leaf_id: str,
    additional_instruction: str
) -> Tuple[Any, Any]:
    """
    Add two critical checks for a requirement under parent_node:
      - Value equality (custom node)
      - Source-supported verification (LLM verify by URLs)
    Returns (value_node, source_verify_node)
    """
    # Value equality check (custom, critical)
    actual_int = parse_int(value)
    value_node = evaluator.add_custom_node(
        result=(actual_int == expected_int),
        id=f"{group_id}_value_correct",
        desc=value_desc,
        parent=parent_node,
        critical=True
    )

    # Require sources presence (custom, critical) – ensures answer provided official sources
    sources_present_node = evaluator.add_custom_node(
        result=(isinstance(sources, list) and len(sources) > 0),
        id=f"{group_id}_sources_provided",
        desc=f"{group_id.replace('_', ' ').title()} official sources are provided in the answer",
        parent=parent_node,
        critical=True
    )

    # Source-supported verification (LLM with URLs)
    source_node = evaluator.add_leaf(
        id=source_leaf_id,
        desc=f"{group_id.replace('_', ' ').title()} requirement is supported by the cited official sources",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=source_claim,
        node=source_node,
        sources=sources,
        additional_instruction=additional_instruction
    )

    return value_node, source_node


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_context_applicability(
    evaluator: Evaluator,
    parent_node,
    ext: FLCEExtraction
) -> None:
    """
    Context: Biennial period (two-year/24-month) and not first renewal.
    """
    ctx_node = evaluator.add_parallel(
        id="Context_Applicability",
        desc="Answer addresses standard biennial renewal (not first renewal) context.",
        parent=parent_node,
        critical=True
    )

    # Single leaf check using the answer text
    biennial_leaf = evaluator.add_leaf(
        id="Biennial_Period_Context",
        desc="Answer makes clear requirements apply to the biennial (2-year/24-month) renewal period for a pharmacist (standard renewal, not first renewal).",
        parent=ctx_node,
        critical=True
    )
    claim = (
        "The answer explicitly indicates the requirements apply to a standard biennial (two-year/24-month) pharmacist renewal "
        "and also clearly indicates it is not the first renewal."
    )
    await evaluator.verify(
        claim=claim,
        node=biennial_leaf,
        additional_instruction=(
            "Look for terms such as 'biennial', 'two-year', '24 months', 'every two years', and phrases indicating "
            "'not first renewal', 'not initial renewal', or 'subsequent renewal'. Allow reasonable synonyms."
        )
    )


async def verify_base_license_ce(
    evaluator: Evaluator,
    parent_node,
    ext: FLCEExtraction
) -> Dict[str, Any]:
    """
    Base Florida pharmacist license renewal CE requirements (with official sources).
    Returns a dict of important nodes to use as prerequisites for later checks.
    """
    base_node = evaluator.add_parallel(
        id="Base_License_CE",
        desc="Base Florida pharmacist license renewal CE requirements (with official sources).",
        parent=parent_node,
        critical=True
    )

    # Base total CE hours = 30
    base_total_val = ext.base_total.value if ext.base_total else None
    base_total_sources = ext.base_total.sources if ext.base_total else []
    base_total_value_node, base_total_source_node = await add_requirement_checks(
        evaluator,
        base_node,
        group_id="Base_Total_CE_Hours",
        value=base_total_val,
        expected_int=GROUND_TRUTH["base_total_hours"],
        sources=base_total_sources,
        value_desc="Base total CE hours for standard biennial pharmacist renewal is correctly stated as 30 hours.",
        source_claim="Florida requires 30 hours of pharmacist continuing education per biennial (two-year) renewal for a standard (non-first) renewal.",
        source_leaf_id="Base_Total_CE_Hours_Source",
        additional_instruction=(
            "Verify this requirement on official sources only: Florida Board of Pharmacy website, Florida Statutes, or Florida Administrative Code. "
            "If the page is non-official (e.g., third-party summaries), treat as not supported."
        )
    )

    # Minimum live CE hours within base = 10
    live_min_val = ext.base_live_min.value if ext.base_live_min else None
    live_min_sources = ext.base_live_min.sources if ext.base_live_min else []
    base_live_value_node, base_live_source_node = await add_requirement_checks(
        evaluator,
        base_node,
        group_id="Base_Live_CE_Minimum",
        value=live_min_val,
        expected_int=GROUND_TRUTH["base_live_min_hours"],
        sources=live_min_sources,
        value_desc="Minimum live CE hours within the base requirement is correctly stated as at least 10 hours.",
        source_claim="Florida requires at least 10 hours of live CE within the 30-hour biennial pharmacist CE requirement.",
        source_leaf_id="Base_Live_CE_Minimum_Source",
        additional_instruction=(
            "Confirm the live-hour minimum on an official source page (Florida Board of Pharmacy website, Florida Statutes, or Florida Administrative Code)."
        )
    )

    # Medication Errors course = 2 hours; FL BOP-approved
    med_err_val = ext.medication_errors.value if ext.medication_errors else None
    med_err_sources = ext.medication_errors.sources if ext.medication_errors else []
    med_err_value_node, med_err_source_node = await add_requirement_checks(
        evaluator,
        base_node,
        group_id="Medication_Errors_Course",
        value=med_err_val,
        expected_int=GROUND_TRUTH["medication_errors_hours"],
        sources=med_err_sources,
        value_desc="Medication Errors course requirement is correctly stated as 2 hours.",
        source_claim="Florida requires a 2-hour Medication Errors course for pharmacist renewal, and the course must be Florida Board of Pharmacy-approved.",
        source_leaf_id="Medication_Errors_Course_Source",
        additional_instruction=(
            "The official source should make clear both the 2-hour requirement and that the course must be approved by the Florida Board of Pharmacy."
        )
    )

    # Controlled Substances course = 2 hours; FL BOP-approved
    cs_val = ext.controlled_substances.value if ext.controlled_substances else None
    cs_sources = ext.controlled_substances.sources if ext.controlled_substances else []
    cs_value_node, cs_source_node = await add_requirement_checks(
        evaluator,
        base_node,
        group_id="Controlled_Substances_Course",
        value=cs_val,
        expected_int=GROUND_TRUTH["controlled_substances_hours"],
        sources=cs_sources,
        value_desc="Controlled Substances course requirement is correctly stated as 2 hours.",
        source_claim="Florida requires a 2-hour Controlled Substances course for pharmacist renewal, and the course must be Florida Board of Pharmacy-approved.",
        source_leaf_id="Controlled_Substances_Course_Source",
        additional_instruction=(
            "The official source should make clear both the 2-hour requirement and that the course must be approved by the Florida Board of Pharmacy."
        )
    )

    return {
        "base_total_value_node": base_total_value_node,
        "base_live_value_node": base_live_value_node,
        "med_err_value_node": med_err_value_node,
        "cs_value_node": cs_value_node
    }


async def verify_certification_ce_additions(
    evaluator: Evaluator,
    parent_node,
    ext: FLCEExtraction
) -> Dict[str, Any]:
    """
    Additional CE requirements for certifications (with official sources), and cumulative additivity statement.
    Returns dict of value nodes for later prerequisites.
    """
    cert_node = evaluator.add_parallel(
        id="Certification_CE_Additions",
        desc="Additional CE requirements for held certifications (beyond base) with official sources.",
        parent=parent_node,
        critical=True
    )

    # Test and Treat additional CE = 3 hours beyond base
    tnt_val = ext.test_and_treat_additional.value if ext.test_and_treat_additional else None
    tnt_sources = ext.test_and_treat_additional.sources if ext.test_and_treat_additional else []
    tnt_value_node, tnt_source_node = await add_requirement_checks(
        evaluator,
        cert_node,
        group_id="Test_and_Treat_Additional_CE",
        value=tnt_val,
        expected_int=GROUND_TRUTH["test_and_treat_additional_hours"],
        sources=tnt_sources,
        value_desc="Test and Treat additional CE requirement is correctly stated as 3 hours beyond the base requirement.",
        source_claim="Pharmacists maintaining Test and Treat certification must complete an additional 3 hours of CE beyond the base requirement, and the CE must be Florida Board of Pharmacy-approved.",
        source_leaf_id="Test_and_Treat_Additional_CE_Source",
        additional_instruction=(
            "Confirm that the official source states 'additional' (beyond base) and specifies 3 hours, and indicates board approval is required."
        )
    )

    # CPA additional CE = 8 hours beyond base (related to collaborative pharmacy practice)
    cpa_val = ext.cpa_additional.value if ext.cpa_additional else None
    cpa_sources = ext.cpa_additional.sources if ext.cpa_additional else []
    cpa_value_node, cpa_source_node = await add_requirement_checks(
        evaluator,
        cert_node,
        group_id="CPA_Additional_CE",
        value=cpa_val,
        expected_int=GROUND_TRUTH["cpa_additional_hours"],
        sources=cpa_sources,
        value_desc="CPA additional CE requirement is correctly stated as 8 hours beyond the base requirement.",
        source_claim="Pharmacists practicing under a Collaborative Practice Agreement must complete an additional 8 hours of CE beyond the base requirement, related to collaborative pharmacy practice, and the CE must be Florida Board of Pharmacy-approved.",
        source_leaf_id="CPA_Additional_CE_Source",
        additional_instruction=(
            "Confirm the page states the requirement is additional (beyond base), specifies 8 hours, is related to collaborative pharmacy practice, and requires board approval."
        )
    )

    # Cumulative additivity statement
    cumulative_leaf = evaluator.add_leaf(
        id="Cumulative_Additivity_Statement",
        desc="Answer clearly states that when a pharmacist holds BOTH Test and Treat and CPA certifications, the additional CE hours are cumulative (i.e., both add on top of the base requirement rather than substituting for each other).",
        parent=cert_node,
        critical=True
    )
    cumulative_claim = (
        "The answer explicitly states that if a pharmacist holds BOTH Test and Treat and CPA certifications, "
        "the additional CE hours are cumulative and both are added on top of the base requirement."
    )
    await evaluator.verify(
        claim=cumulative_claim,
        node=cumulative_leaf,
        additional_instruction=(
            "Accept wording that clearly implies additivity, such as 'stack', 'in addition to', 'added together', "
            "'both apply', or 'sum both extras'."
        )
    )

    return {
        "tnt_value_node": tnt_value_node,
        "cpa_value_node": cpa_value_node,
    }


async def verify_total_minimum_hours(
    evaluator: Evaluator,
    parent_node,
    ext: FLCEExtraction,
    prereq_nodes: List[Any]
) -> None:
    """
    Total minimum hours: must be provided by the answer and consistent with base + additions.
    """
    total_node = evaluator.add_parallel(
        id="Total_Minimum_Hours",
        desc="Answer provides the total minimum CE hours for the scenario and the total is consistent with the stated base + additional requirements.",
        parent=parent_node,
        critical=True
    )

    # Check that total is provided
    total_provided_node = evaluator.add_custom_node(
        result=(ext.total_min_hours is not None and isinstance(ext.total_min_hours, str) and ext.total_min_hours.strip() != ""),
        id="Total_Minimum_Hours_Provided",
        desc="Answer explicitly provides the total minimum CE hours for the scenario.",
        parent=total_node,
        critical=True
    )

    # Consistency check using LLM: computed vs stated
    base_val = parse_int(ext.base_total.value if ext.base_total else None)
    tnt_val = parse_int(ext.test_and_treat_additional.value if ext.test_and_treat_additional else None)
    cpa_val = parse_int(ext.cpa_additional.value if ext.cpa_additional else None)
    total_val = parse_int(ext.total_min_hours)

    expected_sum = None
    if base_val is not None and tnt_val is not None and cpa_val is not None:
        expected_sum = base_val + tnt_val + cpa_val

    total_consistency_leaf = evaluator.add_leaf(
        id="Total_Minimum_Consistency",
        desc="Total minimum hours matches the sum of base hours + Test and Treat additional + CPA additional as stated in the answer.",
        parent=total_node,
        critical=True
    )

    # Build claim that references the answer's breakdown and the computed expected sum
    claim_parts = []
    claim_parts.append(f"Base hours stated: {ext.base_total.value if ext.base_total else 'None'}")
    claim_parts.append(f"Test and Treat additional: {ext.test_and_treat_additional.value if ext.test_and_treat_additional else 'None'}")
    claim_parts.append(f"CPA additional: {ext.cpa_additional.value if ext.cpa_additional else 'None'}")
    claim_parts.append(f"Total stated: {ext.total_min_hours if ext.total_min_hours else 'None'}")
    if expected_sum is not None:
        claim_parts.append(f"Expected total (sum): {expected_sum}")
    else:
        claim_parts.append("Expected total (sum): not computable from stated parts")

    claim = (
        "Verify that the answer's stated total minimum hours equals the sum of the stated base hours plus the additional hours for "
        "Test and Treat and CPA. Specifically:\n"
        + " | ".join(claim_parts)
    )

    await evaluator.verify(
        claim=claim,
        node=total_consistency_leaf,
        additional_instruction=(
            "Treat Medication Errors and Controlled Substances as part of the base requirement; do not double-count them. "
            "Confirm the total equals base + Test and Treat additional + CPA additional, using the values as presented in the answer text. "
            "If any of the components are missing or incorrect, the total consistency should not pass."
        ),
        extra_prerequisites=prereq_nodes  # Gate on prior critical value checks
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
    Evaluate the answer for Florida pharmacist CE requirements with Test and Treat + CPA certifications.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation parallel, we'll add a critical top-level node under it
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

    # Extract structured info from the answer
    ext: FLCEExtraction = await evaluator.extract(
        prompt=prompt_extract_fl_ce(),
        template_class=FLCEExtraction,
        extraction_name="fl_ce_extraction"
    )

    # Add ground truth information for reference
    evaluator.add_ground_truth({
        "expected": GROUND_TRUTH,
        "notes": "Medication Errors and Controlled Substances hours are part of the base 30 hours (not extra). Additional hours for Test and Treat (3) and CPA (8) are cumulative."
    }, gt_type="ground_truth")

    # Build top-level critical node as per rubric
    fl_eval_node = evaluator.add_parallel(
        id="License_Renewal_CE_Requirements_FL",
        desc=(
            "Evaluate whether the answer correctly states the Florida pharmacist biennial (non-first renewal) CE requirements "
            "for base license renewal and for Test and Treat + CPA certifications, including official source references for each hour requirement, "
            "and provides the correct total minimum hours derived from those requirements."
        ),
        parent=root,
        critical=True
    )

    # 1) Context applicability
    await verify_context_applicability(evaluator, fl_eval_node, ext)

    # 2) Base license CE
    base_nodes = await verify_base_license_ce(evaluator, fl_eval_node, ext)

    # 3) Certification CE additions
    cert_nodes = await verify_certification_ce_additions(evaluator, fl_eval_node, ext)

    # 4) Total minimum hours (gate on prior value checks)
    prereqs = [
        base_nodes.get("base_total_value_node"),
        cert_nodes.get("tnt_value_node"),
        cert_nodes.get("cpa_value_node")
    ]
    # Filter out None just in case
    prereqs = [p for p in prereqs if p is not None]
    await verify_total_minimum_hours(evaluator, fl_eval_node, ext, prereqs)

    # Return summary
    return evaluator.get_summary()