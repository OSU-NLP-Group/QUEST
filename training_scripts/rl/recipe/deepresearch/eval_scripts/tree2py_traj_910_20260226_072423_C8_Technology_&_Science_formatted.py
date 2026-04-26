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
TASK_ID = "qis_semiconductor_facilities_2026"
TASK_DESCRIPTION = (
    "A technology investment firm is preparing a comprehensive report on advanced technology infrastructure in the "
    "United States to guide their 2026 investment strategy. They are specifically interested in both quantum computing "
    "research capabilities and semiconductor manufacturing capacity.\n\n"
    "Your task is to identify and provide detailed information about 4 distinct facilities:\n"
    "- 2 facilities must be from the U.S. Department of Energy National Quantum Information Science (QIS) Research Centers\n"
    "- 2 facilities must be semiconductor fabrication plants operated by either Intel Corporation or Taiwan Semiconductor "
    "Manufacturing Company (TSMC)\n\n"
    "All 4 facilities must be located in different U.S. states, and all must be operational or have publicly announced "
    "construction plans as of 2020 or later.\n\n"
    "For each of the 2 quantum research centers, provide:\n"
    "1. The official acronym of the center\n"
    "2. The lead national laboratory managing the center\n"
    "3. The U.S. state where the lead laboratory is located\n"
    "4. A brief description of the center's primary research focus area\n"
    "5. A reference URL from an official Department of Energy or national laboratory source\n\n"
    "For each of the 2 semiconductor facilities, provide:\n"
    "1. The official facility name or designation (e.g., Fab number, campus name)\n"
    "2. The operating company (Intel or TSMC)\n"
    "3. The U.S. state where the facility is located\n"
    "4. The primary process node or technology being manufactured (e.g., 5nm, 3nm, etc.)\n"
    "5. A reference URL from the company's official website or press releases"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class QuantumCenterItem(BaseModel):
    acronym: Optional[str] = None
    lead_lab: Optional[str] = None
    state: Optional[str] = None
    focus: Optional[str] = None
    reference_url: Optional[str] = None


class SemiconductorFabItem(BaseModel):
    name: Optional[str] = None
    company: Optional[str] = None
    state: Optional[str] = None
    process: Optional[str] = None
    reference_url: Optional[str] = None


class FacilitiesExtraction(BaseModel):
    quantum_centers: List[QuantumCenterItem] = Field(default_factory=list)
    semiconductor_fabs: List[SemiconductorFabItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
    Extract up to two DOE National Quantum Information Science (QIS) Research Centers and up to two semiconductor fabrication facilities (Intel or TSMC) described in the answer.

    Return a JSON object with two arrays: 
    - quantum_centers: an array of objects with fields [acronym, lead_lab, state, focus, reference_url]
    - semiconductor_fabs: an array of objects with fields [name, company, state, process, reference_url]

    Rules:
    - Only include U.S.-based facilities.
    - For quantum_centers:
        • acronym: official acronym (e.g., Q-NEXT, SQMS, C2QA, QSA, QSC)
        • lead_lab: the lead national laboratory (e.g., Argonne National Laboratory)
        • state: the U.S. state for the lead lab location
        • focus: brief description of primary research focus per the answer
        • reference_url: a single URL from DOE or the official national lab website (e.g., energy.gov, anl.gov, lbl.gov, fnal.gov, ornl.gov, lanl.gov, pnnl.gov, sandia.gov, bnl.gov, ameslab.gov)
    - For semiconductor_fabs:
        • name: official facility name/designation (e.g., "Fab 52", "Arizona Fab", "Ohio One Campus")
        • company: Intel or TSMC (text as given in the answer)
        • state: the U.S. state location
        • process: the primary process node or technology (e.g., "5nm", "3nm", "Intel 4", "20A", etc.)
        • reference_url: a single URL from the company's official site or press releases (intel.com or tsmc.com domains, including subdomains/newsrooms)
    - Extract the entries in the same order as they appear in the answer. If the answer includes more than two entries per category, extract them all; the evaluator will pick the first two. 
    - If a field is missing, set it to null.
    - For URLs, extract the actual URLs; if not present, set to null. Do not invent URLs.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _normalize_state(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_quantum_center(
    evaluator: Evaluator,
    parent_node,
    item: QuantumCenterItem,
    index: int,
    seen_states: List[str],
) -> None:
    """
    Build and verify a single quantum center subtree.
    """
    idx = index + 1
    qc_node = evaluator.add_parallel(
        id=f"quantum_center_{idx}",
        desc=f"{'First' if idx == 1 else 'Second'} quantum research center identification with all required details",
        parent=parent_node,
        critical=False,
    )

    # Reference presence (critical gate)
    ref_present = evaluator.add_custom_node(
        result=_non_empty(item.reference_url),
        id=f"qc{idx}_reference_present",
        desc=f"QC{idx}: Reference URL is provided",
        parent=qc_node,
        critical=True,
    )

    # Reference official source (critical)
    ref_official = evaluator.add_leaf(
        id=f"qc{idx}_reference",
        desc=f"QC{idx}: Reference URL from official DOE or laboratory source is provided",
        parent=qc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This URL is an official U.S. Department of Energy or U.S. National Laboratory webpage. "
            "Acceptable domains include energy.gov, doe.gov, or official national laboratory .gov domains "
            "such as anl.gov, lbl.gov, fnal.gov, ornl.gov, lanl.gov, sandia.gov, pnnl.gov, bnl.gov, ameslab.gov, etc."
        ),
        node=ref_official,
        sources=item.reference_url,
        additional_instruction=(
            "Use the URL shown to determine the domain ownership. If the URL is missing or not on these official domains, "
            "the claim is not supported."
        ),
    )

    # Acronym (critical) - verify against the reference
    acronym_leaf = evaluator.add_leaf(
        id=f"qc{idx}_acronym",
        desc=f"QC{idx}: Official acronym of the quantum center is provided",
        parent=qc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official acronym of this DOE National QIS Research Center is '{item.acronym}'.",
        node=acronym_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "Allow case-insensitive comparison and minor punctuation variants (e.g., Q-NEXT vs QNEXT). "
            "If the page clearly indicates a different acronym or does not support the given one, mark as not supported."
        ),
    )

    # Lead lab (critical)
    lead_lab_leaf = evaluator.add_leaf(
        id=f"qc{idx}_lead_lab",
        desc=f"QC{idx}: Lead national laboratory operating the center is correctly identified",
        parent=qc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The lead national laboratory managing this center is '{item.lead_lab}'.",
        node=lead_lab_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "Accept reasonable variants of the lab's official name (e.g., 'Argonne' vs 'Argonne National Laboratory'). "
            "The page should explicitly indicate the lead lab."
        ),
    )

    # State (critical): verify state for lead lab
    state_leaf = evaluator.add_leaf(
        id=f"qc{idx}_state",
        desc=f"QC{idx}: State location of the lead laboratory is provided",
        parent=qc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The lead laboratory '{item.lead_lab}' is located in the U.S. state of '{item.state}'.",
        node=state_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "It's acceptable if the page mentions the lab city and state (e.g., Lemont, Illinois). "
            "Infer the state correctly if the city/state are explicitly shown."
        ),
    )

    # Operational/announced >= 2020 (critical)
    operational_leaf = evaluator.add_leaf(
        id=f"qc{idx}_operational",
        desc=f"QC{idx}: Center is operational or has publicly announced construction plans as of 2020 or later",
        parent=qc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This center is operational or had its establishment/launch/construction publicly announced in or after 2020 "
            "(year >= 2020)."
        ),
        node=operational_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "Look for language like 'launched', 'established', 'announced', 'opening', 'operational', etc., together with a year. "
            "If an explicit date/year is not clearly 2020 or later, do not support."
        ),
    )

    # Focus (non-critical)
    focus_leaf = evaluator.add_leaf(
        id=f"qc{idx}_focus",
        desc=f"QC{idx}: Primary research focus area is accurately described",
        parent=qc_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The center's primary research focus can be summarized as: '{item.focus}'.",
        node=focus_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "Allow paraphrasing. The summary should align with the main themes described on the page (e.g., quantum materials, "
            "sensors, networking, algorithms, error correction, etc.). If the summary contradicts the page, do not support."
        ),
    )

    # Uniqueness for the second quantum center: must differ from first QC state
    if idx == 2:
        if seen_states:
            prev_states_list_str = ", ".join(seen_states)
        else:
            prev_states_list_str = "N/A"
        unique_leaf = evaluator.add_custom_node(
            result=_non_empty(item.state) and (_normalize_state(item.state) not in {_normalize_state(s) for s in seen_states}),
            id=f"qc{idx}_state_unique",
            desc=f"QC{idx}: State is different from previously listed quantum center states [{prev_states_list_str}]",
            parent=qc_node,
            critical=True,
        )

    # Update seen states
    if _non_empty(item.state):
        seen_states.append(item.state.strip())


async def verify_semiconductor_fab(
    evaluator: Evaluator,
    parent_node,
    item: SemiconductorFabItem,
    index: int,
    seen_states: List[str],
) -> None:
    """
    Build and verify a single semiconductor fab subtree.
    """
    idx = index + 1
    sf_node = evaluator.add_parallel(
        id=f"semiconductor_fab_{idx}",
        desc=f"{'First' if idx == 1 else 'Second'} semiconductor facility identification with all required details",
        parent=parent_node,
        critical=False,
    )

    # Reference presence (critical gate)
    ref_present = evaluator.add_custom_node(
        result=_non_empty(item.reference_url),
        id=f"sf{idx}_reference_present",
        desc=f"SF{idx}: Reference URL is provided",
        parent=sf_node,
        critical=True,
    )

    # Reference official company source (critical)
    ref_official = evaluator.add_leaf(
        id=f"sf{idx}_reference",
        desc=f"SF{idx}: Reference URL from official company source is provided",
        parent=sf_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This URL is an official Intel (intel.com) or TSMC (tsmc.com) webpage, including subdomains/newsroom/press pages."
        ),
        node=ref_official,
        sources=item.reference_url,
        additional_instruction=(
            "Use the URL shown to confirm the domain is intel.com or tsmc.com (including subdomains). "
            "If missing or not a company domain, the claim is not supported."
        ),
    )

    # Facility name/designation provided (critical existence)
    name_leaf = evaluator.add_custom_node(
        result=_non_empty(item.name),
        id=f"sf{idx}_name",
        desc=f"SF{idx}: Official facility name or designation is provided",
        parent=sf_node,
        critical=True,
    )

    # Company is Intel or TSMC and matches the facility operator (critical)
    company_leaf = evaluator.add_leaf(
        id=f"sf{idx}_company",
        desc=f"SF{idx}: Operating company is Intel or TSMC",
        parent=sf_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The facility is operated by '{item.company}', and the operator is either Intel or TSMC."
        ),
        node=company_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "Confirm the page attributes the facility to Intel or TSMC. Accept 'Taiwan Semiconductor Manufacturing Company' "
            "as TSMC. If the operator is not Intel/TSMC or cannot be confirmed, do not support."
        ),
    )

    # State location supported by the reference (critical)
    state_supported_leaf = evaluator.add_leaf(
        id=f"sf{idx}_state_supported",
        desc=f"SF{idx}: State location is provided",
        parent=sf_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The facility '{item.name}' is located in the U.S. state of '{item.state}'.",
        node=state_supported_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "It's acceptable if the page mentions the city and state (e.g., Chandler, Arizona). "
            "Infer the state correctly if the location is clearly indicated."
        ),
    )

    # State uniqueness (critical)
    if seen_states:
        prev_states_list_str = ", ".join(seen_states)
    else:
        prev_states_list_str = "N/A"

    unique_condition = _non_empty(item.state) and (_normalize_state(item.state) not in {_normalize_state(s) for s in seen_states})
    unique_leaf = evaluator.add_custom_node(
        result=unique_condition,
        id=f"sf{idx}_state_unique",
        desc=f"SF{idx}: State is different from previously listed states [{prev_states_list_str}]",
        parent=sf_node,
        critical=True,
    )

    # Operational/announced >= 2020 (critical)
    operational_leaf = evaluator.add_leaf(
        id=f"sf{idx}_operational",
        desc=f"SF{idx}: Facility is operational or has publicly announced construction plans as of 2020 or later",
        parent=sf_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This facility is operational or has publicly announced construction/groundbreaking/plans in or after 2020 (year >= 2020)."
        ),
        node=operational_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "Look for words like 'announced', 'groundbreaking', 'construction', 'operational', 'opening' with a year. "
            "If the relevant year is not clearly >= 2020, do not support."
        ),
    )

    # Process node/technology provided and supported (critical)
    process_leaf = evaluator.add_leaf(
        id=f"sf{idx}_process",
        desc=f"SF{idx}: Primary process node or technology specification is provided",
        parent=sf_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The facility's primary process/node or technology is '{item.process}'.",
        node=process_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "Accept common naming like '5nm', '3nm', 'N5', 'N3', 'N4P', 'Intel 4', 'Intel 3', '20A', '18A', etc. "
            "If the page suggests a different node or does not support the claim, do not support."
        ),
    )

    # Update seen states
    if _non_empty(item.state):
        seen_states.append(item.state.strip())


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
    Evaluate an answer for the QIS + Semiconductor facilities task and return a structured summary.
    """
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction",
    )

    # Normalize and select first two of each category; pad if fewer provided
    qcs: List[QuantumCenterItem] = list(extracted.quantum_centers or [])
    fabs: List[SemiconductorFabItem] = list(extracted.semiconductor_fabs or [])

    if len(qcs) < 2:
        qcs = qcs + [QuantumCenterItem() for _ in range(2 - len(qcs))]
    else:
        qcs = qcs[:2]

    if len(fabs) < 2:
        fabs = fabs + [SemiconductorFabItem() for _ in range(2 - len(fabs))]
    else:
        fabs = fabs[:2]

    # Build category nodes
    qc_category = evaluator.add_parallel(
        id="quantum_research_centers_category",
        desc="Identify 2 DOE National Quantum Information Science Research Centers",
        parent=root,
        critical=False,
    )

    semi_category = evaluator.add_parallel(
        id="semiconductor_facilities_category",
        desc="Identify 2 semiconductor fabrication facilities operated by Intel or TSMC",
        parent=root,
        critical=False,
    )

    # Track states to enforce uniqueness
    seen_states: List[str] = []

    # Verify Quantum Centers
    await verify_quantum_center(evaluator, qc_category, qcs[0], 0, seen_states)
    await verify_quantum_center(evaluator, qc_category, qcs[1], 1, seen_states)

    # Verify Semiconductor Fabs
    await verify_semiconductor_fab(evaluator, semi_category, fabs[0], 0, seen_states)
    await verify_semiconductor_fab(evaluator, semi_category, fabs[1], 1, seen_states)

    # Optional global check: all four states distinct (critical to overall usefulness? keep non-critical but informative)
    normalized_nonempty = [_normalize_state(s) for s in seen_states if _non_empty(s)]
    all_distinct = (len(normalized_nonempty) == 4) and (len(set(normalized_nonempty)) == 4)
    evaluator.add_custom_node(
        result=all_distinct,
        id="global_states_all_distinct",
        desc="All four facilities are located in different U.S. states",
        parent=root,
        critical=False,
    )

    return evaluator.get_summary()