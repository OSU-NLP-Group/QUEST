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
TASK_ID = "us_5g_sa_nationwide_first_two"
TASK_DESCRIPTION = """Identify the first two telecommunications operators to deploy commercial 5G Standalone (SA) networks with nationwide coverage in the United States. For each operator, provide the following information: (1) Operator/carrier name, (2) Nationwide 5G SA launch or completion date (month and year), (3) Primary mid-band spectrum frequency bands used for the deployment, (4) Approximate US population coverage percentage achieved, (5) Key 5G SA technical capabilities implemented (must include: Service-Based Architecture status, network slicing capability, and cloud-native network functions), (6) At least one authoritative reference URL supporting the deployment details. Requirements and constraints: Only commercial nationwide 5G Standalone deployments qualify (not Non-Standalone NSA, and not regional/city-specific SA deployments); The 5G core network must implement Service-Based Architecture (SBA) as defined by 3GPP standards; Deployments must cover a significant portion of the US population (generally >50 million people or >15% of US population at minimum); List operators in chronological order by their nationwide SA deployment date.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OperatorCapabilities(BaseModel):
    sba: Optional[str] = None                 # e.g., "Yes", "Implemented", "SBA-based", or phrase from answer
    five_gc: Optional[str] = None             # e.g., "Yes", "Dedicated 5G Core", or phrase
    network_slicing: Optional[str] = None     # e.g., "Yes", "Supported", or phrase
    cloud_native: Optional[str] = None        # e.g., "Cloud-native CNFs", "Kubernetes", or phrase


class OperatorInfo(BaseModel):
    name: Optional[str] = None
    date_month: Optional[str] = None          # launch/completion month (e.g., "August")
    date_year: Optional[str] = None           # year (e.g., "2020")
    spectrum_bands: List[str] = Field(default_factory=list)  # e.g., ["2.5 GHz (n41)", "C-band (3.7-3.98 GHz)"]
    coverage: Optional[str] = None            # e.g., "200M people", "80% population", "nationwide"
    capabilities: Optional[OperatorCapabilities] = None
    reference_urls: List[str] = Field(default_factory=list)  # authoritative references


class OperatorsExtraction(BaseModel):
    first: Optional[OperatorInfo] = None
    second: Optional[OperatorInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_operators() -> str:
    return """
You will extract structured information for the first two U.S. telecommunications operators that, according to the answer text, deployed commercial 5G Standalone (SA) networks with nationwide coverage. Extract exactly two operators (the first and the second mentioned or listed), preserving the ordering presented in the answer or the explicit “first/second” labeling in the answer.

For each operator, extract the following fields exactly as they appear in the answer:
- name: The operator/carrier name (e.g., "T-Mobile", "Verizon", "AT&T").
- date_month: The month of the nationwide 5G SA launch (or completion) as written (e.g., "August", "Nov.", "November"). Keep the month text as-is.
- date_year: The year of the nationwide 5G SA launch (or completion) as written (e.g., "2020").
- spectrum_bands: A list of the primary mid-band frequency bands used for the deployment as written in the answer (e.g., ["2.5 GHz", "C-band 3.7-3.98 GHz", "3.45 GHz", "CBRS 3.5 GHz"]). If multiple bands are mentioned, include all. If none are mentioned, return an empty list.
- coverage: The approximate U.S. population coverage percentage or count (e.g., "80% population", "200M+ people", "nationwide"). If not stated, return null.
- capabilities: A nested object with:
  - sba: The Service-Based Architecture (SBA) implementation status if mentioned (e.g., "SBA-based 5G core", "SBA implemented"). If not stated, return null.
  - five_gc: 5G Core presence/status if mentioned (e.g., "dedicated 5G Core", "5GC deployed"). If not stated, return null.
  - network_slicing: Network slicing capability status if mentioned (e.g., "network slicing supported/launched"). If not stated, return null.
  - cloud_native: Cloud-native network functions if mentioned (e.g., "cloud-native CNFs", "containerized", "Kubernetes"). If not stated, return null.
- reference_urls: A list of all authoritative reference URLs explicitly provided in the answer for this operator. Extract actual URLs only. If none, return an empty list.

Return a JSON object with two top-level fields: "first" and "second", each being an object with the fields above. If the answer only provides one operator, fill the missing one with null.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                             #
# --------------------------------------------------------------------------- #
def _join_list(values: List[str]) -> str:
    return ", ".join([v for v in values if v]) if values else ""


def _urls_for_op(op: Optional[OperatorInfo]) -> List[str]:
    return (op.reference_urls if (op and op.reference_urls) else [])


def _date_str(op: Optional[OperatorInfo]) -> str:
    m = op.date_month if (op and op.date_month) else ""
    y = op.date_year if (op and op.date_year) else ""
    return f"{m} {y}".strip()


# --------------------------------------------------------------------------- #
# Verification logic per operator                                             #
# --------------------------------------------------------------------------- #
async def verify_operator(
    evaluator: Evaluator,
    parent_node,
    op: Optional[OperatorInfo],
    which: str,  # "First" or "Second"
    other_op: Optional[OperatorInfo] = None
) -> None:
    """
    Build verification sub-tree for a single operator (First/Second) as specified by the rubric.
    """
    # Create the operator node (parallel aggregation, non-critical at this level to allow partial credit)
    op_node = evaluator.add_parallel(
        id=f"{which}_Operator",
        desc=f"{which} telecommunications operator to deploy nationwide 5G SA in the United States",
        parent=parent_node,
        critical=False
    )

    # References existence (Critical)
    ref_present = bool(op and op.reference_urls)
    evaluator.add_custom_node(
        result=ref_present,
        id=f"{which}_Op_Reference",
        desc=f"URL reference supporting {which.lower()} operator's nationwide 5G SA deployment",
        parent=op_node,
        critical=True
    )

    # Identity (Critical)
    identity_node = evaluator.add_parallel(
        id=f"{which}_Op_Identity",
        desc=f"Correct identification of the {which.lower()} nationwide 5G SA operator",
        parent=op_node,
        critical=True
    )

    # Name leaf (Critical)
    name_leaf = evaluator.add_leaf(
        id=f"{which}_Op_Name",
        desc="Operator/carrier name correctly stated",
        parent=identity_node,
        critical=True
    )
    name_val = op.name if op and op.name else ""
    await evaluator.verify(
        claim=f"The referenced page(s) are about the operator/carrier named '{name_val}'.",
        node=name_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Verify that the operator name appears clearly on the page as the carrier deploying the 5G SA network. Allow minor naming variants (e.g., 'T-Mobile US' vs 'T-Mobile')."
    )

    # Deployment type (Critical)
    dep_node = evaluator.add_parallel(
        id=f"{which}_Op_Deployment_Type",
        desc="Deployment type verification",
        parent=identity_node,
        critical=True
    )

    # Nationwide confirmed (Critical)
    nationwide_leaf = evaluator.add_leaf(
        id=f"{which}_Op_Nationwide_Confirmed",
        desc="Deployment confirmed as nationwide (not regional)",
        parent=dep_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The deployment is nationwide across the United States for {name_val}.",
        node=nationwide_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Look for explicit text such as 'nationwide', 'across the U.S.', coverage across all 50 states or a vast majority of the U.S. population. If the source is clearly regional/city-specific only, this should fail."
    )

    # SA confirmed (Critical)
    sa_leaf = evaluator.add_leaf(
        id=f"{which}_Op_SA_Confirmed",
        desc="Deployment confirmed as Standalone (SA), not Non-Standalone (NSA)",
        parent=dep_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The deployment for {name_val} is 5G Standalone (SA), not Non-Standalone (NSA).",
        node=sa_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Check that the page explicitly states '5G Standalone (SA)' or equivalent terminology indicating SA core/SA mode, not merely NSA."
    )

    # Commercial confirmed (Critical)
    commercial_leaf = evaluator.add_leaf(
        id=f"{which}_Op_Commercial",
        desc="Confirmed as commercial deployment (not trial)",
        parent=dep_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The deployment for {name_val} is a commercial, production 5G SA deployment (not a trial or limited pilot).",
        node=commercial_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Look for language like 'commercial', 'production', 'launched for customers', or availability to subscribers. If only a trial or limited pilot is mentioned, this should fail."
    )

    # Timeline (Critical)
    time_node = evaluator.add_parallel(
        id=f"{which}_Op_Timeline",
        desc=f"{'Launch' if which=='First' else 'Completion'} date accurately provided",
        parent=op_node,
        critical=True
    )

    # Month (Critical)
    month_leaf = evaluator.add_leaf(
        id=f"{which}_Op_Month",
        desc=f"{'Launch' if which=='First' else 'Completion'} month correctly stated",
        parent=time_node,
        critical=True
    )
    month_val = op.date_month if (op and op.date_month) else ""
    await evaluator.verify(
        claim=f"The month of the nationwide 5G SA {'launch' if which=='First' else 'completion'} for {name_val} is '{month_val}'.",
        node=month_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Focus only on confirming the month (ignore the specific day). Accept abbreviated forms (e.g., 'Nov.' for 'November') as equivalent."
    )

    # Year (Critical)
    year_leaf = evaluator.add_leaf(
        id=f"{which}_Op_Year",
        desc=f"{'Launch' if which=='First' else 'Completion'} year correctly stated",
        parent=time_node,
        critical=True
    )
    year_val = op.date_year if (op and op.date_year) else ""
    await evaluator.verify(
        claim=f"The year of the nationwide 5G SA {'launch' if which=='First' else 'completion'} for {name_val} is '{year_val}'.",
        node=year_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Focus only on confirming the year (ignore month and day)."
    )

    # Timeline ordering verification (Critical)
    time_order_leaf = evaluator.add_leaf(
        id=f"{which}_Op_Timeline_Verification",
        desc=f"Timeline verified as {'first' if which=='First' else 'second'} nationwide SA deployment",
        parent=time_node,
        critical=True
    )
    # Build a simple logical claim using both operators' dates (if available)
    other_name = other_op.name if (other_op and other_op.name) else "the other operator"
    this_date = _date_str(op)
    other_date = _date_str(other_op)
    if which == "First":
        order_claim = f"{name_val} date '{this_date}' is earlier than {other_name} date '{other_date}'."
        order_instruction = "Compare month-year chronologically. If one of the dates is missing, use the available information and your best judgment from the strings; if insufficient, judge as incorrect."
    else:
        order_claim = f"{name_val} date '{this_date}' is later than {other_name} date '{other_date}'."
        order_instruction = "Compare month-year chronologically. If one of the dates is missing, use the available information and your best judgment from the strings; if insufficient, judge as incorrect."
    await evaluator.verify(
        claim=order_claim,
        node=time_order_leaf,
        additional_instruction=order_instruction
    )

    # Spectrum (Critical)
    spec_node = evaluator.add_parallel(
        id=f"{which}_Op_Spectrum",
        desc="Primary mid-band spectrum frequencies identified",
        parent=op_node,
        critical=True
    )

    # Frequency band(s) stated (Critical)
    band_leaf = evaluator.add_leaf(
        id=f"{which}_Op_Frequency_Band",
        desc="Specific frequency band(s) stated (e.g., 2.5 GHz, 3.5 GHz)",
        parent=spec_node,
        critical=True
    )
    bands_str = _join_list(op.spectrum_bands if (op and op.spectrum_bands) else [])
    await evaluator.verify(
        claim=f"The primary mid-band frequency bands used by {name_val} for the nationwide 5G SA deployment include: {bands_str}.",
        node=band_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Check that the referenced pages explicitly mention the frequency bands. Accept synonyms like 'n41' for 2.5 GHz, 'C-band (3.7–3.98 GHz)', '3.45 GHz', 'CBRS 3.5 GHz', etc."
    )

    # Mid-band confirmed (Critical)
    midband_leaf = evaluator.add_leaf(
        id=f"{which}_Op_Mid_Band_Confirmed",
        desc="Confirmed as mid-band spectrum (1-6 GHz range)",
        parent=spec_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The bands listed for {name_val} are mid-band (1–6 GHz) and are used as primary spectrum in the SA deployment.",
        node=midband_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Accept if the page calls them 'mid-band' or if the numerical frequencies fall within 1–6 GHz. If only mmWave or sub-1 GHz bands are mentioned without mid-band, this should fail."
    )

    # Coverage (Critical)
    cov_node = evaluator.add_parallel(
        id=f"{which}_Op_Coverage",
        desc="Population coverage information provided",
        parent=op_node,
        critical=True
    )

    # Coverage value stated (Critical)
    cov_val_leaf = evaluator.add_leaf(
        id=f"{which}_Op_Coverage_Value",
        desc="Coverage percentage or population count stated",
        parent=cov_node,
        critical=True
    )
    coverage_str = op.coverage if (op and op.coverage) else ""
    await evaluator.verify(
        claim=f"The coverage achieved for {name_val}'s nationwide 5G SA deployment is approximately '{coverage_str}'.",
        node=cov_val_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Verify that the page provides a coverage metric such as population count (e.g., '200 million people') or a percentage of U.S. population."
    )

    # Nationwide threshold met (Critical)
    cov_thresh_leaf = evaluator.add_leaf(
        id=f"{which}_Op_Nationwide_Threshold",
        desc="Coverage meets nationwide threshold (>15% US population)",
        parent=cov_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val}'s coverage for the SA deployment meets or exceeds a 'nationwide threshold' defined as covering at least 15% of the U.S. population or >50 million people.",
        node=cov_thresh_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Look for coverage statements like 'nationwide', 'covers 200M+ people', 'covers >15% population', etc. If coverage is clearly below the threshold or limited to a few cities, this should fail."
    )

    # Technical capabilities (Critical)
    tech_node = evaluator.add_parallel(
        id=f"{which}_Op_Technical_Capabilities",
        desc="Key 5G SA technical capabilities documented",
        parent=op_node,
        critical=True
    )

    # SBA (Critical)
    sba_leaf = evaluator.add_leaf(
        id=f"{which}_Op_SBA",
        desc="Service-Based Architecture (SBA) implementation confirmed",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val}'s 5G core for the SA deployment implements Service-Based Architecture (SBA) as per 3GPP.",
        node=sba_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Look for 'Service-Based Architecture', 'SBA', or equivalent architectural descriptions tied to the 5G core."
    )

    # 5GC (Critical)
    fivegc_leaf = evaluator.add_leaf(
        id=f"{which}_Op_5GC",
        desc="Dedicated 5G Core (5GC) network deployment confirmed",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"A dedicated 5G Core (5GC) is deployed for {name_val}'s 5G SA network.",
        node=fivegc_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Confirm the presence of a dedicated 5G Core for SA mode. Accept phrases like 'SA core', '5G core', '5GC' deployed for commercial traffic."
    )

    # Network slicing (Critical)
    slicing_leaf = evaluator.add_leaf(
        id=f"{which}_Op_Network_Slicing",
        desc="Network slicing capability stated",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Network slicing capability is implemented or supported in {name_val}'s 5G SA network.",
        node=slicing_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Look for 'network slicing', 'slice', enterprise slicing trials transitioning to commercial, or equivalent capability statements."
    )

    # Cloud-native (Critical)
    cloud_leaf = evaluator.add_leaf(
        id=f"{which}_Op_Cloud_Native",
        desc="Cloud-native network functions noted",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val}'s 5G SA core uses cloud-native network functions (CNFs), such as containerized microservices, Kubernetes, or public/private cloud deployments.",
        node=cloud_leaf,
        sources=_urls_for_op(op),
        additional_instruction="Accept evidence such as 'cloud-native CNFs', 'containerized', 'Kubernetes', 'cloud core', or vendor/operator statements indicating cloud-native 5G core functions."
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
    Evaluate an answer for the 'first two US nationwide 5G SA operators' task.
    """
    # Initialize evaluator with a parallel root
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

    # Add a grouping node for readability (non-critical to allow partial credit across operators)
    group_node = evaluator.add_parallel(
        id="Nationwide_5G_SA_Operators",
        desc="Identification and verification of the first two US telecommunications operators with nationwide 5G Standalone network deployments",
        parent=root,
        critical=False
    )

    # Extract operators info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_operators(),
        template_class=OperatorsExtraction,
        extraction_name="operators_extraction"
    )

    # Run verification for both operators (First and Second)
    first_op = extracted.first
    second_op = extracted.second

    await verify_operator(evaluator, group_node, first_op, "First", other_op=second_op)
    await verify_operator(evaluator, group_node, second_op, "Second", other_op=first_op)

    # Return evaluation summary
    return evaluator.get_summary()