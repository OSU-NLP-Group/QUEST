import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pa_commercial_kennel_requirements"
TASK_DESCRIPTION = (
    "What are the specific operational and regulatory requirements that a commercial dog boarding kennel in Pennsylvania "
    "must meet to obtain and maintain a kennel license from the Pennsylvania Department of Agriculture? Please provide "
    "comprehensive information covering: (1) licensing threshold and application requirements, (2) minimum space standards "
    "per dog, (3) enclosure height requirements, (4) temperature control standards including both minimum and maximum limits, "
    "(5) mechanical ventilation system specifications including airflow rates and fresh air requirements, (6) auxiliary ventilation "
    "requirements for elevated temperatures, (7) air filtration standards, (8) humidity control requirements under different "
    "temperature conditions, (9) environmental monitoring and recording requirements, (10) ammonia level limits, "
    "(11) professional engineer certification requirements for ventilation systems, (12) insurance requirements, and "
    "(13) emergency response protocols for system malfunctions."
)

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class SourceExtraction(BaseModel):
    """Extract all source URLs cited in the answer."""
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_sources() -> str:
    return (
        "Extract all URLs explicitly mentioned in the answer that are presented as references or sources supporting the "
        "regulatory requirements (Pennsylvania Department of Agriculture, Pennsylvania Code/Regulations, official guidance, etc.). "
        "Include URLs whether they appear as plain links or markdown links. Return them in the 'source_urls' array. "
        "If there are no URLs, return an empty array."
    )


# --------------------------------------------------------------------------- #
# Helper for additional instructions                                          #
# --------------------------------------------------------------------------- #
def build_additional_instruction_for_requirement(
    sources_present: bool,
    requirement_hint: Optional[str] = None
) -> str:
    """
    Build an additional instruction for URL-backed verification.
    If sources are present, instruct the judge to verify strictly against the cited webpages.
    If no sources are present, instruct the judge to mark as not supported.
    """
    base = []
    if sources_present:
        base.append(
            "Verify the claim strictly against the provided URL sources. Prefer official Pennsylvania Department of Agriculture "
            "documents, Pennsylvania Code/Regulations, or other authoritative state sources. Allow minor wording variants (e.g., "
            "degree symbol vs 'F', small punctuation differences), but the numeric thresholds, time limits, and specifications must match."
        )
    else:
        base.append(
            "No sources were provided in the answer. You must conclude the claim is NOT SUPPORTED and mark it incorrect."
        )
    if requirement_hint:
        base.append(f"Specific guidance: {requirement_hint}")
    return " ".join(base)


# --------------------------------------------------------------------------- #
# Build verification nodes and claims                                         #
# --------------------------------------------------------------------------- #
def build_requirement_claims(
    evaluator: Evaluator,
    parent_node,
    sources: List[str],
) -> List[tuple[str, List[str] | None, Any, str]]:
    """
    Create leaf nodes for each requirement and return list of (claim, sources, node, additional_instruction)
    for batch verification. Some items (coverage checks) will be verified separately via simple_verify.
    """
    claims_and_nodes: List[tuple[str, List[str] | None, Any, str]] = []
    has_sources = bool(sources)

    def add_req_leaf(node_id: str, desc: str, claim: str, critical: bool, hint: Optional[str] = None):
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent_node,
            critical=critical,
        )
        add_ins = build_additional_instruction_for_requirement(has_sources, requirement_hint=hint)
        srcs: List[str] | None = sources if has_sources else None
        claims_and_nodes.append((claim, srcs, node, add_ins))

    # 1) Kennel license threshold (critical)
    add_req_leaf(
        "Kennel_License_Threshold",
        "Kennel license is required from the Pennsylvania Department of Agriculture for facilities housing/keeping/transferring 26 or more dogs in a calendar year",
        "A kennel license is required by the Pennsylvania Department of Agriculture when a facility houses, keeps, or transfers 26 or more dogs within a single calendar year.",
        critical=True,
        hint="Look for the numeric threshold '26 or more dogs in a calendar year' triggering licensing by PDA."
    )

    # 2) Application and maintenance coverage (simple verify; not URL-based)
    # We verify coverage directly against the answer content.
    coverage_node = evaluator.add_leaf(
        id="License_Application_and_Maintenance_Info_Included",
        desc="Answer includes the licensing application/maintenance requirements (i.e., what must be done to obtain and maintain the license), beyond only stating the threshold",
        parent=parent_node,
        critical=True
    )
    coverage_claim = (
        "The answer includes licensing application and ongoing maintenance requirements beyond just stating the threshold—"
        "such as application submission to the Pennsylvania Department of Agriculture, inspection/approval steps, fees/payment, "
        "renewal timing, compliance with regulations, and recordkeeping."
    )
    coverage_instruction = (
        "Judge this solely by reading the provided answer. If the answer only states the threshold and omits application "
        "process and maintenance obligations (e.g., inspections, fees, renewal cadence, compliance/recordkeeping), mark incorrect."
    )
    # This one is simple_verify (no URLs)
    # We will return it separately since it's not URL-backed:
    # We'll append as a special-case item with sources None and specific instruction.
    claims_and_nodes.append((coverage_claim, None, coverage_node, coverage_instruction))

    # 3) Minimum floor space PA weight-based (critical)
    add_req_leaf(
        "Minimum_Floor_Space_PA_Weight_Based",
        "Minimum floor space per dog follows Pennsylvania weight-based standards: 5 sq ft for dogs ≤25 lbs; 8 sq ft for dogs >25 lbs and ≤50 lbs",
        "Minimum floor space per dog under Pennsylvania standards is: 5 square feet for dogs weighing 25 pounds or less; and 8 square feet for dogs weighing more than 25 pounds up to and including 50 pounds.",
        critical=True,
        hint="Confirm the two explicit categories: 5 sq ft ≤25 lbs; 8 sq ft >25 lbs and ≤50 lbs."
    )

    # 4) USDA alternative method (critical)
    add_req_leaf(
        "Minimum_Floor_Space_USDA_Alternative_Method",
        "Answer includes the USDA alternative space calculation method: measure nose-to-base-of-tail length, add 6 inches, then square the total to get minimum floor space (in square inches)",
        "The USDA alternative minimum floor space method is: measure the dog's length from the tip of the nose to the base of the tail, add six inches, then square the total to calculate the minimum floor space in square inches.",
        critical=True,
        hint="Look for formula language 'length + 6 inches; then square the total' producing square inches."
    )

    # 5) Interior height requirement (critical)
    add_req_leaf(
        "Interior_Height_Requirement",
        "Primary enclosure interior height is at least 6 inches higher than the head of the tallest dog when standing normally",
        "The primary enclosure's interior height must be at least six inches higher than the head of the tallest dog when the dog is standing in a normal posture.",
        critical=True,
        hint="Look for '6 inches higher than the tallest dog's head' requirement."
    )

    # 6) Minimum temperature (acclimated dogs) (critical)
    add_req_leaf(
        "Minimum_Temperature_Acclimated",
        "Indoor temperature does not fall below 45°F for more than 4 consecutive hours for acclimated dogs",
        "For acclimated dogs, indoor temperature may not drop below 45°F for more than four consecutive hours.",
        critical=True,
        hint="Confirm both 45°F and 'more than four consecutive hours'."
    )

    # 7) Minimum temperature (sensitive groups) (critical)
    add_req_leaf(
        "Minimum_Temperature_Sensitive_Groups",
        "Indoor temperature does not fall below 50°F for non-acclimated dogs and cold-sensitive/at-risk groups (short-haired/toy breeds, elderly, young, sick, injured)",
        "For non-acclimated or temperature‑sensitive dogs (e.g., short‑haired or toy breeds, elderly, young, sick, or injured), indoor temperature may not drop below 50°F.",
        critical=True,
        hint="Confirm 50°F threshold and the sensitive categories."
    )

    # 8) Maximum temperature standard (critical)
    add_req_leaf(
        "Maximum_Temperature_Standard",
        "Temperature does not exceed 85°F for more than 4 consecutive hours",
        "Indoor temperature must not exceed 85°F for more than four consecutive hours.",
        critical=True,
        hint="Confirm both 85°F and 'more than four consecutive hours'."
    )

    # 9) Mechanical ventilation total airflow (critical)
    add_req_leaf(
        "Mechanical_Ventilation_Total_Airflow",
        "Mechanical ventilation provides minimum total airflow of 100 CFM per dog at all times",
        "The mechanical ventilation system must provide a minimum total airflow of 100 cubic feet per minute (CFM) per dog at all times.",
        critical=True,
        hint="Look for '100 CFM per dog' as a continuous requirement."
    )

    # 10) Mechanical ventilation fresh air (critical)
    add_req_leaf(
        "Mechanical_Ventilation_Fresh_Air",
        "At least 30 CFM per dog of circulated air is fresh outdoor air",
        "Of the circulated air, at least 30 CFM per dog must be fresh outdoor air.",
        critical=True,
        hint="Confirm '≥30 CFM per dog' of outdoor fresh air component."
    )

    # 11) Auxiliary ventilation trigger (critical)
    add_req_leaf(
        "Auxiliary_Ventilation_Trigger",
        "When temperature exceeds 85°F for any length of time, auxiliary ventilation is provided",
        "Whenever indoor temperature exceeds 85°F for any length of time, auxiliary ventilation must be provided.",
        critical=True,
        hint="Key phrase: trigger once temperature goes above 85°F."
    )

    # 12) Auxiliary ventilation airflow when over 85°F (critical)
    add_req_leaf(
        "Auxiliary_Ventilation_Airflow_When_Over_85F",
        "When temperature exceeds 85°F, auxiliary ventilation increases total airflow to 200 CFM per dog",
        "When temperature exceeds 85°F, auxiliary ventilation must increase total airflow to 200 CFM per dog.",
        critical=True,
        hint="Confirm '200 CFM per dog' when >85°F."
    )

    # 13) Air filtration MERV (critical)
    add_req_leaf(
        "Air_Filtration_MERV",
        "Ventilation uses disposable filters rated MERV 8 or higher",
        "The ventilation system must use disposable air filters rated at least MERV 8.",
        critical=True,
        hint="Look for 'MERV 8 or higher' disposable filters."
    )

    # 14) Filter replacement schedule (critical)
    add_req_leaf(
        "Filter_Replacement_Schedule",
        "Filters are replaced at least quarterly for equipment serving dog-housing areas",
        "Filters for ventilation equipment servicing dog‑housing areas must be replaced at least quarterly.",
        critical=True,
        hint="Confirm 'at least quarterly' replacement."
    )

    # 15) Humidity range when below 85°F (critical)
    add_req_leaf(
        "Humidity_Range_When_Below_85F",
        "When temperature is below 85°F, relative humidity is maintained between 30% and 70%",
        "When indoor temperature is below 85°F, relative humidity shall be maintained between 30% and 70%.",
        critical=True,
        hint="Confirm RH range '30% to 70%' when temperature < 85°F."
    )

    # 16) Heat index target when over 85°F (critical)
    add_req_leaf(
        "Heat_Index_Target_When_Over_85F",
        "When temperature exceeds 85°F, humidity is reduced so that Heat Index reaches 85 or lower within 4 hours",
        "When the indoor temperature exceeds 85°F, humidity must be reduced so that the Heat Index reaches 85 or lower within four hours.",
        critical=True,
        hint="Confirm 'Heat Index ≤ 85 within 4 hours when >85°F'."
    )

    # 17) Heat index maximum never exceeded (critical)
    add_req_leaf(
        "Heat_Index_Maximum_Never_Exceeded",
        "Heat Index never exceeds 90 in any area where dogs are housed",
        "In any area where dogs are housed, the Heat Index must never exceed 90.",
        critical=True,
        hint="Confirm 'Heat Index 90 is never exceeded'."
    )

    # 18) Environmental monitoring hourly recording (critical)
    add_req_leaf(
        "Environmental_Monitoring_Hourly_Recording",
        "Temperature and humidity are measured and recorded hourly in each room housing dogs",
        "Temperature and humidity must be measured and recorded hourly in each room where dogs are housed.",
        critical=True,
        hint="Look for 'hourly measurement and recording in each room housing dogs'."
    )

    # 19) Temperature device accuracy (critical)
    add_req_leaf(
        "Monitoring_Device_Accuracy_Temperature",
        "Temperature measurement devices are accurate to within 1°F",
        "Temperature measurement devices used must be accurate to within 1°F.",
        critical=True,
        hint="Confirm '±1°F accuracy'."
    )

    # 20) Humidity device accuracy (critical)
    add_req_leaf(
        "Monitoring_Device_Accuracy_Humidity",
        "Humidity measurement devices are accurate to within ±2% relative humidity from 10% to 90% RH",
        "Humidity measurement devices must be accurate to within ±2% relative humidity over the range of 10% to 90% RH.",
        critical=True,
        hint="Confirm '±2% RH accuracy from 10% to 90% RH'."
    )

    # 21) Ammonia level limit (critical)
    add_req_leaf(
        "Ammonia_Level_Limit",
        "Ammonia at dog height does not exceed 15 ppm, except within 30 minutes of completion of active sanitation",
        "Measured at dog height, ammonia concentration must not exceed 15 ppm, except within 30 minutes following completion of active sanitation.",
        critical=True,
        hint="Confirm '≤15 ppm at dog height; exception within 30 minutes after active sanitation'."
    )

    # 22) Professional engineer certification required (critical)
    add_req_leaf(
        "Professional_Engineer_Certification_Required",
        "Ventilation system is certified by a licensed professional engineer verifying compliance with Pennsylvania commercial kennel ventilation/humidity/airflow standards",
        "The kennel's ventilation system must be certified by a licensed professional engineer who verifies compliance with Pennsylvania commercial kennel standards for ventilation, humidity, and airflow.",
        critical=True,
        hint="Look for 'licensed professional engineer certification' and scope covering ventilation/humidity/airflow compliance."
    )

    # 23) PE certification contents (critical)
    add_req_leaf(
        "Professional_Engineer_Certification_Contents",
        "The professional engineer certification includes the required elements: total cubic feet of facility; CFM capacity of ventilation equipment; humidity control system description; auxiliary ventilation system description; maximum number of dogs",
        "The professional engineer certification must include: total cubic feet of the facility; CFM capacity of the ventilation equipment; a description of the humidity control system; a description of the auxiliary ventilation system; and the maximum number of dogs the facility may house.",
        critical=True,
        hint="Confirm all five elements are explicitly required in certification contents."
    )

    # 24) General liability insurance (non-critical)
    add_req_leaf(
        "General_Liability_Insurance",
        "Answer addresses insurance requirements (including the stated typical coverage levels: $1,000,000 per occurrence and $2,000,000 aggregate annually)",
        "Kennels must maintain general liability insurance, commonly specified at $1,000,000 per occurrence and $2,000,000 aggregate annually.",
        critical=False,
        hint="Accept 'typical' phrasing; verify if sources mention liability coverage levels around $1M per occurrence and $2M aggregate annually."
    )

    # 25) Emergency protocol: malfunction below 85°F (critical)
    add_req_leaf(
        "Emergency_Protocol_Malfunction_Below_85F",
        "If ventilation malfunction occurs and temperature remains below 85°F, the issue is corrected within 72 hours",
        "If the ventilation system malfunctions and temperature remains below 85°F, the issue must be corrected within 72 hours.",
        critical=True,
        hint="Confirm '≤72 hours correction when temperature < 85°F'."
    )

    # 26) Emergency protocol: malfunction above 85°F (critical)
    add_req_leaf(
        "Emergency_Protocol_Malfunction_Above_85F",
        "If ventilation malfunction occurs and temperature exceeds 85°F, the issue is corrected within 4 hours and the kennel's veterinarian is immediately notified",
        "If the ventilation system malfunctions and the indoor temperature exceeds 85°F, the issue must be corrected within four hours and the kennel's veterinarian must be notified immediately.",
        critical=True,
        hint="Confirm both '≤4 hours correction' and 'immediate veterinarian notification' when >85°F."
    )

    # 27) Emergency protocol: 24h notification to PDA (critical)
    add_req_leaf(
        "Emergency_Protocol_Malfunction_24h_Notification",
        "If malfunction reaches or exceeds 24 hours, the Pennsylvania Department of Agriculture is notified",
        "If a ventilation malfunction reaches or exceeds 24 hours, the Pennsylvania Department of Agriculture must be notified.",
        critical=True,
        hint="Confirm 'PDA notification at ≥24 hours malfunction duration'."
    )

    # 28) Carbon monoxide detectors (critical)
    add_req_leaf(
        "Carbon_Monoxide_Detectors",
        "Carbon monoxide detectors meeting UL 2034 or IAS 6-96 standards are installed in rooms with carbon monoxide-emitting devices",
        "Carbon monoxide detectors meeting UL 2034 or IAS 6‑96 must be installed in rooms that contain carbon monoxide‑emitting devices.",
        critical=True,
        hint="Confirm detector standards UL 2034 or IAS 6‑96."
    )

    return claims_and_nodes


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate the provided answer for Pennsylvania commercial kennel regulatory requirements using the Mind2Web2 framework.
    """
    # Initialize evaluator (root is non-critical to allow partial credit for non-critical child like insurance)
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

    # Extract source URLs from the answer
    sources_extraction = await evaluator.extract(
        prompt=prompt_extract_sources(),
        template_class=SourceExtraction,
        extraction_name="source_urls",
    )
    all_sources = sources_extraction.source_urls if sources_extraction and sources_extraction.source_urls else []

    # Create main verification group node (parallel aggregation, non-critical to permit partial scoring on entire task)
    pa_group = evaluator.add_parallel(
        id="Pennsylvania_Commercial_Kennel_Requirements",
        desc="Evaluate all regulatory and operational requirements for a Pennsylvania commercial dog boarding kennel to obtain/maintain a kennel license",
        parent=root,
        critical=False
    )

    # Build all requirement claims and their nodes
    claims_and_nodes = build_requirement_claims(evaluator, pa_group, all_sources)

    # Perform verifications (parallelize via batch_verify)
    # Each tuple: (claim, sources, node, additional_instruction)
    await evaluator.batch_verify(claims_and_nodes)

    # Add custom info about URLs used
    evaluator.add_custom_info(
        info={
            "total_urls_extracted": len(all_sources),
            "urls": all_sources[:20]  # Limit to 20 for summary compactness
        },
        info_type="url_info",
        info_name="extracted_source_urls"
    )

    # Return evaluation summary
    return evaluator.get_summary()