import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "travel_plan_validation_2026"
TASK_DESCRIPTION = (
    "A US citizen is planning a 28-day vacation in late February 2026 that involves the following itinerary: "
    "(1) Domestic flight from their home city to a West Coast departure city using Allegiant Air, "
    "(2) International flight from the West Coast to Singapore, "
    "(3) A 5-night Royal Caribbean cruise on the Ovation of the Seas departing from Singapore, with ports of call including Penang (Malaysia), Phuket (Thailand), and Manila (Philippines), "
    "(4) Return journey via Singapore and domestic US flight. "
    "The traveler has the following constraints and items: US passport with expiration date of September 15, 2026; "
    "One personal item measuring 7\" x 13\" x 17\"; "
    "One checked bag weighing 48 pounds with dimensions totaling 78 linear inches; "
    "Plans to complete eTravel registration 48 hours before arriving in Manila; "
    "Trip dates: February 18, 2026 (departure) to March 18, 2026 (return to US). "
    "Verify whether this travel plan meets all mandatory requirements for: "
    "(1) Passport validity for Philippines entry (six-month rule), "
    "(2) Allegiant Air's operational capabilities for the domestic flight segments, "
    "(3) Allegiant Air's baggage restrictions for both personal item and checked bag, "
    "(4) Philippines eTravel registration timeframe compliance, "
    "(5) Verification that the described Royal Caribbean cruise operates from Singapore. "
    "Provide a complete analysis identifying which requirements are met and which, if any, are not met. "
    "Include reference URLs from official sources (government travel sites, airline websites, or cruise line websites) that support your verification for each requirement."
)

# Key plan attributes (from task description)
PASSPORT_EXPIRY = "September 15, 2026"
TRIP_DEPARTURE = "February 18, 2026"
TRIP_RETURN = "March 18, 2026"
PERSONAL_ITEM_DIMS = "7 x 13 x 17 inches"
CHECKED_BAG_WEIGHT_LBS = "48"
CHECKED_BAG_LINEAR_INCHES = "78"
CRUISE_SHIP = "Ovation of the Seas"
CRUISE_NIGHTS = "5"
CRUISE_DEPARTURE_PORT = "Singapore"
CRUISE_PORTS = ["Penang (Malaysia)", "Phuket (Thailand)", "Manila (Philippines)"]

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class RequirementURLsExtraction(BaseModel):
    """Extract official reference URLs the answer cites for each requirement."""
    passport_rule_urls: List[str] = Field(default_factory=list)
    visa_rule_urls: List[str] = Field(default_factory=list)
    allegiant_ops_urls: List[str] = Field(default_factory=list)
    baggage_policy_urls: List[str] = Field(default_factory=list)
    etravel_urls: List[str] = Field(default_factory=list)
    cruise_urls: List[str] = Field(default_factory=list)


class RequirementAssessmentsExtraction(BaseModel):
    """Extract the answer's stated assessment for each requirement."""
    # Use normalized labels: "met", "not_met", or "uncertain"
    passport_six_month_assessment: Optional[str] = None
    # Use "visa_not_required", "visa_required", or "uncertain"
    visa_requirement_assessment: Optional[str] = None
    # Allegiant operations assessment ("compatible", "not_compatible", or "uncertain")
    allegiant_domestic_assessment: Optional[str] = None
    # Baggage compliance ("compliant", "non_compliant", or "uncertain")
    baggage_personal_item_assessment: Optional[str] = None
    baggage_checked_bag_assessment: Optional[str] = None
    # eTravel timing ("compliant", "non_compliant", or "uncertain")
    etravel_timing_assessment: Optional[str] = None
    # Cruise operates ("verified", "not_verified", or "uncertain")
    cruise_operates_assessment: Optional[str] = None


class ItineraryDatesExtraction(BaseModel):
    """Extract any Philippines arrival/departure dates the answer provides."""
    manila_arrival_date: Optional[str] = None
    philippines_departure_date: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_requirement_urls() -> str:
    return (
        "Extract official reference URLs that the answer cites or mentions for the following checks. "
        "Only include official sources: government (.gov or .gov.ph/.ph government), the airline's official site (allegiantair.com), "
        "and the cruise line's official site (royalcaribbean.com). "
        "Return arrays for each category; if none are provided, return an empty array.\n"
        "- passport_rule_urls: Official government URLs stating the Philippines six-month passport validity rule.\n"
        "- visa_rule_urls: Official government URLs describing visa rules for US citizens entering the Philippines (30 days or less).\n"
        "- allegiant_ops_urls: Official Allegiant URLs that support their operational scope (domestic-only and/or route network).\n"
        "- baggage_policy_urls: Official Allegiant baggage policy URL(s) that state personal item limits and checked bag limits.\n"
        "- etravel_urls: Official Philippine government/eTravel URL(s) stating timing requirements for eTravel registration.\n"
        "- cruise_urls: Official Royal Caribbean ship/itinerary URL(s) confirming Ovation of the Seas departures from Singapore (preferably 2026).\n"
    )


def prompt_extract_assessments() -> str:
    return (
        "Extract the answer's explicit assessment for each requirement using the normalized values below. "
        "If the answer expresses uncertainty or lacks enough detail, use 'uncertain'. "
        "If the answer does not address the requirement, return null.\n"
        "- passport_six_month_assessment: 'met' | 'not_met' | 'uncertain'\n"
        "- visa_requirement_assessment: 'visa_not_required' | 'visa_required' | 'uncertain'\n"
        "- allegiant_domestic_assessment: 'compatible' | 'not_compatible' | 'uncertain'\n"
        "- baggage_personal_item_assessment: 'compliant' | 'non_compliant' | 'uncertain'\n"
        "- baggage_checked_bag_assessment: 'compliant' | 'non_compliant' | 'uncertain'\n"
        "- etravel_timing_assessment: 'compliant' | 'non_compliant' | 'uncertain'\n"
        "- cruise_operates_assessment: 'verified' | 'not_verified' | 'uncertain'\n"
    )


def prompt_extract_itinerary_dates() -> str:
    return (
        "Extract any explicit Philippines-specific dates mentioned in the answer:\n"
        "- manila_arrival_date: The date the traveler arrives in Manila (Philippines), if provided.\n"
        "- philippines_departure_date: The date the traveler departs the Philippines (e.g., cruises away from Manila), if provided.\n"
        "If unknown or not provided, return null for the corresponding field."
    )


# --------------------------------------------------------------------------- #
# Verification functions (subtrees)                                           #
# --------------------------------------------------------------------------- #
async def verify_passport_six_month_rule(
    evaluator: Evaluator,
    parent_node,
    urls: RequirementURLsExtraction,
    assessments: RequirementAssessmentsExtraction,
    itinerary_dates: ItineraryDatesExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Passport_Validity_Philippines_Six_Month_Rule",
        desc="Verifies whether the traveler’s passport validity satisfies the Philippines six-month rule given the itinerary, with an official government citation.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Assessment correctness (simple logical verification)
    assess_leaf = evaluator.add_leaf(
        id="Passport_Six_Month_Rule_Assessment",
        desc="Answer correctly determines whether the stated passport expiration date satisfies the Philippines passport-validity requirement (≥6 months beyond intended departure from the Philippines) based on the itinerary dates provided (or explicitly notes if the itinerary lacks sufficient detail to determine the Philippines departure date).",
        parent=node,
        critical=True,
    )

    assessment_label = assessments.passport_six_month_assessment or "uncertain"
    manila_arrival = itinerary_dates.manila_arrival_date or "unknown"
    ph_departure = itinerary_dates.philippines_departure_date or "unknown"

    claim_assess = (
        f"The answer’s determination about the Philippines six-month passport validity rule is appropriate for this plan: "
        f"passport expires on {PASSPORT_EXPIRY}; Manila arrival date in the answer: {manila_arrival}; Philippines departure date in the answer: {ph_departure}. "
        f"The rule requires validity for at least 6 months beyond the date of departure from the Philippines. "
        f"The answer’s final classification is '{assessment_label}', which correctly reflects the situation given the provided dates or the lack thereof."
    )
    await evaluator.verify(
        claim=claim_assess,
        node=assess_leaf,
        additional_instruction=(
            "Judge only whether the answer's conclusion is logically sound given the passport expiry and the timing of the Philippines segment. "
            "If the exact Philippines departure date is missing, considering it 'uncertain' can be correct. "
            "Do not verify the rule text here; that is covered by the URL verification."
        ),
    )

    # Leaf: Official reference URL support (government website)
    ref_leaf = evaluator.add_leaf(
        id="Passport_Six_Month_Rule_Official_Reference_URL",
        desc="Provides at least one official government URL supporting the Philippines passport-validity rule used in the assessment.",
        parent=node,
        critical=True,
    )

    claim_rule = (
        "This official government page explicitly states the Philippines passport validity rule that requires a passport to be valid for at least six months "
        "beyond the intended stay or departure from the Philippines."
    )
    await evaluator.verify(
        claim=claim_rule,
        node=ref_leaf,
        sources=urls.passport_rule_urls,
        additional_instruction=(
            "Only accept clear statements from official government sources (e.g., .gov, .gov.ph, official immigration or foreign affairs pages). "
            "If the page is not official or does not clearly state the six-month rule, mark as not supported."
        ),
    )


async def verify_visa_requirement(
    evaluator: Evaluator,
    parent_node,
    urls: RequirementURLsExtraction,
    assessments: RequirementAssessmentsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Philippines_Visa_Requirement_US_Citizen_30_Days_Or_Less",
        desc="Verifies whether a US citizen needs a tourist visa for the Philippines given the plan’s Philippines stay duration, with an official government citation.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Assessment correctness (simple logical verification)
    assess_leaf = evaluator.add_leaf(
        id="Visa_Requirement_Assessment",
        desc="Answer correctly determines whether the plan implies a Philippines stay of 30 days or less (or explicitly states if the Philippines length of stay cannot be determined from the itinerary) and therefore whether a tourist visa is required for a US citizen under the stated rule.",
        parent=node,
        critical=True,
    )

    visa_assess = assessments.visa_requirement_assessment or "uncertain"
    claim_assess = (
        f"The itinerary implies the Philippines stay is at most a short port call during the cruise and not more than 30 days. "
        f"For a US citizen, the answer’s conclusion '{visa_assess}' regarding the need for a tourist visa for stays of 30 days or less is correct given the described plan."
    )
    await evaluator.verify(
        claim=claim_assess,
        node=assess_leaf,
        additional_instruction=(
            "Use the task description and the answer as context. "
            "If the plan does not provide exact stay length in the Philippines, 'uncertain' can be correct. "
            "Do not verify the visa rule text here; that is covered by the URL verification."
        ),
    )

    # Leaf: Official reference URL support (government website)
    ref_leaf = evaluator.add_leaf(
        id="Visa_Requirement_Official_Reference_URL",
        desc="Provides at least one official government URL supporting the US-citizen visa requirement rule used in the assessment.",
        parent=node,
        critical=True,
    )

    claim_rule = (
        "This official government page states that US citizens may enter the Philippines without a tourist visa for stays of 30 days or less, "
        "provided other entry conditions are met (e.g., onward/return ticket, passport validity)."
    )
    await evaluator.verify(
        claim=claim_rule,
        node=ref_leaf,
        sources=urls.visa_rule_urls,
        additional_instruction=(
            "Only accept clear statements from official government sources (e.g., .gov, .gov.ph, official immigration or foreign affairs pages). "
            "If the page is not official or does not clearly state the 30-day visa-free policy for US citizens, mark as not supported."
        ),
    )


async def verify_allegiant_operations(
    evaluator: Evaluator,
    parent_node,
    urls: RequirementURLsExtraction,
    assessments: RequirementAssessmentsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Allegiant_Operational_Capability_Domestic_Segments",
        desc="Verifies Allegiant can be used for the domestic US legs as described (including recognizing Allegiant’s domestic-only scope per constraints), with an official Allegiant citation.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Assessment correctness (simple verify)
    assess_leaf = evaluator.add_leaf(
        id="Allegiant_Domestic_Segments_Assessment",
        desc="Answer correctly assesses whether Allegiant’s operations (as supported by cited official Allegiant information) are compatible with the plan’s domestic US flight usage as described, and does not claim Allegiant provides the international segments (or clearly states what cannot be verified if required route/city details are missing).",
        parent=node,
        critical=True,
    )

    allegiant_assess = assessments.allegiant_domestic_assessment or "compatible"
    claim_assess = (
        f"The answer correctly limits Allegiant Air usage to domestic US segments and does not claim Allegiant operates the international flight to Singapore; "
        f"this assessment ('{allegiant_assess}') is appropriate for Allegiant’s network."
    )
    await evaluator.verify(
        claim=claim_assess,
        node=assess_leaf,
        additional_instruction=(
            "Judge only the answer’s conclusion using task context and common knowledge that Allegiant is a US-based low-cost carrier focused on domestic routes. "
            "Do not verify network details here; that is covered by the URL verification."
        ),
    )

    # Leaf: Official reference URL support (allegiantair.com)
    ref_leaf = evaluator.add_leaf(
        id="Allegiant_Operations_Official_Reference_URL",
        desc="Provides at least one official Allegiant URL supporting the operational-capability claim used in the assessment (airline website pages, not third-party sources).",
        parent=node,
        critical=True,
    )

    claim_rule = (
        "This official Allegiant webpage supports that Allegiant’s operations focus on domestic US service (and do not provide international service to Singapore)."
    )
    await evaluator.verify(
        claim=claim_rule,
        node=ref_leaf,
        sources=urls.allegiant_ops_urls,
        additional_instruction=(
            "Only accept content from allegiantair.com. "
            "If the page is not Allegiant’s official site or does not substantively support the domestic-only nature of their network, mark as not supported."
        ),
    )


async def verify_allegiant_baggage(
    evaluator: Evaluator,
    parent_node,
    urls: RequirementURLsExtraction,
    assessments: RequirementAssessmentsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Allegiant_Baggage_Restrictions_Personal_And_Checked",
        desc="Verifies compliance with Allegiant baggage restrictions for both the personal item and checked bag, with an official Allegiant citation.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Personal item compliance (verify against policy URL)
    personal_leaf = evaluator.add_leaf(
        id="Personal_Item_Compliance_Assessment",
        desc="Answer correctly compares the given personal-item dimensions against Allegiant’s official personal-item limits and addresses any required fit/placement condition (e.g., under-seat requirement) as stated in the cited policy.",
        parent=node,
        critical=True,
    )
    personal_assess = assessments.baggage_personal_item_assessment or "uncertain"
    claim_personal = (
        f"A personal item measuring {PERSONAL_ITEM_DIMS} is '{personal_assess}' under Allegiant’s personal item limits and under-seat fit requirement, "
        f"as stated in the official baggage policy."
    )
    await evaluator.verify(
        claim=claim_personal,
        node=personal_leaf,
        sources=urls.baggage_policy_urls,
        additional_instruction=(
            "Use the official Allegiant baggage policy page(s) to determine whether 7 x 13 x 17 inches qualifies as a personal item. "
            "Consider both dimensional limits and under-seat fit language; if any dimension exceeds the stated limit, treat as non-compliant."
        ),
    )

    # Leaf: Checked bag compliance (verify against policy URL)
    checked_leaf = evaluator.add_leaf(
        id="Checked_Bag_Compliance_Assessment",
        desc="Answer correctly compares the given checked-bag weight and total dimensions against Allegiant’s official checked-bag limits (weight and size) as stated in the cited policy.",
        parent=node,
        critical=True,
    )
    checked_assess = assessments.baggage_checked_bag_assessment or "uncertain"
    claim_checked = (
        f"A checked bag weighing {CHECKED_BAG_WEIGHT_LBS} lbs and totaling {CHECKED_BAG_LINEAR_INCHES} linear inches is '{checked_assess}' "
        f"under Allegiant’s checked baggage weight and size limits per the official baggage policy."
    )
    await evaluator.verify(
        claim=claim_checked,
        node=checked_leaf,
        sources=urls.baggage_policy_urls,
        additional_instruction=(
            "Use Allegiant’s policy for checked bags (weight limit and maximum linear inches). "
            "If 48 lbs exceeds the standard weight limit (e.g., 40 lbs) and the policy treats overweight as non-compliant rather than merely subject to a fee, mark non-compliant; "
            "otherwise, reason according to the policy text."
        ),
    )

    # Leaf: Official reference URL support (allegiantair.com baggage policy)
    ref_leaf = evaluator.add_leaf(
        id="Allegiant_Baggage_Policy_Official_Reference_URL",
        desc="Provides an official Allegiant baggage-policy URL that supports the personal-item and checked-bag limits used in the assessments.",
        parent=node,
        critical=True,
    )
    claim_bag_policy = (
        "This official Allegiant baggage policy page clearly states the personal item dimensional limits and the checked bag weight/size limits used for compliance checks."
    )
    await evaluator.verify(
        claim=claim_bag_policy,
        node=ref_leaf,
        sources=urls.baggage_policy_urls,
        additional_instruction=(
            "Only accept content from allegiantair.com that explicitly states personal item limits and checked bag limits. "
            "If the page does not provide these limits or is not Allegiant’s official site, mark as not supported."
        ),
    )


async def verify_etravel_timing(
    evaluator: Evaluator,
    parent_node,
    urls: RequirementURLsExtraction,
    assessments: RequirementAssessmentsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Philippines_eTravel_Timing",
        desc="Verifies eTravel registration timing compliance for arrival in the Philippines, with an official Philippine government citation.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Timing assessment (verify against official URL)
    assess_leaf = evaluator.add_leaf(
        id="eTravel_Timing_Assessment",
        desc="Answer correctly determines whether completing eTravel registration 48 hours before arriving in Manila satisfies the official timing requirement (as stated in the cited official source).",
        parent=node,
        critical=True,
    )
    etravel_assess = assessments.etravel_timing_assessment or "uncertain"
    claim_assess = (
        f"Completing eTravel registration 48 hours before arrival in Manila is '{etravel_assess}' according to the official eTravel timing requirement."
    )
    await evaluator.verify(
        claim=claim_assess,
        node=assess_leaf,
        sources=urls.etravel_urls,
        additional_instruction=(
            "Verify the timing rule from the official eTravel site (e.g., within 3 days/72 hours before arrival). "
            "Treat non-official pages as unsupported."
        ),
    )

    # Leaf: Official reference URL (government/eTravel site)
    ref_leaf = evaluator.add_leaf(
        id="eTravel_Official_Reference_URL",
        desc="Provides an official Philippine government/eTravel URL supporting the timing requirement used in the assessment.",
        parent=node,
        critical=True,
    )
    claim_rule = (
        "This official Philippine government/eTravel page clearly states the timing requirement (e.g., register within a certain time window before arrival)."
    )
    await evaluator.verify(
        claim=claim_rule,
        node=ref_leaf,
        sources=urls.etravel_urls,
        additional_instruction=(
            "Only accept official government or the official eTravel portal pages. "
            "If the page is not official or the timing window is not stated, mark as not supported."
        ),
    )


async def verify_cruise_operates(
    evaluator: Evaluator,
    parent_node,
    urls: RequirementURLsExtraction,
    assessments: RequirementAssessmentsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Royal_Caribbean_Cruise_From_Singapore",
        desc="Verifies that the described Royal Caribbean Ovation of the Seas cruise operates from Singapore in 2026, with an official Royal Caribbean citation.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Assessment of operation (verify using official RC URLs)
    assess_leaf = evaluator.add_leaf(
        id="Ovation_Departs_Singapore_2026_Assessment",
        desc="Answer verifies (based on cited official Royal Caribbean information) that Ovation of the Seas offers an itinerary departing from Singapore in 2026 consistent with the plan’s description (or explicitly states if it cannot be verified from official sources).",
        parent=node,
        critical=True,
    )
    cruise_assess = assessments.cruise_operates_assessment or "uncertain"
    claim_assess = (
        f"Ovation of the Seas has an itinerary departing from Singapore in 2026; the answer’s conclusion '{cruise_assess}' is correct based on the official Royal Caribbean sources."
    )
    await evaluator.verify(
        claim=claim_assess,
        node=assess_leaf,
        sources=urls.cruise_urls,
        additional_instruction=(
            "Use only royalcaribbean.com official pages (ship or itinerary pages). "
            "If the page indicates Singapore departures for Ovation of the Seas in 2026, mark as supported; otherwise, not supported."
        ),
    )

    # Leaf: Official Royal Caribbean URL support
    ref_leaf = evaluator.add_leaf(
        id="Royal_Caribbean_Official_Itinerary_Reference_URL",
        desc="Provides at least one official Royal Caribbean URL (ship page and/or itinerary page) supporting the Singapore departure claim used in the assessment.",
        parent=node,
        critical=True,
    )
    claim_rc = (
        "This official Royal Caribbean webpage confirms Ovation of the Seas itineraries that depart from Singapore in 2026."
    )
    await evaluator.verify(
        claim=claim_rc,
        node=ref_leaf,
        sources=urls.cruise_urls,
        additional_instruction=(
            "Only accept content from royalcaribbean.com that clearly shows Singapore departures for Ovation of the Seas (preferably 2026). "
            "If not clearly stated, mark as not supported."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate the agent's answer for the 2026 travel plan validation task.
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
        default_model=model,
    )

    # Top-level critical node mirroring rubric's root
    top_node = evaluator.add_parallel(
        id="Root_Travel_Plan_Validation",
        desc="Checks whether the plan satisfies all mandatory verifications requested in the question and constraints, and provides official-source citations for each verification.",
        parent=root,
        critical=True,
    )

    # Parallel extraction of URLs, assessments, and any itinerary dates mentioned
    urls_task = evaluator.extract(
        prompt=prompt_extract_requirement_urls(),
        template_class=RequirementURLsExtraction,
        extraction_name="requirement_urls",
    )
    assessments_task = evaluator.extract(
        prompt=prompt_extract_assessments(),
        template_class=RequirementAssessmentsExtraction,
        extraction_name="requirement_assessments",
    )
    dates_task = evaluator.extract(
        prompt=prompt_extract_itinerary_dates(),
        template_class=ItineraryDatesExtraction,
        extraction_name="itinerary_dates",
    )

    urls, assessments, itinerary_dates = await asyncio.gather(urls_task, assessments_task, dates_task)

    # Add custom info for task context constants (helps interpretation)
    evaluator.add_custom_info(
        info={
            "passport_expiry": PASSPORT_EXPIRY,
            "trip_departure": TRIP_DEPARTURE,
            "trip_return": TRIP_RETURN,
            "personal_item_dims": PERSONAL_ITEM_DIMS,
            "checked_bag_weight_lbs": CHECKED_BAG_WEIGHT_LBS,
            "checked_bag_linear_inches": CHECKED_BAG_LINEAR_INCHES,
            "cruise_ship": CRUISE_SHIP,
            "cruise_nights": CRUISE_NIGHTS,
            "cruise_departure_port": CRUISE_DEPARTURE_PORT,
            "cruise_ports": CRUISE_PORTS,
        },
        info_type="task_context",
        info_name="travel_plan_constants",
    )

    # Build and verify each subtree under the critical top node
    await verify_passport_six_month_rule(evaluator, top_node, urls, assessments, itinerary_dates)
    await verify_visa_requirement(evaluator, top_node, urls, assessments)
    await verify_allegiant_operations(evaluator, top_node, urls, assessments)
    await verify_allegiant_baggage(evaluator, top_node, urls, assessments)
    await verify_etravel_timing(evaluator, top_node, urls, assessments)
    await verify_cruise_operates(evaluator, top_node, urls, assessments)

    # Return unified summary
    return evaluator.get_summary()