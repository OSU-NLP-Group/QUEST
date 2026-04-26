import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tsa_touchless_id_guide"
TASK_DESCRIPTION = (
    "A U.S. citizen with an iPhone 13 running iOS 26.2 wants to use TSA PreCheck Touchless ID for domestic air travel. "
    "They do not currently have TSA PreCheck membership. Provide complete information on: "
    "(1) the requirements and process to enroll in TSA PreCheck, "
    "(2) the additional requirements to activate and use TSA PreCheck Touchless ID, "
    "(3) which U.S. airlines participate in the Touchless ID program, "
    "(4) what physical backup identification they must carry, and "
    "(5) how to verify Touchless ID availability at their departure airport."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class EnrollmentSection(BaseModel):
    online_application: Optional[bool] = None
    inperson_appointment: Optional[bool] = None
    required_documentation: Optional[bool] = None
    enrollment_fee_text: Optional[str] = None
    ktn_issuance: Optional[bool] = None
    enrollment_urls: List[str] = Field(default_factory=list)


class TouchlessActivationSection(BaseModel):
    precheck_membership_required: Optional[bool] = None
    ktn_required: Optional[bool] = None
    airline_profile_requirement: Optional[bool] = None
    ktn_in_profile: Optional[bool] = None
    passport_upload: Optional[bool] = None
    optin_process: Optional[bool] = None
    boarding_pass_indicator: Optional[bool] = None
    device_compatibility_confirmed: Optional[bool] = None
    device_compatibility_detail: Optional[str] = None
    data_privacy_policy: Optional[bool] = None
    touchless_id_requirements_urls: List[str] = Field(default_factory=list)


class AirlinesSection(BaseModel):
    airlines_listed: List[str] = Field(default_factory=list)
    participating_airlines_urls: List[str] = Field(default_factory=list)


class BackupIDSection(BaseModel):
    backup_required: Optional[bool] = None
    realid_compliant: Optional[bool] = None
    realid_enforcement_date_text: Optional[str] = None
    backup_id_urls: List[str] = Field(default_factory=list)


class AirportAvailabilitySection(BaseModel):
    airport_expansion_text: Optional[str] = None
    airline_specific_availability: Optional[bool] = None
    terminal_checkpoint_variation: Optional[bool] = None
    boarding_pass_confirmation: Optional[bool] = None
    availability_verification_urls: List[str] = Field(default_factory=list)


class TouchlessGuideExtraction(BaseModel):
    enrollment: EnrollmentSection = EnrollmentSection()
    activation: TouchlessActivationSection = TouchlessActivationSection()
    airlines: AirlinesSection = AirlinesSection()
    backup_id: BackupIDSection = BackupIDSection()
    availability: AirportAvailabilitySection = AirportAvailabilitySection()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_touchless_guide() -> str:
    return """
Extract the following structured information from the answer text. Only mark a boolean as true if the answer explicitly states it. For each URLs field, extract all explicit URLs in the answer that best support that specific topic. If a requested item is not present, set it to false or null, and return an empty list for missing URLs.

Return a JSON object with this structure:

{
  "enrollment": {
    "online_application": boolean,  // Answer states completing an online pre-enrollment application is part of TSA PreCheck enrollment
    "inperson_appointment": boolean,  // Answer states an in-person appointment at an enrollment center is required
    "required_documentation": boolean,  // Answer mentions bringing identity and citizenship documentation
    "enrollment_fee_text": string|null,  // The fee info as written (e.g., "$78 for 5 years", "$76.75–$85", etc.)
    "ktn_issuance": boolean,  // Answer mentions receiving a Known Traveler Number (KTN) upon approval
    "enrollment_urls": string[]  // All URLs in the answer about TSA PreCheck enrollment (prefer tsa.gov pages)
  },

  "activation": {
    "precheck_membership_required": boolean,  // Answer states that TSA PreCheck membership is a prerequisite for Touchless ID
    "ktn_required": boolean,  // Answer mentions a Known Traveler Number is required
    "airline_profile_requirement": boolean,  // Answer states a frequent flyer profile with a participating airline is needed
    "ktn_in_profile": boolean,  // Answer mentions adding KTN to airline profile
    "passport_upload": boolean,  // Answer mentions uploading a valid passport to airline profile
    "optin_process": boolean,  // Answer mentions opting in to Touchless ID (in profile or at check-in)
    "boarding_pass_indicator": boolean,  // Answer mentions Touchless ID indicator must appear on mobile boarding pass
    "device_compatibility_confirmed": boolean|null,  // Answer confirms the specified user's device meets requirements
    "device_compatibility_detail": string|null,  // The device requirement details as stated in the answer
    "data_privacy_policy": boolean|null,  // Answer mentions biometric images/data deleted within ~24 hours of departure
    "touchless_id_requirements_urls": string[]  // All URLs about Touchless ID requirements (airlines or TSA)
  },

  "airlines": {
    "airlines_listed": string[],  // Names of U.S. airlines that the answer lists as participating in Touchless ID
    "participating_airlines_urls": string[]  // URLs in the answer that list participating airlines
  },

  "backup_id": {
    "backup_required": boolean,  // Answer states a physical backup ID must still be carried
    "realid_compliant": boolean,  // Answer states ID must be REAL ID-compliant (star) or Enhanced Driver's License
    "realid_enforcement_date_text": string|null,  // Any enforcement/effective date mentioned (e.g., "May 7, 2025")
    "backup_id_urls": string[]  // URLs in the answer about REAL ID or acceptable identification requirements
  },

  "availability": {
    "airport_expansion_text": string|null,  // Any mention like "65 airports by Spring 2026" (return the text as stated)
    "airline_specific_availability": boolean,  // Answer explains availability varies by airline at airports
    "terminal_checkpoint_variation": boolean|null,  // Answer mentions variation by terminal/checkpoint
    "boarding_pass_confirmation": boolean,  // Answer states to check for Touchless ID indicator on mobile boarding pass
    "availability_verification_urls": string[]  // URLs in the answer about where/how to verify Touchless ID availability
  }
}
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _any_url_has_domain(urls: List[str], domains: List[str]) -> bool:
    if not urls:
        return False
    for u in urls:
        try:
            netloc = urlparse(u).netloc.lower()
            for d in domains:
                d = d.lower().lstrip(".")
                if netloc.endswith(d):
                    return True
        except Exception:
            continue
    return False


def _has_any_url(urls: List[str]) -> bool:
    return isinstance(urls, list) and len(urls) > 0


# --------------------------------------------------------------------------- #
# Section verifications                                                       #
# --------------------------------------------------------------------------- #
async def verify_enrollment(evaluator: Evaluator, parent, ex: TouchlessGuideExtraction) -> None:
    node = evaluator.add_parallel(
        id="tsa_precheck_enrollment_process",
        desc="TSA PreCheck enrollment requirements and process correctly specified",
        parent=parent,
        critical=False
    )

    # Critical: tsa.gov enrollment URL present
    has_tsa_gov = _any_url_has_domain(ex.enrollment.enrollment_urls, ["tsa.gov"])
    evaluator.add_custom_node(
        result=has_tsa_gov,
        id="tsa_precheck_enrollment_url",
        desc="Provides reference URL from tsa.gov domain about PreCheck enrollment",
        parent=node,
        critical=True
    )

    # Critical mentions (validated against the answer text)
    leaf_online = evaluator.add_leaf(
        id="online_application",
        desc="Mentions completing online pre-enrollment application",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that completing an online pre-enrollment application is part of the TSA PreCheck enrollment process.",
        node=leaf_online,
        additional_instruction="Judge only by the answer's content. Do not require external evidence."
    )

    leaf_inperson = evaluator.add_leaf(
        id="inperson_appointment",
        desc="Specifies in-person appointment at enrollment center required",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that an in-person appointment at a TSA PreCheck enrollment center is required.",
        node=leaf_inperson,
        additional_instruction="Judge only by the answer's content. Do not require external evidence."
    )

    leaf_docs = evaluator.add_leaf(
        id="required_documentation",
        desc="Mentions bringing identity and citizenship documentation",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly mentions bringing identity and citizenship documentation to the enrollment appointment.",
        node=leaf_docs,
        additional_instruction="Judge only by the answer's content."
    )

    # Non-critical: fee info provided (we only check presence in the answer)
    leaf_fee = evaluator.add_leaf(
        id="enrollment_fee",
        desc="Provides enrollment fee information ($76.75-$85 for 5 years)",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The answer includes some enrollment fee information for TSA PreCheck (e.g., a dollar amount and/or 5-year membership).",
        node=leaf_fee,
        additional_instruction="Pass if the answer includes any concrete fee information; do not judge exact numbers."
    )

    leaf_ktn = evaluator.add_leaf(
        id="ktn_issuance",
        desc="Mentions receiving Known Traveler Number (KTN) upon approval",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that upon approval the traveler receives a Known Traveler Number (KTN).",
        node=leaf_ktn,
        additional_instruction="Judge only by the answer's content."
    )


async def verify_activation(evaluator: Evaluator, parent, ex: TouchlessGuideExtraction) -> None:
    node = evaluator.add_parallel(
        id="touchless_id_activation",
        desc="Additional requirements for Touchless ID activation correctly specified",
        parent=parent,
        critical=False
    )

    # Critical: at least one requirements URL present (airline or TSA)
    has_req_url = _has_any_url(ex.activation.touchless_id_requirements_urls)
    evaluator.add_custom_node(
        result=has_req_url,
        id="touchless_id_requirements_url",
        desc="Provides reference URL about Touchless ID requirements",
        parent=node,
        critical=True
    )

    # Critical mentions (answer content)
    claims = [
        ("precheck_membership_required",
         "Specifies that TSA PreCheck membership is prerequisite for Touchless ID",
         "The answer explicitly states that TSA PreCheck membership is a prerequisite for using TSA PreCheck Touchless ID."),
        ("ktn_required",
         "Mentions Known Traveler Number (KTN) is required",
         "The answer explicitly states that a Known Traveler Number (KTN) is required to use Touchless ID."),
        ("airline_profile_requirement",
         "Specifies need for active frequent flyer profile with participating airline",
         "The answer explicitly states that the traveler needs an active frequent flyer profile with a participating airline to use Touchless ID."),
        ("ktn_in_profile",
         "Mentions adding KTN to airline profile",
         "The answer explicitly mentions adding the KTN to the airline profile."),
        ("passport_upload",
         "Specifies uploading valid passport information to airline profile",
         "The answer explicitly states that the traveler must upload a valid passport to the airline profile."),
        ("optin_process",
         "Mentions opt-in selection for Touchless ID in airline profile or at check-in",
         "The answer explicitly mentions opting in to Touchless ID either in the airline profile or during check-in."),
        ("boarding_pass_indicator",
         "Specifies that Touchless ID indicator must appear on mobile boarding pass to use service",
         "The answer explicitly states that a Touchless ID indicator must appear on the mobile boarding pass in order to use the service.")
    ]

    tasks = []
    for leaf_id, leaf_desc, claim_text in claims:
        lf = evaluator.add_leaf(
            id=leaf_id,
            desc=leaf_desc,
            parent=node,
            critical=True
        )
        tasks.append(evaluator.verify(
            claim=claim_text,
            node=lf,
            additional_instruction="Judge only by the answer's content."
        ))
    await asyncio.gather(*tasks)

    # Non-critical details
    dev_leaf = evaluator.add_leaf(
        id="device_compatibility",
        desc="Confirms iPhone 13 with iOS 26.2 meets device requirements (iPhone 11+ with iOS 26.1+ required)",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The answer confirms that an iPhone 13 running iOS 26.2 meets the device requirements to use TSA PreCheck Touchless ID.",
        node=dev_leaf,
        additional_instruction="Judge only by the answer's content."
    )

    privacy_leaf = evaluator.add_leaf(
        id="data_privacy_policy",
        desc="Mentions that biometric images and data are deleted within 24 hours of scheduled departure",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The answer mentions that biometric images/data used for Touchless ID are deleted within about 24 hours of the scheduled departure.",
        node=privacy_leaf,
        additional_instruction="Judge only by the answer's content."
    )


async def verify_participating_airlines(evaluator: Evaluator, parent, ex: TouchlessGuideExtraction) -> None:
    node = evaluator.add_parallel(
        id="participating_airlines",
        desc="List of participating airlines correctly provided",
        parent=parent,
        critical=False
    )

    # Critical: at least one URL listing participating airlines
    has_airlines_url = _has_any_url(ex.airlines.participating_airlines_urls)
    evaluator.add_custom_node(
        result=has_airlines_url,
        id="participating_airlines_url",
        desc="Provides reference URL listing participating airlines",
        parent=node,
        critical=True
    )

    # Verify the answer lists each required airline (answer-content check)
    airline_checks = [
        ("alaska_airlines", "Lists Alaska Airlines as participating", "Alaska Airlines"),
        ("american_airlines", "Lists American Airlines as participating", "American Airlines"),
        ("delta_airlines", "Lists Delta Air Lines as participating", "Delta Air Lines"),
        ("southwest_airlines", "Lists Southwest Airlines as participating", "Southwest Airlines"),
        ("united_airlines", "Lists United Airlines as participating", "United Airlines"),
    ]

    tasks = []
    for leaf_id, desc, name in airline_checks:
        lf = evaluator.add_leaf(
            id=leaf_id,
            desc=desc,
            parent=node,
            critical=True
        )
        tasks.append(evaluator.verify(
            claim=f"The answer lists {name} as a participating airline in the TSA PreCheck Touchless ID program.",
            node=lf,
            additional_instruction="Judge only by the answer's content."
        ))
    await asyncio.gather(*tasks)


async def verify_backup_id(evaluator: Evaluator, parent, ex: TouchlessGuideExtraction) -> None:
    node = evaluator.add_parallel(
        id="physical_backup_id",
        desc="Physical backup ID requirements correctly specified",
        parent=parent,
        critical=False
    )

    # Critical: URL about REAL ID or acceptable identification (prefer tsa.gov or dhs.gov)
    has_realid_url = _any_url_has_domain(ex.backup_id.backup_id_urls, ["tsa.gov", "dhs.gov"])
    evaluator.add_custom_node(
        result=has_realid_url,
        id="backup_id_url",
        desc="Provides reference URL about REAL ID or acceptable identification requirements",
        parent=node,
        critical=True
    )

    # Critical mentions (answer-content check)
    backup_leaf = evaluator.add_leaf(
        id="backup_required",
        desc="Specifies that physical backup ID must be carried despite Touchless ID enrollment",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that a physical backup ID must still be carried even when enrolled in Touchless ID.",
        node=backup_leaf,
        additional_instruction="Judge only by the answer's content."
    )

    realid_leaf = evaluator.add_leaf(
        id="realid_compliant",
        desc="Specifies ID must be REAL ID-compliant with star marking or Enhanced Driver's License",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the physical ID must be REAL ID-compliant (star marking) or an Enhanced Driver's License.",
        node=realid_leaf,
        additional_instruction="Judge only by the answer's content."
    )

    # Non-critical: enforcement date mention
    date_leaf = evaluator.add_leaf(
        id="realid_enforcement_date",
        desc="Mentions REAL ID requirement effective May 7, 2025",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The answer mentions that the REAL ID requirement/enforcement date is May 7, 2025.",
        node=date_leaf,
        additional_instruction="Judge only by the answer's content."
    )


async def verify_airport_availability(evaluator: Evaluator, parent, ex: TouchlessGuideExtraction) -> None:
    node = evaluator.add_parallel(
        id="airport_availability",
        desc="Airport availability verification process correctly explained",
        parent=parent,
        critical=False
    )

    # Critical: URL about airport availability/participating locations
    has_avail_url = _has_any_url(ex.availability.availability_verification_urls)
    evaluator.add_custom_node(
        result=has_avail_url,
        id="availability_verification_url",
        desc="Provides reference URL about airport availability or participating locations",
        parent=node,
        critical=True
    )

    # Non-critical: expansion mention
    expansion_leaf = evaluator.add_leaf(
        id="airport_expansion",
        desc="Mentions 65 airports by Spring 2026 expansion",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The answer mentions an expansion to around 65 airports by Spring 2026 for Touchless ID.",
        node=expansion_leaf,
        additional_instruction="Judge only by the answer's content."
    )

    # Critical: airline-specific availability noted
    airline_var_leaf = evaluator.add_leaf(
        id="airline_specific_availability",
        desc="Explains availability varies by airline at specific airports",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explains that Touchless ID availability varies by airline at specific airports.",
        node=airline_var_leaf,
        additional_instruction="Judge only by the answer's content."
    )

    # Non-critical: terminal/checkpoint variation noted
    terminal_leaf = evaluator.add_leaf(
        id="terminal_checkpoint_variation",
        desc="Mentions that availability may vary by terminal and checkpoint within an airport",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The answer mentions that Touchless ID availability may vary by terminal and checkpoint within an airport.",
        node=terminal_leaf,
        additional_instruction="Judge only by the answer's content."
    )

    # Critical: boarding pass confirmation
    bp_leaf = evaluator.add_leaf(
        id="boarding_pass_confirmation",
        desc="Mentions checking for Touchless ID indicator on mobile boarding pass as confirmation",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer instructs travelers to check for a Touchless ID indicator on the mobile boarding pass as confirmation/eligibility.",
        node=bp_leaf,
        additional_instruction="Judge only by the answer's content."
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
    Evaluate an answer for TSA PreCheck Touchless ID guidance using a rubric-based verification tree.
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
        default_model=model
    )

    # NOTE: Root is set as non-critical in the framework to avoid illegal critical-child constraints.
    # We enforce criticality at leaf/section level per rubric.

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_touchless_guide(),
        template_class=TouchlessGuideExtraction,
        extraction_name="touchless_id_guide_extraction"
    )

    # Build sections according to rubric
    await verify_enrollment(evaluator, root, extraction)
    await verify_activation(evaluator, root, extraction)
    await verify_participating_airlines(evaluator, root, extraction)
    await verify_backup_id(evaluator, root, extraction)
    await verify_airport_availability(evaluator, root, extraction)

    # Return summary
    return evaluator.get_summary()