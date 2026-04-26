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
TASK_ID = "mlb_go_ahead_entry_ballpark"
TASK_DESCRIPTION = (
    "Identify an MLB ballpark in the United States that has implemented the MLB Go-Ahead Entry facial authentication "
    "system as of 2024-2025. For this ballpark, provide: 1) The name of the ballpark, 2) The app required for enrollment, "
    "3) The biometric enrollment method fans must complete, 4) Whether the program is mandatory or voluntary, 5) The core "
    "technology used for entry verification, 6) Whether there are designated gates for this entry method, and 7) A reference "
    "URL confirming the Go-Ahead Entry implementation at this ballpark."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BallparkEntryInfo(BaseModel):
    """
    Structured info extracted from the agent answer about a single MLB ballpark's
    MLB Go-Ahead Entry implementation.
    """
    ballpark_name: Optional[str] = None
    app_required: Optional[str] = None  # e.g., "MLB Ballpark app"
    biometric_enrollment_method: Optional[str] = None  # e.g., "selfie", "face scan"
    program_voluntary_or_mandatory: Optional[str] = None  # e.g., "voluntary", "opt-in", "mandatory"
    entry_technology: Optional[str] = None  # e.g., "facial authentication", "facial recognition"
    designated_gates: Optional[str] = None  # e.g., "Gate 3 only", "Dedicated Go-Ahead Entry lanes"
    reference_urls: List[str] = Field(default_factory=list)  # URLs cited in the answer that support the claims


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_go_ahead_entry_info() -> str:
    """
    Build an extraction prompt for the LLM to pull the relevant fields from the agent's answer.
    """
    return """
    You must extract information about a single MLB ballpark (in the United States) that has implemented the MLB "Go-Ahead Entry" facial authentication system as of 2024–2025, from the provided answer text.

    Extract the following fields (return null when missing; do NOT invent):
    - ballpark_name: The name of the MLB ballpark mentioned in the answer.
    - app_required: The app required to enroll for Go-Ahead Entry (typically "MLB Ballpark app" or similar phrasing).
    - biometric_enrollment_method: What a fan must do to enroll biometrically (e.g., "take a selfie", "face scan").
    - program_voluntary_or_mandatory: Whether Go-Ahead Entry is voluntary (opt-in) or mandatory (e.g., "voluntary", "opt-in", or "mandatory").
    - entry_technology: The core technology used for entry verification (e.g., "facial authentication", "facial recognition", "hands-free facial matching").
    - designated_gates: Whether specific gates/lanes are designated for this entry method. Provide the phrasing given (e.g., mention "designated gates", "dedicated entrance", or specific gate numbers if present).
    - reference_urls: An array of all URLs in the answer that directly pertain to Go-Ahead Entry at the identified ballpark. Include official MLB/team domains or credible news sources if provided; include any relevant URLs present. Do not invent URLs.

    Rules:
    - Extract only what is explicitly present in the answer text.
    - If multiple ballparks are mentioned, extract information for the first one appearing in the answer.
    - For URLs: return only valid URLs that appear in the answer. If none are present, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_ballpark_name(info: BallparkEntryInfo) -> str:
    return info.ballpark_name.strip() if info.ballpark_name else "the ballpark mentioned in the answer"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_go_ahead_entry(evaluator: Evaluator, parent_node, info: BallparkEntryInfo) -> None:
    """
    Build the verification tree and perform checks for MLB Go-Ahead Entry at a specified ballpark.
    All children under the parent node are critical, following the rubric.
    """
    # Add parent node (Critical + Parallel aggregation as per rubric)
    main_node = evaluator.add_parallel(
        id="MLB_Ballpark_Go_Ahead_Entry",
        desc="Identifies an MLB ballpark in the United States that has implemented MLB Go-Ahead Entry and provides accurate information about its requirements and features",
        parent=parent_node,
        critical=True
    )

    # 1) Ballpark_Name (Critical)
    # Implemented as an existence check: ensure a ballpark name is provided in the answer.
    evaluator.add_custom_node(
        result=bool(info.ballpark_name and info.ballpark_name.strip()),
        id="Ballpark_Name",
        desc="Provides the name of an MLB ballpark that has implemented MLB Go-Ahead Entry as of 2024-2025",
        parent=main_node,
        critical=True
    )

    # Prepare common values
    ballpark = _safe_ballpark_name(info)
    urls = info.reference_urls  # may be empty; verification function will handle failures

    # 2) MLB_Ballpark_App_Requirement (Critical)
    app_node = evaluator.add_leaf(
        id="MLB_Ballpark_App_Requirement",
        desc="States that enrollment requires the MLB Ballpark app",
        parent=main_node,
        critical=True
    )
    app_claim = f"Enrollment in MLB Go-Ahead Entry for {ballpark} requires using the MLB Ballpark app."
    app_instruction = (
        "Verify that the page explicitly indicates the MLB Ballpark app is required to enroll in Go-Ahead Entry. "
        "Accept equivalent phrasings like 'MLB Ballpark' or 'Ballpark app'. The claim should relate to the Go-Ahead Entry program."
    )

    # 3) Selfie_Enrollment (Critical)
    selfie_node = evaluator.add_leaf(
        id="Selfie_Enrollment",
        desc="States that fans must take a selfie to enroll in the system",
        parent=main_node,
        critical=True
    )
    selfie_claim = (
        f"To enroll in MLB Go-Ahead Entry for {ballpark}, fans must capture a selfie (face scan) during enrollment in the MLB Ballpark app."
    )
    selfie_instruction = (
        "Check that the page explains the enrollment requires taking a selfie, face scan, or similar facial capture within the MLB Ballpark app. "
        "Accept synonymous phrases such as 'take a selfie', 'scan your face', 'add a face'."
    )

    # 4) Opt_In_Program (Critical)
    optin_node = evaluator.add_leaf(
        id="Opt_In_Program",
        desc="States that Go-Ahead Entry is a voluntary opt-in program",
        parent=main_node,
        critical=True
    )
    optin_claim = "MLB Go-Ahead Entry is voluntary (opt-in) and not mandatory for fans."
    optin_instruction = (
        "Confirm the page indicates Go-Ahead Entry is optional/voluntary/opt-in. "
        "If the page suggests or states participation is required or mandatory, mark as not supported."
    )

    # 5) Facial_Authentication_Function (Critical)
    facial_node = evaluator.add_leaf(
        id="Facial_Authentication_Function",
        desc="States that the system uses facial authentication technology for hands-free entry",
        parent=main_node,
        critical=True
    )
    facial_claim = (
        "The Go-Ahead Entry system uses facial authentication or facial recognition to verify identity for hands-free entry."
    )
    facial_instruction = (
        "Look for language explicitly describing facial authentication/recognition, face matching, or hands-free facial entry. "
        "Accept equivalent phrasings that clearly indicate face-based verification at the gate."
    )

    # 6) Designated_Gates (Critical)
    gates_node = evaluator.add_leaf(
        id="Designated_Gates",
        desc="States that specific gates or entrances are designated for Go-Ahead Entry",
        parent=main_node,
        critical=True
    )
    gates_claim = (
        f"{ballpark} has specific designated gates, lanes, or entrances for MLB Go-Ahead Entry."
    )
    gates_instruction = (
        "Verify that the page mentions designated/assigned gates, lanes, or entrances for Go-Ahead Entry. "
        "Accept statements like 'Go-Ahead Entry-only lanes', 'dedicated entrance', or specific gate numbers reserved for this program."
    )

    # 7) Ballpark_Reference_URL (Critical)
    # This leaf checks that at least one provided URL clearly confirms Go-Ahead Entry implementation at the identified ballpark.
    ref_node = evaluator.add_leaf(
        id="Ballpark_Reference_URL",
        desc="Provides a valid reference URL from official MLB sources, team websites, or credible news sources about the Go-Ahead Entry implementation at the identified ballpark",
        parent=main_node,
        critical=True
    )
    ref_claim = (
        f"This webpage confirms that MLB Go-Ahead Entry is implemented or available at {ballpark}."
    )
    ref_instruction = (
        "Pass only if the page clearly references the specific ballpark/team and explicitly mentions MLB Go-Ahead Entry at that venue. "
        "Prefer official MLB/team domains (e.g., mlb.com subdomains) or credible news outlets. "
        "If the URL is unrelated, generic, non-credible, or does not mention the specified ballpark's Go-Ahead Entry program, mark as not supported."
    )

    # Batch verify the six URL-supported claims in parallel
    claims_and_sources = [
        (app_claim, urls, app_node, app_instruction),
        (selfie_claim, urls, selfie_node, selfie_instruction),
        (optin_claim, urls, optin_node, optin_instruction),
        (facial_claim, urls, facial_node, facial_instruction),
        (gates_claim, urls, gates_node, gates_instruction),
        (ref_claim, urls, ref_node, ref_instruction),
    ]
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the MLB Go-Ahead Entry (ballpark) task using a rubric-based verification tree.
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

    # Extract structured info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_go_ahead_entry_info(),
        template_class=BallparkEntryInfo,
        extraction_name="go_ahead_entry_info"
    )

    # Add a small custom info record for convenience
    evaluator.add_custom_info(
        info={
            "ballpark_name": extracted_info.ballpark_name,
            "app_required": extracted_info.app_required,
            "biometric_enrollment_method": extracted_info.biometric_enrollment_method,
            "program_voluntary_or_mandatory": extracted_info.program_voluntary_or_mandatory,
            "entry_technology": extracted_info.entry_technology,
            "designated_gates": extracted_info.designated_gates,
            "reference_urls": extracted_info.reference_urls,
        },
        info_type="extracted_summary",
        info_name="extracted_go_ahead_entry_summary"
    )

    # Build and run verification
    await verify_go_ahead_entry(evaluator, root, extracted_info)

    # Return standardized summary
    return evaluator.get_summary()