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
TASK_ID = "iphone_sos_verizon_support"
TASK_DESCRIPTION = (
    "You are experiencing an issue where your iPhone is stuck in SOS mode while using Verizon as your carrier. "
    "You need to follow Apple's official troubleshooting guidance and have Verizon's contact information ready. "
    "Identify the following information: (1) What are the first three troubleshooting steps recommended by Apple Support "
    "for resolving iPhone SOS mode? Provide the steps in the correct sequential order as listed on Apple's official support "
    "documentation, along with the reference URL. (2) What is the official technical support phone number for Verizon "
    "Mobile/5G Home/LTE Home Internet customers experiencing network issues, and what are the operating hours for this "
    "technical support line? Provide the reference URL from Verizon's official website. For each piece of information, "
    "include the specific details and the official support page URL where this information can be verified."
)

EXPECTED_APPLE_STEPS = [
    "Toggle Airplane Mode on for at least 15 seconds then turn it off",
    "Restart the iPhone",
    "Contact the carrier to verify account status and check for network issues",
]
EXPECTED_VERIZON_NUMBER = "800-922-0204"
EXPECTED_VERIZON_HOURS = "1 PM to 5 AM GMT, Sunday through Saturday"
EXPECTED_APPLE_URL_HINT = "https://support.apple.com/en-us/120000"
EXPECTED_VERIZON_URL = "https://www.verizon.com/support/contact-us/"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StepInfo(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None


class AppleSOSExtraction(BaseModel):
    step1: Optional[StepInfo] = None
    step2: Optional[StepInfo] = None
    step3: Optional[StepInfo] = None
    apple_urls: List[str] = Field(default_factory=list)


class VerizonSupportExtraction(BaseModel):
    phone_number: Optional[str] = None
    hours: Optional[str] = None
    url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_apple_steps() -> str:
    return (
        "From the provided answer, extract the first three troubleshooting steps that the answer claims Apple Support "
        "recommends for resolving an iPhone stuck in SOS mode (or showing 'SOS only'). Preserve the sequential order "
        "exactly as presented in the answer. For each of the first three steps, return:\n"
        "- step1.text, step1.url\n"
        "- step2.text, step2.url\n"
        "- step3.text, step3.url\n"
        "Also extract all Apple Support URLs mentioned in the answer as an array 'apple_urls'.\n\n"
        "Rules:\n"
        "1) Extract exactly what the answer states; do not invent or infer steps or URLs.\n"
        "2) If a specific step URL is not provided, set it to null.\n"
        "3) If the answer provides a single Apple Support URL for multiple steps, include that URL in each relevant step "
        "field if clearly associated; otherwise leave step-specific url null and still list it under 'apple_urls'.\n"
        "4) Only include URLs explicitly present in the answer text."
    )


def prompt_extract_verizon_support() -> str:
    return (
        "From the provided answer, extract Verizon's technical support contact details that the answer claims apply to "
        "Mobile/5G Home/LTE Home Internet customers experiencing network issues. Return:\n"
        "- phone_number: the phone number string as written in the answer\n"
        "- hours: the operating hours string as written in the answer\n"
        "- url: the Verizon official reference URL cited in the answer (verizon.com)\n\n"
        "Rules:\n"
        "1) Extract only what appears in the answer; do not invent any details.\n"
        "2) If any required item is missing, set it to null.\n"
        "3) The URL must be an official Verizon URL if provided (contain 'verizon.com'); otherwise return null."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def unique_non_empty(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def apple_sources_for_step(extracted: AppleSOSExtraction, step: Optional[StepInfo]) -> List[str]:
    candidates = []
    if step and step.url:
        candidates.append(step.url)
    candidates.extend(extracted.apple_urls)
    return unique_non_empty(candidates)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_apple_sequence(
    evaluator: Evaluator,
    parent_node,
    apple: AppleSOSExtraction,
) -> None:
    """
    Build and verify the Apple troubleshooting sequence (first 3 steps).
    The entire sequence is critical and sequential; each step has existence, match, and source-support checks.
    """
    seq_node = evaluator.add_sequential(
        id="Apple_Official_Troubleshooting_Sequence",
        desc="The first three troubleshooting steps from Apple's official support documentation for iPhone SOS mode, in correct sequential order",
        parent=parent_node,
        critical=True,
    )

    # Step 1: Airplane Mode
    step1_exists = evaluator.add_custom_node(
        result=(
            apple.step1 is not None
            and apple.step1.text is not None
            and apple.step1.text.strip() != ""
            and len(apple_sources_for_step(apple, apple.step1)) > 0
        ),
        id="First_Step_Exists",
        desc="First step text and Apple Support URL are provided in the answer",
        parent=seq_node,
        critical=True,
    )

    step1_match_leaf = evaluator.add_leaf(
        id="First_Step_Match_Text",
        desc="First step text matches expected: 'Toggle Airplane Mode on for at least 15 seconds then turn it off'",
        parent=seq_node,
        critical=True,
    )
    step1_text = apple.step1.text if apple.step1 and apple.step1.text else ""
    await evaluator.verify(
        claim=(
            f"The answer's first step ('{step1_text}') is equivalent to: "
            f"'{EXPECTED_APPLE_STEPS[0]}' for addressing iPhone stuck in SOS mode."
        ),
        node=step1_match_leaf,
        additional_instruction="Judge equivalence leniently; allow minor paraphrases and formatting differences. Ensure the gist is toggling Airplane Mode, with about 15 seconds on, then off.",
    )

    step1_support_leaf = evaluator.add_leaf(
        id="First_Troubleshooting_Step",
        desc="Correctly identify the first troubleshooting step: toggle Airplane Mode on for at least 15 seconds then turn it off, with reference to Apple's official support",
        parent=seq_node,
        critical=True,
    )
    step1_sources = apple_sources_for_step(apple, apple.step1)
    await evaluator.verify(
        claim=(
            "On Apple's official support page for resolving iPhone stuck in SOS mode, "
            "the first troubleshooting step is to turn on Airplane Mode for at least 15 seconds and then turn it off."
        ),
        node=step1_support_leaf,
        sources=step1_sources,
        additional_instruction=(
            "Confirm that the Apple Support documentation lists Airplane Mode toggle as step 1. "
            "Check ordering if steps are numbered or clearly sequenced. Accept minor phrasing variations."
        ),
    )

    # Step 2: Restart iPhone
    step2_exists = evaluator.add_custom_node(
        result=(
            apple.step2 is not None
            and apple.step2.text is not None
            and apple.step2.text.strip() != ""
            and len(apple_sources_for_step(apple, apple.step2)) > 0
        ),
        id="Second_Step_Exists",
        desc="Second step text and Apple Support URL are provided in the answer",
        parent=seq_node,
        critical=True,
    )

    step2_match_leaf = evaluator.add_leaf(
        id="Second_Step_Match_Text",
        desc="Second step text matches expected: 'Restart the iPhone'",
        parent=seq_node,
        critical=True,
    )
    step2_text = apple.step2.text if apple.step2 and apple.step2.text else ""
    await evaluator.verify(
        claim=(
            f"The answer's second step ('{step2_text}') is equivalent to: "
            f"'{EXPECTED_APPLE_STEPS[1]}' for addressing iPhone stuck in SOS mode."
        ),
        node=step2_match_leaf,
        additional_instruction="Judge equivalence leniently; ensure the gist is restarting the iPhone.",
    )

    step2_support_leaf = evaluator.add_leaf(
        id="Second_Troubleshooting_Step",
        desc="Correctly identify the second troubleshooting step: restart the iPhone, with reference to Apple's official support",
        parent=seq_node,
        critical=True,
    )
    step2_sources = apple_sources_for_step(apple, apple.step2)
    await evaluator.verify(
        claim=(
            "On Apple's official support page for resolving iPhone stuck in SOS mode, "
            "the second troubleshooting step is to restart the iPhone."
        ),
        node=step2_support_leaf,
        sources=step2_sources,
        additional_instruction=(
            "Confirm that the Apple Support documentation lists 'restart the iPhone' as step 2. "
            "Check ordering; accept minor phrasing variations."
        ),
    )

    # Step 3: Contact carrier
    step3_exists = evaluator.add_custom_node(
        result=(
            apple.step3 is not None
            and apple.step3.text is not None
            and apple.step3.text.strip() != ""
            and len(apple_sources_for_step(apple, apple.step3)) > 0
        ),
        id="Third_Step_Exists",
        desc="Third step text and Apple Support URL are provided in the answer",
        parent=seq_node,
        critical=True,
    )

    step3_match_leaf = evaluator.add_leaf(
        id="Third_Step_Match_Text",
        desc="Third step text matches expected: 'Contact the carrier to verify account status and check for network issues'",
        parent=seq_node,
        critical=True,
    )
    step3_text = apple.step3.text if apple.step3 and apple.step3.text else ""
    await evaluator.verify(
        claim=(
            f"The answer's third step ('{step3_text}') is equivalent to: "
            f"'{EXPECTED_APPLE_STEPS[2]}' for addressing iPhone stuck in SOS mode."
        ),
        node=step3_match_leaf,
        additional_instruction="Judge equivalence leniently; ensure the gist is contacting the carrier to verify account status and network issues.",
    )

    step3_support_leaf = evaluator.add_leaf(
        id="Third_Troubleshooting_Step",
        desc="Correctly identify the third troubleshooting step: contact the carrier to verify account status and check for network issues, with reference to Apple's official support",
        parent=seq_node,
        critical=True,
    )
    step3_sources = apple_sources_for_step(apple, apple.step3)
    await evaluator.verify(
        claim=(
            "On Apple's official support page for resolving iPhone stuck in SOS mode, "
            "the third troubleshooting step is to contact the carrier to verify account status and check for network issues."
        ),
        node=step3_support_leaf,
        sources=step3_sources,
        additional_instruction=(
            "Confirm ordering and wording; accept minor phrasing variations. "
            "Ensure the step is specifically about contacting the carrier for account/network checks."
        ),
    )


async def verify_verizon_support(
    evaluator: Evaluator,
    parent_node,
    vz: VerizonSupportExtraction,
) -> None:
    """
    Build and verify Verizon technical support contact information.
    The whole subtree is critical and parallel; add existence gate and verify both phone number and hours.
    """
    vz_node = evaluator.add_parallel(
        id="Verizon_Technical_Support_Contact",
        desc="Official Verizon technical support contact information for mobile service issues",
        parent=parent_node,
        critical=True,
    )

    # Existence gate (critical sibling prerequisite)
    exists_vz = evaluator.add_custom_node(
        result=(
            vz is not None
            and vz.phone_number is not None
            and vz.phone_number.strip() != ""
            and vz.hours is not None
            and vz.hours.strip() != ""
            and vz.url is not None
            and vz.url.strip() != ""
        ),
        id="Verizon_Info_Provided",
        desc="Verizon technical support phone number, operating hours, and official reference URL are provided in the answer",
        parent=vz_node,
        critical=True,
    )

    # Phone number: match to expected (simple check)
    phone_match_leaf = evaluator.add_leaf(
        id="Support_Phone_Number_Match_Expected",
        desc=f"Answer's Verizon technical support phone number matches expected '{EXPECTED_VERIZON_NUMBER}'",
        parent=vz_node,
        critical=True,
    )
    provided_num = vz.phone_number or ""
    await evaluator.verify(
        claim=(
            f"The phone number provided in the answer ('{provided_num}') equals the official Verizon technical support number '{EXPECTED_VERIZON_NUMBER}'."
        ),
        node=phone_match_leaf,
        additional_instruction="Allow formatting variations such as spaces, hyphens, or leading '1-'. Focus on numeric equivalence.",
    )

    # Phone number: source-supported by Verizon page
    phone_source_leaf = evaluator.add_leaf(
        id="Support_Phone_Number",
        desc="Provide the official Verizon technical support number 800-922-0204 for Mobile/5G Home/LTE Home Internet customers, verified by Verizon's official contact page",
        parent=vz_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"Verizon's official technical support phone number for Mobile/5G Home/LTE Home Internet customers experiencing network issues is {EXPECTED_VERIZON_NUMBER}."
        ),
        node=phone_source_leaf,
        sources=vz.url,
        additional_instruction=(
            "Verify on the official Verizon contact/support page. Focus on technical support or service support numbers, "
            "not sales or billing. If multiple numbers are listed, ensure the one verified is a technical support line applicable to mobile/home internet."
        ),
    )

    # Hours: match to expected (simple check)
    hours_match_leaf = evaluator.add_leaf(
        id="Support_Hours_Match_Expected",
        desc=f"Answer's Verizon technical support operating hours match expected '{EXPECTED_VERIZON_HOURS}'",
        parent=vz_node,
        critical=True,
    )
    provided_hours = vz.hours or ""
    await evaluator.verify(
        claim=(
            f"The operating hours provided in the answer ('{provided_hours}') match the official hours '{EXPECTED_VERIZON_HOURS}'."
        ),
        node=hours_match_leaf,
        additional_instruction="Allow minor phrasing variations, but time range and days must be equivalent; treat timezone specification (GMT) explicitly.",
    )

    # Hours: source-supported by Verizon page
    hours_source_leaf = evaluator.add_leaf(
        id="Support_Hours",
        desc="Provide the operating hours: 1 PM to 5 AM GMT, Sunday through Saturday, verified by Verizon's official contact page",
        parent=vz_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"Verizon's technical support operating hours are {EXPECTED_VERIZON_HOURS}."
        ),
        node=hours_source_leaf,
        sources=vz.url,
        additional_instruction=(
            "Confirm on Verizon's official contact/support page. If hours are shown in a different timezone, ensure they correspond to the claimed GMT schedule."
        ),
    )


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for iPhone SOS troubleshooting (Apple) and Verizon support contact details.
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

    # Extract both Apple steps and Verizon support info
    apple_task = evaluator.extract(
        prompt=prompt_extract_apple_steps(),
        template_class=AppleSOSExtraction,
        extraction_name="apple_sos_steps",
    )
    vz_task = evaluator.extract(
        prompt=prompt_extract_verizon_support(),
        template_class=VerizonSupportExtraction,
        extraction_name="verizon_support_info",
    )
    apple_extracted, vz_extracted = await asyncio.gather(apple_task, vz_task)

    # Ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_apple_steps": EXPECTED_APPLE_STEPS,
        "expected_apple_url_hint": EXPECTED_APPLE_URL_HINT,
        "expected_verizon_number": EXPECTED_VERIZON_NUMBER,
        "expected_verizon_hours": EXPECTED_VERIZON_HOURS,
        "expected_verizon_url": EXPECTED_VERIZON_URL,
    }, gt_type="expected_values")

    # Create critical guide node to mirror rubric
    guide_node = evaluator.add_parallel(
        id="iPhone_SOS_Troubleshooting_Guide",
        desc="Complete troubleshooting guidance for iPhone SOS mode on Verizon network",
        parent=root,
        critical=True,
    )

    # Verify Apple sequence
    await verify_apple_sequence(evaluator, guide_node, apple_extracted)

    # Verify Verizon support
    await verify_verizon_support(evaluator, guide_node, vz_extracted)

    return evaluator.get_summary()