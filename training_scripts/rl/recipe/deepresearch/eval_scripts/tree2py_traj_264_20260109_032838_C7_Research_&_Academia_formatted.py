import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cvpr_2024_identification"
TASK_DESCRIPTION = (
    "Identify the name of the computer science conference that meets ALL of the following criteria:\n\n"
    "1. The conference took place in 2024\n"
    "2. The conference was held in Seattle, Washington, USA\n"
    "3. The venue was the Seattle Convention Center\n"
    "4. The conference lasted 5 consecutive days\n"
    "5. The conference focused on Computer Vision and Pattern Recognition\n"
    "6. The conference acceptance rate was approximately 23.6%\n"
    "7. The conference received 11,532 paper submissions\n"
    "8. The conference accepted 2,719 papers\n"
    "9. The conference is organized by IEEE/CVF\n"
    "10. The conference took place in June 2024\n"
    "11. The conference is held annually\n"
    "12. The conference included workshops and tutorials in its program\n\n"
    "What is the full name of this conference?"
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ConferenceExtraction(BaseModel):
    full_name: Optional[str] = None
    acronym: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_conference_info() -> str:
    return (
        "Extract the conference information from the answer.\n"
        "You must return:\n"
        "1) full_name: The full, official name of the conference as stated in the answer (spelled out, not just an acronym; include the year if present).\n"
        "2) acronym: The acronym of the conference if it appears (e.g., CVPR); return null if not present.\n"
        "3) sources: A list of all URLs explicitly cited in the answer that are relevant to this conference (official website pages, schedule, stats, program pages, press releases, etc.).\n"
        "Follow the SPECIAL RULES FOR URL SOURCES EXTRACTION. If no URLs are provided, return an empty list."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str]) -> str:
    return name.strip() if name else "the conference referenced in the answer"


def _additional_instruction_full_name() -> str:
    return (
        "Judge whether the response provides the conference's full official name (spelled out), "
        "not just an acronym. Accept names that include both the spelled-out official title and "
        "an acronym in parentheses. Example of acceptable: 'IEEE/CVF Conference on Computer Vision "
        "and Pattern Recognition (CVPR) 2024'. Example of insufficient: 'CVPR 2024' alone."
    )


def _additional_instruction_location() -> str:
    return (
        "Verify that the provided webpage(s) explicitly indicate the event location as Seattle, "
        "Washington, USA. Accept reasonable variants like 'Seattle, WA' or 'Seattle, Washington'."
    )


def _additional_instruction_venue() -> str:
    return (
        "Verify that the venue is the Seattle Convention Center. Accept references to its buildings "
        "such as 'Summit' or 'Arch' as part of the Seattle Convention Center, but the primary venue "
        "must be the Seattle Convention Center."
    )


def _additional_instruction_duration() -> str:
    return (
        "Verify that the event lasted five consecutive days. Often represented as dates like "
        "June 17–21, 2024 (inclusive = 5 days). Use official schedule/program pages."
    )


def _additional_instruction_field() -> str:
    return (
        "Verify that the conference focuses on Computer Vision and Pattern Recognition (CVPR). "
        "Accept reasonable wording variants like 'computer vision' and 'pattern recognition'."
    )


def _additional_instruction_acceptance_rate() -> str:
    return (
        "Verify that the acceptance rate was approximately 23.6%. Allow minor rounding differences "
        "(e.g., 23.5%–23.7%). If a page shows exact stats, use that."
    )


def _additional_instruction_submissions() -> str:
    return (
        "Verify that there were 11,532 submissions. Allow formatting variants like '11,532' or '11532'. "
        "Prefer official statistics pages."
    )


def _additional_instruction_accepted_papers() -> str:
    return (
        "Verify that 2,719 papers were accepted. Allow formatting variants like '2,719' or '2719'. "
        "Prefer official statistics pages."
    )


def _additional_instruction_organizer() -> str:
    return (
        "Verify that the conference is organized by IEEE and CVF (Computer Vision Foundation). "
        "Accept phrasing like 'IEEE/CVF'."
    )


def _additional_instruction_timing() -> str:
    return (
        "Verify that the conference took place in June 2024. Accept pages that show event dates within June 2024."
    )


def _additional_instruction_frequency() -> str:
    return (
        "Verify that the conference is held annually (yearly). "
        "Accept phrasing like 'annual conference'."
    )


def _additional_instruction_program() -> str:
    return (
        "Verify that the program included workshops and tutorials. "
        "Accept references to 'workshops', 'tutorials', or a program schedule listing them."
    )


# --------------------------------------------------------------------------- #
# Tree construction and verification                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_conference_tree(
    evaluator: Evaluator,
    root_node,
    extracted: ConferenceExtraction,
) -> None:
    # Parent node: critical + parallel (as per rubric)
    conf_node = evaluator.add_parallel(
        id="conference_identification",
        desc="Identify the conference by full name and ensure it matches the provided constraints",
        parent=root_node,
        critical=True,
    )

    # Prepare leaf nodes
    # 1. Full name provided
    leaf_full_name = evaluator.add_leaf(
        id="provide_full_conference_name",
        desc="Response provides the full name of the conference (not just an acronym)",
        parent=conf_node,
        critical=True,
    )

    # 2. Location
    leaf_location = evaluator.add_leaf(
        id="conference_location",
        desc="The conference was held in Seattle, Washington, USA",
        parent=conf_node,
        critical=True,
    )

    # 3. Venue
    leaf_venue = evaluator.add_leaf(
        id="conference_venue",
        desc="The conference venue was the Seattle Convention Center",
        parent=conf_node,
        critical=True,
    )

    # 4. Duration
    leaf_duration = evaluator.add_leaf(
        id="conference_duration",
        desc="The conference lasted 5 consecutive days",
        parent=conf_node,
        critical=True,
    )

    # 5. Field
    leaf_field = evaluator.add_leaf(
        id="conference_field",
        desc="The conference focused on Computer Vision and Pattern Recognition",
        parent=conf_node,
        critical=True,
    )

    # 6. Acceptance rate
    leaf_acceptance = evaluator.add_leaf(
        id="acceptance_rate",
        desc="The conference acceptance rate was approximately 23.6%",
        parent=conf_node,
        critical=True,
    )

    # 7. Submission count
    leaf_submissions = evaluator.add_leaf(
        id="submission_count",
        desc="The conference received 11,532 paper submissions",
        parent=conf_node,
        critical=True,
    )

    # 8. Accepted papers
    leaf_accepted = evaluator.add_leaf(
        id="accepted_papers",
        desc="The conference accepted 2,719 papers",
        parent=conf_node,
        critical=True,
    )

    # 9. Organizing body
    leaf_organizer = evaluator.add_leaf(
        id="organizing_body",
        desc="The conference is organized by IEEE/CVF",
        parent=conf_node,
        critical=True,
    )

    # 10. Timing
    leaf_timing = evaluator.add_leaf(
        id="conference_timing",
        desc="The conference took place in June 2024",
        parent=conf_node,
        critical=True,
    )

    # 11. Frequency
    leaf_frequency = evaluator.add_leaf(
        id="conference_frequency",
        desc="The conference is held annually",
        parent=conf_node,
        critical=True,
    )

    # 12. Program structure
    leaf_program = evaluator.add_leaf(
        id="program_structure",
        desc="The conference included workshops and tutorials in its program",
        parent=conf_node,
        critical=True,
    )

    conf_name = _safe_name(extracted.full_name)
    sources = extracted.sources if extracted.sources else None

    claims_and_sources = [
        (
            f"The response provides the full official conference name as '{extracted.full_name}'. "
            f"It is spelled out and not just an acronym.",
            None,
            leaf_full_name,
            _additional_instruction_full_name(),
        ),
        (
            f"{conf_name} was held in Seattle, Washington, USA.",
            sources,
            leaf_location,
            _additional_instruction_location(),
        ),
        (
            f"The venue for {conf_name} was the Seattle Convention Center.",
            sources,
            leaf_venue,
            _additional_instruction_venue(),
        ),
        (
            f"{conf_name} lasted five consecutive days.",
            sources,
            leaf_duration,
            _additional_instruction_duration(),
        ),
        (
            f"{conf_name} focuses on Computer Vision and Pattern Recognition.",
            sources,
            leaf_field,
            _additional_instruction_field(),
        ),
        (
            f"The acceptance rate for {conf_name} was approximately 23.6%.",
            sources,
            leaf_acceptance,
            _additional_instruction_acceptance_rate(),
        ),
        (
            f"{conf_name} received 11,532 submissions.",
            sources,
            leaf_submissions,
            _additional_instruction_submissions(),
        ),
        (
            f"{conf_name} accepted 2,719 papers.",
            sources,
            leaf_accepted,
            _additional_instruction_accepted_papers(),
        ),
        (
            f"{conf_name} is organized by IEEE and CVF.",
            sources,
            leaf_organizer,
            _additional_instruction_organizer(),
        ),
        (
            f"{conf_name} took place in June 2024.",
            sources,
            leaf_timing,
            _additional_instruction_timing(),
        ),
        (
            f"{conf_name} is held annually.",
            sources,
            leaf_frequency,
            _additional_instruction_frequency(),
        ),
        (
            f"The program for {conf_name} included workshops and tutorials.",
            sources,
            leaf_program,
            _additional_instruction_program(),
        ),
    ]

    # Run all verifications in parallel to avoid mutual critical-sibling skip from early failures
    await evaluator.batch_verify(claims_and_sources)


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

    # Extract conference info from the answer
    extracted_conf = await evaluator.extract(
        prompt=prompt_extract_conference_info(),
        template_class=ConferenceExtraction,
        extraction_name="conference_extraction",
    )

    # Optional: add ground truth info for bookkeeping (not used for scoring)
    evaluator.add_ground_truth({
        "expected_example": "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2024",
        "note": "This is a commonly known match for the provided constraints; used for logging only."
    })

    # Build verification tree and run checks
    await build_and_verify_conference_tree(evaluator, root, extracted_conf)

    # Return standardized summary
    return evaluator.get_summary()