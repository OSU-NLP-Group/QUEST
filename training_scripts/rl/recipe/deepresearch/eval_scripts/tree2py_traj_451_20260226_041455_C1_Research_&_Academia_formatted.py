import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "esther_perel_masters"
TASK_DESCRIPTION = "At which university did Esther Perel earn her master's degree, and in what field of study?"

EXPECTED_UNIVERSITY = "Lesley University"
EXPECTED_FIELD_VARIANTS = ["expressive art therapy", "expressive arts therapy"]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MastersInfo(BaseModel):
    university_name: Optional[str] = None
    degree_field: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_masters_info() -> str:
    return """
    Extract the master's degree details for Esther Perel as presented in the agent's answer.

    Return a JSON object with the following fields:
    - university_name: The name of the university the answer claims awarded Esther Perel's master's degree. If multiple institutions are mentioned, choose the one explicitly tied to the master's degree. If unclear or missing, return null.
    - degree_field: The field of study for that master's degree as exactly claimed in the answer (use the answer's wording; do not invent). If missing, return null.
    - sources: An array of all URL(s) explicitly provided in the answer as sources for these degree details (including any "Sources" section). Extract the actual URLs (plain URLs or from markdown links). Remove duplicates. If none are provided, return an empty array.

    Do not infer or add information beyond what is explicitly in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification workflow                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, extracted: MastersInfo) -> None:
    # Create rubric root node as a critical parallel group
    rubric_root = evaluator.add_parallel(
        id="Esther_Perel_Masters_Institution",
        desc="Correctly identify the university where Esther Perel earned her master's degree and the field of study",
        parent=evaluator.root,
        critical=True
    )

    # Prepare sources (may be empty)
    sources_list = extracted.sources or []

    # 1) University verification leaf
    univ_node = evaluator.add_leaf(
        id="University_Name",
        desc="The university name is Lesley University",
        parent=rubric_root,
        critical=True
    )

    univ_claim = (
        f"In the agent's answer, the university where Esther Perel earned her master's degree must be '{EXPECTED_UNIVERSITY}'. "
        f"The answer reports the university as '{(extracted.university_name or '').strip()}'. "
        f"Judge Correct only if BOTH are satisfied: "
        f"(A) The answer text explicitly claims '{EXPECTED_UNIVERSITY}' as the master's institution (case-insensitive; "
        f"accept 'Lesley College' only if it is clearly the historical name referring to Lesley University), AND "
        f"(B) the provided source URL(s) explicitly support that Esther Perel earned a master's degree from {EXPECTED_UNIVERSITY}."
    )

    await evaluator.verify(
        claim=univ_claim,
        node=univ_node,
        sources=sources_list,
        additional_instruction=(
            "Checklist for your decision:\n"
            "1) Read the full answer (provided above) and confirm it explicitly says Lesley University for the master's school. "
            "   Treat capitalization and punctuation flexibly. If the answer states a different institution or omits the school, mark Incorrect.\n"
            "2) If the answer provided NO source URLs, mark Incorrect.\n"
            "3) If URLs are provided, open them and verify they explicitly support that Esther Perel earned a master's degree from Lesley University. "
            "   If they do not support the claim, mark Incorrect.\n"
            "Only if both 1) and 3) are satisfied should you mark Correct."
        )
    )

    # 2) Degree field verification leaf
    field_node = evaluator.add_leaf(
        id="Degree_Field",
        desc="The degree field is expressive art therapy (or expressive arts therapy)",
        parent=rubric_root,
        critical=True
    )

    field_variants_str = "', '".join(EXPECTED_FIELD_VARIANTS)
    field_claim = (
        f"In the agent's answer, the field of study for Esther Perel's master's degree must be either "
        f"'expressive art therapy' or 'expressive arts therapy'. "
        f"The answer reports the field as '{(extracted.degree_field or '').strip()}'. "
        f"Judge Correct only if BOTH are satisfied: "
        f"(A) The answer text explicitly claims one of the accepted variants ('{field_variants_str}') "
        f"(case-insensitive; minor hyphenation is acceptable; do not accept the generic phrase 'expressive therapies' "
        f"unless the source explicitly equates it with 'expressive arts therapy'), AND "
        f"(B) the provided source URL(s) explicitly support that the master's field was expressive arts therapy (or expressive art therapy)."
    )

    await evaluator.verify(
        claim=field_claim,
        node=field_node,
        sources=sources_list,
        additional_instruction=(
            "Checklist for your decision:\n"
            "1) Read the full answer and confirm it explicitly names the field as 'expressive art therapy' or 'expressive arts therapy' "
            "(allow hyphenation/case variations). If the answer only says 'expressive therapies' without clearly equating it to 'expressive arts therapy', mark Incorrect.\n"
            "2) If the answer provided NO source URLs, mark Incorrect.\n"
            "3) Using the provided URLs, verify they explicitly support that the master's field is expressive arts therapy (or expressive art therapy). "
            "If not supported, mark Incorrect.\n"
            "Only if both 1) and 3) are satisfied should you mark Correct."
        )
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
    # Initialize evaluator with a parallel root
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_masters_info(),
        template_class=MastersInfo,
        extraction_name="masters_info"
    )

    # Ground truth info (for transparency in summary)
    evaluator.add_ground_truth({
        "expected_university": EXPECTED_UNIVERSITY,
        "accepted_field_variants": EXPECTED_FIELD_VARIANTS
    }, gt_type="ground_truth_masters")

    # Build verification tree and run checks
    await build_and_verify(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()