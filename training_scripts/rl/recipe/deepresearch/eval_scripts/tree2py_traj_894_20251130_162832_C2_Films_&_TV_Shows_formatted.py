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
TASK_ID = "cara_buono_emmy_role"
TASK_DESCRIPTION = (
    "Cara Buono is an actress from The Bronx, New York City, who graduated from Columbia University with a double major. "
    "During her television career, she received a Primetime Emmy Award nomination for Outstanding Guest Actress in a Drama Series for one specific role. "
    "Identify this Emmy-nominated television role by providing the following information: "
    "(1) the character name and the show name for which she received the Emmy nomination, "
    "(2) the year she received this Emmy nomination, and "
    "(3) the character's profession in the show."
)

EXPECTED_INFO = {
    "character_name": "Dr. Faye Miller",
    "show_name": "Mad Men",
    "nomination_year": "2011",
    "profession_accepted": [
        "market research consultant",
        "strategist",
        "consumer psychologist",
        "market researcher",
        "consumer research consultant",
        "marketing strategist",
        "consumer research strategist",
    ],
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EmmyRoleExtraction(BaseModel):
    """Structured extraction of Cara Buono's Emmy-nominated role from the answer."""
    character_name: Optional[str] = None
    show_name: Optional[str] = None
    nomination_year: Optional[str] = None
    character_profession: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_emmy_role() -> str:
    return """
    Extract the Emmy-nominated television role information for Cara Buono from the answer.

    Required fields:
    1) character_name: The character name for which she received the Primetime Emmy nomination for Outstanding Guest Actress in a Drama Series.
    2) show_name: The TV show name associated with that nomination.
    3) nomination_year: The year of that Emmy nomination (use a 4-digit year if explicitly present, otherwise return null).
    4) character_profession: The character's profession in the show (extract exactly as described in the answer; if multiple are given, choose the most central/profession-like).
    5) sources: Extract any URLs cited in the answer that are relevant to this role/nomination/profession (if none are provided, return an empty list).

    Rules:
    - Extract exactly what is stated in the answer; do not infer or invent.
    - If any field is not mentioned, return null for that field (or an empty list for sources).
    """


# --------------------------------------------------------------------------- #
# Verification logic helpers                                                  #
# --------------------------------------------------------------------------- #
def build_verification_items(extracted: EmmyRoleExtraction, complete_node) -> List[Dict[str, Any]]:
    """
    Build verification items (claim, sources, node, additional_instruction) for batch verification.
    Each item maps to one leaf node and represents a single binary check.
    """
    items = []

    # Character Name
    char_node = VerificationBuilder.add_leaf_node(
        id="Character_Name",
        desc="Correctly identify the character name as Dr. Faye Miller",
        parent=complete_node,
        critical=True,
    )
    char_claim = (
        f"The character name '{(extracted.character_name or '').strip()}' matches 'Dr. Faye Miller' for the Emmy-nominated role."
    )
    char_instruction = (
        "Compare the names for equivalence. Accept minor variants (e.g., 'Faye Miller', 'Dr Faye Miller'), "
        "differences in punctuation/casing, and honorific 'Dr'. Ensure the name refers to Cara Buono's Emmy-nominated guest role."
    )
    items.append({
        "claim": char_claim,
        "sources": None,
        "node": char_node,
        "additional_instruction": char_instruction
    })

    # Show Name
    show_node = VerificationBuilder.add_leaf_node(
        id="Show_Name",
        desc="Correctly identify the show name as Mad Men",
        parent=complete_node,
        critical=True,
    )
    show_claim = (
        f"The show name '{(extracted.show_name or '').strip()}' matches 'Mad Men' for Cara Buono's Emmy-nominated role."
    )
    show_instruction = (
        "Compare the show names. Accept minor formatting differences (e.g., spacing or punctuation). "
        "Focus on whether the answer identifies 'Mad Men' as the Emmy-nominated show."
    )
    items.append({
        "claim": show_claim,
        "sources": None,
        "node": show_node,
        "additional_instruction": show_instruction
    })

    # Nomination Year
    year_node = VerificationBuilder.add_leaf_node(
        id="Nomination_Year",
        desc="Correctly state the year of the Emmy nomination as 2011",
        parent=complete_node,
        critical=True,
    )
    year_claim = (
        f"The Emmy nomination year '{(extracted.nomination_year or '').strip()}' matches '2011' for Outstanding Guest Actress in a Drama Series."
    )
    year_instruction = (
        "Verify the year equality. Accept reasonable textual variants of 2011 in the answer (e.g., 'in 2011'). "
        "Focus specifically on the Primetime Emmy nomination year for Outstanding Guest Actress in a Drama Series."
    )
    items.append({
        "claim": year_claim,
        "sources": None,
        "node": year_node,
        "additional_instruction": year_instruction
    })

    # Character Profession
    prof_node = VerificationBuilder.add_leaf_node(
        id="Character_Profession",
        desc="Correctly identify the character's profession as market research consultant or strategist",
        parent=complete_node,
        critical=True,
    )
    extracted_prof = (extracted.character_profession or "").strip()
    prof_claim = (
        f"The character's profession in the show is '{extracted_prof}', which is equivalent to a market research consultant or strategist."
    )
    prof_instruction = (
        "Determine if the extracted profession is equivalent to 'market research consultant' or 'strategist'. "
        "Accept synonyms such as 'consumer psychologist', 'market researcher', 'consumer research consultant', or 'marketing strategist'. "
        "If the answer clearly states a different profession unrelated to market/consumer research strategy, mark incorrect."
    )
    items.append({
        "claim": prof_claim,
        "sources": None,
        "node": prof_node,
        "additional_instruction": prof_instruction
    })

    return items


class VerificationBuilder:
    """Small helper to keep leaf creation concise."""
    @staticmethod
    def add_leaf_node(id: str, desc: str, parent, critical: bool = True):
        # Leaf nodes must start initialized with binary score (default 0.0) and status
        return parent.add_leaf(
            id=id,
            desc=desc,
            parent=parent,  # this is incorrect; fix to use evaluator.add_leaf with parent
            critical=critical
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
    Evaluate an answer for Cara Buono's Emmy-nominated role task.
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_emmy_role(),
        template_class=EmmyRoleExtraction,
        extraction_name="emmy_role_extraction",
    )

    # Record ground truth for transparency
    evaluator.add_ground_truth({
        "expected_character_name": EXPECTED_INFO["character_name"],
        "expected_show_name": EXPECTED_INFO["show_name"],
        "expected_nomination_year": EXPECTED_INFO["nomination_year"],
        "accepted_profession_synonyms": EXPECTED_INFO["profession_accepted"],
    }, gt_type="expected_emmy_role")

    # Build a critical aggregation node matching rubric root (Complete_Task)
    complete_node = evaluator.add_parallel(
        id="Complete_Task",
        desc="Successfully identify Cara Buono's Emmy-nominated television role and provide all required information",
        parent=root,
        critical=True
    )

    # Create four critical leaf nodes and verify them
    # Note: When parent node is critical, all children must be critical (enforced by framework).
    # We'll construct the leaves and use evaluator.verify for each.
    char_leaf = evaluator.add_leaf(
        id="Character_Name",
        desc="Correctly identify the character name as Dr. Faye Miller",
        parent=complete_node,
        critical=True,
    )
    show_leaf = evaluator.add_leaf(
        id="Show_Name",
        desc="Correctly identify the show name as Mad Men",
        parent=complete_node,
        critical=True,
    )
    year_leaf = evaluator.add_leaf(
        id="Nomination_Year",
        desc="Correctly state the year of the Emmy nomination as 2011",
        parent=complete_node,
        critical=True,
    )
    prof_leaf = evaluator.add_leaf(
        id="Character_Profession",
        desc="Correctly identify the character's profession as market research consultant or strategist",
        parent=complete_node,
        critical=True,
    )

    # Prepare claims with additional instructions
    char_claim = (
        f"The character name '{(extracted.character_name or '').strip()}' matches 'Dr. Faye Miller' for the Emmy-nominated role."
    )
    char_instruction = (
        "Compare names for equivalence. Accept minor variants (e.g., 'Faye Miller', 'Dr Faye Miller'), "
        "differences in punctuation/casing, and honorific 'Dr'. Ensure it refers to Cara Buono's Emmy-nominated guest role."
    )

    show_claim = (
        f"The show name '{(extracted.show_name or '').strip()}' matches 'Mad Men' for Cara Buono's Emmy-nominated role."
    )
    show_instruction = (
        "Compare show names. Accept minor formatting differences (e.g., spacing or punctuation). "
        "Focus on whether the answer identifies 'Mad Men' as the Emmy-nominated show."
    )

    year_claim = (
        f"The Emmy nomination year '{(extracted.nomination_year or '').strip()}' matches '2011' for Outstanding Guest Actress in a Drama Series."
    )
    year_instruction = (
        "Verify year equality. Accept reasonable textual variants of 2011 in the answer (e.g., 'in 2011'). "
        "Focus specifically on the Primetime Emmy nomination year for Outstanding Guest Actress in a Drama Series."
    )

    extracted_profession = (extracted.character_profession or "").strip()
    prof_claim = (
        f"The character's profession in the show is '{extracted_profession}', which is equivalent to a market research consultant or strategist."
    )
    prof_instruction = (
        "Determine if the extracted profession is equivalent to 'market research consultant' or 'strategist'. "
        "Accept synonyms such as 'consumer psychologist', 'market researcher', 'consumer research consultant', or 'marketing strategist'. "
        "If the answer clearly states a different profession unrelated to market/consumer research strategy, mark incorrect."
    )

    # Execute verifications (no URLs required; focus on matching the answer text to expected values)
    await evaluator.batch_verify([
        (char_claim, None, char_leaf, char_instruction),
        (show_claim, None, show_leaf, show_instruction),
        (year_claim, None, year_leaf, year_instruction),
        (prof_claim, None, prof_leaf, prof_instruction),
    ])

    # Return the evaluation summary
    return evaluator.get_summary()