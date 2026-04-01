import asyncio
import logging
from typing import Dict, List, Optional

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "marvel_rivals"
TASK_DESCRIPTION = """
Marvel Rivals is a video game featuring iconic Marvel characters, each with a unique set of abilities.

Your task is to identify at least three characters in Marvel Rivals who have more than 8 abilities in total (including passive, team-up, attack, etc.). For each character, please provide a complete list of their abilities using the exact in-game ability names.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data-models for extracted info                                              #
# --------------------------------------------------------------------------- #
class CharacterAbility(BaseModel):
    character_name: Optional[str] = None
    abilities: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)

class MarvelRivalsCharacters(BaseModel):
    characters: List[CharacterAbility] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Prompt for extracting character information                                  #
# --------------------------------------------------------------------------- #
def prompt_extract_characters() -> str:
    return """
Extract every Marvel Rivals character mentioned in the answer along with:
1. `character_name`: the exact in-game name of the character.
2. `abilities`: a list of all ability names exactly as they appear in-game (including passive, team-up, attack, etc.).
3. `source_urls`: any URLs mentioned in the answer that support this character's abilities.

Return JSON with the structure:
{
  "characters": [
    {
      "character_name": "...",
      "abilities": ["Ability One", "Ability Two", ...],
      "source_urls": ["https://...", ...]
    },
    ...
  ]
}

If a character is listed without abilities or URLs, include empty arrays for those fields.
"""

# --------------------------------------------------------------------------- #
# Verification helpers for each character                                       #
# --------------------------------------------------------------------------- #
async def verify_character(
    evaluator: Evaluator,
    parent_node,
    char_index: int,
    character: CharacterAbility,
) -> None:
    """
    Verify a single character's abilities and provenance.
    """
    # Create parent node for this character (sequential to implement short-circuit logic)
    char_node = evaluator.add_sequential(
        id=f"character_{char_index}",
        desc=f"Evaluation of character slot {char_index}: {character.character_name or '<missing>'}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # Check if character exists
    character_exists = character.character_name is not None and character.character_name.strip() != ""
    
    # 1. Existence check
    existence_node = evaluator.add_custom_node(
        result=character_exists,
        id=f"character_{char_index}_exists",
        desc=f"Check if character slot {char_index} has a valid character name",
        parent=char_node,
        critical=True  # Critical to gate subsequent checks
    )

    # 2. Quantity verification
    quantity_node = evaluator.add_custom_node(
        result=len(character.abilities) >= 9,
        id=f"character_{char_index}_quantity",
        desc=f"Character has {len(character.abilities)} abilities (requires > 8)",
        parent=char_node,
        critical=True
    )

    # 3. Provenance verification
    provenance_node = evaluator.add_leaf(
        id=f"character_{char_index}_provenance",
        desc=f"Ability names for {character.character_name or '<missing>'} exactly match in-game names, as supported by the provided URLs",
        parent=char_node,
        critical=True
    )

    # Build the claim for verification
    claim_text = (
        f"In Marvel Rivals, {character.character_name} has the following abilities: "
        + "; ".join(character.abilities)
        + "."
    )

    # Always call verify regardless of data existence
    await evaluator.verify(
        claim=claim_text,
        node=provenance_node,
        sources=character.source_urls,  # Pass the list directly, even if empty
        additional_instruction="Verify that all listed abilities exactly match the in-game ability names for this character. The source should clearly show these ability names."
    )

# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: openai.AsyncAzureOpenAI,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a Marvel Rivals answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator with parallel strategy for root
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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Extract structured info from the answer ---------------- #
    parsed_info = await evaluator.extract(
        prompt=prompt_extract_characters(),
        template_class=MarvelRivalsCharacters,
        extraction_name="marvel_rivals_characters"
    )

    # -------- 3. Build verification tree -------------------------------- #
    # Evaluate up to 3 characters (allow partial credit)
    chars = parsed_info.characters[:3]
    
    # Pad if fewer than 3
    while len(chars) < 3:
        chars.append(CharacterAbility())

    # Verify each character
    for idx, char in enumerate(chars, start=1):
        await verify_character(evaluator, root, idx, char)

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()