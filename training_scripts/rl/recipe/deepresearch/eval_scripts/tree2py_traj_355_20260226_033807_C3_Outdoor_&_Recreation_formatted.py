import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "whistler_blackcomb_2025_26_late_closing"
TASK_DESCRIPTION = (
    "For the 2025/26 winter season at Whistler Blackcomb ski resort in British Columbia, Canada, "
    "identify which mountain (Whistler Mountain or Blackcomb Mountain) remains open for skiing later into the spring. "
    "Provide the closing date for that mountain, its vertical drop in feet, and its top elevation in feet. "
    "Include URL references from official Whistler Blackcomb sources to verify each specification."
)

# Constraint values to verify (as specified by the rubric)
EXPECTED_CLOSING_DATE = "May 18, 2026"
EXPECTED_VERTICAL_DROP_DIGITS = "5280"  # digits only, we will allow "5,280" equivalently
EXPECTED_TOP_ELEVATION_DIGITS = "7494"  # digits only, we will allow "7,494" equivalently

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ChosenMountainSpec(BaseModel):
    """Information provided in the answer for the chosen later-closing mountain."""
    name: Optional[str] = None
    closing_date: Optional[str] = None
    vertical_drop_ft: Optional[str] = None
    top_elevation_ft: Optional[str] = None
    closing_date_urls: List[str] = Field(default_factory=list)
    vertical_drop_urls: List[str] = Field(default_factory=list)
    top_elevation_urls: List[str] = Field(default_factory=list)


class OtherMountainInfo(BaseModel):
    """Information in the answer for the other mountain (used for comparison of closing dates)."""
    name: Optional[str] = None
    closing_date: Optional[str] = None
    closing_date_urls: List[str] = Field(default_factory=list)


class WhistlerSeasonExtraction(BaseModel):
    """Complete extraction structure from the agent's answer."""
    chosen: Optional[ChosenMountainSpec] = None
    other: Optional[OtherMountainInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_whistler_blackcomb() -> str:
    return """
    From the answer, extract the information about Whistler Blackcomb's two mountains for the 2025/26 winter season.

    You must extract exactly the following fields:

    chosen:
      - name: The mountain the answer claims remains open later into spring (either "Whistler Mountain" or "Blackcomb Mountain"; allow short forms like "Whistler" or "Blackcomb").
      - closing_date: The closing date stated for that chosen mountain (as written in the answer).
      - vertical_drop_ft: The vertical drop for the chosen mountain, in feet (as written in the answer; keep units if present).
      - top_elevation_ft: The top elevation for the chosen mountain, in feet (as written in the answer; keep units if present).
      - closing_date_urls: An array of all URLs cited that support the chosen mountain's closing date.
      - vertical_drop_urls: An array of all URLs cited that support the chosen mountain's vertical drop.
      - top_elevation_urls: An array of all URLs cited that support the chosen mountain's top elevation.

    other:
      - name: The other mountain's name (the mountain not chosen as later-closing, if the answer mentions it).
      - closing_date: The closing date stated for the other mountain (as written).
      - closing_date_urls: An array of all URLs cited to support the other mountain's closing date.

    IMPORTANT:
    - Only extract what is explicitly stated in the answer.
    - If a field is missing in the answer, set it to null or [] as appropriate.
    - URLs can be plain or markdown links; extract the actual URL strings. Do not invent URLs.
    - Do not normalize or transform numbers; keep them as in the answer (e.g., "5,280 ft", "7494 feet", etc.).
    - The answer may use multiple URLs; collect them all per field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _digits_only(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return "".join(ch for ch in s if ch.isdigit())


def _contains_expected_date(answer_date: Optional[str], month_day_year: str) -> bool:
    """Check the answer's closing_date string contains the expected date components."""
    if not answer_date:
        return False
    # Basic heuristic: require month name and day+year presence
    expected_lower = month_day_year.lower()
    ans_lower = answer_date.lower()
    # Split the expected into tokens to allow minor formatting variations
    # Expect "may 18, 2026" => check "may", "18", "2026" presence
    tokens = ["may", "18", "2026"]
    return all(tok in ans_lower for tok in tokens)


def _canonical_mountain_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower()
    if "whistler" in n:
        return "Whistler Mountain"
    if "blackcomb" in n:
        return "Blackcomb Mountain"
    return name.strip()


def _infer_other_name(chosen_name: Optional[str]) -> Optional[str]:
    cn = _canonical_mountain_name(chosen_name)
    if cn is None:
        return None
    return "Blackcomb Mountain" if "Whistler" in cn else "Whistler Mountain"


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_identification_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: WhistlerSeasonExtraction,
) -> None:
    """
    Build the identification subtree:
    - Ensures the chosen mountain is Whistler or Blackcomb (exactly one named).
    - Verifies with official URLs that the chosen mountain's closing date is later than the other mountain's closing date
      by checking each closing date against sources and then logically comparing the two dates.
    """
    identify_node = evaluator.add_parallel(
        id="Identify_Later_Closing_Mountain",
        desc="Correctly identify the later-closing mountain among Whistler Mountain and Blackcomb Mountain for the 2025/26 season, supported by official Whistler Blackcomb source(s).",
        parent=parent_node,
        critical=True,
    )

    chosen = extracted.chosen or ChosenMountainSpec()
    other = extracted.other or OtherMountainInfo()

    # Leaf: Mountain is Whistler or Blackcomb, and exactly one named (use simple verify against the answer text)
    mountain_leaf = evaluator.add_leaf(
        id="Mountain_Is_Whistler_Or_Blackcomb",
        desc="Names exactly one mountain, and it is either Whistler Mountain or Blackcomb Mountain.",
        parent=identify_node,
        critical=True,
    )
    chosen_name = _canonical_mountain_name(chosen.name)
    claim_name = (
        f"The answer selects exactly one mountain as the later-closing one, and it is either Whistler Mountain or Blackcomb Mountain: '{chosen.name or ''}'. "
        f"Treat 'Whistler' as 'Whistler Mountain' and 'Blackcomb' as 'Blackcomb Mountain'."
    )
    await evaluator.verify(
        claim=claim_name,
        node=mountain_leaf,
        additional_instruction="Check the answer text to ensure exactly one mountain is named as the later-closing one, and that it refers to Whistler Mountain or Blackcomb Mountain (allow short forms).",
    )

    # Sequential group: verify each closing date by official source, then compare
    compare_node = evaluator.add_sequential(
        id="Later_Than_Other_Mountain_Verified",
        desc="Provides official Whistler Blackcomb URL reference(s) showing the chosen mountain’s 2025/26 closing date is later than the other mountain’s 2025/26 closing date.",
        parent=identify_node,
        critical=True,
    )

    # Leaf A: Verify chosen mountain closing date with official sources
    chosen_close_leaf = evaluator.add_leaf(
        id="Chosen_Closing_Date_Supported",
        desc="Chosen mountain closing date is supported by official Whistler Blackcomb URL(s).",
        parent=compare_node,
        critical=True,
    )
    chosen_close_claim = (
        f"For the 2025/26 winter season, {chosen_name or 'the chosen mountain'} closes on {chosen.closing_date or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=chosen_close_claim,
        node=chosen_close_leaf,
        sources=chosen.closing_date_urls,
        additional_instruction=(
            "Only accept official Whistler Blackcomb sources (pages on whistlerblackcomb.com or Vail Resorts official "
            "sites clearly about Whistler Blackcomb). The page must explicitly reference the 2025/26 season and the closing date for the stated mountain."
        ),
    )

    # Prepare other mountain name: if not extracted, infer by complement
    other_name = _canonical_mountain_name(other.name) or _infer_other_name(chosen_name)

    # Leaf B: Verify other mountain closing date with official sources
    other_close_leaf = evaluator.add_leaf(
        id="Other_Closing_Date_Supported",
        desc="Other mountain closing date is supported by official Whistler Blackcomb URL(s).",
        parent=compare_node,
        critical=True,
    )
    other_close_claim = (
        f"For the 2025/26 winter season, {other_name or 'the other mountain'} closes on {other.closing_date or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=other_close_claim,
        node=other_close_leaf,
        sources=other.closing_date_urls,
        additional_instruction=(
            "Only accept official Whistler Blackcomb sources (pages on whistlerblackcomb.com or Vail Resorts official "
            "sites clearly about Whistler Blackcomb). The page must explicitly reference the 2025/26 season and the closing date for the stated mountain."
        ),
    )

    # Leaf C: Logical comparison (no URLs needed) – check later than
    compare_logic_leaf = evaluator.add_leaf(
        id="Chosen_Closing_Date_Is_Later",
        desc="The chosen mountain's closing date is later than the other mountain's closing date.",
        parent=compare_node,
        critical=True,
    )
    compare_claim = (
        f"The closing date '{chosen.closing_date or 'UNKNOWN'}' is later than '{other.closing_date or 'UNKNOWN'}'. "
        f"Consider standard date ordering (Month Day, Year) and allow minor formatting differences."
    )
    await evaluator.verify(
        claim=compare_claim,
        node=compare_logic_leaf,
        additional_instruction="Perform a pure logical comparison of the two dates provided in the answer; if either date is missing or ambiguous, mark as Incorrect.",
    )


async def build_specs_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: WhistlerSeasonExtraction,
) -> None:
    """
    Build the specifications subtree for the chosen mountain:
    - Closing date must match the constraint (May 18, 2026) and be verified by official URLs.
    - Vertical drop must be 5,280 feet and be verified by official URLs.
    - Top elevation must be 7,494 feet and be verified by official URLs.
    """
    specs_node = evaluator.add_parallel(
        id="Provide_Required_Specs_For_Chosen_Mountain",
        desc="Provide the required specifications for the chosen later-closing mountain, matching all constraint values, each verified with official Whistler Blackcomb URL reference(s).",
        parent=parent_node,
        critical=True,
    )

    chosen = extracted.chosen or ChosenMountainSpec()
    chosen_name = _canonical_mountain_name(chosen.name) or "the chosen mountain"

    # Closing date check (sequential: value in answer, then source verification)
    closing_seq = evaluator.add_sequential(
        id="Closing_Date_Matches_Constraint_With_Source",
        desc="States the chosen mountain’s 2025/26 closing date as May 18, 2026 and includes official Whistler Blackcomb URL reference(s) verifying this closing date.",
        parent=specs_node,
        critical=True,
    )
    # Value exists and matches
    closing_value_node = evaluator.add_custom_node(
        result=_contains_expected_date(chosen.closing_date, EXPECTED_CLOSING_DATE),
        id="Closing_Date_Value_Equals_May_18_2026",
        desc="The answer explicitly states the chosen mountain’s closing date as May 18, 2026.",
        parent=closing_seq,
        critical=True,
    )
    # Source verification leaf
    closing_src_leaf = evaluator.add_leaf(
        id="Closing_Date_Source_Verified",
        desc="Official Whistler Blackcomb URL(s) verify the closing date is May 18, 2026 for the chosen mountain (2025/26 season).",
        parent=closing_seq,
        critical=True,
    )
    closing_claim = f"For the 2025/26 winter season, {chosen_name} closes on {EXPECTED_CLOSING_DATE}."
    await evaluator.verify(
        claim=closing_claim,
        node=closing_src_leaf,
        sources=chosen.closing_date_urls,
        additional_instruction=(
            "Only accept official Whistler Blackcomb sources (whistlerblackcomb.com or relevant Vail Resorts official pages) "
            "that explicitly confirm the closing date May 18, 2026 for the specified mountain in the 2025/26 season."
        ),
    )

    # Vertical drop check (sequential: value in answer, then source verification)
    vdrop_seq = evaluator.add_sequential(
        id="Vertical_Drop_Matches_Constraint_With_Source",
        desc="States the chosen mountain’s vertical drop as 5,280 feet and includes official Whistler Blackcomb URL reference(s) verifying this vertical drop.",
        parent=specs_node,
        critical=True,
    )
    vdrop_value_node = evaluator.add_custom_node(
        result=_digits_only(chosen.vertical_drop_ft) == EXPECTED_VERTICAL_DROP_DIGITS,
        id="Vertical_Drop_Value_Equals_5280_ft",
        desc="The answer explicitly states the chosen mountain’s vertical drop as 5,280 feet.",
        parent=vdrop_seq,
        critical=True,
    )
    vdrop_src_leaf = evaluator.add_leaf(
        id="Vertical_Drop_Source_Verified",
        desc="Official Whistler Blackcomb URL(s) verify the vertical drop is 5,280 feet for the chosen mountain.",
        parent=vdrop_seq,
        critical=True,
    )
    vdrop_claim = f"The vertical drop of {chosen_name} is 5,280 feet."
    await evaluator.verify(
        claim=vdrop_claim,
        node=vdrop_src_leaf,
        sources=chosen.vertical_drop_urls,
        additional_instruction=(
            "Only accept official Whistler Blackcomb sources (whistlerblackcomb.com or relevant Vail Resorts official pages). "
            "Units must be feet; minor formatting like commas is acceptable."
        ),
    )

    # Top elevation check (sequential: value in answer, then source verification)
    top_seq = evaluator.add_sequential(
        id="Top_Elevation_Matches_Constraint_With_Source",
        desc="States the chosen mountain’s top elevation as 7,494 feet and includes official Whistler Blackcomb URL reference(s) verifying this top elevation.",
        parent=specs_node,
        critical=True,
    )
    top_value_node = evaluator.add_custom_node(
        result=_digits_only(chosen.top_elevation_ft) == EXPECTED_TOP_ELEVATION_DIGITS,
        id="Top_Elevation_Value_Equals_7494_ft",
        desc="The answer explicitly states the chosen mountain’s top elevation as 7,494 feet.",
        parent=top_seq,
        critical=True,
    )
    top_src_leaf = evaluator.add_leaf(
        id="Top_Elevation_Source_Verified",
        desc="Official Whistler Blackcomb URL(s) verify the top elevation is 7,494 feet for the chosen mountain.",
        parent=top_seq,
        critical=True,
    )
    top_claim = f"The top elevation of {chosen_name} is 7,494 feet."
    await evaluator.verify(
        claim=top_claim,
        node=top_src_leaf,
        sources=chosen.top_elevation_urls,
        additional_instruction=(
            "Only accept official Whistler Blackcomb sources (whistlerblackcomb.com or relevant Vail Resorts official pages). "
            "Units must be feet; minor formatting like commas is acceptable."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Whistler Blackcomb 2025/26 late-closing mountain task.
    Returns a structured summary dictionary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Add a critical main node under the evaluator's root to enforce rubric criticality
    main = evaluator.add_sequential(
        id="Root",
        desc=(
            "For the 2025/26 winter season at Whistler Blackcomb, identify which mountain (Whistler Mountain or Blackcomb Mountain) "
            "remains open later into spring, and provide that mountain’s closing date, vertical drop (feet), and top elevation (feet), each supported by official Whistler Blackcomb URLs, meeting all listed constraints."
        ),
        parent=root,
        critical=True,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_whistler_blackcomb(),
        template_class=WhistlerSeasonExtraction,
        extraction_name="whistler_blackcomb_extraction",
    )

    # Add ground truth/constraint info for transparency
    evaluator.add_ground_truth(
        {
            "season": "2025/26",
            "expected_values": {
                "closing_date": EXPECTED_CLOSING_DATE,
                "vertical_drop_ft_digits": EXPECTED_VERTICAL_DROP_DIGITS,
                "top_elevation_ft_digits": EXPECTED_TOP_ELEVATION_DIGITS,
            }
        },
        gt_type="constraints",
    )

    # Build verification subtree for identification
    await build_identification_subtree(evaluator, main, extracted)

    # Build verification subtree for required specs
    await build_specs_subtree(evaluator, main, extracted)

    # Return the standardized summary
    return evaluator.get_summary()