import asyncio
import logging
from typing import Any, List, Optional, Dict
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "march_2026_total_lunar_eclipse_facts"
TASK_DESCRIPTION = (
    "According to NASA's official information about the March 3, 2026 total lunar eclipse, "
    "provide the following three specific facts:\n\n"
    "1. What is the duration of totality (the period when the Moon is completely within Earth's umbra)?\n"
    "2. In which constellation will the Moon be located during the eclipse?\n"
    "3. When is the next total lunar eclipse scheduled to occur after the March 2026 event?\n\n"
    "For each answer, provide a reference URL from NASA's official website (science.nasa.gov) that supports your information."
)

GROUND_TRUTH = {
    "totality_duration": "58 minutes",
    "constellation": "Leo",
    "next_total_lunar_eclipse": "December 31, 2028 / January 1, 2029",
}


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class FactItem(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EclipseFactsExtraction(BaseModel):
    totality_duration: Optional[FactItem] = None
    constellation: Optional[FactItem] = None
    next_total_lunar_eclipse: Optional[FactItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_eclipse_facts() -> str:
    return """
    Extract the three requested facts (as stated in the answer) and the supporting NASA reference URLs.

    For each of the following, extract:
    - value: the exact text the answer provides for the requested fact (do not normalize; extract verbatim).
    - sources: a list of URLs that the answer explicitly cites for this fact. IMPORTANT: include only URLs from NASA's official 'science.nasa.gov' domain. If the answer mentions other NASA subdomains (e.g., eclipse.gsfc.nasa.gov) or non-NASA domains, DO NOT include them in 'sources' here.

    Return a JSON object with exactly these fields:
    {
      "totality_duration": { "value": string|null, "sources": string[] },
      "constellation": { "value": string|null, "sources": string[] },
      "next_total_lunar_eclipse": { "value": string|null, "sources": string[] }
    }

    Field semantics:
    - totality_duration.value: duration of totality for the March 3, 2026 total lunar eclipse (e.g., "58 minutes", "about 58 minutes", "0h 58m").
    - constellation.value: the constellation where the Moon is located during the eclipse (e.g., "Leo").
    - next_total_lunar_eclipse.value: the next total lunar eclipse date after March 3, 2026 (e.g., "December 31, 2028", "January 1, 2029", "Dec 31, 2028 / Jan 1, 2029").
    - For each 'sources', include every science.nasa.gov URL the answer cites for that specific fact. If the answer provides no science.nasa.gov URL for a fact, return an empty list for that fact's 'sources'.

    Do not fabricate any values or URLs. If a value is not given in the answer, set it to null. If no qualifying NASA URLs are present for a fact, return an empty array for that fact's 'sources'.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _filter_science_nasa_urls(urls: List[str]) -> List[str]:
    """Keep only URLs whose netloc equals science.nasa.gov (case-insensitive)."""
    out = []
    for u in urls or []:
        try:
            p = urlparse(u.strip())
            if p.scheme in ("http", "https") and p.netloc.lower().endswith("science.nasa.gov"):
                out.append(u.strip())
        except Exception:
            continue
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def _safe_text(x: Optional[str]) -> str:
    return (x or "").strip()


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_q1_totality_duration(evaluator: Evaluator, parent_node, facts: EclipseFactsExtraction) -> None:
    """Question 1: Duration of totality."""
    qnode = evaluator.add_parallel(
        id="Question_1_Totality_Duration",
        desc="Verify the answer to question 1: duration of totality",
        parent=parent_node,
        critical=False,
    )

    item = facts.totality_duration or FactItem()
    nasa_urls = _filter_science_nasa_urls(item.sources)

    # Value check (critical): Must match 58 minutes (allow reasonable variants).
    val_leaf = evaluator.add_leaf(
        id="Duration_Value",
        desc="The duration of totality is correctly stated as 58 minutes",
        parent=qnode,
        critical=True,
    )
    value_text = _safe_text(item.value)
    await evaluator.verify(
        claim=(
            f"The stated totality duration in the answer is '{value_text}', and this is equivalent to 58 minutes. "
            f"Treat as correct if the answer uses minor variants such as 'about 58 minutes', '≈ 58 minutes', "
            f"'58 min', '0 h 58 m', or '00:58'."
        ),
        node=val_leaf,
        additional_instruction=(
            "This is a simple logical check against the answer text. Focus only on whether the answer's stated number "
            "corresponds to 58 minutes with common formatting or rounding."
        ),
    )

    # Existence of NASA science.nasa.gov reference for this fact (critical).
    evaluator.add_custom_node(
        result=len(nasa_urls) > 0,
        id="Duration_NASA_URL_Provided",
        desc="At least one supporting URL from NASA's science.nasa.gov is provided for the totality duration",
        parent=qnode,
        critical=True,
    )

    # NASA reference support (critical): NASA explicitly supports ~58 minutes.
    ref_leaf = evaluator.add_leaf(
        id="Duration_NASA_Reference",
        desc="A valid reference URL from NASA's official website (science.nasa.gov) is provided to support the totality duration",
        parent=qnode,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "NASA's official information (on science.nasa.gov) states that the duration of totality for the "
            "March 3, 2026 total lunar eclipse is approximately 58 minutes (accept 'about 58 minutes', '≈ 58 min', or '0 h 58 m')."
        ),
        node=ref_leaf,
        sources=nasa_urls,
        additional_instruction=(
            "Verify specifically for the March 3, 2026 total lunar eclipse. "
            "Accept minor formatting differences or rounding that clearly indicate ~58 minutes."
        ),
    )


async def verify_q2_constellation(evaluator: Evaluator, parent_node, facts: EclipseFactsExtraction) -> None:
    """Question 2: Constellation of the Moon during the eclipse."""
    qnode = evaluator.add_parallel(
        id="Question_2_Constellation",
        desc="Verify the answer to question 2: constellation location",
        parent=parent_node,
        critical=False,
    )

    item = facts.constellation or FactItem()
    nasa_urls = _filter_science_nasa_urls(item.sources)

    # Value check (critical): Must be Leo (allow 'in Leo', 'constellation Leo', case-insensitive).
    val_leaf = evaluator.add_leaf(
        id="Constellation_Value",
        desc="The Moon's constellation location during the eclipse is correctly identified as Leo",
        parent=qnode,
        critical=True,
    )
    value_text = _safe_text(item.value)
    await evaluator.verify(
        claim=(
            f"The constellation named in the answer is '{value_text}', which should correspond to Leo. "
            f"Treat as correct if it clearly indicates Leo (e.g., 'Leo', 'in Leo', or 'constellation Leo'), ignoring case."
        ),
        node=val_leaf,
        additional_instruction="This is a simple match check on the answer text; focus on equivalence to 'Leo'.",
    )

    # Existence of NASA science.nasa.gov reference for this fact (critical).
    evaluator.add_custom_node(
        result=len(nasa_urls) > 0,
        id="Constellation_NASA_URL_Provided",
        desc="At least one supporting URL from NASA's science.nasa.gov is provided for the constellation",
        parent=qnode,
        critical=True,
    )

    # NASA reference support (critical): NASA states the Moon is in Leo during the eclipse.
    ref_leaf = evaluator.add_leaf(
        id="Constellation_NASA_Reference",
        desc="A valid reference URL from NASA's official website (science.nasa.gov) is provided to support the constellation location",
        parent=qnode,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "NASA's official information (on science.nasa.gov) states that during the March 3, 2026 total lunar eclipse, "
            "the Moon is located in the constellation Leo."
        ),
        node=ref_leaf,
        sources=nasa_urls,
        additional_instruction="Verify specifically for the March 3, 2026 lunar eclipse; accept phrasing like 'in Leo' or 'constellation Leo'.",
    )


async def verify_q3_next_eclipse(evaluator: Evaluator, parent_node, facts: EclipseFactsExtraction) -> None:
    """Question 3: Next total lunar eclipse date after March 2026."""
    qnode = evaluator.add_parallel(
        id="Question_3_Next_Eclipse",
        desc="Verify the answer to question 3: next total lunar eclipse date",
        parent=parent_node,
        critical=False,
    )

    item = facts.next_total_lunar_eclipse or FactItem()
    nasa_urls = _filter_science_nasa_urls(item.sources)

    # Value check (critical): Must be Dec 31, 2028 or Jan 1, 2029 (timezone/UTC variants acceptable).
    val_leaf = evaluator.add_leaf(
        id="Next_Eclipse_Value",
        desc="The date of the next total lunar eclipse after March 2026 is correctly stated as December 31, 2028 or January 1, 2029",
        parent=qnode,
        critical=True,
    )
    value_text = _safe_text(item.value)
    await evaluator.verify(
        claim=(
            f"The answer states the next total lunar eclipse occurs on '{value_text}', which is correct if and only if "
            f"it corresponds to December 31, 2028 or January 1, 2029 (allowing UTC/time-zone related date rollovers, and "
            f"minor formatting like 'Dec 31, 2028' or 'Jan 1, 2029')."
        ),
        node=val_leaf,
        additional_instruction="Pure logical check against the answer text; accept either date due to time zones/UTC conventions.",
    )

    # Existence of NASA science.nasa.gov reference for this fact (critical).
    evaluator.add_custom_node(
        result=len(nasa_urls) > 0,
        id="Next_Eclipse_NASA_URL_Provided",
        desc="At least one supporting URL from NASA's science.nasa.gov is provided for the next total lunar eclipse date",
        parent=qnode,
        critical=True,
    )

    # NASA reference support (critical): NASA states next total lunar eclipse is Dec 31, 2028 / Jan 1, 2029.
    ref_leaf = evaluator.add_leaf(
        id="Next_Eclipse_NASA_Reference",
        desc="A valid reference URL from NASA's official website (science.nasa.gov) is provided to support the next eclipse date",
        parent=qnode,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "NASA's official information (on science.nasa.gov) states that the next total lunar eclipse after March 3, 2026 "
            "occurs on December 31, 2028 (which may appear as January 1, 2029 in UTC or different time zones)."
        ),
        node=ref_leaf,
        sources=nasa_urls,
        additional_instruction="Accept either 2028-12-31 or 2029-01-01 formulations as indicating the same eclipse, due to UTC/time zone differences.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the March 3, 2026 total lunar eclipse facts task.
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

    # Optional intermediate aggregator to mirror rubric
    facts_root = evaluator.add_parallel(
        id="March_2026_Eclipse_Facts",
        desc="Verify the three requested facts about the March 3, 2026 total lunar eclipse based on NASA's official sources",
        parent=root,
        critical=False,
    )

    # Extract structured data from the answer
    extracted_facts = await evaluator.extract(
        prompt=prompt_extract_eclipse_facts(),
        template_class=EclipseFactsExtraction,
        extraction_name="extracted_facts",
    )

    # Add ground truth context
    evaluator.add_ground_truth(
        {
            "expected_totality_duration": GROUND_TRUTH["totality_duration"],
            "expected_constellation": GROUND_TRUTH["constellation"],
            "expected_next_total_lunar_eclipse": GROUND_TRUTH["next_total_lunar_eclipse"],
        },
        gt_type="ground_truth_facts",
    )

    # Build verification subtrees (parallel across questions)
    await asyncio.gather(
        verify_q1_totality_duration(evaluator, facts_root, extracted_facts),
        verify_q2_constellation(evaluator, facts_root, extracted_facts),
        verify_q3_next_eclipse(evaluator, facts_root, extracted_facts),
    )

    # Return evaluation summary
    return evaluator.get_summary()