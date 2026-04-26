import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dietary_guidelines_2025_2030_release_protein_pyramid"
TASK_DESCRIPTION = """
In January 2026, the U.S. Department of Health and Human Services and the U.S. Department of Agriculture jointly released updated Dietary Guidelines for Americans (2025-2030), which included significant changes to protein recommendations and introduced a redesigned food pyramid. Identify the exact release date (month, day, and year) of these guidelines. State the new recommended protein intake range in grams per kilogram of body weight per day. Using this range, calculate what the minimum and maximum daily protein intake would be in grams for a person weighing 70 kilograms. Finally, describe the major structural change made to the food pyramid visualization, specifically explaining which food groups are now positioned at the base of the pyramid. Provide reference URLs from official USDA or HHS sources to support your answer.
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
OFFICIAL_DOMAINS = (
    "hhs.gov",
    "usda.gov",
    "dietaryguidelines.gov",
    "health.gov",       # HHS
    "nutrition.gov",    # USDA
)


def is_official_url(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(netloc == d or netloc.endswith("." + d) for d in OFFICIAL_DOMAINS)


def filter_official_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    out = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if is_official_url(u) and u not in seen:
            out.append(u)
            seen.add(u)
    return out


def parse_range_from_text(range_text: Optional[str]) -> (Optional[float], Optional[float]):
    """
    Try to parse numeric lower/upper bounds from a textual range like "1.2–1.7 g/kg/day" or "1.2 to 1.7 g/kg/day".
    Returns (lower, upper) as floats or (None, None) if not parsable.
    """
    if not range_text or not isinstance(range_text, str):
        return None, None

    text = range_text.strip().lower()
    # Normalize dashes
    text = text.replace("–", "-").replace("—", "-").replace(" to ", "-")
    # Extract first two numbers in order
    nums = re.findall(r"(\d+(?:\.\d+)?)", text)
    if len(nums) >= 2:
        try:
            lo = float(nums[0])
            hi = float(nums[1])
            if lo <= hi:
                return lo, hi
        except Exception:
            return None, None
    return None, None


def has_complete_january_2026_date(date_text: Optional[str]) -> bool:
    """
    Check that the date string includes:
    - Month: January (allow "January" or "Jan")
    - Day: 1-31
    - Year: 2026
    """
    if not date_text or not isinstance(date_text, str):
        return False
    text = date_text.strip()
    # Accept "January 12, 2026", "Jan 12, 2026", "January 12 2026"
    pattern = re.compile(
        r"\b(January|Jan\.?)\s+(\d{1,2})(?:st|nd|rd|th)?\,?\s*(2026)\b",
        flags=re.IGNORECASE
    )
    m = pattern.search(text)
    if not m:
        return False
    day = int(m.group(2))
    return 1 <= day <= 31


def nearly_equal(a: float, b: float, abs_tol: float = 1.5) -> bool:
    try:
        return abs(float(a) - float(b)) <= abs_tol
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ReleaseSection(BaseModel):
    release_date: Optional[str] = None
    release_date_urls: List[str] = Field(default_factory=list)


class ProteinSection(BaseModel):
    range_text: Optional[str] = None
    lower_g_per_kg: Optional[float] = None
    upper_g_per_kg: Optional[float] = None
    min_protein_g_70kg: Optional[float] = None
    max_protein_g_70kg: Optional[float] = None
    protein_range_urls: List[str] = Field(default_factory=list)


class PyramidSection(BaseModel):
    inversion_description: Optional[str] = None
    base_groups: List[str] = Field(default_factory=list)  # e.g., ["protein", "vegetables"]
    pyramid_urls: List[str] = Field(default_factory=list)


class GuidelinesExtraction(BaseModel):
    release: Optional[ReleaseSection] = None
    protein: Optional[ProteinSection] = None
    pyramid: Optional[PyramidSection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_guidelines() -> str:
    return """
    Extract the requested information exactly as stated in the answer text. Return a JSON object with fields:
    - release:
        - release_date: the complete release date string as written (month name, day, year), e.g., "January 12, 2026".
        - release_date_urls: all URLs the answer cites for the release date; include only official USDA/HHS domains (hhs.gov, usda.gov, dietaryguidelines.gov, health.gov, nutrition.gov).
    - protein:
        - range_text: the stated protein intake range including units (e.g., "1.2–1.7 g/kg/day").
        - lower_g_per_kg: the numeric lower bound of the range (float), if present.
        - upper_g_per_kg: the numeric upper bound of the range (float), if present.
        - min_protein_g_70kg: the stated minimum grams per day for a 70 kg person (float), if present in the answer.
        - max_protein_g_70kg: the stated maximum grams per day for a 70 kg person (float), if present.
        - protein_range_urls: all URLs the answer cites for the protein range; include only official USDA/HHS domains as above.
    - pyramid:
        - inversion_description: short snippet from the answer describing the structural change (e.g., "the pyramid was inverted").
        - base_groups: a list of the food groups the answer claims are now at the base of the pyramid; use short lowercase names (e.g., "protein", "vegetables", "grains", "fruits", "dairy", "oils").
        - pyramid_urls: all URLs the answer cites for the pyramid change; include only official USDA/HHS domains as above.

    Rules:
    - Extract only what appears in the answer. Do not invent values.
    - If a required item is missing, set it to null or an empty list.
    - For numeric fields, return pure numbers when possible.
    - For URL arrays, include only URLs explicitly present in the answer and within the allowed official domains.
    """.strip()


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_release_section(evaluator: Evaluator, parent_node, data: GuidelinesExtraction) -> None:
    node = evaluator.add_parallel(
        id="Guidelines_Release_Date",
        desc="Provides the release date of the guidelines",
        parent=parent_node,
        critical=False
    )

    rel: ReleaseSection = data.release or ReleaseSection()

    # Leaf: Date_With_Month_Day_Year (critical) - format/existence check
    date_ok = has_complete_january_2026_date(rel.release_date)
    evaluator.add_custom_node(
        result=date_ok,
        id="Date_With_Month_Day_Year",
        desc="Includes complete date with month (January), day, and year (2026)",
        parent=node,
        critical=True
    )

    # Leaf: Release_Date_Source_URL (critical) - URL-grounded verification of the date value
    valid_urls = filter_official_urls(rel.release_date_urls)
    if not valid_urls or not rel.release_date:
        evaluator.add_custom_node(
            result=False,
            id="Release_Date_Source_URL",
            desc="Provides official USDA or HHS reference URL confirming the release date",
            parent=node,
            critical=True
        )
    else:
        leaf = evaluator.add_leaf(
            id="Release_Date_Source_URL",
            desc="Provides official USDA or HHS reference URL confirming the release date",
            parent=node,
            critical=True
        )
        claim = f"The Dietary Guidelines for Americans 2025–2030 were released on {rel.release_date}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=valid_urls,
            additional_instruction="Only pass if the official USDA/HHS page clearly states or confirms this exact release date."
        )


async def verify_protein_section(evaluator: Evaluator, parent_node, data: GuidelinesExtraction) -> None:
    node = evaluator.add_sequential(
        id="Protein_Intake_Recommendation_Analysis",
        desc="Provides and applies the new protein recommendation range",
        parent=parent_node,
        critical=False
    )

    prot: ProteinSection = data.protein or ProteinSection()

    # Try to ensure numeric lower/upper
    lo = prot.lower_g_per_kg
    hi = prot.upper_g_per_kg
    if lo is None or hi is None:
        parsed_lo, parsed_hi = parse_range_from_text(prot.range_text)
        if lo is None:
            lo = parsed_lo
        if hi is None:
            hi = parsed_hi

    # Leaf: Protein_Range_In_Grams_Per_Kg (critical) - existence/format within the answer
    units_ok = False
    if isinstance(prot.range_text, str):
        txt = prot.range_text.lower()
        units_ok = ("g/kg" in txt) or ("grams per kilogram" in txt)
        # Prefer daily rate mention if present
        # Accept if "per day" present (tolerant)
        # Not strictly required for pass if g/kg present, but improves specificity
    prange_ok = (lo is not None and hi is not None and isinstance(prot.range_text, str) and units_ok)
    evaluator.add_custom_node(
        result=prange_ok,
        id="Protein_Range_In_Grams_Per_Kg",
        desc="States the protein recommendation range in g/kg/day format",
        parent=node,
        critical=True
    )

    # Child aggregator: Calculation_For_70kg_Person (critical)
    calc_node = evaluator.add_parallel(
        id="Calculation_For_70kg_Person",
        desc="Calculates the daily protein range in grams for a 70 kg person",
        parent=node,
        critical=True
    )

    # Leaf: Calculation_Accuracy (critical) - arithmetic check
    calc_ok = False
    if lo is not None and hi is not None and prot.min_protein_g_70kg is not None and prot.max_protein_g_70kg is not None:
        expected_min = lo * 70.0
        expected_max = hi * 70.0
        calc_ok = nearly_equal(expected_min, prot.min_protein_g_70kg, abs_tol=1.5) and \
                  nearly_equal(expected_max, prot.max_protein_g_70kg, abs_tol=1.5)

    evaluator.add_custom_node(
        result=calc_ok,
        id="Calculation_Accuracy",
        desc="The calculated minimum and maximum values correctly apply the stated g/kg range to 70 kg (i.e., multiplies the lower and upper bounds by 70)",
        parent=calc_node,
        critical=True
    )

    # Leaf: Protein_Recommendation_Source_URL (critical) - URL-grounded verification of the stated range
    valid_urls = filter_official_urls(prot.protein_range_urls)
    if not valid_urls or (lo is None or hi is None):
        evaluator.add_custom_node(
            result=False,
            id="Protein_Recommendation_Source_URL",
            desc="Provides reference URL from official USDA or HHS source confirming the protein recommendation range",
            parent=calc_node,
            critical=True
        )
    else:
        leaf = evaluator.add_leaf(
            id="Protein_Recommendation_Source_URL",
            desc="Provides reference URL from official USDA or HHS source confirming the protein recommendation range",
            parent=calc_node,
            critical=True
        )
        if prot.range_text:
            claim = f"The recommended protein intake range is {prot.range_text} (grams per kilogram of body weight per day)."
        else:
            claim = f"The recommended protein intake range is between {lo} and {hi} g/kg/day."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=valid_urls,
            additional_instruction=(
                "Only pass if the official USDA/HHS page explicitly supports the same numeric range. "
                "Minor rounding (e.g., 1.2 vs 1.20) is acceptable."
            )
        )

    # Record parsed numeric info for transparency
    evaluator.add_custom_info(
        info={
            "parsed_lower_g_per_kg": lo,
            "parsed_upper_g_per_kg": hi,
            "stated_min_g_70kg": prot.min_protein_g_70kg,
            "stated_max_g_70kg": prot.max_protein_g_70kg,
            "expected_min_g_70kg": None if lo is None else lo * 70.0,
            "expected_max_g_70kg": None if hi is None else hi * 70.0
        },
        info_type="parsed_protein_numbers",
        info_name="protein_numbers_debug"
    )


async def verify_pyramid_section(evaluator: Evaluator, parent_node, data: GuidelinesExtraction) -> None:
    node = evaluator.add_parallel(
        id="Food_Pyramid_Structural_Change",
        desc="Describes the major structural change to the food pyramid visualization",
        parent=parent_node,
        critical=False
    )

    pyr: PyramidSection = data.pyramid or PyramidSection()

    # Leaf: Pyramid_Inversion_Described (critical) - ensure the answer mentions inversion
    inv_leaf = evaluator.add_leaf(
        id="Pyramid_Inversion_Described",
        desc="Explains that the pyramid was inverted compared to previous versions",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly explains that the redesigned food pyramid was inverted (flipped/reversed) compared to previous versions.",
        node=inv_leaf,
        additional_instruction="Consider synonyms like 'flipped' or 'reversed' as indicating inversion."
    )

    # Leaf: Food_Group_Repositioning_At_Base (critical) - ensure the answer states base groups
    base_leaf = evaluator.add_leaf(
        id="Food_Group_Repositioning_At_Base",
        desc="Identifies which food groups are now positioned at the base of the pyramid (protein and vegetables)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that protein (or 'protein foods') and vegetables are positioned at the base of the redesigned food pyramid.",
        node=base_leaf,
        additional_instruction="Allow 'protein foods' as equivalent to 'protein'."
    )

    # Leaf: Food_Pyramid_Source_URL (critical) - URL-grounded verification
    valid_urls = filter_official_urls(pyr.pyramid_urls)
    if not valid_urls:
        evaluator.add_custom_node(
            result=False,
            id="Food_Pyramid_Source_URL",
            desc="Provides reference URL from official USDA or HHS source confirming the food pyramid structural changes",
            parent=node,
            critical=True
        )
    else:
        leaf = evaluator.add_leaf(
            id="Food_Pyramid_Source_URL",
            desc="Provides reference URL from official USDA or HHS source confirming the food pyramid structural changes",
            parent=node,
            critical=True
        )
        claim = (
            "The official USDA/HHS source confirms that the redesigned food pyramid is inverted compared to previous "
            "versions and that the base of the pyramid consists of protein (protein foods) and vegetables."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=valid_urls,
            additional_instruction="Only pass if the page clearly supports both the inversion and the base groups being protein and vegetables."
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
    Evaluate an answer for the Dietary Guidelines 2025–2030 release/protein/pyramid task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall, sections are independent; allow partial credit
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

    # Optional top-level grouping node (non-critical to allow partial credit across sections)
    top = evaluator.add_parallel(
        id="Dietary_Guidelines_Complete_Answer",
        desc="Complete and accurate information about the Dietary Guidelines for Americans 2025–2030 released in January 2026",
        parent=root,
        critical=False
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_guidelines(),
        template_class=GuidelinesExtraction,
        extraction_name="guidelines_extraction"
    )

    # Verifications
    await verify_release_section(evaluator, top, extracted)
    await verify_protein_section(evaluator, top, extracted)
    await verify_pyramid_section(evaluator, top, extracted)

    return evaluator.get_summary()