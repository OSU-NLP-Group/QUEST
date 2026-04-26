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
TASK_ID = "tx_school_districts_4"
TASK_DESCRIPTION = (
    "Identify four different public school districts in Texas that each meet all of the following criteria:\n\n"
    "1. The district is located in one of these Texas counties: Tarrant, Dallas, Collin, or Denton\n"
    "2. The district has a student enrollment of at least 25,000 students for the 2024-2025 school year\n"
    "3. The district serves students from kindergarten through grade 12\n"
    "4. The district's minority enrollment (non-white students) is at least 50% of total enrollment\n\n"
    "For each district you identify, provide:\n"
    "- The district's full name\n"
    "- The county where it is located\n"
    "- The total student enrollment for 2024-2025\n"
    "- The percentage of minority enrollment\n"
    "- A reference URL that supports your enrollment and demographic information"
)

ALLOWED_COUNTIES = {"tarrant", "dallas", "collin", "denton"}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DistrictEntry(BaseModel):
    name: Optional[str] = None
    county: Optional[str] = None
    enrollment_2024_2025: Optional[str] = None
    minority_percent: Optional[str] = None
    reference_url: Optional[str] = None
    # Optional free-form field if the answer explicitly mentions grade span
    grade_span: Optional[str] = None


class DistrictList(BaseModel):
    districts: List[DistrictEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    Extract up to four distinct Texas public school districts listed in the answer. For each district, extract:
    1. name: the district's full name as written in the answer text
    2. county: the county name as written in the answer text (e.g., "Dallas County" or "Dallas")
    3. enrollment_2024_2025: the total student enrollment number for the 2024–2025 school year exactly as written (keep commas, plus signs, or words like "approximately" if present; do not convert to a number)
    4. minority_percent: the percentage of minority enrollment exactly as written (e.g., "70%", "about 65%", etc.)
    5. reference_url: a single URL that the answer cites to support enrollment and demographic information for that district (must be an explicit URL in the answer). If multiple are present, pick the most relevant one; if none is present, set to null.
    6. grade_span: if the answer explicitly states the grade span (e.g., "K–12", "Pre-K–12", "EE–12"), extract it exactly; otherwise set to null.

    Return a JSON object with:
    {
      "districts": [
        { "name": ..., "county": ..., "enrollment_2024_2025": ..., "minority_percent": ..., "reference_url": ..., "grade_span": ... },
        ...
      ]
    }

    IMPORTANT:
    - Do not invent or infer any information not explicitly present in the answer text.
    - Preserve the formatting of enrollment and percentage strings exactly as written.
    - Only include explicit URLs that appear in the answer for each district; do not infer URLs.
    - Keep the original county text as written (with or without the word "County").
    - If the answer lists more than four districts, include only the first four in order of appearance.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"#{n}")


def is_nonempty_text(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def has_valid_url(url: Optional[str]) -> bool:
    if not is_nonempty_text(url):
        return False
    url_s = url.strip().lower()
    return url_s.startswith("http://") or url_s.startswith("https://")


def normalize_county_label(county_text: Optional[str]) -> str:
    """
    Normalize county text for messaging only (not for strict matching).
    E.g., "Dallas County" -> "Dallas County"; "Dallas" -> "Dallas County".
    """
    if not is_nonempty_text(county_text):
        return ""
    c = county_text.strip()
    base = c.replace("County", "").replace("county", "").strip()
    if base.lower() in ALLOWED_COUNTIES:
        # Recompose canonical "X County" for clarity in the claim text.
        return f"{base.title()} County"
    # If it's some other formatting, just return original string.
    # The verifier will rely on the web page content.
    return c


# --------------------------------------------------------------------------- #
# Verification logic per district                                             #
# --------------------------------------------------------------------------- #
async def verify_one_district(
    evaluator: Evaluator,
    root_node,
    district: DistrictEntry,
    idx_one_based: int
) -> None:
    """
    Build verification subtree for a single district and run checks.
    All 5 leaf checks are critical under this district node (parallel aggregation).
    """
    # Create the district main node (parallel, non-critical to allow partial credit across districts)
    district_node = evaluator.add_parallel(
        id=f"district_{idx_one_based}",
        desc=f"{ordinal(idx_one_based)} qualifying Texas school district meets all requirements",
        parent=root_node,
        critical=False
    )

    # Prepare commonly used fields
    name = (district.name or "").strip()
    county_display = normalize_county_label(district.county)
    enrollment_str = (district.enrollment_2024_2025 or "").strip()
    minority_str = (district.minority_percent or "").strip()
    ref_url = (district.reference_url or "").strip()

    # ------------------------------------------------------------------- #
    # Reference URL check (must be verified first to serve as a gate)     #
    # ------------------------------------------------------------------- #
    ref_leaf = evaluator.add_leaf(
        id=f"district_{idx_one_based}_reference",
        desc="Provides valid URL reference supporting the district's enrollment and demographic information",
        parent=district_node,
        critical=True
    )

    # If no valid URL present, mark as failed directly and do not call verify
    if not has_valid_url(ref_url):
        ref_leaf.score = 0.0
        ref_leaf.status = "failed"
    else:
        ref_claim = (
            f"This webpage provides enrollment and demographic (race/ethnicity or minority enrollment percentage) "
            f"information for the school district named '{name}'."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=ref_leaf,
            sources=ref_url,
            additional_instruction=(
                "Verify that the page is relevant to the specified school district and contains BOTH: "
                "(1) an overall student enrollment figure (preferably for 2024–2025 or clearly labeled for 2024-25) "
                "and (2) demographic information (either a minority enrollment percentage or a race/ethnicity breakdown "
                "from which minority percentage can be inferred). If either part is missing, mark as not supported."
            )
        )

    # Build a convenience list of extra preconditions so that if the reference page fails,
    # subsequent checks can be skipped meaningfully.
    prereq = [ref_leaf]

    # ------------------------------------------------------------------- #
    # Location (county) check                                             #
    # ------------------------------------------------------------------- #
    location_leaf = evaluator.add_leaf(
        id=f"district_{idx_one_based}_location",
        desc="District is located in one of the following Texas counties: Tarrant, Dallas, Collin, or Denton",
        parent=district_node,
        critical=True
    )
    if not (is_nonempty_text(name) and is_nonempty_text(district.county) and has_valid_url(ref_url)):
        # Missing essential components to verify; fail early
        location_leaf.score = 0.0
        location_leaf.status = "failed"
    else:
        loc_claim = (
            f"The webpage indicates that the school district named '{name}' is located in {county_display}, Texas, "
            f"which is one of the following counties: Tarrant, Dallas, Collin, or Denton."
        )
        await evaluator.verify(
            claim=loc_claim,
            node=location_leaf,
            sources=ref_url,
            additional_instruction=(
                "Accept reasonable variants like 'Dallas County' vs 'Dallas'. If the district spans multiple counties, "
                "pass if at least one of Tarrant, Dallas, Collin, or Denton is clearly indicated. "
                "If the county is not explicitly stated on the page, mark as not supported."
            ),
            extra_prerequisites=prereq
        )

    # ------------------------------------------------------------------- #
    # Enrollment (>= 25,000) for 2024-2025 check                          #
    # ------------------------------------------------------------------- #
    enrollment_leaf = evaluator.add_leaf(
        id=f"district_{idx_one_based}_enrollment",
        desc="District has student enrollment of at least 25,000 students for the 2024-2025 school year",
        parent=district_node,
        critical=True
    )
    if not (is_nonempty_text(name) and is_nonempty_text(enrollment_str) and has_valid_url(ref_url)):
        enrollment_leaf.score = 0.0
        enrollment_leaf.status = "failed"
    else:
        enr_claim = (
            f"The webpage shows that the student enrollment for the 2024–2025 school year for the district '{name}' "
            f"is {enrollment_str}, and this number is at least 25,000 students."
        )
        await evaluator.verify(
            claim=enr_claim,
            node=enrollment_leaf,
            sources=ref_url,
            additional_instruction=(
                "Verify that the page provides a 2024–2025 (or equivalently formatted 2024-25) overall enrollment figure "
                "for the district. Allow minor formatting differences (commas, plus signs, 'approximately'). "
                "Pass only if the page supports the stated enrollment and it is clearly >= 25,000."
            ),
            extra_prerequisites=prereq
        )

    # ------------------------------------------------------------------- #
    # Grade span K-12 check                                               #
    # ------------------------------------------------------------------- #
    grade_span_leaf = evaluator.add_leaf(
        id=f"district_{idx_one_based}_grade_span",
        desc="District serves students from kindergarten through grade 12",
        parent=district_node,
        critical=True
    )
    if not has_valid_url(ref_url):
        grade_span_leaf.score = 0.0
        grade_span_leaf.status = "failed"
    else:
        # We do not require the answer to list grade span explicitly; verify via the reference page.
        gs_claim = (
            f"The webpage indicates that the district '{name}' serves grades kindergarten through 12th grade (K–12)."
        )
        await evaluator.verify(
            claim=gs_claim,
            node=grade_span_leaf,
            sources=ref_url,
            additional_instruction=(
                "Accept variants such as K–12, Pre-K–12, PK–12, EE–12, or similar that clearly include the full span "
                "from Kindergarten through 12th grade. If unclear or not stated, mark as not supported."
            ),
            extra_prerequisites=prereq
        )

    # ------------------------------------------------------------------- #
    # Diversity (minority >= 50%) check                                   #
    # ------------------------------------------------------------------- #
    diversity_leaf = evaluator.add_leaf(
        id=f"district_{idx_one_based}_diversity",
        desc="District's minority enrollment is at least 50% of total enrollment",
        parent=district_node,
        critical=True
    )
    if not (is_nonempty_text(name) and is_nonempty_text(minority_str) and has_valid_url(ref_url)):
        diversity_leaf.score = 0.0
        diversity_leaf.status = "failed"
    else:
        div_claim = (
            f"The webpage shows that the minority enrollment (non-white students) for the district '{name}' is "
            f"{minority_str}, which is at least 50% of total enrollment."
        )
        await evaluator.verify(
            claim=div_claim,
            node=diversity_leaf,
            sources=ref_url,
            additional_instruction=(
                "Interpret 'minority' as non-white students. Accept if the page explicitly provides a minority % "
                ">= 50% or provides a race/ethnicity breakdown from which the non-white sum clearly reaches or exceeds 50%. "
                "Allow minor rounding differences. If the page does not show a clear minority percentage or breakdown, "
                "mark as not supported."
            ),
            extra_prerequisites=prereq
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
    Evaluate an answer for the 'Texas public school districts' task.
    """
    # Initialize evaluator with a parallel root to allow partial credit across districts
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify four public school districts in Texas that each meet all specified criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract up to 4 districts from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictList,
        extraction_name="districts_extraction"
    )

    # Keep only the first four items; pad if fewer
    districts: List[DistrictEntry] = list(extracted.districts[:4])
    while len(districts) < 4:
        districts.append(DistrictEntry())

    # Add custom info for transparency
    evaluator.add_custom_info(
        {
            "allowed_counties": sorted(list(ALLOWED_COUNTIES)),
            "num_districts_parsed": len(extracted.districts),
        },
        info_type="config",
        info_name="evaluation_context"
    )

    # Build and verify each district subtree
    for i in range(4):
        await verify_one_district(
            evaluator=evaluator,
            root_node=root,
            district=districts[i],
            idx_one_based=i + 1
        )

    return evaluator.get_summary()