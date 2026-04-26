import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "vocalist_steelers_2026"
TASK_DESCRIPTION = """
A vocalist performed the national anthem at a Pittsburgh Steelers playoff game in January 2026, delivering what many described as a 'flawless' performance despite freezing temperatures. Identify this vocalist by providing: (1) Her full birth name (first, middle, last) and professional stage name, (2) The city and state where she was born and raised, (3) The university she attended, (4) The year and the artist she worked with for her first major professional gig as a background vocalist. Provide URL references supporting each piece of information.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VocalistExtraction(BaseModel):
    # Core identification
    vocalist_name: Optional[str] = None  # The identified person (could be stage or birth name)
    birth_name: Optional[str] = None
    stage_name: Optional[str] = None

    # Origin and education
    birth_city: Optional[str] = None
    birth_state: Optional[str] = None
    raised_city: Optional[str] = None
    raised_state: Optional[str] = None
    university: Optional[str] = None

    # Career details
    first_major_gig_year: Optional[str] = None
    first_major_gig_artist: Optional[str] = None
    professional_years_statement: Optional[str] = None
    grammy_accreditation_statement: Optional[str] = None

    # Source URLs per fact
    performance_event_urls: List[str] = Field(default_factory=list)
    birth_name_urls: List[str] = Field(default_factory=list)
    stage_name_urls: List[str] = Field(default_factory=list)
    born_and_raised_urls: List[str] = Field(default_factory=list)
    university_urls: List[str] = Field(default_factory=list)
    first_major_gig_urls: List[str] = Field(default_factory=list)
    professional_years_urls: List[str] = Field(default_factory=list)
    grammy_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_vocalist_details() -> str:
    return """
    You must extract the single vocalist identified in the answer who performed the U.S. national anthem at a Pittsburgh Steelers playoff game in January 2026, and all requested biographical/career details, along with the specific supporting URLs cited in the answer for each detail.

    Extract the following fields exactly as they appear in the answer:
    1) vocalist_name: The identified vocalist (can be a stage or birth name).
    2) birth_name: The full legal birth name (first, middle, last) if provided (e.g., "Ashley Norelle Simpson"); else null.
    3) stage_name: The professional stage name if provided (e.g., "Norelle"); else null.
    4) birth_city: City of birth; else null.
    5) birth_state: State of birth; else null.
    6) raised_city: City where she was raised; else null.
    7) raised_state: State where she was raised; else null.
    8) university: University she attended; else null.
    9) first_major_gig_year: The year of her first major professional background vocal gig; else null.
    10) first_major_gig_artist: The artist she supported in that first major background gig; else null.
    11) professional_years_statement: The statement or number of years (approximate is fine) claiming how long she has sung professionally; else null.
    12) grammy_accreditation_statement: The statement indicating "Grammy-accredited"/Grammy credentials (e.g., credited on Grammy-nominated/winning work); else null.

    Also extract the exact URLs (as full links) that the answer cites for each of the following categories (list may be empty if none were given):
    - performance_event_urls: URLs that verify she performed the national anthem at a Pittsburgh Steelers playoff game in January 2026.
    - birth_name_urls: URLs that support the full birth name.
    - stage_name_urls: URLs that support the professional stage name.
    - born_and_raised_urls: URLs that support being born AND raised in Cleveland, Ohio (both birth and upbringing).
    - university_urls: URLs that support that she attended John Carroll University.
    - first_major_gig_urls: URLs that support the first major background vocalist gig details (year + artist).
    - professional_years_urls: URLs that support "approximately 17+ years" (or at least 17 years) of professional singing.
    - grammy_urls: URLs that support "Grammy-accredited" (e.g., credits/recognition tied to the Grammys).

    Important:
    - Only include URLs explicitly present in the answer text. Do not invent or infer URLs.
    - Return null for any missing scalar fields and [] for any missing URL lists.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_list(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls or [])


def _merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if isinstance(u, str):
                u2 = u.strip()
                if u2 and u2 not in seen:
                    seen.add(u2)
                    merged.append(u2)
    return merged


def _display_name(ex: VocalistExtraction) -> str:
    return (ex.stage_name or ex.birth_name or ex.vocalist_name or "the vocalist").strip()


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_event_block(evaluator: Evaluator, parent_node, ex: VocalistExtraction):
    """
    Build the event verification block (critical; first in sequence):
    - Check URL presence
    - Verify the claim with provided URLs
    """
    event_block = evaluator.add_sequential(
        id="event_performance",
        desc="Event performance verification block",
        parent=parent_node,
        critical=True
    )

    # URL presence (critical)
    evaluator.add_custom_node(
        result=_nonempty_list(ex.performance_event_urls),
        id="event_performance_url_provided",
        desc="A supporting URL is provided for the January 2026 Steelers playoff national anthem performance",
        parent=event_block,
        critical=True
    )

    # Evidence-backed verification (critical)
    event_leaf = evaluator.add_leaf(
        id="event_performance_verified",
        desc="Provides a URL that verifies the identified vocalist performed the national anthem at a Pittsburgh Steelers playoff game in January 2026.",
        parent=event_block,
        critical=True
    )
    name_for_claim = _display_name(ex)
    claim = (
        f"{name_for_claim} performed the U.S. national anthem at a Pittsburgh Steelers playoff game in January 2026."
    )
    await evaluator.verify(
        claim=claim,
        node=event_leaf,
        sources=ex.performance_event_urls,
        additional_instruction=(
            "Accept synonymous phrasing like 'Star-Spangled Banner'. "
            "Ensure the game was an NFL playoff game in January 2026 (e.g., Wild Card/Divisional) for the Pittsburgh Steelers."
        ),
    )


async def build_and_verify_identity_constraints(evaluator: Evaluator, parent_node, ex: VocalistExtraction):
    """
    Build and verify all identity and constraint checks with URL support.
    Each sub-item is a critical sequential block: (1) URL present, (2) claim verified by URLs.
    """
    top = evaluator.add_parallel(
        id="identity_and_constraints_verified_with_urls",
        desc="Verifies (with URLs) that the vocalist matches all stated constraints and provides the requested details.",
        parent=parent_node,
        critical=True
    )

    # 1) Full birth name exactly "Ashley Norelle Simpson"
    birthname_block = evaluator.add_sequential(
        id="full_birth_name_main",
        desc="Birth name verification block",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_list(ex.birth_name_urls),
        id="full_birth_name_url_present",
        desc="URL is provided to support the full birth name",
        parent=birthname_block,
        critical=True
    )
    bn_leaf = evaluator.add_leaf(
        id="full_birth_name_matches_constraint_with_url",
        desc="Birth name is exactly 'Ashley Norelle Simpson' (first, middle, last) and a supporting URL is provided.",
        parent=birthname_block,
        critical=True
    )
    await evaluator.verify(
        claim="Her full birth name is Ashley Norelle Simpson.",
        node=bn_leaf,
        sources=ex.birth_name_urls,
        additional_instruction="Require an exact name match ignoring letter casing and extra whitespace; middle name 'Norelle' must be present."
    )

    # 2) Stage name exactly "Norelle"
    stage_block = evaluator.add_sequential(
        id="stage_name_main",
        desc="Stage name verification block",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_list(ex.stage_name_urls),
        id="stage_name_url_present",
        desc="URL is provided to support the professional stage name",
        parent=stage_block,
        critical=True
    )
    st_leaf = evaluator.add_leaf(
        id="stage_name_matches_constraint_with_url",
        desc="Professional stage name is exactly 'Norelle' and a supporting URL is provided.",
        parent=stage_block,
        critical=True
    )
    await evaluator.verify(
        claim="Her professional stage name is Norelle.",
        node=st_leaf,
        sources=ex.stage_name_urls,
        additional_instruction="Accept simple variants like capitalization; must clearly indicate her professional/performing name is 'Norelle'."
    )

    # 3) Born AND raised in Cleveland, Ohio
    bornraised_block = evaluator.add_sequential(
        id="born_and_raised_main",
        desc="Born and raised location verification block",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_list(ex.born_and_raised_urls),
        id="born_and_raised_url_present",
        desc="URL is provided to support 'born and raised in Cleveland, Ohio'",
        parent=bornraised_block,
        critical=True
    )
    br_leaf = evaluator.add_leaf(
        id="born_and_raised_matches_constraint_with_url",
        desc="Born AND raised in Cleveland, Ohio (city + state) and a supporting URL is provided.",
        parent=bornraised_block,
        critical=True
    )
    await evaluator.verify(
        claim="She was born and raised in Cleveland, Ohio.",
        node=br_leaf,
        sources=ex.born_and_raised_urls,
        additional_instruction="Both birth and upbringing must reference Cleveland, OH (allow nearby area phrasing if page explicitly states 'born and raised in Cleveland')."
    )

    # 4) Attended John Carroll University
    university_block = evaluator.add_sequential(
        id="university_main",
        desc="University verification block",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_list(ex.university_urls),
        id="university_url_present",
        desc="URL is provided to support university attendance",
        parent=university_block,
        critical=True
    )
    uni_leaf = evaluator.add_leaf(
        id="university_matches_constraint_with_url",
        desc="Attended John Carroll University and a supporting URL is provided.",
        parent=university_block,
        critical=True
    )
    await evaluator.verify(
        claim="She attended John Carroll University.",
        node=uni_leaf,
        sources=ex.university_urls,
        additional_instruction="Accept 'attended', 'studied at', or 'graduated from' John Carroll University as valid support."
    )

    # 5) First major background vocalist gig in 2008 for John Legend
    firstgig_block = evaluator.add_sequential(
        id="first_major_background_gig_main",
        desc="First major background vocalist gig verification block",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_list(ex.first_major_gig_urls),
        id="first_major_background_gig_url_present",
        desc="URL is provided to support first major background vocalist gig details",
        parent=firstgig_block,
        critical=True
    )
    fg_leaf = evaluator.add_leaf(
        id="first_major_background_gig_matches_constraint_with_url",
        desc="First major professional gig as a background vocalist was in 2008 for John Legend, and supporting URL(s) are provided.",
        parent=firstgig_block,
        critical=True
    )
    await evaluator.verify(
        claim="Her first major professional background vocalist gig was in 2008 for John Legend.",
        node=fg_leaf,
        sources=ex.first_major_gig_urls,
        additional_instruction="The evidence should pair both components: year 2008 and artist John Legend; accept synonyms like 'background singer'/'background vocals'."
    )

    # 6) Professional years approximately 17+ years
    years_block = evaluator.add_sequential(
        id="professional_years_main",
        desc="Professional years verification block",
        parent=top,
        critical=True
    )
    merged_years_urls = _merge_urls(ex.professional_years_urls, ex.first_major_gig_urls)
    evaluator.add_custom_node(
        result=_nonempty_list(merged_years_urls),
        id="professional_years_url_present",
        desc="URL is provided to support 'approximately 17+ years' of professional singing",
        parent=years_block,
        critical=True
    )
    yrs_leaf = evaluator.add_leaf(
        id="professional_years_matches_constraint_with_url",
        desc="Evidence shows she has been singing professionally for approximately 17+ years, and a supporting URL is provided.",
        parent=years_block,
        critical=True
    )
    await evaluator.verify(
        claim="She has been singing professionally for at least 17 years.",
        node=yrs_leaf,
        sources=merged_years_urls,
        additional_instruction=(
            "It is acceptable if the page states an initial professional year such as 2008; "
            "by 2025/2026 this implies 17+ years. Any phrasing like 'over 17 years' or "
            "'since 2008' should be treated as supporting 'approximately 17+ years'."
        )
    )

    # 7) Grammy-accredited
    grammy_block = evaluator.add_sequential(
        id="grammy_accredited_main",
        desc="Grammy accreditation verification block",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_list(ex.grammy_urls),
        id="grammy_accredited_url_present",
        desc="URL is provided to support 'Grammy-accredited'",
        parent=grammy_block,
        critical=True
    )
    gr_leaf = evaluator.add_leaf(
        id="grammy_accredited_matches_constraint_with_url",
        desc="Evidence shows she is Grammy-accredited, and a supporting URL is provided.",
        parent=grammy_block,
        critical=True
    )
    await evaluator.verify(
        claim="She has Grammy credentials (e.g., Grammy-nominated or Grammy-winning credits or accreditation).",
        node=gr_leaf,
        sources=ex.grammy_urls,
        additional_instruction=(
            "Accept evidence indicating Grammy-nominated/winning credits, official Grammy recognition, "
            "or similar credible 'Grammy-accredited' phrasing tied directly to her work."
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
    """
    Evaluate an answer for the Steelers playoff national anthem vocalist identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # high-level flow: event check → identity constraints
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
    extracted: VocalistExtraction = await evaluator.extract(
        prompt=prompt_extract_vocalist_details(),
        template_class=VocalistExtraction,
        extraction_name="vocalist_details",
    )

    # Add ground truth constraints for clarity and auditing (not used as a check)
    evaluator.add_ground_truth({
        "required_birth_name_exact": "Ashley Norelle Simpson",
        "required_stage_name_exact": "Norelle",
        "required_born_and_raised": "Cleveland, Ohio",
        "required_university": "John Carroll University",
        "required_first_gig": {"year": "2008", "artist": "John Legend"},
        "required_prof_years": "approximately 17+ years",
        "required_grammy": "Grammy-accredited (credible Grammy-linked credentials)"
    }, gt_type="constraints")

    # Build the task node (critical, sequential)
    task_node = evaluator.add_sequential(
        id="vocalist_identification_task",
        desc="Identify the vocalist who performed the national anthem at a Pittsburgh Steelers playoff game in January 2026 and verify all required biographical/career constraints with URL support.",
        parent=root,
        critical=True
    )

    # 1) Event performance verification (critical, first)
    await build_and_verify_event_block(evaluator, task_node, extracted)

    # 2) Identity and constraints with URLs (critical, parallel)
    await build_and_verify_identity_constraints(evaluator, task_node, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()