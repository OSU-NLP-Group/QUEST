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
TASK_ID = "film_release_date_2025_debut_actress"
TASK_DESCRIPTION = (
    "What is the exact theatrical release date of the film that satisfies all of the following criteria: "
    "(1) The film is directed by an actress who is making her feature-length directorial debut in 2025; "
    "(2) The film stars June Squibb in the lead role; "
    "(3) The film was presented at the 2025 Cannes Film Festival in the Un Certain Regard selection; "
    "(4) The director has previously appeared at the Cannes Film Festival multiple times as an actress; "
    "(5) The film had its theatrical release in November 2025. Provide the specific date (month, day, and year) "
    "and include at least one reference URL that confirms this release date."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmInfoExtraction(BaseModel):
    """
    Structured information extracted from the answer about the film and its release details.
    """
    film_title: Optional[str] = None
    director_name: Optional[str] = None

    # Director-related constraint fields
    director_debut_year: Optional[str] = None  # e.g., "2025"
    director_is_actress: Optional[bool] = None  # True if identified as an actress in the answer
    director_cannes_appearances_as_actress: Optional[str] = None  # e.g., "multiple", "several", "3 times"

    # Lead actor
    lead_actor: Optional[str] = None  # should be "June Squibb"

    # Cannes selection info
    cannes_year: Optional[str] = None  # expecting "2025"
    cannes_section: Optional[str] = None  # expecting "Un Certain Regard"

    # Release date info
    release_date_str: Optional[str] = None  # e.g., "November 14, 2025"
    release_month: Optional[str] = None     # e.g., "November"
    release_day: Optional[str] = None       # e.g., "14"
    release_year: Optional[str] = None      # e.g., "2025"

    # URLs
    general_sources: List[str] = Field(default_factory=list)  # All URLs mentioned in the answer
    release_date_sources: List[str] = Field(default_factory=list)  # URLs specifically cited for the release date


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_film_info() -> str:
    return """
    Extract the following structured information from the answer about the identified film and its release details.

    Required fields:
    - film_title: The film's title.
    - director_name: The director's full name.
    - director_debut_year: The year of the director's feature-length directorial debut (if stated; do not infer).
    - director_is_actress: Return true if the answer explicitly states or implies the director is an actress (has an acting career); otherwise false or null.
    - director_cannes_appearances_as_actress: If the answer states that the director has previously appeared at the Cannes Film Festival multiple times as an actress, capture the phrasing (e.g., "multiple", "several", "three times"). If not stated, return null.
    - lead_actor: The lead actor's name (the answer must indicate June Squibb in the lead role).
    - cannes_year: The year the film appeared at Cannes (if stated; expected "2025").
    - cannes_section: The specific section at Cannes (if stated; expected "Un Certain Regard").
    - release_date_str: The specific theatrical release date provided in the answer in a natural language format (e.g., "November 14, 2025"); if no specific date is provided, return null.
    - release_month: The month of the release date (e.g., "November"); if unavailable, return null.
    - release_day: The day of the release date (e.g., "14"); if unavailable, return null.
    - release_year: The year of the release date (e.g., "2025"); if unavailable, return null.

    URL fields:
    - general_sources: Extract all URLs mentioned in the answer (including plain URLs or markdown links). These can be any references associated with the film details.
    - release_date_sources: Extract URLs that specifically relate to or are cited to confirm the theatrical release date. If the answer does not distinguish, include any URLs that plausibly confirm the release date. If none are provided, return an empty array.

    Rules:
    - Do not invent any information. Only extract what is explicitly present in the answer.
    - If a field is not mentioned, set it to null (or an empty array for URL lists).
    - For URLs, extract actual URL strings; ignore obviously invalid URLs. If a URL is missing a protocol, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_str(val: Optional[str]) -> str:
    return val or ""


def combine_urls(primary: List[str], fallback: List[str]) -> List[str]:
    """
    Prefer primary list if non-empty; otherwise return fallback list.
    Deduplicate while preserving order.
    """
    seen = set()
    result: List[str] = []
    for url in (primary if primary else fallback):
        if url and url not in seen:
            seen.add(url)
            result.append(url)
    return result


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root,
    data: FilmInfoExtraction
) -> None:
    """
    Build the verification tree according to the rubric and perform all checks.
    """

    # Create the top-level critical node mirroring rubric "Film_and_Release_Date_Verification"
    film_main_node = evaluator.add_parallel(
        id="Film_and_Release_Date_Verification",
        desc="Verify that the answer correctly identifies a film meeting all specified constraints and provides its exact theatrical release date with supporting references",
        parent=root,
        critical=True
    )

    # Sub-node: Verify all five constraints (turn into explicit leaf checks to avoid conflating multiple checks)
    constraints_node = evaluator.add_parallel(
        id="Film_Satisfies_All_Constraints",
        desc="The identified film must satisfy all five criteria: (1) directed by an actress making her feature-length directorial debut in 2025, (2) stars June Squibb in the lead role, (3) was presented at the 2025 Cannes Film Festival in the Un Certain Regard selection, (4) the director has previously appeared at Cannes Film Festival multiple times as an actress, and (5) had its theatrical release in November 2025",
        parent=film_main_node,
        critical=True
    )

    # Prepare URLs for general constraints (film/direction/lead/cannes)
    general_urls = data.general_sources or []
    # Prepare URLs for release date specific checks. Fall back to general URLs if none provided.
    release_urls = combine_urls(data.release_date_sources, data.general_sources)

    # 1) Directed by an actress making her feature-length directorial debut in 2025
    leaf_director_debut = evaluator.add_leaf(
        id="Directed_by_Actress_Debut_2025",
        desc="Film is directed by an actress making her feature-length directorial debut in 2025",
        parent=constraints_node,
        critical=True
    )
    claim_director_debut = (
        f"The film '{safe_str(data.film_title)}' is directed by an actress, {safe_str(data.director_name)}, "
        f"and it is her feature-length directorial debut in 2025."
    )
    await evaluator.verify(
        claim=claim_director_debut,
        node=leaf_director_debut,
        sources=general_urls,
        additional_instruction="Confirm both parts: (a) the director is an actress (has acting background) and (b) the film is her first feature-length directorial effort, debuting in 2025. Accept reasonable variants in phrasing."
    )

    # 2) Stars June Squibb in the lead role
    leaf_lead_june = evaluator.add_leaf(
        id="Stars_June_Squibb_Lead",
        desc="Film stars June Squibb in the lead role",
        parent=constraints_node,
        critical=True
    )
    claim_june_lead = (
        f"June Squibb is the lead actor in the film '{safe_str(data.film_title)}'."
    )
    await evaluator.verify(
        claim=claim_june_lead,
        node=leaf_lead_june,
        sources=general_urls,
        additional_instruction="Verify that June Squibb is credited specifically as the lead role, not merely a supporting role."
    )

    # 3) Presented at 2025 Cannes Film Festival in Un Certain Regard
    leaf_cannes_ucr = evaluator.add_leaf(
        id="Cannes_Un_Certain_Regard_2025",
        desc="Film appeared at the 2025 Cannes Film Festival in the Un Certain Regard selection",
        parent=constraints_node,
        critical=True
    )
    claim_cannes_ucr = (
        f"The film '{safe_str(data.film_title)}' was presented at the 2025 Cannes Film Festival in the Un Certain Regard selection."
    )
    await evaluator.verify(
        claim=claim_cannes_ucr,
        node=leaf_cannes_ucr,
        sources=general_urls,
        additional_instruction="Confirm the Cannes year is 2025 and the section is Un Certain Regard. Allow minor naming variants such as 'Un Certain Regard' vs 'Un Certain Regard section'."
    )

    # 4) Director previously appeared at Cannes multiple times as an actress
    leaf_director_prev_cannes = evaluator.add_leaf(
        id="Director_Previous_Cannes_Appearances",
        desc="Director has previously appeared at Cannes multiple times as an actress",
        parent=constraints_node,
        critical=True
    )
    claim_director_prev_cannes = (
        f"The director {safe_str(data.director_name)} has previously appeared at the Cannes Film Festival multiple times as an actress."
    )
    await evaluator.verify(
        claim=claim_director_prev_cannes,
        node=leaf_director_prev_cannes,
        sources=general_urls,
        additional_instruction="Check that the director has actress credits at Cannes on multiple occasions (i.e., more than once). Accept phrasing like 'multiple', 'several', or explicit counts greater than 1."
    )

    # 5) Theatrical release occurred in November 2025 (month-year check)
    leaf_release_month_year = evaluator.add_leaf(
        id="Release_in_November_2025",
        desc="The film had its theatrical release in November 2025",
        parent=constraints_node,
        critical=True
    )
    claim_release_month_year = (
        f"The film '{safe_str(data.film_title)}' had its theatrical release in November 2025."
    )
    await evaluator.verify(
        claim=claim_release_month_year,
        node=leaf_release_month_year,
        sources=release_urls,
        additional_instruction="Confirm that the theatrical release month is November and the year is 2025 (regional releases are acceptable if explicitly stated)."
    )

    # Leaf: Specific release date provided (month, day, year) in the answer (answer-level presence check)
    leaf_specific_date_provided = evaluator.add_leaf(
        id="Specific_Release_Date_Provided",
        desc="The answer provides a specific theatrical release date (month, day, and year) in November 2025 for the identified film",
        parent=film_main_node,
        critical=True
    )
    # Use simple verification focused on the answer content.
    specific_date_text = safe_str(data.release_date_str)
    claim_specific_date = (
        f"The answer explicitly provides a specific theatrical release date with month, day, and year, "
        f"and that date is in November 2025: '{specific_date_text}'."
    )
    await evaluator.verify(
        claim=claim_specific_date,
        node=leaf_specific_date_provided,
        additional_instruction="Verify based on the provided answer text that a concrete date is given and that it falls in November 2025."
    )

    # Leaf: At least one valid reference URL confirms the exact release date (URL-grounded verification)
    leaf_valid_refs = evaluator.add_leaf(
        id="Valid_Reference_URLs_Provided",
        desc="The answer includes at least one valid reference URL that confirms the theatrical release date of the identified film",
        parent=film_main_node,
        critical=True
    )
    claim_confirm_exact_date = (
        f"The film '{safe_str(data.film_title)}' had its theatrical release on {specific_date_text}."
    )
    # Try verifying by release-date-specific URLs first, falling back to general URLs if necessary.
    verify_urls = release_urls
    await evaluator.verify(
        claim=claim_confirm_exact_date,
        node=leaf_valid_refs,
        sources=verify_urls,
        additional_instruction="Confirm that the provided URL(s) state or clearly support the exact theatrical release date (month, day, year). If multiple regions are listed, a date matching the cited one is acceptable."
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
    Evaluate the answer for the film release date task using the obj_task_eval framework.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level parallel aggregation
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

    # Extract structured film info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_film_info(),
        template_class=FilmInfoExtraction,
        extraction_name="film_info_extraction"
    )

    # Add custom info snapshot to the summary (for debugging/traceability)
    evaluator.add_custom_info(
        info={
            "film_title": extracted.film_title,
            "director_name": extracted.director_name,
            "lead_actor": extracted.lead_actor,
            "cannes_year": extracted.cannes_year,
            "cannes_section": extracted.cannes_section,
            "release_date_str": extracted.release_date_str,
            "release_month": extracted.release_month,
            "release_day": extracted.release_day,
            "release_year": extracted.release_year,
            "director_debut_year": extracted.director_debut_year,
            "director_is_actress": extracted.director_is_actress,
            "director_cannes_appearances_as_actress": extracted.director_cannes_appearances_as_actress,
            "general_sources_count": len(extracted.general_sources or []),
            "release_date_sources_count": len(extracted.release_date_sources or [])
        },
        info_type="extraction_snapshot",
        info_name="film_info_snapshot"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted)

    # Return standardized summary
    return evaluator.get_summary()