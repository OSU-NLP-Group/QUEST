import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "conf_logistics_2025"
TASK_DESCRIPTION = (
    "I'm planning to attend several major computer science and AI conferences in 2025 for research purposes. "
    "I need detailed logistical information for the following four categories of conferences: "
    "(1) One major machine learning conference held in North America, "
    "(2) One major computer vision conference, "
    "(3) One major natural language processing (NLP) conference, and "
    "(4) One major general artificial intelligence conference held in Asia. "
    "For each of these four conferences, please provide the following information: the official conference name "
    "(full name, not abbreviation only), the exact start date (month, day, year), the exact end date (month, day, year), "
    "the host city, and the specific venue name (the actual facility/convention center name, not just the city). "
    "All conferences must be recognized top-tier conferences in their respective fields, scheduled for 2025, "
    "and the information must be verifiable from official conference websites."
)


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class ConferenceItem(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None  # optional, for clarity if answer labels it
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    city: Optional[str] = None
    venue: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)


class ConferencesExtraction(BaseModel):
    ml: Optional[ConferenceItem] = None
    cv: Optional[ConferenceItem] = None
    nlp: Optional[ConferenceItem] = None
    ai: Optional[ConferenceItem] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
Extract exactly one conference item for each of the following categories from the answer, preserving the information as stated in the answer text:
- ml: A major machine learning conference (must be held in North America).
- cv: A major computer vision conference.
- nlp: A major natural language processing conference.
- ai: A major general artificial intelligence conference (must be held in Asia).

For each category (ml, cv, nlp, ai), extract a JSON object with these fields:
- name: The official conference name as provided in the answer (full if present; do not invent).
- category: If the answer labels it, capture that label (e.g., "ML", "Computer Vision"); otherwise null.
- start_date: The start date string as provided (e.g., "December 7, 2025" or "2025-12-07"). Do not reformat.
- end_date: The end date string as provided.
- city: The host city as provided (e.g., "New Orleans, Louisiana").
- venue: The specific facility or convention center name (not just the city), as provided (e.g., "Ernest N. Morial Convention Center").
- official_urls: An array of official conference website URLs cited in the answer that can verify dates and location (include only URLs explicitly present in the answer; no inferred or third‑party links).

Rules:
1) Extract only what is explicitly present in the answer. If a field is not present, set it to null (or empty list for official_urls).
2) If the answer mentions multiple conferences per category, choose the one that most clearly fits the stated category and constraints, but do not add or change data that is not in the answer.
3) Only include URLs that are explicitly listed in the answer; do not create or infer any URLs. Prefer official websites (e.g., conference series official domain or the conference's own domain) if present in the answer.
4) If the answer does not provide any usable conference for a category, set that category to null.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(value: Optional[str]) -> str:
    return value or ""


def _urls_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    # Filter out obviously malformed strings (very light heuristic)
    filtered = [u for u in urls if isinstance(u, str) and len(u.strip()) > 0]
    return filtered if filtered else None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def add_common_conference_checks(
    evaluator: Evaluator,
    parent_node,
    prefix: str,
    item: Optional[ConferenceItem],
    category_label_for_check: str,
    require_region: Optional[str] = None,  # "North America" or "Asia" or None
) -> None:
    """
    Add leaf nodes under parent_node following the rubric for a single conference category.
    All leaves are critical, matching rubric. Uses official URLs when provided for evidence-backed verification.
    """
    # Normalize fields
    name = _safe(item.name if item else None)
    start_date = _safe(item.start_date if item else None)
    end_date = _safe(item.end_date if item else None)
    city = _safe(item.city if item else None)
    venue = _safe(item.venue if item else None)
    urls = _urls_or_none(item.official_urls if item else None)

    # 1) Official full name provided (not abbreviation-only)
    node_fullname = evaluator.add_leaf(
        id=f"{prefix}_official_full_name_provided",
        desc="Provides the official conference name in full (not abbreviation-only).",
        parent=parent_node,
        critical=True,
    )
    claim_fullname = (
        f"The provided conference name '{name}' is the official full name of the conference (not an abbreviation only). "
        f"If the provided name is only an abbreviation like 'NeurIPS', 'ICML', 'CVPR', 'ACL', 'AAAI', or 'IJCAI' without "
        f"the expanded official name, then this statement should be considered false."
    )
    await evaluator.verify(
        claim=claim_fullname,
        node=node_fullname,
        sources=urls,
        additional_instruction=(
            "Rely on the official conference website(s) to determine the official full name. "
            "Minor formatting/casing differences are acceptable. "
            "If the answer's provided name is only an abbreviation with no expanded name, judge as Incorrect."
        ),
    )

    # 2) Category check: ML / CV / NLP / General AI
    node_cat = evaluator.add_leaf(
        id=f"{prefix}_is_{'machine_learning_conference' if prefix=='ml' else ('computer_vision_conference' if prefix=='cv' else ('nlp_conference' if prefix=='nlp' else 'general_ai_conference'))}",
        desc=(
            "Conference is a machine learning conference (fits the ML category)." if prefix == "ml" else
            ("Conference is a computer vision conference (fits the CV category)." if prefix == "cv" else
             ("Conference is a natural language processing conference (fits the NLP category)." if prefix == "nlp" else
              "Conference fits the 'general artificial intelligence' category (not a different specialized category unless still clearly general AI as requested)."))
        ),
        parent=parent_node,
        critical=True,
    )
    cat_claim = (
        f"The conference '{name}' belongs to the field/category: {category_label_for_check}."
    )
    await evaluator.verify(
        claim=cat_claim,
        node=node_cat,
        sources=urls,
        additional_instruction=(
            "Use the provided official conference webpage(s) to confirm the conference's domain. "
            "Allow reasonable synonyms (e.g., 'computer vision' vs 'vision and pattern recognition') "
            "as long as it clearly fits the requested category."
        ),
    )

    # 3) Top-tier check
    node_top_tier = evaluator.add_leaf(
        id=f"{prefix}_is_top_tier",
        desc="Conference is recognized as top-tier in its field (as required by the constraints).",
        parent=parent_node,
        critical=True,
    )
    claim_top_tier = (
        f"The conference '{name}' is widely recognized as a top-tier conference in {category_label_for_check}."
    )
    await evaluator.verify(
        claim=claim_top_tier,
        node=node_top_tier,
        sources=None,
        additional_instruction=(
            "For this check, you MAY use general field knowledge and community consensus "
            "(e.g., NeurIPS/ICML for ML; CVPR/ICCV/ECCV for CV; ACL/EMNLP for NLP; AAAI/IJCAI for general AI). "
            "If the conference is not broadly regarded as top-tier, mark as Incorrect."
        ),
    )

    # 4) Regional constraint (only for ML in North America and AI in Asia)
    if require_region == "North America":
        node_region = evaluator.add_leaf(
            id=f"{prefix}_held_in_north_america",
            desc="Conference location is in North America.",
            parent=parent_node,
            critical=True,
        )
        claim_region = (
            f"The host city '{city}' is in North America; thus the conference is held in North America."
        )
        await evaluator.verify(
            claim=claim_region,
            node=node_region,
            sources=urls,
            additional_instruction=(
                "Use the official site to confirm the city. Then judge if that city is located within North America "
                "(United States, Canada, Mexico, or broadly accepted North American territories)."
            ),
        )
    elif require_region == "Asia":
        node_region = evaluator.add_leaf(
            id=f"{prefix}_held_in_asia",
            desc="Conference location is in Asia.",
            parent=parent_node,
            critical=True,
        )
        claim_region = (
            f"The host city '{city}' is in Asia; thus the conference is held in Asia."
        )
        await evaluator.verify(
            claim=claim_region,
            node=node_region,
            sources=urls,
            additional_instruction=(
                "Use the official site to confirm the city. Then judge if that city is located within Asia "
                "(broadly accepted geographic definition)."
            ),
        )

    # 5) Start date completeness + 2025
    node_start = evaluator.add_leaf(
        id=f"{prefix}_start_date_complete_and_2025",
        desc="Provides the exact start date including month, day, and year, and the year is 2025.",
        parent=parent_node,
        critical=True,
    )
    claim_start = (
        f"The provided start date '{start_date}' explicitly includes a month, a day number, and the year 2025; "
        f"and the year is indeed 2025."
    )
    await evaluator.verify(
        claim=claim_start,
        node=node_start,
        sources=None,
        additional_instruction=(
            "Only pass if the string clearly includes month (name or numeric), day (numeric), and the year '2025'. "
            "Minor format variations are acceptable (e.g., '2025-07-12', 'July 12, 2025')."
        ),
    )

    # 6) End date completeness + 2025
    node_end = evaluator.add_leaf(
        id=f"{prefix}_end_date_complete_and_2025",
        desc="Provides the exact end date including month, day, and year, and the year is 2025.",
        parent=parent_node,
        critical=True,
    )
    claim_end = (
        f"The provided end date '{end_date}' explicitly includes a month, a day number, and the year 2025; "
        f"and the year is indeed 2025."
    )
    await evaluator.verify(
        claim=claim_end,
        node=node_end,
        sources=None,
        additional_instruction=(
            "Only pass if the string clearly includes month (name or numeric), day (numeric), and the year '2025'. "
            "Minor format variations are acceptable (e.g., '2025-07-16', 'July 16, 2025')."
        ),
    )

    # 7) Host city provided (existence)
    evaluator.add_custom_node(
        result=(len(city.strip()) > 0),
        id=f"{prefix}_host_city_provided",
        desc="Provides the host city.",
        parent=parent_node,
        critical=True,
    )

    # 8) Specific venue facility provided (existence and specificity beyond city)
    city_lower = city.strip().lower()
    venue_lower = venue.strip().lower()
    venue_specific = (len(venue.strip()) > 0) and (venue_lower != city_lower)
    evaluator.add_custom_node(
        result=venue_specific,
        id=f"{prefix}_specific_venue_facility_provided",
        desc="Provides the specific venue facility name (not just the city).",
        parent=parent_node,
        critical=True,
    )

    # 9) Verifiable from official website (dates and venue/city can be verified)
    node_verify = evaluator.add_leaf(
        id=f"{prefix}_verifiable_from_official_website",
        desc="Cites/links an official conference website page(s) sufficient to verify the provided dates and venue/city.",
        parent=parent_node,
        critical=True,
    )
    claim_verify = (
        f"On the official conference website page(s), the conference '{name}' is scheduled from '{start_date}' to "
        f"'{end_date}' in '{city}', and the venue/facility is '{venue}'."
    )
    await evaluator.verify(
        claim=claim_verify,
        node=node_verify,
        sources=urls,
        additional_instruction=(
            "Only pass if at least one provided URL is an official conference website page (e.g., conference's own domain "
            "or the official series/organizer domain) that directly confirms both the dates and the city/venue. "
            "If URLs are missing, non-official, or do not clearly confirm the details, mark as not supported."
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
    Evaluate an answer for the 2025 conference logistics task.
    """
    # Initialize evaluator with a parallel root as per rubric
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

    # Extract structured conference info
    extracted = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferencesExtraction,
        extraction_name="conferences_extraction",
    )

    # Build top-level rubric node (parallel)
    top_node = evaluator.add_parallel(
        id="conference_information_compilation",
        desc="Evaluate whether the response provides required logistical details for four 2025 top-tier conferences matching the specified categories and location constraints.",
        parent=root,
        critical=False,
    )

    # ML conference (must be North America)
    ml_parent = evaluator.add_parallel(
        id="ml_conference",
        desc="Machine learning conference item (must be held in North America) with all required fields and official-site verifiability.",
        parent=top_node,
        critical=False,
    )
    await add_common_conference_checks(
        evaluator=evaluator,
        parent_node=ml_parent,
        prefix="ml",
        item=extracted.ml,
        category_label_for_check="machine learning",
        require_region="North America",
    )

    # CV conference
    cv_parent = evaluator.add_parallel(
        id="cv_conference",
        desc="Computer vision conference item with all required fields and official-site verifiability.",
        parent=top_node,
        critical=False,
    )
    await add_common_conference_checks(
        evaluator=evaluator,
        parent_node=cv_parent,
        prefix="cv",
        item=extracted.cv,
        category_label_for_check="computer vision",
        require_region=None,
    )

    # NLP conference
    nlp_parent = evaluator.add_parallel(
        id="nlp_conference",
        desc="Natural language processing conference item with all required fields and official-site verifiability.",
        parent=top_node,
        critical=False,
    )
    await add_common_conference_checks(
        evaluator=evaluator,
        parent_node=nlp_parent,
        prefix="nlp",
        item=extracted.nlp,
        category_label_for_check="natural language processing",
        require_region=None,
    )

    # General AI conference (must be Asia)
    ai_parent = evaluator.add_parallel(
        id="ai_conference",
        desc="General AI conference item (must be held in Asia) with all required fields and official-site verifiability.",
        parent=top_node,
        critical=False,
    )
    await add_common_conference_checks(
        evaluator=evaluator,
        parent_node=ai_parent,
        prefix="ai",
        item=extracted.ai,
        category_label_for_check="general artificial intelligence",
        require_region="Asia",
    )

    # Return evaluation summary
    return evaluator.get_summary()