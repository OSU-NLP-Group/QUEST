import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ai_ml_conferences_2026"
TASK_DESCRIPTION = """
I am a researcher planning my conference attendance for 2026 and need to identify three major international academic conferences in the fields of artificial intelligence, machine learning, computer vision, or natural language processing that are scheduled to take place in 2026. For each of the three conferences, provide the following information: (1) The full conference name, (2) The dates when the main conference will be held (start date and end date), (3) The city where the conference will take place, (4) The country where the conference will take place, (5) The name of the venue (e.g., convention center, expo hall), and (6) The official conference website URL for the 2026 event. The three conferences you select should be well-established, major conferences in the field (not workshops, symposiums, or small regional events).
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConferenceItem(BaseModel):
    name: Optional[str] = None
    start_date: Optional[str] = None  # Keep as string; answers may use ranges or formats
    end_date: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    venue: Optional[str] = None
    official_url: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)


class ConferencesExtraction(BaseModel):
    conferences: List[ConferenceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
    From the provided answer, extract up to THREE distinct, well-established international academic conferences in AI, machine learning, computer vision, or natural language processing that are scheduled for 2026.
    
    For each conference, extract the following fields:
    - name: The full conference name (e.g., "Conference on Neural Information Processing Systems (NeurIPS)").
    - start_date: The start date of the MAIN CONFERENCE in 2026 (exclude workshop/tutorial-only dates). Keep the original format as written in the answer (e.g., "June 15, 2026" or "June 15").
    - end_date: The end date of the MAIN CONFERENCE in 2026. Keep the original format.
    - city: The primary host city of the MAIN CONFERENCE in 2026.
    - country: The country of the MAIN CONFERENCE in 2026.
    - venue: The venue name (e.g., a convention center or expo hall), if provided.
    - official_url: The official website URL for the 2026 event (ideally a 2026 edition page or main site that clearly hosts the 2026 details). If missing, set to null.
    - extra_urls: Any additional URLs cited in the answer that support the 2026 details (e.g., Wikipedia, press releases, blog posts, archived pages). If none, return an empty list.
    
    Rules:
    - Extract only what is explicitly present in the answer text.
    - If more than three conferences are mentioned, keep only the FIRST three.
    - If fewer than three conferences are mentioned, return as many as present.
    - If any field is missing for a conference, set it to null (or [] for extra_urls).
    - For URLs missing a protocol, prepend "http://".
    - The dates should correspond to the MAIN CONFERENCE days (exclude workshop-only dates if the answer distinguishes them).
    
    Return a JSON object with a single key "conferences" that is an array of conference objects with the fields above.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _normalize_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    s = url.strip()
    if not s:
        return None
    if not (s.startswith("http://") or s.startswith("https://")):
        s = "http://" + s
    return s


def collect_sources(conf: ConferenceItem) -> List[str]:
    urls: List[str] = []
    off = _normalize_url(conf.official_url)
    if off:
        urls.append(off)
    for u in conf.extra_urls or []:
        nu = _normalize_url(u)
        if nu and nu not in urls:
            urls.append(nu)
    return urls


# --------------------------------------------------------------------------- #
# Verification logic per conference                                           #
# --------------------------------------------------------------------------- #
async def verify_conference(evaluator: Evaluator, parent_node, conf: ConferenceItem, index: int) -> None:
    # Parent node for this conference (parallel, non-critical)
    conf_node = evaluator.add_parallel(
        id=f"conference_{index}",
        desc=f"Complete and accurate information about the {'first' if index==0 else ('second' if index==1 else 'third')} conference",
        parent=parent_node,
        critical=False
    )

    # Prepare sources
    all_sources = collect_sources(conf)
    official_only = _normalize_url(conf.official_url)

    # Leaf: name is a major, established international conference in target fields
    name_node = evaluator.add_leaf(
        id=f"conference_{index}_name",
        desc="The conference is a major, established international academic conference in AI/ML/CV/NLP",
        parent=conf_node,
        critical=True
    )
    conf_name = conf.name or "the conference"
    name_claim = (
        f"The conference named '{conf_name}' is a major, established international academic conference in "
        f"artificial intelligence, machine learning, computer vision, or natural language processing."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=all_sources if all_sources else None,
        additional_instruction=(
            "Rely on the provided webpage(s). Accept if the pages clearly indicate the conference is a flagship/major/"
            "premier/long-standing international academic conference in the target fields. Allow well-known "
            "abbreviations (e.g., NeurIPS, ICML, ICLR, CVPR, ACL, EMNLP, ECCV, NAACL). "
            "If the name is missing from the answer or the sources are irrelevant/inaccessible, mark as Incorrect."
        )
    )

    # Leaf: dates accurate for 2026 main conference
    dates_node = evaluator.add_leaf(
        id=f"conference_{index}_dates",
        desc="The dates provided match the official 2026 main-conference schedule",
        parent=conf_node,
        critical=True
    )
    start_str = conf.start_date or "[missing start date]"
    end_str = conf.end_date or "[missing end date]"
    dates_claim = (
        f"For the 2026 edition of {conf_name}, the MAIN CONFERENCE runs from {start_str} to {end_str}."
    )
    await evaluator.verify(
        claim=dates_claim,
        node=dates_node,
        sources=all_sources if all_sources else None,
        additional_instruction=(
            "Verify against the 2026 schedule on the official or authoritative page. "
            "Focus on MAIN CONFERENCE dates (exclude workshops/tutorials if distinct). "
            "If either date is missing in the answer or unsupported by sources, mark as Incorrect."
        )
    )

    # Leaf: city correct
    city_node = evaluator.add_leaf(
        id=f"conference_{index}_city",
        desc="The city for the 2026 main conference is correctly identified",
        parent=conf_node,
        critical=True
    )
    city_val = conf.city or "[missing city]"
    city_claim = f"For the 2026 edition of {conf_name}, the MAIN CONFERENCE city is {city_val}."
    await evaluator.verify(
        claim=city_claim,
        node=city_node,
        sources=all_sources if all_sources else None,
        additional_instruction=(
            "Confirm the primary in-person host city for the 2026 main conference (not just workshops). "
            "If the city is missing or unsupported by the sources, mark as Incorrect."
        )
    )

    # Leaf: country correct
    country_node = evaluator.add_leaf(
        id=f"conference_{index}_country",
        desc="The country for the 2026 main conference is correctly identified",
        parent=conf_node,
        critical=True
    )
    country_val = conf.country or "[missing country]"
    country_claim = f"For the 2026 edition of {conf_name}, the MAIN CONFERENCE country is {country_val}."
    await evaluator.verify(
        claim=country_claim,
        node=country_node,
        sources=all_sources if all_sources else None,
        additional_instruction=(
            "Confirm the country corresponding to the 2026 main conference host city. "
            "If missing or unsupported, mark as Incorrect."
        )
    )

    # Leaf: venue correct
    venue_node = evaluator.add_leaf(
        id=f"conference_{index}_venue",
        desc="The venue name for the 2026 main conference is correctly provided",
        parent=conf_node,
        critical=True
    )
    venue_val = conf.venue or "[missing venue]"
    venue_claim = f"For the 2026 edition of {conf_name}, the MAIN CONFERENCE venue is '{venue_val}'."
    await evaluator.verify(
        claim=venue_claim,
        node=venue_node,
        sources=all_sources if all_sources else None,
        additional_instruction=(
            "Verify the named physical venue (e.g., convention center/expo hall) for the 2026 main conference. "
            "If the answer omits the venue or the sources do not support it, mark as Incorrect."
        )
    )

    # Leaf: official 2026 website URL provided and valid
    url_node = evaluator.add_leaf(
        id=f"conference_{index}_url",
        desc="A valid official website URL for the conference's 2026 event is provided",
        parent=conf_node,
        critical=True
    )
    url_claim = (
        f"This webpage is the official website or main 2026 event landing page for the {conf_name} conference."
    )
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=official_only if official_only else None,
        additional_instruction=(
            "Pass if the provided URL leads to the official conference website (or its 2026 edition page) and the page "
            "clearly corresponds to the 2026 event. If no URL is provided in the answer, or the page is clearly a "
            "different year or a non-official site, mark as Incorrect."
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator with root as parallel (three conferences independent)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Complete and accurate information about three major AI/ML conferences scheduled for 2026",
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

    # Ensure exactly 3 items (pad with empty if needed), keep first 3 only
    conferences: List[ConferenceItem] = list(extracted.conferences[:3])
    while len(conferences) < 3:
        conferences.append(ConferenceItem())

    evaluator.add_custom_info(
        info={
            "extracted_count": len(extracted.conferences),
            "used_count": 3,
            "note": "Only the first three conferences from the answer are evaluated. Missing fields will likely fail verification."
        },
        info_type="extraction_meta",
        info_name="extraction_statistics"
    )

    # Build three parallel conference subtrees
    tasks = []
    for idx in range(3):
        tasks.append(verify_conference(evaluator, root, conferences[idx], idx))
    await asyncio.gather(*tasks)

    # Return summary
    return evaluator.get_summary()