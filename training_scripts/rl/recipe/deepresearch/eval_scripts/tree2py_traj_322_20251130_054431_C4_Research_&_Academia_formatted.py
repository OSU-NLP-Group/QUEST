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
TASK_ID = "perseverance_nature_cheyava_falls"
TASK_DESCRIPTION = """
In 2024, NASA's Perseverance rover discovered a rock with distinctive "leopard spots" features on Mars that was later analyzed and reported in a major scientific publication in 2025. Identify the Nature journal publication that reported findings on this rock and provide the following information: (1) The exact publication date of the article, (2) The lead (first) author's full name, (3) The nickname given to the rock that was studied, (4) The name of the core sample that was collected from this rock, (5) The name of the geological formation where the rock was found, (6) The month and year when the sample was collected, (7) The specific location or valley name where the rock was discovered.
"""


# --------------------------------------------------------------------------- #
# Ground truth context (for reporting only; verification relies on sources)   #
# --------------------------------------------------------------------------- #
GROUND_TRUTH = {
    "journal": "Nature",
    "publication_date": "September 10, 2025",
    "lead_author": "Joel A. Hurowitz",
    "rock_name": "Cheyava Falls",
    "sample_name": "Sapphire Canyon",
    "formation_name": "Bright Angel formation",
    "collection_timeframe": "July 2024",
    "location": "Neretva Vallis (western edge of Jezero Crater)"
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NaturePublicationExtraction(BaseModel):
    # Identification
    nature_urls: List[str] = Field(default_factory=list)
    all_urls: List[str] = Field(default_factory=list)
    article_title: Optional[str] = None
    doi: Optional[str] = None
    journal_name: Optional[str] = None

    # Required fields to verify
    publication_date: Optional[str] = None
    lead_author: Optional[str] = None
    rock_name: Optional[str] = None
    sample_name: Optional[str] = None
    formation_name: Optional[str] = None
    collection_timeframe: Optional[str] = None  # e.g., "July 2024"
    location_name: Optional[str] = None         # e.g., "Neretva Vallis" or "Jezero Crater western edge"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nature_publication_info() -> str:
    return """
    From the answer, extract details about the Nature journal publication that reported findings on Perseverance's "leopard spots" rock. Extract exactly what the answer states; do not infer or add information.

    Required fields:
    - nature_urls: array of all URLs in the answer that clearly point to Nature's website (e.g., https://www.nature.com/... or other Nature-branded article pages explicitly present in the answer). Include only URLs explicitly present in the answer. Deduplicate.
    - all_urls: array of all URLs (any domain) explicitly present in the answer. Deduplicate.
    - article_title: the full article title if provided in the answer; else null.
    - doi: the DOI string if explicitly given (e.g., "10.1038/s41586-025-XXXXX"). If a full DOI URL is given (e.g., https://doi.org/...), extract the DOI string (without the https://doi.org/ prefix). If none, return null.
    - journal_name: the journal name as stated in the answer (e.g., "Nature", "Nature Communications", etc.); else null.
    - publication_date: the exact publication date as written in the answer (keep the format the answer uses); else null.
    - lead_author: the first (lead) author's full name as written in the answer; else null.
    - rock_name: the rock nickname/name as written in the answer; else null.
    - sample_name: the core sample name as written in the answer; else null.
    - formation_name: the geological formation name as written in the answer; else null.
    - collection_timeframe: the month and year when the sample was collected, as written in the answer; else null.
    - location_name: the specific location/valley name where the rock was discovered, as written in the answer (e.g., "Neretva Vallis" or "Jezero Crater western edge"); else null.

    URL handling rules:
    - Only extract URLs that are explicitly present. Do not invent URLs.
    - Include full URLs. If a URL is missing the protocol, prepend "http://".
    - Return arrays for nature_urls and all_urls; return empty arrays if no URLs are present.

    If any field is missing from the answer, set it to null (or empty array for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def select_nature_sources(info: NaturePublicationExtraction) -> Optional[List[str]]:
    """
    Choose Nature-like URLs from the extraction to use as verification sources.
    Priority:
      1) info.nature_urls
      2) Filtered nature-like URLs from info.all_urls (nature.com, www.nature.com, or doi.org/10.1038)
    Returns None if no Nature-like URLs are present.
    """
    if info.nature_urls:
        # Deduplicate while preserving order
        seen = set()
        result = []
        for u in info.nature_urls:
            if u and u not in seen:
                seen.add(u)
                result.append(u)
        return result if result else None

    # Try to filter Nature-like URLs from all_urls
    candidates = []
    for u in info.all_urls or []:
        if not u:
            continue
        low = u.lower()
        if ("nature.com" in low) or ("doi.org/10.1038" in low):
            candidates.append(u)

    if not candidates:
        return None

    # Deduplicate
    seen2 = set()
    filtered = []
    for u in candidates:
        if u not in seen2:
            seen2.add(u)
            filtered.append(u)

    return filtered if filtered else None


def _has_text(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_nature_publication(evaluator: Evaluator, parent_node, info: NaturePublicationExtraction) -> None:
    """
    Build the rubric tree and perform verifications for the Nature publication.
    """
    # Group node representing the Nature publication verification (critical)
    group_node = evaluator.add_parallel(
        id="nature_publication_cheyava_falls",
        desc="Identify and provide details about the Nature publication reporting findings on the specified Perseverance rover rock.",
        parent=parent_node,
        critical=True
    )

    nature_sources = select_nature_sources(info)

    # 1) Publication identifier (presence: title and/or DOI and/or Nature URL)
    evaluator.add_custom_node(
        result=(_has_text(info.article_title) or _has_text(info.doi) or (nature_sources is not None and len(nature_sources) > 0)),
        id="publication_identifier",
        desc="The answer must uniquely identify the Nature publication (e.g., article title and/or DOI and/or a Nature URL).",
        parent=group_node,
        critical=True
    )

    # 2) Journal name presence + correctness via Nature page
    evaluator.add_custom_node(
        result=_has_text(info.journal_name),
        id="journal_name_provided",
        desc="The answer provides the journal name.",
        parent=group_node,
        critical=True
    )
    node_journal = evaluator.add_leaf(
        id="journal_name",
        desc="The publication must identify Nature as the journal.",
        parent=group_node,
        critical=True
    )
    claim_journal = f"The journal for the article is '{info.journal_name or ''}'."
    await evaluator.verify(
        claim=claim_journal,
        node=node_journal,
        sources=nature_sources,
        additional_instruction="Verify this ONLY against the Nature article webpage. Accept only 'Nature' (the flagship weekly journal). "
                               "If the page indicates 'Nature Communications', 'Nature Geoscience', 'Nature Astronomy', etc., "
                               "this should be judged as incorrect."
    )

    # 3) Publication date presence + correctness
    evaluator.add_custom_node(
        result=_has_text(info.publication_date),
        id="publication_date_provided",
        desc="The answer provides the publication date.",
        parent=group_node,
        critical=True
    )
    node_pubdate = evaluator.add_leaf(
        id="publication_date",
        desc="The publication must identify September 10, 2025 as the publication date.",
        parent=group_node,
        critical=True
    )
    claim_pubdate = f"The publication date of the article is '{info.publication_date or ''}'."
    await evaluator.verify(
        claim=claim_pubdate,
        node=node_pubdate,
        sources=nature_sources,
        additional_instruction="Check the article page for the exact publication date. The correct date is 10 September 2025 (equivalently, September 10, 2025 or ISO 2025-09-10). "
                               "Minor formatting differences are acceptable, but the value must correspond to that date."
    )

    # 4) Lead (first) author presence + correctness
    evaluator.add_custom_node(
        result=_has_text(info.lead_author),
        id="lead_author_provided",
        desc="The answer provides the lead author's name.",
        parent=group_node,
        critical=True
    )
    node_lead = evaluator.add_leaf(
        id="lead_author",
        desc="The publication must identify Joel A. Hurowitz as the lead or first author.",
        parent=group_node,
        critical=True
    )
    claim_lead = f"The first (lead) author listed for the article is '{info.lead_author or ''}'."
    await evaluator.verify(
        claim=claim_lead,
        node=node_lead,
        sources=nature_sources,
        additional_instruction="Verify the first author shown on the article page. Accept minor variants in name formatting (e.g., middle initial). "
                               "The correct lead author should be Joel A. Hurowitz."
    )

    # 5) Rock nickname/name presence + correctness
    evaluator.add_custom_node(
        result=_has_text(info.rock_name),
        id="rock_name_provided",
        desc="The answer provides the rock nickname or name.",
        parent=group_node,
        critical=True
    )
    node_rock = evaluator.add_leaf(
        id="rock_name",
        desc="The answer must correctly identify Cheyava Falls as the rock nickname/name.",
        parent=group_node,
        critical=True
    )
    claim_rock = f"The article refers to the studied rock as '{info.rock_name or ''}'."
    await evaluator.verify(
        claim=claim_rock,
        node=node_rock,
        sources=nature_sources,
        additional_instruction="Verify that the article identifies the rock by the nickname 'Cheyava Falls'. "
                               "Allow for minor casing or punctuation differences."
    )

    # 6) Core sample name presence + correctness
    evaluator.add_custom_node(
        result=_has_text(info.sample_name),
        id="sample_name_provided",
        desc="The answer provides the core sample name.",
        parent=group_node,
        critical=True
    )
    node_sample = evaluator.add_leaf(
        id="sample_name",
        desc="The answer must correctly identify Sapphire Canyon as the core sample name collected from the rock.",
        parent=group_node,
        critical=True
    )
    claim_sample = f"The core sample collected from this rock is named '{info.sample_name or ''}'."
    await evaluator.verify(
        claim=claim_sample,
        node=node_sample,
        sources=nature_sources,
        additional_instruction="Verify that the article names the core sample 'Sapphire Canyon'. "
                               "Minor casing or punctuation differences are acceptable."
    )

    # 7) Geological formation presence + correctness
    evaluator.add_custom_node(
        result=_has_text(info.formation_name),
        id="formation_name_provided",
        desc="The answer provides the geological formation name.",
        parent=group_node,
        critical=True
    )
    node_form = evaluator.add_leaf(
        id="formation_name",
        desc="The answer must correctly identify Bright Angel or Bright Angel formation as the geological formation.",
        parent=group_node,
        critical=True
    )
    claim_form = f"The geological formation reported for the rock is '{info.formation_name or ''}'."
    await evaluator.verify(
        claim=claim_form,
        node=node_form,
        sources=nature_sources,
        additional_instruction="Verify that the article identifies the geological formation as 'Bright Angel' (often phrased as 'Bright Angel formation'). "
                               "Minor wording differences are acceptable if they clearly refer to 'Bright Angel'."
    )

    # 8) Collection timeframe presence + correctness
    evaluator.add_custom_node(
        result=_has_text(info.collection_timeframe),
        id="collection_timeframe_provided",
        desc="The answer provides the month and year when the sample was collected.",
        parent=group_node,
        critical=True
    )
    node_collect = evaluator.add_leaf(
        id="collection_timeframe",
        desc="The answer must correctly identify July 2024 as when the sample was collected.",
        parent=group_node,
        critical=True
    )
    claim_collect = f"The sample was collected in '{info.collection_timeframe or ''}'."
    await evaluator.verify(
        claim=claim_collect,
        node=node_collect,
        sources=nature_sources,
        additional_instruction="Verify the collection timeframe on the article page. The correct timeframe should be July 2024. "
                               "Allow equivalent phrasings like 'in July 2024'."
    )

    # 9) Discovery location presence + correctness
    evaluator.add_custom_node(
        result=_has_text(info.location_name),
        id="location_provided",
        desc="The answer provides the specific location or valley name.",
        parent=group_node,
        critical=True
    )
    node_loc = evaluator.add_leaf(
        id="location",
        desc="The answer must correctly identify Neretva Vallis or Jezero Crater western edge as the discovery location.",
        parent=group_node,
        critical=True
    )
    claim_loc = f"The article states the rock was discovered in or near '{info.location_name or ''}'."
    await evaluator.verify(
        claim=claim_loc,
        node=node_loc,
        sources=nature_sources,
        additional_instruction="Verify the location on the article page. Accept 'Neretva Vallis' or equivalent phrasing indicating the western edge of Jezero Crater. "
                               "Minor wording differences are acceptable if they unambiguously refer to these locations."
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
    Evaluate an answer for the Nature publication (Perseverance 'leopard spots' rock) task.
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

    # Extraction
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_nature_publication_info(),
        template_class=NaturePublicationExtraction,
        extraction_name="nature_publication_extraction",
    )

    # Add GT info for report (not used for scoring)
    evaluator.add_ground_truth({
        "expected": GROUND_TRUTH,
        "note": "Verification is performed against the Nature article webpage(s) extracted from the answer when available."
    })

    # Verification
    await verify_nature_publication(evaluator, root, extracted_info)

    # Return the evaluation summary
    return evaluator.get_summary()