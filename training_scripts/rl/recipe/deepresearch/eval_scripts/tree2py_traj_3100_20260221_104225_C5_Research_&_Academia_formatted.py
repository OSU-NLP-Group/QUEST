import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs_conf_constraints_2022_2024"
TASK_DESCRIPTION = """Identify three computer science conferences that meet all of the following criteria:

1. The conference must be ranked as A* (A-star) or A in the CORE Computer Science Conference Rankings, or be listed in the top 30 of Research.com's Computer Science Conference Rankings for 2024 or later.

2. The conference must have been held in the United States at least once between 2022 and 2024 (inclusive).

3. The conference's most recently reported acceptance rate (from any year between 2020 and 2024) for full research papers must be between 15% and 30% (inclusive).

4. The conference proceedings must be indexed in either the ACM Digital Library or IEEE Xplore Digital Library.

5. The conference must be primarily focused on one of these computer science subfields: Human-Computer Interaction, Artificial Intelligence, Computer Vision, Natural Language Processing, or Machine Learning.

6. For the most recent occurrence of the conference held between 2022 and 2024, the total number of accepted full research papers must be at least 50.

For each conference, provide: the full conference name, the acronym/abbreviation, the specific year and US city/state where it was held (between 2022-2024), the reported acceptance rate with source year, the number of accepted papers from the most recent edition (2022-2024), and reference URLs supporting each piece of information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConferenceItem(BaseModel):
    # Identification
    full_name: Optional[str] = None
    acronym: Optional[str] = None
    name_acronym_urls: List[str] = Field(default_factory=list)

    # US edition 2022–2024 (a particular occurrence)
    us_year: Optional[str] = None
    us_city: Optional[str] = None
    us_state: Optional[str] = None
    us_edition_urls: List[str] = Field(default_factory=list)

    # Ranking: either CORE (A*/A) or Research.com top-30 for 2024+
    core_rank: Optional[str] = None          # e.g., "A*", "A"
    research_rank: Optional[str] = None      # e.g., "12"
    research_rank_year: Optional[str] = None # e.g., "2024", "2025"
    ranking_urls: List[str] = Field(default_factory=list)

    # Acceptance rate (most recently reported between 2020–2024, full research papers)
    acceptance_rate: Optional[str] = None    # e.g., "23%", "0.23"
    acceptance_year: Optional[str] = None    # e.g., "2023"
    acceptance_urls: List[str] = Field(default_factory=list)

    # Proceedings indexing
    indexing_platform: Optional[str] = None  # e.g., "ACM Digital Library", "IEEE Xplore", "Both"
    indexing_urls: List[str] = Field(default_factory=list)

    # Subfield focus
    subfield: Optional[str] = None           # One of HCI, AI, Computer Vision, NLP, Machine Learning
    subfield_urls: List[str] = Field(default_factory=list)

    # Accepted papers (most recent 2022–2024 occurrence)
    recent_year: Optional[str] = None
    accepted_papers: Optional[str] = None
    papers_urls: List[str] = Field(default_factory=list)


class ConferencesExtraction(BaseModel):
    conferences: List[ConferenceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
    You must extract up to ALL conferences mentioned in the answer (we will later select the first 3). For each conference, extract the following fields exactly as provided in the answer text. If a field is missing in the answer, set it to null (or an empty array for URL lists). Do not fabricate or infer any information beyond the answer.

    Return a JSON object with:
    {
      "conferences": [
        {
          "full_name": string|null,
          "acronym": string|null,
          "name_acronym_urls": [url, ...],

          "us_year": string|null,            // a specific year between 2022–2024 when the conference was held in the United States
          "us_city": string|null,
          "us_state": string|null,
          "us_edition_urls": [url, ...],     // URLs supporting the US occurrence and venue details

          "core_rank": string|null,          // e.g., "A*", "A", "B"
          "research_rank": string|null,      // e.g., "12" (rank number as string)
          "research_rank_year": string|null, // e.g., "2024", "2025"
          "ranking_urls": [url, ...],        // URLs to CORE list and/or Research.com page

          "acceptance_rate": string|null,    // e.g., "23%", "0.23", "23 percent"
          "acceptance_year": string|null,    // e.g., "2023"
          "acceptance_urls": [url, ...],     // URLs that explicitly mention acceptance rate for full research papers

          "indexing_platform": string|null,  // e.g., "ACM Digital Library", "IEEE Xplore", or "Both"
          "indexing_urls": [url, ...],       // URLs that support the indexing claim

          "subfield": string|null,           // one of: "HCI", "AI", "Computer Vision", "NLP", "Machine Learning"
          "subfield_urls": [url, ...],       // URLs supporting the primary subfield focus

          "recent_year": string|null,        // the most recent year between 2022–2024 for which accepted full research papers count is given
          "accepted_papers": string|null,    // numeric string, e.g., "75"
          "papers_urls": [url, ...]          // URLs that explicitly support the accepted full papers count for that recent_year
        },
        ...
      ]
    }

    URL extraction rules:
    - Extract only URLs explicitly present in the answer (including markdown links).
    - Do not invent URLs. If none provided for a field, use an empty array.
    - Always include full URLs with protocol (http/https); if missing, prepend http://.

    Important:
    - Do not merge multiple conferences into one entry.
    - Preserve the original text (e.g., percentages and numbers) as strings.
    - If multiple years/rates are mentioned, pick the most recent one that fits the requested time window for the corresponding field and record that year/rate in the corresponding fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        key = u.strip()
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _parse_first_year(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"(20[2-4][0-9])", s)  # matches 2020-2049
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _parse_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"\d+", s.replace(",", ""))
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def _parse_percent_to_float(rate: Optional[str]) -> Optional[float]:
    if not rate:
        return None
    # Try percentage like "23%" or "23.5 %"
    m = re.search(r"(\d+(\.\d+)?)\s*%", rate)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # Try decimal like "0.23"
    m2 = re.fullmatch(r"\s*(0?\.\d+)\s*", rate)
    if m2:
        try:
            return float(m2.group(1)) * 100.0
        except Exception:
            return None
    # Try integer-only like "23"
    m3 = re.fullmatch(r"\s*(\d+)\s*", rate)
    if m3:
        try:
            return float(m3.group(1))
        except Exception:
            return None
    return None


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len([u for u in urls if _nonempty(u)]) > 0)


def _allowed_subfield(s: Optional[str]) -> bool:
    if not s:
        return False
    allowed = {"hci", "ai", "computer vision", "nlp", "machine learning"}
    return _normalize_text(s) in {_normalize_text(x) for x in allowed}


def _collect_any_sources(item: ConferenceItem) -> List[str]:
    all_lists = [
        item.name_acronym_urls,
        item.us_edition_urls,
        item.ranking_urls,
        item.acceptance_urls,
        item.indexing_urls,
        item.subfield_urls,
        item.papers_urls,
    ]
    merged: List[str] = []
    for lst in all_lists:
        merged.extend(lst or [])
    return _dedup_urls(merged)


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def _verify_conference(
    evaluator: Evaluator,
    parent_node,
    item: ConferenceItem,
    idx: int,
) -> None:
    """
    Build the subtree for a single conference and perform verifications.
    idx is 0-based; use idx+1 for human-readable numbering.
    """
    conf_num = idx + 1
    conf_node = evaluator.add_parallel(
        id=f"Conference_{conf_num}",
        desc=f"Conference #{conf_num}: constraint satisfaction and required fields",
        parent=parent_node,
        critical=False  # The conference contributes soft credit; all its own children will be critical.
    )

    # 1) Name & Acronym
    name_block = evaluator.add_sequential(
        id=f"C{conf_num}_Name_And_Acronym",
        desc=f"C{conf_num}: Provides the full conference name and acronym with supporting URLs",
        parent=conf_node,
        critical=False
    )
    exists_name = evaluator.add_custom_node(
        result=(_nonempty(item.full_name) and _nonempty(item.acronym) and _has_urls(_collect_any_sources(item))),
        id=f"C{conf_num}_Name_And_Acronym_exists",
        desc=f"C{conf_num}: Name and acronym provided and at least one supporting URL available",
        parent=name_block,
        critical=True
    )
    name_leaf = evaluator.add_leaf(
        id=f"C{conf_num}_Name_And_Acronym_supported",
        desc=f"C{conf_num}: The conference is called '{item.full_name}' and uses acronym '{item.acronym}'",
        parent=name_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The conference is commonly known as '{item.full_name}' with the acronym '{item.acronym}'.",
        node=name_leaf,
        sources=_collect_any_sources(item),
        additional_instruction="Verify that the provided sources explicitly support the full name and the acronym/abbreviation."
    )

    # 2) US Edition between 2022–2024 (year, city, state)
    us_block = evaluator.add_sequential(
        id=f"C{conf_num}_US_Edition_2022_2024_With_Venue",
        desc=f"C{conf_num}: US occurrence (2022–2024) with year and city/state, supported by URLs",
        parent=conf_node,
        critical=False
    )
    us_year_int = _parse_first_year(item.us_year)
    us_exists = evaluator.add_custom_node(
        result=(_has_urls(item.us_edition_urls) and _nonempty(item.us_city) and _nonempty(item.us_state)
                and (us_year_int is not None and 2022 <= us_year_int <= 2024)),
        id=f"C{conf_num}_US_Edition_exists",
        desc=f"C{conf_num}: US occurrence fields present with valid 2022–2024 year and supporting URLs",
        parent=us_block,
        critical=True
    )
    us_leaf = evaluator.add_leaf(
        id=f"C{conf_num}_US_Edition_supported",
        desc=f"C{conf_num}: In {item.us_year}, held in {item.us_city}, {item.us_state}, United States",
        parent=us_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {item.us_year}, the conference took place in {item.us_city}, {item.us_state}, United States.",
        node=us_leaf,
        sources=item.us_edition_urls,
        additional_instruction="Verify that the provided page(s) explicitly confirm the US location, the specific year (2022–2024), and the city/state."
    )

    # 3) Ranking: CORE A*/A OR Research.com top-30 for 2024 or later
    rank_block = evaluator.add_sequential(
        id=f"C{conf_num}_Ranking",
        desc=f"C{conf_num}: Meets ranking criterion (CORE A*/A or Research.com top-30 for 2024+)",
        parent=conf_node,
        critical=False
    )
    rank_exists = evaluator.add_custom_node(
        result=(_has_urls(item.ranking_urls) and (_nonempty(item.core_rank) or _nonempty(item.research_rank))),
        id=f"C{conf_num}_Ranking_exists",
        desc=f"C{conf_num}: Ranking information and URLs are provided",
        parent=rank_block,
        critical=True
    )
    rank_leaf = evaluator.add_leaf(
        id=f"C{conf_num}_Ranking_supported",
        desc=f"C{conf_num}: Ranking criterion satisfied",
        parent=rank_block,
        critical=True
    )
    rank_claim_parts: List[str] = []
    if _nonempty(item.core_rank):
        rank_claim_parts.append(f"it is rated '{item.core_rank}' by the CORE rankings")
    if _nonempty(item.research_rank) or _nonempty(item.research_rank_year):
        rank_claim_parts.append(f"it appears in Research.com with rank {item.research_rank} in {item.research_rank_year}")
    rank_claim_detail = " and ".join(rank_claim_parts) if rank_claim_parts else "ranking details as provided"
    await evaluator.verify(
        claim=(
            f"This conference satisfies the ranking requirement: either CORE A* or A OR Research.com top-30 "
            f"for 2024 or later. Specifically, {rank_claim_detail}. At least one of these must hold."
        ),
        node=rank_leaf,
        sources=item.ranking_urls,
        additional_instruction=(
            "Confirm at least one of the following is supported: "
            "(a) The conference has CORE rank A* or A; OR "
            "(b) It is top-30 in Research.com Computer Science Conference Rankings for 2024 or later. "
            "If the Research.com rank is provided, ensure the year is 2024 or newer and the rank is 30 or better."
        )
    )

    # 4) Acceptance Rate (most recent in 2020–2024) ∈ [15%, 30%]
    acc_block = evaluator.add_sequential(
        id=f"C{conf_num}_Acceptance_Rate",
        desc=f"C{conf_num}: Acceptance rate (most recent 2020–2024) between 15%–30%, with year and URLs",
        parent=conf_node,
        critical=False
    )
    acc_year_int = _parse_first_year(item.acceptance_year)
    acc_exists = evaluator.add_custom_node(
        result=(_has_urls(item.acceptance_urls) and _nonempty(item.acceptance_rate) and (acc_year_int is not None and 2020 <= acc_year_int <= 2024)),
        id=f"C{conf_num}_Acceptance_exists",
        desc=f"C{conf_num}: Acceptance rate and year (2020–2024) provided with supporting URLs",
        parent=acc_block,
        critical=True
    )
    acc_leaf = evaluator.add_leaf(
        id=f"C{conf_num}_Acceptance_supported",
        desc=f"C{conf_num}: Acceptance rate between 15% and 30% inclusive",
        parent=acc_block,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The most recently reported acceptance rate for full research papers between 2020–2024 is "
            f"{item.acceptance_rate} in {item.acceptance_year}, and its numeric value lies between 15% and 30% inclusive."
        ),
        node=acc_leaf,
        sources=item.acceptance_urls,
        additional_instruction=(
            "Confirm that the acceptance rate refers to full research papers (main track). "
            "Convert to percentage if needed (e.g., 0.23 => 23%). "
            "Ensure the year is between 2020 and 2024 (inclusive) and the rate is within [15%, 30%]."
        )
    )

    # 5) Proceedings indexing (ACM DL or IEEE Xplore)
    idx_block = evaluator.add_sequential(
        id=f"C{conf_num}_Proceedings_Indexing",
        desc=f"C{conf_num}: Proceedings indexed in ACM Digital Library or IEEE Xplore",
        parent=conf_node,
        critical=False
    )
    idx_exists = evaluator.add_custom_node(
        result=(_has_urls(item.indexing_urls) and _nonempty(item.indexing_platform)),
        id=f"C{conf_num}_Indexing_exists",
        desc=f"C{conf_num}: Indexing/platform and URLs provided",
        parent=idx_block,
        critical=True
    )
    idx_leaf = evaluator.add_leaf(
        id=f"C{conf_num}_Indexing_supported",
        desc=f"C{conf_num}: Proceedings indexed in ACM DL or IEEE Xplore",
        parent=idx_block,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The conference proceedings are indexed in either ACM Digital Library or IEEE Xplore. "
            f"For this conference, the indexing platform is '{item.indexing_platform}'."
        ),
        node=idx_leaf,
        sources=item.indexing_urls,
        additional_instruction="Verify that the provided source(s) explicitly indicate that the proceedings are indexed in ACM DL or IEEE Xplore."
    )

    # 6) Subfield focus (HCI, AI, Computer Vision, NLP, Machine Learning)
    sub_block = evaluator.add_sequential(
        id=f"C{conf_num}_Subfield_Focus",
        desc=f"C{conf_num}: Primary subfield focus is allowed (HCI, AI, CV, NLP, or ML)",
        parent=conf_node,
        critical=False
    )
    sub_exists = evaluator.add_custom_node(
        result=(_has_urls(item.subfield_urls) and _allowed_subfield(item.subfield)),
        id=f"C{conf_num}_Subfield_exists",
        desc=f"C{conf_num}: Subfield is one of the allowed and URLs provided",
        parent=sub_block,
        critical=True
    )
    sub_leaf = evaluator.add_leaf(
        id=f"C{conf_num}_Subfield_supported",
        desc=f"C{conf_num}: Primary focus is {item.subfield}",
        parent=sub_block,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The conference is primarily focused on {item.subfield}, which is one of: "
            f"Human-Computer Interaction (HCI), Artificial Intelligence (AI), Computer Vision, "
            f"Natural Language Processing (NLP), or Machine Learning."
        ),
        node=sub_leaf,
        sources=item.subfield_urls,
        additional_instruction="Verify that the provided source(s) show this conference is primarily in the stated subfield."
    )

    # 7) Accepted full research papers count ≥ 50 (most recent 2022–2024)
    papers_block = evaluator.add_sequential(
        id=f"C{conf_num}_Most_Recent_2022_2024_Paper_Count",
        desc=f"C{conf_num}: Accepted full research papers count (most recent 2022–2024) ≥ 50, with year and URLs",
        parent=conf_node,
        critical=False
    )
    recent_year_int = _parse_first_year(item.recent_year)
    papers_exists = evaluator.add_custom_node(
        result=(_has_urls(item.papers_urls) and _nonempty(item.accepted_papers)
                and (recent_year_int is not None and 2022 <= recent_year_int <= 2024)),
        id=f"C{conf_num}_Papers_exists",
        desc=f"C{conf_num}: Recent year 2022–2024 and accepted papers count provided with URLs",
        parent=papers_block,
        critical=True
    )
    papers_leaf = evaluator.add_leaf(
        id=f"C{conf_num}_Papers_supported",
        desc=f"C{conf_num}: {item.recent_year} accepted full papers count is at least 50",
        parent=papers_block,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"In {item.recent_year}, the total number of accepted full research papers was {item.accepted_papers}, "
            f"which is at least 50."
        ),
        node=papers_leaf,
        sources=item.papers_urls,
        additional_instruction="Confirm that the number refers to accepted full research papers (main track) for a 2022–2024 edition, and that it is ≥ 50."
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
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator; Root uses PARALLEL and is non-critical to allow partial credit aggregation.
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

    # Extract conferences list
    extracted: ConferencesExtraction = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferencesExtraction,
        extraction_name="conferences_extraction",
    )

    all_items: List[ConferenceItem] = list(extracted.conferences or [])
    original_count = len(all_items)

    # Choose the first 3 as per policy. If fewer than 3, pad with empty items to allow tree building.
    selected: List[ConferenceItem] = all_items[:3]
    while len(selected) < 3:
        selected.append(ConferenceItem())

    # Record simple info for debugging
    evaluator.add_custom_info(
        info={
            "original_conference_count": original_count,
            "selected_count": len(selected),
            "selected_names": [c.full_name for c in selected],
            "selected_acronyms": [c.acronym for c in selected]
        },
        info_type="extraction_stats",
        info_name="extraction_overview"
    )

    # Global checks under root
    at_least_three = original_count >= 3
    evaluator.add_custom_node(
        result=at_least_three,
        id="Exactly_Three_Conferences",
        desc="Response identifies at least three conferences (filtering will keep the first three)",
        parent=root,
        critical=True
    )

    # Distinctness among the first three (post-filter selection)
    norm_names = []
    for c in selected:
        rep = _normalize_text(c.full_name) or _normalize_text(c.acronym)
        norm_names.append(rep)
    distinct_ok = (len([n for n in norm_names if n]) == 3) and (len(set([n for n in norm_names if n])) == 3)
    evaluator.add_custom_node(
        result=distinct_ok,
        id="Distinct_Conferences",
        desc="The three selected conferences are distinct (based on normalized names/acronyms)",
        parent=root,
        critical=True
    )

    # Build each conference subtree
    for i, conf in enumerate(selected):
        await _verify_conference(evaluator, root, conf, i)

    # Return structured summary
    return evaluator.get_summary()