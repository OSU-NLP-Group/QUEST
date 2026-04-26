import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "big_ten_2024_expansion"
TASK_DESCRIPTION = (
    "In 2024, the Big Ten Conference expanded by officially admitting four universities from the Pac-12 Conference "
    "on a specific date in August. Identify all four universities that joined the Big Ten Conference in this August "
    "2024 expansion. For each university, provide the following information: (1) the official university name, "
    "(2) the state where it is located, (3) the city where its main campus is located, (4) whether it is a public or "
    "private institution, (5) the name of the conference it left to join the Big Ten, and (6) the exact date (month, "
    "day, and year) it officially became a Big Ten member."
)

# Optional ground truth reference for context (not used to auto-fail branches; for summary/debugging)
EXPECTED_UNIS = [
    {
        "official_name": "University of Oregon",
        "state": "Oregon",
        "city": "Eugene",
        "institution_status": "public",
        "previous_conference": "Pac-12 Conference",
        "join_date": "August 2, 2024",
    },
    {
        "official_name": "University of Washington",
        "state": "Washington",
        "city": "Seattle",
        "institution_status": "public",
        "previous_conference": "Pac-12 Conference",
        "join_date": "August 2, 2024",
    },
    {
        "official_name": "University of California, Los Angeles",
        "state": "California",
        "city": "Los Angeles",
        "institution_status": "public",
        "previous_conference": "Pac-12 Conference",
        "join_date": "August 2, 2024",
    },
    {
        "official_name": "University of Southern California",
        "state": "California",
        "city": "Los Angeles",
        "institution_status": "private",
        "previous_conference": "Pac-12 Conference",
        "join_date": "August 2, 2024",
    },
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityInfo(BaseModel):
    """Information for a single university in the expansion."""
    official_name: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    institution_status: Optional[str] = None  # "public" or "private"
    previous_conference: Optional[str] = None
    join_date: Optional[str] = None  # e.g., "August 2, 2024"
    source_urls: List[str] = Field(default_factory=list)


class ExpansionExtraction(BaseModel):
    """Extraction of the complete expansion set from the answer."""
    universities: List[UniversityInfo] = Field(default_factory=list)
    post_expansion_member_count: Optional[str] = None  # e.g., "18" or "eighteen"


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_expansion_universities() -> str:
    return """
    Extract the four universities that the answer claims officially joined the Big Ten Conference in August 2024 (the expansion entrants from the Pac-12). For each identified university, extract the following fields exactly as stated in the answer:

    - official_name: The official university name (prefer the full formal name; if only an abbreviation is provided, extract that abbreviation).
    - state: The U.S. state where the university’s main campus is located (e.g., "California", "Oregon").
    - city: The city where its main campus is located (e.g., "Los Angeles", "Eugene", "Seattle").
    - institution_status: Whether it is a "public" or "private" institution (simple lowercase string preferred).
    - previous_conference: The conference the university left to join the Big Ten (e.g., "Pac-12 Conference").
    - join_date: The exact official date (month, day, year) it became a Big Ten member (e.g., "August 2, 2024").
    - source_urls: All URLs cited in the answer that directly support this university’s membership change and/or the required attributes. Extract only valid URLs explicitly presented in the answer (including Markdown links).

    Additional rule:
    - If the answer lists more than four universities, include only the first four mentioned in the answer.
    - If any field is missing for a university, set it to null (for strings) or an empty list (for source_urls).
    - Do not invent or infer any information that is not explicitly present in the answer text.
    - For source_urls, extract only the actual URLs; if none are present, return an empty list.

    Also, extract the answer’s stated post-expansion member count for the Big Ten:
    - post_expansion_member_count: The number of member institutions after the August 2024 expansion, as explicitly stated in the answer (e.g., "18" or "eighteen"). If not mentioned, return null.

    Return JSON with:
    {
      "universities": [ ... up to 4 items ... ],
      "post_expansion_member_count": "... or null ..."
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_univ_name(name: Optional[str]) -> str:
    if not name:
        return ""
    s = name.strip().lower()
    # Normalize common abbreviations to help dedup
    abbrev_map = {
        "ucla": "university of california, los angeles",
        "usc": "university of southern california",
    }
    s = abbrev_map.get(s, s)
    # Remove punctuation to make comparisons more robust
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def first_four_unis(unis: List[UniversityInfo]) -> List[UniversityInfo]:
    return unis[:4] if unis else []


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_global_requirements(
    evaluator: Evaluator,
    parent_node,
    extraction: ExpansionExtraction,
) -> None:
    """
    Build and verify the Global Requirements node:
    - Exactly four distinct universities provided
    - After expansion, Big Ten member count stated as 18
    """
    global_node = evaluator.add_parallel(
        id="Global_Requirements",
        desc="Global constraints that apply to the full set of universities",
        parent=parent_node,
        critical=True
    )

    # Exactly four distinct universities (no duplicates)
    uni_list = first_four_unis(extraction.universities)
    unique_names = {normalize_univ_name(u.official_name) for u in uni_list if normalize_univ_name(u.official_name)}
    exactly_four_distinct = (len(uni_list) == 4) and (len(unique_names) == 4)

    evaluator.add_custom_node(
        result=exactly_four_distinct,
        id="Exactly_Four_Distinct_Universities",
        desc="Response identifies exactly four distinct universities (no duplicates) as the expansion entrants",
        parent=global_node,
        critical=True
    )

    # Post-expansion member count = 18
    post_count_leaf = evaluator.add_leaf(
        id="Post_Expansion_Member_Count_18",
        desc="Correctly indicates that after this expansion, the Big Ten Conference has 18 member institutions",
        parent=global_node,
        critical=True
    )

    # Use all provided sources across universities to verify the 18-member claim (if any)
    all_sources: List[str] = []
    for u in uni_list:
        if u.source_urls:
            all_sources.extend(u.source_urls)

    claim = "After the August 2024 expansion, the Big Ten Conference has 18 member institutions."
    add_ins = (
        "Pass only if the evidence or the answer explicitly indicates the Big Ten has 18 members after this expansion. "
        "Accept phrasing like '18 members', '18 member institutions', or '18 full members'."
    )
    await evaluator.verify(
        claim=claim,
        node=post_count_leaf,
        sources=all_sources if all_sources else None,
        additional_instruction=add_ins
    )


async def verify_university(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    uni: UniversityInfo
) -> None:
    """
    Verify all aspects for a single university entry.
    The node is non-critical at university level to allow partial credit across universities,
    but each attribute leaf is critical within the university branch to ensure internal consistency.
    """
    univ_node = evaluator.add_parallel(
        id=f"University_{idx+1}",
        desc=f"Information for the {idx+1}th university among the four expansion entrants",
        parent=parent_node,
        critical=False
    )

    sources = uni.source_urls if uni.source_urls else None

    # 1) Is Expansion Entrant
    entrant_leaf = evaluator.add_leaf(
        id=f"Univ{idx+1}_Is_Expansion_Entrant",
        desc="The university identified is one of the four universities that officially joined the Big Ten in the August 2024 expansion",
        parent=univ_node,
        critical=True
    )
    claim_entrant = (
        f"{uni.official_name or 'This university'} officially joined the Big Ten Conference in August 2024 "
        f"as part of the four-team expansion."
    )
    await evaluator.verify(
        claim=claim_entrant,
        node=entrant_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm via the cited sources (or the provided answer context) that this school is one of the four "
            "Pac-12 universities that became Big Ten members in August 2024."
        ),
    )

    # 2) Official Name
    name_leaf = evaluator.add_leaf(
        id=f"Univ{idx+1}_Official_Name",
        desc="Correctly identifies the official university name",
        parent=univ_node,
        critical=True
    )
    claim_name = f"The official university name is '{uni.official_name}'."
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the provided school name matches the standard official or formal name used by reputable sources "
            "(e.g., university site or Wikipedia). Accept common abbreviations if the page clearly connects them to the "
            "full official name (e.g., 'UCLA' -> 'University of California, Los Angeles', 'USC' -> 'University of Southern California')."
        ),
    )

    # 3) State
    state_leaf = evaluator.add_leaf(
        id=f"Univ{idx+1}_State",
        desc="Correctly identifies the state where the university is located",
        parent=univ_node,
        critical=True
    )
    claim_state = f"The state where the university’s main campus is located is '{uni.state}'."
    await evaluator.verify(
        claim=claim_state,
        node=state_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the state of the main campus (e.g., California, Oregon, Washington). "
            "Accept equivalent formats (e.g., 'CA' vs 'California') if clearly indicated."
        ),
    )

    # 4) City
    city_leaf = evaluator.add_leaf(
        id=f"Univ{idx+1}_City",
        desc="Correctly identifies the city where the main campus is located",
        parent=univ_node,
        critical=True
    )
    claim_city = f"The city where the main campus is located is '{uni.city}'."
    await evaluator.verify(
        claim=claim_city,
        node=city_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the primary campus city (e.g., Los Angeles, Eugene, Seattle). "
            "If the source shows city + state together, that is acceptable."
        ),
    )

    # 5) Institution Status
    status_leaf = evaluator.add_leaf(
        id=f"Univ{idx+1}_Institution_Status",
        desc="Correctly identifies whether the university is public or private",
        parent=univ_node,
        critical=True
    )
    claim_status = f"The university is a '{uni.institution_status}' institution."
    await evaluator.verify(
        claim=claim_status,
        node=status_leaf,
        sources=sources,
        additional_instruction=(
            "Verify whether the institution is public or private based on authoritative sources. "
            "Accept variants like 'public research university' or 'private research university' as equivalent."
        ),
    )

    # 6) Previous Conference
    prev_conf_leaf = evaluator.add_leaf(
        id=f"Univ{idx+1}_Previous_Conference",
        desc="Correctly identifies the conference the university left to join the Big Ten (Pac-12 Conference)",
        parent=univ_node,
        critical=True
    )
    claim_prev_conf = (
        f"This university left the Pac-12 Conference to join the Big Ten Conference."
    )
    await evaluator.verify(
        claim=claim_prev_conf,
        node=prev_conf_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the university previously competed in the Pac-12 Conference before moving to the Big Ten."
        ),
    )

    # 7) Join Date
    join_date_leaf = evaluator.add_leaf(
        id=f"Univ{idx+1}_Join_Date",
        desc="Correctly provides the exact official join date (August 2, 2024)",
        parent=univ_node,
        critical=True
    )
    claim_join_date = (
        f"This university officially became a Big Ten member on August 2, 2024."
    )
    await evaluator.verify(
        claim=claim_join_date,
        node=join_date_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the effective membership date is August 2, 2024. "
            "Accept synonymous phrasing such as 'effective August 2, 2024' or 'joined August 2, 2024'."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Big Ten 2024 expansion task.
    """
    # Initialize evaluator (root is non-critical by design in framework)
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
        default_model=model
    )

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_expansion_universities(),
        template_class=ExpansionExtraction,
        extraction_name="expansion_extraction"
    )

    # Add ground truth info for visibility (not used as gating)
    evaluator.add_ground_truth({
        "expected_universities": EXPECTED_UNIS,
        "expected_member_count_after_expansion": 18,
        "expected_join_date_for_all": "August 2, 2024"
    }, gt_type="reference")

    # Create main task node (non-critical) to allow partial credit across universities
    main_task_node = evaluator.add_parallel(
        id="Big_Ten_2024_Expansion",
        desc="Identify the four August 2024 Big Ten expansion universities and provide required attributes for each",
        parent=root,
        critical=False
    )

    # Global requirements verification
    await verify_global_requirements(evaluator, main_task_node, extraction)

    # Verify up to four universities (pad with empty entries if fewer present to keep structure consistent)
    universities = first_four_unis(extraction.universities)
    while len(universities) < 4:
        universities.append(UniversityInfo())

    for i, uni in enumerate(universities):
        await verify_university(evaluator, main_task_node, i, uni)

    # Return summary report
    return evaluator.get_summary()