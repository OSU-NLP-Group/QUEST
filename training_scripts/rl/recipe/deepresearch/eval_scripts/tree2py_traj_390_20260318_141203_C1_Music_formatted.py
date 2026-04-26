import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "debut_album_release_date_normani"
TASK_DESCRIPTION = """
What is the release date (month, day, and year) of the debut studio album by the American artist who competed in Season 24 of Dancing with the Stars in 2017 and released the Platinum-certified debut solo single 'Motivation' in August 2019?
"""

# Ground truth reference (for summary/info only; not used to bias verification)
GROUND_TRUTH = {
    "artist": "Normani",
    "debut_single": {
        "title": "Motivation",
        "release_date": "August 16, 2019",
        "certification": "RIAA Platinum"
    },
    "debut_album": {
        "title": "Dopamine",
        "release_date": "June 14, 2024",
        "label": "RCA Records"
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AnswerExtraction(BaseModel):
    artist_name: Optional[str] = None
    debut_single_title: Optional[str] = None
    debut_single_release_date: Optional[str] = None
    debut_single_certifications: List[str] = Field(default_factory=list)
    debut_album_title: Optional[str] = None
    debut_album_release_date: Optional[str] = None
    debut_album_label: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
    From the provided answer, extract the following fields exactly as they appear in the answer. Do not infer or add information that is not explicitly stated. If an item is not present in the answer text, return null (or an empty list for list fields).

    Fields to extract:
    - artist_name: The name of the artist the answer identifies.
    - debut_single_title: The title of the artist's debut solo single as stated in the answer (if mentioned).
    - debut_single_release_date: The release date of the debut solo single as stated in the answer (if mentioned).
    - debut_single_certifications: A list of any certifications (e.g., "RIAA Platinum") explicitly mentioned for the debut solo single.
    - debut_album_title: The title of the identified artist's debut studio album (if stated).
    - debut_album_release_date: The release date of that debut studio album exactly as written in the answer (preserve the original formatting).
    - debut_album_label: The record label(s) through which the debut album was released (e.g., "RCA Records", or "Keep Cool/RCA Records") if mentioned.
    - urls: Extract all explicit URLs present in the answer text (including plain URLs or markdown links). If there are no URLs, return an empty list.

    Return a single JSON object with these fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_mdy_date_format(date_str: Optional[str]) -> bool:
    """
    Check if the provided date string matches a month/day/year numeric format, allowing
    1-2 digits for month and day, and 4 digits for year. Examples: 6/14/2024, 06/14/2024.
    """
    if not date_str:
        return False
    pattern = r"^\s*(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/([12]\d{3})\s*$"
    return re.match(pattern, date_str.strip()) is not None


def join_sources(*lists: List[str]) -> List[str]:
    """Combine multiple URL lists into one, preserving order and removing obvious duplicates."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_identify_artist_checks(
    evaluator: Evaluator,
    parent,
    data: AnswerExtraction
) -> None:
    """
    Build the Parallel, critical node to verify the artist identity constraints.
    """
    identify_node = evaluator.add_parallel(
        id="Identify_Correct_Artist",
        desc="Correctly identify the artist that matches all provided descriptors/constraints.",
        parent=parent,
        critical=True
    )

    sources = data.urls

    # 1) Artist Is American
    node_american = evaluator.add_leaf(
        id="Artist_Is_American",
        desc="The identified artist is American.",
        parent=identify_node,
        critical=True
    )
    claim_american = f"The artist {data.artist_name or 'the identified artist'} is American (U.S.)."
    await evaluator.verify(
        claim=claim_american,
        node=node_american,
        sources=sources,
        additional_instruction="Look for phrases like 'American singer', 'American artist', or nationality explicitly indicating United States."
    )

    # 2) DWTS Season 24 competitor in 2017
    node_dwts = evaluator.add_leaf(
        id="DWTS_Season_24_Competitor_2017",
        desc="The identified artist competed in Dancing with the Stars Season 24 (aired in 2017).",
        parent=identify_node,
        critical=True
    )
    claim_dwts = f"{data.artist_name or 'The identified artist'} competed in Season 24 of Dancing with the Stars in 2017."
    await evaluator.verify(
        claim=claim_dwts,
        node=node_dwts,
        sources=sources,
        additional_instruction="Accept references to 'DWTS Season 24' or 'the 24th season' in 2017; mentions of their professional dance partner or placement also support the claim."
    )

    # 3) Debut solo single title is 'Motivation'
    node_single_title = evaluator.add_leaf(
        id="Debut_Solo_Single_Is_Motivation",
        desc="The identified artist's debut solo single is titled 'Motivation'.",
        parent=identify_node,
        critical=True
    )
    claim_single_title = f"The debut solo single by {data.artist_name or 'the identified artist'} is titled 'Motivation'."
    await evaluator.verify(
        claim=claim_single_title,
        node=node_single_title,
        sources=sources,
        additional_instruction="Confirm that 'Motivation' is explicitly described as the artist's debut solo single."
    )

    # 4) 'Motivation' release date August 16, 2019
    node_single_date = evaluator.add_leaf(
        id="Motivation_Release_Date_Is_Aug_16_2019",
        desc="The debut solo single 'Motivation' was released on August 16, 2019.",
        parent=identify_node,
        critical=True
    )
    claim_single_date = f"The single 'Motivation' by {data.artist_name or 'the identified artist'} was released on August 16, 2019."
    await evaluator.verify(
        claim=claim_single_date,
        node=node_single_date,
        sources=sources,
        additional_instruction="Allow minor formatting variants like '16 August 2019' or 'Aug. 16, 2019', but the date must correspond to 2019-08-16."
    )

    # 5) 'Motivation' RIAA Platinum
    node_single_platinum = evaluator.add_leaf(
        id="Motivation_RIAA_Platinum",
        desc="The debut solo single 'Motivation' received a Platinum certification from the RIAA.",
        parent=identify_node,
        critical=True
    )
    claim_single_platinum = f"The single 'Motivation' by {data.artist_name or 'the identified artist'} has been certified Platinum by the RIAA."
    await evaluator.verify(
        claim=claim_single_platinum,
        node=node_single_platinum,
        sources=sources,
        additional_instruction="Accept 'Platinum' or higher RIAA certifications (e.g., Multi-Platinum) as satisfying 'Platinum'."
    )


async def build_album_checks(
    evaluator: Evaluator,
    parent,
    data: AnswerExtraction
) -> None:
    """
    Build the Parallel, critical node to verify the debut album details.
    """
    album_node = evaluator.add_parallel(
        id="Provide_Debut_Album_Release_Date",
        desc="Provide the release date (month/day/year) of the identified artist’s debut studio album, consistent with the constraints.",
        parent=parent,
        critical=True
    )

    sources = data.urls

    # 1) Album is debut studio album
    node_album_debut = evaluator.add_leaf(
        id="Album_Is_Debut_Studio_Album",
        desc="The album referenced is the artist's debut (first) studio album.",
        parent=album_node,
        critical=True
    )
    if data.debut_album_title and data.artist_name:
        claim_album_debut = f"'{data.debut_album_title}' is the debut (first) studio album by {data.artist_name}."
    elif data.artist_name:
        claim_album_debut = f"The debut (first) studio album by {data.artist_name} is correctly identified as their debut studio album."
    else:
        claim_album_debut = "The referenced album is the artist's debut (first) studio album."
    await evaluator.verify(
        claim=claim_album_debut,
        node=node_album_debut,
        sources=sources,
        additional_instruction="Look for wording like 'debut studio album' on authoritative sources (labels, major publications, Wikipedia, etc.)."
    )

    # 2) Album release date checks: world value + answer formatting (month/day/year)
    # Use a sequential sub-node so formatting is only considered if the date itself is correct.
    date_checks = evaluator.add_sequential(
        id="Album_Released_June_14_2024",
        desc="The stated debut studio album release date is June 14, 2024 and is given as month/day/year.",
        parent=album_node,
        critical=True
    )

    # 2.a) World value check
    leaf_date_value = evaluator.add_leaf(
        id="Album_ReleaseDate_Value_June_14_2024",
        desc="The debut studio album's actual release date is June 14, 2024.",
        parent=date_checks,
        critical=True
    )
    if data.debut_album_title and data.artist_name:
        claim_album_date = f"The album '{data.debut_album_title}' by {data.artist_name} was released on June 14, 2024."
    elif data.artist_name:
        claim_album_date = f"The debut studio album by {data.artist_name} was released on June 14, 2024."
    else:
        claim_album_date = "The debut studio album in question was released on June 14, 2024."
    await evaluator.verify(
        claim=claim_album_date,
        node=leaf_date_value,
        sources=sources,
        additional_instruction="Allow variants like '14 June 2024' or numeric formats as long as the date corresponds to 2024-06-14."
    )

    # 2.b) Answer formatting check (must be month/day/year)
    # Implemented as a custom node (binary) based on the extracted string format from the answer.
    date_format_ok = is_mdy_date_format(data.debut_album_release_date)
    evaluator.add_custom_node(
        result=date_format_ok,
        id="Album_ReleaseDate_Format_MDY",
        desc="The stated album release date is given in month/day/year format (e.g., 6/14/2024 or 06/14/2024).",
        parent=date_checks,
        critical=True
    )

    # 3) Album released through RCA
    node_label = evaluator.add_leaf(
        id="Album_Released_Through_RCA",
        desc="The debut studio album is described as released through RCA Records.",
        parent=album_node,
        critical=True
    )
    if data.debut_album_title and data.artist_name:
        claim_label = f"The album '{data.debut_album_title}' by {data.artist_name} was released through RCA Records."
    elif data.artist_name:
        claim_label = f"The debut studio album by {data.artist_name} was released through RCA Records."
    else:
        claim_label = "The debut studio album was released through RCA Records."
    await evaluator.verify(
        claim=claim_label,
        node=node_label,
        sources=sources,
        additional_instruction="Treat 'RCA Records', 'RCA', or 'Keep Cool/RCA Records' as satisfying this condition."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the debut album release date identification task.
    """
    # Initialize evaluator with a sequential strategy so album checks are skipped if artist identification fails
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Record ground truth info (for transparency in the summary)
    evaluator.add_ground_truth(
        {
            "expected_artist": GROUND_TRUTH["artist"],
            "expected_debut_single": GROUND_TRUTH["debut_single"],
            "expected_debut_album": GROUND_TRUTH["debut_album"]
        },
        gt_type="ground_truth"
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=AnswerExtraction,
        extraction_name="parsed_answer_fields"
    )

    # Add a small custom info block for diagnostics
    evaluator.add_custom_info(
        {
            "extracted_artist": extracted.artist_name,
            "extracted_debut_album_title": extracted.debut_album_title,
            "extracted_debut_album_release_date_raw": extracted.debut_album_release_date,
            "total_urls_found": len(extracted.urls)
        },
        info_type="diagnostics",
        info_name="extraction_diagnostics"
    )

    # Build verification subtrees
    await build_identify_artist_checks(evaluator, root, extracted)
    await build_album_checks(evaluator, root, extracted)

    # Return standardized summary
    return evaluator.get_summary()