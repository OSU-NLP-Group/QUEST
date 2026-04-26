import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "most_matches_us_stadium_2026_wc"
TASK_DESCRIPTION = "Which United States stadium will host the most matches during the 2026 FIFA World Cup, and in which city is it located?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StadiumSelection(BaseModel):
    stadium_name: Optional[str] = None
    city: Optional[str] = None
    matches_claimed: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_stadium_selection() -> str:
    return """
    From the answer text, extract the single United States stadium the answer claims will host the most matches during the 2026 FIFA World Cup, and the city where it is located. Also extract any URLs explicitly cited as sources.

    Return a JSON object with:
    - stadium_name: The name of the US stadium identified as hosting the most matches (string). If not provided, return null.
    - city: The city (or municipality) where that stadium is located (string). If not provided, return null.
    - matches_claimed: If the answer explicitly states how many matches this stadium will host, return that number or phrase as a string (e.g., "8", "8 matches", or "tied for most with 7") else null.
    - sources: An array of all URLs that the answer cites as evidence for this claim. Only include actual URLs present in the answer (including markdown links). If no URLs are provided, return an empty list.

    If multiple stadiums are mentioned, choose the one the answer asserts will host the most matches (in the US). If there is ambiguity, pick the most prominent or first explicitly stated one.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _normalize_sources(srcs: Optional[List[str]]) -> List[str]:
    if not srcs:
        return []
    # Keep simple normalization; framework will handle markdown/caching/etc.
    return [s for s in srcs if _has_text(s)]


# --------------------------------------------------------------------------- #
# Main verification builder                                                   #
# --------------------------------------------------------------------------- #
async def _build_and_verify_tree(evaluator: Evaluator, extracted: StadiumSelection) -> None:
    """
    Build the verification tree per rubric and run verifications.
    """
    # Parent node as per rubric (critical, parallel)
    parent = evaluator.add_parallel(
        id="Most_Matches_US_Stadium_2026_WC",
        desc="Answer identifies the United States 2026 FIFA World Cup stadium that will host the most matches (tournament-wide) and gives its city.",
        parent=evaluator.root,
        critical=True
    )

    stadium = extracted.stadium_name or ""
    city = extracted.city or ""
    sources_list = _normalize_sources(extracted.sources)

    # 1) Stadium_Is_Official_US_Venue
    node_desc_1 = "The named stadium is an official 2026 FIFA World Cup venue located in the United States."
    if not _has_text(stadium):
        evaluator.add_custom_node(
            result=False,
            id="Stadium_Is_Official_US_Venue",
            desc=node_desc_1,
            parent=parent,
            critical=True
        )
    else:
        leaf1 = evaluator.add_leaf(
            id="Stadium_Is_Official_US_Venue",
            desc=node_desc_1,
            parent=parent,
            critical=True
        )
        claim1 = f"The stadium named '{stadium}' is one of the official venues for the 2026 FIFA World Cup and is located in the United States."
        await evaluator.verify(
            claim=claim1,
            node=leaf1,
            sources=sources_list,
            additional_instruction=(
                "Verify the stadium appears on the official list of 2026 FIFA World Cup host venues and "
                "that it is within the United States. Accept reputable sources (e.g., FIFA.com, host city pages, "
                "major news outlets) listed in the provided URLs. If sources are absent, judge using the provided "
                "answer context only."
            )
        )

    # 2) Stadium_Hosts_Most_Matches_Among_All_Host_Venues
    node_desc_2 = "The named stadium hosts the highest number of matches among all 16 host venues across Canada, Mexico, and the United States."
    if not _has_text(stadium):
        evaluator.add_custom_node(
            result=False,
            id="Stadium_Hosts_Most_Matches_Among_All_Host_Venues",
            desc=node_desc_2,
            parent=parent,
            critical=True
        )
    else:
        leaf2 = evaluator.add_leaf(
            id="Stadium_Hosts_Most_Matches_Among_All_Host_Venues",
            desc=node_desc_2,
            parent=parent,
            critical=True
        )
        if _has_text(extracted.matches_claimed):
            claim2 = (
                f"Among all 16 host venues for the 2026 FIFA World Cup across Canada, Mexico, and the United States, "
                f"the stadium '{stadium}' will host the highest number of matches (as claimed: {extracted.matches_claimed})."
            )
        else:
            claim2 = (
                f"Among all 16 host venues for the 2026 FIFA World Cup across Canada, Mexico, and the United States, "
                f"the stadium '{stadium}' will host the highest number of matches."
            )
        await evaluator.verify(
            claim=claim2,
            node=leaf2,
            sources=sources_list,
            additional_instruction=(
                "Check the total number of matches assigned to each of the 16 official venues. "
                "Confirm that the named stadium has the strictly highest total or is tied for the highest total "
                "among all venues. If it is not at least tied for the most, mark as not supported."
            )
        )

    # 3) City_Location_Provided_And_Correct
    node_desc_3 = "The answer provides the city where the named stadium is located, and it matches the stadium identified."
    if not (_has_text(stadium) and _has_text(city)):
        evaluator.add_custom_node(
            result=False,
            id="City_Location_Provided_And_Correct",
            desc=node_desc_3,
            parent=parent,
            critical=True
        )
    else:
        leaf3 = evaluator.add_leaf(
            id="City_Location_Provided_And_Correct",
            desc=node_desc_3,
            parent=parent,
            critical=True
        )
        claim3 = f"The stadium '{stadium}' is located in '{city}'."
        await evaluator.verify(
            claim=claim3,
            node=leaf3,
            sources=sources_list,
            additional_instruction=(
                "Verify that the provided city corresponds to the stadium's actual location. "
                "Allow reasonable metropolitan or municipal variants (e.g., Inglewood ~ Los Angeles area; "
                "Arlington ~ Dallas–Fort Worth area; East Rutherford ~ New York/New Jersey area). "
                "Minor spelling or capitalization differences are acceptable."
            )
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
    Evaluate an answer for: Which United States stadium will host the most matches during the 2026 FIFA World Cup, and in which city is it located?
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_stadium_selection(),
        template_class=StadiumSelection,
        extraction_name="stadium_selection"
    )

    # Build tree and verify
    await _build_and_verify_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()