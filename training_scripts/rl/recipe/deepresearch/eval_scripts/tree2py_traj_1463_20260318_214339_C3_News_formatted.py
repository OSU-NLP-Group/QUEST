import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cabinet_confirmation_2025"
TASK_DESCRIPTION = """
Identify the U.S. Cabinet Secretary who was confirmed by the Senate in 2025 with a vote count of exactly 59 Yeas and 38 Nays, where exactly eight Democratic senators voted in favor of the confirmation. This individual must be from the state of Colorado, with both Colorado senators having voted in favor. The individual must also be the CEO of Liberty Energy, an energy company. The confirmation vote must have occurred on February 3, 2025, at 6:13 PM Eastern Time, and must be recorded as Senate Roll Call Vote Number 30. Provide the individual's full name and the official Senate.gov URL documenting this roll call vote.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConfirmationAnswerExtraction(BaseModel):
    """
    Extract the key elements from the agent's answer:
    - name: The individual's full name provided by the agent.
    - senate_vote_url: The official Senate.gov roll call vote URL documenting the confirmation.
    - all_urls: All URLs explicitly mentioned in the answer (deduplicated). Include senate_vote_url if present.
    """
    name: Optional[str] = None
    senate_vote_url: Optional[str] = None
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_confirmation_info() -> str:
    return """
    Extract the following from the answer:

    1) name: The individual's full name explicitly provided.
    2) senate_vote_url: The official Senate.gov roll call vote URL that documents the confirmation vote discussed in the answer.
       - It must be a URL on the senate.gov domain that specifically documents a roll call vote (not a press release or unrelated page).
       - If multiple Senate.gov URLs are present, choose the one that most directly documents the roll call vote for the confirmation.
       - If not provided, set to null.
    3) all_urls: A deduplicated list of all URLs explicitly mentioned anywhere in the answer, including markdown links.
       - Include the senate_vote_url if present.
       - Ensure each URL is complete (prepend http:// if missing a protocol).
    
    Return a JSON object with fields: name, senate_vote_url, all_urls.
    If any field is missing from the answer, set it to null (or [] for all_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_senate_roll_call_url(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
        host_ok = parsed.netloc.lower().endswith("senate.gov")
        return bool(host_ok)
    except Exception:
        return False


def _consolidate_urls(extracted: ConfirmationAnswerExtraction) -> List[str]:
    urls_set = set(extracted.all_urls or [])
    if extracted.senate_vote_url:
        urls_set.add(extracted.senate_vote_url)
    # Filter obviously empty strings
    urls = [u for u in urls_set if isinstance(u, str) and u.strip()]
    return urls


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, extracted: ConfirmationAnswerExtraction) -> None:
    # Add top-level critical task node
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Successfully identify the individual meeting all specified criteria and provide required documentation",
        parent=evaluator.root,
        critical=True
    )

    # Child 1: Individual_Identity (sequential)
    identity_node = evaluator.add_sequential(
        id="Individual_Identity",
        desc="Correctly identify the full name of the individual",
        parent=task_node,
        critical=True
    )

    position_node = evaluator.add_parallel(
        id="Position_Verification",
        desc="Verify the individual's official position and confirmation year",
        parent=identity_node,
        critical=True
    )

    # Leaf under Position_Verification: Reference_URL_Position (custom existence/format check)
    senate_url_ok = evaluator.add_custom_node(
        result=_is_senate_roll_call_url(extracted.senate_vote_url),
        id="Reference_URL_Position",
        desc="Provide official Senate.gov URL documenting the confirmation",
        parent=position_node,
        critical=True
    )

    # Leaf under Position_Verification: Cabinet_Secretary (verify by senate.gov URL)
    cabinet_secretary_leaf = evaluator.add_leaf(
        id="Cabinet_Secretary",
        desc="The individual must be a U.S. Cabinet Secretary",
        parent=position_node,
        critical=True
    )
    await evaluator.verify(
        claim="This Senate roll call vote documents the confirmation of a nominee to serve as a U.S. Cabinet Secretary (e.g., 'to be Secretary of [Department]'). It is not for a deputy, under secretary, or a non-cabinet position.",
        node=cabinet_secretary_leaf,
        sources=extracted.senate_vote_url,
        additional_instruction="Look for wording like 'Nomination Confirmed' and 'to be Secretary of ...' on the page. Reject if it is for a non-cabinet role.",
        extra_prerequisites=[senate_url_ok]
    )

    # Leaf under Position_Verification: Year_2025_Confirmation (verify by senate.gov URL)
    year_2025_leaf = evaluator.add_leaf(
        id="Year_2025_Confirmation",
        desc="The confirmation must have occurred in 2025",
        parent=position_node,
        critical=True
    )
    await evaluator.verify(
        claim="This roll call vote occurred in the year 2025.",
        node=year_2025_leaf,
        sources=extracted.senate_vote_url,
        additional_instruction="Check the date shown on the roll call page; the year should be 2025.",
        extra_prerequisites=[senate_url_ok]
    )

    # Child 2: Vote_Characteristics (parallel)
    vote_char_node = evaluator.add_parallel(
        id="Vote_Characteristics",
        desc="Verify all Senate confirmation vote details",
        parent=task_node,
        critical=True
    )

    # Vote_Count
    vote_count_leaf = evaluator.add_leaf(
        id="Vote_Count",
        desc="The confirmation vote count must be exactly 59 Yeas and 38 Nays",
        parent=vote_char_node,
        critical=True
    )
    await evaluator.verify(
        claim="This roll call vote shows exactly 59 Yea votes and 38 Nay votes.",
        node=vote_count_leaf,
        sources=extracted.senate_vote_url,
        additional_instruction="Ignore 'Present' or 'Not Voting'. Only confirm that Yea=59 and Nay=38.",
        extra_prerequisites=[senate_url_ok]
    )

    # Democratic_Support
    dem_support_leaf = evaluator.add_leaf(
        id="Democratic_Support",
        desc="Exactly eight Democratic senators must have voted in favor",
        parent=vote_char_node,
        critical=True
    )
    await evaluator.verify(
        claim="Exactly eight Democratic senators voted Yea (in favor) on this roll call vote.",
        node=dem_support_leaf,
        sources=extracted.senate_vote_url,
        additional_instruction="Use the party breakdown or the list of votes to count Democratic Yeas. The count must be exactly 8.",
        extra_prerequisites=[senate_url_ok]
    )

    # Colorado_Senators_Support
    co_senators_leaf = evaluator.add_leaf(
        id="Colorado_Senators_Support",
        desc="Both U.S. senators from Colorado must have voted in favor of the confirmation",
        parent=vote_char_node,
        critical=True
    )
    await evaluator.verify(
        claim="Both Colorado senators, Michael Bennet (D-CO) and John Hickenlooper (D-CO), voted Yea on this roll call vote.",
        node=co_senators_leaf,
        sources=extracted.senate_vote_url,
        additional_instruction="Confirm that both Bennet and Hickenlooper are listed among the Yea votes.",
        extra_prerequisites=[senate_url_ok]
    )

    # Child 3: Geographic_and_Professional (parallel)
    geo_prof_node = evaluator.add_parallel(
        id="Geographic_and_Professional",
        desc="Verify state affiliation and professional background",
        parent=task_node,
        critical=True
    )

    # Colorado_Origin (verify via any provided URLs; prefer non-senate sources if available)
    colorado_origin_leaf = evaluator.add_leaf(
        id="Colorado_Origin",
        desc="The individual must be from the state of Colorado",
        parent=geo_prof_node,
        critical=True
    )
    all_urls = _consolidate_urls(extracted)
    non_senate_urls = [u for u in all_urls if not _is_senate_roll_call_url(u)]
    colorado_name = extracted.name or "the individual"
    await evaluator.verify(
        claim=f"{colorado_name} is from the state of Colorado (e.g., Colorado-based, Colorado resident, or commonly identified as being from Colorado).",
        node=colorado_origin_leaf,
        sources=non_senate_urls if non_senate_urls else all_urls,
        additional_instruction="Check reputable bios or company/about pages. Accept phrasing like 'Colorado businessman', 'based in Denver, Colorado', or 'from Colorado'."
    )

    # Company_Leadership (parallel under geo_prof_node)
    company_leadership_node = evaluator.add_parallel(
        id="Company_Leadership",
        desc="Verify CEO position and company identity",
        parent=geo_prof_node,
        critical=True
    )

    # Energy_Company_CEO
    energy_company_ceo_leaf = evaluator.add_leaf(
        id="Energy_Company_CEO",
        desc="The individual must be the CEO of an energy company",
        parent=company_leadership_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{colorado_name} is the Chief Executive Officer (CEO) of an energy company.",
        node=energy_company_ceo_leaf,
        sources=non_senate_urls if non_senate_urls else all_urls,
        additional_instruction="Prefer sources that explicitly say 'CEO'. If the page says 'CEO of Liberty Energy' that qualifies as an energy company."
    )

    # Liberty_Energy
    liberty_energy_leaf = evaluator.add_leaf(
        id="Liberty_Energy",
        desc="The company must be Liberty Energy",
        parent=company_leadership_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{colorado_name} is the CEO of Liberty Energy.",
        node=liberty_energy_leaf,
        sources=non_senate_urls if non_senate_urls else all_urls,
        additional_instruction="Accept if the company is identified as Liberty Energy (formerly Liberty Oilfield Services is acceptable if it clearly refers to the same company)."
    )

    # Child 4: Temporal_Precision (parallel)
    temporal_node = evaluator.add_parallel(
        id="Temporal_Precision",
        desc="Verify exact timing details of the confirmation vote",
        parent=task_node,
        critical=True
    )

    # Confirmation_Date
    confirmation_date_leaf = evaluator.add_leaf(
        id="Confirmation_Date",
        desc="The confirmation vote occurred on February 3, 2025",
        parent=temporal_node,
        critical=True
    )
    await evaluator.verify(
        claim="The date of this roll call vote is February 3, 2025.",
        node=confirmation_date_leaf,
        sources=extracted.senate_vote_url,
        additional_instruction="Verify the vote date shown on the Senate roll call page is February 3, 2025.",
        extra_prerequisites=[senate_url_ok]
    )

    # Vote_Time
    vote_time_leaf = evaluator.add_leaf(
        id="Vote_Time",
        desc="The vote took place at 6:13 PM Eastern Time",
        parent=temporal_node,
        critical=True
    )
    await evaluator.verify(
        claim="The recorded vote time is 6:13 PM Eastern Time.",
        node=vote_time_leaf,
        sources=extracted.senate_vote_url,
        additional_instruction="Allow minor formatting variants like '6:13 p.m. ET' or '6:13 PM'. It must be Eastern Time.",
        extra_prerequisites=[senate_url_ok]
    )

    # Vote_Number
    vote_number_leaf = evaluator.add_leaf(
        id="Vote_Number",
        desc="The Senate roll call vote number must be 30",
        parent=temporal_node,
        critical=True
    )
    await evaluator.verify(
        claim="This is Senate Roll Call Vote Number 30.",
        node=vote_number_leaf,
        sources=extracted.senate_vote_url,
        additional_instruction="Look for 'Vote Number: 30' or similar explicit indicator on the page.",
        extra_prerequisites=[senate_url_ok]
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
        default_model=model,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_confirmation_info(),
        template_class=ConfirmationAnswerExtraction,
        extraction_name="confirmation_extraction"
    )

    # Record some helpful custom info
    evaluator.add_custom_info(
        info={
            "extracted_name": extracted.name,
            "senate_vote_url": extracted.senate_vote_url,
            "total_urls_found": len(extracted.all_urls or []),
        },
        info_type="extraction_overview",
        info_name="extraction_summary"
    )

    # Build verification tree and run checks
    await build_and_verify(evaluator, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()