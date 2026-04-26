import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "cma59_venue_nashville_2025"
TASK_DESCRIPTION = """
Identify the multi-purpose indoor arena venue in Nashville, Tennessee that hosted the 59th Annual CMA Awards in November 2025. For this venue, provide: (1) The official name of the venue, (2) The venue's street address, (3) The venue's seating capacity for full-house concert configuration, and (4) A reference URL from an official source that verifies this information.
"""


class VenueDetails(BaseModel):
    venue_name: Optional[str] = None
    street_address: Optional[str] = None
    concert_capacity_full: Optional[str] = None
    official_source_urls: List[str] = Field(default_factory=list)


def prompt_extract_venue_details() -> str:
    return """
    Extract the single venue identified in the answer that satisfies the task: a multi-purpose indoor arena in Nashville, Tennessee that hosted the 59th Annual CMA Awards in November 2025.
    Return a JSON object with the following fields:
    - venue_name: The official name of the venue (string).
    - street_address: The street address of the venue as provided in the answer (string).
    - concert_capacity_full: The seating capacity for the full-house concert configuration as stated in the answer (string; allow ranges or phrasing like "up to 19,000").
    - official_source_urls: An array of URL strings that the answer cites as official sources (e.g., the venue’s official website, CMA’s official site, or municipal/government pages) that verify the venue identity and details.
    If any required field is missing in the answer, set it to null; if there are no official sources, return an empty array for official_source_urls.
    """


async def verify_constraints(
    evaluator: Evaluator,
    parent_node,
    details: VenueDetails,
) -> None:
    constraints_node = evaluator.add_parallel(
        id="VenueMeetsConstraints",
        desc="The identified venue satisfies all stated identification constraints.",
        parent=parent_node,
        critical=True,
    )

    name = details.venue_name or ""
    urls = details.official_source_urls

    located_leaf = evaluator.add_leaf(
        id="LocatedInNashvilleTN",
        desc="The venue is located in Nashville, Tennessee.",
        parent=constraints_node,
        critical=True,
    )
    arena_leaf = evaluator.add_leaf(
        id="IsMultiPurposeIndoorArena",
        desc="The venue is a multi-purpose indoor arena.",
        parent=constraints_node,
        critical=True,
    )
    hosted_leaf = evaluator.add_leaf(
        id="Hosted59thCMAAwardsInNov2025",
        desc="The venue hosted the 59th Annual CMA Awards in November 2025.",
        parent=constraints_node,
        critical=True,
    )

    claims_and_sources = [
        (
            f"The venue '{name}' is located in Nashville, Tennessee.",
            urls,
            located_leaf,
            "Confirm via the provided official source URLs that the venue is in Nashville, Tennessee. Accept exact address statements such as 'Nashville, TN'.",
        ),
        (
            f"The venue '{name}' is a multi-purpose indoor arena.",
            urls,
            arena_leaf,
            "Confirm via the official sources that the venue is an indoor arena and multi-purpose (synonyms like multiuse/multi-purpose are acceptable).",
        ),
        (
            f"The venue '{name}' hosted the 59th Annual CMA Awards in November 2025.",
            urls,
            hosted_leaf,
            "Prefer CMA’s official site or the venue’s official announcements to verify that the 59th CMA Awards (2025) took place at this venue in November 2025.",
        ),
    ]
    await evaluator.batch_verify(claims_and_sources)


async def verify_required_outputs(
    evaluator: Evaluator,
    parent_node,
    details: VenueDetails,
) -> None:
    required_node = evaluator.add_parallel(
        id="RequiredVenueOutputsProvided",
        desc="All required venue details are provided for the identified venue.",
        parent=parent_node,
        critical=True,
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(details.venue_name and details.venue_name.strip()),
        id="OfficialVenueNameProvided",
        desc="The answer provides the venue’s official name.",
        parent=required_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(details.street_address and details.street_address.strip()),
        id="StreetAddressProvided",
        desc="The answer provides the venue’s street address.",
        parent=required_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(details.concert_capacity_full and details.concert_capacity_full.strip()),
        id="FullHouseConcertCapacityProvided",
        desc="The answer provides the venue’s seating capacity for full-house concert configuration.",
        parent=required_node,
        critical=True,
    )
    # Ensure at least one official URL exists (critical)
    evaluator.add_custom_node(
        result=bool(details.official_source_urls),
        id="OfficialSourceURLExists",
        desc="At least one official source URL is provided.",
        parent=required_node,
        critical=True,
    )

    # Check that provided official URLs are indeed official sources
    official_url_leaf = evaluator.add_leaf(
        id="OfficialSourceReferenceURLProvided",
        desc="At least one reference URL from an official source is provided, and the provided official URL(s) verify the key claims (venue identity and the required details).",
        parent=required_node,
        critical=True,
    )
    urls = details.official_source_urls
    name = details.venue_name or ""
    address = details.street_address or ""
    capacity = details.concert_capacity_full or ""
    # We verify that official URLs cover the identity and the required details collectively (run multiple checks)
    # First, verify that at least one URL is official in nature
    await evaluator.verify(
        claim="At least one of the provided URLs is an official source (e.g., venue’s official website, CMA’s official site, or a municipal/government page).",
        node=official_url_leaf,
        sources=urls,
        additional_instruction="Judge officialness by domain (e.g., venue official domain, cmaawards.com, government domains). If all URLs are non-official, fail.",
    )

    # Verify each required detail is supported by official sources (critical leaves)
    name_verified_leaf = evaluator.add_leaf(
        id="OfficialSourceValidatesName",
        desc="Official source(s) support the venue’s official name.",
        parent=required_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official name of the venue is '{name}'.",
        node=name_verified_leaf,
        sources=urls,
        additional_instruction="Verify the venue name exactly or with minor variations as shown on official sources.",
    )

    address_verified_leaf = evaluator.add_leaf(
        id="OfficialSourceValidatesAddress",
        desc="Official source(s) support the venue’s street address.",
        parent=required_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue’s street address is '{address}'.",
        node=address_verified_leaf,
        sources=urls,
        additional_instruction="Confirm the full street address on an official source page. Allow minor formatting differences (e.g., abbreviations like Ave./Avenue).",
    )

    capacity_verified_leaf = evaluator.add_leaf(
        id="OfficialSourceValidatesCapacity",
        desc="Official source(s) support the venue’s full-house concert seating capacity.",
        parent=required_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue’s full-house concert seating capacity is '{capacity}'.",
        node=capacity_verified_leaf,
        sources=urls,
        additional_instruction="Check official pages or documents for concert configuration capacity. Allow phrasing like 'up to' or ranges; accept close equivalents if clearly stated by official sources.",
    )


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

    details = await evaluator.extract(
        prompt=prompt_extract_venue_details(),
        template_class=VenueDetails,
        extraction_name="venue_details",
    )

    # Top-level critical sequential node per rubric
    task_node = evaluator.add_sequential(
        id="VenueIdentificationTask",
        desc="Identify the multi-purpose indoor arena venue in Nashville, Tennessee that hosted the 59th Annual CMA Awards in November 2025, and provide its official name, street address, full-house concert seating capacity, and official-source reference URL(s).",
        parent=root,
        critical=True,
    )

    await verify_constraints(evaluator, task_node, details)
    await verify_required_outputs(evaluator, task_node, details)

    return evaluator.get_summary()