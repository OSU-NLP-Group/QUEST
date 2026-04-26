import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "leed_sf_address_2020"
TASK_DESCRIPTION = """
A commercial real estate website published an article in December 2020 listing the top 10 LEED-certified buildings in San Francisco. One building mentioned in that article is a 25-story Class A office building in the Financial District that has achieved LEED Platinum certification. This building is located within one block of the Montgomery Street BART/Muni station and offers amenities including a conference center and bicycle storage. What is the street address of this building?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BuildingExtraction(BaseModel):
    """
    Structured data extracted from the answer.
    """
    address: Optional[str] = None  # The street address (street number + street name) claimed in the final answer
    building_name: Optional[str] = None  # The building name if provided
    article_url: Optional[str] = None  # URL to the December 2020 CRE article listing top 10 LEED-certified SF buildings
    urls: List[str] = Field(default_factory=list)  # All citation/reference URLs present in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_building_info() -> str:
    return """
    Extract the structured information about the building and sources from the provided answer.

    You must extract exactly and only what is explicitly stated in the answer text. Do not invent or infer.

    Required fields:
    1) address: The final street address that the answer claims for the building (include the street number and street name; omit suite numbers or floor numbers if present).
    2) building_name: The building name as written in the answer, if any; otherwise null.
    3) article_url: The URL of the December 2020 article on a commercial real estate (CRE) website that lists the top 10 LEED-certified buildings in San Francisco, if the answer provides it; otherwise null.
    4) urls: Collect all explicit URLs that the answer presents as citations or references for any of the building facts (including the article_url if it appears in the answer).

    Rules for URL extraction:
    - Extract only URLs explicitly present in the answer (plain or markdown).
    - Include complete URLs (prepend http:// if missing).
    - If the answer provides no URLs, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_nonempty(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for u in urls:
        if not u:
            continue
        v = u.strip()
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def _entity_phrase(address: Optional[str], name: Optional[str]) -> str:
    """
    Build a flexible phrase to identify the building using either address or name.
    Prefer address; if missing, fall back to name.
    If both are present, include both with an 'or' to allow flexible matching.
    """
    addr = (address or "").strip()
    nm = (name or "").strip()

    if addr and nm:
        return f"the building located at '{addr}' (also known as '{nm}')"
    if addr:
        return f"the building located at '{addr}'"
    if nm:
        return f"the building named '{nm}'"
    # Fallback generic phrase (should be rare; most verifications will then fail for lack of evidence)
    return "the building in question"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_building(evaluator: Evaluator, root_node, ex: BuildingExtraction) -> None:
    """
    Build the verification tree and run checks according to the rubric.
    """

    # Consolidate sources
    all_urls = _unique_nonempty((ex.urls or []) + ([ex.article_url] if ex.article_url else []))
    article_urls = _unique_nonempty([ex.article_url] if ex.article_url else [])

    # Root-level sequential structure per rubric
    # 1) Street_Address_Provided (we break into existence + correctness)
    street_seq = evaluator.add_sequential(
        id="Street_Address_Provided",
        desc="Provides the correct street address (street name and number) with supporting reference URL",
        parent=root_node,
        critical=True
    )

    # 1.a) Existence and source presence (critical gate)
    evaluator.add_custom_node(
        result=bool(ex.address and ex.address.strip()) and (len(all_urls) > 0),
        id="address_and_source_present",
        desc="Street address string is provided and at least one reference URL is present in the answer",
        parent=street_seq,
        critical=True
    )

    # 1.b) Address correctness supported by at least one cited page
    addr_leaf = evaluator.add_leaf(
        id="address_correctness_verified",
        desc="At least one cited source page explicitly shows the claimed street address",
        parent=street_seq,
        critical=True
    )
    addr_claim = f"This webpage explicitly shows the building's street address as '{(ex.address or '').strip()}'. Minor formatting differences like 'St' vs 'Street' or the presence of city/state/ZIP on the page are acceptable; ignore suite or floor numbers."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_leaf,
        sources=all_urls,
        additional_instruction="Focus on whether the page displays the same street number and street name. Minor punctuation or abbreviation differences are acceptable."
    )

    # 2) Building_Requirements_Verified (parallel group, all critical subchecks)
    reqs_group = evaluator.add_parallel(
        id="Building_Requirements_Verified",
        desc="The building at the provided address meets all specified requirements",
        parent=root_node,
        critical=True
    )

    # 2.a) Classification + LEED + Article (break into atomic leaves under a critical parallel node)
    class_article_group = evaluator.add_parallel(
        id="Classification_And_Article_Verified",
        desc="Building is Class A office, has LEED Platinum, and was in the December 2020 CRE article of top 10 LEED-certified SF buildings",
        parent=reqs_group,
        critical=True
    )

    # Class A office
    class_a_leaf = evaluator.add_leaf(
        id="class_a_office_verified",
        desc="Building is confirmed as a Class A office building",
        parent=class_article_group,
        critical=True
    )
    class_a_claim = f"The page shows that {_entity_phrase(ex.address, ex.building_name)} is a Class A office building."
    await evaluator.verify(
        claim=class_a_claim,
        node=class_a_leaf,
        sources=all_urls,
        additional_instruction="Accept variants like 'Class A', 'Class-A', or 'Class A office tower'. The building identity can be matched by address or by name."
    )

    # LEED Platinum
    leed_platinum_leaf = evaluator.add_leaf(
        id="leed_platinum_verified",
        desc="Building is confirmed as having LEED Platinum certification",
        parent=class_article_group,
        critical=True
    )
    leed_claim = f"The page states that {_entity_phrase(ex.address, ex.building_name)} has achieved LEED Platinum certification."
    await evaluator.verify(
        claim=leed_claim,
        node=leed_platinum_leaf,
        sources=all_urls,
        additional_instruction="Accept variants like 'LEED Platinum Certified', 'LEED-CS Platinum', 'LEED v4 Platinum', or explicit mention of 'LEED Platinum'."
    )

    # December 2020 article + mention
    article_leaf = evaluator.add_leaf(
        id="article_dec2020_verified",
        desc="December 2020 CRE article lists top 10 LEED-certified SF buildings and includes this building",
        parent=class_article_group,
        critical=True
    )
    article_claim = (
        f"This article is from December 2020, is on a commercial real estate website, and is about the 'top 10 LEED-certified buildings in San Francisco'. "
        f"It includes {_entity_phrase(ex.address, ex.building_name)} (match by building name or by address)."
    )
    await evaluator.verify(
        claim=article_claim,
        node=article_leaf,
        sources=article_urls if article_urls else all_urls,
        additional_instruction="Confirm three things on the page: (1) publication date in December 2020; (2) topic/title indicates top 10 LEED-certified buildings in San Francisco; (3) the building is included (match by name or the same street address). The site should be a commercial real estate website (brokerage, listing platform, CRE news/publication)."
    )

    # 2.b) Physical specifications: 25 stories + Financial District
    phys_group = evaluator.add_parallel(
        id="Physical_Specifications_Verified",
        desc="Building is exactly 25 stories tall and located in San Francisco's Financial District",
        parent=reqs_group,
        critical=True
    )

    stories_leaf = evaluator.add_leaf(
        id="stories_25_verified",
        desc="Building is exactly 25 stories tall",
        parent=phys_group,
        critical=True
    )
    stories_claim = f"The page indicates that {_entity_phrase(ex.address, ex.building_name)} has exactly 25 stories (i.e., 25 floors)."
    await evaluator.verify(
        claim=stories_claim,
        node=stories_leaf,
        sources=all_urls,
        additional_instruction="Accept wording like '25-story', '25 stories', or '25 floors'."
    )

    fidi_leaf = evaluator.add_leaf(
        id="financial_district_verified",
        desc="Building is located in San Francisco's Financial District",
        parent=phys_group,
        critical=True
    )
    fidi_claim = f"The page indicates that {_entity_phrase(ex.address, ex.building_name)} is located in San Francisco's Financial District."
    await evaluator.verify(
        claim=fidi_claim,
        node=fidi_leaf,
        sources=all_urls,
        additional_instruction="Accept variants like 'Financial District' or 'FiDi'."
    )

    # 2.c) Location + amenities: within one block of Montgomery St. station + conference center + bicycle storage
    loc_amen_group = evaluator.add_parallel(
        id="Location_And_Amenities_Verified",
        desc="Within one block of Montgomery Street BART/Muni station and has both a conference center and bicycle storage",
        parent=reqs_group,
        critical=True
    )

    mont_leaf = evaluator.add_leaf(
        id="within_one_block_montgomery_verified",
        desc="Within one block of Montgomery Street BART/Muni station",
        parent=loc_amen_group,
        critical=True
    )
    mont_claim = f"The page states that {_entity_phrase(ex.address, ex.building_name)} is within one block of the Montgomery Street BART/Muni station."
    await evaluator.verify(
        claim=mont_claim,
        node=mont_leaf,
        sources=all_urls,
        additional_instruction="Accept phrases like 'within one block', 'adjacent to', 'steps from', 'across the street from', or 'next to' Montgomery Street BART/Muni station."
    )

    conf_leaf = evaluator.add_leaf(
        id="conference_center_verified",
        desc="Building offers a conference center amenity",
        parent=loc_amen_group,
        critical=True
    )
    conf_claim = f"The page lists a conference center (or equivalent conference/meeting facilities) as an amenity for {_entity_phrase(ex.address, ex.building_name)}."
    await evaluator.verify(
        claim=conf_claim,
        node=conf_leaf,
        sources=all_urls,
        additional_instruction="Accept equivalents like 'conference center', 'tenant conference center', 'conference facilities', or 'meeting rooms'."
    )

    bike_leaf = evaluator.add_leaf(
        id="bicycle_storage_verified",
        desc="Building offers bicycle storage amenity",
        parent=loc_amen_group,
        critical=True
    )
    bike_claim = f"The page lists bicycle storage (bike room) as an amenity for {_entity_phrase(ex.address, ex.building_name)}."
    await evaluator.verify(
        claim=bike_claim,
        node=bike_leaf,
        sources=all_urls,
        additional_instruction="Accept synonyms like 'bicycle storage', 'bike storage', 'secure bike room', or 'bicycle parking'."
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
    Evaluate an answer for the LEED-certified SF building street address task.
    """
    # Initialize evaluator with a sequential root as the rubric requires ordered gating
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
        default_model=model,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_building_info(),
        template_class=BuildingExtraction,
        extraction_name="building_extraction",
    )

    # Build verification tree and run checks
    await verify_building(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()