import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "tsa_touchless_id_ny_airport"
TASK_DESCRIPTION = "Which New York metropolitan area airport appears on the TSA PreCheck Touchless ID participating airport lists for both American Airlines and Alaska Airlines? Provide the airport's three-letter IATA code and include the official TSA Touchless ID webpage as your reference."


class AirportSelection(BaseModel):
    airport_name: Optional[str] = None
    iata_code: Optional[str] = None
    american_airlines_urls: List[str] = Field(default_factory=list)
    alaska_airlines_urls: List[str] = Field(default_factory=list)
    tsa_touchless_id_url: Optional[str] = None
    digital_id_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)


def prompt_extract_airport_info() -> str:
    return """
    Extract the single airport identified in the answer and all relevant URLs the answer cites.

    Required fields:
    - airport_name: The name of the airport identified in the answer (e.g., "LaGuardia Airport", "JFK", "EWR"). If multiple airports are mentioned, pick the one the answer claims is on both lists.
    - iata_code: The three-letter IATA code of the identified airport (e.g., "LGA", "JFK", "EWR"). If not explicitly provided, return null.

    URL fields (extract only URLs explicitly present in the answer text; include full URLs):
    - american_airlines_urls: All URLs from American Airlines official site that are cited as the "TSA PreCheck Touchless ID participating airports list" or equivalent page(s) showing participating airports.
    - alaska_airlines_urls: All URLs from Alaska Airlines official site that are cited as the "TSA PreCheck Touchless ID participating airports list" or equivalent page(s) showing participating airports.
    - tsa_touchless_id_url: The URL for the official TSA webpage about "Touchless ID" or TSA PreCheck digital ID program. Must be a tsa.gov URL if provided. If multiple TSA URLs are cited, pick the most directly relevant "Touchless ID" page; otherwise return null.
    - digital_id_urls: Any URLs cited that mention TSA Digital ID readers and/or the acceptance of New York mobile driver's license (NY MiD) at TSA checkpoints.
    - location_urls: Any URLs cited that can support that the airport is in the New York metropolitan area.

    If any of the above URLs are not present in the answer, return an empty list for that field (or null for tsa_touchless_id_url).
    """


def is_valid_iata(code: Optional[str]) -> bool:
    if not code:
        return False
    c = code.strip()
    if len(c) != 3:
        return False
    return c.isalpha()


async def verify_airport_requirements(
    evaluator: Evaluator,
    parent_node,
    info: AirportSelection,
) -> None:
    req_node = evaluator.add_parallel(
        id="Airport_Requirements",
        desc="The identified airport satisfies all geographic, digital ID acceptance, and airline participation criteria",
        parent=parent_node,
        critical=True,
    )

    # Geographic Location
    geo_node = evaluator.add_leaf(
        id="Geographic_Location",
        desc="The airport is located in the New York metropolitan area",
        parent=req_node,
        critical=True,
    )
    geo_sources: List[str] = []
    if info.location_urls:
        geo_sources.extend(info.location_urls)
    else:
        # Fallback: use airline pages if present (they may indicate city/airport names)
        geo_sources.extend(info.american_airlines_urls)
        geo_sources.extend(info.alaska_airlines_urls)
    claim_geo = f"The airport {info.airport_name or 'the identified airport'} (IATA {info.iata_code or 'unknown'}) is located in the New York metropolitan area."
    await evaluator.verify(
        claim=claim_geo,
        node=geo_node,
        sources=geo_sources if len(geo_sources) > 0 else None,
        additional_instruction="Accept airports commonly recognized as part of the NYC metro area (e.g., JFK, LGA, EWR). Allow reasonable naming variations.",
    )

    # Digital ID Acceptance (NY MiD through TSA Digital ID at checkpoints with digital ID readers)
    digital_id_node = evaluator.add_leaf(
        id="Digital_ID_Acceptance",
        desc="The airport accepts New York State mobile driver's licenses (NY MiD) through TSA's Digital ID program at checkpoints with digital ID readers",
        parent=req_node,
        critical=True,
    )
    did_sources: List[str] = []
    if info.digital_id_urls:
        did_sources.extend(info.digital_id_urls)
    if info.tsa_touchless_id_url:
        did_sources.append(info.tsa_touchless_id_url)
    claim_did = (
        f"At {info.airport_name or 'the identified airport'}, TSA's Digital ID program accepts New York State mobile driver's licenses (NY MiD) "
        f"at checkpoints equipped with digital ID readers."
    )
    await evaluator.verify(
        claim=claim_did,
        node=digital_id_node,
        sources=did_sources if len(did_sources) > 0 else None,
        additional_instruction="Look for explicit mention that New York Mobile ID (NY MiD) is accepted via TSA Digital ID and that the airport has TSA digital ID readers.",
    )

    # American Airlines participation - add gating existence check
    aa_sources_provided = evaluator.add_custom_node(
        result=bool(info.american_airlines_urls),
        id="AA_Sources_Provided",
        desc="American Airlines participating airports list URL(s) are provided",
        parent=req_node,
        critical=True,
    )
    aa_part_node = evaluator.add_leaf(
        id="American_Airlines_Participation",
        desc="The airport appears on the American Airlines TSA PreCheck Touchless ID participating airports list",
        parent=req_node,
        critical=True,
    )
    claim_aa = (
        f"The airport {info.airport_name or (info.iata_code or 'the identified airport')} appears on the American Airlines TSA PreCheck Touchless ID participating airports list."
    )
    await evaluator.verify(
        claim=claim_aa,
        node=aa_part_node,
        sources=info.american_airlines_urls if len(info.american_airlines_urls) > 0 else None,
        additional_instruction="Verify that the provided American Airlines page(s) explicitly list the airport among Touchless ID participating airports. Allow reasonable naming variations (e.g., city or terminal references).",
    )

    # Alaska Airlines participation - add gating existence check
    alaska_sources_provided = evaluator.add_custom_node(
        result=bool(info.alaska_airlines_urls),
        id="Alaska_Sources_Provided",
        desc="Alaska Airlines participating airports list URL(s) are provided",
        parent=req_node,
        critical=True,
    )
    alaska_part_node = evaluator.add_leaf(
        id="Alaska_Airlines_Participation",
        desc="The airport appears on the Alaska Airlines TSA PreCheck Touchless ID participating airports list",
        parent=req_node,
        critical=True,
    )
    claim_alaska = (
        f"The airport {info.airport_name or (info.iata_code or 'the identified airport')} appears on the Alaska Airlines TSA PreCheck Touchless ID participating airports list."
    )
    await evaluator.verify(
        claim=claim_alaska,
        node=alaska_part_node,
        sources=info.alaska_airlines_urls if len(info.alaska_airlines_urls) > 0 else None,
        additional_instruction="Verify that the provided Alaska Airlines page(s) explicitly list the airport among Touchless ID participating airports. Allow reasonable naming variations.",
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
) -> Dict[str, Any]:
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

    # Extraction
    info = await evaluator.extract(
        prompt=prompt_extract_airport_info(),
        template_class=AirportSelection,
        extraction_name="airport_selection",
    )

    # Build tree
    airport_node = evaluator.add_sequential(
        id="Airport_Identification",
        desc="Identify the New York metropolitan area airport that appears on TSA PreCheck Touchless ID participating lists for both American Airlines and Alaska Airlines",
        parent=root,
        critical=False,
    )

    # Requirements group
    await verify_airport_requirements(evaluator, airport_node, info)

    # IATA code format check
    iata_node = evaluator.add_custom_node(
        result=is_valid_iata(info.iata_code),
        id="IATA_Code_Format",
        desc="The airport's three-letter IATA code is provided",
        parent=airport_node,
        critical=True,
    )

    # Source documentation: restructure as a parallel group to ensure existence + correctness checks
    src_parent = evaluator.add_parallel(
        id="Source_Documentation",
        desc="Official TSA Touchless ID webpage URL is provided as reference",
        parent=airport_node,
        critical=True,
    )

    tsa_url_provided = evaluator.add_custom_node(
        result=bool(info.tsa_touchless_id_url),
        id="TSA_URL_Provided",
        desc="TSA Touchless ID webpage URL is provided",
        parent=src_parent,
        critical=True,
    )

    tsa_url_correct = evaluator.add_leaf(
        id="TSA_URL_Correct",
        desc="The provided URL is the official TSA 'Touchless ID' webpage",
        parent=src_parent,
        critical=True,
    )
    tsa_claim = "This webpage is the official Transportation Security Administration (TSA) 'Touchless ID' page describing TSA PreCheck digital ID capabilities."
    await evaluator.verify(
        claim=tsa_claim,
        node=tsa_url_correct,
        sources=info.tsa_touchless_id_url if info.tsa_touchless_id_url else None,
        additional_instruction="Confirm the page is on tsa.gov and explicitly about 'Touchless ID' or TSA PreCheck Digital ID. If the URL is missing or the page is irrelevant, mark as not supported.",
    )

    return evaluator.get_summary()