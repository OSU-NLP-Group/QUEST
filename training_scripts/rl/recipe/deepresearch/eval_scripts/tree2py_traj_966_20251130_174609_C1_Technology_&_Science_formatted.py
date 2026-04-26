import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "gvtc_info"
TASK_DESCRIPTION = (
    "GVTC Communications is a fiber optic service provider in Texas. "
    "Identify the city where GVTC's headquarters is located, and name at least two Texas counties that are part of GVTC's service area."
)


class GVTCExtraction(BaseModel):
    company_reference: Optional[str] = None
    headquarters_city: Optional[str] = None
    counties: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


def prompt_extract_gvtc() -> str:
    return (
        "From the answer, extract the following fields about GVTC Communications:\n"
        "1) company_reference: The company name as referenced in the answer text (e.g., 'GVTC Communications', 'GVTC', or 'Guadalupe Valley Telephone Cooperative'). "
        "Return exactly what appears; if the answer does not clearly reference GVTC Communications, return null.\n"
        "2) headquarters_city: The city name where GVTC's headquarters is stated (city only, without the state). "
        "If the answer provides 'City, State' or 'City, TX', return only the city. If the city is not provided, return null.\n"
        "3) counties: A list of county names in Texas that the answer claims are part of GVTC's service area. "
        "Extract them exactly as they appear (e.g., 'Comal County', 'Kendall County'). If none are provided, return an empty list.\n"
        "4) source_urls: All URLs present in the answer that appear to support claims about the headquarters city or the service area counties. "
        "Include plain URLs and markdown links, and return the actual URLs. If none are present, return an empty list.\n"
        "Do not invent information; only extract what is explicitly in the answer."
    )


def _format_counties_for_claim(counties: List[str]) -> str:
    if not counties:
        return "none"
    return ", ".join(counties)


async def evaluate_answer(
    client: LLMClient,
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_gvtc(),
        template_class=GVTCExtraction,
        extraction_name="gvtc_extraction",
    )

    gvtc_node = evaluator.add_parallel(
        id="GVTC_Info",
        desc="Evaluate whether the answer identifies GVTC Communications' headquarters city and names at least two Texas counties in GVTC's service area.",
        parent=root,
        critical=True,
    )

    company_identity_node = evaluator.add_leaf(
        id="Company_Identity",
        desc="The answer addresses GVTC Communications (the referenced Texas fiber optic service provider).",
        parent=gvtc_node,
        critical=True,
    )
    company_identity_claim = (
        "The answer is about GVTC Communications (also referred to as GVTC or the Guadalupe Valley Telephone Cooperative), "
        "a fiber optic service provider in Texas."
    )
    await evaluator.verify(
        claim=company_identity_claim,
        node=company_identity_node,
        additional_instruction=(
            "Judge whether the answer is clearly addressing GVTC Communications (GVTC). "
            "Accept synonyms or shorthand such as 'GVTC' and references to the cooperative behind the brand. "
            "Focus on whether the company being discussed matches GVTC Communications."
        ),
    )

    hq_city_provided_node = evaluator.add_custom_node(
        result=bool(extracted.headquarters_city and extracted.headquarters_city.strip()),
        id="Headquarters_City_Provided",
        desc="The answer identifies the city where GVTC Communications' headquarters is located.",
        parent=gvtc_node,
        critical=True,
    )

    counties_list = extracted.counties or []
    service_area_counties_node = evaluator.add_custom_node(
        result=len([c for c in counties_list if c and c.strip()]) >= 2,
        id="Service_Area_Counties",
        desc="The answer names at least two Texas counties that are part of GVTC Communications' service area.",
        parent=gvtc_node,
        critical=True,
    )

    evaluator.add_custom_info(
        info={
            "extracted_company_reference": extracted.company_reference,
            "extracted_headquarters_city": extracted.headquarters_city,
            "extracted_counties": counties_list,
            "extracted_source_urls": extracted.source_urls,
            "num_counties_extracted": len(counties_list),
        },
        info_type="extraction_summary",
    )

    return evaluator.get_summary()