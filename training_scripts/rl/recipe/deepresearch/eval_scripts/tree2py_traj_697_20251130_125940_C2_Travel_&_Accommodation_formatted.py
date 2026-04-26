import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "yellowstone_winter_lodge_2025_2026"
TASK_DESCRIPTION = (
    "For travelers planning a winter trip to Yellowstone National Park during the 2025-2026 winter season who need to access their lodging by personal vehicle, "
    "identify which in-park winter lodge is accessible by automobile. Provide the lodge's name, its winter season opening date, and closing date for the 2025-2026 season, "
    "along with URL references supporting this information."
)


class IdentificationAccess(BaseModel):
    lodge_name: Optional[str] = None
    car_access_statement: Optional[str] = None
    identification_access_urls: List[str] = Field(default_factory=list)


class LodgeDates(BaseModel):
    opening_date: Optional[str] = None
    closing_date: Optional[str] = None
    dates_urls: List[str] = Field(default_factory=list)


class WinterLodgeExtraction(BaseModel):
    identification: Optional[IdentificationAccess] = None
    dates: Optional[LodgeDates] = None


def prompt_extract_winter_lodge() -> str:
    return (
        "From the provided answer, extract the in-park winter lodge in Yellowstone National Park that the answer asserts is accessible by personal vehicle (automobile) "
        "during the 2025-2026 winter season, along with the season dates and supporting URLs.\n\n"
        "Return a JSON object with two sections:\n"
        "1) identification:\n"
        "   - lodge_name: The lodge name claimed to be accessible by personal vehicle in winter (in-park only).\n"
        "   - car_access_statement: A quote or paraphrase from the answer explicitly stating that this lodge is accessible by personal vehicle/automobile/road in winter.\n"
        "   - identification_access_urls: All URLs cited that support the lodge identification and its automobile accessibility in winter. Extract actual URLs only; if none are present, return an empty list.\n"
        "2) dates:\n"
        "   - opening_date: The winter 2025-2026 opening date for the identified lodge, exactly as presented in the answer (preserve formatting, e.g., 'Dec 15, 2025'). If missing, return null.\n"
        "   - closing_date: The winter 2025-2026 closing date for the identified lodge, exactly as presented in the answer. If missing, return null.\n"
        "   - dates_urls: All URLs cited that specifically support the opening and closing dates for the 2025-2026 winter season. Extract actual URLs only; if none are present, return an empty list.\n\n"
        "Rules:\n"
        "- Only extract information explicitly present in the answer. Do not invent.\n"
        "- If multiple lodges are mentioned, choose the one explicitly stated as accessible by personal vehicle and in-park; otherwise pick the first stated as accessible by car.\n"
        "- If any field is missing, use null (for strings) or [] (for URL lists).\n"
        "- Extract full URLs (including protocol)."
    )


async def verify_identification_and_access(
    evaluator: Evaluator,
    parent_node,
    extracted: WinterLodgeExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Lodge_Identification_and_Access_By_Automobile",
        desc="Identify which in-park winter lodge is accessible by personal automobile during the 2025-2026 winter season.",
        parent=parent_node,
        critical=True,
    )

    ident = extracted.identification or IdentificationAccess()

    # Lodge name provided (existence check)
    evaluator.add_custom_node(
        result=bool(ident.lodge_name and ident.lodge_name.strip()),
        id="Lodge_Name_Provided",
        desc="Provide the lodge name that is accessible by personal automobile (in-park, winter 2025-2026 context).",
        parent=node,
        critical=True,
    )

    # Reference URLs exist for identification/access
    evaluator.add_custom_node(
        result=bool(ident.identification_access_urls and len(ident.identification_access_urls) > 0),
        id="Reference_URL_For_Identification_Access",
        desc="Provide at least one URL reference supporting the lodge identification and its automobile accessibility in winter.",
        parent=node,
        critical=True,
    )

    # Automobile accessibility explicitly stated in the answer (simple verify on the answer text)
    auto_stmt_leaf = evaluator.add_leaf(
        id="Automobile_Accessibility_Stated",
        desc="Explicitly state that the identified lodge is accessible by personal vehicle/automobile in winter.",
        parent=node,
        critical=True,
    )
    # Build a claim evaluated against the answer text
    lodge_name_for_claim = ident.lodge_name or "the identified lodge"
    claim_answer_level = (
        f"The answer explicitly states that {lodge_name_for_claim} is accessible by personal vehicle (automobile) during the winter season."
    )
    await evaluator.verify(
        claim=claim_answer_level,
        node=auto_stmt_leaf,
        additional_instruction=(
            "Judge based only on the answer text. Accept synonyms such as 'car', 'personal vehicle', 'automobile', "
            "'open to wheeled vehicles', 'road-accessible', 'plowed road open to vehicles'. "
            "It must be clear that personal vehicles can reach the in-park lodge in winter (not just by snowcoach or snowmobile)."
        ),
    )

    # Additionally verify the accessibility claim against the provided identification/access URLs
    access_support_leaf = evaluator.add_leaf(
        id="Automobile_Access_Supported_By_URLs",
        desc="The lodge's winter automobile accessibility is supported by the cited identification/access URLs.",
        parent=node,
        critical=True,
    )
    claim_sources_level = (
        f"{lodge_name_for_claim} is accessible by personal vehicle (automobile) during the winter season in Yellowstone National Park."
    )
    await evaluator.verify(
        claim=claim_sources_level,
        node=access_support_leaf,
        sources=ident.identification_access_urls,
        additional_instruction=(
            "Use the provided URLs to confirm that personal vehicles (automobiles) can reach the lodge in winter via open roads. "
            "Accept phrasing like 'open to wheeled vehicles', 'road remains open', 'drive-in access'. "
            "Ensure the lodge is in-park and that the winter context is applicable."
        ),
    )


async def verify_operational_dates(
    evaluator: Evaluator,
    parent_node,
    extracted: WinterLodgeExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Lodge_Operational_Dates_2025_2026",
        desc="Provide the winter season opening and closing dates for the identified lodge for the 2025-2026 season, with supporting URL reference(s).",
        parent=parent_node,
        critical=True,
    )

    dates = extracted.dates or LodgeDates()
    ident = extracted.identification or IdentificationAccess()
    lodge_name_for_claim = ident.lodge_name or "the identified lodge"

    # Reference URLs exist for dates
    evaluator.add_custom_node(
        result=bool(dates.dates_urls and len(dates.dates_urls) > 0),
        id="Reference_URL_For_Dates",
        desc="Provide at least one URL reference supporting the opening and closing dates for winter 2025-2026.",
        parent=node,
        critical=True,
    )

    # Opening date verification against cited URLs
    open_leaf = evaluator.add_leaf(
        id="Opening_Date",
        desc="State the winter 2025-2026 opening date for the identified lodge.",
        parent=node,
        critical=True,
    )
    opening = dates.opening_date or ""
    claim_opening = (
        f"The winter 2025-2026 opening date for {lodge_name_for_claim} is '{opening}'."
        if opening
        else f"The answer provides the winter 2025-2026 opening date for {lodge_name_for_claim}."
    )
    await evaluator.verify(
        claim=claim_opening,
        node=open_leaf,
        sources=dates.dates_urls,
        additional_instruction=(
            "Verify in the provided URLs that the stated opening date for the lodge's winter 2025-2026 season matches. "
            "Allow minor format variations (e.g., 'Dec 15, 2025' vs 'December 15, 2025')."
        ),
    )

    # Closing date verification against cited URLs
    close_leaf = evaluator.add_leaf(
        id="Closing_Date",
        desc="State the winter 2025-2026 closing date for the identified lodge.",
        parent=node,
        critical=True,
    )
    closing = dates.closing_date or ""
    claim_closing = (
        f"The winter 2025-2026 closing date for {lodge_name_for_claim} is '{closing}'."
        if closing
        else f"The answer provides the winter 2025-2026 closing date for {lodge_name_for_claim}."
    )
    await evaluator.verify(
        claim=claim_closing,
        node=close_leaf,
        sources=dates.dates_urls,
        additional_instruction=(
            "Verify in the provided URLs that the stated closing date for the lodge's winter 2025-2026 season matches. "
            "Allow minor format variations."
        ),
    )


async def evaluate_answer(
    client: LLMClient,
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_winter_lodge(),
        template_class=WinterLodgeExtraction,
        extraction_name="winter_lodge_extraction",
    )

    # Create a critical task node (since initialize creates a non-critical root)
    task_node = evaluator.add_sequential(
        id="Winter_Lodge_Planning_Task",
        desc="Identify the in-park winter lodge (2025-2026 season) accessible by personal automobile and provide its opening/closing dates with supporting URLs.",
        parent=root,
        critical=True,
    )

    await verify_identification_and_access(evaluator, task_node, extracted)
    await verify_operational_dates(evaluator, task_node, extracted)

    return evaluator.get_summary()