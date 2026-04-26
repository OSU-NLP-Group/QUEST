import asyncio
import logging
from typing import Optional, Dict, Any

from pydantic import BaseModel

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "dollywood_2026_adult_1day"
TASK_DESCRIPTION = (
    "What is the starting price for an adult 1-day ticket to Dollywood theme park for the 2026 season (before taxes), "
    "what age range qualifies for adult pricing, and what is the official Dollywood website URL where this ticket "
    "pricing information can be verified?"
)


class DollywoodTicketExtraction(BaseModel):
    adult_starting_price: Optional[str] = None
    adult_age_range: Optional[str] = None
    source_url: Optional[str] = None


def prompt_extract_ticket_info() -> str:
    return """
Extract the following fields from the provided answer as they are written (do not infer or invent values):
1) adult_starting_price: The starting price for an adult 1-day Dollywood theme park ticket for the 2026 season, before taxes. Return the text exactly as stated (e.g., "$99", "from $99", "$99+ tax").
2) adult_age_range: The age range that qualifies for adult pricing at Dollywood (e.g., "ages 10-61", "10-61", "10+"). Return it exactly as stated.
3) source_url: A single URL to Dollywood's official website (must be on dollywood.com) where this ticket pricing information can be verified. If multiple URLs are provided, pick the single most directly relevant page about Dollywood Theme Park 1-Day ticket pricing. If no URL is present in the answer text, return null.

If any of these are not explicitly present in the answer text, set the field to null.
Only extract URLs that are explicitly mentioned. Do not invent URLs.
"""


async def verify_ticket_information(evaluator: Evaluator, root_node, extracted: DollywoodTicketExtraction) -> None:
    # Create the main critical node as per rubric
    main_node = evaluator.add_parallel(
        id="Complete_Dollywood_Ticket_Information",
        desc="Provides both the starting price for adult 1-day Dollywood theme park tickets for the 2026 season and the age range that qualifies for adult pricing",
        parent=root_node,
        critical=True
    )

    # 1) Official source reference (critical)
    # If no source URL extracted, directly fail this critical check to gate subsequent verifications.
    if not extracted.source_url or not isinstance(extracted.source_url, str) or extracted.source_url.strip() == "":
        evaluator.add_custom_node(
            result=False,
            id="Official_Source_Reference",
            desc="Provides a reference URL from Dollywood's official website (dollywood.com) that supports the ticket pricing information",
            parent=main_node,
            critical=True
        )
    else:
        official_src_node = evaluator.add_leaf(
            id="Official_Source_Reference",
            desc="Provides a reference URL from Dollywood's official website (dollywood.com) that supports the ticket pricing information",
            parent=main_node,
            critical=True
        )
        claim_official = (
            "The provided webpage URL is hosted on dollywood.com (the official Dollywood website) and is a page related "
            "to Dollywood Theme Park ticketing/pricing where adult 1-day ticket pricing information can be found or verified."
        )
        await evaluator.verify(
            claim=claim_official,
            node=official_src_node,
            sources=extracted.source_url,
            additional_instruction=(
                "Use both the URL string and the page content/screenshot. Confirm the domain is dollywood.com and that "
                "the page is about Dollywood Theme Park tickets (not Dollywood's Splash Country or unrelated pages). "
                "Ticket/pricing pages, buy tickets pages, or ticket-selection pages are acceptable as long as adult ticket pricing "
                "can be verified there."
            ),
        )

    # 2) Adult ticket starting price (critical)
    price_node = evaluator.add_leaf(
        id="Adult_Ticket_Starting_Price",
        desc="States the starting price for adult 1-day Dollywood theme park tickets for the 2026 season (before tax)",
        parent=main_node,
        critical=True
    )
    price_text = extracted.adult_starting_price or "None"
    claim_price = (
        f"The starting price for a 1-day Adult ticket to Dollywood Theme Park for the 2026 season (before taxes) is {price_text}."
    )
    await evaluator.verify(
        claim=claim_price,
        node=price_node,
        sources=extracted.source_url if extracted.source_url else None,
        additional_instruction=(
            "Verify the price on the cited official page. It must refer to Dollywood Theme Park adult 1-day admission "
            "and reflect a 'starting at' base price before taxes/fees. If the page lists dynamic date-based pricing, "
            "use the 'starting at' figure for the 2026 season. If the provided URL is missing or not on dollywood.com, "
            "or if the page does not show adult 1-day pricing for the 2026 season, mark as not supported."
        ),
    )

    # 3) Adult age range (critical)
    age_node = evaluator.add_leaf(
        id="Adult_Age_Range",
        desc="States the age range that qualifies for adult pricing at Dollywood",
        parent=main_node,
        critical=True
    )
    age_text = extracted.adult_age_range or "None"
    claim_age = f"At Dollywood, the Adult ticket age range is {age_text}."
    await evaluator.verify(
        claim=claim_age,
        node=age_node,
        sources=extracted.source_url if extracted.source_url else None,
        additional_instruction=(
            "Verify on the cited official page that the stated ages correspond to the Adult category for Dollywood Theme Park tickets. "
            "Allow reasonable formatting variants (e.g., 'Ages 10-61', '10–61', '10+'). Ensure it refers to the Theme Park "
            "(not the water park) and the adult pricing category."
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_ticket_info(),
        template_class=DollywoodTicketExtraction,
        extraction_name="dollywood_ticket_info"
    )

    # Build verification tree and verify
    await verify_ticket_information(evaluator, root, extracted)

    return evaluator.get_summary()