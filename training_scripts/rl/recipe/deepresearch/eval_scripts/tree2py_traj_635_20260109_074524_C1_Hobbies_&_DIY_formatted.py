import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "largest_craft_contact"
TASK_DESCRIPTION = (
    "I want to contact the largest arts and crafts retail chain in the United States (by number of store locations) "
    "to inquire about a product. What is the customer service phone number for this chain, and what are the hours "
    "when I can call their customer service line on a weekday in January 2026?"
)

# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class ChainInfo(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PhoneInfo(BaseModel):
    number: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HoursInfo(BaseModel):
    weekday_hours: Optional[str] = None
    timezone: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ContactExtraction(BaseModel):
    chain: Optional[ChainInfo] = None
    phone: Optional[PhoneInfo] = None
    hours: Optional[HoursInfo] = None


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_contact_info() -> str:
    return """
    Extract, from the provided answer text only, the following structured information related to contacting the largest arts and crafts retail chain in the United States (by number of store locations):

    1) chain:
       - name: The chain the answer claims is the largest by number of store locations in the U.S. (e.g., "Michaels", "Hobby Lobby", "JOANN").
       - sources: All URLs cited in the answer that support this identification (e.g., official company pages, Wikipedia pages, news articles showing store counts or stating "largest").

    2) phone:
       - number: The customer service phone number provided for the identified chain (e.g., a toll-free 1-800 number or a listed customer care number).
       - sources: All URLs in the answer that support or display this phone number (e.g., "Contact Us" page, customer service page).

    3) hours:
       - weekday_hours: The stated hours during which the customer service phone line is available on a weekday (Monday–Friday) in January 2026 (e.g., "Mon–Fri 8am–7pm CT").
       - timezone: If the timezone is mentioned, extract it (e.g., "CT", "ET", "PT"); otherwise return null.
       - sources: All URLs cited in the answer that support these customer service phone hours (e.g., help center page, customer service hours page).

    Rules:
    - Extract values exactly as stated in the answer; do not invent or infer missing information.
    - If a field is not present in the answer, return null for that field (or an empty list for sources).
    - For URLs, include only valid, complete URLs. Accept plain URLs or markdown links; extract the actual URLs.
    - Do not deduplicate or filter sources; include all cited URLs that relate to the specific field.
    """


# -----------------------------------------------------------------------------
# Verification Helpers
# -----------------------------------------------------------------------------
async def verify_largest_chain(
    evaluator: Evaluator,
    parent_node,
    extracted: ContactExtraction,
) -> None:
    """
    Verify identification of the largest arts and crafts retail chain (by number of store locations).
    """
    chain_name = (extracted.chain.name.strip() if extracted.chain and extracted.chain.name else "")
    chain_sources = (extracted.chain.sources if extracted.chain and extracted.chain.sources else [])

    # Leaf as per rubric: Largest_Chain_Identification
    largest_leaf = evaluator.add_leaf(
        id="Largest_Chain_Identification",
        desc="The solution must identify the craft store chain that has the highest number of retail locations in the United States among major craft store chains (based on verifiable store count data).",
        parent=parent_node,
        critical=True,
    )

    claim = f"The largest arts and crafts retail chain in the United States by number of store locations is {chain_name}."
    add_ins = (
        "Verify that the provided sources explicitly support that this chain is the largest by number of store locations. "
        "Accept phrasing like 'largest arts and crafts retailer/chain' or clearly higher store count than other major U.S. craft chains. "
        "Allow minor name variants (e.g., 'Michaels Stores' vs 'Michaels'). If no source provides explicit support, mark as not supported."
    )

    await evaluator.verify(
        claim=claim,
        node=largest_leaf,
        sources=chain_sources,
        additional_instruction=add_ins,
    )


async def verify_customer_service_phone(
    evaluator: Evaluator,
    parent_node,
    extracted: ContactExtraction,
) -> None:
    """
    Verify that a valid customer service phone number is provided for the identified chain.
    """
    chain_name = (extracted.chain.name.strip() if extracted.chain and extracted.chain.name else "")
    phone_number = (extracted.phone.number.strip() if extracted.phone and extracted.phone.number else "")
    phone_sources = (extracted.phone.sources if extracted.phone and extracted.phone.sources else [])

    # Leaf as per rubric: Customer_Service_Phone
    phone_leaf = evaluator.add_leaf(
        id="Customer_Service_Phone",
        desc="The solution must provide a valid customer service phone number for the craft store chain identified in the previous step.",
        parent=parent_node,
        critical=True,
    )

    claim = f"The phone number '{phone_number}' is a valid customer service phone number for {chain_name}."
    add_ins = (
        "Confirm that the provided number is explicitly described as customer service/customer care/support or a primary contact number for the chain, "
        "not a single local store's number. Allow formatting variations (e.g., hyphens, parentheses). "
        "If the sources do not explicitly state this number as customer service or general customer care for the chain, do not support."
    )

    await evaluator.verify(
        claim=claim,
        node=phone_leaf,
        sources=phone_sources,
        additional_instruction=add_ins,
    )


async def verify_service_hours(
    evaluator: Evaluator,
    parent_node,
    extracted: ContactExtraction,
) -> None:
    """
    Verify the weekday customer service phone availability hours for January 2026.
    """
    chain_name = (extracted.chain.name.strip() if extracted.chain and extracted.chain.name else "")
    hours_text = (extracted.hours.weekday_hours.strip() if extracted.hours and extracted.hours.weekday_hours else "")
    timezone = (extracted.hours.timezone.strip() if extracted.hours and extracted.hours.timezone else "")
    hours_sources = (extracted.hours.sources if extracted.hours and extracted.hours.sources else [])

    # Leaf as per rubric: Service_Hours
    hours_leaf = evaluator.add_leaf(
        id="Service_Hours",
        desc="The solution must provide the customer service phone availability hours for a weekday (Monday-Friday) in January 2026 for the craft store chain identified in the first step.",
        parent=parent_node,
        critical=True,
    )

    tz_part = f" ({timezone})" if timezone else ""
    claim = (
        f"On a weekday (Monday–Friday) in January 2026, the customer service phone line for {chain_name} is available during: {hours_text}{tz_part}."
    )
    add_ins = (
        "Verify that the hours apply to customer service phone availability for weekdays (Mon–Fri). "
        "If the source lists general Mon–Fri customer service hours without seasonal caveats, assume they apply to January 2026 except explicit holiday closures. "
        "Allow minor variations in time formatting (e.g., '8 am' vs '8:00 AM') and accept any provided timezone if stated."
    )

    await evaluator.verify(
        claim=claim,
        node=hours_leaf,
        sources=hours_sources,
        additional_instruction=add_ins,
    )


# -----------------------------------------------------------------------------
# Main Evaluation Entry
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the 'largest arts and crafts chain contact information' task.
    """
    # Initialize evaluator with a sequential root per rubric
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_contact_info(),
        template_class=ContactExtraction,
        extraction_name="contact_info",
    )

    # Build the verification tree according to rubric (sequential critical leaves)
    # 1) Largest chain identification
    await verify_largest_chain(evaluator, root, extracted)

    # 2) Customer service phone (auto-skips if step 1 fails due to sequential aggregation)
    await verify_customer_service_phone(evaluator, root, extracted)

    # 3) Service hours (auto-skips if prior step(s) fail)
    await verify_service_hours(evaluator, root, extracted)

    # Return standardized summary
    return evaluator.get_summary()