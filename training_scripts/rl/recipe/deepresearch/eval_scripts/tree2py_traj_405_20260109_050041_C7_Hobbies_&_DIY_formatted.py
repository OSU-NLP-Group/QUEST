import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "craft_subscription_boxes"
TASK_DESCRIPTION = """
Find three different monthly craft subscription box services designed for adults or teenagers (ages 12 and up) that are available in the United States. For each subscription box service, provide the following information:

1. Service Name: The name of the subscription box service
2. Official Website: The URL of the official website where customers can subscribe
3. Monthly Price: The cost for a month-to-month subscription (excluding shipping), which must be $40 or less
4. Target Age: The intended age group for the subscription
5. What's Included: Confirmation that the box includes all necessary materials and supplies to complete the projects
6. Instructions: Confirmation that instructions or tutorials are provided
7. US Shipping: Confirmation that the service ships to addresses in the United States

Each subscription service must offer a month-to-month subscription option (not only pre-paid multi-month packages) and must be independently operated (not different product lines from the same company).
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SubscriptionBox(BaseModel):
    service_name: Optional[str] = None
    official_website: Optional[str] = None
    monthly_price: Optional[str] = None  # Prefer text (e.g., "$39.99", "39", "Under $40")
    target_age: Optional[str] = None     # e.g., "Adults", "Teens 13+", "12+"
    includes_materials: Optional[str] = None  # yes/no/phrase
    instructions_provided: Optional[str] = None  # yes/no/phrase
    us_shipping: Optional[str] = None  # yes/no/phrase
    month_to_month_option: Optional[str] = None  # yes/no/phrase


class SubscriptionBoxesExtraction(BaseModel):
    boxes: List[SubscriptionBox] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_boxes() -> str:
    return """
    Extract all craft subscription box services mentioned in the answer. For each service, extract these fields exactly as stated in the answer:

    - service_name: The name of the subscription box service.
    - official_website: The URL of the official website where customers can subscribe (include full URL; if missing protocol, prepend http://).
    - monthly_price: The stated month-to-month price (excluding shipping), as text (e.g., "$39.99"). If not explicitly stated, return null.
    - target_age: The intended age group (e.g., "Adults", "Teens 12+", "12+", "Adults & Teens"). If not stated, return null.
    - includes_materials: Whether the box includes all necessary materials/supplies to complete the projects. Return a brief phrase or "yes"/"no" based on the answer; if unclear, return null.
    - instructions_provided: Whether instructions or tutorials are provided. Return a brief phrase or "yes"/"no"; if unclear, return null.
    - us_shipping: Whether the service ships to addresses in the United States. Return a brief phrase or "yes"/"no"; if unclear, return null.
    - month_to_month_option: Whether a month-to-month subscription option is available (not only prepaid multi-month). Return a brief phrase or "yes"/"no"; if unclear, return null.

    Return a JSON object with a 'boxes' array containing one object per service. Extract all services mentioned; do not invent data.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_domain(url: Optional[str]) -> Optional[str]:
    if not url or not isinstance(url, str):
        return None
    try:
        parsed = urlparse(url.strip())
        netloc = parsed.netloc.lower()
        if not netloc and parsed.path:
            # Handle cases like "example.com/path" when scheme missing
            temp = urlparse("http://" + url.strip())
            netloc = temp.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        if netloc.startswith("m."):
            netloc = netloc[2:]
        return netloc or None
    except Exception:
        return None


def is_nonempty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def select_first_three_valid(boxes: List[SubscriptionBox]) -> List[SubscriptionBox]:
    valid = [b for b in boxes if is_nonempty_str(b.service_name) and is_nonempty_str(b.official_website)]
    return valid[:3]


# --------------------------------------------------------------------------- #
# Verification logic per box                                                  #
# --------------------------------------------------------------------------- #
async def verify_box(
    evaluator: Evaluator,
    parent_node,
    box: SubscriptionBox,
    index: int
) -> None:
    """
    Build verification nodes for a single subscription box service and run verifications.
    """
    idx = index + 1
    box_node = evaluator.add_parallel(
        id=f"Box_{idx}",
        desc=f"{['First','Second','Third'][index]} craft subscription box service meets all per-service requirements",
        parent=parent_node,
        critical=False
    )

    # 1) Name - existence check (critical)
    name_exists = is_nonempty_str(box.service_name)
    evaluator.add_custom_node(
        result=name_exists,
        id=f"Box_{idx}_Name",
        desc=f"Provides the name of the {['first','second','third'][index]} subscription box service",
        parent=box_node,
        critical=True
    )

    # 2) Official Website - verify that URL is official and active, with subscription capability
    website_node = evaluator.add_leaf(
        id=f"Box_{idx}_Website",
        desc=f"Provides an active, official website URL where customers can subscribe for the {['first','second','third'][index]} service",
        parent=box_node,
        critical=True
    )
    claim_website = (
        f"The answer provides an active, official website URL where customers can subscribe to the "
        f"{box.service_name or 'subscription box service'}."
    )
    await evaluator.verify(
        claim=claim_website,
        node=website_node,
        sources=box.official_website,
        additional_instruction=(
            "Verify the provided URL leads to the official site or official store page for the service and shows a subscription or purchase option. "
            "Pages on platforms like Cratejoy are acceptable if they clearly represent the brand's official store page for subscribing."
        ),
    )

    # 3) Monthly Price <= $40 (excluding shipping)
    price_node = evaluator.add_leaf(
        id=f"Box_{idx}_Price",
        desc="Provides the month-to-month subscription price (excluding shipping) and it is $40 or less",
        parent=box_node,
        critical=True
    )
    if is_nonempty_str(box.monthly_price):
        claim_price = (
            f"The answer states the month-to-month subscription price (excluding shipping) is {box.monthly_price}, "
            f"and it is $40 or less."
        )
    else:
        claim_price = (
            "The answer explicitly states the month-to-month subscription price (excluding shipping), and it is $40 or less."
        )
    await evaluator.verify(
        claim=claim_price,
        node=price_node,
        sources=box.official_website,
        additional_instruction=(
            "On the website, check the month-to-month price (not prepaid bundles). The price must be $40 or less, excluding shipping. "
            "Accept minor formatting (e.g., $39, $39.99). If only prepaid multi-month packages exist (no monthly option), this should be considered incorrect."
        ),
    )

    # 4) Month-to-month option available
    m2m_node = evaluator.add_leaf(
        id=f"Box_{idx}_MonthToMonth",
        desc="Confirms a month-to-month subscription option is available (not only prepaid multi-month packages)",
        parent=box_node,
        critical=True
    )
    claim_m2m = (
        "The answer confirms that a month-to-month subscription option is available (not only prepaid multi-month packages)."
    )
    await evaluator.verify(
        claim=claim_m2m,
        node=m2m_node,
        sources=box.official_website,
        additional_instruction=(
            "Verify the presence of a monthly plan or wording such as 'month-to-month', 'monthly', 'pay monthly', or 'cancel anytime'. "
            "If only prepaid multi-month options are offered, this should fail."
        ),
    )

    # 5) Age suitability (adults or teens 12+)
    age_node = evaluator.add_leaf(
        id=f"Box_{idx}_Age",
        desc="Confirms the service is designed for adults or teenagers with minimum age 12+",
        parent=box_node,
        critical=True
    )
    claim_age = (
        "The answer confirms the service is designed for adults or teenagers ages 12+."
    )
    await evaluator.verify(
        claim=claim_age,
        node=age_node,
        sources=box.official_website,
        additional_instruction=(
            "Look for age guidance or target audience statements. Accept 'Adults', 'Teens', '12+', '13+', etc. "
            "If the service is clearly for children under 12, this should fail."
        ),
    )

    # 6) Materials included
    materials_node = evaluator.add_leaf(
        id=f"Box_{idx}_Materials",
        desc="Confirms the box includes all necessary materials and supplies needed to complete the projects",
        parent=box_node,
        critical=True
    )
    claim_materials = (
        "The answer confirms the box includes all necessary materials and supplies to complete the projects."
    )
    await evaluator.verify(
        claim=claim_materials,
        node=materials_node,
        sources=box.official_website,
        additional_instruction=(
            "Confirm that project-specific materials are included. Minor basic tools (e.g., scissors, glue) may be excluded. "
            "If the kit requires separate purchases of key project materials, this should fail."
        ),
    )

    # 7) Instructions provided
    instructions_node = evaluator.add_leaf(
        id=f"Box_{idx}_Instructions",
        desc="Confirms instructions or tutorials are provided",
        parent=box_node,
        critical=True
    )
    claim_instructions = (
        "The answer confirms instructions or tutorials are provided with the projects."
    )
    await evaluator.verify(
        claim=claim_instructions,
        node=instructions_node,
        sources=box.official_website,
        additional_instruction=(
            "Accept written instructions, printed booklets, online guides, or video tutorials. "
            "If there is no guidance to complete the projects, this should fail."
        ),
    )

    # 8) US shipping
    us_node = evaluator.add_leaf(
        id=f"Box_{idx}_US_Shipping",
        desc="Confirms the service ships to addresses in the United States",
        parent=box_node,
        critical=True
    )
    claim_us = (
        "The answer confirms the service ships to addresses in the United States."
    )
    await evaluator.verify(
        claim=claim_us,
        node=us_node,
        sources=box.official_website,
        additional_instruction=(
            "Look for shipping policy that includes the US. If shipping is limited to non-US only, or unclear, this should fail."
        ),
    )


# --------------------------------------------------------------------------- #
# Set-level requirements                                                      #
# --------------------------------------------------------------------------- #
def add_set_level_requirements(
    evaluator: Evaluator,
    parent_node,
    all_boxes: List[SubscriptionBox],
    selected_boxes: List[SubscriptionBox]
) -> None:
    set_node = evaluator.add_parallel(
        id="Set_Level_Requirements",
        desc="Requirements that apply to the set of three services as a whole",
        parent=parent_node,
        critical=True
    )

    # Exactly three services (no more, no fewer) with name & website provided in the answer
    valid_count = sum(1 for b in all_boxes if is_nonempty_str(b.service_name) and is_nonempty_str(b.official_website))
    evaluator.add_custom_node(
        result=(valid_count == 3),
        id="Exactly_Three_Services",
        desc="The response provides exactly three subscription box services (no more, no fewer)",
        parent=set_node,
        critical=True
    )

    # Distinct services among the three selected (unique names and/or official websites)
    names = [b.service_name.strip().lower() for b in selected_boxes if is_nonempty_str(b.service_name)]
    domains = [normalize_domain(b.official_website) for b in selected_boxes if is_nonempty_str(b.official_website)]
    distinct_services = (len(names) == 3 and len(set(names)) == 3) and (len(domains) == 3 and len(set(domains)) == 3)
    evaluator.add_custom_node(
        result=distinct_services,
        id="Distinct_Services",
        desc="The three services are different/distinct (not duplicates, e.g., unique service names and/or official websites)",
        parent=set_node,
        critical=True
    )

    # Independent operation: proxy by requiring different domains (not product lines under same site)
    independent = (len(domains) == 3 and len(set(domains)) == 3)
    evaluator.add_custom_node(
        result=independent,
        id="Independent_Operation",
        desc="The three services are independently operated (not different product lines from the same company/parent company as constrained)",
        parent=set_node,
        critical=True
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
    """
    Evaluate an answer for the craft subscription boxes task.
    """
    # Initialize evaluator and root (parallel aggregation)
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

    # Extract boxes from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_boxes(),
        template_class=SubscriptionBoxesExtraction,
        extraction_name="subscription_boxes"
    )
    all_boxes: List[SubscriptionBox] = extraction.boxes or []

    # Select up to three valid boxes (with name & website)
    selected_boxes = select_first_three_valid(all_boxes)
    # Pad to exactly three for per-box verification (placeholders will naturally fail critical checks)
    while len(selected_boxes) < 3:
        selected_boxes.append(SubscriptionBox())

    # Build verification nodes for the three boxes
    for i in range(3):
        await verify_box(evaluator, root, selected_boxes[i], i)

    # Set-level requirements (critical gating for the whole task)
    add_set_level_requirements(evaluator, root, all_boxes, selected_boxes)

    # Add custom info (optional diagnostics)
    evaluator.add_custom_info(
        info={
            "total_services_extracted": len(all_boxes),
            "selected_for_verification": sum(1 for b in selected_boxes if is_nonempty_str(b.service_name) and is_nonempty_str(b.official_website)),
            "selected_domains": [normalize_domain(b.official_website) for b in selected_boxes]
        },
        info_type="diagnostics",
        info_name="extraction_summary"
    )

    # Return the evaluation summary
    return evaluator.get_summary()