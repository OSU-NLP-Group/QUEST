import asyncio
import logging
import re
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "broadway_booking"
TASK_DESCRIPTION = (
    "I am planning to attend a Broadway musical in New York City and need detailed information to make my booking decision. "
    "Please identify one currently running Broadway musical and provide comprehensive information about its venue and ticketing options.\n\n"
    "Specifically, I need:\n"
    "1. The name of the musical and confirmation that it is currently running\n"
    "2. The name of the Broadway theater where it is performed\n"
    "3. The theater's total seating capacity (which must meet the Broadway theater requirement of 500 or more seats)\n"
    "4. Information about the theater's wheelchair-accessible seating and ADA compliance features\n"
    "5. A description of the ticket pricing structure, including:\n"
    "   - Orchestra seating section details and pricing\n"
    "   - Mezzanine seating section details and pricing\n"
    "   - Premium or VIP seating options with enhanced pricing\n\n"
    "For each piece of information, please provide reference URLs from official Broadway theater websites, ticketing platforms, or venue websites that confirm these details."
)


class ShowExtraction(BaseModel):
    show_name: Optional[str] = None
    show_urls: List[str] = Field(default_factory=list)
    running_status_text: Optional[str] = None


class VenueExtraction(BaseModel):
    theater_name: Optional[str] = None
    theater_urls: List[str] = Field(default_factory=list)
    seating_capacity: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)
    accessibility_info: Optional[str] = None
    accessibility_urls: List[str] = Field(default_factory=list)


class PricingSection(BaseModel):
    section_name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PricingExtraction(BaseModel):
    orchestra: PricingSection = Field(default_factory=PricingSection)
    mezzanine: PricingSection = Field(default_factory=PricingSection)
    premium_vip: PricingSection = Field(default_factory=PricingSection)


def prompt_extract_show_info() -> str:
    return (
        "Extract exactly one currently running Broadway musical referenced in the answer.\n"
        "Return a JSON object with the following fields:\n"
        "1. show_name: The musical's name as stated in the answer\n"
        "2. show_urls: An array of 1–5 URLs that the answer cites to support the show's current running status on Broadway. "
        "   Prefer official show or theater websites, or reputable ticketing platforms (e.g., Telecharge, Ticketmaster, SeatGeek, Broadway.com, Playbill).\n"
        "3. running_status_text: A short text snippet from the answer that asserts the show is currently running (e.g., 'now playing', 'currently running', 'performances scheduled').\n"
        "If multiple shows are mentioned, pick the first or most emphasized as the primary show. If any field is missing, return null or an empty array accordingly."
    )


def prompt_extract_venue_info() -> str:
    return (
        "Extract the Broadway theater venue info for the identified show. Return a JSON object with:\n"
        "1. theater_name: The name of the Broadway theater where the show is performed\n"
        "2. theater_urls: 1–5 URLs cited in the answer that confirm the venue association (official venue/owner site or reputable Broadway listings)\n"
        "3. seating_capacity: The total seating capacity as stated in the answer (string; may be a number or text like '1,080 seats')\n"
        "4. capacity_urls: 1–5 URLs that confirm the capacity\n"
        "5. accessibility_info: A concise description of wheelchair-accessible seating/ADA features mentioned in the answer\n"
        "6. accessibility_urls: 1–5 URLs confirming the accessibility/ADA accommodations\n"
        "If any item is not present in the answer, set the corresponding field to null or an empty array."
    )


def prompt_extract_pricing_info() -> str:
    return (
        "Extract ticket pricing details for the identified show, organized by seating sections. Return a JSON object with nested objects:\n"
        "orchestra: { section_name, description, price, urls }\n"
        "mezzanine: { section_name, description, price, urls }\n"
        "premium_vip: { section_name, description, price, urls }\n"
        "Where:\n"
        "- section_name: the section name (e.g., 'Orchestra', 'Mezzanine', 'Premium/VIP') if present\n"
        "- description: brief details of the section (e.g., front orchestra, center mezzanine)\n"
        "- price: a price or range as a string (e.g., '$89-$199', 'from $129')\n"
        "- urls: 1–5 URLs supporting pricing/tier info (prefer reputable ticketing platforms or official sources)\n"
        "If any field is missing for a section, set it to null or an empty array."
    )


def _has_any_url(urls: List[str]) -> bool:
    return isinstance(urls, list) and len(urls) > 0 and any(isinstance(u, str) and len(u.strip()) > 0 for u in urls)


def _extract_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    nums = re.findall(r"\d{3,5}", text.replace(",", ""))
    try:
        return int(nums[0]) if nums else None
    except Exception:
        return None


async def build_and_verify_tree(
    evaluator: Evaluator,
    show: ShowExtraction,
    venue: VenueExtraction,
    pricing: PricingExtraction,
) -> None:
    top = evaluator.add_sequential(
        id="Broadway_Musical_Booking_Info",
        desc="Evaluate whether the response identifies one currently running Broadway musical and provides required venue + ticketing details with appropriate reference URLs.",
        parent=evaluator.root,
        critical=True,
    )

    show_group = evaluator.add_parallel(
        id="Show_Identification",
        desc="Show is identified and confirmed currently running on Broadway with an acceptable reference URL.",
        parent=top,
        critical=True,
    )

    show_existence = evaluator.add_custom_node(
        result=(show.show_name is not None and show.show_name.strip() != "" and _has_any_url(show.show_urls)),
        id="Show_Name_and_URL_Provided",
        desc="Show name is provided and at least one supporting URL is present.",
        parent=show_group,
        critical=True,
    )

    show_leaf = evaluator.add_leaf(
        id="Show_Name_and_Running_Status_with_URL",
        desc="Provides the musical name and confirms it is currently running on Broadway with ≥1 reference URL.",
        parent=show_group,
        critical=True,
    )
    show_claim = (
        f"The musical '{(show.show_name or '').strip()}' is currently running on Broadway (New York City) "
        f"with scheduled performances or a 'now playing' indication."
    )
    await evaluator.verify(
        claim=show_claim,
        node=show_leaf,
        sources=show.show_urls,
        additional_instruction="Confirm from the provided URL(s) that the show is currently running on a Broadway stage in NYC (not Off-Broadway or touring). Look for schedules, 'now playing', or performance listings.",
    )

    venue_group = evaluator.add_parallel(
        id="Venue_Details",
        desc="Broadway theater venue information is provided and satisfies capacity and accessibility constraints, with reference URLs.",
        parent=top,
        critical=True,
    )

    theater_exist = evaluator.add_custom_node(
        result=(venue.theater_name is not None and venue.theater_name.strip() != "" and _has_any_url(venue.theater_urls)),
        id="Theater_Name_URL_Provided",
        desc="Theater name and venue association URL(s) are provided.",
        parent=venue_group,
        critical=True,
    )

    theater_leaf = evaluator.add_leaf(
        id="Theater_Name_with_URL",
        desc="Provides the Broadway theater name where the show is performed, with ≥1 reference URL confirming the venue association.",
        parent=venue_group,
        critical=True,
    )
    theater_claim = (
        f"The musical '{(show.show_name or '').strip()}' is performed at the Broadway theater '{(venue.theater_name or '').strip()}'."
    )
    await evaluator.verify(
        claim=theater_claim,
        node=theater_leaf,
        sources=venue.theater_urls,
        additional_instruction="Verify that the provided URL(s) explicitly associate the identified show with the specified Broadway theater venue.",
    )

    capacity_exist = evaluator.add_custom_node(
        result=(venue.seating_capacity is not None and venue.seating_capacity.strip() != "" and _has_any_url(venue.capacity_urls)),
        id="Capacity_Info_URL_Provided",
        desc="Seating capacity value and supporting capacity URL(s) are provided.",
        parent=venue_group,
        critical=True,
    )

    capacity_leaf = evaluator.add_leaf(
        id="Seating_Capacity_500plus_with_URL",
        desc="Provides the theater’s total seating capacity and verifies it is ≥ 500 seats, with ≥1 reference URL confirming the capacity.",
        parent=venue_group,
        critical=True,
    )
    capacity_number = _extract_first_int(venue.seating_capacity)
    capacity_desc = venue.seating_capacity or ""
    capacity_claim = (
        f"The theater '{(venue.theater_name or '').strip()}' has a total seating capacity of {capacity_desc}, "
        f"which is at least 500 seats."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=venue.capacity_urls,
        additional_instruction="Confirm from the URL(s) the stated total seating capacity and that it meets or exceeds 500 seats (Broadway requirement). If multiple figures are shown, use the official capacity.",
    )

    access_exist = evaluator.add_custom_node(
        result=(venue.accessibility_info is not None and venue.accessibility_info.strip() != "" and _has_any_url(venue.accessibility_urls)),
        id="Accessibility_Info_URL_Provided",
        desc="Accessibility/ADA details and supporting URL(s) are provided.",
        parent=venue_group,
        critical=True,
    )

    access_leaf = evaluator.add_leaf(
        id="Accessibility_ADA_with_URL",
        desc="Provides wheelchair-accessible seating and ADA/accessibility feature information for the theater, with ≥1 reference URL confirming accommodations.",
        parent=venue_group,
        critical=True,
    )
    access_claim = (
        f"The theater '{(venue.theater_name or '').strip()}' offers wheelchair-accessible seating and ADA accommodations: "
        f"{(venue.accessibility_info or '').strip()}."
    )
    await evaluator.verify(
        claim=access_claim,
        node=access_leaf,
        sources=venue.accessibility_urls,
        additional_instruction="Verify that the URL(s) describe wheelchair-accessible seating and ADA features (e.g., ramps, elevators, assistive listening, accessible restrooms) for the theater.",
    )

    pricing_group = evaluator.add_parallel(
        id="Ticket_Pricing",
        desc="Ticket pricing structure includes required tiers/sections and premium/VIP options, each supported by reference URLs.",
        parent=top,
        critical=True,
    )

    orch_exist = evaluator.add_custom_node(
        result=(pricing.orchestra.price is not None and pricing.orchestra.price.strip() != "" and _has_any_url(pricing.orchestra.urls)),
        id="Orchestra_Info_URL_Provided",
        desc="Orchestra section pricing and URL(s) are provided.",
        parent=pricing_group,
        critical=True,
    )

    orch_leaf = evaluator.add_leaf(
        id="Orchestra_Pricing_with_URL",
        desc="Describes orchestra seating section details and pricing with ≥1 reference URL confirming orchestra pricing/tier information.",
        parent=pricing_group,
        critical=True,
    )
    orch_claim = (
        f"For '{(show.show_name or '').strip()}', Orchestra section pricing is {(pricing.orchestra.price or '').strip()}; "
        f"section details: {(pricing.orchestra.description or '').strip()}."
    )
    await evaluator.verify(
        claim=orch_claim,
        node=orch_leaf,
        sources=pricing.orchestra.urls,
        additional_instruction="Confirm that the provided URL(s) show Orchestra section pricing for this show. Accept ranges or 'from $X'.",
    )

    mezz_exist = evaluator.add_custom_node(
        result=(pricing.mezzanine.price is not None and pricing.mezzanine.price.strip() != "" and _has_any_url(pricing.mezzanine.urls)),
        id="Mezzanine_Info_URL_Provided",
        desc="Mezzanine section pricing and URL(s) are provided.",
        parent=pricing_group,
        critical=True,
    )

    mezz_leaf = evaluator.add_leaf(
        id="Mezzanine_Pricing_with_URL",
        desc="Describes mezzanine seating section details and pricing with ≥1 reference URL confirming mezzanine pricing/tier information.",
        parent=pricing_group,
        critical=True,
    )
    mezz_claim = (
        f"For '{(show.show_name or '').strip()}', Mezzanine section pricing is {(pricing.mezzanine.price or '').strip()}; "
        f"section details: {(pricing.mezzanine.description or '').strip()}."
    )
    await evaluator.verify(
        claim=mezz_claim,
        node=mezz_leaf,
        sources=pricing.mezzanine.urls,
        additional_instruction="Confirm that the provided URL(s) show Mezzanine section pricing for this show. Accept ranges or 'from $X'.",
    )

    prem_exist = evaluator.add_custom_node(
        result=(pricing.premium_vip.price is not None and pricing.premium_vip.price.strip() != "" and _has_any_url(pricing.premium_vip.urls)),
        id="Premium_or_VIP_Info_URL_Provided",
        desc="Premium/VIP pricing and URL(s) are provided.",
        parent=pricing_group,
        critical=True,
    )

    prem_leaf = evaluator.add_leaf(
        id="Premium_or_VIP_Pricing_with_URL",
        desc="Describes premium/VIP ticket options and pricing with ≥1 reference URL confirming premium/VIP availability and pricing/tier information.",
        parent=pricing_group,
        critical=True,
    )
    prem_claim = (
        f"Premium or VIP tickets are available for '{(show.show_name or '').strip()}', and pricing is "
        f"{(pricing.premium_vip.price or '').strip()}."
    )
    await evaluator.verify(
        claim=prem_claim,
        node=prem_leaf,
        sources=pricing.premium_vip.urls,
        additional_instruction="Confirm from the URL(s) that Premium/VIP (enhanced-price) tickets exist for this show and the stated pricing/range is correct.",
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

    show_task = evaluator.extract(
        prompt=prompt_extract_show_info(),
        template_class=ShowExtraction,
        extraction_name="show_info",
    )
    venue_task = evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_info",
    )
    pricing_task = evaluator.extract(
        prompt=prompt_extract_pricing_info(),
        template_class=PricingExtraction,
        extraction_name="pricing_info",
    )
    show_info, venue_info, pricing_info = await asyncio.gather(show_task, venue_task, pricing_task)

    await build_and_verify_tree(evaluator, show_info, venue_info, pricing_info)

    return evaluator.get_summary()