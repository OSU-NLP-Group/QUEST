import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_family_march_2026"
TASK_DESCRIPTION = """
Identify a Broadway theater in New York City that meets ALL of the following requirements for a family planning to attend a show in March 2026:
(1) The theater must have a seating capacity between 1,500 and 2,000 seats;
(2) The theater must offer a weekly discounted ticket program with tickets available for under $50;
(3) The theater must provide wheelchair accessible seating;
(4) The theater must permit children ages 4 and above to attend performances;
(5) The theater must currently be showing a production (as of March 2026).
Provide the theater name, the box office phone number and address, and a reference URL from the theater's official website or ticketing page that confirms this information.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TheaterExtraction(BaseModel):
    theater_name: Optional[str] = None
    seating_capacity: Optional[str] = None
    weekly_discount_program: Optional[str] = None  # name/description (e.g., Rush, Lottery)
    weekly_discount_price: Optional[str] = None    # e.g., "$39", "40", "<$50"
    wheelchair_accessibility: Optional[str] = None  # free-form text from answer
    age_policy: Optional[str] = None               # text describing minimum age policy
    current_production: Optional[str] = None       # production name if provided
    box_office_phone: Optional[str] = None
    address: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theater_info() -> str:
    return """
Extract the single Broadway theater mentioned in the answer (use the first one if multiple are listed). Return the following fields exactly as stated in the answer text:

- theater_name: The name of the theater.
- seating_capacity: The stated seating capacity number or phrase (e.g., "1,761", "about 1,800 seats").
- weekly_discount_program: The specific weekly discount program name/description if provided (e.g., "Rush", "Lottery"), else null.
- weekly_discount_price: The price associated with the weekly discount program if mentioned (e.g., "$39", "40", "<$50"), else null.
- wheelchair_accessibility: Summary phrase from the answer regarding wheelchair accessible seating (or ADA), else null.
- age_policy: The policy text from the answer concerning children (e.g., "No children under 4", "Ages 4+ permitted"), else null.
- current_production: The production currently being shown (as stated), if provided; else null.
- box_office_phone: The box office phone number as written in the answer, else null.
- address: The full address provided in the answer, else null.
- reference_urls: A list of all URLs explicitly present in the answer that are intended as supporting references (official theater website pages or official ticketing pages). Return actual URLs only; do not fabricate any.

Important:
- Do not infer or invent any information. Only extract what the answer explicitly states.
- For URLs, extract all that are present in the answer and intended as references. Accept plain URLs or markdown links; return the raw URL strings.
- If a field is not present, set it to null (or an empty list for reference_urls).
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _sanitize_urls(urls: List[str]) -> List[str]:
    clean = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u  # extractor may already normalize, but ensure protocol
        clean.append(u)
    # de-duplicate preserving order
    seen = set()
    unique = []
    for u in clean:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _parse_seat_count(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Find the first plausible integer with optional thousands separators
    m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{3,4})\b", text.replace("\u202f", "").replace(" ", ""))
    if not m:
        # try to catch simple ranges like "1,500-2,000" -> take first number
        m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{3,4})\s*[-–]\s*(\d{1,3}(?:,\d{3})+|\d{3,4})\b", text)
        if m:
            first = m.group(1).replace(",", "")
            try:
                return int(first)
            except Exception:
                return None
        return None
    num = m.group(1).replace(",", "")
    try:
        return int(num)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: TheaterExtraction) -> None:
    # Root-level (provided by evaluator.initialize). Create a single critical child matching the rubric root.
    root = evaluator.root
    if root is None:
        raise RuntimeError("Evaluator root not initialized")

    # Create the main critical node aggregating all checks
    main = evaluator.add_parallel(
        id="broadway_theater_selection",
        desc="Evaluate whether the identified Broadway theater meets all specified requirements for a family-friendly ticketed venue",
        parent=root,
        critical=True
    )

    # Normalize sources once
    sources = _sanitize_urls(extracted.reference_urls or [])
    theater_name = extracted.theater_name or "the theater"

    # 1) Reference URL (make it a small subtree: provided + official/ticketing)
    ref_parent = evaluator.add_parallel(
        id="reference_url",
        desc="Provide a valid reference URL from the theater's official website or ticketing page that confirms the information",
        parent=main,
        critical=True
    )

    ref_provided = evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="reference_url_provided",
        desc="At least one reference URL is provided in the answer",
        parent=ref_parent,
        critical=True
    )

    ref_official = evaluator.add_leaf(
        id="reference_url_official",
        desc="At least one provided URL is an official theater page or an official ticketing page for this theater",
        parent=ref_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one of these pages is either the official website for '{theater_name}' OR an official ticketing page for that specific theater (e.g., Telecharge, Ticketmaster, BroadwayDirect).",
        node=ref_official,
        sources=sources,
        additional_instruction="Accept if the page appears to be operated by the theater owner/operator (e.g., Shubert, Nederlander) or an official ticketing provider. Reject obvious blogs/aggregators."
    )

    # 2) Theater Location (Broadway in NYC)
    theater_location = evaluator.add_leaf(
        id="theater_location",
        desc="The theater is a Broadway theater located in New York City",
        parent=main,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms that '{theater_name}' is a Broadway theater located in New York City (Manhattan, NY).",
        node=theater_location,
        sources=sources,
        additional_instruction="Allow phrases like 'Broadway theatre', 'Theatre District', or references to Shubert/Nederlander/Jujamcyn Broadway houses. Reject Off-Broadway/Off-Off-Broadway."
    )

    # 3) Seating Capacity (subtree: supported value + in-range check)
    cap_parent = evaluator.add_parallel(
        id="seating_capacity",
        desc="The theater has a seating capacity between 1,500 and 2,000 seats",
        parent=main,
        critical=True
    )

    capacity_supported = evaluator.add_leaf(
        id="capacity_supported",
        desc="The seating capacity value is supported by the cited source",
        parent=cap_parent,
        critical=True
    )
    cap_text = extracted.seating_capacity or ""
    await evaluator.verify(
        claim=f"The seating capacity of {theater_name} is {cap_text} seats (or an equivalent phrasing).",
        node=capacity_supported,
        sources=sources,
        additional_instruction="Look for phrases like 'seating capacity', 'capacity', 'seats'. Small rounding differences are acceptable."
    )

    cap_value = _parse_seat_count(extracted.seating_capacity)
    in_range = cap_value is not None and 1500 <= cap_value <= 2000
    evaluator.add_custom_node(
        result=in_range,
        id="capacity_in_range",
        desc=f"Parsed seating capacity {cap_value if cap_value is not None else 'N/A'} is between 1,500 and 2,000",
        parent=cap_parent,
        critical=True
    )

    # 4) Weekly Discount Program under $50
    weekly_discount = evaluator.add_leaf(
        id="weekly_discount_program",
        desc="The theater offers a weekly discounted ticket program with tickets priced under $50",
        parent=main,
        critical=True
    )
    claimed_price = extracted.weekly_discount_price or "under $50"
    await evaluator.verify(
        claim=f"The theater offers a weekly discounted ticket program (e.g., rush, lottery, weekly offer) with tickets priced under $50 (e.g., {claimed_price}).",
        node=weekly_discount,
        sources=sources,
        additional_instruction="Accept weekly recurring programs (e.g., weekly lottery releases, rush policies tied to weekly schedules). Reject one-off promotions. Verify price is < $50."
    )

    # 5) Wheelchair Accessibility
    wheelchair = evaluator.add_leaf(
        id="wheelchair_accessibility",
        desc="The theater provides wheelchair accessible seating in compliance with ADA requirements",
        parent=main,
        critical=True
    )
    await evaluator.verify(
        claim=f"{theater_name} provides wheelchair-accessible seating (ADA-compliant) and related accommodations.",
        node=wheelchair,
        sources=sources,
        additional_instruction="Look for wheelchair icons, ADA language, accessible seating notes, or specific instructions for purchasing accessible seats."
    )

    # 6) Age Restriction Policy (children ages 4+ permitted)
    age_policy = evaluator.add_leaf(
        id="age_restriction_policy",
        desc="The theater permits children ages 4 and above to attend performances",
        parent=main,
        critical=True
    )
    await evaluator.verify(
        claim=f"Children ages 4 and above are permitted to attend performances at {theater_name}.",
        node=age_policy,
        sources=sources,
        additional_instruction="Common phrasing is 'No children under 4 admitted' which implies ages 4+ permitted. That should be accepted."
    )

    # 7) Current Production (as of March 2026)
    current_prod = evaluator.add_leaf(
        id="current_production_march_2026",
        desc="The theater is currently showing a production (as of March 2026)",
        parent=main,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of March 2026, {theater_name} has an active production scheduled/performing during that month.",
        node=current_prod,
        sources=sources,
        additional_instruction="Accept if the page shows a production with performance dates in March 2026 (e.g., calendar, schedule, 'now playing' during March 2026)."
    )

    # 8) Box Office Contact Information (subtree: phone + address, each provided and supported)
    box_parent = evaluator.add_parallel(
        id="box_office_contact",
        desc="Provide the theater's box office phone number and address",
        parent=main,
        critical=True
    )

    # Phone: provided?
    phone_provided = evaluator.add_custom_node(
        result=bool(extracted.box_office_phone and extracted.box_office_phone.strip()),
        id="box_office_phone_provided",
        desc="Box office phone number is provided in the answer",
        parent=box_parent,
        critical=True
    )

    # Phone: supported
    phone_supported = evaluator.add_leaf(
        id="box_office_phone_supported",
        desc="Box office phone number is supported by the cited source",
        parent=box_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The box office phone number for {theater_name} is '{(extracted.box_office_phone or '').strip()}'.",
        node=phone_supported,
        sources=sources,
        additional_instruction="Match the digits ignoring common formatting differences (spaces, dashes, parentheses)."
    )

    # Address: provided?
    address_provided = evaluator.add_custom_node(
        result=bool(extracted.address and extracted.address.strip()),
        id="box_office_address_provided",
        desc="Box office street address is provided in the answer",
        parent=box_parent,
        critical=True
    )

    # Address: supported
    address_supported = evaluator.add_leaf(
        id="box_office_address_supported",
        desc="Box office address is supported by the cited source",
        parent=box_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The address of {theater_name} is '{(extracted.address or '').strip()}', located in New York, NY.",
        node=address_supported,
        sources=sources,
        additional_instruction="Allow minor formatting differences (abbrev vs full street names). Must indicate NYC location."
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
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root aggregation
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
        prompt=prompt_extract_theater_info(),
        template_class=TheaterExtraction,
        extraction_name="theater_extraction",
    )

    # Record constraints as custom info for transparency
    evaluator.add_custom_info(
        info={
            "capacity_required_range": [1500, 2000],
            "discount_requirement": "weekly discounted program with tickets under $50",
            "accessibility": "wheelchair accessible seating (ADA)",
            "age_policy": "children ages 4+ permitted",
            "current_production_month": "March 2026",
        },
        info_type="constraints",
        info_name="evaluation_constraints",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()