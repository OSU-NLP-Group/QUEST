import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "philly_diy_4"
TASK_DESCRIPTION = """
Find 4 beginner-friendly DIY hobby classes or workshops in the greater Philadelphia metropolitan area (including nearby New Jersey suburbs such as Cherry Hill, Burlington, or Collingswood) that offer in-person, hands-on instruction. Each of the 4 options must represent a different type of DIY hobby or craft category (such as pottery, painting, woodworking, fiber arts, leather crafting, candle making, etc.). For each class or workshop, provide: (1) The name of the studio, school, or organization offering the class, (2) The specific type of DIY hobby or craft taught, (3) The location (city and state), (4) Clear pricing information for the class or workshop, (5) Schedule information or contact details for enrollment, and (6) A reference URL to the organization's website or class information page. Note: Options must offer actual instruction or guided workshops, not just retail craft supply stores.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DIYClassItem(BaseModel):
    studio_name: Optional[str] = None
    hobby_type: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    pricing: Optional[str] = None
    schedule_info: Optional[str] = None
    contact_info: Optional[str] = None
    url: Optional[str] = None
    beginner_text: Optional[str] = None        # e.g., "beginner-friendly" / "no experience necessary"
    in_person_text: Optional[str] = None       # e.g., "in-person", "hands-on", "workshop", "in-studio"
    instruction_text: Optional[str] = None     # any phrase indicating guided instruction or workshop


class DIYClassesExtraction(BaseModel):
    classes: List[DIYClassItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_diy_classes() -> str:
    return """
    Extract up to four (4) DIY hobby classes or workshops described in the answer. For each option, return a JSON object with:
    - studio_name: Name of the studio, school, or organization offering the class/workshop
    - hobby_type: The specific type of DIY hobby or craft taught (e.g., pottery, woodworking, fiber arts, painting, leather crafting, candle making, etc.)
    - location_city: The city where the class takes place
    - location_state: The state (e.g., PA, NJ) where the class takes place
    - pricing: Clear pricing information exactly as stated (e.g., "$65 per person", "$95 including materials", "Class $200", etc.). If multiple prices, summarize briefly.
    - schedule_info: Schedule details (dates/times/upcoming sessions/calendar) if provided. If missing, set to null.
    - contact_info: Contact/enrollment details (email/phone/enroll link/booking form/call-to-action) if provided. If missing, set to null.
    - url: A single reference URL to the organization’s website or the specific class information page. Use the URL explicitly present in the answer. If missing, set to null.
    - beginner_text: Exact phrase(s) indicating beginner suitability or "no experience necessary" if present; else null.
    - in_person_text: Exact phrase(s) indicating the class is in-person and/or hands-on (e.g., "in-person", "hands-on", "workshop", "in-studio"); else null.
    - instruction_text: Any phrase that clearly indicates guided instruction or a workshop (e.g., "class", "course", "workshop", "taught by", "instructor-led"); else null.

    Rules:
    - Do not invent or infer information. Extract only what appears in the answer.
    - If more than 4 options are provided in the answer, extract only the first four in the same order.
    - If fewer than 4 options are provided, still return all found ones; missing fields should be null.
    - Prefer the organization’s own webpage or a dedicated class page for 'url'. If an aggregator/listing page is cited in the answer, extract that URL as-is.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _loc_str(item: DIYClassItem) -> str:
    city = (item.location_city or "").strip()
    state = (item.location_state or "").strip()
    if city and state:
        return f"{city}, {state}"
    return (city or state or "").strip()


def _normalize_label(s: Optional[str]) -> str:
    return (s or "").strip().lower()


# --------------------------------------------------------------------------- #
# Verification logic per class                                                #
# --------------------------------------------------------------------------- #
async def verify_one_class(
    evaluator: Evaluator,
    parent_node,
    item: DIYClassItem,
    idx: int,
) -> None:
    """
    Build and run verification nodes for a single DIY class/workshop.
    """
    class_node = evaluator.add_sequential(
        id=f"hobby_class_{idx+1}",
        desc=f"DIY hobby class/workshop #{idx+1} meets all required criteria",
        parent=parent_node,
        critical=False,
    )

    # Step 0: Required info presence (critical gate)
    required_present = (
        _nonempty(item.studio_name)
        and _nonempty(item.hobby_type)
        and _nonempty(item.location_city)
        and _nonempty(item.location_state)
        and _nonempty(item.pricing)
        and _nonempty(item.url)
        and (_nonempty(item.schedule_info) or _nonempty(item.contact_info))
    )
    evaluator.add_custom_node(
        result=required_present,
        id=f"class_{idx+1}_required_info",
        desc=f"Class #{idx+1} has all required fields (name, hobby type, city, state, pricing, URL, and schedule/contact)",
        parent=class_node,
        critical=True,
    )

    # Step 1: Organization/page support
    org_leaf = evaluator.add_leaf(
        id=f"class_{idx+1}_org_supported",
        desc=f"URL page is about the cited organization '{item.studio_name}' or clearly lists its class",
        parent=class_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page is about or clearly lists a class/workshop by '{item.studio_name}'.",
        node=org_leaf,
        sources=item.url,
        additional_instruction="Accept if the organization/studio name appears prominently, or the listing page clearly attributes the class to this organization.",
    )

    # Step 2: Hobby type supported
    hobby_leaf = evaluator.add_leaf(
        id=f"class_{idx+1}_hobby_supported",
        desc=f"Page explicitly supports the cited hobby type '{item.hobby_type}' for this class/workshop",
        parent=class_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page describes a class/workshop that teaches '{item.hobby_type}' (or a closely related subcategory).",
        node=hobby_leaf,
        sources=item.url,
        additional_instruction="Look for explicit mentions of the craft type (e.g., pottery, woodworking, fiber arts, painting, leather, candles). Allow close subcategory matches.",
    )

    # Step 3: Location supported by page
    loc_leaf = evaluator.add_leaf(
        id=f"class_{idx+1}_location_supported",
        desc=f"Page indicates the class occurs at/near {_loc_str(item)} (address or city/state shown)",
        parent=class_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page explicitly indicates the class occurs in {_loc_str(item)} (via city/state or address).",
        node=loc_leaf,
        sources=item.url,
        additional_instruction="Accept if the page shows an address in the cited city/state or an explicit city+state mention.",
    )

    # Step 3b: City belongs to greater Philadelphia metro or nearby NJ suburb (non-critical semantic check)
    metro_leaf = evaluator.add_leaf(
        id=f"class_{idx+1}_in_philly_metro",
        desc=f"Location {_loc_str(item)} is in the greater Philadelphia metropolitan area (or a nearby NJ suburb)",
        parent=class_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"{_loc_str(item)} is part of (or immediately adjacent to) the Philadelphia metropolitan area, including NJ suburbs.",
        node=metro_leaf,
        additional_instruction="Use general U.S. geography knowledge. If uncertain, mark as not supported.",
    )

    # Step 4: Beginner-friendly
    beginner_leaf = evaluator.add_leaf(
        id=f"class_{idx+1}_beginner",
        desc=f"Page indicates beginner-friendly or no-experience-necessary offering",
        parent=class_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The page states the class/workshop is beginner-friendly or that no prior experience is required.",
        node=beginner_leaf,
        sources=item.url,
        additional_instruction="Look for phrases like 'beginner-friendly', 'no experience necessary', 'intro/beginner class', or similar.",
    )

    # Step 5: In-person, hands-on instruction
    inperson_leaf = evaluator.add_leaf(
        id=f"class_{idx+1}_in_person_hands_on",
        desc="Class/workshop is in-person and hands-on (guided instruction)",
        parent=class_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The page indicates the class/workshop is in-person and provides hands-on, guided instruction.",
        node=inperson_leaf,
        sources=item.url,
        additional_instruction="Accept synonyms like 'in-person', 'in-studio', 'on-site', 'workshop', 'hands-on', 'instructor-led'. Reject purely online/self-paced content.",
    )

    # Step 6: Clear pricing
    price_leaf = evaluator.add_leaf(
        id=f"class_{idx+1}_pricing_supported",
        desc="Page provides clear pricing information for the class/workshop",
        parent=class_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The page provides explicit pricing information for the class/workshop.",
        node=price_leaf,
        sources=item.url,
        additional_instruction="Accept any explicit price (fixed, per-person, per-session, deposit + balance, range) clearly tied to attending the class/workshop.",
    )

    # Step 7: Schedule or enrollment contact
    enroll_leaf = evaluator.add_leaf(
        id=f"class_{idx+1}_enrollment_info",
        desc="Page provides schedule details or a clear enrollment/contact method (email/phone/booking/form)",
        parent=class_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The page provides schedule details (dates/times/upcoming sessions/calendar) OR a clear enrollment/contact method (email, phone, booking link, sign-up form).",
        node=enroll_leaf,
        sources=item.url,
        additional_instruction="At least one of schedule details or a clear way to enroll/contact must be present.",
    )

    # Step 8: Not retail-only (must be actual instruction)
    not_retail_leaf = evaluator.add_leaf(
        id=f"class_{idx+1}_not_retail_only",
        desc="Page is about an actual class/workshop with instruction, not just a retail craft supply store/product",
        parent=class_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This page offers an actual class or workshop with instruction (not merely a retail craft supply store or product page).",
        node=not_retail_leaf,
        sources=item.url,
        additional_instruction="Look for terms like 'class', 'course', 'workshop', 'lessons', 'register', 'book', 'taught by', etc. Reject pages that only sell supplies without instructional offerings.",
    )


# --------------------------------------------------------------------------- #
# Hobby diversity verification                                                #
# --------------------------------------------------------------------------- #
async def verify_diversity(
    evaluator: Evaluator,
    parent_node,
    items: List[DIYClassItem],
) -> None:
    diversity_node = evaluator.add_parallel(
        id="hobby_diversity",
        desc="The 4 identified classes represent 4 different DIY/craft categories",
        parent=parent_node,
        critical=False,
    )

    categories = [(_normalize_label(it.hobby_type)) for it in items[:4]]
    nonnull_cats = [c for c in categories if c]
    all_four_present = len(nonnull_cats) == 4

    evaluator.add_custom_node(
        result=all_four_present,
        id="hobby_diversity_all_four_present",
        desc="All four hobby categories are provided (non-empty)",
        parent=diversity_node,
        critical=True,  # if not all present, diversity fails
    )

    diversity_leaf = evaluator.add_leaf(
        id="hobby_diversity_distinct",
        desc="The four hobby categories are distinct (not the same or trivial variants)",
        parent=diversity_node,
        critical=False,
    )

    cats_str = ", ".join([c if c else "MISSING" for c in categories])
    await evaluator.verify(
        claim=f"The following four categories represent four different types of DIY/craft: {cats_str}.",
        node=diversity_leaf,
        additional_instruction="Use common sense. Treat synonyms/subtypes (e.g., 'painting' vs 'watercolor painting') as the same category. Categories must be meaningfully different.",
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
    Evaluate an answer for the Philly DIY classes task.
    """
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

    # 1) Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_diy_classes(),
        template_class=DIYClassesExtraction,
        extraction_name="diy_classes_extraction",
    )

    # Keep only the first 4; pad with empty if fewer than 4
    items = list(extracted.classes[:4])
    while len(items) < 4:
        items.append(DIYClassItem())

    # Record custom info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "categories": [it.hobby_type for it in items],
            "locations": [f"{(it.location_city or '').strip()}, {(it.location_state or '').strip()}" for it in items],
            "urls": [it.url for it in items],
        },
        info_type="extraction_overview",
        info_name="extraction_overview",
    )

    # 2) Build verification tree for each of the 4 classes
    for i, item in enumerate(items):
        await verify_one_class(evaluator, root, item, i)

    # 3) Diversity check across the 4 categories
    await verify_diversity(evaluator, root, items)

    # 4) Return structured evaluation summary
    return evaluator.get_summary()