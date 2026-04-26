import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dc_md_craft_stores_2025"
TASK_DESCRIPTION = """
Identify 4 craft stores located in Washington DC or Maryland that meet all of the following requirements for planning holiday DIY ornament crafting activities:

1. The store must be located in Washington DC or Maryland with a verifiable physical street address.

2. The store must carry DIY ornament-making supplies, specifically stocking at least 3 of the following materials: wood beads, ribbon, paint/brushes, embroidery thread, or pipe cleaners. The store must also offer holiday-themed Christmas ornament craft supplies.

3. The store's 2025 holiday hours must include: being closed on both Thanksgiving Day (November 27, 2025) and Christmas Day (December 25, 2025), being open on the Saturday after Thanksgiving (November 29, 2025), and having regular weekday hours that include closing at 8:00 PM or later on at least one weekday.

4. The store must offer at least one in-store holiday craft event or workshop during December 2025, with the event schedule publicly announced or advertised. If the event is free, it must not require advance registration or fees. If it is a paid workshop, it must be at least 1.5 hours in duration, and the materials cost per person must be clearly stated or indicated as included.

For each of the 4 stores, provide: the store name, complete street address, confirmation of the required craft supplies carried, verification of the 2025 holiday hours, and details of the December 2025 holiday craft event or workshop offered, including all relevant scheduling information and any registration or fee requirements.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None

    # URLs cited in the answer for each verification area
    location_urls: List[str] = Field(default_factory=list)
    supplies_urls: List[str] = Field(default_factory=list)
    hours_urls: List[str] = Field(default_factory=list)
    event_urls: List[str] = Field(default_factory=list)

    # Optional details the answer may provide (used as hints only)
    materials_listed: List[str] = Field(default_factory=list)  # e.g., ["wood beads", "ribbon", "paint"]
    holiday_ornament_supplies: Optional[bool] = None

    event_title: Optional[str] = None
    event_free_or_paid: Optional[str] = None  # "free" or "paid"
    event_duration: Optional[str] = None      # e.g., "2 hours", "90 minutes"
    event_registration: Optional[str] = None  # e.g., "none", "registration required", "RSVP"
    event_materials_cost_info: Optional[str] = None  # e.g., "$15 materials", "materials included"


class StoresExtraction(BaseModel):
    stores: List[StoreItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stores() -> str:
    return """
    Extract up to 4 craft stores mentioned in the answer (take the first 4 if more are provided). For each store, return:
    - name: store name (string)
    - address: complete street address as written in the answer (string; include city/state and ZIP if present)
    - location_urls: URL(s) in the answer that verify the physical street address (official site, Google listing, map/location page, etc.)
    - supplies_urls: URL(s) in the answer that show the store carries craft supplies (product/category pages, in-stock pages, etc.)
    - hours_urls: URL(s) in the answer that show 2025 holiday hours or clearly state closures/open dates and weekday hours
    - event_urls: URL(s) in the answer that publicly announce or advertise at least one in-store holiday craft event or workshop in December 2025 (event page, calendar, Facebook event, etc.)
    - materials_listed: list of any of the following materials explicitly claimed in the answer as carried by the store (use canonical names exactly from this list where applicable): ["wood beads", "ribbon", "paint/brushes", "embroidery thread", "pipe cleaners"]. If the answer uses synonyms (e.g., "embroidery floss" for embroidery thread), normalize to the canonical name.
    - holiday_ornament_supplies: true/false if the answer explicitly states the store offers holiday-themed Christmas ornament craft supplies; null if unclear
    - event_title: the title/name of a December 2025 in-store holiday craft event/workshop, if provided; else null
    - event_free_or_paid: "free" if the answer explicitly states free; "paid" if it explicitly states fees; null if unclear
    - event_duration: textual duration mentioned for the event if provided (e.g., "90 minutes", "2 hours"); else null
    - event_registration: textual registration requirement if provided (e.g., "no registration", "RSVP required"); else null
    - event_materials_cost_info: textual info if materials cost per person is clearly stated or indicated as included; else null

    Notes:
    - Only include URLs that are explicitly present in the answer text.
    - Do not invent or infer any URLs or details.
    - If a field is not present in the answer, return null (or empty list for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _material_instructions(material_key: str) -> str:
    mapping = {
        "wood beads": [
            "wood beads", "wooden beads", "unfinished wood beads", "natural wood beads", "craft wood beads"
        ],
        "ribbon": [
            "ribbon", "ribbons", "holiday ribbon", "satin ribbon", "grosgrain ribbon"
        ],
        "paint/brushes": [
            "paint", "paints", "acrylic paint", "tempera paint", "paint set", "paintbrush", "paint brush", "brushes", "brush set"
        ],
        "embroidery thread": [
            "embroidery thread", "embroidery floss", "floss", "DMC floss", "DMC embroidery"
        ],
        "pipe cleaners": [
            "pipe cleaners", "chenille stems", "fuzzy sticks"
        ],
    }
    synonyms = mapping.get(material_key, [material_key])
    return (
        f"For this verification, consider the following synonyms/variants for '{material_key}': {synonyms}. "
        "Treat close matches and common variants as acceptable."
    )


async def _verify_material_presence(
    evaluator: Evaluator,
    store_name: str,
    material_key: str,
    urls: List[str],
) -> bool:
    """
    Standalone verification (not added to the tree) that a store carries a given material.
    Returns True/False. This is used to compute the '>= 3 materials' requirement.
    """
    claim = (
        f"The store '{store_name}' carries {material_key} suitable for DIY ornament making."
    )
    add_ins = (
        "Verify on any of the provided URLs that the store sells/stock this material (product page, category, or search result on the store site is acceptable). "
        "The page should be relevant to the specific store/location rather than a generic blog post. "
        + _material_instructions(material_key)
    )
    return await evaluator.verify(claim=claim, node=None, sources=urls, additional_instruction=add_ins)


def _nonempty(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# --------------------------------------------------------------------------- #
# Verification logic per store                                                #
# --------------------------------------------------------------------------- #
async def verify_store(
    evaluator: Evaluator,
    parent_node,
    store: StoreItem,
    idx: int,
) -> None:
    """
    Build verification subtree for a single store.
    Node IDs follow the rubric names exactly: store_1, store_1_location, etc.
    """
    store_id = idx + 1
    store_node = evaluator.add_parallel(
        id=f"store_{store_id}",
        desc=["First", "Second", "Third", "Fourth"][idx] + " qualifying craft store identified",
        parent=parent_node,
        critical=False
    )

    name = store.name or f"Store #{store_id}"
    address = store.address or ""

    # 1) Location (critical)
    loc_node = evaluator.add_parallel(
        id=f"store_{store_id}_location",
        desc="Store is located in Washington DC or Maryland with verifiable street address",
        parent=store_node,
        critical=True
    )

    # 1.a) URL reference existence (critical)
    evaluator.add_custom_node(
        result=len(_nonempty(store.location_urls)) > 0,
        id=f"store_{store_id}_location_reference",
        desc="URL reference for location verification",
        parent=loc_node,
        critical=True
    )

    # 1.b) Address verification (critical)
    loc_verify_leaf = evaluator.add_leaf(
        id=f"store_{store_id}_location_verification",
        desc="Street address in DC or MD is confirmed from official source",
        parent=loc_node,
        critical=True
    )
    loc_claim = (
        f"The store '{name}' has a verifiable physical street address located in Washington, DC or Maryland: '{address}'."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_verify_leaf,
        sources=_nonempty(store.location_urls),
        additional_instruction=(
            "Confirm that the page shows a physical street address (not a P.O. Box). "
            "Accept 'Washington, DC' or 'District of Columbia' and any city in the state of Maryland (MD). "
            "Suite numbers are acceptable. If multiple locations exist, the cited page should match this store."
        )
    )

    # 2) Supplies (critical)
    supplies_node = evaluator.add_parallel(
        id=f"store_{store_id}_supplies",
        desc="Store carries at least 3 of the required DIY ornament-making materials",
        parent=store_node,
        critical=True
    )

    # 2.a) URL reference existence (critical)
    evaluator.add_custom_node(
        result=len(_nonempty(store.supplies_urls)) > 0,
        id=f"store_{store_id}_supplies_reference",
        desc="URL reference for supplies verification",
        parent=supplies_node,
        critical=True
    )

    # 2.b) Holiday-themed ornament craft supplies offered (critical)
    holiday_leaf = evaluator.add_leaf(
        id=f"store_{store_id}_supplies_holiday",
        desc="Holiday-themed Christmas ornament craft supplies are offered",
        parent=supplies_node,
        critical=True
    )
    holiday_claim = (
        f"The store '{name}' offers holiday-themed Christmas ornament craft supplies (e.g., ornament kits, holiday ribbons or embellishments)."
    )
    await evaluator.verify(
        claim=holiday_claim,
        node=holiday_leaf,
        sources=_nonempty(store.supplies_urls),
        additional_instruction=(
            "Look for explicit holiday/Christmas ornament craft supplies on the provided URLs: "
            "e.g., 'ornament', 'Christmas', 'holiday', 'seasonal', 'make-and-take ornament kits'. "
            "Generic, non-holiday craft supplies alone are not sufficient."
        )
    )

    # 2.c) Verify presence of at least 3 materials out of the 5; compute using standalone checks,
    # then add a single critical custom node reflecting the >=3 requirement.
    materials_keys = ["wood beads", "ribbon", "paint/brushes", "embroidery thread", "pipe cleaners"]
    mat_results: List[Tuple[str, bool]] = []
    for mk in materials_keys:
        try:
            present = await _verify_material_presence(
                evaluator, name, mk, _nonempty(store.supplies_urls)
            )
        except Exception:
            present = False
        mat_results.append((mk, bool(present)))

    count_present = sum(1 for _, ok in mat_results if ok)

    # Add the required inventory node (critical)
    evaluator.add_custom_node(
        result=(count_present >= 3),
        id=f"store_{store_id}_supplies_inventory",
        desc="Verified availability of wood beads, ribbon, paint/brushes, embroidery thread, or pipe cleaners (minimum 3 types)",
        parent=supplies_node,
        critical=True
    )

    # Record details for transparency
    evaluator.add_custom_info(
        info={
            "store_index": store_id,
            "store_name": name,
            "materials_verified": {k: v for k, v in mat_results},
            "count_verified": count_present
        },
        info_type="materials_verification",
        info_name=f"store_{store_id}_materials_check"
    )

    # 3) Hours (critical)
    hours_node = evaluator.add_parallel(
        id=f"store_{store_id}_hours",
        desc="Store operating hours meet all specified requirements",
        parent=store_node,
        critical=True
    )

    # 3.e) URL reference existence (critical)
    evaluator.add_custom_node(
        result=len(_nonempty(store.hours_urls)) > 0,
        id=f"store_{store_id}_hours_reference",
        desc="URL reference for holiday hours verification",
        parent=hours_node,
        critical=True
    )

    # 3.a) Closed on Thanksgiving Day (Nov 27, 2025) (critical)
    tg_leaf = evaluator.add_leaf(
        id=f"store_{store_id}_hours_thanksgiving",
        desc="Store is closed on Thanksgiving Day (November 27, 2025)",
        parent=hours_node,
        critical=True
    )
    tg_claim = "The store is closed on Thanksgiving Day, Thursday, November 27, 2025."
    await evaluator.verify(
        claim=tg_claim,
        node=tg_leaf,
        sources=_nonempty(store.hours_urls),
        additional_instruction=(
            "This must be specific to 2025 holiday hours. Accept phrasing like 'Closed Thanksgiving Day'. "
            "If a page shows a 2025 holiday schedule or clearly labels 2025, that is preferred."
        )
    )

    # 3.b) Open on Saturday after Thanksgiving (Nov 29, 2025) (critical)
    sat_leaf = evaluator.add_leaf(
        id=f"store_{store_id}_hours_thanksgiving_saturday",
        desc="Store is open on Saturday after Thanksgiving (November 29, 2025)",
        parent=hours_node,
        critical=True
    )
    sat_claim = "The store is open on Saturday, November 29, 2025 (the Saturday after Thanksgiving)."
    await evaluator.verify(
        claim=sat_claim,
        node=sat_leaf,
        sources=_nonempty(store.hours_urls),
        additional_instruction=(
            "Look for 'Open Saturday after Thanksgiving', 'Small Business Saturday', or explicit date 11/29/2025 with listed open hours. "
            "Opening with 'regular Saturday hours' also satisfies the requirement."
        )
    )

    # 3.c) Closed on Christmas Day (Dec 25, 2025) (critical)
    xmas_leaf = evaluator.add_leaf(
        id=f"store_{store_id}_hours_christmas",
        desc="Store is closed on Christmas Day (December 25, 2025)",
        parent=hours_node,
        critical=True
    )
    xmas_claim = "The store is closed on Christmas Day, December 25, 2025."
    await evaluator.verify(
        claim=xmas_claim,
        node=xmas_leaf,
        sources=_nonempty(store.hours_urls),
        additional_instruction="Prefer pages that explicitly label 2025 holiday hours or a 2025 holiday closure list."
    )

    # 3.d) At least one weekday closes at 8:00 PM or later (critical)
    eve_leaf = evaluator.add_leaf(
        id=f"store_{store_id}_hours_weekday_evening",
        desc="Regular hours include at least one weekday closing at 8:00 PM or later",
        parent=hours_node,
        critical=True
    )
    eve_claim = "On at least one weekday, the store's regular closing time is 8:00 PM or later."
    await evaluator.verify(
        claim=eve_claim,
        node=eve_leaf,
        sources=_nonempty(store.hours_urls),
        additional_instruction=(
            "Accept any weekday (Mon–Fri) schedule indicating close at 8:00 PM, 9:00 PM, etc. "
            "If seasonal/holiday weekday hours differ, either regular weekday hours or clearly-labeled 2025 holiday weekday hours are acceptable."
        )
    )

    # 4) Events (critical)
    events_node = evaluator.add_parallel(
        id=f"store_{store_id}_events",
        desc="Store offers qualifying December 2025 holiday craft event or workshop",
        parent=store_node,
        critical=True
    )

    # 4.d) URL reference existence (critical)
    evaluator.add_custom_node(
        result=len(_nonempty(store.event_urls)) > 0,
        id=f"store_{store_id}_events_reference",
        desc="URL reference for event information verification",
        parent=events_node,
        critical=True
    )

    # 4.a) Existence of at least one in-store December 2025 holiday craft event/workshop (critical)
    exist_leaf = evaluator.add_leaf(
        id=f"store_{store_id}_events_existence",
        desc="At least one in-store holiday craft event or workshop is offered during December 2025",
        parent=events_node,
        critical=True
    )
    exist_claim = (
        f"The store '{name}' offers at least one in-store holiday craft event or workshop during December 2025."
    )
    await evaluator.verify(
        claim=exist_claim,
        node=exist_leaf,
        sources=_nonempty(store.event_urls),
        additional_instruction=(
            "Evidence should show an event/workshop in December 2025 and indicate it is in-store/at the store location. "
            "Accept 'in-store', 'at this location', or a specific address/branch name on the event page."
        )
    )

    # 4.b) Publicly announced/advertised schedule (critical)
    ann_leaf = evaluator.add_leaf(
        id=f"store_{store_id}_events_announcement",
        desc="December 2025 event schedule is publicly announced or advertised",
        parent=events_node,
        critical=True
    )
    ann_claim = (
        "The December 2025 event/workshop schedule is publicly announced or advertised with date/time details on the provided page(s)."
    )
    await evaluator.verify(
        claim=ann_claim,
        node=ann_leaf,
        sources=_nonempty(store.event_urls),
        additional_instruction=(
            "The page should include date(s) and timing in December 2025, such as a calendar entry, event listing, or promotional post with specifics."
        )
    )

    # 4.c) Requirements (we adjust criticality to satisfy framework constraints)
    # The parent 'events' is critical; therefore all children must be critical in this framework.
    req_node = evaluator.add_parallel(
        id=f"store_{store_id}_events_requirements",
        desc="Event meets registration and fee requirements",
        parent=events_node,
        critical=True  # Adjusted to meet framework rule: critical parent cannot have non-critical children
    )

    # 4.c.i) Free events: no advance registration/fees required (conditional, treated as critical check)
    free_leaf = evaluator.add_leaf(
        id=f"store_{store_id}_events_free_no_registration",
        desc="If free event, no advance registration or fees required",
        parent=req_node,
        critical=True
    )
    free_claim = (
        "For the December 2025 in-store holiday craft event: "
        "If the event is free, the page indicates no advance registration or fees are required. "
        "If the event is paid, this condition is not applicable and should be considered satisfied."
    )
    await evaluator.verify(
        claim=free_claim,
        node=free_leaf,
        sources=_nonempty(store.event_urls),
        additional_instruction=(
            "Look for phrases like 'free drop-in', 'no registration', 'no RSVP', 'no fee'. "
            "If the event clearly charges a fee (paid workshop), treat this check as satisfied (N/A)."
        )
    )

    # 4.c.ii) Paid workshop duration >= 1.5 hours (conditional, treated as critical check)
    dur_leaf = evaluator.add_leaf(
        id=f"store_{store_id}_events_paid_duration",
        desc="If paid workshop, duration is at least 1.5 hours",
        parent=req_node,
        critical=True
    )
    dur_claim = (
        "For the December 2025 in-store holiday craft workshop: "
        "If the workshop is paid, its listed duration is at least 1.5 hours (90 minutes). "
        "If the event is free, this condition is not applicable and should be considered satisfied."
    )
    await evaluator.verify(
        claim=dur_claim,
        node=dur_leaf,
        sources=_nonempty(store.event_urls),
        additional_instruction=(
            "Accept durations such as '1 hour 30 minutes', '90 minutes', '2 hours', etc. "
            "If the event is free, mark this as satisfied (N/A)."
        )
    )

    # 4.c.iii) Materials cost stated or included for paid workshops (conditional, treated as critical check)
    mat_cost_leaf = evaluator.add_leaf(
        id=f"store_{store_id}_events_materials_cost",
        desc="Materials cost per person is clearly stated or included",
        parent=req_node,
        critical=True
    )
    mat_cost_claim = (
        "For the December 2025 in-store holiday craft workshop: "
        "If the workshop is paid, the page clearly states the materials cost per person or indicates materials are included. "
        "If the event is free, this condition is not applicable and should be considered satisfied."
    )
    await evaluator.verify(
        claim=mat_cost_claim,
        node=mat_cost_leaf,
        sources=_nonempty(store.event_urls),
        additional_instruction=(
            "Look for explicit fee breakdowns (e.g., 'Workshop $30, materials included' or 'Materials fee $10'). "
            "If the event is free, treat this as satisfied (N/A)."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
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
    """
    Evaluate an answer for the DC/MD craft stores 2025 holiday task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # 4 stores evaluated independently
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

    # Extract structured store list from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_stores(),
        template_class=StoresExtraction,
        extraction_name="stores_extraction"
    )

    # Normalize to exactly 4 stores (pad with empty placeholders if fewer)
    stores: List[StoreItem] = list(extracted.stores[:4])
    while len(stores) < 4:
        stores.append(StoreItem())

    # Build verification subtrees for each store
    tasks = []
    for i in range(4):
        tasks.append(verify_store(evaluator, root, stores[i], i))
    # Run verifications sequentially (safer for dependent custom nodes); could also gather if desired
    for t in tasks:
        await t

    # Add a quick summary of requirements as ground truth reference
    evaluator.add_ground_truth({
        "jurisdictions": ["Washington, DC", "Maryland"],
        "materials_required_any_3": ["wood beads", "ribbon", "paint/brushes", "embroidery thread", "pipe cleaners"],
        "holiday_hours_2025": {
            "closed": ["2025-11-27 (Thanksgiving Day)", "2025-12-25 (Christmas Day)"],
            "open": ["2025-11-29 (Saturday after Thanksgiving)"],
            "weekday_close_at_least": "20:00"
        },
        "event_requirements_dec_2025": {
            "in_store": True,
            "public_announcement": True,
            "conditional_rules": {
                "free": "no advance registration or fees",
                "paid": ">= 1.5 hours and materials cost stated or included"
            }
        }
    }, gt_type="requirements")

    return evaluator.get_summary()