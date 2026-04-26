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
TASK_ID = "holiday_wreath_bf_2025_plan"
TASK_DESCRIPTION = (
    "I'm planning to host an adult fresh wreath-making workshop in Ann Arbor, Michigan in early December 2025, and I want to purchase all necessary supplies during Black Friday sales on November 28, 2025. "
    "The workshop will be 2-3 hours long for 8 adult participants (ages 18+). Please help me plan my Black Friday shopping trip by providing: "
    "(1) Which major craft store (Michaels or Hobby Lobby) I should shop at, considering their Black Friday 2025 opening and closing hours; "
    "(2) A recommended shopping time window that allows me to shop within the store's Black Friday hours; "
    "(3) A complete materials checklist including wreath forms (specify size: 12\" or 18\"), fresh greenery type (pine, fir, or cedar), assembly tools (floral wire and wire cutters), and any optional decorative elements; "
    "(4) Confirmation that my workshop duration (2-3 hours) and participant capacity (8 adults) align with standard practices for hands-on fresh wreath-making workshops; "
    "(5) Information about any special Black Friday sales or discounts available at the chosen store. "
    "Please provide URL references to support your recommendations for store hours, material requirements, and workshop standards."
)

# Expected Black Friday 2025 hours constraints (as per rubric)
MICHAELS_BF_OPEN = "7am"
MICHAELS_BF_CLOSE = "10pm"
HOBBY_LOBBY_BF_OPEN = "8am"
HOBBY_LOBBY_BF_CLOSE = "9pm"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PlanExtraction(BaseModel):
    # Store selection and hours
    chosen_store: Optional[str] = None
    store_open_time: Optional[str] = None       # e.g., "7am"
    store_close_time: Optional[str] = None      # e.g., "10pm"
    store_hours_urls: List[str] = Field(default_factory=list)

    # Shopping time window and rationale
    shopping_window_start: Optional[str] = None
    shopping_window_end: Optional[str] = None
    shopping_window_rationale: Optional[str] = None

    # Materials: wreath base
    wreath_form_size: Optional[str] = None          # e.g., '12"' or '18"' or "12-inch"
    wreath_form_quantity: Optional[str] = None      # e.g., "8", "one per participant"
    wreath_materials_urls: List[str] = Field(default_factory=list)

    # Materials: fresh greenery
    greenery_types: List[str] = Field(default_factory=list)  # e.g., ["pine", "cedar"]
    greenery_quantity_guidance: Optional[str] = None
    greenery_urls: List[str] = Field(default_factory=list)

    # Tools
    tools_included: List[str] = Field(default_factory=list)  # e.g., ["floral wire", "wire cutters"]
    tools_urls: List[str] = Field(default_factory=list)

    # Optional decorative elements
    optional_decor: List[str] = Field(default_factory=list)

    # Workshop logistics confirmations + URLs
    duration_confirm_text: Optional[str] = None
    duration_urls: List[str] = Field(default_factory=list)
    capacity_confirm_text: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)
    age_requirement_confirm_text: Optional[str] = None
    age_urls: List[str] = Field(default_factory=list)

    # Sales and location
    sales_info_text: Optional[str] = None
    sales_urls: List[str] = Field(default_factory=list)
    store_address: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
Extract the following information from the answer text, strictly as presented. Do NOT invent or infer any information not explicitly present.

1) Store Selection & Hours
- chosen_store: The selected store, exactly "Michaels" or "Hobby Lobby" (return null if not clearly specified).
- store_open_time: Black Friday 2025 opening time for the chosen store (e.g., "7am", return null if missing).
- store_close_time: Black Friday 2025 closing time for the chosen store (e.g., "10pm", return null if missing).
- store_hours_urls: All URL(s) cited that explicitly support or state the Black Friday hours for the chosen store.

2) Shopping Time Window
- shopping_window_start: The recommended shopping window start time (e.g., "9am", return null if missing).
- shopping_window_end: The recommended shopping window end time (e.g., "11am", return null if not provided).
- shopping_window_rationale: Any rationale provided for the timing (e.g., “less crowded, better selection”; return null if missing).

3) Materials: Wreath Base
- wreath_form_size: The specified wreath form size (e.g., '12"', '18"', '12-inch' or '18-inch'; return null if missing).
- wreath_form_quantity: The quantity guidance for wreath forms (e.g., "8", "one per participant", "8 forms"; return null if missing).
- wreath_materials_urls: All URL(s) cited that support the wreath base requirements (sizes or one base per participant).

4) Materials: Fresh Greenery
- greenery_types: Array of greenery types listed; only include from: ["pine","fir","cedar"] exactly if any are mentioned.
- greenery_quantity_guidance: Any quantity guidance for greenery (e.g., “X bunches per wreath”, “Y pounds”, “Z stems”; return null if missing).
- greenery_urls: All URL(s) cited that support greenery requirements.

5) Tools
- tools_included: Array of tool names included in the materials list (e.g., "floral wire", "wire cutters", "pruning shears").
- tools_urls: All URL(s) cited that support the assembly tools requirements.

6) Optional Decor
- optional_decor: Array of optional decorative elements listed (e.g., "ribbon", "bows", "pinecones", "berries").

7) Workshop Logistics Confirmation
- duration_confirm_text: Text (if any) that confirms 2–3 hours is appropriate for wreath-making workshops (return null if missing).
- duration_urls: All URL(s) cited supporting typical wreath workshop duration.
- capacity_confirm_text: Text (if any) that confirms 8 participants is within a standard effective range (return null if missing).
- capacity_urls: All URL(s) cited supporting typical workshop size standards.
- age_requirement_confirm_text: Text (if any) that confirms 18+ as an adult workshop standard (return null if missing).
- age_urls: All URL(s) cited supporting adult workshop age standards.

8) Sales & Location
- sales_info_text: Text (if any) describing Black Friday sales/discounts for the chosen store (return null if missing).
- sales_urls: All URL(s) cited confirming sales details.
- store_address: The specific Ann Arbor, Michigan store address provided, if any (return null if missing).
- location_urls: All URL(s) cited confirming the store location/address.

Special URL rules:
- Only extract URLs that actually appear in the answer (plain URL or markdown link).
- Include full URLs with protocol where possible.
- If no URLs are given for a field, return an empty list for that field.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_store(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    l = s.strip().lower()
    if "michaels" in l:
        return "Michaels"
    if "hobby lobby" in l:
        return "Hobby Lobby"
    return None


def _has_any_keyword(s: Optional[str], keywords: List[str]) -> bool:
    if not s:
        return False
    low = s.lower()
    return any(k in low for k in keywords)


def _list_contains_keyword(lst: List[str], keywords: List[str]) -> bool:
    for item in lst or []:
        if _has_any_keyword(item, keywords):
            return True
    return False


def _quantity_implies_eight(q: Optional[str]) -> bool:
    if not q:
        return False
    low = q.lower()
    # Numeric check
    digits = "".join(ch for ch in low if ch.isdigit())
    try:
        if digits and int(digits) >= 8:
            return True
    except Exception:
        pass
    # Phrase-based check
    if "per participant" in low or "each participant" in low or "for 8" in low or "eight" in low:
        return True
    return False


async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    claim: str,
    leaf_id: str,
    leaf_desc: str,
    parent,
    urls: List[str],
    critical: bool = True,
    add_ins: str = "None"
):
    """
    Convenience: if URLs provided, perform URL-based verification; otherwise mark leaf failed.
    """
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=leaf_desc,
        parent=parent,
        critical=critical
    )
    if urls and len(urls) > 0:
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=add_ins
        )
    else:
        node.score = 0.0
        node.status = "failed"
    return node


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_store_selection_and_timing(evaluator: Evaluator, root, ex: PlanExtraction):
    # Parent (make non-critical to allow optional rationale as non-critical child)
    parent = evaluator.add_parallel(
        id="Store_Selection_and_Timing",
        desc="Identify the optimal craft store and shopping time window for Black Friday 2025",
        parent=root,
        critical=False
    )

    # Store Choice Identification (sequential)
    sci = evaluator.add_sequential(
        id="Store_Choice_Identification",
        desc="Clearly identify which craft store (Michaels or Hobby Lobby) to shop at",
        parent=parent,
        critical=False
    )

    # Store_Name_Specified (critical)
    chosen = _normalize_store(ex.chosen_store)
    evaluator.add_custom_node(
        result=(chosen in ("Michaels", "Hobby Lobby")),
        id="Store_Name_Specified",
        desc="Answer explicitly names Michaels or Hobby Lobby as the chosen store",
        parent=sci,
        critical=True
    )

    # Store_Hours_Accuracy (critical)
    hours_claim = (
        f"Given the chosen store '{chosen}', the provided Black Friday 2025 hours "
        f"('{ex.store_open_time}' to '{ex.store_close_time}') match the constraint: "
        f"Michaels should be {MICHAELS_BF_OPEN}-{MICHAELS_BF_CLOSE} and "
        f"Hobby Lobby should be {HOBBY_LOBBY_BF_OPEN}-{HOBBY_LOBBY_BF_CLOSE}."
    )
    node_hours_accuracy = evaluator.add_leaf(
        id="Store_Hours_Accuracy",
        desc="Answer provides Black Friday 2025 hours that match constraints: Michaels (7am-10pm) or Hobby Lobby (8am-9pm)",
        parent=sci,
        critical=True
    )
    await evaluator.verify(
        claim=hours_claim,
        node=node_hours_accuracy,
        additional_instruction="Interpret times in local store time. Allow minor formatting differences like '7:00 AM' vs '7am'."
    )

    # Store_Hours_URL_Reference (critical)
    await _verify_with_urls_or_fail(
        evaluator,
        claim=f"The page confirms {chosen} Black Friday 2025 hours are from {ex.store_open_time} to {ex.store_close_time} (local time).",
        leaf_id="Store_Hours_URL_Reference",
        leaf_desc="Answer includes URL reference confirming the Black Friday hours for the chosen store",
        parent=sci,
        urls=ex.store_hours_urls,
        critical=True,
        add_ins="Confirm that the page explicitly mentions Black Friday hours (or holiday hours specific to the day) matching the stated open/close times."
    )

    # Shopping_Time_Window (parallel)
    stw = evaluator.add_parallel(
        id="Shopping_Time_Window",
        desc="Provide a specific recommended shopping time window",
        parent=parent,
        critical=False
    )

    # Time_Window_Specified (critical)
    evaluator.add_custom_node(
        result=bool(ex.shopping_window_start and ex.shopping_window_start.strip()),
        id="Time_Window_Specified",
        desc="Answer specifies a shopping time window with start time (and optionally end time)",
        parent=stw,
        critical=True
    )

    # Time_Within_Store_Hours (critical)
    within_claim = (
        f"The recommended shopping window from '{ex.shopping_window_start}'"
        f"{f' to {ex.shopping_window_end}' if ex.shopping_window_end else ''} "
        f"falls within the {chosen} Black Friday hours '{ex.store_open_time}' to '{ex.store_close_time}'."
    )
    node_within = evaluator.add_leaf(
        id="Time_Within_Store_Hours",
        desc="Specified shopping time falls within the chosen store's Black Friday hours",
        parent=stw,
        critical=True
    )
    await evaluator.verify(
        claim=within_claim,
        node=node_within,
        additional_instruction="Treat time comparisons in the same local timezone. Allow reasonable rounding or formatting differences."
    )

    # Timing_Rationale (non-critical)
    evaluator.add_custom_node(
        result=bool(ex.shopping_window_rationale and ex.shopping_window_rationale.strip()),
        id="Timing_Rationale",
        desc="Answer explains reasoning for the chosen time window (e.g., early access, better selection)",
        parent=stw,
        critical=False
    )


async def build_essential_materials(evaluator: Evaluator, root, ex: PlanExtraction):
    parent = evaluator.add_parallel(
        id="Essential_Workshop_Materials",
        desc="Complete checklist of required materials for fresh wreath-making workshop",
        parent=root,
        critical=False
    )

    # Wreath_Base_Components (critical, all children critical)
    wbc = evaluator.add_parallel(
        id="Wreath_Base_Components",
        desc="Core structural materials for wreath construction",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_any_keyword(ex.wreath_form_size, ["12", "18"]),
        id="Wreath_Form_Size_Specification",
        desc="Answer specifies wreath forms in 12-inch or 18-inch diameter",
        parent=wbc,
        critical=True
    )

    evaluator.add_custom_node(
        result=_quantity_implies_eight(ex.wreath_form_quantity),
        id="Wreath_Form_Quantity_Specification",
        desc="Answer indicates obtaining wreath forms for all workshop participants (8 forms needed)",
        parent=wbc,
        critical=True
    )

    await _verify_with_urls_or_fail(
        evaluator,
        claim="The page indicates appropriate wreath base/form sizes (12-inch or 18-inch) and that each participant needs one base.",
        leaf_id="Wreath_Materials_URL_Reference",
        leaf_desc="Answer includes URL reference supporting wreath form requirements",
        parent=wbc,
        urls=ex.wreath_materials_urls,
        critical=True,
        add_ins="Accept pages that recommend standard wreath base sizes (e.g., 12–18 inches) and/or indicate one base per person/wreath."
    )

    # Fresh_Greenery_Requirements (critical)
    fgr = evaluator.add_parallel(
        id="Fresh_Greenery_Requirements",
        desc="Fresh evergreen branches needed for wreath decoration",
        parent=parent,
        critical=True
    )

    greenery_types_norm = [t.strip().lower() for t in (ex.greenery_types or [])]
    evaluator.add_custom_node(
        result=any(t in {"pine", "fir", "cedar"} for t in greenery_types_norm),
        id="Greenery_Type_Specification",
        desc="Answer specifies acceptable fresh greenery types: pine, fir, or cedar branches",
        parent=fgr,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(ex.greenery_quantity_guidance and ex.greenery_quantity_guidance.strip()),
        id="Greenery_Quantity_Guidance",
        desc="Answer provides guidance on quantity of greenery needed for the workshop",
        parent=fgr,
        critical=True
    )

    await _verify_with_urls_or_fail(
        evaluator,
        claim="The page supports using fresh evergreen branches (pine, fir, or cedar) and offers quantity guidance suitable for one wreath.",
        leaf_id="Greenery_Materials_URL_Reference",
        leaf_desc="Answer includes URL reference supporting fresh greenery requirements",
        parent=fgr,
        urls=ex.greenery_urls,
        critical=True,
        add_ins="Look for mentions of appropriate greenery types (pine/fir/cedar) and guideline amounts (e.g., bunches, stems, or pounds per wreath)."
    )

    # Assembly_Tools_and_Wire (critical)
    atw = evaluator.add_parallel(
        id="Assembly_Tools_and_Wire",
        desc="Tools required for attaching greenery to wreath forms",
        parent=parent,
        critical=True
    )

    tools_lower = [t.lower() for t in (ex.tools_included or [])]

    evaluator.add_custom_node(
        result=_list_contains_keyword(tools_lower, ["floral wire"]),
        id="Floral_Wire_Included",
        desc="Answer includes floral wire in the materials list",
        parent=atw,
        critical=True
    )

    evaluator.add_custom_node(
        result=_list_contains_keyword(tools_lower, ["wire cutters", "wire cutter", "cutter", "cutters", "snips", "pruning shears", "pruners"]),
        id="Wire_Cutters_Included",
        desc="Answer includes wire cutters or cutting tools in the materials list",
        parent=atw,
        critical=True
    )

    await _verify_with_urls_or_fail(
        evaluator,
        claim="The page supports including floral wire and cutting tools (e.g., wire cutters/shears) for assembling fresh wreaths.",
        leaf_id="Tools_URL_Reference",
        leaf_desc="Answer includes URL reference supporting assembly tools requirements",
        parent=atw,
        urls=ex.tools_urls,
        critical=True,
        add_ins="Look for mentions of floral wire and wire cutters (or equivalent cutting tools) as required for wreath assembly."
    )

    # Optional_Decorative_Elements (non-critical)
    ode = evaluator.add_parallel(
        id="Optional_Decorative_Elements",
        desc="Additional decorative items for wreath embellishment",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(ex.optional_decor),
        id="Decorative_Elements_Listed",
        desc="Answer mentions optional decorative elements such as ribbon, bows, pinecones, or berries",
        parent=ode,
        critical=False
    )


async def build_workshop_logistics(evaluator: Evaluator, root, ex: PlanExtraction):
    parent = evaluator.add_parallel(
        id="Workshop_Logistics_Confirmation",
        desc="Confirmation that workshop parameters align with standard practices",
        parent=root,
        critical=False
    )

    # Duration
    dur = evaluator.add_parallel(
        id="Duration_Alignment_Confirmation",
        desc="Confirm 2-3 hour workshop duration aligns with standards",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(ex.duration_confirm_text and ex.duration_confirm_text.strip()),
        id="Duration_Standard_Cited",
        desc="Answer confirms that 2-3 hours is appropriate for fresh wreath-making workshops based on standard practices",
        parent=dur,
        critical=True
    )

    await _verify_with_urls_or_fail(
        evaluator,
        claim="The page indicates that fresh wreath-making workshops typically take about 2–3 hours.",
        leaf_id="Duration_URL_Reference",
        leaf_desc="Answer includes URL reference confirming typical wreath workshop duration standards",
        parent=dur,
        urls=ex.duration_urls,
        critical=True,
        add_ins="Allow slight phrasing differences like 'around two hours' or '2–3 hours typical'."
    )

    # Participant Capacity
    cap = evaluator.add_parallel(
        id="Participant_Capacity_Alignment_Confirmation",
        desc="Confirm 8 participants aligns with standards for hands-on workshops",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(ex.capacity_confirm_text and ex.capacity_confirm_text.strip()),
        id="Capacity_Standard_Cited",
        desc="Answer confirms that 8 participants falls within the standard range (4-10) for effective hands-on instruction",
        parent=cap,
        critical=True
    )

    await _verify_with_urls_or_fail(
        evaluator,
        claim="The page supports that small hands-on craft workshops often run effectively with about 4–10 participants (8 is typical within this range).",
        leaf_id="Capacity_URL_Reference",
        leaf_desc="Answer includes URL reference confirming typical workshop size standards",
        parent=cap,
        urls=ex.capacity_urls,
        critical=True,
        add_ins="Accept pages that describe ideal class sizes for hands-on craft instruction around 4–10 participants."
    )

    # Age Requirement
    age = evaluator.add_parallel(
        id="Age_Requirement_Alignment_Confirmation",
        desc="Confirm 18+ age requirement aligns with adult workshop standards",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(ex.age_requirement_confirm_text and ex.age_requirement_confirm_text.strip()),
        id="Age_Standard_Cited",
        desc="Answer confirms that 18+ age requirement aligns with typical adult workshop standards",
        parent=age,
        critical=True
    )

    await _verify_with_urls_or_fail(
        evaluator,
        claim="The page indicates that adult workshops commonly set an age minimum at 18+.",
        leaf_id="Age_Requirement_URL_Reference",
        leaf_desc="Answer includes URL reference confirming typical adult workshop age requirements",
        parent=age,
        urls=ex.age_urls,
        critical=True,
        add_ins="Accept pages that define 'adult classes' or 'adult workshops' as 18+ or similar."
    )


async def build_sales_and_location(evaluator: Evaluator, root, ex: PlanExtraction):
    parent = evaluator.add_parallel(
        id="Sales_and_Location_Information",
        desc="Black Friday sales details and store location information",
        parent=root,
        critical=False
    )

    # Sales
    sales = evaluator.add_parallel(
        id="Black_Friday_Sales_Details",
        desc="Information about Black Friday sale periods and discounts at the chosen store",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(ex.sales_info_text and ex.sales_info_text.strip()),
        id="Sale_Information_Provided",
        desc="Answer provides information about Black Friday sales or discounts (e.g., Michaels 70% off, sale period Nov 21-30)",
        parent=sales,
        critical=False
    )

    await _verify_with_urls_or_fail(
        evaluator,
        claim="The page confirms Black Friday sales or discount promotions for the chosen store around Nov 28, 2025 (e.g., percentage-off deals, multi-day sales).",
        leaf_id="Sales_URL_Reference",
        leaf_desc="Answer includes URL reference confirming sales details",
        parent=sales,
        urls=ex.sales_urls,
        critical=False,
        add_ins="Check that the page indicates Black Friday or holiday sales/discounts relevant to late November 2025."
    )

    # Location
    loc = evaluator.add_parallel(
        id="Store_Location_Details",
        desc="Specific store location information for Ann Arbor, Michigan",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(ex.store_address and ex.store_address.strip()),
        id="Ann_Arbor_Address_Provided",
        desc="Answer provides the specific store address in Ann Arbor (e.g., Michaels at 3655 Washtenaw Ave)",
        parent=loc,
        critical=False
    )

    await _verify_with_urls_or_fail(
        evaluator,
        claim=f"The page lists the Ann Arbor, MI address for the chosen store matching: {ex.store_address}.",
        leaf_id="Location_URL_Reference",
        leaf_desc="Answer includes URL reference confirming store location",
        parent=loc,
        urls=ex.location_urls,
        critical=False,
        add_ins="Accept store locator pages, Google Maps business pages, or the store’s official location page that show this address."
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
    Evaluate the answer for the Black Friday wreath-making workshop supply shopping plan.
    """
    # Initialize evaluator (root non-critical parallel to allow partial scores and non-critical children)
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

    # Extract structured plan from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction"
    )

    # Build verification subtrees
    await build_store_selection_and_timing(evaluator, root, extracted)
    await build_essential_materials(evaluator, root, extracted)
    await build_workshop_logistics(evaluator, root, extracted)
    await build_sales_and_location(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()