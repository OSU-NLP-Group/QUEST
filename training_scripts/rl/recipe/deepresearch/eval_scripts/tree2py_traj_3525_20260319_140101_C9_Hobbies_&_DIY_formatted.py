import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task metadata
# -----------------------------------------------------------------------------
TASK_ID = "holiday_workshop_shopping_plan_2025"
TASK_DESCRIPTION = """
You are planning to host a holiday craft workshop on December 26, 2025, and need to purchase all supplies on Christmas Eve (December 24, 2025) from major craft retailers before they close for the holiday. The workshop will have three craft stations for participants:

1. Wreath Making Station — Participants will create decorative holiday wreaths
2. Ornament Decorating Station — Participants will decorate clear plastic ornaments
3. Gift Basket Assembly Station — Participants will assemble and wrap gift baskets

Create a comprehensive shopping plan that includes:

- A verified shopping timeline showing the latest time you can complete shopping at Michaels or Hobby Lobby on Christmas Eve 2025, including specific store closing times
- A complete supply list for each of the three craft stations, with specific requirements for:
  - Wreath Making: wreath bases (with size specifications), deco mesh (width and quantity per wreath), wide ribbon (width and quantity), narrow ribbon (width and quantity), attachment supplies (pipe cleaners or zip ties), and cutting tools
  - Ornament Decorating: clear plastic ornaments (type and quantity), paints or markers (type suitable for ornaments), adhesive supplies (hot glue or craft glue), and embellishments (variety for decoration)
  - Gift Basket Assembly: baskets (quantity and appropriate size), cellophane wrap (standard width dimensions and roll length), ribbon (for tying baskets), and filler material (type such as shredded paper or crinkle cut)
- For each supply item, provide the specific dimensions, quantities, or specifications that are standard for that type of craft project
- Include URL references from your research that verify the supply requirements and store information

Your shopping plan must ensure that all required supplies can be obtained from Michaels and/or Hobby Lobby before their Christmas Eve 2025 closing times, and all supply specifications must align with standard practices for these craft projects as documented in craft supply guides and tutorials.
""".strip()


# -----------------------------------------------------------------------------
# Extraction data models
# -----------------------------------------------------------------------------
class StoreHoursInfo(BaseModel):
    closing_time: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TimelineExtraction(BaseModel):
    shopping_date_text: Optional[str] = None
    latest_completion_time: Optional[str] = None
    michaels: StoreHoursInfo = Field(default_factory=StoreHoursInfo)
    hobby_lobby: StoreHoursInfo = Field(default_factory=StoreHoursInfo)


class WreathSupplies(BaseModel):
    wreath_base_size: Optional[str] = None
    deco_mesh_width: Optional[str] = None
    deco_mesh_quantity_per_wreath: Optional[str] = None
    wide_ribbon_width: Optional[str] = None
    wide_ribbon_rolls_per_wreath: Optional[str] = None
    narrow_ribbon_width: Optional[str] = None
    narrow_ribbon_rolls_per_wreath: Optional[str] = None
    attachments_type: Optional[str] = None
    attachments_quantity_per_wreath: Optional[str] = None
    cutting_tools: List[str] = Field(default_factory=list)
    supply_urls: List[str] = Field(default_factory=list)


class OrnamentSupplies(BaseModel):
    ornaments_type: Optional[str] = None
    ornaments_quantity_per_participant: Optional[str] = None
    paints_or_markers_type: Optional[str] = None
    adhesive_type: Optional[str] = None
    embellishments_desc: Optional[str] = None
    supply_urls: List[str] = Field(default_factory=list)


class GiftBasketSupplies(BaseModel):
    basket_size: Optional[str] = None
    basket_quantity_per_participant: Optional[str] = None
    cellophane_width: Optional[str] = None
    cellophane_roll_length: Optional[str] = None
    ribbon_type: Optional[str] = None
    filler_type: Optional[str] = None
    scissors_included: Optional[bool] = None
    supply_urls: List[str] = Field(default_factory=list)


class SourcingAndScaling(BaseModel):
    retailer_statement: Optional[str] = None
    participant_count: Optional[str] = None
    uses_per_participant_quantities: Optional[bool] = None
    total_quantities_listed: Optional[bool] = None
    other_urls: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompts
# -----------------------------------------------------------------------------
def prompt_extract_timeline() -> str:
    return """
Extract the shopping timeline details as they appear in the answer.

Fields to extract:
- shopping_date_text: The exact text indicating the shopping date (e.g., "Christmas Eve 2025", "December 24, 2025").
- latest_completion_time: The specific latest time by which shopping should be completed (e.g., "5:15 PM", "by 5:30 PM").
- michaels.closing_time: The stated closing time for Michaels on Christmas Eve 2025, if given (e.g., "6:00 PM").
- michaels.urls: All URLs in the answer that are explicitly provided to verify Michaels store hours for Christmas Eve.
- hobby_lobby.closing_time: The stated closing time for Hobby Lobby on Christmas Eve 2025, if given (e.g., "5:30 PM").
- hobby_lobby.urls: All URLs in the answer that are explicitly provided to verify Hobby Lobby store hours for Christmas Eve.

Rules:
- Do not invent information; only extract what the answer explicitly states.
- For any missing field, return null (for strings) or an empty list (for URLs).
- Only include full URLs that are explicitly present in the answer.
""".strip()


def prompt_extract_wreath_supplies() -> str:
    return """
Extract the wreath station supply details, including specs and per-wreath quantities as written in the answer.

Fields:
- wreath_base_size: Size/dimension of the wreath base/form (e.g., "14-inch", "16-inch", "18-inch").
- deco_mesh_width: Width of deco mesh (e.g., "10 inch").
- deco_mesh_quantity_per_wreath: Quantity needed per wreath (e.g., "1 roll per wreath", "20 yards per wreath").
- wide_ribbon_width: Width of wide ribbon (e.g., "2.5 inch").
- wide_ribbon_rolls_per_wreath: Quantity of wide ribbon per wreath (e.g., "2 rolls").
- narrow_ribbon_width: Width of narrow ribbon (e.g., "1.5 inch").
- narrow_ribbon_rolls_per_wreath: Quantity of narrow ribbon per wreath (e.g., "2 rolls").
- attachments_type: Attachment supplies type (e.g., "pipe cleaners", "zip ties").
- attachments_quantity_per_wreath: Quantity per wreath (e.g., "10–15 per wreath").
- cutting_tools: List of cutting tools mentioned (e.g., ["wire cutters", "scissors"]).
- supply_urls: All URLs in the answer used as supply guides/tutorials for the wreath station.

Rules:
- Extract exactly as written; do not infer.
- Use null for missing text fields, and [] for missing lists.
- Only include full URLs explicitly present in the answer.
""".strip()


def prompt_extract_ornament_supplies() -> str:
    return """
Extract the ornament decorating station supply details, including specs and per-participant quantities as written.

Fields:
- ornaments_type: Type and size of clear plastic ornaments (e.g., "clear plastic shatterproof ball ornaments, 70mm").
- ornaments_quantity_per_participant: Quantity per participant (e.g., "2 ornaments per participant").
- paints_or_markers_type: Paint or marker types suitable for ornaments (e.g., "acrylic paint", "oil-based paint pens").
- adhesive_type: Adhesives specified (e.g., "hot glue", "craft glue").
- embellishments_desc: Description/variety of embellishments (e.g., "glitter, sequins, ribbon").
- supply_urls: All URLs in the answer used as supply guides/tutorials for the ornament station.

Rules:
- Extract exactly as written; no inference.
- Use null for missing text fields and [] for URLs.
""".strip()


def prompt_extract_giftbasket_supplies() -> str:
    return """
Extract the gift basket station supply details, including specs and per-participant quantities as written.

Fields:
- basket_size: Appropriate basket/container size (e.g., "medium 12–14 inch", "10x12 inch").
- basket_quantity_per_participant: Quantity per participant (e.g., "1 basket per participant").
- cellophane_width: Stated cellophane roll width (e.g., "24 inch", "30 inch").
- cellophane_roll_length: Stated roll length (e.g., "100 ft", "150 ft").
- ribbon_type: Type/spec for tying (e.g., "curling ribbon", "fabric ribbon 1.5 inch").
- filler_type: Filler material (e.g., "shredded paper", "crinkle cut").
- scissors_included: true/false if scissors are included for this station in the list.
- supply_urls: All URLs in the answer used as supply guides/tutorials for the gift basket station.

Rules:
- Extract exactly as written; do not infer.
- Use null for missing text fields, false for missing boolean, [] for missing lists.
""".strip()


def prompt_extract_sourcing_and_scaling() -> str:
    return """
Extract statements about sourcing from retailers and how quantities scale for a group workshop.

Fields:
- retailer_statement: The exact sentence(s) indicating that required supplies can be obtained from Michaels and/or Hobby Lobby (and not relying on other retailers).
- participant_count: If an explicit participant count is stated for totals, extract it (e.g., "20 participants"); else null.
- uses_per_participant_quantities: true/false depending on whether per-participant or per-project quantities are provided in the plan (across stations).
- total_quantities_listed: true/false indicating whether totals derived from a participant count are provided.
- other_urls: Any additional URLs in the answer not already captured that relate to sourcing or supply guidance (exclude the already captured station supply URLs and store-hours URLs, if specified explicitly in the answer).

Rules:
- Extract only what is explicitly written.
- Use null for missing strings, false for booleans if not present, [] for missing lists.
""".strip()


# -----------------------------------------------------------------------------
# Verification helpers
# -----------------------------------------------------------------------------
async def verify_timeline_and_finish(
    evaluator: Evaluator,
    parent,
    timeline: TimelineExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="timeline_and_latest_finish_time",
        desc="Provides a Christmas Eve 2025 (Dec 24, 2025) shopping timeline that includes store closing times and the latest time shopping can be completed.",
        parent=parent,
        critical=True,
    )

    # 1) Date alignment with Christmas Eve 2025 (Dec 24, 2025)
    date_node = evaluator.add_leaf(
        id="includes_christmas_eve_date_alignment",
        desc="Timeline explicitly indicates shopping occurs on Christmas Eve 2025 (Dec 24, 2025).",
        parent=node,
        critical=True,
    )
    claim_date = "The shopping timeline explicitly indicates that shopping occurs on Christmas Eve 2025 (December 24, 2025)."
    await evaluator.verify(
        claim=claim_date,
        node=date_node,
        additional_instruction="Judge this only by whether the answer explicitly names 'Christmas Eve 2025' or 'December 24, 2025' in the shopping timeline."
    )

    # 2) Includes required store closing times
    times_node = evaluator.add_leaf(
        id="includes_required_store_closing_times",
        desc="Timeline states Michaels closes at 6:00 PM and Hobby Lobby closes at 5:30 PM on Dec 24, 2025.",
        parent=node,
        critical=True,
    )
    claim_times = "The timeline states that Michaels closes at 6:00 PM and Hobby Lobby closes at 5:30 PM on December 24, 2025."
    await evaluator.verify(
        claim=claim_times,
        node=times_node,
        additional_instruction="Focus on whether BOTH closing times are present as written, allowing for small formatting variants like '6 PM' or '5:30 p.m.'."
    )

    # 3) Latest completion time respects whichever store(s) are used
    latest_node = evaluator.add_leaf(
        id="latest_completion_time_respects_closing",
        desc="Timeline states a latest completion time for shopping at Michaels or Hobby Lobby that is not later than the stated closing time(s) for the store(s) used.",
        parent=node,
        critical=True,
    )
    claim_latest = (
        "The plan's stated 'latest completion time' for shopping is not later than the closing time(s) of the store(s) it uses on December 24, 2025 "
        "(i.e., not later than 6:00 PM for Michaels and not later than 5:30 PM for Hobby Lobby; if using both stores, the latest time should be "
        "no later than the earlier closing among them)."
    )
    await evaluator.verify(
        claim=claim_latest,
        node=latest_node,
        additional_instruction="Use the answer's own latest time and stated store usage to judge whether this constraint is satisfied. If the plan uses both stores, the latest time must be no later than 5:30 PM."
    )


async def verify_station_supply_lists(
    evaluator: Evaluator,
    parent,
    wreath: WreathSupplies,
    ornament: OrnamentSupplies,
    gift: GiftBasketSupplies,
) -> None:
    node = evaluator.add_parallel(
        id="station_supply_lists_complete",
        desc="Provides supply lists for all three stations with standard specifications/dimensions and quantities, matching the required components.",
        parent=parent,
        critical=True,
    )

    # Wreath station requirements
    wreath_node = evaluator.add_leaf(
        id="wreath_station_requirements_met",
        desc="Wreath station list includes: wreath base/form (standard size such as 14/16/18-inch diameter), deco mesh (~10-inch width; quantity per wreath), ribbon in two widths (~2.5\" and ~1.5\"; with quantities), attachment supplies (pipe cleaners or zip ties), and cutting tools; each with specifications/quantities.",
        parent=node,
        critical=True,
    )
    claim_wreath = (
        "The wreath station supply list in the answer includes ALL of the following with specific sizes/specs/quantities: "
        "1) a wreath base/form with a standard diameter (around 14–18 inches) and the size used is stated; "
        "2) deco mesh of about 10-inch width and a stated quantity per wreath; "
        "3) ribbon in two widths (about 2.5 inch and about 1.5 inch) with the number of rolls or length per wreath; "
        "4) attachment supplies (pipe cleaners or zip ties) with a per-wreath quantity; and "
        "5) cutting tools (e.g., wire cutters/scissors)."
    )
    await evaluator.verify(
        claim=claim_wreath,
        node=wreath_node,
        additional_instruction="Accept minor phrasing/format differences. Check presence and specificity within the answer."
    )

    # Ornament station requirements
    ornament_node = evaluator.add_leaf(
        id="ornament_station_requirements_met",
        desc="Ornament station list includes: clear plastic ornaments (type and quantity), paints/markers suitable for ornaments (type and quantity/coverage), adhesive supplies (hot glue and/or craft glue), and embellishments (variety/coverage).",
        parent=node,
        critical=True,
    )
    claim_ornament = (
        "The ornament station supply list in the answer includes ALL of the following with explicit specs/quantities: "
        "1) clear plastic ornaments with a stated type/size and per-participant quantity; "
        "2) paints or markers suitable for ornaments (e.g., acrylic paints or oil-based paint pens), including type/sufficiency; "
        "3) adhesive supplies (hot glue and/or craft glue) with availability/coverage; and "
        "4) a variety of embellishments (e.g., glitter, sequins, ribbon) with sufficient quantity/coverage details."
    )
    await evaluator.verify(
        claim=claim_ornament,
        node=ornament_node,
        additional_instruction="Judge by completeness and specificity as written in the plan."
    )

    # Gift basket station requirements
    gift_node = evaluator.add_leaf(
        id="gift_basket_station_requirements_met",
        desc="Gift basket station list includes: baskets/containers (quantity and appropriate size), cellophane wrap (common widths such as 16/24/30 inches and roll length typically ≥100 ft), ribbon for tying (spec/type and quantity), scissors, and filler material (e.g., shredded/crinkle cut and quantity).",
        parent=node,
        critical=True,
    )
    claim_gift = (
        "The gift basket station supply list in the answer includes ALL of the following with explicit specs/quantities: "
        "1) baskets/containers with appropriate size and per-participant quantity; "
        "2) cellophane wrap specifying a common width (e.g., 16, 24, or 30 inches) and a roll length around or above 100 ft; "
        "3) ribbon for tying with type/width and quantity; "
        "4) scissors; and "
        "5) filler material (e.g., shredded paper or crinkle cut) with sufficient quantity."
    )
    await evaluator.verify(
        claim=claim_gift,
        node=gift_node,
        additional_instruction="Confirm all listed components are present with practical specs/quantities for a workshop."
    )


async def verify_group_scaling(
    evaluator: Evaluator,
    parent,
    sourcing_scaling: SourcingAndScaling,
) -> None:
    node = evaluator.add_leaf(
        id="group_workshop_quantity_scaling",
        desc="Quantities are presented to support a group workshop (per-participant/per-project or totals from a participant count) across all three stations.",
        parent=parent,
        critical=True,
    )
    claim = (
        "The plan presents quantities in a group-workshop-scalable way: it provides either per-participant/per-project quantities or totals computed from an explicit participant count, and this applies across all three stations."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Look for per-participant numbers (e.g., '1 wreath per person', '2 ornaments per participant') or totals derived from a stated participant count."
    )


async def verify_retailer_sourcing(
    evaluator: Evaluator,
    parent,
    sourcing_scaling: SourcingAndScaling,
) -> None:
    node = evaluator.add_leaf(
        id="retailer_sourcing_constraint",
        desc="Plan indicates the required supplies can be obtained from Michaels and/or Hobby Lobby (does not rely on other retailers), consistent with completing purchases by Christmas Eve 2025 closing times.",
        parent=parent,
        critical=True,
    )
    claim = (
        "The plan explicitly indicates that all required supplies can be purchased from Michaels and/or Hobby Lobby on Christmas Eve 2025 and does not rely on other retailers to meet the required supply set; it is consistent with completing purchases by the stated closing times."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Judge only whether the plan states sourcing from Michaels and/or Hobby Lobby without depending on other retailers, and that this is compatible with the holiday hours stated in the plan."
    )


async def verify_url_references(
    evaluator: Evaluator,
    parent,
    timeline: TimelineExtraction,
    wreath: WreathSupplies,
    ornament: OrnamentSupplies,
    gift: GiftBasketSupplies,
) -> None:
    node = evaluator.add_leaf(
        id="url_references_verification",
        desc="Includes URL references that (a) verify the Christmas Eve 2025 store-hours information used and (b) support the supply standards/specifications used for the stations (at least one supply-guidance source overall).",
        parent=parent,
        critical=True,
    )

    claim = (
        "The answer includes URL references that: (a) provide store-hours information for Christmas Eve 2025 for Michaels and for Hobby Lobby; and "
        "(b) include at least one craft supply guide or tutorial that supports the supply specifications used for the stations."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Judge this purely by presence/coverage of URLs in the answer text: at least one Michaels hours URL, at least one Hobby Lobby hours URL, and at least one supply-guidance/tutorial URL supporting the listed supplies and specs. Do not assess the external content—only the inclusion and intended use as described."
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
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

    # Extract structured info from the answer (in parallel)
    timeline_task = evaluator.extract(
        prompt=prompt_extract_timeline(),
        template_class=TimelineExtraction,
        extraction_name="timeline_extraction",
    )
    wreath_task = evaluator.extract(
        prompt=prompt_extract_wreath_supplies(),
        template_class=WreathSupplies,
        extraction_name="wreath_supplies",
    )
    ornament_task = evaluator.extract(
        prompt=prompt_extract_ornament_supplies(),
        template_class=OrnamentSupplies,
        extraction_name="ornament_supplies",
    )
    gift_task = evaluator.extract(
        prompt=prompt_extract_giftbasket_supplies(),
        template_class=GiftBasketSupplies,
        extraction_name="gift_basket_supplies",
    )
    sourcing_task = evaluator.extract(
        prompt=prompt_extract_sourcing_and_scaling(),
        template_class=SourcingAndScaling,
        extraction_name="sourcing_and_scaling",
    )

    timeline, wreath_supplies, ornament_supplies, gift_supplies, sourcing_scaling = await asyncio.gather(
        timeline_task, wreath_task, ornament_task, gift_task, sourcing_task
    )

    # Optional: record GT-style expectations for transparency
    evaluator.add_ground_truth({
        "expected_store_closing_times_dec_24_2025": {
            "Michaels": "6:00 PM",
            "Hobby Lobby": "5:30 PM"
        },
        "typical_supply_specs_reference_points": {
            "wreath_base_diameter": "≈14–18 inches",
            "deco_mesh_width": "≈10 inches",
            "ribbon_widths": ["≈2.5 inch (wide)", "≈1.5 inch (narrow)"],
            "cellophane_common_widths": ["16 in", "24 in", "30 in"],
            "cellophane_roll_length": "≈100 ft or more"
        }
    }, gt_type="expectations")

    # Custom info: summarize URLs counts (for debugging/reporting)
    evaluator.add_custom_info(
        info={
            "michaels_hours_urls": timeline.michaels.urls,
            "hobby_lobby_hours_urls": timeline.hobby_lobby.urls,
            "wreath_supply_urls": wreath_supplies.supply_urls,
            "ornament_supply_urls": ornament_supplies.supply_urls,
            "gift_supply_urls": gift_supplies.supply_urls,
            "other_urls": sourcing_scaling.other_urls
        },
        info_type="url_collections",
        info_name="extracted_urls_summary"
    )

    # Build verification tree as specified by the rubric JSON
    # Root in Evaluator is non-critical by framework design; children are set critical per rubric.
    await verify_timeline_and_finish(evaluator, root, timeline)
    await verify_station_supply_lists(evaluator, root, wreath_supplies, ornament_supplies, gift_supplies)
    await verify_group_scaling(evaluator, root, sourcing_scaling)
    await verify_retailer_sourcing(evaluator, root, sourcing_scaling)
    await verify_url_references(evaluator, root, timeline, wreath_supplies, ornament_supplies, gift_supplies)

    return evaluator.get_summary()