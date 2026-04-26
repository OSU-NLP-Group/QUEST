import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "diy_christmas_workshop_materials_2026"
TASK_DESCRIPTION = (
    "You are planning to host a beginner-friendly DIY Christmas craft workshop where participants will create four "
    "specific holiday projects: (1) a cardboard advent calendar, (2) a fresh greenery Christmas wreath using a wreath "
    "form, (3) a sewn Christmas stocking with both outer fabric and lining, and (4) a water-filled decorative snow globe. "
    "For each of the four projects, provide a comprehensive list of the essential materials and tools needed, including "
    "any specific quantities where applicable (such as fabric yardage). Additionally, identify a major craft supply store "
    "chain that operates multiple locations throughout New York state where these materials could be purchased, and specify "
    "the exact number of locations this store chain has in New York state as of 2026."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class AdventCalendarExtraction(BaseModel):
    cardboard: Optional[bool] = None
    ruler: Optional[bool] = None
    glue_or_adhesive: Optional[bool] = None
    containers_24: Optional[bool] = None  # True only if 24 numbered containers/bags/pockets are explicitly listed


class WreathExtraction(BaseModel):
    wreath_form: Optional[bool] = None  # wreath form/ring/frame/base (wire, grapevine, foam, straw acceptable)
    florist_wire: Optional[bool] = None  # floral/paddle wire acceptable
    scissors_or_secateurs: Optional[bool] = None  # scissors, pruners, secateurs, garden shears acceptable
    moss: Optional[bool] = None  # sheet/sphagnum moss acceptable
    greenery: Optional[bool] = None  # fresh evergreen foliage/greenery


class StockingExtraction(BaseModel):
    outer_fabric_qty: Optional[str] = None  # should indicate 1/2 yard (0.5 yd, half yard, ½ yd)
    lining_fabric_qty: Optional[str] = None  # should indicate 1/2 yard
    batting_qty: Optional[str] = None  # should indicate 1/2 yard
    pattern: Optional[bool] = None  # stocking pattern/template is mentioned


class SnowGlobeExtraction(BaseModel):
    container: Optional[bool] = None  # glass jar, mason jar, or plastic globe ornament
    glycerin: Optional[bool] = None  # glycerin/glycerine
    distilled_water: Optional[bool] = None  # requires "distilled water" (or phrasing like water, preferably distilled)
    fine_glitter: Optional[bool] = None  # fine/superfine/microfine glitter
    figurines: Optional[bool] = None  # small figurines/miniatures/decor pieces


class StoreChainExtraction(BaseModel):
    chain_name: Optional[str] = None
    ny_location_count: Optional[str] = None  # keep as string for robust extraction
    source_urls: List[str] = Field(default_factory=list)  # URLs supporting the claimed NY store count


class WorkshopExtraction(BaseModel):
    advent: Optional[AdventCalendarExtraction] = None
    wreath: Optional[WreathExtraction] = None
    stocking: Optional[StockingExtraction] = None
    snow_globe: Optional[SnowGlobeExtraction] = None
    store: Optional[StoreChainExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_workshop() -> str:
    return """
Extract the following structured information exactly as it appears in the answer for the four DIY projects and store info.

GENERAL INSTRUCTIONS:
- Return booleans as true/false only if the answer EXPLICITLY includes that material/tool for the specified project.
- Use synonyms where reasonable (listed below). If any accepted synonym is explicitly present, set the boolean true.
- For quantities, return the exact text as shown in the answer (e.g., "1/2 yard", "0.5 yd", "half yard", "½ yard").
- For the advent calendar "containers_24", only return true if the answer explicitly states 24 numbered containers/bags/pockets/days.
- For distilled water, only set true if the answer explicitly mentions "distilled" (e.g., "distilled water", "water (preferably distilled)").
- For fine glitter, set true only if the answer indicates "fine", "superfine", "microfine", or similar qualifiers; plain "glitter" alone is insufficient.
- For the store info, extract the chain name, the exact New York location count as a string, and any URLs the answer cites for that store count (store locator pages, official statements, press releases, etc.).

ACCEPTED SYNONYMS (non-exhaustive guidance):
- Advent glue_or_adhesive: "glue", "hot glue gun", "tacky glue", "PVA", "adhesive", "double-sided tape"
- Advent containers_24: "24" for days/boxes/bags/pockets/envelopes/compartments/doors
- Wreath wreath_form: "wreath form", "wreath ring", "wreath frame", "wire wreath frame", "grapevine wreath", "foam wreath", "straw wreath"
- Wreath florist_wire: "florist wire", "floral wire", "paddle wire"
- Wreath scissors_or_secateurs: "scissors", "secateurs", "pruners", "pruning shears", "garden shears", "snips"
- Wreath moss: "moss", "sheet moss", "sphagnum moss"
- Wreath greenery: "fresh evergreen", "greenery", "foliage", "spruce", "fir", "pine", "cedar", etc.
- Snow globe container: "glass jar", "mason jar", "plastic globe", "ornament globe"
- Snow globe glycerin: "glycerin", "glycerine"
- Snow globe distilled_water: must mention "distilled"
- Snow globe fine_glitter: "fine", "superfine", "microfine" modifiers appear with "glitter"
- Snow globe figurines: "small figurines", "miniatures", "tiny trees", "bottle brush trees", "mini decor"

Return JSON matching this schema:

{
  "advent": {
    "cardboard": boolean or null,
    "ruler": boolean or null,
    "glue_or_adhesive": boolean or null,
    "containers_24": boolean or null
  },
  "wreath": {
    "wreath_form": boolean or null,
    "florist_wire": boolean or null,
    "scissors_or_secateurs": boolean or null,
    "moss": boolean or null,
    "greenery": boolean or null
  },
  "stocking": {
    "outer_fabric_qty": string or null,
    "lining_fabric_qty": string or null,
    "batting_qty": string or null,
    "pattern": boolean or null
  },
  "snow_globe": {
    "container": boolean or null,
    "glycerin": boolean or null,
    "distilled_water": boolean or null,
    "fine_glitter": boolean or null,
    "figurines": boolean or null
  },
  "store": {
    "chain_name": string or null,
    "ny_location_count": string or null,
    "source_urls": [string, ...]
  }
}

IMPORTANT:
- If any field is not explicitly present in the answer, return null for that field (or empty array for URLs).
- Do NOT fabricate or infer any values.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _bool(v: Optional[bool]) -> bool:
    return bool(v) if isinstance(v, bool) else False


def _is_half_yard(text: Optional[str]) -> bool:
    if not text:
        return False
    s = text.strip().lower()
    # Normalize common variations
    repls = [
        ("yards", "yard"),
        ("yds", "yd"),
        (" ", ""),
    ]
    for a, b in repls:
        s = s.replace(a, b)
    # Accept common half-yard expressions
    candidates = [
        "1/2yard",
        "1/2yd",
        "0.5yard",
        "0.5yd",
        "½yard",
        "½yd",
        "halfyard",
        "half-yd",
        "halfyd",
    ]
    return any(c in s for c in candidates)


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def _build_and_verify_tree(evaluator: Evaluator, extracted: WorkshopExtraction) -> None:
    # Create top-level critical parallel node (as per rubric)
    top = evaluator.add_parallel(
        id="workshop_materials_complete",
        desc="All essential materials, tools, and store information for the four-project DIY Christmas craft workshop are provided",
        parent=evaluator.root,
        critical=True,
    )

    # Use safe defaults for nested structures
    advent = extracted.advent or AdventCalendarExtraction()
    wreath = extracted.wreath or WreathExtraction()
    stocking = extracted.stocking or StockingExtraction()
    snow = extracted.snow_globe or SnowGlobeExtraction()
    store = extracted.store or StoreChainExtraction()

    # Advent calendar checks (custom nodes as presence checks)
    evaluator.add_custom_node(
        result=_bool(advent.cardboard),
        id="advent_calendar_cardboard",
        desc="Cardboard is identified as required material for the advent calendar",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_bool(advent.ruler),
        id="advent_calendar_ruler",
        desc="Ruler is identified as required tool for the advent calendar",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_bool(advent.glue_or_adhesive),
        id="advent_calendar_glue",
        desc="Glue or adhesive is identified as required material for the advent calendar",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_bool(advent.containers_24),
        id="advent_calendar_containers",
        desc="24 numbered containers, bags, or decorative elements are identified for the advent calendar days",
        parent=top,
        critical=True,
    )

    # Wreath checks
    evaluator.add_custom_node(
        result=_bool(wreath.wreath_form),
        id="wreath_form",
        desc="Wreath form or wreath ring is identified as required base structure for the wreath",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_bool(wreath.florist_wire),
        id="wreath_wire",
        desc="Florist wire is identified as required material for the wreath",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_bool(wreath.scissors_or_secateurs),
        id="wreath_scissors",
        desc="Scissors or secateurs are identified as required cutting tools for the wreath",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_bool(wreath.moss),
        id="wreath_moss",
        desc="Moss is identified as required material for the wreath",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_bool(wreath.greenery),
        id="wreath_greenery",
        desc="Fresh evergreen foliage or greenery is identified as required material for the wreath",
        parent=top,
        critical=True,
    )

    # Stocking checks (quantities must be 1/2 yard)
    evaluator.add_custom_node(
        result=_is_half_yard(stocking.outer_fabric_qty),
        id="stocking_outer_fabric",
        desc="Outer fabric with specified quantity (1/2 yard) is identified for the stocking",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_is_half_yard(stocking.lining_fabric_qty),
        id="stocking_lining_fabric",
        desc="Lining fabric with specified quantity (1/2 yard) is identified for the stocking",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_is_half_yard(stocking.batting_qty),
        id="stocking_batting",
        desc="Batting with specified quantity (1/2 yard) is identified for the stocking",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_bool(stocking.pattern),
        id="stocking_pattern",
        desc="Stocking pattern is identified as required for the stocking",
        parent=top,
        critical=True,
    )

    # Snow globe checks
    evaluator.add_custom_node(
        result=_bool(snow.container),
        id="snow_globe_container",
        desc="Glass jar or plastic globe container is identified as required for the snow globe",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_bool(snow.glycerin),
        id="snow_globe_glycerin",
        desc="Glycerin is identified as required material for the snow globe",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_bool(snow.distilled_water),
        id="snow_globe_water",
        desc="Distilled water is identified as required material for the snow globe",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_bool(snow.fine_glitter),
        id="snow_globe_glitter",
        desc="Fine or superfine glitter is identified as required material for the snow globe",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_bool(snow.figurines),
        id="snow_globe_figurines",
        desc="Small figurines or decorative miniatures are identified as required for the snow globe",
        parent=top,
        critical=True,
    )

    # Store chain identified (presence)
    evaluator.add_custom_node(
        result=bool(store.chain_name and store.chain_name.strip()),
        id="store_chain_identified",
        desc="A major craft supply store chain with New York state locations is identified",
        parent=top,
        critical=True,
    )

    # Store location count - verify against provided URLs if available; if missing, fail this critical node
    chain_name = (store.chain_name or "").strip()
    count_str = (store.ny_location_count or "").strip()
    urls = store.source_urls if isinstance(store.source_urls, list) else []

    if chain_name and count_str:
        # Create a verification leaf and check with cited sources (prefer multi-URL verification)
        loc_node = evaluator.add_leaf(
            id="store_location_count",
            desc="The exact number of store locations in New York state is provided",
            parent=top,
            critical=True,
        )
        claim = f"The craft supply store chain '{chain_name}' has exactly {count_str} locations in New York state as of 2026."
        add_ins = (
            "Only mark the claim as supported if at least one of the provided URL(s) explicitly shows the total number "
            "of New York state locations for the specified chain (e.g., an official store locator that lists and counts "
            "NY locations, an official press release, or an authoritative page that clearly states the current NY count). "
            "If the URLs are missing, irrelevant, or do not clearly state or allow counting to the exact number provided, "
            "mark the claim as NOT supported."
        )
        await evaluator.verify(
            claim=claim,
            node=loc_node,
            sources=urls if urls else None,  # If empty, the verifier will do simple check; instruction forces strictness
            additional_instruction=add_ins,
        )
    else:
        # If the answer doesn't provide both a chain name and an explicit count string, fail this critical requirement
        evaluator.add_custom_node(
            result=False,
            id="store_location_count",
            desc="The exact number of store locations in New York state is provided",
            parent=top,
            critical=True,
        )


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
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
    Evaluate an answer for the DIY Christmas craft workshop materials and store info task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root container; actual rubric root added as a critical child
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

    # Extract structured info
    extracted: WorkshopExtraction = await evaluator.extract(
        prompt=prompt_extract_workshop(),
        template_class=WorkshopExtraction,
        extraction_name="workshop_extraction",
    )

    # Optional: record minimal GT/context info for traceability
    evaluator.add_ground_truth({
        "projects": [
            "Cardboard advent calendar",
            "Fresh greenery wreath (with wreath form)",
            "Sewn Christmas stocking (outer + lining, quantities)",
            "Water-filled snow globe",
        ],
        "store_info_required": "Major craft chain in NY and exact number of NY locations as of 2026",
    })

    # Build the verification tree and run checks
    await _build_and_verify_tree(evaluator, extracted)

    # Return standard evaluation summary
    return evaluator.get_summary()