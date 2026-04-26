import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "thanksgiving_craft_2025"
TASK_DESCRIPTION = (
    "You are planning a Thanksgiving weekend DIY craft session for 2025 and need to create a comprehensive plan. "
    "Your plan must include:\n\n"
    "1. Craft Store Selection: Identify a major craft store chain that operates retail locations in all U.S. states "
    "EXCEPT Alaska and Hawaii, is open on Black Friday 2025 (November 28, 2025), and opens at 8:00 AM or earlier on that day.\n\n"
    "2. Project 1 - Fabric Wreath: A beginner-level, Thanksgiving-appropriate project completed within 2 hours, "
    "specifying wreath form size (12-inch or 14-inch), fabric quantity (2.5–4 yards for 14-inch OR ~3 fat quarters for 12-inch), "
    "and including wire wreath form and cutting tools in the materials list; optionally incorporates 2025 trends "
    "(dried florals, woodland motifs, or burgundy/navy palettes).\n\n"
    "3. Project 2 - Thanksgiving Centerpiece: A beginner-level project completed within 1 hour, specifying a container/tray/base "
    "and at least 2 decorative element types; optionally incorporates 2025 trends.\n\n"
    "4. Project 3 - Woodworking or Safety-Required Project: A beginner-level fall/Thanksgiving-themed project completed within 3 hours, "
    "specifying the primary material with approximate dimensions, at least 2 basic tools, and safety requirements that include "
    "safety glasses/goggles AND at least one additional item (hearing protection, dust mask, or closed-toe shoes).\n\n"
    "For each component, provide the name/description and a reference URL that confirms the information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreInfo(BaseModel):
    store_name: Optional[str] = None
    operations_urls: List[str] = Field(default_factory=list)
    operations_claim: Optional[str] = None
    excluded_states_mentioned: List[str] = Field(default_factory=list)
    black_friday_url: Optional[str] = None
    black_friday_open_time: Optional[str] = None
    black_friday_is_open: Optional[str] = None
    black_friday_date: Optional[str] = None


class FabricWreathProject(BaseModel):
    project_name: Optional[str] = None
    project_url: Optional[str] = None
    skill_level: Optional[str] = None
    time_required: Optional[str] = None
    wreath_form_size: Optional[str] = None  # e.g., "12-inch", "14-inch"
    fabric_quantity: Optional[str] = None   # free text as in the answer
    materials_list: List[str] = Field(default_factory=list)
    optional_trend_elements: List[str] = Field(default_factory=list)
    extra_urls: List[str] = Field(default_factory=list)


class CenterpieceProject(BaseModel):
    project_name: Optional[str] = None
    project_url: Optional[str] = None
    skill_level: Optional[str] = None
    time_required: Optional[str] = None
    container_or_base: Optional[str] = None
    decorative_elements: List[str] = Field(default_factory=list)
    materials_list: List[str] = Field(default_factory=list)
    optional_trend_elements: List[str] = Field(default_factory=list)
    extra_urls: List[str] = Field(default_factory=list)


class SafetyProject(BaseModel):
    project_name: Optional[str] = None
    project_url: Optional[str] = None
    skill_level: Optional[str] = None
    time_required: Optional[str] = None
    theme: Optional[str] = None
    primary_material: Optional[str] = None
    primary_material_dimensions: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    safety_equipment: List[str] = Field(default_factory=list)
    safety_reference_url: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)


class PlanExtraction(BaseModel):
    store: Optional[StoreInfo] = None
    fabric_wreath: Optional[FabricWreathProject] = None
    centerpiece: Optional[CenterpieceProject] = None
    safety_project: Optional[SafetyProject] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract the structured plan details from the provided answer. Return a JSON object with the following schema:

    {
      "store": {
        "store_name": string|null,
        "operations_urls": string[] (URLs confirming where the chain operates; extract only URLs present in the answer),
        "operations_claim": string|null (the answer's wording about operational coverage),
        "excluded_states_mentioned": string[] (list the states explicitly mentioned as excluded, e.g., ["Alaska","Hawaii"]),
        "black_friday_url": string|null (URL that states Black Friday 2025 hours; extract from the answer),
        "black_friday_open_time": string|null (the opening time quoted for Black Friday 2025, e.g., "6:00 AM", "8:00 AM"),
        "black_friday_is_open": string|null (e.g., "open", "closed", or phrasing from answer),
        "black_friday_date": string|null (e.g., "November 28, 2025" or "11/28/2025")
      },
      "fabric_wreath": {
        "project_name": string|null,
        "project_url": string|null,
        "skill_level": string|null,
        "time_required": string|null,
        "wreath_form_size": string|null,       // e.g., "12-inch", "14\"", "14 inch"
        "fabric_quantity": string|null,        // e.g., "3 fat quarters", "3 yards", "2.5-4 yards"
        "materials_list": string[],            // list each material as a string exactly as in the answer
        "optional_trend_elements": string[],   // any of: dried florals, woodland motifs, burgundy, navy
        "extra_urls": string[]                 // any additional URLs cited for this project
      },
      "centerpiece": {
        "project_name": string|null,
        "project_url": string|null,
        "skill_level": string|null,
        "time_required": string|null,
        "container_or_base": string|null,      // e.g., "tray", "vase", "compote", "dough bowl", etc.
        "decorative_elements": string[],       // list elements like "pumpkins", "candles", "florals", "pinecones"
        "materials_list": string[],
        "optional_trend_elements": string[],   // any of: dried florals, woodland motifs, burgundy, navy
        "extra_urls": string[]
      },
      "safety_project": {
        "project_name": string|null,
        "project_url": string|null,
        "skill_level": string|null,
        "time_required": string|null,
        "theme": string|null,                        // "Thanksgiving" or "fall" themes
        "primary_material": string|null,             // e.g., "1x8 pine board" or "wooden pumpkin cutout"
        "primary_material_dimensions": string|null,  // approximate dimensions or quantity if stated
        "tools": string[],                           // list beginner tools (drill, saw, sander, hammer, measuring tape, etc.)
        "safety_equipment": string[],                // list items like "safety glasses", "goggles", "hearing protection", "dust mask", "closed-toe shoes"
        "safety_reference_url": string|null,         // a URL that explicitly mentions safety requirements; if not provided, null
        "extra_urls": string[]
      }
    }

    Guidelines:
    - Extract ONLY what appears explicitly in the answer text. Do not invent values.
    - For URLs, extract only valid URLs that appear in the answer (including markdown links).
    - Keep all fields as strings as they appear (do not normalize numbers or times).
    - If a field is not provided in the answer, set it to null (or empty array for lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(primary: Optional[str], extras: Optional[List[str]] = None) -> Optional[List[str]]:
    urls: List[str] = []
    if primary and isinstance(primary, str) and primary.strip():
        urls.append(primary.strip())
    if extras:
        for u in extras:
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())
    return urls if urls else None


def _wreath_size_category(size_text: Optional[str]) -> Optional[str]:
    if not size_text:
        return None
    text = size_text.lower()
    if "14" in text:
        return "14"
    if "12" in text:
        return "12"
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_store_selection(evaluator: Evaluator, parent_node, store: Optional[StoreInfo]) -> None:
    node = evaluator.add_sequential(
        id="CraftStoreSelection",
        desc="Identification of a craft store chain that operates in states excluding Alaska and Hawaii, with specific Black Friday 2025 operating hours",
        parent=parent_node,
        critical=False
    )

    store_name = store.store_name if store else "the selected store"
    ops_sources = (store.operations_urls if store else []) or []
    hours_source = store.black_friday_url if (store and store.black_friday_url) else None

    # StoreChainIdentification (critical, sequential)
    chain_id = evaluator.add_sequential(
        id="StoreChainIdentification",
        desc="The craft store chain must operate in all U.S. states except Alaska and Hawaii",
        parent=node,
        critical=True
    )

    scope = evaluator.add_parallel(
        id="StoreOperationalScope",
        desc="Verification of store's operational scope and type",
        parent=chain_id,
        critical=True
    )

    # StateOperations (leaf)
    leaf_state_ops = evaluator.add_leaf(
        id="StateOperations",
        desc="Store operates in at least 45 states, excluding Alaska and Hawaii",
        parent=scope,
        critical=True
    )
    claim_state_ops = (
        f"The chain '{store_name}' operates retail locations in at least 45 U.S. states and does not operate in Alaska or Hawaii."
    )
    await evaluator.verify(
        claim=claim_state_ops,
        node=leaf_state_ops,
        sources=ops_sources if ops_sources else None,
        additional_instruction="Accept if the referenced page(s) explicitly show state coverage indicating no stores in Alaska or Hawaii."
    )

    # StoreType (leaf)
    leaf_store_type = evaluator.add_leaf(
        id="StoreType",
        desc="Store must be a dedicated arts and crafts retail chain",
        parent=scope,
        critical=True
    )
    claim_store_type = f"'{store_name}' is a dedicated arts-and-crafts retail chain (craft supplies and related goods)."
    await evaluator.verify(
        claim=claim_store_type,
        node=leaf_store_type,
        sources=ops_sources if ops_sources else None,
        additional_instruction="Look for descriptions that it specializes in arts & crafts supplies or crafts retail."
    )

    # StoreChainReference (leaf)
    leaf_chain_ref = evaluator.add_leaf(
        id="StoreChainReference",
        desc="Reference URL confirming store chain operations",
        parent=chain_id,
        critical=True
    )
    claim_chain_ref = (
        "This reference explicitly provides information about the chain's U.S. operations/state coverage."
    )
    await evaluator.verify(
        claim=claim_chain_ref,
        node=leaf_chain_ref,
        sources=ops_sources if ops_sources else None,
        additional_instruction="The page should mention locations/states covered, store locator with states, or a corporate facts page."
    )

    # BlackFridayHours (critical, sequential)
    hours = evaluator.add_sequential(
        id="BlackFridayHours",
        desc="Store's Black Friday 2025 (November 28) opening time must be 8:00 AM or earlier",
        parent=node,
        critical=True
    )

    hv = evaluator.add_parallel(
        id="HoursVerification",
        desc="Verification of store hours on Black Friday 2025",
        parent=hours,
        critical=True
    )

    # OpeningTime (leaf)
    leaf_open_time = evaluator.add_leaf(
        id="OpeningTime",
        desc="Store opens at or before 8:00 AM on November 28, 2025",
        parent=hv,
        critical=True
    )
    claim_open_time = f"On Black Friday 2025 (November 28, 2025), {store_name} stores open at or before 8:00 AM local time."
    await evaluator.verify(
        claim=claim_open_time,
        node=leaf_open_time,
        sources=hours_source if hours_source else None,
        additional_instruction="Confirm the listed opening hour is 8:00 AM or earlier for Black Friday 2025."
    )

    # OperationalStatus (leaf)
    leaf_open_status = evaluator.add_leaf(
        id="OperationalStatus",
        desc="Store is confirmed open on Black Friday 2025",
        parent=hv,
        critical=True
    )
    claim_open_status = f"{store_name} is open on Black Friday 2025 (November 28, 2025)."
    await evaluator.verify(
        claim=claim_open_status,
        node=leaf_open_status,
        sources=hours_source if hours_source else None,
        additional_instruction="The reference should indicate the store operates (is open) on that date."
    )

    # HoursReference (leaf)
    leaf_hours_ref = evaluator.add_leaf(
        id="HoursReference",
        desc="Reference URL confirming Black Friday hours",
        parent=hours,
        critical=True
    )
    claim_hours_ref = "This reference specifies the Black Friday 2025 opening time for the chain."
    await evaluator.verify(
        claim=claim_hours_ref,
        node=leaf_hours_ref,
        sources=hours_source if hours_source else None,
        additional_instruction="An official hours page, ad, or reputable listing explicitly mentioning Black Friday 2025 hours qualifies."
    )


async def verify_project_fabric_wreath(evaluator: Evaluator, parent_node, proj: Optional[FabricWreathProject]) -> None:
    node = evaluator.add_sequential(
        id="Project1_FabricWreath",
        desc="A beginner-level fabric wreath project meeting time, material quantity, and trend requirements",
        parent=parent_node,
        critical=False
    )

    proj_url = proj.project_url if proj else None
    all_sources = _combine_sources(proj_url, proj.extra_urls if proj else [])

    # ProjectIdentification (critical, sequential)
    pid = evaluator.add_sequential(
        id="P1_ProjectIdentification",
        desc="Project must be a fabric-based wreath suitable for Thanksgiving",
        parent=node,
        critical=True
    )
    pinfo = evaluator.add_parallel(
        id="P1_ProjectBasicInfo",
        desc="Basic project type and skill level verification",
        parent=pid,
        critical=True
    )

    # ProjectType (leaf)
    leaf_type = evaluator.add_leaf(
        id="P1_ProjectType",
        desc="Project is identified as a fabric wreath or fabric rag wreath",
        parent=pinfo,
        critical=True
    )
    claim_type = "This tutorial describes a fabric wreath (also called a rag wreath) that is suitable for Thanksgiving decoration."
    await evaluator.verify(
        claim=claim_type,
        node=leaf_type,
        sources=all_sources,
        additional_instruction="Look for 'fabric', 'rag wreath', 'wreath', and Thanksgiving/fall context."
    )

    # SkillLevel (leaf)
    leaf_skill = evaluator.add_leaf(
        id="P1_SkillLevel",
        desc="Project is classified as beginner-friendly with simple techniques",
        parent=pinfo,
        critical=True
    )
    claim_skill = "This project is beginner-friendly or easy to make."
    await evaluator.verify(
        claim=claim_skill,
        node=leaf_skill,
        sources=all_sources,
        additional_instruction="Accept if the page explicitly says 'beginner', 'easy', or clearly indicates it's suitable for beginners."
    )

    # ProjectIdentificationReference (leaf)
    leaf_pid_ref = evaluator.add_leaf(
        id="P1_ProjectIdentificationReference",
        desc="Reference URL for project identification",
        parent=pid,
        critical=True
    )
    claim_pid_ref = "This URL is a tutorial or instruction page for the identified fabric wreath project."
    await evaluator.verify(
        claim=claim_pid_ref,
        node=leaf_pid_ref,
        sources=all_sources,
        additional_instruction="The page should present steps or materials for making the fabric wreath."
    )

    # TimeRequirement (leaf, critical)
    leaf_time = evaluator.add_leaf(
        id="P1_TimeRequirement",
        desc="Project completion time must not exceed 2 hours for a beginner",
        parent=node,
        critical=True
    )
    time_excerpt = proj.time_required if proj and proj.time_required else "the stated time"
    claim_time = f"A beginner can complete this fabric wreath project in 2 hours or less (answer cites {time_excerpt})."
    await evaluator.verify(
        claim=claim_time,
        node=leaf_time,
        sources=all_sources,
        additional_instruction="Prefer explicit time statements; otherwise, accept if the tutorial indicates a short, simple build suitable within ~2 hours."
    )

    # MaterialSpecifications (critical, sequential)
    mats = evaluator.add_sequential(
        id="P1_MaterialSpecifications",
        desc="Material quantities must align with standard requirements for the specified wreath size",
        parent=node,
        critical=True
    )
    mdetails = evaluator.add_parallel(
        id="P1_MaterialDetails",
        desc="Detailed material requirements and quantities",
        parent=mats,
        critical=True
    )

    # WreathFormSize (leaf)
    leaf_size = evaluator.add_leaf(
        id="P1_WreathFormSize",
        desc="Wreath form size must be specified (e.g., 12-inch or 14-inch)",
        parent=mdetails,
        critical=True
    )
    size_txt = proj.wreath_form_size if proj else None
    claim_size = (
        f"The tutorial specifies a wreath form size of 12-inch or 14-inch"
        + (f"; it mentions {size_txt}." if size_txt else ".")
    )
    await evaluator.verify(
        claim=claim_size,
        node=leaf_size,
        sources=all_sources,
        additional_instruction="Look for '12-inch', '12\"', '14-inch', or '14\"' in the materials or description."
    )

    # FabricQuantity (leaf) - dependent on size if available
    leaf_fabric_qty = evaluator.add_leaf(
        id="P1_FabricQuantity",
        desc="For a 14-inch form: 2.5-4 yards of fabric, or for a 12-inch form: approximately 3 fat quarters",
        parent=mdetails,
        critical=True
    )
    size_cat = _wreath_size_category(size_txt)
    qty_txt = proj.fabric_quantity if proj and proj.fabric_quantity else "the cited fabric amount"
    if size_cat == "14":
        claim_qty = f"The tutorial uses a 14-inch wreath form and specifies about 2.5 to 4 yards of fabric; it lists {qty_txt}."
    elif size_cat == "12":
        claim_qty = f"The tutorial uses a 12-inch wreath form and specifies approximately three fat quarters of fabric; it lists {qty_txt}."
    else:
        claim_qty = (
            f"The tutorial specifies fabric quantity consistent with standard guidance—"
            f"for a 14-inch form: 2.5–4 yards; or for a 12-inch form: ~3 fat quarters; it lists {qty_txt}."
        )
    await evaluator.verify(
        claim=claim_qty,
        node=leaf_fabric_qty,
        sources=all_sources,
        additional_instruction="Confirm the fabric amount matches the size-specific expectation stated in the claim."
    )

    # AdditionalSupplies (leaf)
    leaf_add_sup = evaluator.add_leaf(
        id="P1_AdditionalSupplies",
        desc="Wire wreath form and cutting tools (scissors or rotary cutter) must be listed",
        parent=mdetails,
        critical=True
    )
    claim_add_sup = "The materials list includes a wire wreath form and cutting tools (scissors or a rotary cutter)."
    await evaluator.verify(
        claim=claim_add_sup,
        node=leaf_add_sup,
        sources=all_sources,
        additional_instruction="Look for explicit mentions of a wire wreath form and either scissors or a rotary cutter."
    )

    # MaterialReference (leaf)
    leaf_mat_ref = evaluator.add_leaf(
        id="P1_MaterialReference",
        desc="Reference URL for material specifications",
        parent=mats,
        critical=True
    )
    claim_mat_ref = "This reference provides the materials list and quantities for the fabric wreath, including wreath size and fabric amount."
    await evaluator.verify(
        claim=claim_mat_ref,
        node=leaf_mat_ref,
        sources=all_sources,
        additional_instruction="A proper tutorial page with materials listed qualifies."
    )

    # TrendAlignment (leaf, non-critical)
    leaf_trend = evaluator.add_leaf(
        id="P1_TrendAlignment",
        desc="Project should incorporate 2025 Thanksgiving trends (dried florals, woodland motifs, or burgundy/navy colors)",
        parent=node,
        critical=False
    )
    claim_trend = "The project incorporates at least one 2025 Thanksgiving trend: dried florals, woodland motifs, or burgundy/navy colors."
    await evaluator.verify(
        claim=claim_trend,
        node=leaf_trend,
        sources=all_sources,
        additional_instruction="Accept if any of the listed trends appear in the styling, materials, or color palette."
    )


async def verify_project_centerpiece(evaluator: Evaluator, parent_node, proj: Optional[CenterpieceProject]) -> None:
    node = evaluator.add_sequential(
        id="Project2_ThanksgivingCenterpiece",
        desc="A beginner-level Thanksgiving centerpiece project meeting time and material requirements",
        parent=parent_node,
        critical=False
    )

    proj_url = proj.project_url if proj else None
    all_sources = _combine_sources(proj_url, proj.extra_urls if proj else [])

    # ProjectIdentification (critical, sequential)
    pid = evaluator.add_sequential(
        id="P2_ProjectIdentification",
        desc="Project must be a Thanksgiving table centerpiece",
        parent=node,
        critical=True
    )
    pinfo = evaluator.add_parallel(
        id="P2_ProjectBasicInfo",
        desc="Basic project type and skill level verification",
        parent=pid,
        critical=True
    )

    # ProjectType (leaf)
    leaf_type = evaluator.add_leaf(
        id="P2_ProjectType",
        desc="Project is identified as a Thanksgiving centerpiece or table decoration",
        parent=pinfo,
        critical=True
    )
    claim_type = "This tutorial is a Thanksgiving centerpiece or table decoration project."
    await evaluator.verify(
        claim=claim_type,
        node=leaf_type,
        sources=all_sources,
        additional_instruction="Look for 'centerpiece', 'table decor', 'Thanksgiving' keywords."
    )

    # SkillLevel (leaf)
    leaf_skill = evaluator.add_leaf(
        id="P2_SkillLevel",
        desc="Project is classified as beginner-friendly or easy DIY",
        parent=pinfo,
        critical=True
    )
    claim_skill = "This project is beginner-friendly or easy to make."
    await evaluator.verify(
        claim=claim_skill,
        node=leaf_skill,
        sources=all_sources,
        additional_instruction="Accept if 'beginner' or 'easy' is stated or strongly implied."
    )

    # ProjectIdentificationReference (leaf)
    leaf_pid_ref = evaluator.add_leaf(
        id="P2_ProjectIdentificationReference",
        desc="Reference URL for project identification",
        parent=pid,
        critical=True
    )
    claim_pid_ref = "This URL is a tutorial or instruction page for the identified Thanksgiving centerpiece project."
    await evaluator.verify(
        claim=claim_pid_ref,
        node=leaf_pid_ref,
        sources=all_sources,
        additional_instruction="The page should present steps or materials for making the centerpiece."
    )

    # TimeRequirement (leaf, critical)
    leaf_time = evaluator.add_leaf(
        id="P2_TimeRequirement",
        desc="Project completion time must not exceed 1 hour",
        parent=node,
        critical=True
    )
    time_excerpt = proj.time_required if proj and proj.time_required else "the stated time"
    claim_time = f"A beginner can complete this centerpiece project in 1 hour or less (answer cites {time_excerpt})."
    await evaluator.verify(
        claim=claim_time,
        node=leaf_time,
        sources=all_sources,
        additional_instruction="Prefer explicit time statements; accept if clearly a quick build within ~1 hour."
    )

    # MaterialSpecifications (critical, sequential)
    mats = evaluator.add_sequential(
        id="P2_MaterialSpecifications",
        desc="Required materials must be clearly specified and commonly available",
        parent=node,
        critical=True
    )
    mdetails = evaluator.add_parallel(
        id="P2_MaterialDetails",
        desc="Detailed material requirements",
        parent=mats,
        critical=True
    )

    # ContainerOrBase (leaf)
    leaf_base = evaluator.add_leaf(
        id="P2_ContainerOrBase",
        desc="A container, tray, or base for the centerpiece must be specified",
        parent=mdetails,
        critical=True
    )
    base_txt = proj.container_or_base if proj and proj.container_or_base else "a container/tray/base"
    claim_base = f"The tutorial specifies {base_txt} as the container/tray/base for the centerpiece."
    await evaluator.verify(
        claim=claim_base,
        node=leaf_base,
        sources=all_sources,
        additional_instruction="Look for a named vessel such as tray, bowl, vase, compote, dough bowl, etc."
    )

    # DecorativeElements (leaf)
    leaf_decor = evaluator.add_leaf(
        id="P2_DecorativeElements",
        desc="At least 2 types of decorative elements (e.g., pumpkins, candles, florals, pinecones) must be listed",
        parent=mdetails,
        critical=True
    )
    listed = proj.decorative_elements if proj and proj.decorative_elements else []
    claim_decor = f"The project lists at least two types of decorative elements, e.g., pumpkins, candles, florals, or pinecones; listed: {listed}."
    await evaluator.verify(
        claim=claim_decor,
        node=leaf_decor,
        sources=all_sources,
        additional_instruction="Verify that at least two distinct decorative element types are present in the materials or instructions."
    )

    # MaterialReference (leaf)
    leaf_mat_ref = evaluator.add_leaf(
        id="P2_MaterialReference",
        desc="Reference URL for material specifications",
        parent=mats,
        critical=True
    )
    claim_mat_ref = "This reference provides the materials list for the centerpiece project."
    await evaluator.verify(
        claim=claim_mat_ref,
        node=leaf_mat_ref,
        sources=all_sources,
        additional_instruction="A proper tutorial page with materials listed qualifies."
    )

    # TrendAlignment (leaf, non-critical)
    leaf_trend = evaluator.add_leaf(
        id="P2_TrendAlignment",
        desc="Project should incorporate 2025 Thanksgiving trends (dried florals, woodland motifs, or burgundy/navy colors)",
        parent=node,
        critical=False
    )
    claim_trend = "The project incorporates at least one 2025 Thanksgiving trend: dried florals, woodland motifs, or burgundy/navy colors."
    await evaluator.verify(
        claim=claim_trend,
        node=leaf_trend,
        sources=all_sources,
        additional_instruction="Accept if any of the listed trends appear in the styling, materials, or color palette."
    )


async def verify_project_safety(evaluator: Evaluator, parent_node, proj: Optional[SafetyProject]) -> None:
    node = evaluator.add_sequential(
        id="Project3_WoodworkingOrSafetyProject",
        desc="A beginner woodworking project OR any project requiring safety equipment, meeting time, material, and safety requirements",
        parent=parent_node,
        critical=False
    )

    proj_url = proj.project_url if proj else None
    safety_ref = proj.safety_reference_url if proj and proj.safety_reference_url else None
    extra = proj.extra_urls if proj else []
    all_sources = _combine_sources(proj_url, extra)
    safety_sources = _combine_sources(safety_ref, ([] if not all_sources else all_sources))

    # ProjectIdentification (critical, sequential)
    pid = evaluator.add_sequential(
        id="P3_ProjectIdentification",
        desc="Project must be suitable for beginners and Thanksgiving-themed or fall-themed",
        parent=node,
        critical=True
    )
    pinfo = evaluator.add_parallel(
        id="P3_ProjectBasicInfo",
        desc="Basic project type and skill level verification",
        parent=pid,
        critical=True
    )

    # ProjectType (leaf)
    leaf_type = evaluator.add_leaf(
        id="P3_ProjectType",
        desc="Project is a woodworking item (e.g., wood sign, wooden decoration) or involves tools requiring safety equipment",
        parent=pinfo,
        critical=True
    )
    claim_type = (
        "This tutorial is a woodworking or tool-based project (e.g., wooden sign/decoration) with a Thanksgiving or fall theme."
    )
    await evaluator.verify(
        claim=claim_type,
        node=leaf_type,
        sources=all_sources,
        additional_instruction="Look for wood/wooden materials or clear tool usage; also confirm Thanksgiving/fall context."
    )

    # SkillLevel (leaf)
    leaf_skill = evaluator.add_leaf(
        id="P3_SkillLevel",
        desc="Project is classified as beginner-friendly",
        parent=pinfo,
        critical=True
    )
    claim_skill = "This project is beginner-friendly or easy to make."
    await evaluator.verify(
        claim=claim_skill,
        node=leaf_skill,
        sources=all_sources,
        additional_instruction="Accept if 'beginner' or 'easy' is stated or strongly implied."
    )

    # ProjectIdentificationReference (leaf)
    leaf_pid_ref = evaluator.add_leaf(
        id="P3_ProjectIdentificationReference",
        desc="Reference URL for project identification",
        parent=pid,
        critical=True
    )
    claim_pid_ref = "This URL is a tutorial or instruction page for the identified woodworking/safety-required project."
    await evaluator.verify(
        claim=claim_pid_ref,
        node=leaf_pid_ref,
        sources=all_sources,
        additional_instruction="The page should present steps or materials for making the project."
    )

    # TimeRequirement (leaf, critical)
    leaf_time = evaluator.add_leaf(
        id="P3_TimeRequirement",
        desc="Project completion time must not exceed 3 hours for a beginner",
        parent=node,
        critical=True
    )
    time_excerpt = proj.time_required if proj and proj.time_required else "the stated time"
    claim_time = f"A beginner can complete this project in 3 hours or less (answer cites {time_excerpt})."
    await evaluator.verify(
        claim=claim_time,
        node=leaf_time,
        sources=all_sources,
        additional_instruction="Prefer explicit time statements; accept if clearly a short/simple build within ~3 hours."
    )

    # MaterialAndToolSpecifications (critical, sequential)
    mats_tools = evaluator.add_sequential(
        id="P3_MaterialAndToolSpecifications",
        desc="Required materials and basic tools must be specified",
        parent=node,
        critical=True
    )
    mtdetails = evaluator.add_parallel(
        id="P3_MaterialAndToolDetails",
        desc="Detailed material and tool requirements",
        parent=mats_tools,
        critical=True
    )

    # PrimaryMaterial (leaf)
    leaf_primary = evaluator.add_leaf(
        id="P3_PrimaryMaterial",
        desc="Primary material (e.g., wood board, wooden cutout) must be specified with approximate dimensions or quantity",
        parent=mtdetails,
        critical=True
    )
    pm = proj.primary_material if proj and proj.primary_material else "the primary wooden material"
    pd = proj.primary_material_dimensions if proj and proj.primary_material_dimensions else "approximate dimensions/quantity"
    claim_primary = f"The tutorial specifies {pm} with {pd}."
    await evaluator.verify(
        claim=claim_primary,
        node=leaf_primary,
        sources=all_sources,
        additional_instruction="Look for named wood species/board sizes or wooden cutout with approximate dimensions or count."
    )

    # BasicTools (leaf)
    leaf_tools = evaluator.add_leaf(
        id="P3_BasicTools",
        desc="At least 2 basic tools from beginner essentials (drill, saw, sander, hammer, measuring tape, etc.) must be listed",
        parent=mtdetails,
        critical=True
    )
    tools_list = proj.tools if proj and proj.tools else []
    claim_tools = f"The tutorial lists at least two beginner tools (e.g., drill, saw, sander, hammer, measuring tape); listed: {tools_list}."
    await evaluator.verify(
        claim=claim_tools,
        node=leaf_tools,
        sources=all_sources,
        additional_instruction="Verify that at least two of the named beginner tools are present."
    )

    # MaterialToolReference (leaf)
    leaf_mt_ref = evaluator.add_leaf(
        id="P3_MaterialToolReference",
        desc="Reference URL for material and tool specifications",
        parent=mats_tools,
        critical=True
    )
    claim_mt_ref = "This reference provides the materials and tools list for the project."
    await evaluator.verify(
        claim=claim_mt_ref,
        node=leaf_mt_ref,
        sources=all_sources,
        additional_instruction="A proper tutorial page with materials and tools listed qualifies."
    )

    # SafetyRequirements (critical, sequential)
    safety = evaluator.add_sequential(
        id="P3_SafetyRequirements",
        desc="If the project involves power tools or woodworking, safety equipment must be specified",
        parent=node,
        critical=True
    )
    sdetails = evaluator.add_parallel(
        id="P3_SafetyEquipmentDetails",
        desc="Detailed safety equipment requirements",
        parent=safety,
        critical=True
    )

    # EyeProtection (leaf)
    leaf_eye = evaluator.add_leaf(
        id="P3_EyeProtection",
        desc="Safety glasses or goggles must be included in safety requirements",
        parent=sdetails,
        critical=True
    )
    claim_eye = "The project specifies wearing safety glasses or goggles."
    await evaluator.verify(
        claim=claim_eye,
        node=leaf_eye,
        sources=safety_sources,
        additional_instruction="Look for 'safety glasses' or 'goggles' explicitly."
    )

    # AdditionalProtection (leaf)
    leaf_add = evaluator.add_leaf(
        id="P3_AdditionalProtection",
        desc="At least one additional safety item (hearing protection, dust mask, or closed-toe shoes) must be specified",
        parent=sdetails,
        critical=True
    )
    claim_add = "The project specifies at least one of: hearing protection, dust mask/respirator, or closed-toe shoes."
    await evaluator.verify(
        claim=claim_add,
        node=leaf_add,
        sources=safety_sources,
        additional_instruction="Accept if any one additional item is present alongside eye protection."
    )

    # SafetyReference (leaf)
    leaf_s_ref = evaluator.add_leaf(
        id="P3_SafetyReference",
        desc="Reference URL for safety requirements",
        parent=safety,
        critical=True
    )
    claim_s_ref = "This reference explicitly states safety equipment requirements for the project."
    await evaluator.verify(
        claim=claim_s_ref,
        node=leaf_s_ref,
        sources=safety_sources,
        additional_instruction="A tutorial section or separate safety page is acceptable if it lists the required safety equipment."
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel as per rubric
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

    # Extract the entire plan from the answer
    plan: PlanExtraction = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction"
    )

    # Build main rubric root node
    session_root = evaluator.add_parallel(
        id="ThanksgivingCraftSession",
        desc="A complete Thanksgiving craft session plan including store selection and three beginner-friendly projects that meet time, material, safety, and trend requirements",
        parent=root,
        critical=False
    )

    # Craft Store Selection subtree
    await verify_store_selection(evaluator, session_root, plan.store if plan else None)

    # Project 1 - Fabric Wreath subtree
    await verify_project_fabric_wreath(evaluator, session_root, plan.fabric_wreath if plan else None)

    # Project 2 - Thanksgiving Centerpiece subtree
    await verify_project_centerpiece(evaluator, session_root, plan.centerpiece if plan else None)

    # Project 3 - Woodworking/Safety-required subtree
    await verify_project_safety(evaluator, session_root, plan.safety_project if plan else None)

    # Return evaluation summary
    return evaluator.get_summary()