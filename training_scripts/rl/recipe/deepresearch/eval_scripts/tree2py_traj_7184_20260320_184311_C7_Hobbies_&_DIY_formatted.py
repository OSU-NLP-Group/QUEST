import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ma_craft_store_michaels_hobbylobby_202603"
TASK_DESCRIPTION = (
    "Identify a physical retail location of either Michaels or Hobby Lobby in Massachusetts that is currently "
    "operational as of March 2026 and offers all of the following product categories and services:\n"
    "- Woodworking supplies (including at least 5 of these essential beginner tools: tape measure, cordless drill, "
    "circular saw, random orbital sander, jigsaw, chisels, carpenter's square, clamps, or miter saw)\n"
    "- Fabric and sewing supplies\n"
    "- Painting supplies (including paint, brushes, and related materials)\n"
    "- Home decor items\n"
    "- Seasonal decorations\n"
    "- Framing services or supplies\n"
    "- Craft kits suitable for adults\n"
    "- Art supplies (including canvases, drawing materials, or similar items)\n"
    "- DIY home improvement supplies or materials\n"
    "- Gardening or outdoor project supplies\n\n"
    "Provide the store name, complete physical address, and a reference URL confirming the store's product offerings."
)

ALLOWED_WOODWORKING_TOOLS = [
    "tape measure",
    "cordless drill",
    "circular saw",
    "random orbital sander",
    "jigsaw",
    "chisels",
    "carpenter's square",
    "clamps",
    "miter saw",
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProductCategories(BaseModel):
    woodworking_supplies: Optional[bool] = None
    fabric_sewing: Optional[bool] = None
    painting_supplies: Optional[bool] = None
    home_decor: Optional[bool] = None
    seasonal_decorations: Optional[bool] = None
    framing: Optional[bool] = None
    adult_craft_kits: Optional[bool] = None
    art_supplies: Optional[bool] = None
    diy_home_materials: Optional[bool] = None
    gardening_outdoor: Optional[bool] = None


class StoreExtraction(BaseModel):
    chain: Optional[str] = None  # "Michaels" or "Hobby Lobby"
    store_name: Optional[str] = None
    address: Optional[str] = None  # Single-line full street address
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None

    store_url: Optional[str] = None  # Location-specific store page if provided
    reference_urls: List[str] = Field(default_factory=list)  # URLs confirming existence and offerings

    woodworking_tools: List[str] = Field(default_factory=list)  # Only from the allowed list, normalized
    categories: Optional[ProductCategories] = None

    operational_as_of_2026: Optional[bool] = None  # If explicitly stated in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_store() -> str:
    allowed_tools_str = "; ".join(ALLOWED_WOODWORKING_TOOLS)
    return f"""
Extract a single Massachusetts store location for Michaels or Hobby Lobby as presented in the answer.

Return the following fields:
- chain: The retail chain, exactly "Michaels" or "Hobby Lobby" (or null if not clearly stated).
- store_name: The store/location name as given in the answer (or null).
- address: The complete physical street address in a single line, including street number/name, city, state, and zip if provided in the answer. Normalize common abbreviations (e.g., "St." vs "Street") only if needed; otherwise keep verbatim.
- city: City name (or null).
- state: State name or code as presented (e.g., "MA" or "Massachusetts") (or null).
- zip: ZIP code (or null).

- store_url: The specific store location page URL if the answer includes one; otherwise null.
- reference_urls: All URLs explicitly present in the answer that could confirm the store's existence and/or product offerings (include the store_url too if present). Include only valid full URLs. Deduplicate.

- woodworking_tools: A list of essential beginner woodworking tools that the answer explicitly claims this store offers.
  IMPORTANT: Only include tools from this canonical allowed list, normalized to the exact canonical spelling:
  [{allowed_tools_str}]
  If the answer uses a close synonym, normalize it to the canonical name (e.g., "measuring tape" -> "tape measure", "power drill" -> "cordless drill", "orbital sander" -> "random orbital sander", "mitre saw" -> "miter saw", "combination square"/"speed square"/"try square" -> "carpenter's square", etc.).
  Do not invent tools not clearly claimed in the answer.

- categories: Booleans indicating whether the answer claims the store offers the following categories:
  woodworking_supplies
  fabric_sewing
  painting_supplies
  home_decor
  seasonal_decorations
  framing
  adult_craft_kits
  art_supplies
  diy_home_materials
  gardening_outdoor

- operational_as_of_2026: true if the answer explicitly indicates the location is currently open/operational (as of March 2026) or shows business hours; false if it suggests closed; null if not stated.

If any requested field is missing in the answer, set it to null (or empty list for URLs and tools). Do not invent any information or URLs not present in the answer text.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for u in urls:
        if not u:
            continue
        u_str = u.strip()
        if not u_str:
            continue
        if u_str not in seen:
            seen.add(u_str)
            result.append(u_str)
    return result


def _canonicalize_tool(name: str) -> str:
    s = (name or "").strip().lower()
    if not s:
        return s

    # Simple normalization for common synonyms/variants
    synonyms = {
        "measuring tape": "tape measure",
        "tape-measure": "tape measure",

        "power drill": "cordless drill",
        "drill": "cordless drill",
        "drill/driver": "cordless drill",
        "drill driver": "cordless drill",

        "orbital sander": "random orbital sander",
        "random-orbital sander": "random orbital sander",
        "random sander": "random orbital sander",
        "sander": "random orbital sander",  # permissive mapping

        "jig saw": "jigsaw",

        "carpenters square": "carpenter's square",
        "combination square": "carpenter's square",
        "speed square": "carpenter's square",
        "try square": "carpenter's square",

        "mitre saw": "miter saw",
        "miter-saw": "miter saw",
        "chop saw": "miter saw",
    }
    if s in synonyms:
        return synonyms[s]
    return s


def _normalize_tools(tools: List[str]) -> List[str]:
    normalized = []
    for t in tools or []:
        ct = _canonicalize_tool(t)
        if ct in ALLOWED_WOODWORKING_TOOLS and ct not in normalized:
            normalized.append(ct)
    return normalized


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, parent, extracted: StoreExtraction) -> None:
    """
    Build the verification tree according to the rubric and run LLM verifications.
    """
    # Top-level critical node mirroring the rubric root
    task_root = evaluator.add_parallel(
        id="Root",
        desc="Identify one craft store location that meets all specified criteria",
        parent=parent,
        critical=True,
    )

    # Collate all sources (store_url + reference_urls)
    sources_all = _unique_urls(([extracted.store_url] if extracted.store_url else []) + (extracted.reference_urls or []))

    # Normalize the woodworking tools claimed in the answer
    normalized_tools = _normalize_tools(extracted.woodworking_tools or [])
    tools_display = ", ".join(normalized_tools) if normalized_tools else "(none provided)"

    # Create all leaf nodes (all critical under critical parent)
    n_chain = evaluator.add_leaf(
        id="Chain_Membership",
        desc="The store is part of either Michaels or Hobby Lobby chain",
        parent=task_root,
        critical=True,
    )
    n_address = evaluator.add_leaf(
        id="Physical_Location",
        desc="The store is a physical retail location with a verifiable address",
        parent=task_root,
        critical=True,
    )
    n_wood = evaluator.add_leaf(
        id="Woodworking_Tools",
        desc="The store carries woodworking supplies including at least 5 essential beginner tools from the specified list",
        parent=task_root,
        critical=True,
    )
    n_fabric = evaluator.add_leaf(
        id="Fabric_Sewing",
        desc="The store offers fabric and sewing supplies",
        parent=task_root,
        critical=True,
    )
    n_paint = evaluator.add_leaf(
        id="Painting_Supplies",
        desc="The store stocks painting supplies including paint, brushes, and related materials",
        parent=task_root,
        critical=True,
    )
    n_decor = evaluator.add_leaf(
        id="Home_Decor",
        desc="The store carries home decor items",
        parent=task_root,
        critical=True,
    )
    n_seasonal = evaluator.add_leaf(
        id="Seasonal_Decorations",
        desc="The store sells seasonal decorations",
        parent=task_root,
        critical=True,
    )
    n_framing = evaluator.add_leaf(
        id="Framing_Services",
        desc="The store provides framing services or supplies",
        parent=task_root,
        critical=True,
    )
    n_kits = evaluator.add_leaf(
        id="Adult_Craft_Kits",
        desc="The store carries craft kits suitable for adults",
        parent=task_root,
        critical=True,
    )
    n_art = evaluator.add_leaf(
        id="Art_Supplies",
        desc="The store offers art supplies including canvases, drawing materials, or similar items",
        parent=task_root,
        critical=True,
    )
    n_diy = evaluator.add_leaf(
        id="DIY_Home_Materials",
        desc="The store sells DIY home improvement supplies or materials",
        parent=task_root,
        critical=True,
    )
    n_garden = evaluator.add_leaf(
        id="Gardening_Outdoor",
        desc="The store carries gardening or outdoor project supplies",
        parent=task_root,
        critical=True,
    )
    n_state = evaluator.add_leaf(
        id="State_Location",
        desc="The store is located in Massachusetts",
        parent=task_root,
        critical=True,
    )
    n_open = evaluator.add_leaf(
        id="Currently_Operational",
        desc="The store location is currently operational as of March 2026",
        parent=task_root,
        critical=True,
    )
    n_ref = evaluator.add_leaf(
        id="Reference_URL",
        desc="A valid reference URL is provided that confirms the store's existence and product offerings",
        parent=task_root,
        critical=True,
    )

    # Build claims and additional instructions
    chain_label = extracted.chain if extracted.chain else "Michaels or Hobby Lobby"
    claim_chain = (
        f"The referenced page(s) indicate that the identified store is part of the {chain_label} retail chain."
    )

    address_text = extracted.address or ""
    claim_address = (
        f"The referenced page(s) show a physical retail store location with the following verifiable address (or an "
        f"equivalent formatting of it): '{address_text}'."
    )

    # For woodworking tools: must check that at least 5 of the specified essential tools are available per sources
    claim_wood = (
        "The referenced page(s) confirm that the store offers woodworking supplies and at least five (5) of the "
        "following essential beginner tools are available for purchase: "
        f"{', '.join(ALLOWED_WOODWORKING_TOOLS)}. "
        f"From the tools actually claimed in the answer [{tools_display}], confirm that at least five distinct tools "
        "from the allowed list are clearly available."
    )

    claim_fabric = "The referenced page(s) show that the store offers fabric and sewing supplies (e.g., fabric by the yard, thread, notions)."
    claim_paint = "The referenced page(s) show that the store stocks painting supplies, including paint and brushes (and related materials)."
    claim_decor = "The referenced page(s) show that the store carries home decor items."
    claim_seasonal = "The referenced page(s) show that the store sells seasonal decorations."
    claim_framing = "The referenced page(s) show that the store provides framing services or sells framing supplies (e.g., frames, mats, custom framing)."
    claim_kits = "The referenced page(s) show that the store carries craft kits suitable for adults."
    claim_art = "The referenced page(s) show that the store offers art supplies, including canvases and drawing materials."
    claim_diy = "The referenced page(s) show that the store sells DIY home improvement supplies or materials (e.g., craft wood, tools, adhesives, hardware)."
    claim_garden = "The referenced page(s) show that the store carries gardening or outdoor project supplies (e.g., planters, outdoor decor, outdoor craft materials)."

    claim_state = "The referenced page(s) indicate that the store is located in Massachusetts (MA)."

    claim_open = (
        "As of March 2026, the referenced page(s) indicate that this store location is currently open and operational "
        "(e.g., shows store hours, 'Open' status, or no 'permanently closed' notice)."
    )

    claim_ref = (
        "At least one of the provided reference URLs is valid and confirms both the store's existence (e.g., address or "
        "store details) and at least some of the required product offerings listed in the task."
    )

    # Additional instruction for robust checking and reasonable variants/synonyms
    add_ins_common = (
        "Allow reasonable name and formatting variants (e.g., 'St' vs 'Street', abbreviations, case differences). "
        "If multiple URLs are provided, it is sufficient that any one URL clearly supports the claim. "
        "Focus on explicit evidence from the webpage text and/or visible screenshot content."
    )

    add_ins_wood = (
        "You must verify that at least five distinct essential beginner woodworking tools are clearly available. "
        "Treat close synonyms as acceptable if obviously equivalent on the page (e.g., measuring tape=tape measure; "
        "power drill/driver=cordless drill; orbital sander=random orbital sander; jig saw=jigsaw; "
        "combination/speed/try square=carpenter's square; mitre saw= miter saw; chop saw≈miter saw). "
        "Do not count the same tool twice under different names. Count only tools from the allowed list. "
        f"Allowed tools list: {', '.join(ALLOWED_WOODWORKING_TOOLS)}. "
        f"Tools claimed in the answer: {tools_display}."
    )

    add_ins_address = (
        "Match the address content on the page to the given address allowing minor formatting differences, "
        "standard postal abbreviations, and punctuation. The page must clearly present a physical street address."
    )

    add_ins_ref = (
        "A valid reference URL should be a reachable webpage that explicitly shows the store's existence (like an address "
        "or store details) and at least some relevant product categories from the task (it does not need to show all). "
        "If none of the URLs provide such evidence, judge as not supported."
    )

    # Prepare batch verifications
    claims_and_sources = [
        (claim_chain, sources_all, n_chain, add_ins_common),
        (claim_address, sources_all, n_address, add_ins_address),
        (claim_wood, sources_all, n_wood, add_ins_wood),
        (claim_fabric, sources_all, n_fabric, add_ins_common),
        (claim_paint, sources_all, n_paint, add_ins_common),
        (claim_decor, sources_all, n_decor, add_ins_common),
        (claim_seasonal, sources_all, n_seasonal, add_ins_common),
        (claim_framing, sources_all, n_framing, add_ins_common),
        (claim_kits, sources_all, n_kits, add_ins_common),
        (claim_art, sources_all, n_art, add_ins_common),
        (claim_diy, sources_all, n_diy, add_ins_common),
        (claim_garden, sources_all, n_garden, add_ins_common),
        (claim_state, sources_all, n_state, add_ins_common),
        (claim_open, sources_all, n_open, add_ins_common),
        (claim_ref, sources_all, n_ref, add_ins_ref),
    ]

    await evaluator.batch_verify(claims_and_sources)

    # Optionally record some custom info for transparency
    evaluator.add_custom_info(
        info={
            "normalized_tools_from_answer": normalized_tools,
            "sources_used": sources_all,
        },
        info_type="extraction_summary",
        info_name="parsed_inputs_overview",
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
    Evaluate an answer for the Massachusetts Michaels/Hobby Lobby store task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Children criteria are independent checks
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
        prompt=prompt_extract_store(),
        template_class=StoreExtraction,
        extraction_name="store_extraction",
    )

    # Build tree and verify
    await build_and_verify(evaluator, root, extracted)

    return evaluator.get_summary()