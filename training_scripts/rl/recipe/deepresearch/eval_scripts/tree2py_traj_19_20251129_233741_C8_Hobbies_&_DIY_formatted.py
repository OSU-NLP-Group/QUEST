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
TASK_ID = "thanksgiving_craft_shopping_guide_2024"
TASK_DESCRIPTION = (
    "You are planning to create DIY Thanksgiving decorations and want to shop for craft materials on Black Friday 2024 (November 29, 2024). "
    "For the three major craft store chains—Hobby Lobby, Michaels, and JoAnn Fabrics:\n\n"
    "1. Indicate whether each store will be open on Black Friday 2024\n"
    "2. For any stores that are open, provide their Black Friday operating hours (both opening and closing times)\n"
    "3. For any stores that are open, describe their specific Black Friday 2024 yarn deals or promotions\n\n"
    "Additionally, provide comprehensive materials lists for creating the following DIY Thanksgiving projects:\n"
    "4. A Thanksgiving wreath\n"
    "5. A Thanksgiving centerpiece\n\n"
    "For each project, your materials list should include all necessary categories of supplies: structural bases or forms, decorative elements, "
    "fastening or attachment materials, and accent items or finishing touches."
)

BLACK_FRIDAY_DATE_TEXT = "Black Friday 2024 (November 29, 2024)"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreHours(BaseModel):
    opening_time: Optional[str] = None
    closing_time: Optional[str] = None
    hours_text: Optional[str] = None  # e.g., "6:00 AM – 9:00 PM"


class StoreInfo(BaseModel):
    status: Optional[str] = None  # Expect "open" or "closed" (case-insensitive). If unknown, null.
    hours: Optional[StoreHours] = None
    yarn_deal: Optional[str] = None  # Description string for yarn-specific promotion/deal
    sources: List[str] = Field(default_factory=list)  # URLs explicitly cited in the answer for this store


class StoresExtraction(BaseModel):
    hobby_lobby: Optional[StoreInfo] = None
    michaels: Optional[StoreInfo] = None
    joann_fabrics: Optional[StoreInfo] = None


class ProjectMaterials(BaseModel):
    base_form: List[str] = Field(default_factory=list)               # Structural base/form items
    decorative_elements: List[str] = Field(default_factory=list)     # Leaves, florals, pumpkins, ribbons, etc.
    fastening_materials: List[str] = Field(default_factory=list)     # Floral wire, hot glue, tape, pipe cleaners, etc.
    accent_touches: List[str] = Field(default_factory=list)          # Bows, signage, LED candles, glitter, etc.


class ProjectsExtraction(BaseModel):
    wreath: Optional[ProjectMaterials] = None
    centerpiece: Optional[ProjectMaterials] = None


class BlackFridayCraftExtraction(BaseModel):
    stores: Optional[StoresExtraction] = None
    projects: Optional[ProjectsExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_black_friday_craft_info() -> str:
    return """
    Extract the requested information from the answer text. Follow these instructions carefully.

    Part A — Stores (Hobby Lobby, Michaels, JoAnn Fabrics):
    For each store, extract the following:
    - status: Whether the answer says the store is "open" or "closed" on Black Friday 2024 (use the lowercase literal "open" or "closed"; if not stated, set to null).
    - hours: If the store is open, extract both opening_time and closing_time in text form (e.g., "6 AM", "8:00 am", "9 PM"). Also provide hours_text as a single combined range string if presented (e.g., "6 AM–9 PM"). If hours are not given or the store is closed, set opening_time, closing_time, and hours_text to null or an 'N/A' text only if the answer explicitly says so.
    - yarn_deal: If the store is open, extract the yarn-specific Black Friday 2024 deal description exactly as stated. If no yarn-specific deal is given, set to null. If the store is closed and the answer explicitly marks this not applicable, set to something like "N/A" or null.
    - sources: Extract all URLs explicitly cited in the answer that support the store's Black Friday status/hours/deal. Include only valid URLs. If none are provided, return an empty list.

    Stores to extract under 'stores':
    - hobby_lobby
    - michaels
    - joann_fabrics

    Part B — Projects:
    Extract the materials lists for:
    1) wreath (Thanksgiving wreath)
    2) centerpiece (Thanksgiving centerpiece)

    For each project, extract four arrays of strings:
    - base_form: structural bases or forms
    - decorative_elements: decorative elements (florals/leaves/mini pumpkins/ribbons/etc.)
    - fastening_materials: items used to assemble/attach/secure components (floral wire, hot glue, tape, pipe cleaners, floral picks, etc.)
    - accent_touches: accent items or finishing touches (bows, signage, LED candles, glitter, berries, etc.)

    Rules:
    - Extract only information explicitly present in the answer text.
    - Normalize the 'status' to exactly "open" or "closed" when clearly stated. If unclear or missing, set to null.
    - Times should be kept as strings as presented (e.g., "6 AM", "8:00 am").
    - For sources, extract only URLs that appear in the answer (plain or in markdown).
    - If an item/category is not present for a project, return an empty array for that category.

    Return a JSON object consistent with the expected schema.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_status(status: Optional[str]) -> Optional[str]:
    if not status:
        return None
    s = status.strip().lower()
    if "open" in s and "closed" not in s:
        return "open"
    if "closed" in s and "open" not in s:
        return "closed"
    # Ambiguous or not provided
    return None


def _is_na_text(text: Optional[str]) -> bool:
    if not text:
        return True
    t = text.strip().lower()
    return t in {"n/a", "na", "not applicable", "none", "no deal", "no hours", "no information"} or t == ""


def _compose_hours_claim(hours: Optional[StoreHours]) -> Optional[str]:
    if not hours:
        return None
    if hours.opening_time and hours.closing_time:
        return f"{hours.opening_time} to {hours.closing_time}"
    if hours.hours_text:
        return hours.hours_text
    return None


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_store_information(
    evaluator: Evaluator,
    parent_node,
    store_label: str,          # Human-readable name for claims (e.g., "Hobby Lobby")
    status_node_id: str,       # e.g., "Hobby_Lobby_Black_Friday_Status"
    hours_node_id: str,        # e.g., "Hobby_Lobby_Hours"
    deal_node_id: str,         # e.g., "Hobby_Lobby_Yarn_Deal"
    store_info: Optional[StoreInfo],
):
    """
    Build a sequential verification sub-tree for one store and run checks:
    1) Black Friday status (critical)
    2) Hours correctness or N/A if closed (critical)
    3) Yarn deal correctness or N/A if closed (critical)
    """
    # Container node for the store (sequential)
    container = evaluator.add_sequential(
        id=f"{store_label.replace(' ', '_')}_Information",
        desc=f"{store_label} Black Friday 2024 status, and (if open) hours and yarn deals.",
        parent=parent_node,
        critical=False
    )

    # Prepare normalized data
    s_info = store_info or StoreInfo()
    normalized_status = _normalize_status(s_info.status)

    # 1) Status (critical)
    # If status is clearly "open" or "closed", verify via sources; otherwise fail this leaf.
    if normalized_status in {"open", "closed"}:
        status_leaf = evaluator.add_leaf(
            id=status_node_id,
            desc=f"Correctly indicates whether {store_label} is open on {BLACK_FRIDAY_DATE_TEXT}.",
            parent=container,
            critical=True
        )
        status_claim = f"{store_label} will be {normalized_status} on {BLACK_FRIDAY_DATE_TEXT}."
        await evaluator.verify(
            claim=status_claim,
            node=status_leaf,
            sources=s_info.sources,
            additional_instruction=f"Use the cited source(s) to confirm whether {store_label} is open or closed on {BLACK_FRIDAY_DATE_TEXT}. "
                                   f"If the page mentions Black Friday hours or a holiday schedule for 2024-11-29, that should be considered."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=status_node_id,
            desc=f"Correctly indicates whether {store_label} is open on {BLACK_FRIDAY_DATE_TEXT}.",
            parent=container,
            critical=True
        )

    # 2) Hours (critical)
    # If open: must provide both opening and closing times or a clear hours_text; verify via sources.
    # If closed: hours should be omitted or marked not applicable (custom check).
    if normalized_status == "open":
        hours_claim_text = _compose_hours_claim(s_info.hours)
        if hours_claim_text:
            hours_leaf = evaluator.add_leaf(
                id=hours_node_id,
                desc=f"If {store_label} is open, provides correct {BLACK_FRIDAY_DATE_TEXT} operating hours (opening and closing). "
                     f"If closed, hours are omitted or N/A.",
                parent=container,
                critical=True
            )
            claim = f"On {BLACK_FRIDAY_DATE_TEXT}, {store_label}'s hours are {hours_claim_text}."
            await evaluator.verify(
                claim=claim,
                node=hours_leaf,
                sources=s_info.sources,
                additional_instruction=f"Verify the Black Friday 2024 hours shown for {store_label}. "
                                       f"The claim should be supported by the cited page(s); allow minor formatting differences in times."
            )
        else:
            evaluator.add_custom_node(
                result=False,
                id=hours_node_id,
                desc=f"If {store_label} is open, provides correct {BLACK_FRIDAY_DATE_TEXT} operating hours (opening and closing). "
                     f"If closed, hours are omitted or N/A.",
                parent=container,
                critical=True
            )
    elif normalized_status == "closed":
        # Check omission / N/A for hours
        hours_na_ok = True
        if s_info.hours:
            # If any of these appear filled in, that's not acceptable for closed
            if s_info.hours.opening_time or s_info.hours.closing_time:
                hours_na_ok = False
            elif s_info.hours.hours_text and not _is_na_text(s_info.hours.hours_text):
                hours_na_ok = False
        evaluator.add_custom_node(
            result=hours_na_ok,
            id=hours_node_id,
            desc=f"If {store_label} is open, provides correct {BLACK_FRIDAY_DATE_TEXT} operating hours (opening and closing). "
                 f"If closed, hours are omitted or N/A.",
            parent=container,
            critical=True
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=hours_node_id,
            desc=f"If {store_label} is open, provides correct {BLACK_FRIDAY_DATE_TEXT} operating hours (opening and closing). "
                 f"If closed, hours are omitted or N/A.",
            parent=container,
            critical=True
        )

    # 3) Yarn Deal (critical)
    if normalized_status == "open":
        if s_info.yarn_deal and s_info.yarn_deal.strip():
            deal_leaf = evaluator.add_leaf(
                id=deal_node_id,
                desc=f"If {store_label} is open, correctly describes its specific {BLACK_FRIDAY_DATE_TEXT} yarn deal/promotion; "
                     f"if closed, deal is omitted or N/A.",
                parent=container,
                critical=True
            )
            claim = f"On {BLACK_FRIDAY_DATE_TEXT}, {store_label} has the following yarn deal or promotion: {s_info.yarn_deal}"
            await evaluator.verify(
                claim=claim,
                node=deal_leaf,
                sources=s_info.sources,
                additional_instruction="Focus specifically on yarn-related Black Friday 2024 promotions. "
                                       "The described deal should be explicitly present on the cited page(s). "
                                       "Allow paraphrasing but ensure the essential discount/offer matches."
            )
        else:
            evaluator.add_custom_node(
                result=False,
                id=deal_node_id,
                desc=f"If {store_label} is open, correctly describes its specific {BLACK_FRIDAY_DATE_TEXT} yarn deal/promotion; "
                     f"if closed, deal is omitted or N/A.",
                parent=container,
                critical=True
            )
    elif normalized_status == "closed":
        # Deal should be omitted or N/A
        deal_na_ok = _is_na_text(s_info.yarn_deal)
        evaluator.add_custom_node(
            result=deal_na_ok,
            id=deal_node_id,
            desc=f"If {store_label} is open, correctly describes its specific {BLACK_FRIDAY_DATE_TEXT} yarn deal/promotion; "
                 f"if closed, deal is omitted or N/A.",
            parent=container,
            critical=True
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=deal_node_id,
            desc=f"If {store_label} is open, correctly describes its specific {BLACK_FRIDAY_DATE_TEXT} yarn deal/promotion; "
                 f"if closed, deal is omitted or N/A.",
            parent=container,
            critical=True
        )


def add_project_materials_checks(
    evaluator: Evaluator,
    parent_node,
    project_node_id: str,
    project_desc: str,
    materials: Optional[ProjectMaterials],
    category_nodes: Dict[str, str]
):
    """
    Adds a parallel node for a project's materials verification and four critical custom checks
    that ensure each required category includes at least one item.
    """
    container = evaluator.add_parallel(
        id=project_node_id,
        desc=project_desc,
        parent=parent_node,
        critical=False
    )

    mats = materials or ProjectMaterials()

    # Structural base/form
    evaluator.add_custom_node(
        result=bool(mats.base_form and len(mats.base_form) > 0),
        id=category_nodes["base"],
        desc="Includes a structural base/form.",
        parent=container,
        critical=True
    )

    # Decorative elements
    evaluator.add_custom_node(
        result=bool(mats.decorative_elements and len(mats.decorative_elements) > 0),
        id=category_nodes["decor"],
        desc="Includes decorative elements.",
        parent=container,
        critical=True
    )

    # Fastening/attachment materials
    evaluator.add_custom_node(
        result=bool(mats.fastening_materials and len(mats.fastening_materials) > 0),
        id=category_nodes["fasten"],
        desc="Includes fastening/attachment materials used to assemble/secure decorations.",
        parent=container,
        critical=True
    )

    # Accent/finishing touches
    evaluator.add_custom_node(
        result=bool(mats.accent_touches and len(mats.accent_touches) > 0),
        id=category_nodes["accent"],
        desc="Includes accent items/finishing touches.",
        parent=container,
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
    Evaluate an answer for the Thanksgiving Craft Shopping Guide task.
    """
    # Initialize evaluator with a parallel root
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

    # Extraction
    extracted: BlackFridayCraftExtraction = await evaluator.extract(
        prompt=prompt_extract_black_friday_craft_info(),
        template_class=BlackFridayCraftExtraction,
        extraction_name="black_friday_craft_info"
    )

    # Top-level container (parallel) to reflect rubric's main node
    guide_node = evaluator.add_parallel(
        id="Thanksgiving_Craft_Shopping_Guide",
        desc="Information about craft store Black Friday 2024 availability/hours/yarn deals, plus materials lists for two Thanksgiving DIY projects.",
        parent=root,
        critical=False
    )

    stores = extracted.stores or StoresExtraction()

    # Hobby Lobby
    await verify_store_information(
        evaluator=evaluator,
        parent_node=guide_node,
        store_label="Hobby Lobby",
        status_node_id="Hobby_Lobby_Black_Friday_Status",
        hours_node_id="Hobby_Lobby_Hours",
        deal_node_id="Hobby_Lobby_Yarn_Deal",
        store_info=stores.hobby_lobby
    )

    # Michaels
    await verify_store_information(
        evaluator=evaluator,
        parent_node=guide_node,
        store_label="Michaels",
        status_node_id="Michaels_Black_Friday_Status",
        hours_node_id="Michaels_Hours",
        deal_node_id="Michaels_Yarn_Deal",
        store_info=stores.michaels
    )

    # JoAnn Fabrics (often branded as JOANN)
    await verify_store_information(
        evaluator=evaluator,
        parent_node=guide_node,
        store_label="JoAnn Fabrics",
        status_node_id="JoAnn_Black_Friday_Status",
        hours_node_id="JoAnn_Hours",
        deal_node_id="JoAnn_Yarn_Deal",
        store_info=stores.joann_fabrics
    )

    # Projects: Wreath and Centerpiece
    projects = extracted.projects or ProjectsExtraction()

    add_project_materials_checks(
        evaluator=evaluator,
        parent_node=guide_node,
        project_node_id="Thanksgiving_Wreath_Materials",
        project_desc="Materials list for a DIY Thanksgiving wreath includes all required supply categories.",
        materials=projects.wreath,
        category_nodes={
            "base": "Wreath_Structural_Base_or_Form",
            "decor": "Wreath_Decorative_Elements",
            "fasten": "Wreath_Fastening_or_Attachment_Materials",
            "accent": "Wreath_Accent_or_Finishing_Touches"
        }
    )

    add_project_materials_checks(
        evaluator=evaluator,
        parent_node=guide_node,
        project_node_id="Thanksgiving_Centerpiece_Materials",
        project_desc="Materials list for a DIY Thanksgiving centerpiece includes all required supply categories.",
        materials=projects.centerpiece,
        category_nodes={
            "base": "Centerpiece_Structural_Base_or_Form",
            "decor": "Centerpiece_Decorative_Elements",
            "fasten": "Centerpiece_Fastening_or_Attachment_Materials",
            "accent": "Centerpiece_Accent_or_Finishing_Touches"
        }
    )

    # Return structured evaluation summary
    return evaluator.get_summary()