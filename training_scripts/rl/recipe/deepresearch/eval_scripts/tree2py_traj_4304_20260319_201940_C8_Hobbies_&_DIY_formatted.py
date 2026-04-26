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
TASK_ID = "march_2026_woodworking_preparation"
TASK_DESCRIPTION = (
    "A parent wants to plan a month of DIY woodworking activities with their children in March 2026. "
    "They intend to attend both the free Home Depot and Lowe's kids workshops that month, and also start a family project making a cutting board at home. "
    "Create a comprehensive preparation checklist that includes: "
    "(1) the specific dates and times for both workshops in March 2026, including age requirements where applicable; "
    "(2) a complete list of all essential safety equipment needed for the home woodworking project; "
    "(3) identification of appropriate food-safe wood types for the cutting board; "
    "(4) the proper finishing method for the cutting board; and "
    "(5) note any material planning considerations such as waste allowance. "
    "Include reference URLs for all major information categories."
)

# Ground truth anchors for the month (used for claims; verification is URL-grounded)
EXPECTED_HD_DATE = "March 7, 2026"
EXPECTED_HD_START_TIME = "9:00 AM"
EXPECTED_HD_AGE = "ages 5–12"

EXPECTED_LOWES_DATE = "March 21, 2026"
EXPECTED_LOWES_TIME_WINDOW = "10:00 AM to 1:00 PM"
EXPECTED_LOWES_DURATION = "30–45 minutes"  # Approximately


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class WorkshopInfo(BaseModel):
    date: Optional[str] = None           # e.g., "March 7, 2026"
    start_time: Optional[str] = None     # e.g., "9:00 AM" (Home Depot)
    time_window: Optional[str] = None    # e.g., "10:00 AM to 1:00 PM" (Lowe's)
    duration: Optional[str] = None       # e.g., "30–45 minutes" (Lowe's)
    age_requirement: Optional[str] = None  # e.g., "ages 5–12" or "not specified"
    urls: List[str] = Field(default_factory=list)  # Reference URLs explicitly cited in the answer


class SafetyInfo(BaseModel):
    reference_urls: List[str] = Field(default_factory=list)


class WoodsInfo(BaseModel):
    woods: List[str] = Field(default_factory=list)     # e.g., ["maple", "walnut", "cherry"]
    reference_urls: List[str] = Field(default_factory=list)


class FinishingInfo(BaseModel):
    finishes: List[str] = Field(default_factory=list)  # e.g., ["mineral oil", "beeswax + mineral oil"]
    reference_urls: List[str] = Field(default_factory=list)


class MaterialPlanningInfo(BaseModel):
    waste_allowance: Optional[str] = None  # e.g., "add ~15–20% extra material"
    reference_urls: List[str] = Field(default_factory=list)


class ChecklistExtraction(BaseModel):
    home_depot: Optional[WorkshopInfo] = None
    lowes: Optional[WorkshopInfo] = None
    safety: Optional[SafetyInfo] = None
    woods: Optional[WoodsInfo] = None
    finishing: Optional[FinishingInfo] = None
    material_planning: Optional[MaterialPlanningInfo] = None
    beginner_friendly_note: Optional[bool] = None  # True if the answer explicitly notes cutting board is beginner-friendly


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_checklist() -> str:
    return """
    Extract the structured information for the March 2026 workshops and the at-home cutting board project from the answer.

    Output JSON fields:
    - home_depot: {
        date: string or null,               // e.g., "March 7, 2026"
        start_time: string or null,         // e.g., "9:00 AM"
        time_window: string or null,        // not typically used for Home Depot; set null if not provided
        duration: string or null,           // set null if not provided
        age_requirement: string or null,    // e.g., "ages 5–12"
        urls: string[]                      // URLs the answer cites for Home Depot Kids Workshop info/schedule
      }
    - lowes: {
        date: string or null,               // e.g., "March 21, 2026"
        start_time: string or null,         // set null if not provided
        time_window: string or null,        // e.g., "10:00 AM to 1:00 PM"
        duration: string or null,           // e.g., "30–45 minutes" if stated
        age_requirement: string or null,    // if the answer states “not specified” or similar, return exactly "not specified"
        urls: string[]                      // URLs the answer cites for Lowe's Kids Workshop info/schedule
      }

    - safety: {
        reference_urls: string[]            // URLs the answer cites for PPE/safety guidance for woodworking
      }

    - woods: {
        woods: string[],                    // wood species named as food-safe for cutting boards (e.g., maple, walnut, cherry)
        reference_urls: string[]            // URLs the answer cites to support the food-safe wood guidance
      }

    - finishing: {
        finishes: string[],                 // finishing methods named for cutting boards (e.g., "mineral oil", "beeswax + mineral oil")
        reference_urls: string[]            // URLs the answer cites to support the finishing guidance
      }

    - material_planning: {
        waste_allowance: string or null,    // any text referencing ~15–20% extra or similar waste allowance
        reference_urls: string[]            // URLs the answer cites to support waste allowance/material planning guidance
      }

    - beginner_friendly_note: boolean or null // true if the answer explicitly notes a cutting board is beginner-friendly; false if explicitly says it's not; null if not mentioned

    IMPORTANT:
    - Only extract data explicitly present in the answer text. Do NOT invent values.
    - For URL fields, include only valid URLs explicitly mentioned in the answer. If none are given, return an empty list.
    - Preserve the original text phrasing for times/dates/age/duration if they appear.
    - If the answer states that Lowe's age requirement is not given on the official site, set age_requirement to exactly "not specified".
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_not_specified(text: Optional[str]) -> bool:
    if text is None:
        return True
    t = text.strip().lower()
    return t in {
        "not specified", "unspecified", "n/a", "none", "not stated",
        "no age requirement", "no age requirement specified", "no age listed"
    }


def _urls_or_empty(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_workshops_verifications(evaluator: Evaluator, parent_node, data: ChecklistExtraction) -> None:
    """
    Build the 'WorkshopsInMarch2026' subtree.
    All children are critical per rubric.
    """
    node = evaluator.add_parallel(
      id="WorkshopsInMarch2026",
      desc="Includes correct March 2026 dates/times for both kids workshops, with age requirements where applicable, and workshop reference URLs",
      parent=parent_node,
      critical=True
    )

    hd = data.home_depot or WorkshopInfo()
    lowes = data.lowes or WorkshopInfo()
    hd_urls = _urls_or_empty(hd.urls)
    lowes_urls = _urls_or_empty(lowes.urls)

    # Home Depot: specific date
    hd_date_leaf = evaluator.add_leaf(
        id="HomeDepotWorkshopDateSpecific",
        desc="States the specific Home Depot workshop date in March 2026 as March 7, 2026 (first Saturday of March 2026)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Home Depot Kids Workshop for March 2026 is scheduled on {EXPECTED_HD_DATE}.",
        node=hd_date_leaf,
        sources=hd_urls,
        additional_instruction="Check the official Home Depot Kids Workshop info/schedule page(s). Allow formatting variations (e.g., Mar 7, 2026)."
    )

    # Home Depot: start time
    hd_time_leaf = evaluator.add_leaf(
        id="HomeDepotWorkshopTime",
        desc="States Home Depot workshop start time as 9:00 AM",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Home Depot Kids Workshop starts at {EXPECTED_HD_START_TIME}.",
        node=hd_time_leaf,
        sources=hd_urls,
        additional_instruction="Treat '9am', '9 a.m.', or '9:00AM' as equivalent. Focus on the start time, not end time or window."
    )

    # Home Depot: age requirement
    hd_age_leaf = evaluator.add_leaf(
        id="HomeDepotAgeRequirement",
        desc="States Home Depot Kids Workshop age requirement as ages 5–12",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Home Depot Kids Workshop is intended for {EXPECTED_HD_AGE} (or equivalent phrasing such as 5-12).",
        node=hd_age_leaf,
        sources=hd_urls,
        additional_instruction="Allow minor phrasing variants (e.g., 'for kids ages 5 to 12')."
    )

    # Home Depot: URL provided (existence check)
    hd_url_exists = evaluator.add_custom_node(
        result=len(hd_urls) > 0,
        id="HomeDepotWorkshopURL",
        desc="Provides a reference URL for the Home Depot Kids Workshop schedule/info",
        parent=node,
        critical=True
    )

    # Lowe's: specific date
    lowes_date_leaf = evaluator.add_leaf(
        id="LowesWorkshopDateSpecific",
        desc="States the specific Lowe's workshop date in March 2026 as March 21, 2026 (third Saturday of March 2026)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Lowe's Kids Workshop for March 2026 is scheduled on {EXPECTED_LOWES_DATE}.",
        node=lowes_date_leaf,
        sources=lowes_urls,
        additional_instruction="Check the official Lowe's Kids Workshop info/schedule page(s). Allow 'Mar 21, 2026' variants."
    )

    # Lowe's: time window
    lowes_time_leaf = evaluator.add_leaf(
        id="LowesWorkshopTimeRange",
        desc="States Lowe's workshop time window as 10:00 AM to 1:00 PM",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Lowe's Kids Workshop runs from {EXPECTED_LOWES_TIME_WINDOW}.",
        node=lowes_time_leaf,
        sources=lowes_urls,
        additional_instruction="Treat '10am–1pm', '10 a.m. to 1 p.m.' etc. as equivalent."
    )

    # Lowe's: duration
    lowes_duration_leaf = evaluator.add_leaf(
        id="LowesWorkshopDuration",
        desc="Notes Lowe's Kids Workshop duration as approximately 30–45 minutes",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Lowe's Kids Workshop typically takes approximately {EXPECTED_LOWES_DURATION}.",
        node=lowes_duration_leaf,
        sources=lowes_urls,
        additional_instruction="Allow minor phrasing differences such as 'about 30 to 45 minutes' or similar approximations."
    )

    # Lowe's: age requirement (either specific or 'not specified' supported by URL)
    lowes_age_leaf = evaluator.add_leaf(
        id="LowesAgeRequirement",
        desc="Provides Lowe’s Kids Workshop eligible age requirement (or explicitly states the official source does not specify an age requirement) and this claim is supported by a provided Lowe’s workshop reference URL",
        parent=node,
        critical=True
    )
    if _is_not_specified(lowes.age_requirement):
        lowes_age_claim = "The official Lowe's Kids Workshop page does not specify any age requirement."
        add_ins = "Verify that the referenced Lowe's page does not clearly list a specific age requirement; absence should be treated as 'not specified'."
    else:
        lowes_age_claim = f"The Lowe's Kids Workshop eligible age requirement is '{lowes.age_requirement}' (or equivalent phrasing)."
        add_ins = "Allow minor variants or ranges to be considered equivalent if they convey the same eligibility."
    await evaluator.verify(
        claim=lowes_age_claim,
        node=lowes_age_leaf,
        sources=lowes_urls,
        additional_instruction=add_ins
    )

    # Lowe's: URL provided (existence check)
    lowes_url_exists = evaluator.add_custom_node(
        result=len(lowes_urls) > 0,
        id="LowesWorkshopURL",
        desc="Provides a reference URL for the Lowe's Kids Workshop schedule/info",
        parent=node,
        critical=True
    )


async def build_safety_verifications(evaluator: Evaluator, parent_node, data: ChecklistExtraction) -> None:
    """
    Build 'HomeProjectSafetyEquipment' subtree.
    """
    node = evaluator.add_parallel(
        id="HomeProjectSafetyEquipment",
        desc="Lists all essential safety equipment for the at-home woodworking cutting board project and provides a safety-equipment reference URL",
        parent=parent_node,
        critical=True
    )

    safety_urls = _urls_or_empty(getattr(data.safety or SafetyInfo(), "reference_urls", []))

    # Complete list presence in the answer text (simple verify against the answer itself)
    safety_complete = evaluator.add_leaf(
        id="SafetyEquipmentCompleteList",
        desc="Includes all essential safety equipment: safety glasses/goggles, hearing protection, dust mask/respirator, gloves, and proper footwear",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The checklist includes: safety glasses or goggles, hearing protection, a dust mask or respirator, gloves, and proper footwear.",
        node=safety_complete,
        additional_instruction=(
            "Check the answer text. Accept common synonyms: safety spectacles/eye protection; earmuffs/earplugs for hearing protection; "
            "N95/respirator/dust mask for respiratory protection; work gloves; closed-toe shoes or work boots for proper footwear."
        )
    )

    # Reference URL: verify the page actually provides PPE/safety guidance (URL-verified)
    safety_ref = evaluator.add_leaf(
        id="SafetyEquipmentReferenceURL",
        desc="Provides a reference URL supporting the woodworking safety equipment guidance",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This referenced page provides woodworking safety or PPE guidance (e.g., eye, hearing, respiratory protection, gloves, and proper footwear).",
        node=safety_ref,
        sources=safety_urls,
        additional_instruction="The page should be about woodworking/tool safety or PPE recommendations, not unrelated topics."
    )


async def build_woods_verifications(evaluator: Evaluator, parent_node, data: ChecklistExtraction) -> None:
    """
    Build 'FoodSafeWoodTypes' subtree.
    """
    node = evaluator.add_parallel(
        id="FoodSafeWoodTypes",
        desc="Identifies appropriate food-safe wood types for the cutting board and provides a reference URL",
        parent=parent_node,
        critical=True
    )

    woods_urls = _urls_or_empty(getattr(data.woods or WoodsInfo(), "reference_urls", []))

    # Woods identified in the answer
    woods_identified = evaluator.add_leaf(
        id="FoodSafeWoodsIdentified",
        desc="Identifies maple, walnut, and cherry as food-safe cutting-board wood types (may list additional suitable woods as well)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The checklist identifies maple, walnut, and cherry as food-safe wood types for cutting boards.",
        node=woods_identified,
        additional_instruction="Check the answer text. Allow 'hard maple/sugar maple' as maple. Additional woods may be listed, but these three must be present."
    )

    # Reference URL: verify the page supports the guidance
    woods_ref = evaluator.add_leaf(
        id="FoodSafeWoodsReferenceURL",
        desc="Provides a reference URL supporting the food-safe wood guidance",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="According to the referenced source(s), maple, walnut, and cherry are suitable/food-safe choices for cutting boards.",
        node=woods_ref,
        sources=woods_urls,
        additional_instruction="The page should explicitly or clearly support these woods as appropriate for cutting boards."
    )


async def build_finishing_verifications(evaluator: Evaluator, parent_node, data: ChecklistExtraction) -> None:
    """
    Build 'CuttingBoardFinishingMethod' subtree.
    """
    node = evaluator.add_parallel(
        id="CuttingBoardFinishingMethod",
        desc="States the proper finishing method for the cutting board and provides a finishing reference URL",
        parent=parent_node,
        critical=True
    )

    finish_urls = _urls_or_empty(getattr(data.finishing or FinishingInfo(), "reference_urls", []))

    # Mineral oil required to be mentioned
    finish_req = evaluator.add_leaf(
        id="MineralOilFinishRequired",
        desc="Specifies mineral oil as the cutting board finish (may also mention an optional beeswax/mineral-oil mixture, but mineral oil must be included)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The checklist specifies mineral oil as the cutting board finish (optionally also a beeswax + mineral oil blend).",
        node=finish_req,
        additional_instruction="Check the answer text. Accept 'food-grade mineral oil', 'butcher block oil' as equivalent when clearly mineral oil based."
    )

    # Reference URL supports finishing method
    finish_ref = evaluator.add_leaf(
        id="FinishingReferenceURL",
        desc="Provides a reference URL supporting the cutting board finishing method",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The referenced source(s) recommend(s) mineral oil (optionally beeswax + mineral oil) as a food-safe finish for cutting boards.",
        node=finish_ref,
        sources=finish_urls,
        additional_instruction="The page should endorse mineral oil or a beeswax/mineral oil mixture as a food-safe cutting board finish."
    )


async def build_material_verifications(evaluator: Evaluator, parent_node, data: ChecklistExtraction) -> None:
    """
    Build 'MaterialPlanningConsiderations' subtree.
    """
    node = evaluator.add_parallel(
        id="MaterialPlanningConsiderations",
        desc="Notes material planning considerations for the DIY project (including waste allowance) and provides a reference URL",
        parent=parent_node,
        critical=True
    )

    mat_urls = _urls_or_empty(getattr(data.material_planning or MaterialPlanningInfo(), "reference_urls", []))

    # Waste allowance mentioned in answer
    waste_leaf = evaluator.add_leaf(
        id="WasteAllowanceMentioned",
        desc="Mentions adding approximately 15–20% extra material to account for waste/mistakes",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The checklist mentions adding approximately 15–20% extra material to allow for waste or mistakes.",
        node=waste_leaf,
        additional_instruction="Check the answer text. Accept formatting like 15-20%, 15 to 20 percent, ~15–20%, etc."
    )

    # Reference URL supports the guidance
    mat_ref = evaluator.add_leaf(
        id="MaterialPlanningReferenceURL",
        desc="Provides a reference URL supporting the material planning/waste allowance guidance",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The referenced source(s) recommend(s) including roughly 15–20% extra material for waste in woodworking or similar projects.",
        node=mat_ref,
        sources=mat_urls,
        additional_instruction="The page should discuss waste allowance, overage, or extra material planning around the 10–20% range (15–20% preferred)."
    )


async def build_beginner_note_verification(evaluator: Evaluator, parent_node, data: ChecklistExtraction) -> None:
    """
    Build 'BeginnerFriendlyNote' leaf.
    """
    leaf = evaluator.add_leaf(
        id="BeginnerFriendlyNote",
        desc="Notes that a cutting board is beginner-friendly (as stated in constraints)",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="The checklist notes that a cutting board is a beginner-friendly project.",
        node=leaf,
        additional_instruction="Check the answer text for phrases like 'beginner-friendly', 'great for beginners', 'good first project', etc."
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
    Evaluate an answer for the March 2026 woodworking preparation checklist task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Root aggregates categories in parallel
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

    # IMPORTANT: Root is critical per rubric; all children under it must also be critical (framework constraint)
    root.critical = True

    # Add ground truth anchors for transparency
    evaluator.add_ground_truth({
        "expected_workshops": {
            "home_depot": {
                "date": EXPECTED_HD_DATE,
                "start_time": EXPECTED_HD_START_TIME,
                "age_requirement": EXPECTED_HD_AGE
            },
            "lowes": {
                "date": EXPECTED_LOWES_DATE,
                "time_window": EXPECTED_LOWES_TIME_WINDOW,
                "duration": EXPECTED_LOWES_DURATION
            }
        },
        "at_home_project_requirements": {
            "safety_items_required": [
                "safety glasses/goggles",
                "hearing protection",
                "dust mask/respirator",
                "gloves",
                "proper footwear"
            ],
            "food_safe_woods_required": ["maple", "walnut", "cherry"],
            "finish_required": "mineral oil (optionally beeswax + mineral oil)",
            "material_waste_allowance": "approximately 15–20%"
        }
    }, gt_type="ground_truth")

    # 1) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_checklist(),
        template_class=ChecklistExtraction,
        extraction_name="checklist_extraction"
    )

    # 2) Build verification tree according to rubric
    # Workshops subtree
    await build_workshops_verifications(evaluator, root, extracted)

    # Safety equipment subtree
    await build_safety_verifications(evaluator, root, extracted)

    # Food-safe wood types subtree
    await build_woods_verifications(evaluator, root, extracted)

    # Finishing method subtree
    await build_finishing_verifications(evaluator, root, extracted)

    # Material planning considerations subtree
    await build_material_verifications(evaluator, root, extracted)

    # Beginner-friendly note leaf
    await build_beginner_note_verification(evaluator, root, extracted)

    # 3) Return structured summary
    return evaluator.get_summary()