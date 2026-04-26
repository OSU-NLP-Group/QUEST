import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "il_craft_vendor_plan_2026"
TASK_DESCRIPTION = """
You are planning to launch a craft vendor business selling handmade wooden cutting boards at Illinois craft fairs in spring 2026. To prepare, you need to create a comprehensive project plan that includes 3D printed display components, woodworking specifications, fair selection, and business requirements.

Your task is to identify and specify:

1. Four 3D Printed Display Components:
For each of the following display items, identify an Illinois public library makerspace that can accommodate the specifications, and provide the makerspace's URL:

- Primary Display Stand: Dimensions 9" × 9" × 5", estimated print time ≤4 hours, weight ≤75 grams
- Small Hanging Hooks: Dimensions 3" × 2" × 1", estimated print time ≤2 hours, weight ≤25 grams
- Sign Holders: Dimensions 7" × 8" × 5", estimated print time ≤8 hours, weight ≤95 grams
- Price Tag Holders: Dimensions 4" × 3" × 2", estimated print time ≤3 hours, weight ≤40 grams

For each item, verify that the identified makerspace:
- Has build volume that accommodates the specified dimensions
- Allows print time limits that meet or exceed the requirement
- Has weight limits that meet or exceed the requirement
- Uses PLA filament
- Accepts STL file format

2. Cutting Board Specifications:
Specify the cutting board design with:
- Finished thickness of at least 1.25 inches (minimum safe thickness for cutting boards)
- Length between 10-16 inches
- Width between 8-12 inches
- A food-safe hardwood species
- Food-safe finish using mineral oil and beeswax mixture

3. Spring 2026 Illinois Craft Fair:
Identify a specific craft fair in Illinois occurring in spring 2026 (March-May) that:
- Has an application deadline allowing at least 6 weeks of preparation time before the event
- Provides standard 10' × 10' booth space or clearly specified booth dimensions
- Provide the fair's URL

4. Business Requirements:
Specify:
- General liability insurance with minimum coverage of $1 million per occurrence
- Estimated insurance cost (monthly or per-event basis)
- Illinois sellers permit or vendor license requirement
- Display table height within ergonomic range (32-38 inches)

5. Technical Specifications:
Provide:
- PLA nozzle temperature range (should be 190-220°C)
- PLA bed temperature range (should be 50-60°C)
- Confirmation that PLA material is suitable for non-food-contact display items

Provide all information with supporting URL references where specifications are found.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MakerspaceInfo(BaseModel):
    makerspace_name: Optional[str] = None
    makerspace_url: Optional[str] = None
    build_volume: Optional[str] = None  # e.g., "220x220x250 mm" or "9x9x5 in"
    time_limit_policy: Optional[str] = None  # e.g., "Up to 4 hours per reservation"
    weight_limit_policy: Optional[str] = None  # e.g., "Max weight 100g" or "No specified limit"
    materials_mentioned: List[str] = Field(default_factory=list)  # e.g., ["PLA", "PETG"]
    file_formats_accepted: List[str] = Field(default_factory=list)  # e.g., ["STL", "OBJ"]


class DisplayItemsExtraction(BaseModel):
    primary_display_stand: Optional[MakerspaceInfo] = None
    small_hanging_hooks: Optional[MakerspaceInfo] = None
    sign_holders: Optional[MakerspaceInfo] = None
    price_tag_holders: Optional[MakerspaceInfo] = None


class CuttingBoardExtraction(BaseModel):
    thickness: Optional[str] = None
    length: Optional[str] = None
    width: Optional[str] = None
    wood_species: Optional[str] = None
    finish: Optional[str] = None
    sources: List[str] = Field(default_factory=list)  # URLs supporting woodworking specs


class CraftFairExtraction(BaseModel):
    fair_name: Optional[str] = None
    fair_url: Optional[str] = None
    location: Optional[str] = None  # e.g., "Springfield, IL"
    event_date: Optional[str] = None  # e.g., "May 12, 2026"
    application_deadline: Optional[str] = None  # e.g., "March 25, 2026"
    booth_size: Optional[str] = None  # e.g., "10x10 feet"


class InsuranceExtraction(BaseModel):
    coverage_min: Optional[str] = None  # e.g., "$1,000,000 per occurrence"
    cost_estimate: Optional[str] = None  # e.g., "$25 per event" or "$40/month"
    urls: List[str] = Field(default_factory=list)  # insurer or event vendor policy pages


class LicensingExtraction(BaseModel):
    requirement_text: Optional[str] = None  # e.g., "Illinois seller's permit required"
    url: Optional[str] = None  # IL Dept. of Revenue or official page


class DisplaySetupExtraction(BaseModel):
    table_height: Optional[str] = None  # e.g., "36 inches"
    url: Optional[str] = None  # ergonomic guideline URL if provided


class TechnicalSpecsExtraction(BaseModel):
    nozzle_temp_range: Optional[str] = None  # e.g., "190-220°C"
    bed_temp_range: Optional[str] = None  # e.g., "50-60°C"
    material_suitability: Optional[str] = None  # e.g., "PLA suitable for non-food-contact display items"
    urls: List[str] = Field(default_factory=list)  # technical reference URLs


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_display_items() -> str:
    return """
    Extract 3D printed display makerspace info for four items. For each item, identify the Illinois public library makerspace used and extract the requested fields exactly as stated in the answer. If a field is missing, return null for that field; lists should be empty if not specified.

    Items and required fields:
    - primary_display_stand:
        makerspace_name
        makerspace_url
        build_volume
        time_limit_policy
        weight_limit_policy
        materials_mentioned (list)
        file_formats_accepted (list)
    - small_hanging_hooks:
        makerspace_name
        makerspace_url
        build_volume
        time_limit_policy
        weight_limit_policy
        materials_mentioned (list)
        file_formats_accepted (list)
    - sign_holders:
        makerspace_name
        makerspace_url
        build_volume
        time_limit_policy
        weight_limit_policy
        materials_mentioned (list)
        file_formats_accepted (list)
    - price_tag_holders:
        makerspace_name
        makerspace_url
        build_volume
        time_limit_policy
        weight_limit_policy
        materials_mentioned (list)
        file_formats_accepted (list)
    """


def prompt_extract_cutting_board() -> str:
    return """
    Extract the cutting board design specifications and any supporting URLs mentioned in the answer.
    Required fields:
    - thickness: finished thickness (string as written)
    - length: finished length (string as written)
    - width: finished width (string as written)
    - wood_species: the chosen hardwood species
    - finish: the food-safe finish description (should mention mineral oil and beeswax)
    - sources: array of URLs supporting these specs (if present)
    """


def prompt_extract_craft_fair() -> str:
    return """
    Extract the Illinois craft fair details mentioned in the answer.
    Required fields:
    - fair_name
    - fair_url
    - location: city/state or description indicating Illinois
    - event_date: date string of the event
    - application_deadline: date string for the application deadline
    - booth_size: booth dimensions or "10x10" if standard is stated
    """


def prompt_extract_insurance() -> str:
    return """
    Extract vendor liability insurance info and supporting URLs from the answer.
    Required fields:
    - coverage_min: minimum coverage specified (string, e.g., "$1,000,000 per occurrence")
    - cost_estimate: estimated cost (string, monthly or per-event)
    - urls: list of URLs supporting the coverage and/or cost
    """


def prompt_extract_licensing() -> str:
    return """
    Extract Illinois business licensing requirement information and its supporting URL from the answer.
    Required fields:
    - requirement_text: description of seller's permit or vendor license requirement
    - url: the URL (prefer official IL Department of Revenue or government site)
    """


def prompt_extract_display_setup() -> str:
    return """
    Extract the vendor booth display setup info for table height and any supporting URL.
    Required fields:
    - table_height: specified table height in inches (string as written)
    - url: URL supporting ergonomic range if provided
    """


def prompt_extract_technical_specs() -> str:
    return """
    Extract PLA technical specifications and supporting URLs.
    Required fields:
    - nozzle_temp_range: PLA nozzle temperature range (string)
    - bed_temp_range: PLA bed temperature range (string)
    - material_suitability: confirmation text that PLA is suitable for non-food-contact display items
    - urls: array of URLs supporting these technical specs
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dims_str(dims: Tuple[int, int, int]) -> str:
    L, W, H = dims
    return f'{L}" × {W}" × {H}"'


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_display_item(
    evaluator: Evaluator,
    parent_node,
    item_node_id: str,
    item_desc: str,
    item_info: Optional[MakerspaceInfo],
    required_dims_in: Tuple[int, int, int],
    required_hours: int,
    required_weight_g: int,
    leaf_prefix: str,
) -> None:
    """
    Verify one display item makerspace meets specifications.
    Parent is a critical parallel node. Each leaf is critical.
    """
    # Create item node under Display_Items (must be critical because parent is critical)
    item_node = evaluator.add_parallel(
        id=item_node_id,
        desc=item_desc,
        parent=parent_node,
        critical=True,
    )

    # Existence of makerspace URL reference (critical)
    ms_url_present = bool(item_info and item_info.makerspace_url and item_info.makerspace_url.strip())
    evaluator.add_custom_node(
        result=ms_url_present,
        id=f"{leaf_prefix}_Reference",
        desc=f"URL reference for makerspace {leaf_prefix[-1]}",
        parent=item_node,
        critical=True,
    )

    ms_url = item_info.makerspace_url if item_info else None
    dims_text = _dims_str(required_dims_in)

    # Build Volume check
    vol_leaf = evaluator.add_leaf(
        id=f"{leaf_prefix}_Build_Volume",
        desc=f"Makerspace build volume accommodates {dims_text}",
        parent=item_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The makerspace's 3D printer build volume meets or exceeds {dims_text} "
            f"(any printer at the makerspace qualifies; conversions between mm and inches are acceptable)."
        ),
        node=vol_leaf,
        sources=ms_url,
        additional_instruction=(
            "Look for build volume or maximum print size on the page (e.g., 220x220x250 mm). "
            "If multiple printers are listed, it's sufficient if at least one meets or exceeds the required volume. "
            "You may convert mm to inches: 25.4 mm = 1 inch."
        ),
    )

    # Time Limit check
    time_leaf = evaluator.add_leaf(
        id=f"{leaf_prefix}_Time_Limit",
        desc=f"Makerspace time limit allows ≥{required_hours} hours",
        parent=item_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The makerspace allows at least {required_hours} hours per print, session, reservation, or booking."
        ),
        node=time_leaf,
        sources=ms_url,
        additional_instruction=(
            "Check reservation policies, printer booking rules, or posted time limits. "
            "Accept per-session or per-reservation limits meeting/exceeding the requirement."
        ),
    )

    # Weight Limit check
    weight_leaf = evaluator.add_leaf(
        id=f"{leaf_prefix}_Weight_Limit",
        desc=f"Makerspace weight limit allows ≥{required_weight_g} grams",
        parent=item_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The makerspace explicitly allows printed object weights of at least {required_weight_g} grams "
            f"or has no stated weight limit below {required_weight_g} grams."
        ),
        node=weight_leaf,
        sources=ms_url,
        additional_instruction=(
            "Look for any stated limits related to print weight, material usage, or similar constraints. "
            "If the page does not mention weight limits, conclude NOT SUPPORTED."
        ),
    )

    # Material PLA check
    material_leaf = evaluator.add_leaf(
        id=f"{leaf_prefix}_Material",
        desc="Makerspace uses PLA filament",
        parent=item_node,
        critical=True,
    )
    await evaluator.verify(
        claim="PLA filament is used or permitted by the makerspace for 3D printing.",
        node=material_leaf,
        sources=ms_url,
        additional_instruction=(
            "Search the page for allowed materials. Accept mentions such as 'PLA', 'PLA+', or 'PLA is recommended'."
        ),
    )

    # File format STL check
    format_leaf = evaluator.add_leaf(
        id=f"{leaf_prefix}_Format",
        desc="Makerspace accepts STL file format",
        parent=item_node,
        critical=True,
    )
    await evaluator.verify(
        claim="STL files are accepted for 3D printing at this makerspace.",
        node=format_leaf,
        sources=ms_url,
        additional_instruction=(
            "Check file preparation guidelines or acceptable file formats. Accept 'STL' or similar phrasing."
        ),
    )


async def verify_woodworking_component(
    evaluator: Evaluator,
    parent_node,
    cb: CuttingBoardExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Woodworking_Component",
        desc="Cutting board specifications",
        parent=parent_node,
        critical=True,
    )

    # Thickness ≥ 1.25"
    leaf_thickness = evaluator.add_leaf(
        id="Board_Thickness_Spec",
        desc="Finished thickness at least 1.25 inches",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The cutting board finished thickness specified in the answer is at least 1.25 inches.",
        node=leaf_thickness,
        additional_instruction=(
            "Check the answer text for the thickness value and confirm it is ≥ 1.25 inches."
        ),
    )

    # Length 10-16"
    leaf_length = evaluator.add_leaf(
        id="Board_Length_Spec",
        desc="Length between 10-16 inches",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The cutting board length specified in the answer is between 10 and 16 inches (inclusive).",
        node=leaf_length,
        additional_instruction="Check the answer text for the length value and confirm the range.",
    )

    # Width 8-12"
    leaf_width = evaluator.add_leaf(
        id="Board_Width_Spec",
        desc="Width between 8-12 inches",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The cutting board width specified in the answer is between 8 and 12 inches (inclusive).",
        node=leaf_width,
        additional_instruction="Check the answer text for the width value and confirm the range.",
    )

    # Food-safe hardwood species
    leaf_species = evaluator.add_leaf(
        id="Wood_Species",
        desc="Food-safe hardwood species specified",
        parent=node,
        critical=True,
    )
    species = cb.wood_species or ""
    await evaluator.verify(
        claim=f"The specified wood species '{species}' is a food-safe hardwood suitable for cutting boards.",
        node=leaf_species,
        sources=cb.sources,  # Use any woodworking references provided in the answer
        additional_instruction=(
            "Verify that the referenced page(s) indicate the species is hardwood and suitable for cutting boards "
            "(e.g., maple, walnut, cherry)."
        ),
    )

    # Food-safe finish: mineral oil + beeswax
    leaf_finish = evaluator.add_leaf(
        id="Finish_Type",
        desc="Food-safe finish using mineral oil and beeswax",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="A mineral oil and beeswax mixture is a food-safe finish for cutting boards.",
        node=leaf_finish,
        sources=cb.sources,
        additional_instruction="Verify from referenced page(s) that mineral oil + beeswax is safe for cutting boards.",
    )


async def verify_craft_fair_component(
    evaluator: Evaluator,
    parent_node,
    fair: CraftFairExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Craft_Fair_Component",
        desc="Illinois spring 2026 craft fair identification",
        parent=parent_node,
        critical=True,
    )

    # Presence of fair URL
    url_present = bool(fair and fair.fair_url and fair.fair_url.strip())
    evaluator.add_custom_node(
        result=url_present,
        id="Fair_Reference",
        desc="URL reference for craft fair",
        parent=node,
        critical=True,
    )
    url = fair.fair_url if fair else None

    # Location in IL
    leaf_loc = evaluator.add_leaf(
        id="Fair_Location_IL",
        desc="Fair located in Illinois",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This craft fair is located in Illinois.",
        node=leaf_loc,
        sources=url,
        additional_instruction="Look for city/state on the page and confirm the state is Illinois.",
    )

    # Season spring 2026 (Mar-May)
    leaf_season = evaluator.add_leaf(
        id="Fair_Season_Spring",
        desc="Fair occurs in spring 2026 (March-May)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The fair date is between March 1, 2026 and May 31, 2026.",
        node=leaf_season,
        sources=url,
        additional_instruction="Check event dates on the page and confirm the date falls within Spring 2026.",
    )

    # Deadline ≥ 6 weeks before event
    leaf_deadline = evaluator.add_leaf(
        id="Fair_Application_Deadline",
        desc="Application deadline allows minimum 6 weeks before event",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The craft fair application deadline is at least 6 weeks (42 days) prior to the event date.",
        node=leaf_deadline,
        sources=url,
        additional_instruction=(
            "Use the event date and deadline date from the page. If dates are explicit, calculate the difference. "
            "If no clear deadline or event date, treat as NOT SUPPORTED."
        ),
    )

    # Booth size 10x10 or specified dimensions
    leaf_booth = evaluator.add_leaf(
        id="Fair_Booth_Size",
        desc="Booth size standard 10' × 10' or clearly specified",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The fair provides standard 10' × 10' booth space or clearly specifies the booth dimensions.",
        node=leaf_booth,
        sources=url,
        additional_instruction="Look for booth size details such as '10x10' or explicit dimensions.",
    )


async def verify_insurance_component(
    evaluator: Evaluator,
    parent_node,
    ins: InsuranceExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Insurance_Component",
        desc="Vendor liability insurance requirements",
        parent=parent_node,
        critical=True,
    )

    # Coverage minimum $1M per occurrence
    leaf_cov = evaluator.add_leaf(
        id="Insurance_Coverage",
        desc="General liability minimum $1 million per occurrence",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="General liability insurance minimum coverage of $1,000,000 per occurrence is required or recommended for the event/vendor.",
        node=leaf_cov,
        sources=ins.urls,
        additional_instruction=(
            "Verify the referenced page(s) state a $1,000,000 per occurrence requirement or standard for vendors."
        ),
    )

    # Cost estimate
    leaf_cost = evaluator.add_leaf(
        id="Insurance_Cost_Est",
        desc="Cost estimate provided (monthly or per-event)",
        parent=node,
        critical=True,
    )
    cost_text = ins.cost_estimate or ""
    await evaluator.verify(
        claim=f"The insurance cost estimate '{cost_text}' is supported by the referenced page(s).",
        node=leaf_cost,
        sources=ins.urls,
        additional_instruction="Confirm that the cost estimate aligns with pricing information on the referenced page(s).",
    )


async def verify_licensing_component(
    evaluator: Evaluator,
    parent_node,
    lic: LicensingExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Licensing_Component",
        desc="State business licensing requirements",
        parent=parent_node,
        critical=True,
    )

    leaf_lic = evaluator.add_leaf(
        id="Illinois_Seller_Permit",
        desc="Illinois sellers permit or vendor license requirement identified",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Illinois requires sellers to register for sales tax collection (seller's permit/vendor license) for retail sales at events.",
        node=leaf_lic,
        sources=lic.url,
        additional_instruction=(
            "Verify on the referenced official page (prefer IL Dept. of Revenue) that sellers must register "
            "to collect sales tax (e.g., sales tax registration, IBT number, or similar)."
        ),
    )


async def verify_display_setup_component(
    evaluator: Evaluator,
    parent_node,
    ds: DisplaySetupExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Display_Setup_Component",
        desc="Vendor booth display setup specifications",
        parent=parent_node,
        critical=True,
    )

    leaf_tbl = evaluator.add_leaf(
        id="Table_Height",
        desc="Display table height within ergonomic range (32-38 inches)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The display table height specified in the answer is between 32 and 38 inches (inclusive).",
        node=leaf_tbl,
        additional_instruction="Check the answer text for the table height and confirm it lies within 32–38 inches.",
    )


async def verify_technical_specs_component(
    evaluator: Evaluator,
    parent_node,
    ts: TechnicalSpecsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Technical_Specs_Component",
        desc="3D printing technical specifications",
        parent=parent_node,
        critical=True,
    )

    # PLA nozzle temp 190–220°C
    leaf_nozzle = evaluator.add_leaf(
        id="PLA_Nozzle_Temp",
        desc="PLA nozzle temperature 190-220°C specified",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="PLA nozzle temperature range is typically 190–220°C.",
        node=leaf_nozzle,
        sources=ts.urls,
        additional_instruction="Verify that the referenced technical page(s) state or support a 190–220°C range for PLA nozzle.",
    )

    # PLA bed temp 50–60°C
    leaf_bed = evaluator.add_leaf(
        id="PLA_Bed_Temp",
        desc="PLA bed temperature 50-60°C specified",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="PLA bed temperature range is typically 50–60°C.",
        node=leaf_bed,
        sources=ts.urls,
        additional_instruction="Verify that the referenced technical page(s) state or support a 50–60°C range for PLA bed.",
    )

    # PLA material suitability
    leaf_mat = evaluator.add_leaf(
        id="Material_Type",
        desc="PLA material confirmed suitable for non-food-contact display items",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="PLA is suitable for non-food-contact display items.",
        node=leaf_mat,
        sources=ts.urls,
        additional_instruction="Verify from referenced page(s) that PLA is appropriate for non-food-contact use.",
    )


async def verify_display_items_component(
    evaluator: Evaluator,
    parent_node,
    di: DisplayItemsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Display_Items",
        desc="All four 3D printed display items have identified makerspaces meeting specifications",
        parent=parent_node,
        critical=True,
    )

    # Item 1: Primary Display Stand (9×9×5, ≤4hr, ≤75g)
    await verify_display_item(
        evaluator=evaluator,
        parent_node=node,
        item_node_id="Display_Item_1_Stand",
        item_desc='Primary display stand (9" × 9" × 5", ≤4hr, ≤75g) - makerspace identified and verified',
        item_info=di.primary_display_stand,
        required_dims_in=(9, 9, 5),
        required_hours=4,
        required_weight_g=75,
        leaf_prefix="Item1_MS",
    )

    # Item 2: Small Hanging Hooks (3×2×1, ≤2hr, ≤25g)
    await verify_display_item(
        evaluator=evaluator,
        parent_node=node,
        item_node_id="Display_Item_2_Hooks",
        item_desc='Small hanging hooks (3" × 2" × 1", ≤2hr, ≤25g) - makerspace identified and verified',
        item_info=di.small_hanging_hooks,
        required_dims_in=(3, 2, 1),
        required_hours=2,
        required_weight_g=25,
        leaf_prefix="Item2_MS",
    )

    # Item 3: Sign Holders (7×8×5, ≤8hr, ≤95g)
    await verify_display_item(
        evaluator=evaluator,
        parent_node=node,
        item_node_id="Display_Item_3_Signs",
        item_desc='Sign holders (7" × 8" × 5", ≤8hr, ≤95g) - makerspace identified and verified',
        item_info=di.sign_holders,
        required_dims_in=(7, 8, 5),
        required_hours=8,
        required_weight_g=95,
        leaf_prefix="Item3_MS",
    )

    # Item 4: Price Tag Holders (4×3×2, ≤3hr, ≤40g)
    await verify_display_item(
        evaluator=evaluator,
        parent_node=node,
        item_node_id="Display_Item_4_Price_Tags",
        item_desc='Price tag holders (4" × 3" × 2", ≤3hr, ≤40g) - makerspace identified and verified',
        item_info=di.price_tag_holders,
        required_dims_in=(4, 3, 2),
        required_hours=3,
        required_weight_g=40,
        leaf_prefix="Item4_MS",
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
    Evaluate the comprehensive Illinois craft vendor project plan answer.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level checks can be done in parallel
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

    # Ground-truth requirement summary (for reference)
    evaluator.add_ground_truth({
        "display_items_requirements": {
            "Primary Display Stand": {"dims_in": "9x9x5 inches", "time_limit": "≤4 hours", "weight": "≤75g"},
            "Small Hanging Hooks": {"dims_in": "3x2x1 inches", "time_limit": "≤2 hours", "weight": "≤25g"},
            "Sign Holders": {"dims_in": "7x8x5 inches", "time_limit": "≤8 hours", "weight": "≤95g"},
            "Price Tag Holders": {"dims_in": "4x3x2 inches", "time_limit": "≤3 hours", "weight": "≤40g"},
            "materials": "PLA", "file_format": "STL"
        },
        "cutting_board_spec_requirements": {
            "thickness_min": "≥1.25 inches", "length_range": "10-16 inches", "width_range": "8-12 inches",
            "wood_species": "food-safe hardwood", "finish": "mineral oil + beeswax (food-safe)"
        },
        "craft_fair_requirements": {
            "location": "Illinois", "season": "Spring 2026 (Mar-May)",
            "deadline": "≥6 weeks before event", "booth": "10x10 or specified dimensions"
        },
        "insurance_requirements": {"coverage": "$1,000,000 per occurrence"},
        "display_setup": {"table_height": "32-38 inches"},
        "technical_specs": {"PLA_nozzle": "190-220°C", "PLA_bed": "50-60°C", "PLA_use": "non-food-display OK"}
    })

    # Extract all components in parallel
    (
        display_items,
        cutting_board,
        craft_fair,
        insurance,
        licensing,
        display_setup,
        technical_specs,
    ) = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_display_items(),
            template_class=DisplayItemsExtraction,
            extraction_name="display_items",
        ),
        evaluator.extract(
            prompt=prompt_extract_cutting_board(),
            template_class=CuttingBoardExtraction,
            extraction_name="cutting_board",
        ),
        evaluator.extract(
            prompt=prompt_extract_craft_fair(),
            template_class=CraftFairExtraction,
            extraction_name="craft_fair",
        ),
        evaluator.extract(
            prompt=prompt_extract_insurance(),
            template_class=InsuranceExtraction,
            extraction_name="insurance",
        ),
        evaluator.extract(
            prompt=prompt_extract_licensing(),
            template_class=LicensingExtraction,
            extraction_name="licensing",
        ),
        evaluator.extract(
            prompt=prompt_extract_display_setup(),
            template_class=DisplaySetupExtraction,
            extraction_name="display_setup",
        ),
        evaluator.extract(
            prompt=prompt_extract_technical_specs(),
            template_class=TechnicalSpecsExtraction,
            extraction_name="technical_specs",
        ),
    )

    # Build root node (critical, parallel aggregation)
    # Root is already created in initialize with non-critical default; upgrade root to critical by wrapping under a new critical node?
    # Instead, we will use the existing root and treat overall passing through children nodes (all critical) to reflect rubric.
    # Since verification_tree enforces critical parent having only critical children, we ensure all direct children added here are critical.

    # Display Items Component
    await verify_display_items_component(evaluator, root, display_items)

    # Woodworking Component
    await verify_woodworking_component(evaluator, root, cutting_board)

    # Craft Fair Component
    await verify_craft_fair_component(evaluator, root, craft_fair)

    # Insurance Component
    await verify_insurance_component(evaluator, root, insurance)

    # Licensing Component
    await verify_licensing_component(evaluator, root, licensing)

    # Display Setup Component
    await verify_display_setup_component(evaluator, root, display_setup)

    # Technical Specs Component
    await verify_technical_specs_component(evaluator, root, technical_specs)

    # Return evaluation summary
    return evaluator.get_summary()