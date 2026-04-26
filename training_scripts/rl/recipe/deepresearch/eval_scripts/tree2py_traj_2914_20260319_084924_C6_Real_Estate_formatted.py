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
TASK_ID = "houston_mixed_use_codes"
TASK_DESCRIPTION = (
    "A real estate developer is planning a mixed-use commercial development project in Houston, Texas, "
    "consisting of three separate buildings: an office building, a retail space, and a warehouse facility. "
    "The developer needs to ensure that all three components meet applicable building codes, industry standards, "
    "and Houston-specific requirements.\n\n"
    "For this development project, provide detailed specifications for each of the three buildings that satisfy the following requirements:\n\n"
    "Office Building:\n"
    "- Parking ratio that meets or exceeds the industry standard of at least 4 spaces per 1,000 square feet\n"
    "- Parking stall dimensions that comply with minimum code requirements\n"
    "- Door widths that meet ADA accessibility standards\n"
    "- Accessible parking spaces as required by ADA\n"
    "- Office space allocation that falls within the standard range for employee workspace\n"
    "- Ceiling height that meets minimum code requirements\n"
    "- An approved fire alarm system that complies with Houston fire code\n"
    "- An automatic sprinkler system\n"
    "- HVAC system that meets ASHRAE 90.1 minimum energy efficiency requirements\n\n"
    "Retail Space:\n"
    "- Ceiling clear height that meets the common industry standard for retail spaces\n"
    "- Occupancy load calculation using the standard square footage per person for retail\n"
    "- An approved fire alarm system that complies with Houston fire code\n"
    "- An automatic sprinkler system\n"
    "- Door widths that meet ADA accessibility standards\n\n"
    "Warehouse Facility:\n"
    "- Loading dock doors with widths within the standard range\n"
    "- Loading dock platform height within the standard range\n"
    "- Parking ratio that meets or exceeds the industrial building standard\n"
    "- Parking stall dimensions that comply with minimum code requirements\n"
    "- An approved fire alarm system that complies with Houston fire code\n"
    "- An automatic sprinkler system\n"
    "- Door widths that meet ADA accessibility standards\n\n"
    "For each specification provided, include the specific numerical values or standards being met, along with reference URLs from the search results that support each requirement."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class OfficeExtraction(BaseModel):
    parking_ratio_per_1000_sf: Optional[str] = None
    parking_stall_length_ft: Optional[str] = None
    parking_stall_width_ft: Optional[str] = None
    parking_sources: List[str] = Field(default_factory=list)

    door_min_width_in: Optional[str] = None
    accessible_parking_description: Optional[str] = None
    ada_sources: List[str] = Field(default_factory=list)

    space_per_employee_sf: Optional[str] = None
    ceiling_height_ft: Optional[str] = None
    space_sources: List[str] = Field(default_factory=list)

    fire_alarm_spec: Optional[str] = None
    sprinkler_spec: Optional[str] = None
    fire_sources: List[str] = Field(default_factory=list)

    hvac_ashrae_901_compliance: Optional[str] = None
    hvac_sources: List[str] = Field(default_factory=list)


class RetailExtraction(BaseModel):
    retail_clear_height_ft: Optional[str] = None
    retail_height_sources: List[str] = Field(default_factory=list)

    occupancy_sf_per_person: Optional[str] = None
    occupancy_sources: List[str] = Field(default_factory=list)

    retail_fire_alarm_spec: Optional[str] = None
    retail_sprinkler_spec: Optional[str] = None
    retail_fire_sources: List[str] = Field(default_factory=list)

    retail_door_min_width_in: Optional[str] = None
    retail_ada_sources: List[str] = Field(default_factory=list)


class WarehouseExtraction(BaseModel):
    dock_door_width_ft: Optional[str] = None
    dock_platform_height_in: Optional[str] = None
    dock_sources: List[str] = Field(default_factory=list)

    parking_ratio_per_1000_sf: Optional[str] = None
    parking_stall_length_ft: Optional[str] = None
    parking_stall_width_ft: Optional[str] = None
    parking_sources: List[str] = Field(default_factory=list)

    fire_alarm_spec: Optional[str] = None
    sprinkler_spec: Optional[str] = None
    fire_sources: List[str] = Field(default_factory=list)

    door_min_width_in: Optional[str] = None
    ada_sources: List[str] = Field(default_factory=list)


class MixedUseSpecs(BaseModel):
    office: Optional[OfficeExtraction] = None
    retail: Optional[RetailExtraction] = None
    warehouse: Optional[WarehouseExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_specs() -> str:
    return """
Extract structured specifications exactly as stated in the answer for three buildings (office, retail, warehouse). 
Do NOT infer or invent values. Return null for any missing scalar field and [] for any missing URL list.

For each building, extract the following fields:

office:
- parking_ratio_per_1000_sf: string (e.g., "4 per 1,000 sf", "5/1000 sf", "4.5")
- parking_stall_length_ft: string (feet, may include units; keep as given)
- parking_stall_width_ft: string (feet, may include units; keep as given)
- parking_sources: array of URLs cited for office parking standards (ratio and/or stall dimensions)

- door_min_width_in: string (inches; clear width as given)
- accessible_parking_description: string description indicating accessible spaces are provided (if present)
- ada_sources: array of URLs cited for ADA requirements (door width and/or accessible parking)

- space_per_employee_sf: string (e.g., "175 sf/employee", "150-250 sf")
- ceiling_height_ft: string (feet; keep as given)
- space_sources: array of URLs cited for office space and/or ceiling height standards

- fire_alarm_spec: string indicating presence of an approved fire alarm system
- sprinkler_spec: string indicating presence of an automatic sprinkler system
- fire_sources: array of URLs cited for Houston fire code requirements

- hvac_ashrae_901_compliance: string indicating compliance with ASHRAE 90.1 minimum energy efficiency
- hvac_sources: array of URLs cited for ASHRAE 90.1 standards

retail:
- retail_clear_height_ft: string (feet; clear height as given)
- retail_height_sources: array of URLs supporting retail ceiling/clear height standards
- occupancy_sf_per_person: string for occupant load factor used (e.g., "60 sf/person")
- occupancy_sources: array of URLs supporting retail occupant load factor
- retail_fire_alarm_spec: string indicating presence of an approved fire alarm system
- retail_sprinkler_spec: string indicating presence of an automatic sprinkler system
- retail_fire_sources: array of URLs for Houston fire code requirements (retail)
- retail_door_min_width_in: string (inches; ADA door width for retail)
- retail_ada_sources: array of URLs supporting ADA requirements for doors

warehouse:
- dock_door_width_ft: string (feet; e.g., "8.5 ft", "9'")
- dock_platform_height_in: string (inches; e.g., "46 in")
- dock_sources: array of URLs for loading dock standards
- parking_ratio_per_1000_sf: string for industrial parking ratio used (e.g., "2 per 1,000 sf")
- parking_stall_length_ft: string (feet)
- parking_stall_width_ft: string (feet)
- parking_sources: array of URLs for industrial parking standards
- fire_alarm_spec: string indicating presence of an approved fire alarm system
- sprinkler_spec: string indicating presence of an automatic sprinkler system
- fire_sources: array of URLs for Houston fire code requirements (warehouse)
- door_min_width_in: string (inches; ADA door width for warehouse)
- ada_sources: array of URLs supporting ADA requirements for doors

Rules:
- Only extract URLs explicitly present in the answer (plain or markdown). If none for a field, return [].
- Preserve units and text as given; do not normalize.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def s(value: Optional[str]) -> str:
    return value if value is not None else ""


def has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def urls_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    return urls if has_urls(urls) else None


# --------------------------------------------------------------------------- #
# Verification builders: create nodes and collect verifications               #
# Each verification is a tuple: (claim, sources, node, additional_instruction)
# --------------------------------------------------------------------------- #
def build_office_verifications(evaluator: Evaluator, parent_node, office: Optional[OfficeExtraction]) -> List[Tuple[str, Any, Any, Optional[str]]]:
    verifs: List[Tuple[str, Any, Any, Optional[str]]] = []

    office_node = evaluator.add_parallel(
        id="Office_Building_Component",
        desc="Office building specifications and compliance requirements",
        parent=parent_node,
        critical=True
    )

    # -------- Office Parking Requirements --------
    parking_node = evaluator.add_sequential(
        id="Office_Parking_Requirements",
        desc="Parking specifications for the office building",
        parent=office_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Office_Parking_Ratio",
        desc="Parking ratio meets or exceeds the standard of 4 spaces per 1,000 square feet of office space",
        parent=parking_node,
        critical=True
    )
    pratio = s(office.parking_ratio_per_1000_sf) if office else ""
    verifs.append((
        f"The proposed office building parking ratio is at least 4 parking spaces per 1,000 square feet. "
        f"The answer's stated ratio is: '{pratio}'. Consider numeric equivalence (e.g., 4/1000 sf equals 4 per 1,000 sf).",
        None,
        leaf,
        "Decide True if the provided ratio equals or exceeds 4 per 1,000 square feet based solely on the given text value; allow minor formatting variations."
    ))

    leaf = evaluator.add_leaf(
        id="Office_Parking_Stall_Dimensions",
        desc="Each parking stall meets minimum dimensions of 18 feet long by 8.5 feet wide",
        parent=parking_node,
        critical=True
    )
    plen = s(office.parking_stall_length_ft) if office else ""
    pwid = s(office.parking_stall_width_ft) if office else ""
    verifs.append((
        "The proposed standard office parking stall dimensions meet or exceed 18 feet in length and 8.5 feet in width. "
        f"The answer's stall dimensions are: length='{plen}', width='{pwid}'.",
        None,
        leaf,
        "Treat 9 ft width as >= 8.5 ft; accept equivalent imperial/metric values; small rounding acceptable."
    ))

    # Reference leaf: if no sources, mark failed immediately
    if office and has_urls(office.parking_sources):
        leaf = evaluator.add_leaf(
            id="Office_Parking_Reference",
            desc="Provide reference URL supporting parking requirements",
            parent=parking_node,
            critical=True
        )
        verifs.append((
            "The cited source(s) support office parking standards (e.g., typical ratio near 4 spaces per 1,000 sf and/or minimum stall size around 8.5 ft by 18 ft). "
            "You only need to check that the source discusses such standards; you do not need to confirm this project's own dimensions.",
            office.parking_sources,
            leaf,
            "Focus on whether the source text explicitly states or clearly implies those standards; allow reasonable phrasing variations."
        ))
    else:
        evaluator.add_leaf(
            id="Office_Parking_Reference",
            desc="Provide reference URL supporting parking requirements",
            parent=parking_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # -------- Office ADA Compliance --------
    ada_node = evaluator.add_sequential(
        id="Office_ADA_Compliance",
        desc="ADA accessibility requirements for the office building",
        parent=office_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Office_Door_Width",
        desc="All office doors meet the minimum width requirement of 32 inches for ADA compliance",
        parent=ada_node,
        critical=True
    )
    dwidth = s(office.door_min_width_in) if office else ""
    verifs.append((
        "The proposed minimum clear width for office doors is at least 32 inches. "
        f"The answer's stated door width is: '{dwidth}'.",
        None,
        leaf,
        "Accept values >= 32 inches; consider minor variants (e.g., 2'-8\" equals 32 inches)."
    ))

    leaf = evaluator.add_leaf(
        id="Office_Accessible_Parking",
        desc="Accessible parking spaces are provided as required by ADA standards",
        parent=ada_node,
        critical=True
    )
    acc_desc = s(office.accessible_parking_description) if office else ""
    verifs.append((
        "The answer explicitly includes accessible parking spaces as required by ADA.",
        None,
        leaf,
        f"Use the provided text snippet for evidence: '{acc_desc}'. Consider True only if presence is clear."
    ))

    if office and has_urls(office.ada_sources):
        leaf = evaluator.add_leaf(
            id="Office_ADA_Reference",
            desc="Provide reference URL supporting ADA requirements",
            parent=ada_node,
            critical=True
        )
        verifs.append((
            "The cited source(s) support ADA requirements relevant here, including that minimum door clear width is 32 inches and that accessible parking spaces are required with defined criteria.",
            office.ada_sources,
            leaf,
            "Verify the source discusses at least the 32-inch door clear width and/or accessible parking requirements."
        ))
    else:
        evaluator.add_leaf(
            id="Office_ADA_Reference",
            desc="Provide reference URL supporting ADA requirements",
            parent=ada_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # -------- Office Space Standards --------
    space_node = evaluator.add_sequential(
        id="Office_Space_Standards",
        desc="Office space allocation and layout standards",
        parent=office_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Office_Space_Per_Employee",
        desc="Office space allocation falls within the standard range of 150-250 square feet per employee",
        parent=space_node,
        critical=True
    )
    spe = s(office.space_per_employee_sf) if office else ""
    verifs.append((
        "The office space allocation per employee falls within 150 to 250 square feet. "
        f"The answer's stated allocation is: '{spe}'.",
        None,
        leaf,
        "Consider ranges that overlap 150-250 as acceptable; allow minor rounding."
    ))

    leaf = evaluator.add_leaf(
        id="Office_Ceiling_Height",
        desc="Office ceiling height meets the minimum requirement of 7.5 feet",
        parent=space_node,
        critical=True
    )
    ceil = s(office.ceiling_height_ft) if office else ""
    verifs.append((
        "The office ceiling height is at least 7.5 feet. "
        f"The answer's stated ceiling height is: '{ceil}'.",
        None,
        leaf,
        "Accept values >= 7.5 ft; allow small rounding."
    ))

    if office and has_urls(office.space_sources):
        leaf = evaluator.add_leaf(
            id="Office_Space_Reference",
            desc="Provide reference URL supporting office space standards",
            parent=space_node,
            critical=True
        )
        verifs.append((
            "The cited source(s) support typical office planning standards, such as space per employee near 150–250 sf and/or minimum office ceiling height around 7.5 ft.",
            office.space_sources,
            leaf,
            "You only need to confirm the standard(s) are stated; do not verify this project's own value."
        ))
    else:
        evaluator.add_leaf(
            id="Office_Space_Reference",
            desc="Provide reference URL supporting office space standards",
            parent=space_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # -------- Office Fire Safety --------
    fire_node = evaluator.add_sequential(
        id="Office_Fire_Safety",
        desc="Fire safety systems for the office building",
        parent=office_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Office_Fire_Alarm_System",
        desc="Office building includes an approved fire alarm system as required by Houston fire code",
        parent=fire_node,
        critical=True
    )
    fas = s(office.fire_alarm_spec) if office else ""
    verifs.append((
        "The answer specifies an approved fire alarm system for the office building.",
        None,
        leaf,
        f"Use the provided text snippet for evidence: '{fas}'."
    ))

    leaf = evaluator.add_leaf(
        id="Office_Sprinkler_System",
        desc="Office building includes an automatic sprinkler system",
        parent=fire_node,
        critical=True
    )
    sps = s(office.sprinkler_spec) if office else ""
    verifs.append((
        "The answer specifies an automatic sprinkler system for the office building.",
        None,
        leaf,
        f"Use the provided text snippet for evidence: '{sps}'."
    ))

    if office and has_urls(office.fire_sources):
        leaf = evaluator.add_leaf(
            id="Office_Fire_Reference",
            desc="Provide reference URL supporting Houston fire code requirements",
            parent=fire_node,
            critical=True
        )
        verifs.append((
            "The cited source(s) support Houston Fire Code requirements for fire alarm and/or automatic sprinkler systems in relevant occupancies.",
            office.fire_sources,
            leaf,
            "Confirm that the source references Houston Fire Code or City of Houston Fire Marshal guidance on alarm/sprinkler requirements."
        ))
    else:
        evaluator.add_leaf(
            id="Office_Fire_Reference",
            desc="Provide reference URL supporting Houston fire code requirements",
            parent=fire_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # -------- Office HVAC Energy (ASHRAE 90.1) --------
    hvac_node = evaluator.add_sequential(
        id="Office_HVAC_Energy",
        desc="HVAC system energy efficiency requirements",
        parent=office_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Office_ASHRAE_Compliance",
        desc="HVAC system meets ASHRAE 90.1 minimum energy efficiency requirements",
        parent=hvac_node,
        critical=True
    )
    ash = s(office.hvac_ashrae_901_compliance) if office else ""
    verifs.append((
        "The answer specifies that the HVAC system meets ASHRAE 90.1 minimum energy efficiency requirements.",
        None,
        leaf,
        f"Use the provided text snippet for evidence: '{ash}'."
    ))

    if office and has_urls(office.hvac_sources):
        leaf = evaluator.add_leaf(
            id="Office_HVAC_Reference",
            desc="Provide reference URL supporting ASHRAE 90.1 standards",
            parent=hvac_node,
            critical=True
        )
        verifs.append((
            "The cited source(s) support that ASHRAE Standard 90.1 defines minimum energy efficiency requirements for HVAC systems.",
            office.hvac_sources,
            leaf,
            "Confirm the page discusses ASHRAE 90.1 scope and HVAC efficiency requirements."
        ))
    else:
        evaluator.add_leaf(
            id="Office_HVAC_Reference",
            desc="Provide reference URL supporting ASHRAE 90.1 standards",
            parent=hvac_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    return verifs


def build_retail_verifications(evaluator: Evaluator, parent_node, retail: Optional[RetailExtraction]) -> List[Tuple[str, Any, Any, Optional[str]]]:
    verifs: List[Tuple[str, Any, Any, Optional[str]]] = []

    retail_node = evaluator.add_parallel(
        id="Retail_Building_Component",
        desc="Retail space specifications and compliance requirements",
        parent=parent_node,
        critical=True
    )

    # -------- Retail Ceiling Height --------
    ceil_node = evaluator.add_sequential(
        id="Retail_Ceiling_Height",
        desc="Retail space ceiling height requirements",
        parent=retail_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Retail_Clear_Height",
        desc="Retail space meets the industry standard minimum clear height of 16 feet",
        parent=ceil_node,
        critical=True
    )
    rch = s(retail.retail_clear_height_ft) if retail else ""
    verifs.append((
        "The retail space clear ceiling height is at least 16 feet. "
        f"The answer's stated retail clear height is: '{rch}'.",
        None,
        leaf,
        "Accept values >= 16 ft; allow small rounding."
    ))

    if retail and has_urls(retail.retail_height_sources):
        leaf = evaluator.add_leaf(
            id="Retail_Height_Reference",
            desc="Provide reference URL supporting retail ceiling height standards",
            parent=ceil_node,
            critical=True
        )
        verifs.append((
            "The cited source(s) support common retail industry clear height standards near or above 16 feet.",
            retail.retail_height_sources,
            leaf,
            "Confirm the page discusses typical retail clear heights (e.g., 16 ft or more)."
        ))
    else:
        evaluator.add_leaf(
            id="Retail_Height_Reference",
            desc="Provide reference URL supporting retail ceiling height standards",
            parent=ceil_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # -------- Retail Occupancy Load --------
    occ_node = evaluator.add_sequential(
        id="Retail_Occupancy_Load",
        desc="Occupancy load calculation for retail space",
        parent=retail_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Retail_Occupancy_Calculation",
        desc="Occupancy load is calculated using the standard of 60 square feet per person for retail spaces",
        parent=occ_node,
        critical=True
    )
    ofac = s(retail.occupancy_sf_per_person) if retail else ""
    verifs.append((
        "The occupancy load factor used for the retail space is 60 square feet per person. "
        f"The answer's stated factor is: '{ofac}'.",
        None,
        leaf,
        "Accept equivalence like '60 sf/person' or '60 sq ft per person'."
    ))

    if retail and has_urls(retail.occupancy_sources):
        leaf = evaluator.add_leaf(
            id="Retail_Occupancy_Reference",
            desc="Provide reference URL supporting occupancy load requirements",
            parent=occ_node,
            critical=True
        )
        verifs.append((
            "The cited source(s) support an occupant load factor for retail/mercantile areas of about 60 square feet per person (gross), or similar recognized code factor.",
            retail.occupancy_sources,
            leaf,
            "Confirm IBC or equivalent code table indicating mercantile/retail occupant load factor ≈ 60 sf/person (gross) or a clearly stated equivalent."
        ))
    else:
        evaluator.add_leaf(
            id="Retail_Occupancy_Reference",
            desc="Provide reference URL supporting occupancy load requirements",
            parent=occ_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # -------- Retail Fire Safety --------
    rfire_node = evaluator.add_sequential(
        id="Retail_Fire_Safety",
        desc="Fire safety systems for the retail space",
        parent=retail_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Retail_Fire_Alarm_System",
        desc="Retail space includes an approved fire alarm system as required by Houston fire code",
        parent=rfire_node,
        critical=True
    )
    rfas = s(retail.retail_fire_alarm_spec) if retail else ""
    verifs.append((
        "The answer specifies an approved fire alarm system for the retail space.",
        None,
        leaf,
        f"Use the provided text snippet for evidence: '{rfas}'."
    ))

    leaf = evaluator.add_leaf(
        id="Retail_Sprinkler_System",
        desc="Retail space includes an automatic sprinkler system",
        parent=rfire_node,
        critical=True
    )
    rspr = s(retail.retail_sprinkler_spec) if retail else ""
    verifs.append((
        "The answer specifies an automatic sprinkler system for the retail space.",
        None,
        leaf,
        f"Use the provided text snippet for evidence: '{rspr}'."
    ))

    if retail and has_urls(retail.retail_fire_sources):
        leaf = evaluator.add_leaf(
            id="Retail_Fire_Reference",
            desc="Provide reference URL supporting Houston fire code requirements",
            parent=rfire_node,
            critical=True
        )
        verifs.append((
            "The cited source(s) support Houston Fire Code requirements for alarm and/or sprinkler systems in retail occupancies.",
            retail.retail_fire_sources,
            leaf,
            "Confirm the page references City of Houston fire code or official guidance."
        ))
    else:
        evaluator.add_leaf(
            id="Retail_Fire_Reference",
            desc="Provide reference URL supporting Houston fire code requirements",
            parent=rfire_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # -------- Retail ADA Compliance (Door width) --------
    rada_node = evaluator.add_sequential(
        id="Retail_ADA_Compliance",
        desc="ADA accessibility requirements for the retail space",
        parent=retail_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Retail_Door_Width",
        desc="All retail doors meet the minimum width requirement of 32 inches for ADA compliance",
        parent=rada_node,
        critical=True
    )
    rdw = s(retail.retail_door_min_width_in) if retail else ""
    verifs.append((
        "The proposed minimum clear width for retail doors is at least 32 inches. "
        f"The answer's stated door width is: '{rdw}'.",
        None,
        leaf,
        "Accept values >= 32 inches; consider minor variants like 2'-8\"."
    ))

    if retail and has_urls(retail.retail_ada_sources):
        leaf = evaluator.add_leaf(
            id="Retail_ADA_Reference",
            desc="Provide reference URL supporting ADA requirements",
            parent=rada_node,
            critical=True
        )
        verifs.append((
            "The cited source(s) support ADA minimum door clear width of 32 inches or equivalent accessibility criteria.",
            retail.retail_ada_sources,
            leaf,
            "Confirm the page discusses the 32-inch minimum door clear width."
        ))
    else:
        evaluator.add_leaf(
            id="Retail_ADA_Reference",
            desc="Provide reference URL supporting ADA requirements",
            parent=rada_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    return verifs


def build_warehouse_verifications(evaluator: Evaluator, parent_node, wh: Optional[WarehouseExtraction]) -> List[Tuple[str, Any, Any, Optional[str]]]:
    verifs: List[Tuple[str, Any, Any, Optional[str]]] = []

    wh_node = evaluator.add_parallel(
        id="Warehouse_Building_Component",
        desc="Warehouse facility specifications and compliance requirements",
        parent=parent_node,
        critical=True
    )

    # -------- Loading Dock --------
    dock_node = evaluator.add_sequential(
        id="Warehouse_Loading_Dock",
        desc="Loading dock design specifications for the warehouse",
        parent=wh_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Warehouse_Dock_Door_Width",
        desc="Loading dock doors are between 8.5 and 9 feet wide",
        parent=dock_node,
        critical=True
    )
    ddw = s(wh.dock_door_width_ft) if wh else ""
    verifs.append((
        "The loading dock door width falls within 8.5 to 9.0 feet. "
        f"The answer's stated dock door width is: '{ddw}'.",
        None,
        leaf,
        "Accept 8'-6\" to 9'-0\" (or metric equivalents) as compliant; small rounding acceptable."
    ))

    leaf = evaluator.add_leaf(
        id="Warehouse_Dock_Platform_Height",
        desc="Loading dock platform height is between 44 and 48 inches",
        parent=dock_node,
        critical=True
    )
    dph = s(wh.dock_platform_height_in) if wh else ""
    verifs.append((
        "The loading dock platform height falls within 44 to 48 inches. "
        f"The answer's stated platform height is: '{dph}'.",
        None,
        leaf,
        "Accept typical 45–48 in; small rounding acceptable."
    ))

    if wh and has_urls(wh.dock_sources):
        leaf = evaluator.add_leaf(
            id="Warehouse_Dock_Reference",
            desc="Provide reference URL supporting loading dock design standards",
            parent=dock_node,
            critical=True
        )
        verifs.append((
            "The cited source(s) support typical loading dock design standards, including door widths near 8.5–9 ft and platform heights near 44–48 in.",
            wh.dock_sources,
            leaf,
            "Confirm the page discusses either or both of those typical dimensions."
        ))
    else:
        evaluator.add_leaf(
            id="Warehouse_Dock_Reference",
            desc="Provide reference URL supporting loading dock design standards",
            parent=dock_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # -------- Warehouse Parking --------
    wpark_node = evaluator.add_sequential(
        id="Warehouse_Parking_Requirements",
        desc="Parking specifications for the warehouse facility",
        parent=wh_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Warehouse_Parking_Ratio",
        desc="Parking ratio meets or exceeds the industrial standard of 2-3 spaces per 1,000 square feet",
        parent=wpark_node,
        critical=True
    )
    wpr = s(wh.parking_ratio_per_1000_sf) if wh else ""
    verifs.append((
        "The warehouse parking ratio meets or exceeds 2 spaces per 1,000 square feet (typical industrial standard often cited as 2–3/1000 sf). "
        f"The answer's stated ratio is: '{wpr}'.",
        None,
        leaf,
        "Decide True if the provided ratio is >= 2 per 1,000 sf; allow minor formatting variations."
    ))

    leaf = evaluator.add_leaf(
        id="Warehouse_Parking_Stall_Dimensions",
        desc="Each parking stall meets minimum dimensions of 18 feet long by 8.5 feet wide",
        parent=wpark_node,
        critical=True
    )
    wpl = s(wh.parking_stall_length_ft) if wh else ""
    wpw = s(wh.parking_stall_width_ft) if wh else ""
    verifs.append((
        "The proposed warehouse parking stall dimensions meet or exceed 18 feet in length and 8.5 feet in width. "
        f"The answer's stall dimensions are: length='{wpl}', width='{wpw}'.",
        None,
        leaf,
        "Treat 9 ft width as >= 8.5 ft; accept metric equivalents; small rounding acceptable."
    ))

    if wh and has_urls(wh.parking_sources):
        leaf = evaluator.add_leaf(
            id="Warehouse_Parking_Reference",
            desc="Provide reference URL supporting parking requirements",
            parent=wpark_node,
            critical=True
        )
        verifs.append((
            "The cited source(s) support industrial parking standards, including ratios near 2–3 spaces per 1,000 sf and/or typical stall dimensions near 8.5 ft by 18 ft.",
            wh.parking_sources,
            leaf,
            "Confirm the page discusses such industrial parking norms; do not verify the project's own value."
        ))
    else:
        evaluator.add_leaf(
            id="Warehouse_Parking_Reference",
            desc="Provide reference URL supporting parking requirements",
            parent=wpark_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # -------- Warehouse Fire Safety --------
    wfire_node = evaluator.add_sequential(
        id="Warehouse_Fire_Safety",
        desc="Fire safety systems for the warehouse facility",
        parent=wh_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Warehouse_Fire_Alarm_System",
        desc="Warehouse includes an approved fire alarm system as required by Houston fire code",
        parent=wfire_node,
        critical=True
    )
    wfas = s(wh.fire_alarm_spec) if wh else ""
    verifs.append((
        "The answer specifies an approved fire alarm system for the warehouse.",
        None,
        leaf,
        f"Use the provided text snippet for evidence: '{wfas}'."
    ))

    leaf = evaluator.add_leaf(
        id="Warehouse_Sprinkler_System",
        desc="Warehouse includes an automatic sprinkler system",
        parent=wfire_node,
        critical=True
    )
    wspr = s(wh.sprinkler_spec) if wh else ""
    verifs.append((
        "The answer specifies an automatic sprinkler system for the warehouse.",
        None,
        leaf,
        f"Use the provided text snippet for evidence: '{wspr}'."
    ))

    if wh and has_urls(wh.fire_sources):
        leaf = evaluator.add_leaf(
            id="Warehouse_Fire_Reference",
            desc="Provide reference URL supporting Houston fire code requirements",
            parent=wfire_node,
            critical=True
        )
        verifs.append((
            "The cited source(s) support Houston Fire Code requirements for alarm and/or sprinkler systems in warehouse/industrial occupancies.",
            wh.fire_sources,
            leaf,
            "Confirm the page references City of Houston fire code or official guidance."
        ))
    else:
        evaluator.add_leaf(
            id="Warehouse_Fire_Reference",
            desc="Provide reference URL supporting Houston fire code requirements",
            parent=wfire_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # -------- Warehouse ADA Compliance (Door width) --------
    wada_node = evaluator.add_sequential(
        id="Warehouse_ADA_Compliance",
        desc="ADA accessibility requirements for the warehouse facility",
        parent=wh_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Warehouse_Door_Width",
        desc="All warehouse doors meet the minimum width requirement of 32 inches for ADA compliance",
        parent=wada_node,
        critical=True
    )
    wdw = s(wh.door_min_width_in) if wh else ""
    verifs.append((
        "The proposed minimum clear width for warehouse doors is at least 32 inches. "
        f"The answer's stated door width is: '{wdw}'.",
        None,
        leaf,
        "Accept values >= 32 inches; consider minor variants like 2'-8\"."
    ))

    if wh and has_urls(wh.ada_sources):
        leaf = evaluator.add_leaf(
            id="Warehouse_ADA_Reference",
            desc="Provide reference URL supporting ADA requirements",
            parent=wada_node,
            critical=True
        )
        verifs.append((
            "The cited source(s) support ADA minimum door clear width of 32 inches or equivalent accessibility criteria.",
            wh.ada_sources,
            leaf,
            "Confirm the page discusses the 32-inch minimum door clear width."
        ))
    else:
        evaluator.add_leaf(
            id="Warehouse_ADA_Reference",
            desc="Provide reference URL supporting ADA requirements",
            parent=wada_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    return verifs


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

    # Wrap rubric root as a top-level critical node under framework root
    project_root = evaluator.add_parallel(
        id="Mixed_Use_Development_Project",
        desc="Evaluate a proposed mixed-use commercial development in Houston, Texas, consisting of an office building, a retail space, and a warehouse facility, ensuring all components meet applicable building codes, zoning requirements, and industry standards",
        parent=root,
        critical=True
    )

    # Extract structured specs
    specs = await evaluator.extract(
        prompt=prompt_extract_specs(),
        template_class=MixedUseSpecs,
        extraction_name="mixed_use_specs"
    )

    # Build nodes and collect verifications
    all_verifs: List[Tuple[str, Any, Any, Optional[str]]] = []
    all_verifs.extend(build_office_verifications(evaluator, project_root, specs.office))
    all_verifs.extend(build_retail_verifications(evaluator, project_root, specs.retail))
    all_verifs.extend(build_warehouse_verifications(evaluator, project_root, specs.warehouse))

    # Run all verifications in parallel to avoid cross-skip due to critical sibling prerequisites
    if all_verifs:
        await evaluator.batch_verify(all_verifs, majority_vote=True, num_trials=3)

    return evaluator.get_summary()