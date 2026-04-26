import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "seattle_class_a_office_investment"
TASK_DESCRIPTION = """
Identify a Class A office building in downtown Seattle that meets the following institutional investment criteria: 
(1) Building must be at least 50,000 square feet in total size; 
(2) Building must have LEED Gold certification (60-79 points) or higher; 
(3) Building must have Energy Star certification with a score of 75 or higher; 
(4) Building must be located within 1/2 mile walking distance of a major transit stop (light rail or major bus line); 
(5) Building must provide parking at a ratio of at least 4 spaces per 1,000 square feet of office space; 
(6) Building must have an automatic fire sprinkler system and meet ADA accessibility standards; 
(7) Building must operate under triple net (NNN) lease arrangements where tenants pay property taxes, insurance, and common area maintenance; 
(8) Building must have at least one major corporate tenant (Fortune 1000 company or equivalent); 
(9) Major tenants must have lease terms of at least 5 years; 
(10) Building must maintain a minimum occupancy rate of 90%. 
For your identified building, provide the building name, address, and verification sources (URLs) confirming each of these requirements.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ClassAInfo(BaseModel):
    statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class SizeInfo(BaseModel):
    sqft: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class LocationInfo(BaseModel):
    statement: Optional[str] = None  # e.g., "Downtown Seattle", "CBD", "Denny Triangle"
    urls: List[str] = Field(default_factory=list)


class LEEDInfo(BaseModel):
    level: Optional[str] = None  # e.g., "LEED Gold", "LEED Platinum"
    urls: List[str] = Field(default_factory=list)


class EnergyStarInfo(BaseModel):
    status: Optional[str] = None  # e.g., "Energy Star certified"
    score: Optional[str] = None   # e.g., "78"
    urls: List[str] = Field(default_factory=list)


class TransitInfo(BaseModel):
    description: Optional[str] = None  # e.g., "0.3 miles to Westlake Station"
    urls: List[str] = Field(default_factory=list)


class ParkingInfo(BaseModel):
    ratio: Optional[str] = None  # e.g., "4.2/1000", "4 per 1,000 sf"
    urls: List[str] = Field(default_factory=list)


class SafetyInfo(BaseModel):
    statement: Optional[str] = None  # Used for sprinkler and ADA
    urls: List[str] = Field(default_factory=list)


class LeaseStructureInfo(BaseModel):
    lease_type: Optional[str] = None  # e.g., "NNN", "triple net"
    urls: List[str] = Field(default_factory=list)


class TenantInfo(BaseModel):
    tenant_name: Optional[str] = None  # e.g., "Amazon", "Boeing"
    urls: List[str] = Field(default_factory=list)


class LeaseTermInfo(BaseModel):
    term_years: Optional[str] = None  # e.g., "7-year lease"
    urls: List[str] = Field(default_factory=list)


class OccupancyInfo(BaseModel):
    rate: Optional[str] = None  # e.g., "92%"
    urls: List[str] = Field(default_factory=list)


class BuildingExtraction(BaseModel):
    building_name: Optional[str] = None
    address: Optional[str] = None

    class_a: Optional[ClassAInfo] = None
    size: Optional[SizeInfo] = None
    location: Optional[LocationInfo] = None

    leed: Optional[LEEDInfo] = None
    energy_star: Optional[EnergyStarInfo] = None

    transit: Optional[TransitInfo] = None
    parking: Optional[ParkingInfo] = None
    sprinkler: Optional[SafetyInfo] = None
    ada: Optional[SafetyInfo] = None

    lease_structure: Optional[LeaseStructureInfo] = None
    major_tenant: Optional[TenantInfo] = None
    lease_term: Optional[LeaseTermInfo] = None
    occupancy: Optional[OccupancyInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_building() -> str:
    return """
Extract the single primary office building proposed in the answer and the evidence the answer cites for each investment criterion. 
If multiple buildings are mentioned, select the first one that appears to be the recommended candidate.

Return a JSON with the following fields (use null for any missing value; for URLs, include only URLs explicitly present in the answer):

- building_name: The building name
- address: The full street address as stated

- class_a:
  - statement: The text snippet or phrase used in the answer indicating Class A status
  - urls: A list of URLs cited to support Class A designation

- size:
  - sqft: The total building size (e.g., "520,000 SF", "52,000 sq ft", etc.) as stated in the answer
  - urls: URLs cited to support the size claim

- location:
  - statement: The area/description indicating it is in downtown Seattle (e.g., "CBD", "Downtown", "Denny Triangle", "Belltown", "Pioneer Square", "Financial District", "Retail Core", "West Edge")
  - urls: URLs cited to support the downtown location

- leed:
  - level: The LEED certification level (e.g., "LEED Gold", "LEED Platinum") as stated
  - urls: URLs cited to support LEED certification

- energy_star:
  - status: Energy Star certification status if provided
  - score: The Energy Star score as stated (e.g., "79")
  - urls: URLs cited to support Energy Star status/score

- transit:
  - description: The proximity statement used in the answer (e.g., "0.3 miles to Westlake Station", "5–10 min walk to light rail")
  - urls: URLs cited to support transit proximity

- parking:
  - ratio: The provided parking ratio (e.g., "4.0 spaces per 1,000 SF", "4/1000")
  - urls: URLs cited to support parking capacity/ratio

- sprinkler:
  - statement: The statement indicating an automatic fire sprinkler system throughout
  - urls: URLs cited to support fire sprinkler system

- ada:
  - statement: The statement indicating ADA accessibility/compliance
  - urls: URLs cited to support ADA compliance

- lease_structure:
  - lease_type: The lease structure as stated (e.g., "NNN" or "triple net")
  - urls: URLs cited to support lease structure

- major_tenant:
  - tenant_name: The major corporate tenant name (Fortune 1000 or equivalent) as cited
  - urls: URLs cited to confirm the tenant presence

- lease_term:
  - term_years: The stated term length for the major tenant or tenants (e.g., "5-year lease", "10-year lease")
  - urls: URLs cited to confirm lease terms

- occupancy:
  - rate: The current occupancy rate as stated (e.g., "90%", "95 percent")
  - urls: URLs cited to confirm occupancy rate

IMPORTANT:
- Extract only URLs explicitly present in the answer (plain URLs or in markdown link format).
- Do not invent URLs; if none are provided for a criterion, return an empty list for that criterion's urls.
- Keep numbers and units as free-form strings; do not normalize (e.g., "4/1000" is fine).
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def _format_bldg_ref(data: BuildingExtraction) -> str:
    parts = []
    if data and data.building_name:
        parts.append(f"{data.building_name}")
    if data and data.address:
        parts.append(f"({data.address})")
    return " ".join(parts).strip() or "the building"


def _add_url_ref_check(
    evaluator: Evaluator,
    parent: VerificationNode,
    node_id: str,
    desc: str,
    urls: Optional[List[str]],
    critical: bool = True
) -> VerificationNode:
    exists = len(_non_empty_urls(urls)) > 0
    return evaluator.add_custom_node(
        result=exists,
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )


async def _verify_with_sources(
    evaluator: Evaluator,
    parent: VerificationNode,
    node_id: str,
    desc: str,
    claim: str,
    urls: Optional[List[str]],
    additional_instruction: str,
    critical: bool = True,
    extra_prereq: Optional[List[VerificationNode]] = None
) -> VerificationNode:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_non_empty_urls(urls),
        additional_instruction=additional_instruction,
        extra_prerequisites=extra_prereq
    )
    return leaf


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_classification_quality(
    evaluator: Evaluator,
    parent: VerificationNode,
    data: BuildingExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Building_Classification_Quality",
        desc="Verification that the building qualifies as Class A with appropriate quality standards",
        parent=parent,
        critical=True
    )

    bref = _format_bldg_ref(data)

    # Class A URL existence (gate)
    class_a_urls = data.class_a.urls if data and data.class_a else []
    class_a_url_node = _add_url_ref_check(
        evaluator, node, "Class_A_URL_Reference",
        "Provides URL reference confirming Class A designation",
        class_a_urls, critical=True
    )
    # Class A designation
    await _verify_with_sources(
        evaluator, node, "Class_A_Designation",
        "Building is officially designated or recognized as Class A office space",
        claim=f"{bref} is recognized or marketed as a Class A office building in Seattle.",
        urls=class_a_urls,
        additional_instruction=(
            "Check whether the cited page(s) explicitly describe the property as 'Class A' "
            "or equivalent phrasing (e.g., 'Class A office tower')."
        ),
        critical=True,
        extra_prereq=[class_a_url_node]
    )

    # Size URL existence (gate)
    size_urls = data.size.urls if data and data.size else []
    size_url_node = _add_url_ref_check(
        evaluator, node, "Size_URL_Reference",
        "Provides URL reference confirming building size",
        size_urls, critical=True
    )
    # Building size >= 50,000 SF
    size_str = data.size.sqft if data and data.size and data.size.sqft else ""
    await _verify_with_sources(
        evaluator, node, "Building_Size_Requirement",
        "Building size meets minimum 50,000 square feet threshold",
        claim=f"The total size of {bref} is {size_str} and at least 50,000 square feet.",
        urls=size_urls,
        additional_instruction=(
            "Confirm the building's total rentable or gross area is ≥ 50,000 square feet. "
            "If the page states a figure (e.g., 200,000 SF), that satisfies the condition."
        ),
        critical=True,
        extra_prereq=[size_url_node]
    )

    # Downtown location URL existence (gate)
    loc_urls = data.location.urls if data and data.location else []
    loc_url_node = _add_url_ref_check(
        evaluator, node, "Location_URL_Reference",
        "Provides URL reference confirming downtown location",
        loc_urls, critical=True
    )
    # Prime downtown location
    loc_stmt = data.location.statement if data and data.location and data.location.statement else "Downtown Seattle"
    await _verify_with_sources(
        evaluator, node, "Prime_Location",
        "Building is located in downtown Seattle CBD with high accessibility",
        claim=f"{bref} is located in downtown Seattle (CBD). It may also be described as '{loc_stmt}'.",
        urls=loc_urls,
        additional_instruction=(
            "Verify that the building is in Downtown Seattle. Accept canonical downtown subareas like "
            "Belltown, Denny Triangle, Financial District, Pioneer Square, West Edge, Retail Core, and Waterfront "
            "as within downtown."
        ),
        critical=True,
        extra_prereq=[loc_url_node]
    )


async def build_sustainability_energy(
    evaluator: Evaluator,
    parent: VerificationNode,
    data: BuildingExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Sustainability_Energy_Performance",
        desc="Verification of environmental certifications and energy efficiency",
        parent=parent,
        critical=True
    )

    # LEED sub-sequential: gate by URL then status
    leed_seq = evaluator.add_sequential(
        id="LEED_Gold_Certification",
        desc="Building has achieved LEED Gold certification (60-79 points) or higher",
        parent=node,
        critical=True
    )

    leed_urls = data.leed.urls if data and data.leed else []
    leed_url_node = _add_url_ref_check(
        evaluator, leed_seq, "LEED_URL_Reference",
        "Provides URL reference confirming LEED certification level",
        leed_urls, critical=True
    )

    leed_level = data.leed.level if data and data.leed and data.leed.level else ""
    await _verify_with_sources(
        evaluator, leed_seq, "LEED_Gold_Status",
        "Building has LEED Gold or Platinum certification",
        claim=f"The building { _format_bldg_ref(data) } has LEED certification at least Gold (e.g., '{leed_level}') or higher.",
        urls=leed_urls,
        additional_instruction=(
            "Confirm that the cited page(s) state LEED Gold, Platinum, or otherwise indicate Gold-or-higher certification. "
            "Explicit mention of 'LEED Gold' or 'LEED Platinum' qualifies."
        ),
        critical=True,
        extra_prereq=[leed_url_node]
    )

    # ENERGY STAR sub-sequential: gate by URL then status and score
    es_seq = evaluator.add_sequential(
        id="Energy_Star_Certification",
        desc="Building has Energy Star certification with score of 75 or higher",
        parent=node,
        critical=True
    )

    es_urls = data.energy_star.urls if data and data.energy_star else []
    es_url_node = _add_url_ref_check(
        evaluator, es_seq, "Energy_Star_URL_Reference",
        "Provides URL reference confirming Energy Star certification and score",
        es_urls, critical=True
    )

    es_status = data.energy_star.status if data and data.energy_star and data.energy_star.status else ""
    await _verify_with_sources(
        evaluator, es_seq, "Energy_Star_Status",
        "Building has active Energy Star certification",
        claim=f"{_format_bldg_ref(data)} is Energy Star certified. {es_status}",
        urls=es_urls,
        additional_instruction=(
            "Verify the building is Energy Star certified (active or recent). "
            "Accept clear statements on the page indicating Energy Star certification."
        ),
        critical=True,
        extra_prereq=[es_url_node]
    )

    es_score = data.energy_star.score if data and data.energy_star and data.energy_star.score else ""
    await _verify_with_sources(
        evaluator, es_seq, "Energy_Star_Score",
        "Energy Star score is 75 or higher",
        claim=f"The Energy Star score for {_format_bldg_ref(data)} is {es_score} and is 75 or higher.",
        urls=es_urls,
        additional_instruction=(
            "Check that the Energy Star score shown or stated on the cited page(s) is ≥ 75. "
            "If multiple years/scores are listed, any score ≥ 75 suffices."
        ),
        critical=True,
        extra_prereq=[es_url_node]
    )


async def build_physical_infrastructure(
    evaluator: Evaluator,
    parent: VerificationNode,
    data: BuildingExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Physical_Infrastructure_Safety",
        desc="Verification of building systems, safety features, and physical specifications",
        parent=parent,
        critical=True
    )

    # Transit accessibility (sequential): gate by URL then proximity content
    transit_seq = evaluator.add_sequential(
        id="Transit_Accessibility",
        desc="Building is within 1/2 mile walking distance of major transit stop",
        parent=node,
        critical=True
    )
    transit_urls = data.transit.urls if data and data.transit else []
    transit_url_node = _add_url_ref_check(
        evaluator, transit_seq, "Transit_URL_Reference",
        "Provides URL reference confirming transit proximity",
        transit_urls, critical=True
    )
    transit_desc = data.transit.description if data and data.transit and data.transit.description else "within 0.5 mile of a major transit stop"
    await _verify_with_sources(
        evaluator, transit_seq, "Transit_Proximity",
        "Building is located within 1/2 mile (800 meters) of light rail or major bus transit",
        claim=f"{_format_bldg_ref(data)} is {transit_desc}, satisfying ≤ 0.5 mile to a major transit stop (light rail or major bus).",
        urls=transit_urls,
        additional_instruction=(
            "Verify that the cited page(s) indicate the property is within 0.5 miles (approx. 800 m) walking distance "
            "to a major transit stop such as a Link light rail station or a major bus hub. "
            "Phrases like '5–10 minute walk' may be acceptable if it reasonably corresponds to ≤0.5 miles."
        ),
        critical=True,
        extra_prereq=[transit_url_node]
    )

    # Parking facilities (sequential): gate by URL then ratio
    parking_seq = evaluator.add_sequential(
        id="Parking_Facilities",
        desc="Building provides adequate parking at required ratio",
        parent=node,
        critical=True
    )
    parking_urls = data.parking.urls if data and data.parking else []
    parking_url_node = _add_url_ref_check(
        evaluator, parking_seq, "Parking_URL_Reference",
        "Provides URL reference confirming parking facilities and capacity",
        parking_urls, critical=True
    )
    pratio = data.parking.ratio if data and data.parking and data.parking.ratio else ""
    await _verify_with_sources(
        evaluator, parking_seq, "Parking_Ratio_Met",
        "Parking provided at minimum 4 spaces per 1,000 SF office space",
        claim=f"{_format_bldg_ref(data)} provides parking at a ratio of {pratio}, which is at least 4 spaces per 1,000 SF.",
        urls=parking_urls,
        additional_instruction=(
            "Confirm that the provided ratio is ≥ 4 per 1,000 SF (e.g., 4.0/1000, 4.2 per 1000). "
            "If a garage ratio is stated but for a mixed-use building, ensure it applies to office users."
        ),
        critical=True,
        extra_prereq=[parking_url_node]
    )

    # Fire safety systems (sequential): gate by URL then sprinkler content
    fire_seq = evaluator.add_sequential(
        id="Fire_Safety_Systems",
        desc="Building has required fire suppression and safety systems",
        parent=node,
        critical=True
    )
    sprinkler_urls = data.sprinkler.urls if data and data.sprinkler else []
    fire_url_node = _add_url_ref_check(
        evaluator, fire_seq, "Fire_Safety_URL_Reference",
        "Provides URL reference confirming fire safety systems",
        sprinkler_urls, critical=True
    )
    spr_stmt = data.sprinkler.statement if data and data.sprinkler and data.sprinkler.statement else "sprinklered throughout"
    await _verify_with_sources(
        evaluator, fire_seq, "Sprinkler_System",
        "Building has automatic fire sprinkler system throughout",
        claim=f"{_format_bldg_ref(data)} has an automatic fire sprinkler system throughout the building ({spr_stmt}).",
        urls=sprinkler_urls,
        additional_instruction=(
            "Look for explicit mentions such as 'fully sprinklered', 'automatic fire sprinkler system', or 'NFPA-compliant sprinklers'. "
            "General code references alone are insufficient unless directly tied to the specific building."
        ),
        critical=True,
        extra_prereq=[fire_url_node]
    )

    # ADA compliance (sequential): gate by URL then ADA content
    ada_seq = evaluator.add_sequential(
        id="ADA_Compliance",
        desc="Building meets ADA accessibility requirements",
        parent=node,
        critical=True
    )
    ada_urls = data.ada.urls if data and data.ada else []
    ada_url_node = _add_url_ref_check(
        evaluator, ada_seq, "ADA_URL_Reference",
        "Provides URL reference confirming ADA compliance",
        ada_urls, critical=True
    )
    ada_stmt = data.ada.statement if data and data.ada and data.ada.statement else "ADA accessible"
    await _verify_with_sources(
        evaluator, ada_seq, "ADA_Standards_Met",
        "Building meets ADA accessibility standards including door widths and accessible routes",
        claim=f"{_format_bldg_ref(data)} meets ADA accessibility standards (e.g., accessible routes, entrances, restrooms): '{ada_stmt}'.",
        urls=ada_urls,
        additional_instruction=(
            "Confirm that the cited page(s) indicate ADA compliance or accessibility features for the building. "
            "Accept clear phrases such as 'ADA compliant', 'ADA accessible', or explicit descriptions of accessible facilities."
        ),
        critical=True,
        extra_prereq=[ada_url_node]
    )


async def build_financial_operational(
    evaluator: Evaluator,
    parent: VerificationNode,
    data: BuildingExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Financial_Operational_Viability",
        desc="Verification of lease structure, occupancy, and financial performance",
        parent=parent,
        critical=True
    )

    # Lease Structure (sequential): gate by URL then NNN type
    lease_seq = evaluator.add_sequential(
        id="Lease_Structure",
        desc="Building operates under triple net (NNN) lease arrangements",
        parent=node,
        critical=True
    )
    lease_urls = data.lease_structure.urls if data and data.lease_structure else []
    lease_url_node = _add_url_ref_check(
        evaluator, lease_seq, "Lease_URL_Reference",
        "Provides URL reference confirming lease structure type",
        lease_urls, critical=True
    )
    lease_type = data.lease_structure.lease_type if data and data.lease_structure and data.lease_structure.lease_type else "NNN"
    await _verify_with_sources(
        evaluator, lease_seq, "NNN_Lease_Type",
        "Major tenants have triple net leases (tenants pay taxes, insurance, CAM)",
        claim=f"{_format_bldg_ref(data)} utilizes a triple net (NNN) lease structure (e.g., '{lease_type}').",
        urls=lease_urls,
        additional_instruction=(
            "Look for 'NNN', 'triple net', or language indicating tenants pay property taxes, insurance, and CAM."
        ),
        critical=True,
        extra_prereq=[lease_url_node]
    )

    # Tenant quality & lease terms (parallel critical): two sequential subnodes
    tenant_lt_parallel = evaluator.add_parallel(
        id="Tenant_Quality_Lease_Terms",
        desc="Building has quality tenants with appropriate lease terms",
        parent=node,
        critical=True
    )

    # Major corporate tenant (sequential): gate by URL then presence
    corp_seq = evaluator.add_sequential(
        id="Major_Corporate_Tenants",
        desc="Building has major corporate tenants (Fortune 1000 or equivalent)",
        parent=tenant_lt_parallel,
        critical=True
    )
    tenant_urls = data.major_tenant.urls if data and data.major_tenant else []
    tenant_url_node = _add_url_ref_check(
        evaluator, corp_seq, "Tenant_URL_Reference",
        "Provides URL reference confirming tenant identity and presence",
        tenant_urls, critical=True
    )
    tenant_name = data.major_tenant.tenant_name if data and data.major_tenant and data.major_tenant.tenant_name else "a major corporate tenant"
    await _verify_with_sources(
        evaluator, corp_seq, "Corporate_Tenant_Present",
        "At least one major corporate tenant identified",
        claim=f"{_format_bldg_ref(data)} has at least one major corporate tenant such as '{tenant_name}'.",
        urls=tenant_urls,
        additional_instruction=(
            "Confirm that the cited page(s) show at least one sizable corporate tenant. "
            "If the page or an accompanying citation indicates the tenant is Fortune 1000 (or equivalent), that satisfies the requirement. "
            "If not explicitly stated, rely on credible indicators (e.g., well-known large corporation)."
        ),
        critical=True,
        extra_prereq=[tenant_url_node]
    )

    # Lease term length (sequential): gate by URL then ≥5 years
    term_seq = evaluator.add_sequential(
        id="Lease_Term_Length",
        desc="Major tenants have lease terms of at least 5 years",
        parent=tenant_lt_parallel,
        critical=True
    )
    term_urls = data.lease_term.urls if data and data.lease_term else []
    term_url_node = _add_url_ref_check(
        evaluator, term_seq, "Lease_Term_URL_Reference",
        "Provides URL reference confirming lease term duration",
        term_urls, critical=True
    )
    term_str = data.lease_term.term_years if data and data.lease_term and data.lease_term.term_years else ""
    await _verify_with_sources(
        evaluator, term_seq, "Five_Year_Minimum",
        "Lease terms are 5 years or longer for major tenants",
        claim=f"The major tenant(s) at {_format_bldg_ref(data)} have lease terms of at least 5 years (e.g., '{term_str}').",
        urls=term_urls,
        additional_instruction=(
            "Verify lease terms for major tenant(s) are ≥ 5 years based on the cited page(s). "
            "If multiple terms are listed, at least one major tenant with ≥ 5 years satisfies the criterion."
        ),
        critical=True,
        extra_prereq=[term_url_node]
    )

    # Occupancy performance (sequential): gate by URL then ≥90%
    occ_seq = evaluator.add_sequential(
        id="Occupancy_Performance",
        desc="Building maintains high occupancy rate indicating strong demand",
        parent=node,
        critical=True
    )
    occ_urls = data.occupancy.urls if data and data.occupancy else []
    occ_url_node = _add_url_ref_check(
        evaluator, occ_seq, "Occupancy_URL_Reference",
        "Provides URL reference confirming current occupancy rate",
        occ_urls, critical=True
    )
    occ_rate = data.occupancy.rate if data and data.occupancy and data.occupancy.rate else ""
    await _verify_with_sources(
        evaluator, occ_seq, "Ninety_Percent_Occupancy",
        "Building maintains minimum 90% occupancy rate",
        claim=f"{_format_bldg_ref(data)} has a current occupancy of {occ_rate}, which is at least 90%.",
        urls=occ_urls,
        additional_instruction=(
            "Confirm from the cited page(s) that occupancy is ≥ 90%. "
            "Accept explicit occupancy percentages or clear statements like 'above 90% occupancy'."
        ),
        critical=True,
        extra_prereq=[occ_url_node]
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
    Evaluate an answer for the Seattle Class A office institutional investment criteria.
    """
    evaluator = Evaluator()
    _ = evaluator.initialize(
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

    # Step 1: Extract structured building information from the answer
    extracted: BuildingExtraction = await evaluator.extract(
        prompt=prompt_extract_building(),
        template_class=BuildingExtraction,
        extraction_name="building_extraction"
    )

    # Optional: record the identified building basic info for summary
    evaluator.add_custom_info(
        {
            "building_name": extracted.building_name,
            "address": extracted.address
        },
        info_type="basic_building_info"
    )

    # Step 2: Build verification tree according to rubric
    # Create a top-level critical sequential node to reflect the rubric's root criticality
    suitability_root = evaluator.add_sequential(
        id="Building_Investment_Suitability",
        desc="Overall evaluation of whether the proposed Class A office building meets all institutional investment criteria",
        parent=evaluator.root,
        critical=True
    )

    # Section 1: Building classification & quality
    await build_classification_quality(evaluator, suitability_root, extracted)

    # Section 2: Sustainability & energy performance
    await build_sustainability_energy(evaluator, suitability_root, extracted)

    # Section 3: Physical infrastructure & safety
    await build_physical_infrastructure(evaluator, suitability_root, extracted)

    # Section 4: Financial & operational viability
    await build_financial_operational(evaluator, suitability_root, extracted)

    # Step 3: Return the summarized evaluation results
    return evaluator.get_summary()