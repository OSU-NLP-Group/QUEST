import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nashville_mixed_use_2024_2025"
TASK_DESCRIPTION = """Identify one major mixed-use development project in Nashville, Tennessee that was active, under construction, or completed during 2024-2025, and provide a comprehensive compliance evaluation across the specified regulatory domains with supporting URLs."""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Identification(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    county: Optional[str] = None
    description: Optional[str] = None
    primary_source_urls: List[str] = Field(default_factory=list)

    uses: List[str] = Field(default_factory=list)  # e.g., ["residential", "retail", "office"]
    mixed_use_sources: List[str] = Field(default_factory=list)

    timeframe_statement: Optional[str] = None  # e.g., "under construction in 2024", "completed in 2025"
    timeframe_sources: List[str] = Field(default_factory=list)

    major_scale_statement: Optional[str] = None  # e.g., "22-story tower", "450,000 sq ft"
    scale_sources: List[str] = Field(default_factory=list)


class ZoningLandUse(BaseModel):
    zoning_approval_statement: Optional[str] = None
    zoning_height_statement: Optional[str] = None
    zoning_doc_urls: List[str] = Field(default_factory=list)


class ParkingTransportation(BaseModel):
    parking_requirements_statement: Optional[str] = None
    parking_landscaping_screening_statement: Optional[str] = None
    accessways_statement: Optional[str] = None
    parking_doc_urls: List[str] = Field(default_factory=list)


class SustainableBuildingCertification(BaseModel):
    leed_applicability_statement: Optional[str] = None
    leed_status_statement: Optional[str] = None
    leed_points_statement: Optional[str] = None
    sustainability_doc_urls: List[str] = Field(default_factory=list)


class AffordableHousing(BaseModel):
    affordable_presence_statement: Optional[str] = None  # "includes X%" or "none/unknown"
    affordable_details_statement: Optional[str] = None
    affordable_doc_urls: List[str] = Field(default_factory=list)


class StormwaterManagement(BaseModel):
    stormwater_plan_statement: Optional[str] = None
    lid_or_limitations_statement: Optional[str] = None
    drainage_statement: Optional[str] = None
    stormwater_doc_urls: List[str] = Field(default_factory=list)


class EnvironmentalReview(BaseModel):
    hud_home_funding_statement: Optional[str] = None  # e.g., "receives HUD/HOME funding" or "no federal funding"
    environmental_review_statement: Optional[str] = None
    environmental_doc_urls: List[str] = Field(default_factory=list)


class ADAAccessibility(BaseModel):
    accessible_parking_statement: Optional[str] = None
    accessible_routes_statement: Optional[str] = None
    ada_doc_urls: List[str] = Field(default_factory=list)


class BuildingPermitsSitePlans(BaseModel):
    permits_obtained_statement: Optional[str] = None
    site_plans_content_statement: Optional[str] = None
    inspections_statement: Optional[str] = None
    co_statement: Optional[str] = None
    permits_doc_urls: List[str] = Field(default_factory=list)


class PublicApprovalProcess(BaseModel):
    planning_hearing_statement: Optional[str] = None
    public_notice_statement: Optional[str] = None
    council_approval_statement: Optional[str] = None
    approval_doc_urls: List[str] = Field(default_factory=list)


class ConstructionTimeline(BaseModel):
    phases_documented_statement: Optional[str] = None
    current_status_statement: Optional[str] = None
    duration_alignment_statement: Optional[str] = None
    timeline_doc_urls: List[str] = Field(default_factory=list)


class CommunityBenefits(BaseModel):
    cba_or_amenities_statement: Optional[str] = None
    community_benefits_doc_urls: List[str] = Field(default_factory=list)


class TransitOrientedDevelopment(BaseModel):
    transit_proximity_statement: Optional[str] = None
    tod_guidelines_statement: Optional[str] = None
    tod_doc_urls: List[str] = Field(default_factory=list)


class GreenInfrastructure(BaseModel):
    green_infrastructure_practices_statement: Optional[str] = None
    energy_efficiency_beyond_code_statement: Optional[str] = None
    green_infrastructure_doc_urls: List[str] = Field(default_factory=list)


class ProjectExtraction(BaseModel):
    identification: Optional[Identification] = None

    zoning_land_use: Optional[ZoningLandUse] = None
    parking_transportation: Optional[ParkingTransportation] = None
    sustainable_building_certification: Optional[SustainableBuildingCertification] = None
    affordable_housing_component: Optional[AffordableHousing] = None
    stormwater_management: Optional[StormwaterManagement] = None
    environmental_review: Optional[EnvironmentalReview] = None
    ada_accessibility: Optional[ADAAccessibility] = None
    building_permits_site_plans: Optional[BuildingPermitsSitePlans] = None
    public_approval_process: Optional[PublicApprovalProcess] = None
    construction_timeline: Optional[ConstructionTimeline] = None
    community_benefits: Optional[CommunityBenefits] = None
    transit_oriented_development: Optional[TransitOrientedDevelopment] = None
    green_infrastructure: Optional[GreenInfrastructure] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project() -> str:
    return """
    Extract exactly one mixed-use development project from the answer (choose the primary/most emphasized one if multiple are mentioned).
    Return a JSON object following the ProjectExtraction schema. Rules:
    - Extract only what is explicitly stated in the answer.
    - For each domain, capture the stated findings/claims as short text strings and the supporting documentation URLs as arrays of URLs.
    - If the answer does not provide information for a field, set the field to null (or empty list for URLs).
    - For URLs, extract only valid, complete URLs that appear in the answer (plain or markdown).

    Fields to extract:

    identification:
      - name: project name
      - location: site address/neighborhood
      - county: county name
      - description: brief basic description
      - primary_source_urls: list of primary URLs confirming project identification/details
      - uses: list of uses explicitly mentioned (e.g., residential, retail, office)
      - mixed_use_sources: list of URLs that support the mixed-use designation
      - timeframe_statement: text indicating activity during 2024–2025 (e.g., under construction 2024, completed 2025)
      - timeframe_sources: URLs supporting the timeframe statement
      - major_scale_statement: text indicating major scale (e.g., multi-story, sq ft, investment)
      - scale_sources: URLs supporting major scale

    zoning_land_use:
      - zoning_approval_statement: text confirming zoning approval or rezoning
      - zoning_height_statement: text confirming height compliance or variance
      - zoning_doc_urls: URLs supporting zoning/land-use/height info

    parking_transportation:
      - parking_requirements_statement: text confirming parking compliance or exemption per 2022 downtown parking ordinance
      - parking_landscaping_screening_statement: text confirming parking landscaping/screening compliance
      - accessways_statement: text confirming approved vehicle/pedestrian access ways
      - parking_doc_urls: URLs supporting parking/transportation/access info

    sustainable_building_certification:
      - leed_applicability_statement: text determining LEED requirement under BL2007-1374
      - leed_status_statement: text confirming pursuit/achievement or not required
      - leed_points_statement: text about points threshold if claiming LEED Certified
      - sustainability_doc_urls: URLs supporting LEED applicability/status

    affordable_housing_component:
      - affordable_presence_statement: text indicating presence or none/unknown of affordable units
      - affordable_details_statement: text with details if any (e.g., % units, AMI)
      - affordable_doc_urls: URLs supporting affordable housing info (if applicable)

    stormwater_management:
      - stormwater_plan_statement: text confirming approved stormwater plan per Volume 1
      - lid_or_limitations_statement: text confirming LID techniques or documented limitations
      - drainage_statement: text confirming drainage infrastructure addressed
      - stormwater_doc_urls: URLs supporting stormwater/LID/drainage info

    environmental_review:
      - hud_home_funding_statement: text indicating whether project receives HUD/HOME federal funding
      - environmental_review_statement: text confirming environmental review completion if federally funded
      - environmental_doc_urls: URLs supporting federal-funding/environmental-review determination

    ada_accessibility:
      - accessible_parking_statement: text confirming ADA accessible parking spaces provided
      - accessible_routes_statement: text confirming accessible pathways with minimum 36-inch clear width
      - ada_doc_urls: URLs supporting ADA/accessibility info

    building_permits_site_plans:
      - permits_obtained_statement: text confirming building permits obtained from Metro Codes
      - site_plans_content_statement: text confirming approved site plans show property lines, setbacks, structures, access ways
      - inspections_statement: text confirming required inspections scheduled/completed
      - co_statement: text confirming Certificate of Occupancy/Completion obtained or not yet applicable
      - permits_doc_urls: URLs supporting permits/site plan/inspection/CO info

    public_approval_process:
      - planning_hearing_statement: text confirming Planning Commission public hearing occurred if required (or not applicable)
      - public_notice_statement: text confirming public notice mailed within 300 feet if required (or not applicable)
      - council_approval_statement: text confirming Metro Council approval obtained if required (or not applicable)
      - approval_doc_urls: URLs supporting the public approval process info

    construction_timeline:
      - phases_documented_statement: text documenting phases (pre-construction, site work, construction, finishing)
      - current_status_statement: text identifying current status in 2024–2025 (planned/under construction/completed)
      - duration_alignment_statement: text stating duration alignment with typical 6–18+ months or explanation
      - timeline_doc_urls: URLs supporting timeline/status info

    community_benefits:
      - cba_or_amenities_statement: text indicating presence or none/unknown of CBA/community amenities
      - community_benefits_doc_urls: URLs supporting CBA/amenities info (if applicable)

    transit_oriented_development:
      - transit_proximity_statement: text identifying proximity to public transit (or not proximate/unknown)
      - tod_guidelines_statement: text documenting TOD guidelines compliance if applicable (or not applicable)
      - tod_doc_urls: URLs supporting transit proximity/TOD guideline info

    green_infrastructure:
      - green_infrastructure_practices_statement: text documenting green infrastructure/sustainable site design (or none/unknown)
      - energy_efficiency_beyond_code_statement: text documenting energy efficiency beyond code (or none/unknown)
      - green_infrastructure_doc_urls: URLs supporting green infrastructure/energy measures info
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_not_applicable(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    keywords = ["not applicable", "n/a", "none", "no", "unknown", "not funded", "no federal funding"]
    return any(k in t for k in keywords)


def combine_sources(primary: List[str], secondary: List[str]) -> List[str]:
    """Return secondary if non-empty; else primary; else empty list."""
    if secondary:
        return secondary
    if primary:
        return primary
    return []


def ensure_list(val: Optional[List[str]]) -> List[str]:
    return val if val else []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_project_identification(
    evaluator: Evaluator,
    parent_node,
    data: ProjectExtraction
) -> None:
    ident = data.identification or Identification()

    # Project identification node (critical, parallel)
    pid_node = evaluator.add_parallel(
        id="project_identification",
        desc="Project identification and eligibility verification (must be satisfied before compliance evaluation).",
        parent=parent_node,
        critical=True
    )

    # Existence: name, location, description
    name_loc_desc_ok = bool(ident.name) and bool(ident.location) and bool(ident.description)
    evaluator.add_custom_node(
        result=name_loc_desc_ok,
        id="project_name_location_description",
        desc="Provides project name, location, and a basic description.",
        parent=pid_node,
        critical=True
    )

    # Mixed-use designation
    mixed_claim = "The project is mixed-use combining residential with commercial, retail, or office uses."
    mix_sources = combine_sources(ensure_list(ident.primary_source_urls), ensure_list(ident.mixed_use_sources))
    mixed_leaf = evaluator.add_leaf(
        id="mixed_use_designation",
        desc="Project is mixed-use combining residential with commercial/retail/office uses.",
        parent=pid_node,
        critical=True
    )
    await evaluator.verify(
        claim=mixed_claim,
        node=mixed_leaf,
        sources=mix_sources,
        additional_instruction=f"Verify that the cited page(s) explicitly mention residential and at least one non-residential use (retail/commercial/office). Uses extracted from the answer: {ident.uses}."
    )

    # Nashville location (Davidson County)
    loc_claim = "The project is located in Nashville/Davidson County, Tennessee."
    loc_leaf = evaluator.add_leaf(
        id="nashville_location",
        desc="Project is located in Nashville/Davidson County, Tennessee.",
        parent=pid_node,
        critical=True
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=ensure_list(ident.primary_source_urls),
        additional_instruction=f"Check the address or location text for 'Nashville' or 'Davidson County'. Extracted location: {ident.location}, county: {ident.county}."
    )

    # Timeframe (2024–2025)
    timeframe_leaf = evaluator.add_leaf(
        id="timeframe_2024_2025",
        desc="Project is active, under construction, or completed during 2024–2025.",
        parent=pid_node,
        critical=True
    )
    timeframe_claim = "During 2024–2025, the project was active, under construction, or completed."
    tf_sources = combine_sources(ensure_list(ident.primary_source_urls), ensure_list(ident.timeframe_sources))
    await evaluator.verify(
        claim=timeframe_claim,
        node=timeframe_leaf,
        sources=tf_sources,
        additional_instruction=f"Look for dates/status indicating activity within 2024–2025. Extracted timeframe statement: {ident.timeframe_statement}."
    )

    # Major development scale
    major_leaf = evaluator.add_leaf(
        id="major_development_scale",
        desc="Project qualifies as a major development (multi-story/significant scale, e.g., significant square footage or multi-million dollar investment).",
        parent=pid_node,
        critical=True
    )
    major_claim = "The project qualifies as a major development (multi-story or significant scale)."
    scale_sources = combine_sources(ensure_list(ident.primary_source_urls), ensure_list(ident.scale_sources))
    await evaluator.verify(
        claim=major_claim,
        node=major_leaf,
        sources=scale_sources,
        additional_instruction=f"Accept indicators such as multi-story tower, large square footage, or major investment. Extracted scale indicator: {ident.major_scale_statement}."
    )

    # Primary source presence gate
    primary_present = evaluator.add_custom_node(
        result=len(ensure_list(ident.primary_source_urls)) > 0,
        id="primary_source_present",
        desc="Primary source URL(s) provided in the answer.",
        parent=pid_node,
        critical=True
    )

    # Primary source URL verification
    primary_leaf = evaluator.add_leaf(
        id="primary_source_url",
        desc="Provides at least one verifiable primary source URL confirming the project identification/details.",
        parent=pid_node,
        critical=True
    )
    first_primary = ensure_list(ident.primary_source_urls)[0] if ensure_list(ident.primary_source_urls) else None
    primary_claim = f"This URL corresponds to the project named '{ident.name}' located in Nashville." if ident.name else "This URL corresponds to the identified project."
    await evaluator.verify(
        claim=primary_claim,
        node=primary_leaf,
        sources=first_primary,
        additional_instruction="Verify the page is clearly about the identified project, matching name/location/description.",
        extra_prerequisites=[primary_present]
    )


async def build_zoning_land_use(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    info = data.zoning_land_use or ZoningLandUse()
    node = evaluator.add_parallel(
        id="zoning_land_use",
        desc="Zoning & land use compliance.",
        parent=parent,
        critical=True
    )
    doc_presence = evaluator.add_custom_node(
        result=len(ensure_list(info.zoning_doc_urls)) > 0,
        id="zoning_documentation_url",
        desc="Provides documentation URL for zoning/land-use/height compliance information.",
        parent=node,
        critical=True
    )
    # Zoning approval/rezoning
    leaf1 = evaluator.add_leaf(
        id="zoning_approval_or_rezoning",
        desc="Verifies appropriate zoning approval or approved rezoning under Nashville UDO.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.zoning_approval_statement or "The project has appropriate zoning approval or rezoning.",
        node=leaf1,
        sources=ensure_list(info.zoning_doc_urls),
        additional_instruction="Check for explicit zoning approval, rezoning ordinance/SP, or UDO compliance references.",
        extra_prerequisites=[doc_presence]
    )
    # Height compliance/variance
    leaf2 = evaluator.add_leaf(
        id="height_compliance_or_variance",
        desc="Verifies building height complies with applicable district limits or a variance was obtained.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.zoning_height_statement or "The building height complies with applicable district limits or has an approved variance.",
        node=leaf2,
        sources=ensure_list(info.zoning_doc_urls),
        additional_instruction="Look for stated height and district limits or Board of Zoning Appeals variance documentation.",
        extra_prerequisites=[doc_presence]
    )


async def build_parking_transportation(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    info = data.parking_transportation or ParkingTransportation()
    node = evaluator.add_parallel(
        id="parking_transportation",
        desc="Parking & transportation compliance.",
        parent=parent,
        critical=True
    )
    doc_presence = evaluator.add_custom_node(
        result=len(ensure_list(info.parking_doc_urls)) > 0,
        id="parking_documentation_url",
        desc="Provides documentation URL for parking/transportation/access compliance information.",
        parent=node,
        critical=True
    )
    # Parking req/exemption
    leaf1 = evaluator.add_leaf(
        id="parking_requirements_or_exemption",
        desc="Confirms project meets parking requirements OR qualifies for reduced/eliminated minimums per 2022 downtown parking ordinance.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.parking_requirements_statement or "The project meets parking requirements or qualifies for reduced/eliminated minimums under the 2022 downtown parking ordinance.",
        node=leaf1,
        sources=ensure_list(info.parking_doc_urls),
        additional_instruction="Verify compliance against Nashville's 2022 downtown parking ordinance or stated exemptions.",
        extra_prerequisites=[doc_presence]
    )
    # Landscaping/screening
    leaf2 = evaluator.add_leaf(
        id="parking_landscaping_screening",
        desc="Confirms parking areas meet landscaping and screening requirements per Metro Code.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.parking_landscaping_screening_statement or "Parking areas meet required landscaping and screening standards.",
        node=leaf2,
        sources=ensure_list(info.parking_doc_urls),
        additional_instruction="Check for landscape/screening plans or Metro Code references for parking areas.",
        extra_prerequisites=[doc_presence]
    )
    # Access ways
    leaf3 = evaluator.add_leaf(
        id="vehicle_pedestrian_accessways",
        desc="Confirms approved vehicle and pedestrian access ways are provided/described.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.accessways_statement or "Approved vehicle and pedestrian access ways are provided.",
        node=leaf3,
        sources=ensure_list(info.parking_doc_urls),
        additional_instruction="Look for circulation plans, ingress/egress approvals, and pedestrian pathway layouts.",
        extra_prerequisites=[doc_presence]
    )


async def build_sustainability_leed(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    info = data.sustainable_building_certification or SustainableBuildingCertification()
    node = evaluator.add_parallel(
        id="sustainable_building_certification",
        desc="Sustainable building certification (LEED) evaluation.",
        parent=parent,
        critical=True
    )
    doc_presence = evaluator.add_custom_node(
        result=len(ensure_list(info.sustainability_doc_urls)) > 0,
        id="sustainability_documentation_url",
        desc="Provides documentation URL supporting the LEED applicability and status determination.",
        parent=node,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="leed_applicability_determination",
        desc="Determines whether LEED is required under BL2007-1374 (public/publicly-funded buildings ≥5,000 sq ft).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.leed_applicability_statement or "LEED applicability has been determined under BL2007-1374.",
        node=leaf1,
        sources=ensure_list(info.sustainability_doc_urls),
        additional_instruction="Check whether the project is public/publicly-funded and ≥5,000 sq ft to require LEED.",
        extra_prerequisites=[doc_presence]
    )
    leaf2 = evaluator.add_leaf(
        id="leed_pursuit_or_achievement_if_required",
        desc="If LEED is required, verifies that LEED certification is being pursued or has been achieved; if not required, explicitly states not required.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.leed_status_statement or "LEED status (pursuing/achieved or not required) is correctly stated.",
        node=leaf2,
        sources=ensure_list(info.sustainability_doc_urls),
        additional_instruction="Verify stated pursuit/achievement or that LEED is not required per applicability determination.",
        extra_prerequisites=[doc_presence]
    )
    leaf3 = evaluator.add_leaf(
        id="leed_points_threshold_if_claiming_certified",
        desc="If claiming LEED Certified level, verifies minimum 40 points threshold is met (or equivalent supporting evidence is cited).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.leed_points_statement or "If claiming LEED Certified, the minimum 40 points threshold is met.",
        node=leaf3,
        sources=ensure_list(info.sustainability_doc_urls),
        additional_instruction="Only enforce points threshold if 'LEED Certified' is claimed; otherwise verify 'not applicable'.",
        extra_prerequisites=[doc_presence]
    )


async def build_affordable_housing(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    info = data.affordable_housing_component or AffordableHousing()
    node = evaluator.add_parallel(
        id="affordable_housing_component",
        desc="Affordable housing component evaluation.",
        parent=parent,
        critical=True
    )
    # Presence statement
    leaf1 = evaluator.add_leaf(
        id="affordable_housing_presence",
        desc="States whether the project includes voluntary affordable housing units or states none/unknown.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.affordable_presence_statement or "Affordable housing component is stated (present or none/unknown).",
        node=leaf1,
        sources=ensure_list(info.affordable_doc_urls),
        additional_instruction="Verify the presence or explicit none/unknown; accept statements indicating % or AMI if present."
    )
    # Details statement
    leaf2 = evaluator.add_leaf(
        id="affordable_housing_details_if_any",
        desc="If affordable housing is included, documents the specific commitments/terms; otherwise marks as not applicable.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.affordable_details_statement or "Affordable housing details (if any) are correctly stated or marked not applicable.",
        node=leaf2,
        sources=ensure_list(info.affordable_doc_urls),
        additional_instruction="If presence indicates 'none/unknown', treat details as not applicable; otherwise verify cited specifics."
    )
    # Documentation URL conditional presence
    doc_ok = is_not_applicable(info.affordable_presence_statement) or len(ensure_list(info.affordable_doc_urls)) > 0
    evaluator.add_custom_node(
        result=doc_ok,
        id="affordable_housing_documentation_url_if_applicable",
        desc="Provides documentation URL if an affordable housing component is claimed/applicable.",
        parent=node,
        critical=True
    )


async def build_stormwater(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    info = data.stormwater_management or StormwaterManagement()
    node = evaluator.add_parallel(
        id="stormwater_management",
        desc="Stormwater management compliance.",
        parent=parent,
        critical=True
    )
    doc_presence = evaluator.add_custom_node(
        result=len(ensure_list(info.stormwater_doc_urls)) > 0,
        id="stormwater_documentation_url",
        desc="Provides documentation URL for stormwater/LID/drainage information.",
        parent=node,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="approved_stormwater_plan",
        desc="Verifies an approved stormwater management plan per Nashville Stormwater Management Manual Volume 1.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.stormwater_plan_statement or "An approved stormwater management plan is in place per Volume 1.",
        node=leaf1,
        sources=ensure_list(info.stormwater_doc_urls),
        additional_instruction="Look for explicit plan approvals or references to Nashville Stormwater Management Manual Volume 1.",
        extra_prerequisites=[doc_presence]
    )
    leaf2 = evaluator.add_leaf(
        id="lid_or_limitations",
        desc="Verifies Low Impact Development techniques are incorporated OR site limitations are documented.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.lid_or_limitations_statement or "LID techniques are incorporated or site limitations are documented.",
        node=leaf2,
        sources=ensure_list(info.stormwater_doc_urls),
        additional_instruction="Verify mention of LID practices or documented infeasibility.",
        extra_prerequisites=[doc_presence]
    )
    leaf3 = evaluator.add_leaf(
        id="drainage_infrastructure_addressed",
        desc="Verifies drainage infrastructure is addressed.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.drainage_statement or "Drainage infrastructure requirements are addressed.",
        node=leaf3,
        sources=ensure_list(info.stormwater_doc_urls),
        additional_instruction="Check drainage plans, pipes, detention/retention facilities descriptions.",
        extra_prerequisites=[doc_presence]
    )


async def build_environmental_review(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    info = data.environmental_review or EnvironmentalReview()
    node = evaluator.add_parallel(
        id="environmental_review",
        desc="Environmental review (federal funding contingent) evaluation.",
        parent=parent,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="hud_home_funding_determination",
        desc="Determines whether the project receives federal funding (HUD/HOME).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.hud_home_funding_statement or "The determination of HUD/HOME federal funding is stated.",
        node=leaf1,
        sources=ensure_list(info.environmental_doc_urls),
        additional_instruction="Verify explicit statements about federal funding involvement."
    )
    leaf2 = evaluator.add_leaf(
        id="environmental_review_if_federally_funded",
        desc="If federally funded, verifies environmental review completion before project approval; otherwise states not applicable.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.environmental_review_statement or "Environmental review completion is verified if federally funded, or 'not applicable' otherwise.",
        node=leaf2,
        sources=ensure_list(info.environmental_doc_urls),
        additional_instruction="If HUD/HOME funded, there must be an environmental review (e.g., NEPA/HUD ER) completed prior to approval."
    )
    # Documentation conditional presence
    doc_ok = is_not_applicable(info.hud_home_funding_statement) or len(ensure_list(info.environmental_doc_urls)) > 0
    evaluator.add_custom_node(
        result=doc_ok,
        id="environmental_documentation_url",
        desc="Provides documentation URL supporting the federal-funding and environmental-review determination.",
        parent=node,
        critical=True
    )


async def build_ada(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    info = data.ada_accessibility or ADAAccessibility()
    node = evaluator.add_parallel(
        id="ada_accessibility",
        desc="ADA accessibility compliance.",
        parent=parent,
        critical=True
    )
    doc_presence = evaluator.add_custom_node(
        result=len(ensure_list(info.ada_doc_urls)) > 0,
        id="ada_documentation_url",
        desc="Provides documentation URL for ADA/accessibility information.",
        parent=node,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="accessible_parking_provided",
        desc="Confirms accessible parking spaces are provided per ADA requirements.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.accessible_parking_statement or "Accessible parking spaces are provided per ADA.",
        node=leaf1,
        sources=ensure_list(info.ada_doc_urls),
        additional_instruction="Look for counts/markings of accessible parking spaces and compliance references.",
        extra_prerequisites=[doc_presence]
    )
    leaf2 = evaluator.add_leaf(
        id="accessible_routes_36_inch_min",
        desc="Confirms accessible pathways/routes meet minimum 36-inch clear width requirement.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.accessible_routes_statement or "Accessible routes meet the minimum 36-inch clear width requirement.",
        node=leaf2,
        sources=ensure_list(info.ada_doc_urls),
        additional_instruction="Check for ADA route dimensions or compliance statements.",
        extra_prerequisites=[doc_presence]
    )


async def build_permits(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    info = data.building_permits_site_plans or BuildingPermitsSitePlans()
    node = evaluator.add_parallel(
        id="building_permits_site_plans",
        desc="Building permits & site plans (and related completion requirement) evaluation.",
        parent=parent,
        critical=True
    )
    doc_presence = evaluator.add_custom_node(
        result=len(ensure_list(info.permits_doc_urls)) > 0,
        id="permits_documentation_url",
        desc="Provides documentation URL for permits/site plan/inspection/CO information.",
        parent=node,
        critical=True
    )
    # Permits obtained
    leaf1 = evaluator.add_leaf(
        id="permits_obtained",
        desc="Verifies building permits were obtained from Metro Codes Department.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.permits_obtained_statement or "Building permits were obtained from Metro Codes Department.",
        node=leaf1,
        sources=ensure_list(info.permits_doc_urls),
        additional_instruction="Verify permit numbers or approvals from Metro Codes.",
        extra_prerequisites=[doc_presence]
    )
    # Site plan content
    leaf2 = evaluator.add_leaf(
        id="site_plans_content",
        desc="Verifies approved site plans show property lines, setbacks, structures, and access ways.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.site_plans_content_statement or "Approved site plans show property lines, setbacks, structures, and access ways.",
        node=leaf2,
        sources=ensure_list(info.permits_doc_urls),
        additional_instruction="Look for plan sheets listing property lines, setbacks, building footprints, and access.",
        extra_prerequisites=[doc_presence]
    )
    # Inspections
    leaf3 = evaluator.add_leaf(
        id="inspections_scheduled_completed",
        desc="Verifies required inspections are scheduled/completed as required.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.inspections_statement or "Required inspections are scheduled or completed.",
        node=leaf3,
        sources=ensure_list(info.permits_doc_urls),
        additional_instruction="Check inspection schedules or completion logs.",
        extra_prerequisites=[doc_presence]
    )
    # Certificate of Occupancy/Completion
    leaf4 = evaluator.add_leaf(
        id="certificate_of_occupancy_completion",
        desc="Verifies Certificate of Occupancy/Completion is obtained before building use (or states not yet applicable if still under construction).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.co_statement or "Certificate of Occupancy/Completion obtained (or not yet applicable if under construction).",
        node=leaf4,
        sources=ensure_list(info.permits_doc_urls),
        additional_instruction="Verify CO issuance or valid explanation of inapplicability.",
        extra_prerequisites=[doc_presence]
    )


async def build_public_approval(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    info = data.public_approval_process or PublicApprovalProcess()
    node = evaluator.add_parallel(
        id="public_approval_process",
        desc="Public approval process evaluation.",
        parent=parent,
        critical=True
    )
    doc_presence = evaluator.add_custom_node(
        result=len(ensure_list(info.approval_doc_urls)) > 0,
        id="approval_documentation_url",
        desc="Provides documentation URL for public approval process information.",
        parent=node,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="planning_commission_hearing_if_required",
        desc="Confirms Planning Commission public hearing occurred if rezoning/special permits were required; otherwise states not applicable.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.planning_hearing_statement or "Planning Commission hearing occurred if required, otherwise not applicable.",
        node=leaf1,
        sources=ensure_list(info.approval_doc_urls),
        additional_instruction="Check rezoning/SP cases and associated public hearing records.",
        extra_prerequisites=[doc_presence]
    )
    leaf2 = evaluator.add_leaf(
        id="public_notice_300_feet_if_required",
        desc="Confirms public notice mailed to property owners within 300 feet when required; otherwise states not applicable.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.public_notice_statement or "Public notice (within 300 feet) was mailed when required, otherwise not applicable.",
        node=leaf2,
        sources=ensure_list(info.approval_doc_urls),
        additional_instruction="Verify notice requirements and evidence of mailing.",
        extra_prerequisites=[doc_presence]
    )
    leaf3 = evaluator.add_leaf(
        id="metro_council_approval_if_required",
        desc="Confirms Metro Council approval obtained if required; otherwise states not applicable.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.council_approval_statement or "Metro Council approval was obtained if required; otherwise not applicable.",
        node=leaf3,
        sources=ensure_list(info.approval_doc_urls),
        additional_instruction="Check council votes/ordinances for approvals.",
        extra_prerequisites=[doc_presence]
    )


async def build_construction_timeline(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    info = data.construction_timeline or ConstructionTimeline()
    node = evaluator.add_parallel(
        id="construction_timeline",
        desc="Construction timeline and phases documentation.",
        parent=parent,
        critical=True
    )
    doc_presence = evaluator.add_custom_node(
        result=len(ensure_list(info.timeline_doc_urls)) > 0,
        id="construction_timeline_documentation_url",
        desc="Provides at least one supporting URL reference for the stated timeline/status information (per overall requirement to cite sources for each domain).",
        parent=node,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="construction_phases_documented",
        desc="Documents construction phases (pre-construction, site work, construction, finishing).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.phases_documented_statement or "Construction phases are documented.",
        node=leaf1,
        sources=ensure_list(info.timeline_doc_urls),
        additional_instruction="Look for phase descriptions or schedules.",
        extra_prerequisites=[doc_presence]
    )
    leaf2 = evaluator.add_leaf(
        id="current_project_status_identified",
        desc="Identifies current project status (e.g., planned, under construction, completed) for 2024–2025 context.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.current_status_statement or "Current project status is identified for 2024–2025.",
        node=leaf2,
        sources=ensure_list(info.timeline_doc_urls),
        additional_instruction="Confirm status like planned, under construction, or completed during 2024–2025.",
        extra_prerequisites=[doc_presence]
    )
    leaf3 = evaluator.add_leaf(
        id="duration_alignment_or_explanation",
        desc="States whether the described duration aligns with typical Nashville timelines (6–18+ months) OR provides an explanation for any difference.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.duration_alignment_statement or "The project's construction duration aligns with typical 6–18+ months or is explained.",
        node=leaf3,
        sources=ensure_list(info.timeline_doc_urls),
        additional_instruction="Assess timeline alignment or provided justification for deviations.",
        extra_prerequisites=[doc_presence]
    )


async def build_community_benefits(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    info = data.community_benefits or CommunityBenefits()
    node = evaluator.add_parallel(
        id="community_benefits",
        desc="Community benefits evaluation.",
        parent=parent,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="cba_or_amenities_presence",
        desc="Documents whether a Community Benefits Agreement and/or community amenities/benefits exist; if none found, states none/unknown.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.cba_or_amenities_statement or "Community Benefits Agreement or amenities presence is stated (or none/unknown).",
        node=leaf1,
        sources=ensure_list(info.community_benefits_doc_urls),
        additional_instruction="Verify claims of CBA or amenities such as public spaces, living wages, local hiring."
    )
    doc_ok = is_not_applicable(info.cba_or_amenities_statement) or len(ensure_list(info.community_benefits_doc_urls)) > 0
    evaluator.add_custom_node(
        result=doc_ok,
        id="community_benefits_documentation_url_if_applicable",
        desc="Provides documentation URL if a CBA/benefits are claimed/applicable.",
        parent=node,
        critical=True
    )


async def build_tod(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    info = data.transit_oriented_development or TransitOrientedDevelopment()
    node = evaluator.add_parallel(
        id="transit_oriented_development",
        desc="Transit-oriented development (TOD) evaluation.",
        parent=parent,
        critical=True
    )
    doc_presence = evaluator.add_custom_node(
        result=len(ensure_list(info.tod_doc_urls)) > 0,
        id="tod_documentation_url",
        desc="Provides documentation URL for transit proximity/TOD guideline discussion.",
        parent=node,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="transit_proximity_identified",
        desc="Identifies project proximity to public transit (or states not proximate/unknown).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.transit_proximity_statement or "Project proximity to public transit is identified or stated as unknown.",
        node=leaf1,
        sources=ensure_list(info.tod_doc_urls),
        additional_instruction="Check maps or transit descriptions (bus routes, stations).",
        extra_prerequisites=[doc_presence]
    )
    leaf2 = evaluator.add_leaf(
        id="tod_guidelines_if_applicable",
        desc="If TOD design guidelines are applicable, documents compliance; otherwise states not applicable.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.tod_guidelines_statement or "TOD guideline compliance is documented if applicable, otherwise 'not applicable'.",
        node=leaf2,
        sources=ensure_list(info.tod_doc_urls),
        additional_instruction="Verify mention of TOD overlays/guidelines and compliance details.",
        extra_prerequisites=[doc_presence]
    )


async def build_green_infrastructure(evaluator: Evaluator, parent, data: ProjectExtraction) -> None:
    info = data.green_infrastructure or GreenInfrastructure()
    node = evaluator.add_parallel(
        id="green_infrastructure",
        desc="Green infrastructure and beyond-code energy measures evaluation.",
        parent=parent,
        critical=True
    )
    doc_presence = evaluator.add_custom_node(
        result=len(ensure_list(info.green_infrastructure_doc_urls)) > 0,
        id="green_infrastructure_documentation_url",
        desc="Provides documentation URL for green infrastructure/energy measures information.",
        parent=node,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="green_infrastructure_practices",
        desc="Documents sustainable site design/green infrastructure practices per Nashville Green Infrastructure Master Plan (or states none/unknown).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.green_infrastructure_practices_statement or "Sustainable site design/green infrastructure practices are documented (or none/unknown).",
        node=leaf1,
        sources=ensure_list(info.green_infrastructure_doc_urls),
        additional_instruction="Look for rain gardens, permeable pavements, green roofs, etc.",
        extra_prerequisites=[doc_presence]
    )
    leaf2 = evaluator.add_leaf(
        id="energy_efficiency_beyond_code",
        desc="Documents any energy efficiency measures beyond code minimum (or states none/unknown).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=info.energy_efficiency_beyond_code_statement or "Energy efficiency measures beyond code are documented (or none/unknown).",
        node=leaf2,
        sources=ensure_list(info.green_infrastructure_doc_urls),
        additional_instruction="Verify measures beyond baseline codes (enhanced envelope, high-efficiency HVAC, etc.).",
        extra_prerequisites=[doc_presence]
    )


async def build_compliance_domains(evaluator: Evaluator, parent_node, data: ProjectExtraction) -> None:
    # Parent node aggregating all domains (critical, parallel)
    comp_node = evaluator.add_parallel(
        id="compliance_evaluation_domains",
        desc="Required regulatory/compliance evaluation across all specified domains (each domain must be addressed with findings and URLs as required).",
        parent=parent_node,
        critical=True
    )
    # Build individual domains
    await build_zoning_land_use(evaluator, comp_node, data)
    await build_parking_transportation(evaluator, comp_node, data)
    await build_sustainability_leed(evaluator, comp_node, data)
    await build_affordable_housing(evaluator, comp_node, data)
    await build_stormwater(evaluator, comp_node, data)
    await build_environmental_review(evaluator, comp_node, data)
    await build_ada(evaluator, comp_node, data)
    await build_permits(evaluator, comp_node, data)
    await build_public_approval(evaluator, comp_node, data)
    await build_construction_timeline(evaluator, comp_node, data)
    await build_community_benefits(evaluator, comp_node, data)
    await build_tod(evaluator, comp_node, data)
    await build_green_infrastructure(evaluator, comp_node, data)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Nashville mixed-use project compliance task.
    """
    # Initialize evaluator with a sequential root, then create a critical main node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Critical main node to enforce criticality across entire evaluation tree
    main_node = evaluator.add_sequential(
        id="overall_evaluation",
        desc="Identify exactly one qualifying major mixed-use Nashville project active in 2024–2025 and provide the required compliance evaluation across all specified domains with supporting URLs (where required).",
        parent=root,
        critical=True
    )

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_project(),
        template_class=ProjectExtraction,
        extraction_name="project_extraction"
    )

    # Build identification and compliance domains (sequential: compliance depends on identification)
    await build_project_identification(evaluator, main_node, extracted)
    await build_compliance_domains(evaluator, main_node, extracted)

    # Return evaluation summary
    return evaluator.get_summary()