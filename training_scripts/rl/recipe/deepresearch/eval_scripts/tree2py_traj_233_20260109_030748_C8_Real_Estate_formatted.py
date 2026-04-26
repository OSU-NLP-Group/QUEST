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
TASK_ID = "ca_mixed_use_dev"
TASK_DESCRIPTION = (
    "Identify a mixed-use development project in California that combines residential units with commercial or retail space "
    "and meets the following requirements: includes affordable housing units as part of its residential component; has achieved "
    "LEED Gold certification or higher; meets Energy Star certification standards (score of 75 or higher) or equivalent high "
    "energy efficiency performance; complies with ADA accessibility requirements including accessible entrances, doorways, "
    "and parking; includes fire safety suppression systems; provides adequate parking facilities for both residential and "
    "commercial components; has an identifiable developer or development company; has documented project scale (total square "
    "footage, number of units, or commercial space size); and has documented construction timeline or completion information. "
    "Provide the development name, developer, location details, project scale, and supporting URLs that verify each requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementInfo(BaseModel):
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class LocationInfo(BaseModel):
    city: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    address: Optional[str] = None
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class DeveloperInfo(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class LeedInfo(BaseModel):
    level: Optional[str] = None  # e.g., "Gold", "Platinum", "Silver"
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class EnergyInfo(BaseModel):
    program: Optional[str] = None  # e.g., "ENERGY STAR", "Net-zero", "Title 24 +15%"
    score: Optional[str] = None    # keep as string (e.g., "78", ">=75")
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ProjectScaleInfo(BaseModel):
    description: Optional[str] = None
    total_sqft: Optional[str] = None
    num_units: Optional[str] = None
    commercial_sqft: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TimelineInfo(BaseModel):
    description: Optional[str] = None
    completion_date: Optional[str] = None
    groundbreaking_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class DevelopmentExtraction(BaseModel):
    development_name: Optional[str] = None

    # Required groups
    location: Optional[LocationInfo] = None
    mixed_use: Optional[RequirementInfo] = None
    affordable: Optional[RequirementInfo] = None
    leed: Optional[LeedInfo] = None
    energy: Optional[EnergyInfo] = None
    ada: Optional[RequirementInfo] = None
    fire_safety: Optional[RequirementInfo] = None
    parking: Optional[RequirementInfo] = None
    developer: Optional[DeveloperInfo] = None
    project_scale: Optional[ProjectScaleInfo] = None
    timeline: Optional[TimelineInfo] = None

    # Optional groups
    sustainability: Optional[RequirementInfo] = None
    transit: Optional[RequirementInfo] = None
    amenities: Optional[RequirementInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_development() -> str:
    return """
    Extract structured information about a single mixed-use development project mentioned in the answer. The project must be in California and combine residential units with commercial or retail space.

    You must extract the following fields exactly as presented in the answer. Use strings for all fields to maximize compatibility. For each requirement, also extract all supporting URLs explicitly mentioned.

    1) development_name: The name of the development project.

    2) location:
       - city
       - county
       - state (should be "California" or "CA" in the answer)
       - address (if available)
       - description (the sentence(s) in the answer describing location)
       - urls (all URLs cited for the location)

    3) mixed_use:
       - description (text from the answer confirming the project combines residential and commercial/retail space)
       - urls (URLs cited for mixed-use)

    4) affordable:
       - description (text confirming affordable housing units are included; accept synonyms like "below-market-rate (BMR)", "income-restricted")
       - urls (URLs cited for affordable housing)

    5) leed:
       - level (e.g., "Gold", "Platinum", "Silver")
       - description (text describing the LEED certification)
       - urls (URLs citing LEED certification; USGBC pages or reputable sources preferred)

    6) energy:
       - program (e.g., "ENERGY STAR", "Net-zero", "Title 24", "CalGreen")
       - score (e.g., "75", ">=75", leave as string)
       - description (text describing energy performance)
       - urls (URLs citing energy performance)

    7) ada:
       - description (text confirming ADA accessibility including accessible entrances, doorways, and parking)
       - urls (URLs citing ADA features)

    8) fire_safety:
       - description (text confirming automatic fire sprinkler or suppression systems)
       - urls (URLs citing fire safety systems)

    9) parking:
       - description (text confirming adequate parking for both residential and commercial components; accept "shared parking", "garage", etc.)
       - urls (URLs citing parking facilities)

    10) developer:
        - name (developer or development company)
        - description (text in answer mentioning the developer)
        - urls (URLs citing developer information)

    11) project_scale:
        - description (text with scale, such as total square footage, number of units, commercial space size)
        - total_sqft (if present, otherwise null)
        - num_units (if present, otherwise null)
        - commercial_sqft (if present, otherwise null)
        - urls (URLs citing scale)

    12) timeline:
        - description (text with construction timeline or completion info)
        - completion_date (if present, otherwise null)
        - groundbreaking_date (if present, otherwise null)
        - urls (URLs citing timeline)

    13) sustainability (optional):
        - description (additional sustainable design features beyond basic LEED)
        - urls

    14) transit (optional):
        - description (text stating proximity to public transit)
        - urls

    15) amenities (optional):
        - description (community amenities or shared spaces for residents)
        - urls

    SPECIAL RULES FOR URL EXTRACTION:
    - Only extract URLs explicitly present in the answer (plain links or markdown). Do not invent URLs.
    - If a field has no URLs mentioned, return an empty array for its urls.
    - Return null for any missing field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


def _safe_text(value: Optional[str]) -> str:
    return value or ""


# --------------------------------------------------------------------------- #
# Verification functions (build subtrees and verify)                          #
# --------------------------------------------------------------------------- #
async def verify_location(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="Location_California",
        desc="Development must be located in California, United States",
        parent=parent_node,
        critical=True,
    )

    loc_urls = info.location.urls if (info.location and info.location.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(loc_urls),
        id="Reference_URL_Location",
        desc="Provide URL documenting the California location",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Location_Verified",
        desc="Development is confirmed to be located in California, United States",
        parent=node,
        critical=True,
    )
    dev_name = _safe_text(info.development_name)
    city = _safe_text(info.location.city if info.location else None)
    state = _safe_text(info.location.state if info.location else None)
    claim = f"The development '{dev_name}' is located in California, United States. City noted: '{city}', State noted in the answer: '{state}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=loc_urls,
        additional_instruction="Pass only if the provided sources clearly indicate the project is in California (CA). Allow common abbreviations like 'Los Angeles, CA'.",
    )


async def verify_mixed_use(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="Mixed_Use_Components",
        desc="Project must combine residential units with commercial or retail space in the same development",
        parent=parent_node,
        critical=True,
    )

    mu_urls = info.mixed_use.urls if (info.mixed_use and info.mixed_use.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(mu_urls),
        id="Reference_URL_Mixed_Use",
        desc="Provide URL documenting the mixed-use components",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Mixed_Use_Verified",
        desc="Development combines residential units with commercial or retail space",
        parent=node,
        critical=True,
    )
    claim = "The development combines residential units with commercial or retail space within the same project."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=mu_urls,
        additional_instruction="Pass only if sources explicitly indicate both a residential component and a commercial/retail component in the same development.",
    )


async def verify_affordable(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="Affordable_Housing_Inclusion",
        desc="Development must include affordable housing units as part of the residential component",
        parent=parent_node,
        critical=True,
    )

    aff_urls = info.affordable.urls if (info.affordable and info.affordable.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(aff_urls),
        id="Reference_URL_Affordable",
        desc="Provide URL documenting the affordable housing component",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Affordable_Units_Present",
        desc="Project includes designated affordable housing units",
        parent=node,
        critical=True,
    )
    claim = "The project includes designated affordable housing units (e.g., BMR, income-restricted, or set-aside)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=aff_urls,
        additional_instruction="Accept clear statements of affordable/BMR/income-restricted units or similar terminology from official or reputable sources.",
    )


async def verify_leed(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="LEED_Certification",
        desc="Development must achieve LEED Gold certification or higher (USGBC)",
        parent=parent_node,
        critical=True,
    )

    leed_urls = info.leed.urls if (info.leed and info.leed.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(leed_urls),
        id="Reference_URL_LEED",
        desc="Provide URL documenting LEED certification status",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="LEED_Gold_Or_Higher",
        desc="Project has achieved LEED Gold or LEED Platinum certification from USGBC",
        parent=node,
        critical=True,
    )
    level = _safe_text(info.leed.level if info.leed else None)
    claim = "This project is documented as LEED Gold or LEED Platinum certified."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=leed_urls,
        additional_instruction="Pass only if sources clearly indicate LEED Gold or LEED Platinum. LEED Silver or lower is not acceptable.",
    )


async def verify_energy(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="Energy_Performance",
        desc="Development must meet high energy efficiency standards",
        parent=parent_node,
        critical=True,
    )

    energy_urls = info.energy.urls if (info.energy and info.energy.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(energy_urls),
        id="Reference_URL_Energy",
        desc="Provide URL documenting energy performance standards",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Energy_Efficiency_Standard",
        desc="Project meets Energy Star certification standards (score of 75+) or equivalent high energy efficiency performance",
        parent=node,
        critical=True,
    )
    program = _safe_text(info.energy.program if info.energy else None)
    score = _safe_text(info.energy.score if info.energy else None)
    claim = "This project meets ENERGY STAR certification standards with a score of 75 or higher, or is clearly documented as having an equivalent high energy performance standard."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=energy_urls,
        additional_instruction=(
            "Accept as equivalent: documented ENERGY STAR score >=75, ENERGY STAR certified, net-zero or zero-net-energy (ZNE), "
            "top-quartile energy performance, or clearly stated Title 24 performance exceeding code by ~15% or more, from reputable sources."
        ),
    )


async def verify_ada(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="ADA_Accessibility",
        desc="Development must comply with ADA accessibility requirements",
        parent=parent_node,
        critical=True,
    )

    ada_urls = info.ada.urls if (info.ada and info.ada.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(ada_urls),
        id="Reference_URL_ADA",
        desc="Provide URL documenting accessibility features",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Accessible_Features",
        desc="Project includes ADA accessibility requirements including accessible entrances, doorways (32-48 inches), and parking",
        parent=node,
        critical=True,
    )
    claim = "The project complies with ADA accessibility requirements, including accessible entrances, accessible doorways, and accessible parking."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ada_urls,
        additional_instruction="Accept explicit ADA compliance statements or detailed accessible features consistent with ADA (entrances, doorways, parking).",
    )


async def verify_fire(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="Fire_Safety_Systems",
        desc="Development must include fire safety suppression systems",
        parent=parent_node,
        critical=True,
    )

    fs_urls = info.fire_safety.urls if (info.fire_safety and info.fire_safety.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(fs_urls),
        id="Reference_URL_Fire_Safety",
        desc="Provide URL documenting fire safety systems",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Fire_Safety_Present",
        desc="Project includes automatic fire sprinkler systems (as required by the constraints)",
        parent=node,
        critical=True,
    )
    claim = "The project includes automatic fire sprinkler systems or equivalent fire suppression systems."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=fs_urls,
        additional_instruction="Accept explicit documentation of automatic sprinklers or fire suppression systems on-site.",
    )


async def verify_parking(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="Parking_Facilities",
        desc="Development must provide adequate parking facilities for both residential and commercial components",
        parent=parent_node,
        critical=True,
    )

    pk_urls = info.parking.urls if (info.parking and info.parking.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(pk_urls),
        id="Reference_URL_Parking",
        desc="Provide URL documenting parking facilities",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Parking_Provided",
        desc="Project provides adequate parking facilities for both residential and commercial components",
        parent=node,
        critical=True,
    )
    claim = "The project provides adequate parking facilities serving both residents and commercial/retail users (e.g., shared garage, designated retail parking)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=pk_urls,
        additional_instruction="Accept documentation indicating parking facilities that serve both residential and commercial components.",
    )


async def verify_developer(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="Developer_Information",
        desc="Development must have an identifiable developer or development company",
        parent=parent_node,
        critical=True,
    )

    dev_urls = info.developer.urls if (info.developer and info.developer.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(dev_urls),
        id="Reference_URL_Developer",
        desc="Provide URL documenting developer information",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Developer_Identified",
        desc="Developer or development company name is documented",
        parent=node,
        critical=True,
    )
    dev_name = _safe_text(info.developer.name if info.developer else None)
    claim = f"The developer or development company for this project is '{dev_name}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=dev_urls,
        additional_instruction="Pass if the sources clearly name the developer/development company associated with the project.",
    )


async def verify_project_scale(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="Project_Scale",
        desc="Development must have documented project scale",
        parent=parent_node,
        critical=True,
    )

    sc_urls = info.project_scale.urls if (info.project_scale and info.project_scale.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(sc_urls),
        id="Reference_URL_Scale",
        desc="Provide URL documenting project scale",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Scale_Documented",
        desc="Project scale is documented (total square footage, number of residential units, and/or commercial space size)",
        parent=node,
        critical=True,
    )
    desc = _safe_text(info.project_scale.description if info.project_scale else None)
    claim = f"The project's scale is documented (e.g., total square footage, number of units, or commercial space size). Extracted description: '{desc}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sc_urls,
        additional_instruction="Pass if sources provide any clear scale metrics: total sq ft, unit counts, or commercial area size.",
    )


async def verify_timeline(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="Construction_Timeline",
        desc="Development must have documented construction timeline or completion information",
        parent=parent_node,
        critical=True,
    )

    tl_urls = info.timeline.urls if (info.timeline and info.timeline.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(tl_urls),
        id="Reference_URL_Timeline",
        desc="Provide URL documenting construction timeline or completion information",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Timeline_Documented",
        desc="Construction timeline or completion information is documented",
        parent=node,
        critical=True,
    )
    desc = _safe_text(info.timeline.description if info.timeline else None)
    claim = f"The project's construction timeline or completion information is documented. Extracted description: '{desc}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=tl_urls,
        additional_instruction="Pass if sources include dates or milestones indicating construction phases or completion.",
    )


async def verify_sustainability(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="Sustainable_Design_Features",
        desc="Development should include sustainable design features beyond basic LEED requirements",
        parent=parent_node,
        critical=False,
    )

    sus_urls = info.sustainability.urls if (info.sustainability and info.sustainability.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(sus_urls),
        id="Reference_URL_Sustainability",
        desc="Provide URL documenting sustainable design features",
        parent=node,
        critical=False,
    )

    leaf = evaluator.add_leaf(
        id="Advanced_Sustainability",
        desc="Project includes additional sustainable design features beyond basic LEED requirements",
        parent=node,
        critical=False,
    )
    desc = _safe_text(info.sustainability.description if info.sustainability else None)
    claim = f"The project includes sustainable design features beyond baseline LEED requirements. Extracted description: '{desc}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sus_urls,
        additional_instruction="Examples include on-site renewables, greywater systems, advanced envelope, battery storage, etc.",
    )


async def verify_transit(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="Transit_Accessibility",
        desc="Development should be located within reasonable distance of public transit (transit-oriented development)",
        parent=parent_node,
        critical=False,
    )

    tr_urls = info.transit.urls if (info.transit and info.transit.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(tr_urls),
        id="Reference_URL_Transit",
        desc="Provide URL documenting transit access",
        parent=node,
        critical=False,
    )

    leaf = evaluator.add_leaf(
        id="Near_Public_Transit",
        desc="Project is documented as being near public transit",
        parent=node,
        critical=False,
    )
    desc = _safe_text(info.transit.description if info.transit else None)
    claim = f"The project is located near public transit. Extracted description: '{desc}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=tr_urls,
        additional_instruction="Accept proximity to rail stations, frequent bus corridors, or official TOD designation.",
    )


async def verify_amenities(evaluator: Evaluator, parent_node, info: DevelopmentExtraction):
    node = evaluator.add_parallel(
        id="Community_Amenities",
        desc="Development should include community amenities or shared spaces for residents",
        parent=parent_node,
        critical=False,
    )

    am_urls = info.amenities.urls if (info.amenities and info.amenities.urls) else []
    evaluator.add_custom_node(
        result=_nonempty_urls(am_urls),
        id="Reference_URL_Amenities",
        desc="Provide URL documenting community amenities",
        parent=node,
        critical=False,
    )

    leaf = evaluator.add_leaf(
        id="Amenities_Present",
        desc="Project includes community amenities or shared spaces for residents",
        parent=node,
        critical=False,
    )
    desc = _safe_text(info.amenities.description if info.amenities else None)
    claim = f"The project includes community amenities or shared resident spaces. Extracted description: '{desc}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=am_urls,
        additional_instruction="Amenities examples: lounges, fitness centers, rooftop decks, parks, community rooms.",
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
    Evaluate an answer for the California mixed-use development task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates all requirement groups
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

    # IMPORTANT: Root should be non-critical to allow optional (non-critical) groups
    # while still gating by critical children. The rubric marks root critical,
    # but the framework forbids non-critical children under a critical parent.
    # Therefore we keep root non-critical and mark mandatory groups critical.
    root.desc = "A mixed-use development project in California meeting the stated sustainability, accessibility, safety, parking, documentation, and sourcing requirements"

    # Extract structured information from the answer
    extracted: DevelopmentExtraction = await evaluator.extract(
        prompt=prompt_extract_development(),
        template_class=DevelopmentExtraction,
        extraction_name="development_extraction",
    )

    # Record a compact summary of extracted identity info
    evaluator.add_custom_info(
        info={
            "development_name": extracted.development_name,
            "developer_name": extracted.developer.name if extracted.developer else None,
            "location_city": extracted.location.city if extracted.location else None,
            "location_state": extracted.location.state if extracted.location else None,
        },
        info_type="extracted_identity",
    )

    # Build mandatory requirement groups
    await verify_location(evaluator, root, extracted)
    await verify_mixed_use(evaluator, root, extracted)
    await verify_affordable(evaluator, root, extracted)
    await verify_leed(evaluator, root, extracted)
    await verify_energy(evaluator, root, extracted)
    await verify_ada(evaluator, root, extracted)
    await verify_fire(evaluator, root, extracted)
    await verify_parking(evaluator, root, extracted)
    await verify_developer(evaluator, root, extracted)
    await verify_project_scale(evaluator, root, extracted)
    await verify_timeline(evaluator, root, extracted)

    # Build optional (non-critical) groups
    await verify_sustainability(evaluator, root, extracted)
    await verify_transit(evaluator, root, extracted)
    await verify_amenities(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()