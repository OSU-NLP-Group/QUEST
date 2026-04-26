import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "atl_class_a_office"
TASK_DESCRIPTION = """
Identify a Class A office building in Atlanta, Georgia that meets ALL of the following requirements:

1. Location: Located in Midtown Atlanta, Buckhead, or Downtown Atlanta
2. Building Class: Classified as Class A office space
3. Size: Contains at least 500,000 square feet of office space
4. Sustainability: Has LEED certification (any level: Certified, Silver, Gold, or Platinum)
5. Parking: Provides at least 4 parking spaces per 1,000 square feet
6. Ceiling Height: Features minimum 9-foot floor-to-ceiling heights
7. Fitness Amenity: Includes an on-site fitness center or health club
8. Conference Facilities: Offers conference center, meeting rooms, or shared conference space
9. Covered Parking: Has covered parking or parking garage available
10. Building Age/Quality: Either built after 2000 OR underwent major renovation after 2010
11. Height: At least 20 stories tall
12. Technology: Has high-speed internet and modern technology infrastructure
13. Retail/Dining: Has retail shops, restaurants, or food service in-building or immediately adjacent
14. Management: Professionally managed with on-site property management

Provide the building name, address, and specific details demonstrating how it satisfies each requirement, with reference URLs supporting your answer.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BuildingInfoExtraction(BaseModel):
    # Identification
    name: Optional[str] = None
    address: Optional[str] = None

    # Core attributes/values (prefer strings for robustness)
    submarket: Optional[str] = None                          # e.g., Midtown Atlanta, Buckhead, Downtown Atlanta
    building_class: Optional[str] = None                     # e.g., Class A, Trophy Class A
    size_sqft: Optional[str] = None                          # e.g., "1,100,000 SF", "1.2M SF"
    leed_certification: Optional[str] = None                 # e.g., LEED Gold
    parking_ratio_per_1000: Optional[str] = None             # e.g., "4.0 / 1000", "4/1,000"
    min_ceiling_height_ft: Optional[str] = None              # e.g., "9'", "10 ft"
    fitness_amenity: Optional[str] = None                    # e.g., "On-site fitness center"
    conference_facilities: Optional[str] = None              # e.g., "Shared conference center"
    covered_parking: Optional[str] = None                    # e.g., "Parking garage"
    built_year: Optional[str] = None                         # e.g., "2005"
    major_renovation_year: Optional[str] = None              # e.g., "2018"
    stories: Optional[str] = None                            # e.g., "25"
    technology_infrastructure: Optional[str] = None          # e.g., "Fiber, redundant backbone"
    retail_dining: Optional[str] = None                      # e.g., "Food hall; in-building retail"
    professional_management: Optional[str] = None            # e.g., "On-site property management"

    # Source URLs per attribute (as explicitly cited in the answer)
    name_address_sources: List[str] = Field(default_factory=list)
    general_sources: List[str] = Field(default_factory=list)
    location_sources: List[str] = Field(default_factory=list)
    building_class_sources: List[str] = Field(default_factory=list)
    size_sources: List[str] = Field(default_factory=list)
    leed_sources: List[str] = Field(default_factory=list)
    parking_sources: List[str] = Field(default_factory=list)
    ceiling_sources: List[str] = Field(default_factory=list)
    fitness_sources: List[str] = Field(default_factory=list)
    conference_sources: List[str] = Field(default_factory=list)
    covered_parking_sources: List[str] = Field(default_factory=list)
    age_sources: List[str] = Field(default_factory=list)
    height_sources: List[str] = Field(default_factory=list)
    technology_sources: List[str] = Field(default_factory=list)
    retail_sources: List[str] = Field(default_factory=list)
    management_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_building_info() -> str:
    return """
Extract the details for a single Atlanta office building that the answer is presenting as satisfying ALL criteria.
Return a JSON with these fields (use strings for values; if missing, set to null; for each `*_sources`, include ONLY URLs explicitly cited in the answer text):

Identification:
- name: Building name
- address: Full address

Core attributes and values (strings are fine; don't coerce to numbers):
- submarket: The submarket (e.g., "Midtown Atlanta", "Buckhead", or "Downtown Atlanta")
- building_class: The stated class (e.g., "Class A" or "Trophy Class A")
- size_sqft: Total office size stated (e.g., "1,000,000 SF", "1.1M SF")
- leed_certification: The stated LEED level (e.g., "LEED Gold", "LEED Certified")
- parking_ratio_per_1000: The stated ratio (e.g., "4/1,000", "4.0 per 1,000 SF")
- min_ceiling_height_ft: The stated minimum floor-to-ceiling height (e.g., "9 ft", "10'")
- fitness_amenity: Brief phrase indicating on-site fitness/health club (or null)
- conference_facilities: Brief phrase indicating conference/meeting facilities (or null)
- covered_parking: Brief phrase indicating covered parking/garage (or null)
- built_year: The year built as stated (string)
- major_renovation_year: The year of major renovation/modernization as stated (string)
- stories: The number of stories stated (string)
- technology_infrastructure: Brief phrase indicating high-speed internet / modern tech infrastructure (or null)
- retail_dining: Brief phrase indicating retail/restaurant/food service in-building or adjacent (or null)
- professional_management: Brief phrase indicating professional management with on-site presence (or null)

Cited source URLs (only include URLs explicitly present in the answer; return [] if none):
- name_address_sources: []
- general_sources: []
- location_sources: []
- building_class_sources: []
- size_sources: []
- leed_sources: []
- parking_sources: []
- ceiling_sources: []
- fitness_sources: []
- conference_sources: []
- covered_parking_sources: []
- age_sources: []
- height_sources: []
- technology_sources: []
- retail_sources: []
- management_sources: []
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_non_empty(val: Optional[str]) -> bool:
    return isinstance(val, str) and val.strip() != ""


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def combine_sources(ex: BuildingInfoExtraction, attr_field: str) -> List[str]:
    attr_urls = getattr(ex, attr_field, []) or []
    general = ex.general_sources or []
    return _dedup_urls([*attr_urls, *general])


def parse_year_int(year_str: Optional[str]) -> Optional[int]:
    if not _is_non_empty(year_str):
        return None
    m = re.search(r"\b(19|20)\d{2}\b", year_str)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root, ex: BuildingInfoExtraction) -> None:
    # 1) Location (Midtown / Buckhead / Downtown)
    loc_parent = evaluator.add_sequential(
        id="Location_Requirement_main",
        desc="The building must be located in one of Atlanta's premier office submarkets: Midtown, Buckhead, or Downtown Atlanta",
        parent=root,
        critical=True,
    )
    loc_sources = combine_sources(ex, "location_sources")
    evaluator.add_custom_node(
        result=_is_non_empty(ex.submarket) and len(loc_sources) > 0,
        id="Location_Requirement_provided",
        desc="Location information and sources are provided in the answer",
        parent=loc_parent,
        critical=True,
    )
    loc_leaf = evaluator.add_leaf(
        id="Location_Requirement",
        desc="The building is in Midtown Atlanta, Buckhead, or Downtown Atlanta",
        parent=loc_parent,
        critical=True,
    )
    loc_claim = f"The building is located in {ex.submarket} in Atlanta, Georgia, which is one of Midtown Atlanta, Buckhead, or Downtown Atlanta."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=loc_sources,
        additional_instruction="Accept 'Midtown', 'Buckhead', or 'Downtown' as the three valid Atlanta submarkets. The page should clearly place the subject building in one of these.",
    )

    # 2) Building class = Class A
    cls_parent = evaluator.add_sequential(
        id="Building_Class_Verification_main",
        desc="The building must be classified as Class A office space",
        parent=root,
        critical=True,
    )
    cls_sources = combine_sources(ex, "building_class_sources")
    evaluator.add_custom_node(
        result=_is_non_empty(ex.building_class) and len(cls_sources) > 0,
        id="Building_Class_Verification_provided",
        desc="Building class information and sources are provided",
        parent=cls_parent,
        critical=True,
    )
    cls_leaf = evaluator.add_leaf(
        id="Building_Class_Verification",
        desc="The building is classified as Class A",
        parent=cls_parent,
        critical=True,
    )
    cls_claim = f"The building is classified as {ex.building_class} office space, which is Class A."
    await evaluator.verify(
        claim=cls_claim,
        node=cls_leaf,
        sources=cls_sources,
        additional_instruction="Look for phrases like 'Class A', 'Trophy Class A'. The source should explicitly describe the building as Class A or equivalent.",
    )

    # 3) Size >= 500,000 SF
    size_parent = evaluator.add_sequential(
        id="Minimum_Size_Requirement_main",
        desc="The building must contain at least 500,000 square feet of office space",
        parent=root,
        critical=True,
    )
    size_sources = combine_sources(ex, "size_sources")
    evaluator.add_custom_node(
        result=_is_non_empty(ex.size_sqft) and len(size_sources) > 0,
        id="Minimum_Size_Requirement_provided",
        desc="Size information and sources are provided",
        parent=size_parent,
        critical=True,
    )
    size_leaf = evaluator.add_leaf(
        id="Minimum_Size_Requirement",
        desc="The building has at least 500,000 square feet of office space",
        parent=size_parent,
        critical=True,
    )
    size_claim = f"The building contains {ex.size_sqft} of office space, and this meets or exceeds 500,000 square feet."
    await evaluator.verify(
        claim=size_claim,
        node=size_leaf,
        sources=size_sources,
        additional_instruction="Verify that the stated office size is at least 500,000 SF. Accept approximations like 0.5M SF or higher.",
    )

    # 4) LEED certification (any level)
    leed_parent = evaluator.add_sequential(
        id="LEED_Certification_main",
        desc="The building must have LEED certification at any level",
        parent=root,
        critical=True,
    )
    leed_sources = combine_sources(ex, "leed_sources")
    evaluator.add_custom_node(
        result=_is_non_empty(ex.leed_certification) and len(leed_sources) > 0,
        id="LEED_Certification_provided",
        desc="LEED certification information and sources are provided",
        parent=leed_parent,
        critical=True,
    )
    leed_leaf = evaluator.add_leaf(
        id="LEED_Certification",
        desc="The building has LEED certification",
        parent=leed_parent,
        critical=True,
    )
    leed_claim = f"The building has LEED {ex.leed_certification} certification."
    await evaluator.verify(
        claim=leed_claim,
        node=leed_leaf,
        sources=leed_sources,
        additional_instruction="Look for explicit 'LEED' mentions and level keywords like Certified, Silver, Gold, or Platinum (any version e.g., LEED-EB O+M, LEED-CS).",
    )

    # 5) Parking ratio >= 4 / 1,000 SF
    park_parent = evaluator.add_sequential(
        id="Parking_Ratio_Standard_main",
        desc="The building must provide at least 4 parking spaces per 1,000 square feet",
        parent=root,
        critical=True,
    )
    park_sources = combine_sources(ex, "parking_sources")
    evaluator.add_custom_node(
        result=_is_non_empty(ex.parking_ratio_per_1000) and len(park_sources) > 0,
        id="Parking_Ratio_Standard_provided",
        desc="Parking ratio information and sources are provided",
        parent=park_parent,
        critical=True,
    )
    park_leaf = evaluator.add_leaf(
        id="Parking_Ratio_Standard",
        desc="The building provides at least 4 spaces per 1,000 SF",
        parent=park_parent,
        critical=True,
    )
    park_claim = f"The building provides a parking ratio of {ex.parking_ratio_per_1000} per 1,000 square feet, meeting or exceeding 4 per 1,000."
    await evaluator.verify(
        claim=park_claim,
        node=park_leaf,
        sources=park_sources,
        additional_instruction="Look for formats like '4/1000', '4 per 1,000', or equivalent. Confirm the stated ratio is >= 4 per 1,000 SF.",
    )

    # 6) Ceiling heights >= 9 ft
    ceil_parent = evaluator.add_sequential(
        id="Ceiling_Height_Standard_main",
        desc="The building must have minimum 9-foot floor-to-ceiling heights",
        parent=root,
        critical=True,
    )
    ceil_sources = combine_sources(ex, "ceiling_sources")
    evaluator.add_custom_node(
        result=_is_non_empty(ex.min_ceiling_height_ft) and len(ceil_sources) > 0,
        id="Ceiling_Height_Standard_provided",
        desc="Ceiling height information and sources are provided",
        parent=ceil_parent,
        critical=True,
    )
    ceil_leaf = evaluator.add_leaf(
        id="Ceiling_Height_Standard",
        desc="The building features minimum 9-foot floor-to-ceiling heights",
        parent=ceil_parent,
        critical=True,
    )
    ceil_claim = f"The building features minimum {ex.min_ceiling_height_ft} floor-to-ceiling heights, which are at least 9 feet."
    await evaluator.verify(
        claim=ceil_claim,
        node=ceil_leaf,
        sources=ceil_sources,
        additional_instruction="Accept formats like '9 ft', '9''', or 'nine-foot'. The minimum height across office floors should be >= 9 ft.",
    )

    # 7) Fitness center amenity
    fit_parent = evaluator.add_sequential(
        id="Fitness_Center_Amenity_main",
        desc="The building must include an on-site fitness center or health club",
        parent=root,
        critical=True,
    )
    fit_sources = combine_sources(ex, "fitness_sources")
    evaluator.add_custom_node(
        result=_is_non_empty(ex.fitness_amenity) and len(fit_sources) > 0,
        id="Fitness_Center_Amenity_provided",
        desc="Fitness amenity information and sources are provided",
        parent=fit_parent,
        critical=True,
    )
    fit_leaf = evaluator.add_leaf(
        id="Fitness_Center_Amenity",
        desc="The building includes an on-site fitness center or health club",
        parent=fit_parent,
        critical=True,
    )
    fit_claim = "The building includes an on-site fitness center or health club."
    await evaluator.verify(
        claim=fit_claim,
        node=fit_leaf,
        sources=fit_sources,
        additional_instruction="Evidence should indicate an on-site fitness facility (gym, wellness center).",
    )

    # 8) Conference facilities
    conf_parent = evaluator.add_sequential(
        id="Conference_Facilities_main",
        desc="The building must provide conference center, meeting rooms, training facilities, or shared conference space",
        parent=root,
        critical=True,
    )
    conf_sources = combine_sources(ex, "conference_sources")
    evaluator.add_custom_node(
        result=_is_non_empty(ex.conference_facilities) and len(conf_sources) > 0,
        id="Conference_Facilities_provided",
        desc="Conference/meeting facilities information and sources are provided",
        parent=conf_parent,
        critical=True,
    )
    conf_leaf = evaluator.add_leaf(
        id="Conference_Facilities",
        desc="The building offers conference/meeting facilities",
        parent=conf_parent,
        critical=True,
    )
    conf_claim = "The building offers conference center, meeting rooms, training facilities, or shared conference space."
    await evaluator.verify(
        claim=conf_claim,
        node=conf_leaf,
        sources=conf_sources,
        additional_instruction="Look for terms like 'conference center', 'shared conference', 'meeting rooms', 'training room', etc.",
    )

    # 9) Covered parking / garage
    covp_parent = evaluator.add_sequential(
        id="Covered_Parking_Availability_main",
        desc="The building must offer covered parking or parking garage access",
        parent=root,
        critical=True,
    )
    covp_sources = combine_sources(ex, "covered_parking_sources")
    evaluator.add_custom_node(
        result=_is_non_empty(ex.covered_parking) and len(covp_sources) > 0,
        id="Covered_Parking_Availability_provided",
        desc="Covered parking / garage information and sources are provided",
        parent=covp_parent,
        critical=True,
    )
    covp_leaf = evaluator.add_leaf(
        id="Covered_Parking_Availability",
        desc="The building has covered parking or a parking garage",
        parent=covp_parent,
        critical=True,
    )
    covp_claim = "The building has covered parking or a parking garage."
    await evaluator.verify(
        claim=covp_claim,
        node=covp_leaf,
        sources=covp_sources,
        additional_instruction="Evidence should indicate a garage or covered parking (not solely surface parking).",
    )

    # 10) Building age/quality (built after 2000 OR major renovation after 2010)
    age_parent = evaluator.add_sequential(
        id="Building_Age_Quality_main",
        desc="The building must be either constructed after 2000 OR have undergone major renovation/modernization after 2010",
        parent=root,
        critical=True,
    )
    age_sources = combine_sources(ex, "age_sources")
    age_value_present = ( _is_non_empty(ex.built_year) or _is_non_empty(ex.major_renovation_year) ) and len(age_sources) > 0
    evaluator.add_custom_node(
        result=age_value_present,
        id="Building_Age_Quality_provided",
        desc="Building age/major renovation information and sources are provided",
        parent=age_parent,
        critical=True,
    )
    age_leaf = evaluator.add_leaf(
        id="Building_Age_Quality",
        desc="Building meets age/quality requirement (built after 2000 or major renovation after 2010)",
        parent=age_parent,
        critical=True,
    )
    by = parse_year_int(ex.built_year)
    ry = parse_year_int(ex.major_renovation_year)
    if ry is not None and ry >= 2011:
        age_claim = f"The building underwent a major renovation in {ex.major_renovation_year}, which is after 2010."
    elif by is not None and by >= 2001:
        age_claim = f"The building was built in {ex.built_year}, which is after 2000."
    else:
        # Fall back to a general disjunctive claim using whatever years are given
        if _is_non_empty(ex.major_renovation_year):
            age_claim = f"The building underwent a major renovation in {ex.major_renovation_year}, satisfying the requirement of renovation after 2010."
        elif _is_non_empty(ex.built_year):
            age_claim = f"The building was built in {ex.built_year}, satisfying the requirement of being built after 2000."
        else:
            age_claim = "The building meets at least one of the following: built after 2000 or major renovation after 2010."
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=age_sources,
        additional_instruction="Pass if EITHER the construction year is 2001 or later OR a major renovation year is 2011 or later, as supported by the source.",
    )

    # 11) Height: at least 20 stories
    hgt_parent = evaluator.add_sequential(
        id="Tower_Height_Requirement_main",
        desc="The building must be at least 20 stories tall",
        parent=root,
        critical=True,
    )
    hgt_sources = combine_sources(ex, "height_sources")
    evaluator.add_custom_node(
        result=_is_non_empty(ex.stories) and len(hgt_sources) > 0,
        id="Tower_Height_Requirement_provided",
        desc="Number of stories and sources are provided",
        parent=hgt_parent,
        critical=True,
    )
    hgt_leaf = evaluator.add_leaf(
        id="Tower_Height_Requirement",
        desc="The building is at least 20 stories tall",
        parent=hgt_parent,
        critical=True,
    )
    hgt_claim = f"The building is {ex.stories} stories tall, which is at least 20 stories."
    await evaluator.verify(
        claim=hgt_claim,
        node=hgt_leaf,
        sources=hgt_sources,
        additional_instruction="Confirm that the building has 20 or more stories.",
    )

    # 12) Technology infrastructure (high-speed internet / modern systems)
    tech_parent = evaluator.add_sequential(
        id="Technology_Infrastructure_main",
        desc="The building must have high-speed internet connectivity and modern technology infrastructure systems",
        parent=root,
        critical=True,
    )
    tech_sources = combine_sources(ex, "technology_sources")
    evaluator.add_custom_node(
        result=_is_non_empty(ex.technology_infrastructure) and len(tech_sources) > 0,
        id="Technology_Infrastructure_provided",
        desc="Technology infrastructure information and sources are provided",
        parent=tech_parent,
        critical=True,
    )
    tech_leaf = evaluator.add_leaf(
        id="Technology_Infrastructure",
        desc="The building has high-speed internet and modern technology infrastructure",
        parent=tech_parent,
        critical=True,
    )
    tech_claim = "The building has high-speed internet connectivity (e.g., fiber) and modern technology infrastructure."
    await evaluator.verify(
        claim=tech_claim,
        node=tech_leaf,
        sources=tech_sources,
        additional_instruction="Look for mentions like 'fiber', 'redundant backbone', 'carrier-neutral', 'high-speed internet', 'Wi‑Fi 6', or modern building tech systems.",
    )

    # 13) Retail / Dining access
    rtl_parent = evaluator.add_sequential(
        id="Retail_Dining_Access_main",
        desc="The building must have retail shops, restaurants, cafes, or food service options within the building or immediately adjacent",
        parent=root,
        critical=True,
    )
    rtl_sources = combine_sources(ex, "retail_sources")
    evaluator.add_custom_node(
        result=_is_non_empty(ex.retail_dining) and len(rtl_sources) > 0,
        id="Retail_Dining_Access_provided",
        desc="Retail/dining access information and sources are provided",
        parent=rtl_parent,
        critical=True,
    )
    rtl_leaf = evaluator.add_leaf(
        id="Retail_Dining_Access",
        desc="The building provides retail/restaurant/food service in-building or immediately adjacent",
        parent=rtl_parent,
        critical=True,
    )
    rtl_claim = "The building has retail shops, restaurants, cafes, or food service options within the building or immediately adjacent."
    await evaluator.verify(
        claim=rtl_claim,
        node=rtl_leaf,
        sources=rtl_sources,
        additional_instruction="Evidence should indicate in-building or immediately adjacent retail/food service options (e.g., lobby retail, on-site café, food hall, adjacent retail podium).",
    )

    # 14) Professional on-site property management
    mgt_parent = evaluator.add_sequential(
        id="Professional_Management_main",
        desc="The building must have professional property management with on-site management presence",
        parent=root,
        critical=True,
    )
    mgt_sources = combine_sources(ex, "management_sources")
    evaluator.add_custom_node(
        result=_is_non_empty(ex.professional_management) and len(mgt_sources) > 0,
        id="Professional_Management_provided",
        desc="Professional on-site management information and sources are provided",
        parent=mgt_parent,
        critical=True,
    )
    mgt_leaf = evaluator.add_leaf(
        id="Professional_Management",
        desc="The building is professionally managed with on-site management",
        parent=mgt_parent,
        critical=True,
    )
    mgt_claim = "The building is professionally managed and has on-site property management."
    await evaluator.verify(
        claim=mgt_claim,
        node=mgt_leaf,
        sources=mgt_sources,
        additional_instruction="Look for explicit mentions of 'on-site management' or professionally managed with an on-site office.",
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
    # Initialize evaluator (root: parallel aggregator per rubric)
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

    # Extract structured building info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_building_info(),
        template_class=BuildingInfoExtraction,
        extraction_name="building_info",
    )

    # Build verification tree according to rubric
    await build_verification_tree(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()