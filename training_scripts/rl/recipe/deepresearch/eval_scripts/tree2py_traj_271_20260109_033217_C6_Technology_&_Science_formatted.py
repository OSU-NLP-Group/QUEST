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
TASK_ID = "3nm_fabs_2027"
TASK_DESCRIPTION = """
Identify three semiconductor fabrication facilities worldwide that are currently producing, or are scheduled to begin volume production by the end of 2027, of chips using 3-nanometer (3nm) or more advanced process node technology. For each facility, provide: (1) The official facility name/designation and the operating company; (2) The specific city and country where the facility is located; (3) Confirmation that the facility uses 12-inch (300mm) wafer technology; (4) Confirmation that the facility has Extreme Ultraviolet (EUV) lithography capability; (5) The production timeline showing volume production has begun or will begin by December 31, 2027; (6) The monthly production capacity, which must be at least 20,000 wafers per month; (7) Supporting URL reference(s) for each facility. Additionally, verify that the three identified facilities are located in at least two different countries to ensure geographic diversity in the global 3nm manufacturing ecosystem.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityItem(BaseModel):
    """Information for a single semiconductor fabrication facility."""
    name: Optional[str] = None  # Official facility name/designation (e.g., "TSMC Fab 18")
    company: Optional[str] = None  # Operating company (e.g., "TSMC")
    city: Optional[str] = None
    country: Optional[str] = None
    process_node: Optional[str] = None  # e.g., "3nm", "N3E", "2nm", "1.4nm"
    wafer_technology: Optional[str] = None  # e.g., "300mm", "12-inch", "12” 300 mm"
    euv_capability: Optional[str] = None  # any indication string like "EUV", "ASML EUV"
    volume_production_timeline: Optional[str] = None  # e.g., "HVM 2025", "starts 2027"
    monthly_capacity: Optional[str] = None  # e.g., "25,000 wafers/month", "30k wpm"
    sources: List[str] = Field(default_factory=list)  # supporting URL references


class FacilitiesExtraction(BaseModel):
    facilities: List[FacilityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
    Extract up to THREE semiconductor fabrication facilities (a.k.a. fabs) mentioned in the answer that produce, or are scheduled to begin volume production by the end of 2027, chips using a 3-nanometer (3nm) or more advanced (smaller) process node.
    
    For each facility, extract the following fields exactly as written in the answer:
    - name: Official facility name/designation (e.g., "Fab 18", "Pyeongtaek Line 3", "Samsung Hwasung EUV line")
    - company: Operating company (e.g., "TSMC", "Samsung", "Intel")
    - city: City where the facility is located
    - country: Country where the facility is located
    - process_node: The process node (e.g., "3nm", "N3", "N3E", "2nm", "1.4nm"). Return the string as stated.
    - wafer_technology: Any mention indicating wafer size (e.g., "300mm", "12-inch"). Return the string as stated.
    - euv_capability: Any mention indicating EUV lithography capability (e.g., "EUV", "ASML NXE"). Return the string as stated.
    - volume_production_timeline: The statement or date indicating when volume production began or will begin (e.g., "HVM started in 2024", "volume starts in 2027").
    - monthly_capacity: The statement mentioning monthly capacity (e.g., "25,000 wafers/month", "30k wpm").
    - sources: A list of URLs cited in the answer and specifically associated with this facility. Only include actual URLs; do not infer or fabricate.

    Rules:
    - If the answer lists more than three facilities, include only the first three.
    - If any field is missing for a facility, set it to null (except sources should be an empty list).
    - Extract only what is explicitly present in the answer; do not invent information.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal_word(idx: int) -> str:
    mapping = {0: "First", 1: "Second", 2: "Third"}
    return mapping.get(idx, f"#{idx + 1}")


def facility_desc(idx: int) -> str:
    return f"{ordinal_word(idx)} facility meets all specified criteria and is fully sourced."


# --------------------------------------------------------------------------- #
# Verification functions for each facility                                    #
# --------------------------------------------------------------------------- #
async def verify_single_facility(
    evaluator: Evaluator,
    parent_node,
    facility: FacilityItem,
    idx: int,
) -> None:
    """
    Build verification nodes for a single facility and run verifications.
    All nodes under a critical parent must themselves be critical to satisfy framework constraints.
    """
    # Create facility parent node (critical to satisfy the critical parent constraint)
    facility_node = evaluator.add_parallel(
        id=f"Facility_{idx + 1}",
        desc=facility_desc(idx),
        parent=parent_node,
        critical=True,
    )

    # Add sources presence first (critical gating for other checks)
    sources_present = bool(facility.sources)
    evaluator.add_custom_node(
        result=sources_present,
        id=f"Facility_{idx + 1}_Sources",
        desc="Provides publicly available source URL(s) that support the facility claims.",
        parent=facility_node,
        critical=True,
    )

    # Name and Company verification
    name_company_node = evaluator.add_leaf(
        id=f"Facility_{idx + 1}_Name_And_Company",
        desc="Provides official facility name/designation AND operating company name.",
        parent=facility_node,
        critical=True,
    )
    name_val = facility.name or ""
    company_val = facility.company or ""
    claim_name_company = (
        f"The semiconductor fabrication facility '{name_val}' is operated by {company_val}."
    )
    await evaluator.verify(
        claim=claim_name_company,
        node=name_company_node,
        sources=facility.sources,
        additional_instruction=(
            "Verify that the sources explicitly mention both the official facility name/designation and "
            "the operating company, and that they are correctly associated. Accept reasonable name variants."
        ),
    )

    # Location verification
    location_node = evaluator.add_leaf(
        id=f"Facility_{idx + 1}_Location",
        desc="Provides the facility location including city AND country.",
        parent=facility_node,
        critical=True,
    )
    city_val = facility.city or ""
    country_val = facility.country or ""
    claim_location = f"The facility '{name_val}' is located in {city_val}, {country_val}."
    await evaluator.verify(
        claim=claim_location,
        node=location_node,
        sources=facility.sources,
        additional_instruction=(
            "Check that the source(s) state the specific city and country of the facility. "
            "Accept reasonable local language transliterations or minor spelling variants."
        ),
    )

    # Process node verification (3nm or more advanced)
    process_node_leaf = evaluator.add_leaf(
        id=f"Facility_{idx + 1}_Process_Node",
        desc="Facility produces (or will produce by the deadline) chips on a 3nm or more advanced (smaller) process node.",
        parent=facility_node,
        critical=True,
    )
    pn_val = facility.process_node or ""
    claim_process_node = (
        f"The facility '{name_val}' produces or is scheduled to produce chips using the {pn_val} process node, "
        "which is 3nm or more advanced (smaller)."
    )
    await evaluator.verify(
        claim=claim_process_node,
        node=process_node_leaf,
        sources=facility.sources,
        additional_instruction=(
            "Confirm that the facility's process node is 3nm-class or smaller (e.g., N3/N3E/3nm, 2nm, 1.4nm). "
            "If the source indicates 4nm, 5nm, or larger, this should be considered NOT meeting the requirement."
        ),
    )

    # 300mm wafer technology verification
    wafer_leaf = evaluator.add_leaf(
        id=f"Facility_{idx + 1}_300mm_Wafers",
        desc="Confirms use of 12-inch (300mm) wafer technology.",
        parent=facility_node,
        critical=True,
    )
    claim_wafer = (
        f"The facility '{name_val}' uses 300 mm (12-inch) wafer technology."
    )
    await evaluator.verify(
        claim=claim_wafer,
        node=wafer_leaf,
        sources=facility.sources,
        additional_instruction=(
            "Look for explicit mentions such as '300mm', '12-inch', or equivalent. "
            "If only 200mm or smaller legacy wafers are mentioned, this should fail."
        ),
    )

    # EUV capability verification
    euv_leaf = evaluator.add_leaf(
        id=f"Facility_{idx + 1}_EUV",
        desc="Confirms Extreme Ultraviolet (EUV) lithography capability.",
        parent=facility_node,
        critical=True,
    )
    euv_val = facility.euv_capability or ""
    claim_euv = (
        f"The facility '{name_val}' has EUV lithography capability (e.g., EUV scanners installed or planned for HVM). "
        f"{euv_val}"
    )
    await evaluator.verify(
        claim=claim_euv,
        node=euv_leaf,
        sources=facility.sources,
        additional_instruction=(
            "Check for explicit EUV capability mentions (e.g., ASML NXE systems, EUV lines). "
            "If only DUV is mentioned with no EUV capability, this should fail."
        ),
    )

    # Volume production timeline verification (by end of 2027)
    timeline_leaf = evaluator.add_leaf(
        id=f"Facility_{idx + 1}_Volume_Production_Timeline",
        desc="Shows volume production has begun or will begin by December 31, 2027.",
        parent=facility_node,
        critical=True,
    )
    timeline_val = facility.volume_production_timeline or ""
    claim_timeline = (
        f"Volume production at the facility '{name_val}' has started or is scheduled to start by December 31, 2027. {timeline_val}"
    )
    await evaluator.verify(
        claim=claim_timeline,
        node=timeline_leaf,
        sources=facility.sources,
        additional_instruction=(
            "Accept any credible source indicating HVM/volume production start on or before 2027-12-31. "
            "If sources indicate 2028 or later, or only pilot/R&D without a committed 2027 timeline, this should fail."
        ),
    )

    # Monthly capacity verification (≥ 20,000 wafers/month)
    capacity_leaf = evaluator.add_leaf(
        id=f"Facility_{idx + 1}_Monthly_Capacity",
        desc="States monthly production capacity AND confirms it is ≥ 20,000 wafers/month.",
        parent=facility_node,
        critical=True,
    )
    capacity_val = facility.monthly_capacity or ""
    claim_capacity = (
        f"The facility '{name_val}' has a monthly capacity of at least 20,000 wafers per month. {capacity_val}"
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_leaf,
        sources=facility.sources,
        additional_instruction=(
            "Verify that the stated monthly capacity (current or planned by 2027) is ≥ 20,000 wafers/month. "
            "Accept 'k wafers/month' notation (e.g., '20k wpm' == 20,000). If the only figure is below 20k, fail."
        ),
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
    Evaluate an answer for the global 3nm-by-2027 fabs identification task.
    """
    # Initialize evaluator with sequential root (as specified in rubric)
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
        default_model=model,
    )

    # Extract facilities from the answer
    extracted_facilities = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction",
    )

    # Select up to the first three facilities; pad if fewer to always build 3
    facilities: List[FacilityItem] = list(extracted_facilities.facilities[:3])
    while len(facilities) < 3:
        facilities.append(FacilityItem())

    # Add "Facilities" node (critical, parallel)
    facilities_parent = evaluator.add_parallel(
        id="Facilities",
        desc="Three facilities are provided and each is evaluated against the required constraints.",
        parent=root,
        critical=True,  # critical as per rubric
    )

    # Build verification sub-trees for each facility
    for i, facility in enumerate(facilities):
        await verify_single_facility(evaluator, facilities_parent, facility, i)

    # Geographic diversity check (critical)
    geo_parent = evaluator.add_parallel(
        id="Geographic_Diversity",
        desc="Verify the three facilities are located in at least two different countries.",
        parent=root,
        critical=True,
    )

    # Compute distinct countries among provided facilities
    countries = [f.country.strip() for f in facilities if f.country and f.country.strip()]
    distinct_country_count = len(set(countries))
    geo_diversity_result = distinct_country_count >= 2

    evaluator.add_custom_node(
        result=geo_diversity_result,
        id="At_Least_Two_Countries",
        desc="The set of three facilities spans ≥ 2 distinct countries.",
        parent=geo_parent,
        critical=True,
    )

    # Add some custom info for transparency
    evaluator.add_custom_info(
        info={
            "extracted_facility_count": len([f for f in facilities if any([
                f.name, f.company, f.city, f.country, f.process_node, f.wafer_technology,
                f.euv_capability, f.volume_production_timeline, f.monthly_capacity, f.sources
            ])]),
            "countries_list": countries,
            "distinct_country_count": distinct_country_count,
        },
        info_type="extraction_summary",
        info_name="facility_extraction_stats"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()