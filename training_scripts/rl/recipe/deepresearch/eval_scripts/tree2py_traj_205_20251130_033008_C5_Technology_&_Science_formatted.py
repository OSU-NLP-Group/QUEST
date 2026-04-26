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
TASK_ID = "tech_milestones_2024"
TASK_DESCRIPTION = """
Identify three major technology developments from 2024 based on the following criteria. For each development, provide the product/facility name, key specifications, and reference URL(s) from official sources.

1. Quantum Computing Chip:
Find a quantum computing chip that meets ALL of the following requirements:
- Announced in December 2024
- Contains exactly 105 qubits
- Achieved "below threshold" quantum error correction, demonstrating exponential error reduction as the system scales up
- Fabricated at a facility in California (specifically in the Santa Barbara region)

2. Semiconductor Manufacturing Facility:
Find a semiconductor manufacturing facility in the United States that meets ALL of the following requirements:
- Located in Phoenix, Arizona
- First fab (fabrication plant) uses N4 process technology
- First fab started high-volume production in Q4 2024 (October-December 2024)
- Total investment in the Arizona facility expanded to $165 billion

3. Spatial Computing Product:
Find a consumer spatial computing headset that meets ALL of the following requirements:
- Launched in the United States in February 2024
- Starting price of $3,499 (U.S.)
- Expanded to 9 new countries/regions in two waves during mid-2024:
  - First wave (June 28, 2024): China mainland, Hong Kong, Japan, and Singapore
  - Second wave (July 12, 2024): Australia, Canada, France, Germany, and United Kingdom

For each of the three items, provide: (1) the product/facility name, (2) a brief description confirming it meets the specified criteria, and (3) reference URL(s) to official announcements or credible sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ItemBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class QuantumChipInfo(ItemBase):
    announced_date: Optional[str] = None
    qubits: Optional[str] = None
    qec_note: Optional[str] = None
    fabrication_site: Optional[str] = None


class FacilityInfo(ItemBase):
    location: Optional[str] = None
    first_fab_process: Optional[str] = None
    hvm_start: Optional[str] = None
    total_investment: Optional[str] = None


class SpatialProductInfo(ItemBase):
    us_launch_date: Optional[str] = None
    starting_price: Optional[str] = None
    wave1_date: Optional[str] = None
    wave1_regions: List[str] = Field(default_factory=list)
    wave2_date: Optional[str] = None
    wave2_regions: List[str] = Field(default_factory=list)


class TechMilestonesExtraction(BaseModel):
    quantum_chip: Optional[QuantumChipInfo] = None
    semiconductor_facility: Optional[FacilityInfo] = None
    spatial_product: Optional[SpatialProductInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tech_milestones() -> str:
    return """
    Extract structured information for exactly three items from the answer: a quantum computing chip, a U.S. semiconductor manufacturing facility, and a consumer spatial computing headset. If multiple candidates are mentioned for a category, pick the one that best fits the constraints in the task description. Return null for a category if it is missing.

    For each category, extract the following fields from the answer exactly as written (do not infer):

    quantum_chip:
      - name: The chip's product/name/designation
      - description: A brief description provided in the answer that attempts to confirm the constraints
      - references: All URLs associated with this chip, preferably official/credible sources
      - announced_date: The announcement date as stated (e.g., "December 2024" or a specific date in Dec 2024)
      - qubits: The stated number of qubits (keep as string)
      - qec_note: Any text indicating "below threshold" quantum error correction with exponential error reduction
      - fabrication_site: The fabrication facility/site/location as stated (e.g., "Santa Barbara, California")

    semiconductor_facility:
      - name: The facility/campus/fab name/designation
      - description: A brief description provided in the answer confirming the constraints
      - references: All URLs associated with this facility, preferably official/credible sources
      - location: The stated location (city, state)
      - first_fab_process: The stated process node for the first fab (e.g., "N4", "4nm")
      - hvm_start: The stated high-volume production start time (e.g., "Q4 2024", a specific month between Oct–Dec 2024)
      - total_investment: The stated total investment figure for the Arizona facility (keep units, e.g., "$165 billion")

    spatial_product:
      - name: The consumer spatial computing headset name
      - description: A brief description provided in the answer confirming the constraints
      - references: All URLs associated with this product, preferably official/credible sources (e.g., manufacturer newsroom)
      - us_launch_date: The U.S. launch month/year or date (e.g., "February 2024")
      - starting_price: The starting price as stated (e.g., "$3,499")
      - wave1_date: The date for the first expansion wave (expected: June 28, 2024)
      - wave1_regions: The list of regions/countries in the first wave (expected: China mainland, Hong Kong, Japan, Singapore)
      - wave2_date: The date for the second expansion wave (expected: July 12, 2024)
      - wave2_regions: The list of regions/countries in the second wave (expected: Australia, Canada, France, Germany, United Kingdom)

    URL extraction rules:
      - Extract only URLs explicitly present in the answer (including markdown links). Do not invent any URL.
      - Return an empty list for references if none are provided.

    Return a single JSON matching the TechMilestonesExtraction schema.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _sources_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    return urls if (urls and len(urls) > 0) else None


# --------------------------------------------------------------------------- #
# Verification logic: Quantum Computing Chip                                  #
# --------------------------------------------------------------------------- #
async def verify_quantum_chip(
    evaluator: Evaluator,
    parent_node,
    qc: Optional[QuantumChipInfo],
) -> None:
    chip_node = evaluator.add_parallel(
        id="Quantum_Computing_Chip",
        desc="Quantum computing chip matching all specified constraints, with name, confirmation description, and official/credible reference URL(s).",
        parent=parent_node,
        critical=False,
    )

    # Critical existence: name provided
    name_ok = _nonempty(qc.name) if qc else False
    evaluator.add_custom_node(
        result=name_ok,
        id="Chip_Name_Provided",
        desc="Provides the quantum computing chip product/name/designation.",
        parent=chip_node,
        critical=True,
    )

    # Critical existence: references provided
    refs_ok = bool(qc and qc.references and len(qc.references) > 0)
    evaluator.add_custom_node(
        result=refs_ok,
        id="Chip_References_Provided",
        desc="Provides at least one valid reference URL from an official announcement/source or otherwise credible source supporting the chip claims.",
        parent=chip_node,
        critical=True,
    )

    # Critical: description confirms constraints (simple verify against answer)
    desc_leaf = evaluator.add_leaf(
        id="Chip_Confirmation_Description",
        desc="Provides a brief description explicitly confirming the chip meets the listed constraints (not just unrelated description).",
        parent=chip_node,
        critical=True,
    )
    claim_desc = (
        "In the provided answer, the chip's description explicitly confirms ALL of the following: "
        "announced in December 2024; exactly 105 qubits; achieved 'below threshold' quantum error correction with exponential error reduction as the system scales; "
        "and fabricated at a facility in the Santa Barbara region of California."
    )
    await evaluator.verify(
        claim=claim_desc,
        node=desc_leaf,
        additional_instruction="Judge only using the answer text. The description must explicitly mention or clearly affirm each constraint, not just generalities.",
    )

    # Prepare sources
    sources = _sources_or_none(qc.references if qc else None)
    chip_name = qc.name if qc and qc.name else "the chip"

    # Critical: Announced in December 2024
    announced_leaf = evaluator.add_leaf(
        id="Chip_Announced_Dec_2024",
        desc="Chip was announced in December 2024.",
        parent=chip_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{chip_name} was announced in December 2024.",
        node=announced_leaf,
        sources=sources,
        additional_instruction="Verify that the page explicitly indicates an announcement in December 2024 (any exact date within Dec 2024 is acceptable).",
    )

    # Critical: Exactly 105 qubits
    qubits_leaf = evaluator.add_leaf(
        id="Chip_Exactly_105_Qubits",
        desc="Chip contains exactly 105 qubits.",
        parent=chip_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{chip_name} contains exactly 105 qubits.",
        node=qubits_leaf,
        sources=sources,
        additional_instruction="Check the content for an explicit statement of 105 qubits (allow minor formatting differences, but not ranges or different numbers).",
    )

    # Critical: Below-threshold QEC with exponential error reduction
    qec_leaf = evaluator.add_leaf(
        id="Chip_Below_Threshold_QEC",
        desc="Chip achieved 'below threshold' quantum error correction demonstrating exponential error reduction as the system scales up.",
        parent=chip_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{chip_name} achieved below-threshold quantum error correction with exponential error reduction as the system scales up.",
        node=qec_leaf,
        sources=sources,
        additional_instruction="Look for phrases like 'below threshold', 'logical error decreases with code size', or 'exponential error suppression' tied to this chip.",
    )

    # Critical: Fabricated in Santa Barbara region, California
    fab_leaf = evaluator.add_leaf(
        id="Chip_Fabricated_Santa_Barbara_CA",
        desc="Chip was fabricated at a facility in California, specifically in the Santa Barbara region.",
        parent=chip_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{chip_name} was fabricated at a facility in California, in or around the Santa Barbara region (e.g., Goleta/Santa Barbara County).",
        node=fab_leaf,
        sources=sources,
        additional_instruction="Accept equivalent phrasing like 'fabricated in Santa Barbara/Goleta' or 'at our Santa Barbara facility'.",
    )


# --------------------------------------------------------------------------- #
# Verification logic: Semiconductor Manufacturing Facility                    #
# --------------------------------------------------------------------------- #
async def verify_semiconductor_facility(
    evaluator: Evaluator,
    parent_node,
    fac: Optional[FacilityInfo],
) -> None:
    fac_node = evaluator.add_parallel(
        id="Semiconductor_Manufacturing_Facility",
        desc="US semiconductor manufacturing facility matching all specified constraints, with name, confirmation description, and official/credible reference URL(s).",
        parent=parent_node,
        critical=False,
    )

    # Critical existence: name provided
    name_ok = _nonempty(fac.name) if fac else False
    evaluator.add_custom_node(
        result=name_ok,
        id="Facility_Name_Provided",
        desc="Provides the facility name/designation.",
        parent=fac_node,
        critical=True,
    )

    # Critical existence: references provided
    refs_ok = bool(fac and fac.references and len(fac.references) > 0)
    evaluator.add_custom_node(
        result=refs_ok,
        id="Facility_References_Provided",
        desc="Provides at least one valid reference URL from an official announcement/source or otherwise credible source supporting the facility claims.",
        parent=fac_node,
        critical=True,
    )

    # Critical: description confirms constraints
    desc_leaf = evaluator.add_leaf(
        id="Facility_Confirmation_Description",
        desc="Provides a brief description explicitly confirming the facility meets the listed constraints.",
        parent=fac_node,
        critical=True,
    )
    claim_desc = (
        "In the provided answer, the facility description explicitly confirms ALL of the following: "
        "located in Phoenix, Arizona; first fab uses N4 (4-nanometer) process technology; "
        "first fab started high-volume production in Q4 2024; and total investment expanded to $165 billion."
    )
    await evaluator.verify(
        claim=claim_desc,
        node=desc_leaf,
        additional_instruction="Judge using the answer text only. The description must explicitly address each constraint."
    )

    sources = _sources_or_none(fac.references if fac else None)
    fac_name = fac.name if fac and fac.name else "the facility"

    # Critical: Located in Phoenix, AZ
    loc_leaf = evaluator.add_leaf(
        id="Facility_Located_Phoenix_AZ",
        desc="Facility is located in Phoenix, Arizona.",
        parent=fac_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{fac_name} is located in Phoenix, Arizona.",
        node=loc_leaf,
        sources=sources,
        additional_instruction="Confirm the site/campus/facility location is Phoenix, AZ.",
    )

    # Critical: First fab uses N4 process
    proc_leaf = evaluator.add_leaf(
        id="Facility_First_Fab_Uses_N4",
        desc="Facility's first fab uses N4 (4-nanometer) process technology.",
        parent=fac_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The first fab at {fac_name} uses N4 (4-nanometer) process technology.",
        node=proc_leaf,
        sources=sources,
        additional_instruction="Look for 'N4', '4nm', or '4-nanometer' explicitly associated with the FIRST fab at this facility.",
    )

    # Critical: HVM in Q4 2024
    hvm_leaf = evaluator.add_leaf(
        id="Facility_HVM_Q4_2024",
        desc="Facility's first fab started high-volume production in Q4 2024 (October–December 2024).",
        parent=fac_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The first fab at {fac_name} started high-volume production in Q4 2024 (Oct–Dec 2024).",
        node=hvm_leaf,
        sources=sources,
        additional_instruction="Confirm explicit 'high-volume production' (or equivalent 'HVM') timing in Oct/Nov/Dec 2024.",
    )

    # Critical: Total investment expanded to $165B
    inv_leaf = evaluator.add_leaf(
        id="Facility_Investment_Expanded_165B",
        desc="Total investment in the Arizona facility expanded to $165 billion.",
        parent=fac_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The total investment in the Arizona facility expanded to $165 billion.",
        node=inv_leaf,
        sources=sources,
        additional_instruction="The figure must be $165 billion for the Arizona facility (not other amounts or other locations).",
    )


# --------------------------------------------------------------------------- #
# Verification logic: Spatial Computing Product                               #
# --------------------------------------------------------------------------- #
async def verify_spatial_product(
    evaluator: Evaluator,
    parent_node,
    sp: Optional[SpatialProductInfo],
) -> None:
    sp_node = evaluator.add_parallel(
        id="Spatial_Computing_Product",
        desc="Consumer spatial computing headset matching all specified constraints, with name, confirmation description, and official/credible reference URL(s).",
        parent=parent_node,
        critical=False,
    )

    # Critical existence: product name
    name_ok = _nonempty(sp.name) if sp else False
    evaluator.add_custom_node(
        result=name_ok,
        id="Product_Name_Provided",
        desc="Provides the spatial computing headset product name.",
        parent=sp_node,
        critical=True,
    )

    # Critical existence: references
    refs_ok = bool(sp and sp.references and len(sp.references) > 0)
    evaluator.add_custom_node(
        result=refs_ok,
        id="Product_References_Provided",
        desc="Provides at least one valid reference URL from the manufacturer's official announcement/newsroom or otherwise credible source supporting the product claims.",
        parent=sp_node,
        critical=True,
    )

    # Critical: description confirms constraints
    desc_leaf = evaluator.add_leaf(
        id="Product_Confirmation_Description",
        desc="Provides a brief description explicitly confirming the product meets the listed constraints.",
        parent=sp_node,
        critical=True,
    )
    claim_desc = (
        "In the provided answer, the headset description explicitly confirms ALL of the following: "
        "a consumer headset launched in the U.S. in February 2024; starting price $3,499; "
        "expansion wave on June 28, 2024 to (China mainland, Hong Kong, Japan, Singapore); "
        "and expansion wave on July 12, 2024 to (Australia, Canada, France, Germany, United Kingdom)."
    )
    await evaluator.verify(
        claim=claim_desc,
        node=desc_leaf,
        additional_instruction="Judge using the answer text only. The description must directly mention these specifics."
    )

    sources = _sources_or_none(sp.references if sp else None)
    prod_name = sp.name if sp and sp.name else "the product"

    # Critical: U.S. launch in Feb 2024
    us_launch_leaf = evaluator.add_leaf(
        id="Product_Consumer_Headset_US_Launch_Feb_2024",
        desc="Product is a consumer headset that launched in the United States in February 2024.",
        parent=sp_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{prod_name} is a consumer headset and launched in the United States in February 2024.",
        node=us_launch_leaf,
        sources=sources,
        additional_instruction="Confirm the product launched (became available) in the U.S. in Feb 2024. Accept explicit dates in Feb 2024.",
    )

    # Critical: Starting price $3,499
    price_leaf = evaluator.add_leaf(
        id="Product_Starting_Price_3499_USD",
        desc="Product starting price is $3,499 (U.S.).",
        parent=sp_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The starting price of {prod_name} is $3,499 (U.S.).",
        node=price_leaf,
        sources=sources,
        additional_instruction="Price must be $3,499 (USD), i.e., three thousand four hundred ninety-nine dollars.",
    )

    # Critical: Wave 1 expansion (June 28, 2024)
    wave1_leaf = evaluator.add_leaf(
        id="Product_Expansion_Wave1_June28_2024",
        desc="Product expanded to China mainland, Hong Kong, Japan, and Singapore on June 28, 2024.",
        parent=sp_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{prod_name} expanded to mainland China, Hong Kong, Japan, and Singapore on June 28, 2024.",
        node=wave1_leaf,
        sources=sources,
        additional_instruction="Allow minor phrasing like 'mainland China' vs 'China mainland'. Check the exact date June 28, 2024 and all four markets listed.",
    )

    # Critical: Wave 2 expansion (July 12, 2024)
    wave2_leaf = evaluator.add_leaf(
        id="Product_Expansion_Wave2_July12_2024",
        desc="Product expanded to Australia, Canada, France, Germany, and the United Kingdom on July 12, 2024.",
        parent=sp_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{prod_name} expanded to Australia, Canada, France, Germany, and the United Kingdom on July 12, 2024.",
        node=wave2_leaf,
        sources=sources,
        additional_instruction="Verify the exact date July 12, 2024 and that all five markets are included.",
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
    Evaluate an answer for the 2024 technology milestones task.
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
        default_model=model,
    )

    # Optional wrapper node to mirror rubric root
    main_node = evaluator.add_parallel(
        id="Technology_Milestones_2024",
        desc="Identify three 2024 technology developments (quantum chip, semiconductor facility, spatial computing product) and for each provide name, confirmation description, and reference URL(s) consistent with the given constraints.",
        parent=root,
        critical=False,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_tech_milestones(),
        template_class=TechMilestonesExtraction,
        extraction_name="tech_milestones_extraction",
    )

    # Add expected constraint summary as ground truth context (for transparency)
    evaluator.add_ground_truth({
        "quantum_chip_required": [
            "Announced in December 2024",
            "Exactly 105 qubits",
            "Below-threshold QEC with exponential error reduction",
            "Fabricated in Santa Barbara region, California"
        ],
        "semiconductor_facility_required": [
            "Located in Phoenix, Arizona",
            "First fab uses N4 (4nm) technology",
            "First fab HVM in Q4 2024",
            "Total investment expanded to $165B"
        ],
        "spatial_product_required": [
            "Consumer headset launched in US in February 2024",
            "Starting price $3,499 (U.S.)",
            "Expansion on June 28, 2024 to (China mainland, Hong Kong, Japan, Singapore)",
            "Expansion on July 12, 2024 to (Australia, Canada, France, Germany, United Kingdom)"
        ]
    }, gt_type="expected_constraints")

    # Build and run verification subtasks
    await verify_quantum_chip(evaluator, main_node, extracted.quantum_chip)
    await verify_semiconductor_facility(evaluator, main_node, extracted.semiconductor_facility)
    await verify_spatial_product(evaluator, main_node, extracted.spatial_product)

    # Return evaluation summary
    return evaluator.get_summary()