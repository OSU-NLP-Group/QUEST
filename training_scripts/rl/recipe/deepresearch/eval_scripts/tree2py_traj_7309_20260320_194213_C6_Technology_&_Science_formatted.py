import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "intel_18a_arizona_facility"
TASK_DESCRIPTION = """
Identify the Intel semiconductor fabrication facility that meets ALL of the following criteria:

1. Location and Classification:
   - Located in Chandler, Arizona, United States
   - Part of Intel's Ocotillo campus
   - The fifth high-volume fabrication facility at this campus

2. Process Technology:
   - Manufactures chips using Intel's 18A process node
   - The 18A process must be a 2nm-class process node
   - The 18A process must be the first 2nm-class semiconductor manufacturing in the United States
   - The 18A process must utilize two specific key technologies: (a) RibbonFET gate-all-around transistor technology, and (b) PowerVia backside power delivery technology

3. Production Specifications:
   - Became fully operational for 18A production in 2025
   - Has a production capacity of 10,000 wafer starts per week

4. Manufactured Product:
   - Manufactures Panther Lake processors (also known as Intel Core Ultra Series 3)
   - This product was announced at CES 2026 on January 5, 2026
   - Systems containing this product became available starting January 27, 2026

Provide the following information:
- The name/designation of the facility
- Confirmation of all location and classification details
- Confirmation of all process technology specifications
- Confirmation of all production specifications
- Confirmation of all product details including announcement and availability dates
- Reference URLs for each major category of information (facility identification, process technology, production specifications, and product details)
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class FacilityInfo(BaseModel):
    name: Optional[str] = None
    location_text: Optional[str] = None  # e.g., "Chandler, Arizona, USA" and "Ocotillo campus"
    classification_text: Optional[str] = None  # e.g., "fifth high-volume fab at Ocotillo"
    urls: List[str] = Field(default_factory=list)  # reference URLs for facility identity/location/classification


class ProcessTechInfo(BaseModel):
    node_name: Optional[str] = None  # e.g., "Intel 18A"
    class_text: Optional[str] = None  # e.g., "2nm-class"
    us_first_text: Optional[str] = None  # e.g., "first 2nm-class manufacturing in the US"
    ribbonfet_text: Optional[str] = None  # mentions RibbonFET gate-all-around
    powervia_text: Optional[str] = None  # mentions PowerVia backside power
    process_urls: List[str] = Field(default_factory=list)  # URLs tying the facility to 18A / 2nm-class / US-first
    technology_urls: List[str] = Field(default_factory=list)  # URLs confirming RibbonFET and PowerVia for 18A


class ProductionSpecs(BaseModel):
    operational_timeline_text: Optional[str] = None  # e.g., "fully operational for 18A in 2025"
    capacity_text: Optional[str] = None  # e.g., "10,000 wafer starts per week"
    urls: List[str] = Field(default_factory=list)  # URLs confirming timeline and capacity


class ProductInfo(BaseModel):
    product_name: Optional[str] = None  # e.g., "Panther Lake"
    also_known_as_text: Optional[str] = None  # e.g., "Intel Core Ultra Series 3"
    announcement_event_text: Optional[str] = None  # e.g., "CES 2026"
    announcement_date_text: Optional[str] = None  # e.g., "January 5, 2026"
    availability_date_text: Optional[str] = None  # e.g., "January 27, 2026"
    product_urls: List[str] = Field(default_factory=list)  # URLs tying product + 18A + facility
    announcement_urls: List[str] = Field(default_factory=list)  # URLs for CES announcement/date
    availability_urls: List[str] = Field(default_factory=list)  # URLs for availability date


class Intel18AArizonaExtraction(BaseModel):
    facility: Optional[FacilityInfo] = None
    process: Optional[ProcessTechInfo] = None
    production: Optional[ProductionSpecs] = None
    product: Optional[ProductInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_structured() -> str:
    return """
    Extract structured information about the Intel fabrication facility in Arizona and the associated process, production, and product details as presented in the answer.

    You must return a JSON object with these nested fields:

    facility:
      - name: The exact facility name/designation provided in the answer (e.g., "Fab 52", "Fab 52 at Ocotillo", "Intel Fab 52", "Fab 52/62" etc.). Use the main single facility name if multiple are mentioned; if ambiguous, choose the one the answer ultimately identifies as meeting all criteria.
      - location_text: The text from the answer that states (or implies) the facility is in Chandler, Arizona, United States and part of Intel's Ocotillo campus. Keep as a short string.
      - classification_text: The text from the answer that states it is the fifth high-volume fabrication facility at the Ocotillo campus.
      - urls: All reference URLs the answer provides that support the facility identity, location, and "fifth high-volume" classification. Include both Intel official and credible tech news links if present.

    process:
      - node_name: The process node used at this facility (expected "Intel 18A" or similar).
      - class_text: The text from the answer that states Intel 18A is a 2nm-class process.
      - us_first_text: The text that states Intel 18A is the first 2nm-class semiconductor manufacturing in the United States.
      - ribbonfet_text: The text that states Intel 18A uses RibbonFET gate-all-around transistor technology.
      - powervia_text: The text that states Intel 18A uses PowerVia (backside power delivery) technology.
      - process_urls: All URLs that support the node name, 2nm-class classification, and US-first milestone (especially tying 18A to the identified facility).
      - technology_urls: All URLs that confirm RibbonFET and PowerVia are key features of 18A.

    production:
      - operational_timeline_text: The text that states the facility became fully operational for 18A production in 2025.
      - capacity_text: The text that states the facility's capacity is 10,000 wafer starts per week.
      - urls: All URLs that support the operational timeline and capacity.

    product:
      - product_name: The product manufactured at this facility using 18A (expected "Panther Lake").
      - also_known_as_text: The alias/codename such as "Intel Core Ultra Series 3".
      - announcement_event_text: The event name where it was announced (expected "CES 2026").
      - announcement_date_text: The announcement date (expected "January 5, 2026").
      - availability_date_text: The availability date for systems (expected "January 27, 2026").
      - product_urls: URLs that confirm the product identity and that it is manufactured on Intel 18A at this facility (Ocotillo/Chandler).
      - announcement_urls: URLs that confirm the CES 2026 announcement and the Jan 5, 2026 date.
      - availability_urls: URLs that confirm systems availability starting Jan 27, 2026.

    Rules:
    - Extract exactly what the answer states. Do not invent data.
    - Return null for any missing field. For URL lists, return an empty list if none were provided.
    - For URLs, extract only actual URLs explicitly present in the answer text (including those formatted as markdown links).
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _safe_list(x: Optional[List[str]]) -> List[str]:
    return x if isinstance(x, list) else []


def _merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _or_unknown(x: Optional[str], fallback: str = "the facility") -> str:
    s = (x or "").strip()
    return s if s else fallback


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_facility(evaluator: Evaluator, parent_node, data: Intel18AArizonaExtraction) -> None:
    fac = data.facility or FacilityInfo()
    fac_name = _or_unknown(fac.name, "the facility")
    fac_urls = _safe_list(fac.urls)

    # Facility Identification (critical parallel group)
    node = evaluator.add_parallel(
        id="Facility_Identification",
        desc="Correctly identify the Intel fabrication facility that meets the specified location and classification criteria.",
        parent=parent_node,
        critical=True
    )

    # Geographic Location
    leaf = evaluator.add_leaf(
        id="Geographic_Location",
        desc="The facility must be located in Chandler, Arizona, United States, specifically at Intel's Ocotillo campus.",
        parent=node,
        critical=True
    )
    claim = f"The facility named '{fac_name}' is located in Chandler, Arizona, United States and is part of Intel's Ocotillo campus."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=fac_urls,
        additional_instruction="Accept reasonable variants of location phrasing. The page must clearly indicate Chandler, Arizona and Ocotillo campus affiliation."
    )

    # Facility Sequence (fifth high-volume fab at Ocotillo)
    leaf = evaluator.add_leaf(
        id="Facility_Sequence",
        desc="The facility must be identified as the fifth high-volume fabrication facility at the Ocotillo campus.",
        parent=node,
        critical=True
    )
    claim = f"The facility '{fac_name}' is identified as the fifth high-volume fabrication facility at Intel's Ocotillo campus."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=fac_urls,
        additional_instruction="Look for wording like 'fifth high-volume fab at Ocotillo' or equivalent phrasing. Allow minor paraphrases."
    )

    # Facility Reference URL credibility + confirmation
    leaf = evaluator.add_leaf(
        id="Facility_Reference_URL",
        desc="Provide a reference URL from an official Intel announcement or credible technology news source that confirms the facility's identity, location, and classification.",
        parent=node,
        critical=True
    )
    claim = (
        "At least one of the provided URLs is either an official Intel source (intel.com domain) or a reputable technology news outlet, "
        "and explicitly confirms the facility's identity (name/designation), its location at the Ocotillo campus in Chandler, Arizona, "
        "and that it is the fifth high-volume fabrication facility at that campus."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=fac_urls,
        additional_instruction="When judging credibility, consider intel.com official pages and well-known tech outlets (e.g., AnandTech, Tom's Hardware, The Verge, EE Times, etc.). The page must explicitly mention all three aspects."
    )


async def verify_process(evaluator: Evaluator, parent_node, data: Intel18AArizonaExtraction) -> None:
    fac = data.facility or FacilityInfo()
    fac_name = _or_unknown(fac.name, "the facility")

    proc = data.process or ProcessTechInfo()
    proc_urls = _safe_list(proc.process_urls)
    tech_urls = _safe_list(proc.technology_urls)
    all_proc_urls = _merge_urls(proc_urls, tech_urls)

    node = evaluator.add_parallel(
        id="Process_Technology",
        desc="Correctly identify and describe the semiconductor process node used at this facility and its key technological features.",
        parent=parent_node,
        critical=True
    )

    # Process Node Name (must be Intel 18A used at this facility)
    leaf = evaluator.add_leaf(
        id="Process_Node_Name",
        desc="The process node must be identified as Intel 18A.",
        parent=node,
        critical=True
    )
    claim = f"The facility '{fac_name}' manufactures chips using the Intel 18A process node."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=proc_urls,
        additional_instruction="The source should tie Intel 18A specifically to the identified facility (Ocotillo/Chandler Arizona). General 18A info without linking to the facility is insufficient."
    )

    # Process Class Classification (2nm-class)
    leaf = evaluator.add_leaf(
        id="Process_Class_Classification",
        desc="The process must be identified as a 2nm-class process node.",
        parent=node,
        critical=True
    )
    claim = "Intel 18A is a 2nm-class semiconductor process node."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=all_proc_urls,
        additional_instruction="Accept minor wording variations like '2 nm class' or 'two-nanometer-class'."
    )

    # US Manufacturing Milestone (first 2nm-class in US)
    leaf = evaluator.add_leaf(
        id="US_Manufacturing_Milestone",
        desc="The process must be noted as the first 2nm-class semiconductor manufacturing in the United States.",
        parent=node,
        critical=True
    )
    claim = "Intel 18A represents the first 2nm-class semiconductor manufacturing in the United States."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=proc_urls,
        additional_instruction="The page should explicitly state the 'first in the US' milestone or equivalent wording."
    )

    # Key Technologies (RibbonFET and PowerVia)
    key_node = evaluator.add_parallel(
        id="Key_Technologies",
        desc="Identify the two primary technological innovations in the 18A process.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="RibbonFET_Technology",
        desc="The process must utilize RibbonFET gate-all-around transistor technology.",
        parent=key_node,
        critical=True
    )
    claim = "Intel 18A uses RibbonFET gate-all-around transistor technology."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=tech_urls,
        additional_instruction="The page should explicitly mention 'RibbonFET' and describe it as gate-all-around transistor technology for 18A."
    )

    leaf = evaluator.add_leaf(
        id="PowerVia_Technology",
        desc="The process must utilize PowerVia backside power delivery technology.",
        parent=key_node,
        critical=True
    )
    claim = "Intel 18A uses PowerVia backside power delivery technology."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=tech_urls,
        additional_instruction="The page should explicitly mention 'PowerVia' as a backside power delivery network for 18A."
    )

    leaf = evaluator.add_leaf(
        id="Technology_Reference_URL",
        desc="Provide a reference URL that confirms these specific technological features of the 18A process.",
        parent=key_node,
        critical=True
    )
    claim = "At least one of the provided sources explicitly states that Intel 18A uses both RibbonFET (gate-all-around) and PowerVia (backside power)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=tech_urls,
        additional_instruction="The same page can mention both features, or multiple pages can each confirm one; in either case, the claim should be considered supported if the collection of URLs together confirms both features."
    )

    # Process Reference URL (name + class + US-first)
    leaf = evaluator.add_leaf(
        id="Process_Reference_URL",
        desc="Provide a reference URL that confirms the process node name, its classification as 2nm-class, and its status as first in the US.",
        parent=node,
        critical=True
    )
    claim = (
        "At least one of the provided URLs explicitly confirms the Intel 18A process name, that it is a 2nm-class node, "
        "and that it is the first 2nm-class semiconductor manufacturing in the United States."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=proc_urls,
        additional_instruction="A single authoritative page that mentions all three points is preferred; otherwise, confirm that the provided set of URLs collectively covers the three points unambiguously."
    )


async def verify_production(evaluator: Evaluator, parent_node, data: Intel18AArizonaExtraction) -> None:
    fac = data.facility or FacilityInfo()
    fac_name = _or_unknown(fac.name, "the facility")
    prod = data.production or ProductionSpecs()
    prod_urls = _safe_list(prod.urls)

    node = evaluator.add_parallel(
        id="Production_Specifications",
        desc="Provide the operational timeline and production capacity specifications for this facility.",
        parent=parent_node,
        critical=True
    )

    # Operational Timeline (fully operational for 18A in 2025)
    leaf = evaluator.add_leaf(
        id="Operational_Timeline",
        desc="The facility must have become fully operational for 18A production in 2025.",
        parent=node,
        critical=True
    )
    claim = f"The facility '{fac_name}' became fully operational for Intel 18A production in 2025."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=prod_urls,
        additional_instruction="The wording should clearly indicate 2025 as the year of full operational status for 18A at the facility."
    )

    # Production Capacity (10,000 wafer starts per week)
    leaf = evaluator.add_leaf(
        id="Production_Capacity",
        desc="The facility's production capacity must be stated as 10,000 wafer starts per week.",
        parent=node,
        critical=True
    )
    claim = f"The facility '{fac_name}' has a production capacity of 10,000 wafer starts per week."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=prod_urls,
        additional_instruction="Look for numeric capacity statements like '10,000 wafer starts per week' (allow minor formatting differences like commas/spaces)."
    )

    # Production Reference URL (both timeline and capacity)
    leaf = evaluator.add_leaf(
        id="Production_Reference_URL",
        desc="Provide a reference URL that confirms the operational timeline and production capacity specifications.",
        parent=node,
        critical=True
    )
    claim = "At least one of the provided URLs confirms both that the facility reached full 18A operations in 2025 and that its capacity is 10,000 wafer starts per week."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=prod_urls,
        additional_instruction="If a single page confirms both items, that is sufficient; otherwise, confirm that the collection of URLs unambiguously covers both."
    )


async def verify_product(evaluator: Evaluator, parent_node, data: Intel18AArizonaExtraction) -> None:
    fac = data.facility or FacilityInfo()
    fac_name = _or_unknown(fac.name, "the facility")

    prod = data.product or ProductInfo()
    product_urls = _safe_list(prod.product_urls)
    announce_urls = _safe_list(prod.announcement_urls)
    avail_urls = _safe_list(prod.availability_urls)

    product_name = _or_unknown(prod.product_name, "Panther Lake")
    aka = _or_unknown(prod.also_known_as_text, "Intel Core Ultra Series 3")

    node = evaluator.add_parallel(
        id="Manufactured_Product",
        desc="Identify the first consumer product manufactured at this facility using the 18A process, including its announcement and availability details.",
        parent=parent_node,
        critical=True
    )

    # Product Identification (Panther Lake / Intel Core Ultra Series 3)
    leaf = evaluator.add_leaf(
        id="Product_Identification",
        desc="The product must be identified as Panther Lake, also known as Intel Core Ultra Series 3 processors.",
        parent=node,
        critical=True
    )
    claim = f"The first consumer product manufactured using Intel 18A at '{fac_name}' is '{product_name}', also known as '{aka}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=product_urls,
        additional_instruction="The page(s) should tie Panther Lake (Intel Core Ultra Series 3) to Intel 18A production at the identified facility."
    )

    # Announcement details (CES 2026, Jan 5, 2026)
    ann_node = evaluator.add_parallel(
        id="Announcement_Details",
        desc="Provide accurate details about when and where the product was announced.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Event_and_Date",
        desc="The product must have been announced at CES 2026 on January 5, 2026.",
        parent=ann_node,
        critical=True
    )
    claim = f"'{product_name}' (Intel Core Ultra Series 3) was announced at CES 2026 on January 5, 2026."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=announce_urls,
        additional_instruction="Confirm both the event (CES 2026) and the date (January 5, 2026). Accept minor date format variations."
    )

    leaf = evaluator.add_leaf(
        id="Announcement_Reference_URL",
        desc="Provide a reference URL that confirms the announcement event and date.",
        parent=ann_node,
        critical=True
    )
    claim = "At least one of the provided URLs explicitly confirms CES 2026 as the announcement event and January 5, 2026 as the date for Panther Lake (Intel Core Ultra Series 3)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=announce_urls,
        additional_instruction="The page must explicitly mention both the event and the date."
    )

    # Availability details (Jan 27, 2026)
    avail_node = evaluator.add_parallel(
        id="Availability_Details",
        desc="Provide accurate information about when systems containing this product became available.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Availability_Date",
        desc="Systems containing the product must have become available starting January 27, 2026.",
        parent=avail_node,
        critical=True
    )
    claim = f"Systems containing '{product_name}' (Intel Core Ultra Series 3) became available starting January 27, 2026."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=avail_urls,
        additional_instruction="The page must explicitly mention systems availability beginning on January 27, 2026. Accept minor date formatting differences."
    )

    leaf = evaluator.add_leaf(
        id="Availability_Reference_URL",
        desc="Provide a reference URL that confirms the product availability date.",
        parent=avail_node,
        critical=True
    )
    claim = "At least one of the provided URLs explicitly confirms that systems with Panther Lake (Intel Core Ultra Series 3) became available starting January 27, 2026."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=avail_urls,
        additional_instruction="The statement should be explicit and unambiguous on the availability date."
    )

    # Product reference URL (product + 18A + this facility)
    leaf = evaluator.add_leaf(
        id="Product_Reference_URL",
        desc="Provide a reference URL that confirms the product name, codename, and its manufacturing using the 18A process at this facility.",
        parent=node,
        critical=True
    )
    claim = (
        f"At least one of the provided URLs confirms that '{product_name}' (Intel Core Ultra Series 3) "
        f"is manufactured using Intel 18A at the identified facility in Chandler's Ocotillo campus."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=product_urls,
        additional_instruction="The page(s) should explicitly connect the product, Intel 18A, and the specific Ocotillo/Chandler facility."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    # Initialize evaluator
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

    # Create a critical task root under the (non-critical) framework root to enforce all-or-nothing grading
    task_root = evaluator.add_parallel(
        id="Root_Task",
        desc="Identify Intel's advanced semiconductor manufacturing facility in Arizona that pioneered 2nm-class chip production in the United States, along with its process technology, production specifications, and first manufactured product.",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_structured(),
        template_class=Intel18AArizonaExtraction,
        extraction_name="extracted_info"
    )

    # Build verification tree according to rubric
    await verify_facility(evaluator, task_root, extracted)
    await verify_process(evaluator, task_root, extracted)
    await verify_production(evaluator, task_root, extracted)
    await verify_product(evaluator, task_root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()