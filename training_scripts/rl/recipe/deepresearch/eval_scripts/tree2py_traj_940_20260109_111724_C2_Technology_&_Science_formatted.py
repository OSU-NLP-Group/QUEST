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
TASK_ID = "ces2026_solid_state_ev_battery"
TASK_DESCRIPTION = (
    "At CES 2026 in January, a company announced the world's first production-ready all-solid-state battery for electric "
    "vehicles, which is being implemented in a production motorcycle with Q1 2026 deliveries. Identify the battery technology "
    "company and its CEO, provide the claimed cell-level energy density (in Wh/kg) and full charging time (in minutes), and "
    "identify the motorcycle manufacturer and the stated delivery timeline."
)


# --------------------------------------------------------------------------- #
# Data models for extracting information from the answer                      #
# --------------------------------------------------------------------------- #
class CESBatteryExtraction(BaseModel):
    # Company and announcement context
    company: Optional[str] = None
    company_sources: List[str] = Field(default_factory=list)

    ceo: Optional[str] = None
    ceo_sources: List[str] = Field(default_factory=list)

    announcement_event: Optional[str] = None  # e.g., "CES 2026", "Consumer Electronics Show 2026"
    announcement_month: Optional[str] = None  # e.g., "January"
    announcement_sources: List[str] = Field(default_factory=list)

    production_ready_claim_text: Optional[str] = None
    production_ready_sources: List[str] = Field(default_factory=list)

    worlds_first_claim_text: Optional[str] = None
    worlds_first_sources: List[str] = Field(default_factory=list)

    # Technical specifications
    battery_type: Optional[str] = None  # e.g., "all-solid-state", "solid-state"
    battery_type_sources: List[str] = Field(default_factory=list)

    energy_density: Optional[str] = None  # as written in answer (e.g., "400 Wh/kg", "up to 400 Wh/kg")
    energy_density_sources: List[str] = Field(default_factory=list)

    charging_time_full: Optional[str] = None  # e.g., "5 minutes", "5 min"
    charging_time_sources: List[str] = Field(default_factory=list)

    # Production vehicle & delivery
    vehicle_type: Optional[str] = None  # e.g., "electric motorcycle"
    motorcycle_manufacturer: Optional[str] = None
    delivery_timeline: Optional[str] = None  # e.g., "Q1 2026", "first quarter of 2026"
    vehicle_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ces_battery_info() -> str:
    return """
Extract the following information exactly as it appears in the answer. Do not infer or fabricate any details.

1) Company & Announcement
- company: The battery technology company's name that made the announcement.
- company_sources: All URLs in the answer that directly relate to or substantiate the company identity or the announcement. Extract only valid URLs.
- ceo: The name of the CEO of the company (as stated in the answer).
- ceo_sources: All URLs supporting the CEO identification; if none are explicitly provided, return an empty list.
- announcement_event: The name of the event for the announcement (e.g., "CES 2026", "Consumer Electronics Show 2026").
- announcement_month: The month of the announcement if stated (e.g., "January").
- announcement_sources: All URLs supporting the announcement context (event and timing).
- production_ready_claim_text: The phrase/wording in the answer that indicates the technology is "production-ready".
- production_ready_sources: All URLs supporting the 'production-ready' claim.
- worlds_first_claim_text: The phrase/wording in the answer that indicates a "world's first" claim.
- worlds_first_sources: All URLs supporting the "world's first" claim.

2) Technical Specifications
- battery_type: The battery type as stated (e.g., "all-solid-state", "solid-state battery with no liquid electrolyte").
- battery_type_sources: All URLs that support the stated battery type.
- energy_density: The claimed cell-level energy density (e.g., "400 Wh/kg" or "up to 400 Wh/kg").
- energy_density_sources: All URLs that support the energy density claim.
- charging_time_full: The claimed full charging time (e.g., "5 minutes"). If the answer only states "0–80% in 5 minutes", extract exactly what is written.
- charging_time_sources: All URLs that support the charging time claim.

3) Production Vehicle & Deliveries
- vehicle_type: The production vehicle type using this battery (e.g., "electric motorcycle").
- motorcycle_manufacturer: The motorcycle manufacturer's name.
- delivery_timeline: The delivery timing as stated (e.g., "Q1 2026", "first quarter of 2026").
- vehicle_sources: All URLs that support the vehicle implementation and the delivery timeline.

Rules:
- Extract only what is explicitly present in the answer. If a field is not present, set it to null (for strings) or [] for lists.
- For URLs, extract only valid URLs found in the answer (including those inside markdown links).
- Preserve the original phrasing (e.g., include "up to" if present).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(*lists: List[str]) -> List[str]:
    """Merge and de-duplicate URL lists while preserving order."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                merged.append(url)
                seen.add(url)
    return merged


# --------------------------------------------------------------------------- #
# Subtree verifications                                                       #
# --------------------------------------------------------------------------- #
async def build_announcement_and_company_subtree(
    evaluator: Evaluator,
    parent_node,
    data: CESBatteryExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Announcement_and_Company",
        desc="Identify the announcing battery technology company and validate the CES 2026 announcement claims",
        parent=parent_node,
        critical=True,
    )

    # Battery_Technology_Company
    sources_company = merge_sources(data.company_sources, data.announcement_sources)
    company_name = data.company or ""
    company_leaf = evaluator.add_leaf(
        id="Battery_Technology_Company",
        desc="Name of the battery technology company that made the announcement",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The battery technology company that made the announcement is {company_name}.",
        node=company_leaf,
        sources=sources_company,
        additional_instruction=(
            "Verify that the provided source(s) identify this company as the one making the announcement about the "
            "all-solid-state battery."
        ),
    )

    # Company_CEO
    sources_ceo = merge_sources(data.ceo_sources, data.company_sources, data.announcement_sources)
    ceo_name = data.ceo or ""
    ceo_leaf = evaluator.add_leaf(
        id="Company_CEO",
        desc="Name of the CEO of the battery technology company",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{ceo_name} is the CEO of {company_name}.",
        node=ceo_leaf,
        sources=sources_ceo,
        additional_instruction=(
            "Check that the source explicitly identifies this person as the CEO (Chief Executive Officer) of the named company. "
            "Allow minor name variants (e.g., middle initials)."
        ),
    )

    # Announcement_Context
    sources_announce = merge_sources(data.announcement_sources, data.company_sources)
    event_text = data.announcement_event or "CES 2026"
    month_text = data.announcement_month or "January"
    announcement_context_leaf = evaluator.add_leaf(
        id="Announcement_Context",
        desc="Announcement was made at CES 2026 in January",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The announcement was made at {event_text} in {month_text} 2026.",
        node=announcement_context_leaf,
        sources=sources_announce,
        additional_instruction=(
            "Treat 'CES' as 'Consumer Electronics Show'. The source should clearly indicate the announcement occurred at "
            "CES 2026 in January (2026). Equivalent phrasing like 'at CES 2026 in Las Vegas in January' is acceptable."
        ),
    )

    # Production_Ready_Claim
    sources_prod_ready = merge_sources(data.production_ready_sources, data.announcement_sources, data.company_sources)
    production_ready_leaf = evaluator.add_leaf(
        id="Production_Ready_Claim",
        desc="Announcement explicitly claims the technology is production-ready (not prototype or concept)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The announcement explicitly claims the battery technology is production-ready (not just a prototype or concept).",
        node=production_ready_leaf,
        sources=sources_prod_ready,
        additional_instruction=(
            "Look for explicit wording like 'production-ready', 'ready for mass production' or equivalent. "
            "If the source frames it as prototype, demo, trial, pilot, or concept only, this claim should fail."
        ),
    )

    # Worlds_First_Claim
    sources_first = merge_sources(data.worlds_first_sources, data.announcement_sources, data.company_sources)
    worlds_first_leaf = evaluator.add_leaf(
        id="Worlds_First_Claim",
        desc="Announcement explicitly claims this is the world's first (as stated in the prompt/constraints)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The announcement explicitly claims this is the world's first production-ready all-solid-state battery.",
        node=worlds_first_leaf,
        sources=sources_first,
        additional_instruction=(
            "Look for phrases like 'world's first', 'first in the world', or equivalent, tied to being production-ready "
            "and all-solid-state. Generic 'first' unrelated to this context should not count."
        ),
    )


async def build_technical_specifications_subtree(
    evaluator: Evaluator,
    parent_node,
    data: CESBatteryExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Technical_Specifications",
        desc="Key technical specifications stated in the announcement",
        parent=parent_node,
        critical=True,
    )

    # Battery_Type
    sources_batt_type = merge_sources(data.battery_type_sources, data.announcement_sources, data.company_sources)
    battery_type_leaf = evaluator.add_leaf(
        id="Battery_Type",
        desc="Battery is all-solid-state (no liquid electrolyte)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The battery is an all-solid-state battery with no liquid electrolyte.",
        node=battery_type_leaf,
        sources=sources_batt_type,
        additional_instruction=(
            "Accept equivalent phrasing such as 'solid-state battery' explicitly indicating solid electrolyte and no liquid electrolyte."
        ),
    )

    # Energy_Density
    sources_energy = merge_sources(data.energy_density_sources, data.announcement_sources, data.company_sources)
    energy_density_text = data.energy_density or ""
    energy_leaf = evaluator.add_leaf(
        id="Energy_Density",
        desc="Claimed cell-level energy density is 400 Wh/kg",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The claimed cell-level energy density is 400 Wh/kg.",
        node=energy_leaf,
        sources=sources_energy,
        additional_instruction=(
            "Confirm the numeric value '400 Wh/kg' at cell level (not pack level). "
            "Minor wording variants like 'up to 400 Wh/kg' are acceptable."
        ),
    )

    # Charging_Time
    sources_charge = merge_sources(data.charging_time_sources, data.announcement_sources, data.company_sources)
    charge_leaf = evaluator.add_leaf(
        id="Charging_Time",
        desc="Claimed full charging time is 5 minutes (not limited to 80%)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The claimed full charging time is 5 minutes to 100% (not merely 0–80%).",
        node=charge_leaf,
        sources=sources_charge,
        additional_instruction=(
            "The source must clearly indicate a 5-minute full charge (to 100%). If it only claims 5 minutes to 80%, "
            "this should fail."
        ),
    )


async def build_production_vehicle_subtree(
    evaluator: Evaluator,
    parent_node,
    data: CESBatteryExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Production_Vehicle_Implementation",
        desc="Details of the production vehicle implementation and delivery timing",
        parent=parent_node,
        critical=True,
    )

    # Implemented_in_Production_Vehicle
    sources_vehicle = merge_sources(data.vehicle_sources, data.announcement_sources, data.company_sources)
    impl_leaf = evaluator.add_leaf(
        id="Implemented_in_Production_Vehicle",
        desc="Battery is implemented in an actual production vehicle (not a concept/prototype-only vehicle)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The battery is implemented in an actual production vehicle (not just a concept or prototype).",
        node=impl_leaf,
        sources=sources_vehicle,
        additional_instruction=(
            "Look for explicit mentions of 'production model', 'production vehicle', or equivalent. "
            "If it is only a prototype, concept, development mule, or demo bike, this should fail."
        ),
    )

    # Vehicle_Is_Electric_Motorcycle
    vehicle_type_leaf = evaluator.add_leaf(
        id="Vehicle_Is_Electric_Motorcycle",
        desc="The production vehicle is an electric motorcycle",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The production vehicle is an electric motorcycle.",
        node=vehicle_type_leaf,
        sources=sources_vehicle,
        additional_instruction="The source should unambiguously identify the vehicle as an electric motorcycle.",
    )

    # Motorcycle_Manufacturer
    manufacturer_name = data.motorcycle_manufacturer or ""
    mfg_leaf = evaluator.add_leaf(
        id="Motorcycle_Manufacturer",
        desc="Name of the motorcycle manufacturer implementing the battery",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The motorcycle manufacturer implementing the battery is {manufacturer_name}.",
        node=mfg_leaf,
        sources=sources_vehicle,
        additional_instruction="Verify that the source names this manufacturer as the one implementing the battery.",
    )

    # Delivery_Timeline
    delivery_leaf = evaluator.add_leaf(
        id="Delivery_Timeline",
        desc="First customer deliveries are scheduled for Q1 2026 (delivery timeline explicitly stated)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="First customer deliveries are scheduled for Q1 2026.",
        node=delivery_leaf,
        sources=sources_vehicle,
        additional_instruction=(
            "Accept 'Q1 2026' or synonymous wording such as 'first quarter of 2026'. "
            "If deliveries are stated differently (e.g., 2025 or H2 2026), this should fail."
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
    Evaluate an answer for the CES 2026 all-solid-state EV battery announcement task.
    Returns a structured result dictionary with a verification tree and final score.
    """
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
        default_model=model,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_ces_battery_info(),
        template_class=CESBatteryExtraction,
        extraction_name="extracted_ces_battery_info",
    )

    # Build top-level critical node representing the entire rubric
    top = evaluator.add_parallel(
        id="First_Production_Solid_State_Battery_Information",
        desc="Complete and accurate information about the claimed world's first production-ready all-solid-state battery announced at CES 2026 and its production electric motorcycle implementation",
        parent=root,
        critical=True,
    )

    # Build three critical subtrees
    await build_announcement_and_company_subtree(evaluator, top, extracted)
    await build_technical_specifications_subtree(evaluator, top, extracted)
    await build_production_vehicle_subtree(evaluator, top, extracted)

    # Return evaluation summary
    return evaluator.get_summary()